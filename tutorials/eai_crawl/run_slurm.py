# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Scaled EAI WARC -> PDF URL (+ optional CDX) pipeline (single node or multi-node SLURM).

Writes Parquet to a local path or ``s3://`` (SwiftStack). Reads WARCs from a
local dir or S3/S3-compatible store. Input and output often use **different**
credentials on the same endpoint — pass ``--output-rclone-remote eai-data``
(or ``EAI_OUT_AWS_*``) so writes do not reuse the read ``AWS_*`` keys.

    # Day smoke -> eai-data bucket (rclone remote):
    export AWS_*                 # team-vendor-data (read)
    uv run --group eai-warcs --extra text_cpu python tutorials/eai_crawl/run_slurm.py \\
        --s3-bucket vdi-169-essentialai-essentialai-data \\
        --s3-prefix eai-warc/20240814/ --stream --url-limit 2 \\
        --s3-endpoint-url https://pdx.s8k.io \\
        --output-dir s3://eai-warcs/pdf_url_idx/20240814/ \\
        --cdx-output-dir s3://eai-warcs/cdx/20240814/ \\
        --output-rclone-remote eai-data

    # Multi-node: add --slurm and launch via tutorials/eai_crawl/submit.sh (srun).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from loguru import logger

# ``uv run`` / bare ``python path/to/script.py`` only put the script dir on
# sys.path; tutorials.* imports need the repo root (same pattern as run_local.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nemo_curator.backends.xenna import XennaExecutor  # noqa: E402
from nemo_curator.core.client import RayClient, SlurmRayClient  # noqa: E402
from nemo_curator.core.constants import DEFAULT_RAY_TEMP_DIR  # noqa: E402
from nemo_curator.pipeline import Pipeline  # noqa: E402
from nemo_curator.stages.text.io.writer import ParquetWriter  # noqa: E402
from tutorials.eai_crawl.s3_storage import (  # noqa: E402
    is_remote_url,
    resolve_output_storage_options,
)
from tutorials.eai_crawl.stage import EaiCrawlDownloadExtractStage  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slurm", action="store_true", help="Use SlurmRayClient (set when running via srun)")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--warc-dir", help="Directory of local/shared-FS WARC files")
    source.add_argument("--s3-bucket", help="S3/S3-compatible bucket holding WARC objects")

    parser.add_argument("--s3-prefix", default="", help="S3 key prefix to list under (with --s3-bucket)")
    parser.add_argument(
        "--s3-endpoint-url",
        default=None,
        help="Endpoint for S3-compatible stores (e.g. SwiftStack https://pdx.s8k.io). "
        "Falls back to the AWS_ENDPOINT_URL env var.",
    )
    parser.add_argument("--s3-region", default=None, help="Region name for the S3 client (optional)")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream each object through warcio (required for compressed .warc.gz). "
        "Without this, S3 mode uses metadata-only range reads which need UNcompressed .warc.",
    )
    parser.add_argument(
        "--s3-suffix",
        default=None,
        help="Object key suffix filter (default '.warc.gz' in --stream mode, else '.warc')",
    )
    parser.add_argument("--download-dir", default="./eai_warc_downloads", help="Scratch dir (local mode; not copied)")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Parquet output dir (local path or s3://bucket/prefix/, e.g. s3://eai-warcs/pdf_url_idx/20240814/)",
    )
    parser.add_argument("--url-limit", type=int, default=None, help="Max WARC files/objects to process")
    parser.add_argument(
        "--s3-keys-file",
        default=None,
        help="File with one S3 object key per line (S3 --stream mode). Bypasses prefix listing so "
        "one job can process a byte-sized chunk of WARCs spanning multiple days. Blank lines and "
        "'#' comments are ignored.",
    )
    parser.add_argument("--record-limit", type=int, default=None, help="Max PDF records per WARC")
    parser.add_argument("--header-bytes", type=int, default=16384, help="Bytes per record range read (S3 range mode)")
    parser.add_argument(
        "--cdx-output-dir",
        default=None,
        help="If set (S3 --stream mode), write one CDX Parquet per WARC "
        "(local or s3://eai-warcs/cdx/<day>/)",
    )
    parser.add_argument(
        "--output-rclone-remote",
        default=None,
        help="rclone remote name for S3 writes (e.g. eai-data). Used when --output-dir / "
        "--cdx-output-dir are s3:// and EAI_OUT_AWS_* are unset.",
    )
    return parser.parse_args()


def _load_keys_file(path: str) -> list[str]:
    """Read S3 object keys (one per line) from a chunk manifest; skip blanks/comments."""
    keys: list[str] = []
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if line and not line.startswith("#"):
                keys.append(line)
    if not keys:
        msg = f"--s3-keys-file {path} contained no usable keys"
        raise SystemExit(msg)
    logger.info(f"Loaded {len(keys)} WARC key(s) from {path}")
    return keys


def build_pipeline(args: argparse.Namespace) -> Pipeline:
    pipeline = Pipeline(
        name="eai_warc_pdf_pipeline",
        description="Collect PDF URLs/metadata from application/pdf WARC responses",
    )

    needs_remote_write = is_remote_url(args.output_dir) or (
        args.cdx_output_dir is not None and is_remote_url(args.cdx_output_dir)
    )
    write_opts = None
    if needs_remote_write:
        write_opts = resolve_output_storage_options(rclone_remote=args.output_rclone_remote)
        if write_opts is None:
            msg = (
                "Remote s3:// output requires --output-rclone-remote (e.g. eai-data) "
                "or EAI_OUT_AWS_ACCESS_KEY_ID / EAI_OUT_AWS_SECRET_ACCESS_KEY"
            )
            raise SystemExit(msg)

    keys = _load_keys_file(args.s3_keys_file) if args.s3_keys_file else None
    if args.s3_keys_file and not (args.s3_bucket and args.stream):
        msg = "--s3-keys-file requires --s3-bucket and --stream"
        raise SystemExit(msg)

    if args.s3_bucket and args.stream:
        from tutorials.eai_crawl.s3_streaming import S3StreamEaiCrawlStage

        pipeline.add_stage(
            S3StreamEaiCrawlStage(
                bucket=args.s3_bucket,
                prefix=args.s3_prefix,
                suffix=args.s3_suffix or ".warc.gz",
                endpoint_url=args.s3_endpoint_url,
                region=args.s3_region,
                url_limit=args.url_limit,
                record_limit=args.record_limit,
                cdx_output_dir=args.cdx_output_dir,
                cdx_storage_options=write_opts if args.cdx_output_dir and is_remote_url(args.cdx_output_dir) else None,
                keys=keys,
            )
        )
    elif args.s3_bucket:
        from tutorials.eai_crawl.s3_stage import S3EaiCrawlStage

        pipeline.add_stage(
            S3EaiCrawlStage(
                bucket=args.s3_bucket,
                prefix=args.s3_prefix,
                suffix=args.s3_suffix or ".warc",
                url_limit=args.url_limit,
                record_limit=args.record_limit,
                header_bytes=args.header_bytes,
            )
        )
    else:
        pipeline.add_stage(
            EaiCrawlDownloadExtractStage(
                warc_dir=args.warc_dir,
                download_dir=args.download_dir,
                url_limit=args.url_limit,
                record_limit=args.record_limit,
            )
        )

    writer_kwargs: dict = {}
    if is_remote_url(args.output_dir):
        writer_kwargs["storage_options"] = write_opts
    pipeline.add_stage(ParquetWriter(path=args.output_dir, write_kwargs=writer_kwargs, mode="overwrite"))
    return pipeline


def main() -> int:
    args = parse_args()

    # RayClient defaults to /tmp/ray which is often unwritable on shared login nodes.
    # Honor RAY_TMPDIR (set by submit.sh / the day-scale checklist).
    ray_temp_dir = os.environ.get("RAY_TMPDIR", DEFAULT_RAY_TEMP_DIR)
    os.makedirs(ray_temp_dir, exist_ok=True)

    client_cls = SlurmRayClient if args.slurm else RayClient
    ray_client = client_cls(ray_temp_dir=ray_temp_dir)
    ray_client.start()
    # On SLURM worker nodes (SLURM_NODEID > 0) start() blocks; only the head continues.

    try:
        pipeline = build_pipeline(args)
        logger.info(f"\n{pipeline.describe()}")
        executor = XennaExecutor(config={"execution_mode": "streaming"})
        results = pipeline.run(executor=executor)
    finally:
        ray_client.stop()

    total_records = sum(task.num_items for task in results) if results else 0
    logger.info(f"Collected {total_records} PDF URL(s) across {len(results) if results else 0} output batch(es)")
    logger.info(f"Parquet output written to: {args.output_dir}")
    if args.cdx_output_dir:
        logger.info(f"CDX output written to: {args.cdx_output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
