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
    source .venv/bin/activate
    python tutorials/eai_crawl/run_slurm.py \\
        --s3-bucket vdi-169-essentialai-essentialai-data \\
        --s3-prefix eai-warc/20240814/ --stream --url-limit 2 \\
        --s3-endpoint-url https://pdx.s8k.io \\
        --output-dir s3://eai-warcs/pdf_url_idx/20240814/ \\
        --cdx-output-dir s3://eai-warcs/cdx/20240814/ \\
        --output-rclone-remote eai-data

    # Multi-node: add --slurm and launch via tutorials/eai_crawl/submit.sh (srun).
"""

from __future__ import annotations

import os
import socket
import sys
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

# ``uv run`` / bare ``python path/to/script.py`` only put the script dir on
# sys.path; tutorials.* imports need the repo root (same pattern as run_local.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse  # noqa: E402

from nemo_curator.core.client import RayClient, SlurmRayClient  # noqa: E402
from nemo_curator.core.constants import DEFAULT_RAY_TEMP_DIR  # noqa: E402
from nemo_curator.pipeline import Pipeline  # noqa: E402
from nemo_curator.stages.text.io.writer import ParquetWriter  # noqa: E402
from tutorials.eai_crawl.resume import (  # noqa: E402
    OUTPUT_LAYOUT_VERSION,
    initialize_output,
    manifest_identity,
    success_marker_path,
    write_json,
)
from tutorials.eai_crawl.s3_storage import (  # noqa: E402
    is_remote_url,
    resolve_output_storage_options,
)
from tutorials.eai_crawl.stage import EaiCrawlDownloadExtractStage  # noqa: E402

# Backends are imported lazily in build_executor() so a missing experimental
# dependency for one backend doesn't break the others.
#
_BACKEND_CHOICES = ("ray_actor_pool", "ray_data", "xenna")


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
    parser.add_argument(
        "--stream-cpus",
        type=float,
        default=float(os.environ.get("EAI_STREAM_CPUS", "1.0")),
        help="CPU reservation per WARC stream task (S3 --stream mode). Lower (e.g. 0.25) packs more "
        "concurrent I/O-bound streams per node. Default 1.0 (or EAI_STREAM_CPUS).",
    )
    parser.add_argument(
        "--warcs-per-task",
        type=int,
        default=int(os.environ.get("EAI_WARCS_PER_TASK", "32")),
        help="Deterministic WARC source-group size. Each group writes idempotent consolidated parts "
        "and is the native resume unit. Default 32 or EAI_WARCS_PER_TASK.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=os.environ.get("EAI_CHECKPOINT_PATH"),
        help="Shared local/Lustre checkpoint directory for native resumability.",
    )
    parser.add_argument("--header-bytes", type=int, default=16384, help="Bytes per record range read (S3 range mode)")
    parser.add_argument(
        "--cdx-output-dir",
        default=None,
        help="If set (S3 --stream mode), write consolidated CDX Parquet part files "
        "(local or s3://eai-warcs/cdx/<day>/). Rows are consolidated within each "
        "deterministic WARC source group.",
    )
    parser.add_argument(
        "--cdx-rows-per-file",
        type=int,
        default=int(os.environ.get("EAI_CDX_ROWS_PER_FILE", "2000000")),
        help="Target CDX rows per consolidated Parquet part (S3 --stream mode). Higher = "
        "fewer/larger files. Default 2_000_000 (~250 MiB) or EAI_CDX_ROWS_PER_FILE.",
    )
    parser.add_argument(
        "--pdf-rows-per-file",
        type=int,
        default=int(os.environ.get("EAI_PDF_ROWS_PER_FILE", "2000000")),
        help="Target PDF-index rows per consolidated Parquet part (S3 --stream mode). PDFs "
        "are rare, so a source group usually emits at most one PDF part. "
        "Default 2_000_000 or EAI_PDF_ROWS_PER_FILE.",
    )
    parser.add_argument(
        "--backend",
        choices=_BACKEND_CHOICES,
        default=os.environ.get("EAI_BACKEND", "ray_actor_pool"),
        help="Ray execution backend. Default 'ray_actor_pool' (or EAI_BACKEND).",
    )
    parser.add_argument(
        "--output-rclone-remote",
        default=None,
        help="rclone remote name for S3 writes (e.g. eai-data). Used when --output-dir / "
        "--cdx-output-dir are s3:// and EAI_OUT_AWS_* are unset.",
    )
    return parser.parse_args()


def build_executor(backend: str):  # noqa: ANN201
    """Instantiate the requested Ray execution backend (imported lazily)."""
    if backend == "ray_actor_pool":
        from nemo_curator.backends.ray_actor_pool import RayActorPoolExecutor

        return RayActorPoolExecutor()
    if backend == "ray_data":
        from nemo_curator.backends.ray_data import RayDataExecutor

        return RayDataExecutor()
    if backend == "xenna":
        from nemo_curator.backends.xenna import XennaExecutor

        return XennaExecutor(config={"execution_mode": "streaming"})
    msg = f"Unknown backend: {backend!r} (choose from {_BACKEND_CHOICES})"
    raise SystemExit(msg)


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

        # Terminal, memory-safe writer: the stream stage buffers PDF + CDX rows per
        # worker and writes consolidated parts itself (no per-WARC tiny files, and
        # nothing large flows back to the RayActorPool driver). So we do NOT add a
        # downstream ParquetWriter for this path.
        remote_out = is_remote_url(args.output_dir)
        pipeline.add_stage(
            S3StreamEaiCrawlStage(
                bucket=args.s3_bucket,
                prefix=args.s3_prefix,
                suffix=args.s3_suffix or ".warc.gz",
                endpoint_url=args.s3_endpoint_url,
                region=args.s3_region,
                url_limit=args.url_limit,
                record_limit=args.record_limit,
                pdf_output_dir=args.output_dir,
                pdf_storage_options=write_opts if remote_out else None,
                pdf_rows_per_file=args.pdf_rows_per_file,
                cdx_output_dir=args.cdx_output_dir,
                cdx_storage_options=write_opts if args.cdx_output_dir and is_remote_url(args.cdx_output_dir) else None,
                keys=keys,
                warcs_per_task=args.warcs_per_task,
                stream_cpus=args.stream_cpus,
                cdx_rows_per_file=args.cdx_rows_per_file,
            )
        )
        return pipeline

    if args.s3_bucket:
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

    # Non-stream paths still return DocumentBatches; write them with the standard writer.
    writer_kwargs: dict = {}
    if is_remote_url(args.output_dir):
        writer_kwargs["storage_options"] = write_opts
    pipeline.add_stage(ParquetWriter(path=args.output_dir, write_kwargs=writer_kwargs, mode="overwrite"))
    return pipeline


def _log_ray_dashboard(ray_client: RayClient) -> None:
    """Log the Ray dashboard URL + SSH tunnel hint (works for any Ray backend).

    The dashboard's cluster views (per-node CPU under "Logical Resources", the
    Sent/Received network panels, actors) are populated by Ray core, so they are
    available regardless of the execution backend — only Ray Data adds an extra
    "Ray Data" progress tab on top.
    """
    host = socket.gethostname()
    try:
        ip = socket.gethostbyname(host)
    except OSError:
        ip = host
    port = getattr(ray_client, "ray_dashboard_port", None)
    if not port:
        return
    logger.info(
        f"Ray dashboard: http://{ip}:{port}  (head node: {host})\n"
        f"  SSH tunnel from your laptop:  ssh -N -L {port}:{ip}:{port} <login-host>\n"
        f"  then open http://localhost:{port}"
    )


def _code_snapshot_identity() -> dict[str, str]:
    identity = {"path": str(REPO_ROOT)}
    info_path = REPO_ROOT / "SNAPSHOT_INFO.txt"
    if info_path.is_file():
        for raw in info_path.read_text().splitlines():
            key, separator, value = raw.partition(":")
            if separator and key.strip() and value.strip():
                identity[key.strip()] = value.strip()
    return identity


def main() -> int:  # noqa: PLR0915
    args = parse_args()

    if bool(args.checkpoint_path) != bool(args.s3_keys_file):
        msg = "Resumable streaming requires both --checkpoint-path and --s3-keys-file"
        raise SystemExit(msg)

    # RayClient defaults to /tmp/ray which is often unwritable on shared login nodes.
    # Honor RAY_TMPDIR (set by submit.sh / the day-scale checklist).
    ray_temp_dir = os.environ.get("RAY_TMPDIR", DEFAULT_RAY_TEMP_DIR)
    os.makedirs(ray_temp_dir, exist_ok=True)

    client_cls = SlurmRayClient if args.slurm else RayClient
    ray_client = client_cls(ray_temp_dir=ray_temp_dir)
    ray_client.start()
    # On SLURM worker nodes (SLURM_NODEID > 0) start() blocks; only the head continues.
    _log_ray_dashboard(ray_client)

    manifest_sha256 = ""
    manifest_warcs = 0
    write_opts = None
    try:
        if args.s3_bucket and args.stream and args.checkpoint_path:
            manifest_sha256, manifest_warcs = manifest_identity(args.s3_keys_file)
            write_opts = resolve_output_storage_options(rclone_remote=args.output_rclone_remote)
            if is_remote_url(args.output_dir) and write_opts is None:
                msg = (
                    "Remote output requires --output-rclone-remote or EAI_OUT_AWS_ACCESS_KEY_ID/"
                    "EAI_OUT_AWS_SECRET_ACCESS_KEY"
                )
                raise SystemExit(msg)
            migrated = initialize_output(
                checkpoint_path=args.checkpoint_path,
                manifest_sha256=manifest_sha256,
                pdf_output_dir=args.output_dir,
                cdx_output_dir=args.cdx_output_dir,
                storage_options=write_opts,
            )
            logger.info(
                f"{'Migrated legacy output' if migrated else 'Resuming initialized output'}: "
                f"manifest={manifest_sha256} WARCs={manifest_warcs}"
            )
        pipeline = build_pipeline(args)
        logger.info(f"\n{pipeline.describe()}")
        logger.info(f"Executor backend: {args.backend}")
        executor = build_executor(args.backend)
        results = pipeline.run(executor=executor, checkpoint_path=args.checkpoint_path)
    finally:
        ray_client.stop()

    if args.s3_bucket and args.stream and args.checkpoint_path:
        from nemo_curator.backends.failed_task_markers import failed_task_manifest_exists
        from nemo_curator.backends.slurm_array import find_slurm_array_retries

        if failed_task_manifest_exists():
            msg = "One or more WARC source groups failed; refusing to write _SUCCESS"
            raise RuntimeError(msg)
        retry_plan = find_slurm_array_retries(args.checkpoint_path)
        if retry_plan is None or retry_plan.shard_indices:
            pending = retry_plan.shard_indices if retry_plan else ("unknown",)
            msg = f"Native resumability reports incomplete shard(s): {pending}; refusing to write _SUCCESS"
            raise RuntimeError(msg)
        marker = {
            "status": "completed",
            "layout_version": OUTPUT_LAYOUT_VERSION,
            "manifest_sha256": manifest_sha256,
            "warc_count": manifest_warcs,
            "pdf_output_dir": args.output_dir.rstrip("/") + "/",
            "cdx_output_dir": args.cdx_output_dir.rstrip("/") + "/" if args.cdx_output_dir else None,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "code_snapshot": _code_snapshot_identity(),
            "completed_at": datetime.now(UTC).isoformat(),
        }
        marker_path = success_marker_path(args.output_dir)
        write_json(marker_path, marker, write_opts)
        logger.info(f"Uploaded authoritative success marker: {marker_path}")
        # Terminal self-writing stage returns no tasks; per-file/row counts are in
        # the per-worker "Flushed N ... row(s)" log lines.
        logger.info("Streaming complete; PDF/CDX parts written by workers (see 'Flushed ...' log lines).")
    elif args.s3_bucket and args.stream:
        logger.info("Streaming complete without native checkpointing or an authoritative _SUCCESS marker.")
    else:
        total_records = sum(task.num_items for task in results) if results else 0
        logger.info(f"Collected {total_records} PDF URL(s) across {len(results) if results else 0} output batch(es)")
    logger.info(f"PDF index output written to: {args.output_dir}")
    if args.cdx_output_dir:
        logger.info(f"CDX output written to: {args.cdx_output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
