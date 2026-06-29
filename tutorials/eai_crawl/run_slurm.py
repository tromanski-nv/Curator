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

"""Scaled EAI WARC -> PDF URL pipeline (single node or multi-node SLURM).

Writes Parquet. Reads either local/shared-FS WARC files or S3 objects:

    # Local single-node (a directory of WARC shards):
    python tutorials/eai_crawl/run_slurm.py \\
        --warc-dir /shared/warcs --output-dir /shared/out

    # S3 metadata-only (range reads; uncompressed WARCs):
    python tutorials/eai_crawl/run_slurm.py \\
        --s3-bucket my-bucket --s3-prefix crawl/warcs/ --output-dir /shared/out

    # Multi-node: add --slurm and launch via tutorials/eai_crawl/submit.sh (srun).

The only code difference between local and SLURM is the Ray client
(``SlurmRayClient`` vs ``RayClient``), mirroring tutorials/slurm/pipeline.py.
"""

from __future__ import annotations

import argparse

from loguru import logger

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.core.client import RayClient, SlurmRayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.writer import ParquetWriter
from tutorials.eai_crawl.stage import EaiCrawlDownloadExtractStage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slurm", action="store_true", help="Use SlurmRayClient (set when running via srun)")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--warc-dir", help="Directory of local/shared-FS WARC files")
    source.add_argument("--s3-bucket", help="S3 bucket holding WARC objects")

    parser.add_argument("--s3-prefix", default="", help="S3 key prefix to list under (with --s3-bucket)")
    parser.add_argument("--download-dir", default="./eai_warc_downloads", help="Scratch dir (local mode; not copied)")
    parser.add_argument("--output-dir", required=True, help="Directory for Parquet output")
    parser.add_argument("--url-limit", type=int, default=None, help="Max WARC files/objects to process")
    parser.add_argument("--record-limit", type=int, default=None, help="Max PDF records per WARC")
    parser.add_argument("--header-bytes", type=int, default=16384, help="Bytes per record range read (S3 mode)")
    return parser.parse_args()


def build_pipeline(args: argparse.Namespace) -> Pipeline:
    pipeline = Pipeline(
        name="eai_warc_pdf_pipeline",
        description="Collect PDF URLs/metadata from application/pdf WARC responses",
    )

    if args.s3_bucket:
        # Imported lazily so local-mode runs don't require boto3.
        from tutorials.eai_crawl.s3_stage import S3EaiCrawlStage

        pipeline.add_stage(
            S3EaiCrawlStage(
                bucket=args.s3_bucket,
                prefix=args.s3_prefix,
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

    pipeline.add_stage(ParquetWriter(path=args.output_dir))
    return pipeline


def main() -> int:
    args = parse_args()

    ray_client = SlurmRayClient() if args.slurm else RayClient()
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
