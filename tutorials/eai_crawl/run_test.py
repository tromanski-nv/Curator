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

"""Run the EAI WARC PDF URL collection pipeline on a local WARC file.

Requires a Linux environment with NeMo Curator installed:
    uv sync --extra text_cpu

For local smoke testing on macOS, use ``run_local.py`` instead.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from loguru import logger

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.writer import ParquetWriter
from tutorials.eai_crawl.stage import EaiCrawlDownloadExtractStage

DEFAULT_WARC = Path.home() / "github/eai-warc-analysis/0a33ff05-01bf-4ef3-8437-c5413097d899.warc.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--warc",
        type=Path,
        default=DEFAULT_WARC,
        help=f"Path to a local .warc or .warc.gz file (default: {DEFAULT_WARC})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for JSONL output (default: temp directory)",
    )
    parser.add_argument(
        "--record-limit",
        type=int,
        default=None,
        help="Maximum PDF records to extract per WARC file",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    warc_path = args.warc.expanduser().resolve()
    if not warc_path.is_file():
        logger.error(f"WARC file not found: {warc_path}")
        return 1

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="eai_warc_pdf_"))
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline(
        name="eai_warc_pdf_pipeline",
        description="Collect PDF URLs from application/pdf WARC responses",
    )
    pipeline.add_stage(
        EaiCrawlDownloadExtractStage(
            warc_paths=[str(warc_path)],
            record_limit=args.record_limit,
        )
    )
    pipeline.add_stage(ParquetWriter(path=str(output_dir)))

    logger.info(f"Processing WARC: {warc_path}")
    results = pipeline.run()

    total_records = sum(task.num_items for task in results)
    logger.info(f"Collected {total_records} PDF URL(s)")
    logger.info(f"Parquet output written to: {output_dir}")

    if total_records:
        sample = results[0].to_pandas().iloc[0]
        logger.info(f"Sample URL: {sample['url']}")
        logger.info(f"Sample filename: {sample['filename']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
