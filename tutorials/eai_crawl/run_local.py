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

"""Local smoke test for the EAI PDF extractor (works without the Curator pipeline).

Use this on macOS or anywhere you want to validate iterator/extractor logic
without importing ``nemo_curator``. Requires ``warcio`` plus ``pandas``/``pyarrow``
(all in the text_cpu extras) for Parquet output.

Install deps:
    uv sync --extra text_cpu
    # or: pip install warcio pandas pyarrow

Run:
    python tutorials/eai_crawl/run_local.py \\
        --warc ~/github/eai-warc-analysis/0a33ff05-01bf-4ef3-8437-c5413097d899.warc.gz \\
        --output-dir /tmp/eai_pdf_urls
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tutorials.eai_crawl.pdf_records import (  # noqa: E402
    PDF_OUTPUT_COLUMNS,
    extract_pdf_record,
    iterate_pdf_warc_records,
)

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
        required=True,
        help="Directory for JSONL output",
    )
    parser.add_argument(
        "--record-limit",
        type=int,
        default=None,
        help="Maximum PDF records to collect",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    warc_path = args.warc.expanduser().resolve()
    if not warc_path.is_file():
        print(f"WARC file not found: {warc_path}")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / f"{warc_path.name}.parquet"

    records = []
    for raw_record in iterate_pdf_warc_records(str(warc_path)):
        extracted = extract_pdf_record(raw_record)
        if extracted is None:
            continue
        records.append(extracted)
        if args.record_limit is not None and len(records) >= args.record_limit:
            break

    df = pd.DataFrame(records, columns=PDF_OUTPUT_COLUMNS)
    df.to_parquet(output_file, index=False)

    print(f"Collected {len(records)} PDF URL(s)")
    print(f"Wrote {output_file}")
    if records:
        print(f"Sample URL: {records[0]['url']}")
        print(f"Sample filename: {records[0]['filename']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
