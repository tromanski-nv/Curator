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

"""S3 metadata-only WARC reading wired into the Curator pipeline.

``S3WarcMetadataStage`` replaces the download + iterate steps of the local
pipeline: instead of fetching whole WARC files, it range-reads only the header
bytes of each record (via ``WarcMetadataScanner``) and emits a ``DocumentBatch``
of PDF URLs/metadata directly. ``S3EaiCrawlStage`` is the user-facing composite
that lists S3 keys and feeds them to the metadata stage.

Note: ``iter_pdf_metadata`` does an index-free *sequential* scan, which requires
**uncompressed** WARCs. For per-record-gzipped ``.warc.gz`` you need a byte-offset
index (CDX) and ``WarcMetadataScanner.read_record_metadata`` instead.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import pandas as pd
from loguru import logger

from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.download.base.url_generation import URLGenerationStage
from nemo_curator.tasks import DocumentBatch, EmptyTask, FileGroupTask
from tutorials.eai_crawl.pdf_records import PDF_OUTPUT_COLUMNS, extract_pdf_record
from tutorials.eai_crawl.s3_download import (
    DEFAULT_HEADER_BYTES,
    RangeReader,
    S3RangeReader,
    WarcMetadataScanner,
)
from tutorials.eai_crawl.s3_url_generation import S3WarcUrlGenerator


@dataclass
class S3WarcMetadataStage(ProcessingStage[FileGroupTask, DocumentBatch]):
    """Range-read PDF metadata from S3 WARC objects, skipping PDF payloads."""

    bucket: str | None = None
    header_bytes: int = DEFAULT_HEADER_BYTES
    record_limit: int | None = None
    add_filename_column: bool | str = True
    reader: RangeReader | None = None  # inject for local/testing; else S3RangeReader(bucket)
    name: str = "s3_warc_pdf_metadata"
    resources = Resources(cpus=1.0)

    def __post_init__(self) -> None:
        self.filename_col = self.add_filename_column if isinstance(self.add_filename_column, str) else "file_name"
        self._s3_reader: RangeReader | None = None

    def inputs(self) -> tuple[list[str], list[str]]:
        return (["data"], [])

    def outputs(self) -> tuple[list[str], list[str]]:
        cols = list(PDF_OUTPUT_COLUMNS)
        if self.add_filename_column:
            cols.append(self.filename_col)
        return (["data"], cols)

    def _get_reader(self) -> RangeReader:
        if self.reader is not None:
            return self.reader
        if self._s3_reader is None:
            if not self.bucket:
                msg = "S3WarcMetadataStage requires a bucket (or an injected reader)"
                raise ValueError(msg)
            self._s3_reader = S3RangeReader(self.bucket)
        return self._s3_reader

    def process(self, task: FileGroupTask) -> DocumentBatch:
        scanner = WarcMetadataScanner(self._get_reader(), header_bytes=self.header_bytes)
        rows: list[dict[str, Any]] = []

        for key in task.data:
            count = 0
            try:
                for raw_record in scanner.iter_pdf_metadata(key):
                    extracted = extract_pdf_record(raw_record)
                    if extracted is None:
                        continue
                    if self.add_filename_column:
                        extracted[self.filename_col] = os.path.basename(key)
                    rows.append(extracted)
                    count += 1
                    if self.record_limit and count >= self.record_limit:
                        break
            except Exception:  # noqa: BLE001
                logger.exception(f"Failed scanning WARC object {key}")
                continue

        return DocumentBatch(
            dataset_name=task.dataset_name,
            data=pd.DataFrame(rows),
            _metadata={**task._metadata},
            _stage_perf=task._stage_perf,
        )


class S3EaiCrawlStage(CompositeStage[EmptyTask, DocumentBatch]):
    """List S3 WARC objects and collect PDF URLs/metadata via range reads."""

    def __init__(  # noqa: PLR0913
        self,
        bucket: str,
        prefix: str = "",
        suffix: str = ".warc",
        url_limit: int | None = None,
        record_limit: int | None = None,
        header_bytes: int = DEFAULT_HEADER_BYTES,
        add_filename_column: bool | str = True,
    ) -> None:
        super().__init__()
        self.url_generator = S3WarcUrlGenerator(bucket=bucket, prefix=prefix, suffix=suffix, limit=url_limit)
        self.stages = [
            URLGenerationStage(url_generator=self.url_generator, limit=url_limit),
            S3WarcMetadataStage(
                bucket=bucket,
                header_bytes=header_bytes,
                record_limit=record_limit,
                add_filename_column=add_filename_column,
            ),
        ]
        self.name = "s3_eai_crawl_pdf_extract"

    def decompose(self) -> list[ProcessingStage]:
        return self.stages

    def get_description(self) -> str:
        return "Collect PDF URLs/metadata from application/pdf records in S3 WARC files (metadata-only range reads)"
