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

"""Streaming WARC reading for compressed (``.warc.gz``) objects on S3/SwiftStack.

Unlike ``s3_stage.S3WarcMetadataStage`` (which range-reads only header bytes and
therefore requires *uncompressed* WARCs), this stage streams each ``.warc.gz``
object through ``warcio`` and extracts PDF URL/metadata on the fly. The PDF
*bodies* are decompressed and discarded in-flight, so nothing is written to disk,
but the whole object must be read over the network (unavoidable for whole-file
gzip without a byte-offset/CDX index).

This is the correct path for the EssentialAI crawl layout:
``s3://<bucket>/eai-warc/<YYYYMMDD>/<uuid>.warc.gz``.

For S3-compatible stores (SwiftStack), set ``endpoint_url`` (or the
``AWS_ENDPOINT_URL`` env var); credentials come from the boto3 default chain
(``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` / ``AWS_SESSION_TOKEN``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import IO, Any

import pandas as pd
from loguru import logger

from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.download.base.url_generation import URLGenerationStage
from nemo_curator.tasks import DocumentBatch, EmptyTask, FileGroupTask
from tutorials.eai_crawl.pdf_records import PDF_OUTPUT_COLUMNS, extract_pdf_record, iterate_pdf_warc_stream
from tutorials.eai_crawl.s3_url_generation import S3WarcUrlGenerator


class ObjectStreamer:
    """Open a readable binary stream for an object key. Subclass per backend."""

    def open(self, key: str) -> IO[bytes]:
        raise NotImplementedError


class S3ObjectStreamer(ObjectStreamer):
    """Stream whole objects from S3 (or S3-compatible storage) via boto3.

    Credentials/region resolve through boto3's default chain; ``endpoint_url``
    (or ``AWS_ENDPOINT_URL``) selects an S3-compatible endpoint (e.g. SwiftStack
    ``https://pdx.s8k.io``).
    """

    def __init__(
        self,
        bucket: str,
        *,
        client: Any = None,  # noqa: ANN401 - boto3 S3 client has no type stubs
        endpoint_url: str | None = None,
        region: str | None = None,
        max_retries: int = 5,
        timeout: int = 60,
    ) -> None:
        self.bucket = bucket
        self.endpoint_url = endpoint_url
        self.region = region
        self.max_retries = max_retries
        self.timeout = timeout
        self._client = client

    def _get_client(self) -> Any:  # noqa: ANN401 - boto3 S3 client has no type stubs
        if self._client is None:
            try:
                import boto3
            except ModuleNotFoundError as exc:
                msg = "boto3 is required for S3ObjectStreamer. Install with: pip install boto3"
                raise RuntimeError(msg) from exc
            from botocore.config import Config as BotoConfig

            boto_cfg = BotoConfig(
                retries={"max_attempts": self.max_retries, "mode": "adaptive"},
                connect_timeout=self.timeout,
                read_timeout=self.timeout,
            )
            endpoint = self.endpoint_url or os.environ.get("AWS_ENDPOINT_URL") or None
            self._client = boto3.client("s3", config=boto_cfg, endpoint_url=endpoint, region_name=self.region)
        return self._client

    def open(self, key: str) -> IO[bytes]:
        resp = self._get_client().get_object(Bucket=self.bucket, Key=key)
        # StreamingBody is a file-like object (.read()) that warcio can consume.
        return resp["Body"]


@dataclass
class S3WarcStreamStage(ProcessingStage[FileGroupTask, DocumentBatch]):
    """Stream ``.warc.gz`` objects and emit a DocumentBatch of PDF URLs/metadata."""

    bucket: str | None = None
    endpoint_url: str | None = None
    region: str | None = None
    record_limit: int | None = None
    add_filename_column: bool | str = True
    streamer: ObjectStreamer | None = None  # inject for tests; else S3ObjectStreamer(bucket)
    name: str = "s3_warc_pdf_stream"
    resources = Resources(cpus=1.0)

    def __post_init__(self) -> None:
        self.filename_col = self.add_filename_column if isinstance(self.add_filename_column, str) else "file_name"
        self._s3_streamer: ObjectStreamer | None = None

    def inputs(self) -> tuple[list[str], list[str]]:
        return (["data"], [])

    def outputs(self) -> tuple[list[str], list[str]]:
        cols = list(PDF_OUTPUT_COLUMNS)
        if self.add_filename_column:
            cols.append(self.filename_col)
        return (["data"], cols)

    def _get_streamer(self) -> ObjectStreamer:
        if self.streamer is not None:
            return self.streamer
        if self._s3_streamer is None:
            if not self.bucket:
                msg = "S3WarcStreamStage requires a bucket (or an injected streamer)"
                raise ValueError(msg)
            self._s3_streamer = S3ObjectStreamer(
                self.bucket, endpoint_url=self.endpoint_url, region=self.region
            )
        return self._s3_streamer

    def process(self, task: FileGroupTask) -> DocumentBatch:
        streamer = self._get_streamer()
        rows: list[dict[str, Any]] = []

        for key in task.data:
            count = 0
            try:
                stream = streamer.open(key)
                try:
                    for raw_record in iterate_pdf_warc_stream(stream, source_name=os.path.basename(key)):
                        extracted = extract_pdf_record(raw_record)
                        if extracted is None:
                            continue
                        if self.add_filename_column:
                            extracted[self.filename_col] = os.path.basename(key)
                        rows.append(extracted)
                        count += 1
                        if self.record_limit and count >= self.record_limit:
                            break
                finally:
                    close = getattr(stream, "close", None)
                    if callable(close):
                        close()
            except Exception:  # noqa: BLE001
                logger.exception(f"Failed streaming WARC object {key}")
                continue

        return DocumentBatch(
            dataset_name=task.dataset_name,
            data=pd.DataFrame(rows, columns=self.outputs()[1]),
            _metadata={**task._metadata},
            _stage_perf=task._stage_perf,
        )


class S3StreamEaiCrawlStage(CompositeStage[EmptyTask, DocumentBatch]):
    """List S3/SwiftStack WARC objects and collect PDF URLs by streaming each object."""

    def __init__(  # noqa: PLR0913
        self,
        bucket: str,
        prefix: str = "",
        suffix: str = ".warc.gz",
        endpoint_url: str | None = None,
        region: str | None = None,
        url_limit: int | None = None,
        record_limit: int | None = None,
        add_filename_column: bool | str = True,
    ) -> None:
        super().__init__()
        self.url_generator = S3WarcUrlGenerator(
            bucket=bucket,
            prefix=prefix,
            suffix=suffix,
            limit=url_limit,
            endpoint_url=endpoint_url,
            region=region,
        )
        self.stages = [
            URLGenerationStage(url_generator=self.url_generator, limit=url_limit),
            S3WarcStreamStage(
                bucket=bucket,
                endpoint_url=endpoint_url,
                region=region,
                record_limit=record_limit,
                add_filename_column=add_filename_column,
            ),
        ]
        self.name = "s3_stream_eai_crawl_pdf_extract"

    def decompose(self) -> list[ProcessingStage]:
        return self.stages

    def get_description(self) -> str:
        return "Collect PDF URLs/metadata by streaming application/pdf records from S3/SwiftStack .warc.gz objects"
