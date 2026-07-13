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
object through ``warcio`` and extracts PDF URL/metadata on the fly. Optionally
also writes a per-WARC CDX-style index (``offset`` / ``length``) for later O(1)
range-fetch of filtered records.

``URLGenerationStage`` emits deterministic groups of WARC objects. Grouping
reduces output-file count while preserving source-level retry boundaries.

This is the correct path for the EssentialAI crawl layout:
``s3://<bucket>/eai-warc/<YYYYMMDD>/<uuid>.warc.gz``.

For S3-compatible stores (SwiftStack), set ``endpoint_url`` (or the
``AWS_ENDPOINT_URL`` env var); credentials come from the boto3 default chain
(``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` / ``AWS_SESSION_TOKEN``).
Path-style addressing is used (required by many S3-compatible endpoints).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

import pandas as pd
from loguru import logger

from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.download.base.url_generation import URLGenerationStage
from nemo_curator.tasks import DocumentBatch, EmptyTask, FailedTask, FileGroupTask
from tutorials.eai_crawl.cdx_index import CDX_COLUMNS, iterate_cdx_and_pdfs
from tutorials.eai_crawl.s3_storage import ensure_parent, is_remote_url, write_parquet
from tutorials.eai_crawl.s3_url_generation import S3WarcUrlGenerator

# Trimmed PDF-URL output schema. Dropped vs. the raw record: ``id`` (identical to
# ``warc_id``), ``source_id``/``file_name`` (both just the basename of
# ``warc_filename``), and ``content_type`` (always ``application/pdf`` here).
# ``filename`` is kept for convenience; ``warc_filename`` + offset + length are the
# coords needed to range-fetch the PDF later.
PDF_INDEX_COLUMNS = [
    "url",
    "warc_id",
    "content_length",
    "http_status",
    "warc_date",
    "filename",
    "warc_filename",
    "warc_record_offset",
    "warc_record_length",
]


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

    def __init__(  # noqa: PLR0913
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
                s3={"addressing_style": "path"},
                signature_version="s3v4",
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
    """Stream ``.warc.gz`` objects; write PDF URL rows and CDX offsets.

    Each input task contains a deterministic group of WARC keys. Both outputs
    are buffered only for that group and written to deterministic
    ``part-<group-hash>-<seq>.parquet`` names before ``process`` returns. Native
    resumability therefore marks a group complete only after its output is
    durable, while retries overwrite rather than append duplicate parts.

    When ``pdf_output_dir`` is set the stage is **terminal**: it writes PDF parts
    itself and returns ``None`` (no ``DocumentBatch`` flows downstream), which
    keeps the RayActorPool driver's memory bounded regardless of chunk size (the
    executor otherwise materializes every intermediate task on the head node). If
    ``pdf_output_dir`` is unset it falls back to returning a ``DocumentBatch`` so
    the stage stays composable with a downstream writer.

    A failed WARC returns ``FailedTask`` for the whole group. Successfully parsed
    rows may already have been written, but the stable paths are overwritten when
    that same source group is retried.
    """

    bucket: str | None = None
    endpoint_url: str | None = None
    region: str | None = None
    record_limit: int | None = None
    # PDF index output. When set, PDF rows are buffered + written here per worker
    # and the stage becomes terminal (returns None). When unset, the stage returns
    # a DocumentBatch for a downstream writer instead.
    pdf_output_dir: str | None = None
    pdf_storage_options: dict[str, Any] | None = None
    # Consolidated PDF file target, in rows within one source group.
    pdf_rows_per_file: int = 2_000_000
    cdx_output_dir: str | None = None
    cdx_storage_options: dict[str, Any] | None = None
    # Consolidated CDX file target within one source group.
    cdx_rows_per_file: int = 2_000_000
    # file_name is redundant with warc_filename's basename; off by default.
    add_filename_column: bool | str = False
    # Per-WARC task is I/O-bound (boto3 stream + gzip), not CPU-bound. Lowering the
    # CPU reservation lets the backend pack more concurrent streams per node (e.g.
    # cpus=0.25 -> ~4 streams/core) to better saturate the NIC. Default 1.0 keeps
    # prior behavior.
    stream_cpus: float = 1.0
    streamer: ObjectStreamer | None = None  # inject for tests; else S3ObjectStreamer(bucket)
    name: str = "s3_warc_pdf_stream"
    resources = Resources(cpus=1.0)

    def __post_init__(self) -> None:
        self.filename_col = self.add_filename_column if isinstance(self.add_filename_column, str) else "file_name"
        self._s3_streamer: ObjectStreamer | None = None
        # Instance-level override of the class-default resources (base reads self.resources).
        self.resources = Resources(cpus=self.stream_cpus)
        for out_dir in (self.pdf_output_dir, self.cdx_output_dir):
            if out_dir and not is_remote_url(out_dir):
                ensure_parent(out_dir)

    def inputs(self) -> tuple[list[str], list[str]]:
        return (["data"], [])

    def outputs(self) -> tuple[list[str], list[str]]:
        return (["data"], self._pdf_columns())

    def _pdf_columns(self) -> list[str]:
        cols = list(PDF_INDEX_COLUMNS)
        if self.add_filename_column:
            cols.append(self.filename_col)
        return cols

    def _get_streamer(self) -> ObjectStreamer:
        if self.streamer is not None:
            return self.streamer
        if self._s3_streamer is None:
            if not self.bucket:
                msg = "S3WarcStreamStage requires a bucket (or an injected streamer)"
                raise ValueError(msg)
            self._s3_streamer = S3ObjectStreamer(self.bucket, endpoint_url=self.endpoint_url, region=self.region)
        return self._s3_streamer

    def _init_buffers(self, group_tag: str) -> None:
        # Per-source-group state. It never survives a process() boundary.
        self._pdf_buffer: list[dict[str, Any]] = []
        self._pdf_buffer_rows = 0
        self._pdf_flush_seq = 0
        self._cdx_buffer: list[dict[str, Any]] = []
        self._cdx_buffer_rows = 0
        self._cdx_flush_seq = 0
        self._group_tag = group_tag

    def _part_path(self, out_dir: str, seq: int) -> str:
        name = f"part-{self._group_tag}-{seq:05d}.parquet"
        if is_remote_url(out_dir):
            return out_dir.rstrip("/") + f"/{name}"
        return str(Path(out_dir) / name)

    def _buffer_pdf(self, pdf_rows: list[dict[str, Any]]) -> None:
        if not self.pdf_output_dir or not pdf_rows:
            return
        self._pdf_buffer.extend(pdf_rows)
        self._pdf_buffer_rows += len(pdf_rows)
        if self._pdf_buffer_rows >= self.pdf_rows_per_file:
            self._flush_pdf()

    def _flush_pdf(self, *, force: bool = False) -> None:
        if not self.pdf_output_dir:
            return
        if not self._pdf_buffer or (not force and self._pdf_buffer_rows < self.pdf_rows_per_file):
            return
        out = self._part_path(self.pdf_output_dir, self._pdf_flush_seq)
        write_parquet(
            pd.DataFrame(self._pdf_buffer, columns=self._pdf_columns()),
            out,
            storage_options=self.pdf_storage_options,
        )
        logger.info(f"Flushed {self._pdf_buffer_rows} PDF row(s) -> {out}")
        self._pdf_flush_seq += 1
        self._pdf_buffer = []
        self._pdf_buffer_rows = 0

    def _buffer_cdx(self, cdx_rows: list[dict[str, Any]]) -> None:
        if not self.cdx_output_dir or not cdx_rows:
            return
        self._cdx_buffer.extend(cdx_rows)
        self._cdx_buffer_rows += len(cdx_rows)
        if self._cdx_buffer_rows >= self.cdx_rows_per_file:
            self._flush_cdx()

    def _flush_cdx(self, *, force: bool = False) -> None:
        if not self.cdx_output_dir:
            return
        if not self._cdx_buffer or (not force and self._cdx_buffer_rows < self.cdx_rows_per_file):
            return
        out = self._part_path(self.cdx_output_dir, self._cdx_flush_seq)
        write_parquet(
            pd.DataFrame(self._cdx_buffer, columns=CDX_COLUMNS),
            out,
            storage_options=self.cdx_storage_options,
        )
        logger.info(f"Flushed {self._cdx_buffer_rows} CDX row(s) -> {out}")
        self._cdx_flush_seq += 1
        self._cdx_buffer = []
        self._cdx_buffer_rows = 0

    def process(self, task: FileGroupTask) -> DocumentBatch | FailedTask | None:
        streamer = self._get_streamer()
        group_tag = task.get_deterministic_id()
        if not group_tag:
            msg = "FileGroupTask must provide a deterministic ID for resumable output"
            raise ValueError(msg)
        self._init_buffers(group_tag)
        # Terminal mode: self-write PDF parts and return None (nothing flows to the
        # driver). Composable mode (no pdf_output_dir): accumulate + return a batch.
        self_writing = bool(self.pdf_output_dir)
        rows: list[dict[str, Any]] = []
        failed = False

        for key in task.data:
            try:
                stream = streamer.open(key)
                try:
                    result = iterate_cdx_and_pdfs(
                        stream,
                        warc_filename=key,
                        pdf_record_limit=self.record_limit,
                    )
                finally:
                    close = getattr(stream, "close", None)
                    if callable(close):
                        close()

                self._buffer_cdx(result.cdx_rows)
                if self.add_filename_column:
                    base = os.path.basename(key)
                    for extracted in result.pdf_rows:
                        extracted[self.filename_col] = base
                if self_writing:
                    self._buffer_pdf(result.pdf_rows)
                else:
                    rows.extend(result.pdf_rows)
                logger.info(
                    f"{key}: layout={result.gzip_layout} responses={result.num_responses} "
                    f"pdfs={len(result.pdf_rows)} cdx={len(result.cdx_rows)}"
                )
            except Exception:  # noqa: BLE001
                logger.exception(f"Failed streaming WARC object {key}")
                failed = True

        # Commit this source group's final partial files before the adapter can
        # decrement its resumability counter.
        self._flush_pdf(force=True)
        self._flush_cdx(force=True)
        if failed:
            return FailedTask()

        if self_writing:
            # PDF written as a side effect; emit nothing downstream.
            return None
        return DocumentBatch(
            dataset_name=task.dataset_name,
            data=pd.DataFrame(rows, columns=self._pdf_columns()),
            _metadata={**task._metadata, "source_files": list(task.data)},
            _stage_perf=task._stage_perf,
        )


class S3StreamEaiCrawlStage(CompositeStage[EmptyTask, DocumentBatch]):
    """List S3/SwiftStack WARC objects and collect PDF URLs by streaming each object.

    Fan-out is one Ray task per deterministic WARC group.
    """

    def __init__(  # noqa: PLR0913
        self,
        bucket: str,
        prefix: str = "",
        suffix: str = ".warc.gz",
        endpoint_url: str | None = None,
        region: str | None = None,
        url_limit: int | None = None,
        record_limit: int | None = None,
        pdf_output_dir: str | None = None,
        pdf_storage_options: dict[str, Any] | None = None,
        pdf_rows_per_file: int = 2_000_000,
        cdx_output_dir: str | None = None,
        cdx_storage_options: dict[str, Any] | None = None,
        add_filename_column: bool | str = False,
        keys: list[str] | None = None,
        warcs_per_task: int = 32,
        stream_cpus: float = 1.0,
        cdx_rows_per_file: int = 2_000_000,
    ) -> None:
        super().__init__()
        self.url_generator = S3WarcUrlGenerator(
            bucket=bucket,
            prefix=prefix,
            suffix=suffix,
            limit=url_limit,
            endpoint_url=endpoint_url,
            region=region,
            keys=keys,
        )
        self.stages = [
            URLGenerationStage(
                url_generator=self.url_generator,
                limit=url_limit,
                urls_per_task=warcs_per_task,
            ),
            S3WarcStreamStage(
                bucket=bucket,
                endpoint_url=endpoint_url,
                region=region,
                record_limit=record_limit,
                pdf_output_dir=pdf_output_dir,
                pdf_storage_options=pdf_storage_options,
                pdf_rows_per_file=pdf_rows_per_file,
                cdx_output_dir=cdx_output_dir,
                cdx_storage_options=cdx_storage_options,
                add_filename_column=add_filename_column,
                stream_cpus=stream_cpus,
                cdx_rows_per_file=cdx_rows_per_file,
            ),
        ]
        self.name = "s3_stream_eai_crawl_pdf_extract"

    def decompose(self) -> list[ProcessingStage]:
        return self.stages

    def get_description(self) -> str:
        return (
            "Collect PDF URLs (+ optional per-WARC CDX index) by streaming "
            "application/pdf records from S3/SwiftStack .warc.gz objects"
        )
