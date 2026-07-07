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

"""Fetch only the metadata bytes of WARC PDF records from S3 (or any range source).

A WARC response record is laid out as::

    WARC headers (WARC-Target-URI, WARC-Record-ID, WARC-Date, Content-Length, ...)
    <blank line>
    HTTP headers (Content-Type, Content-Length, ...)
    <blank line>
    HTTP body  <-- the PDF bytes; we never read these

All the metadata we want lives in the two header blocks at the *start* of each
record. This module fetches only those leading bytes via HTTP range requests,
so the (potentially multi-MB) PDF body is never transferred.

Two access modes are supported:

* ``WarcMetadataScanner.iter_pdf_metadata`` — sequential scan of an *uncompressed*
  WARC. Uses the WARC ``Content-Length`` to compute each record's size and skips
  the body by jumping the byte offset forward. No external index required.

* ``WarcMetadataScanner.read_record_metadata`` — random access to a single record
  at a known byte ``offset`` (e.g. from a CDX index). Works for ``.warc`` and
  per-record-gzipped ``.warc.gz`` (the CC layout), decompressing only the
  truncated header prefix.
"""

from __future__ import annotations

import logging
import os
import zlib
from typing import TYPE_CHECKING, Any

from tutorials.eai_crawl.pdf_records import PDF_CONTENT_TYPE, extract_pdf_record

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

DEFAULT_HEADER_BYTES = 16 * 1024
MAX_HEADER_BYTES = 256 * 1024
RECORD_SEPARATOR = b"\r\n\r\n"


class RangeReader:
    """Base class for byte-range readers. Tracks total bytes transferred."""

    def __init__(self) -> None:
        self.bytes_read = 0

    def size(self, key: str) -> int:
        raise NotImplementedError

    def _read(self, key: str, start: int, end: int) -> bytes:
        raise NotImplementedError

    def read(self, key: str, start: int, end: int) -> bytes:
        """Read inclusive byte range ``[start, end]`` and count the bytes."""
        data = self._read(key, start, end)
        self.bytes_read += len(data)
        return data


class S3RangeReader(RangeReader):
    """Range reader backed by S3 (or S3-compatible) storage via boto3 (lazy-imported).

    Credentials and region are resolved by boto3's standard chain (``AWS_*`` env
    vars, ``~/.aws/`` config, instance profiles). For S3-compatible object stores
    (SwiftStack, MinIO, ...) set ``endpoint_url`` or the ``AWS_ENDPOINT_URL`` env
    var, e.g. ``endpoint_url="https://pdx.s8k.io"``.

    Note: range-reading record metadata only works on *uncompressed* WARCs (or
    per-record-gzipped WARCs with a CDX offset index). For whole-file ``.warc.gz``
    use the streaming path (``s3_streaming.S3WarcStreamStage``) instead.
    """

    def __init__(
        self,
        bucket: str,
        *,
        client: Any = None,  # noqa: ANN401 - boto3 S3 client has no type stubs
        endpoint_url: str | None = None,
        region: str | None = None,
        max_retries: int = 3,
        timeout: int = 30,
    ) -> None:
        super().__init__()
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
                msg = "boto3 is required for S3RangeReader. Install with: pip install boto3"
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

    def size(self, key: str) -> int:
        resp = self._get_client().head_object(Bucket=self.bucket, Key=key)
        return int(resp["ContentLength"])

    def _read(self, key: str, start: int, end: int) -> bytes:
        resp = self._get_client().get_object(
            Bucket=self.bucket,
            Key=key,
            Range=f"bytes={start}-{end}",
        )
        return resp["Body"].read()


class LocalFileRangeReader(RangeReader):
    """Range reader backed by a local file (handy for testing the S3 logic)."""

    def __init__(self, base_dir: str = "") -> None:
        super().__init__()
        self.base_dir = base_dir

    def _path(self, key: str) -> str:
        return os.path.join(self.base_dir, key) if self.base_dir else key

    def size(self, key: str) -> int:
        return os.path.getsize(self._path(key))

    def _read(self, key: str, start: int, end: int) -> bytes:
        with open(self._path(key), "rb") as handle:
            handle.seek(start)
            return handle.read(end - start + 1)


class BytesRangeReader(RangeReader):
    """Range reader backed by in-memory bytes (used by tests)."""

    def __init__(self, data: bytes | dict[str, bytes]) -> None:
        super().__init__()
        # Single-bytes mode responds to any key; dict mode keys per object.
        self._single = data if isinstance(data, bytes) else None
        self._data = None if isinstance(data, bytes) else dict(data)

    def _get(self, key: str) -> bytes:
        return self._single if self._single is not None else self._data[key]

    def size(self, key: str) -> int:
        return len(self._get(key))

    def _read(self, key: str, start: int, end: int) -> bytes:
        return self._get(key)[start : end + 1]


def _gunzip_prefix(raw: bytes) -> bytes:
    """Decompress as much of a (possibly truncated) gzip member as possible."""
    decompressor = zlib.decompressobj(wbits=31)
    try:
        return decompressor.decompress(raw)
    except zlib.error:
        return b""


def _split_header_blocks(data: bytes) -> tuple[bytes | None, bytes | None]:
    """Split decompressed bytes into (WARC header block, HTTP header block).

    Returns ``None`` for a block that is not fully present in ``data``.
    """
    first = data.find(RECORD_SEPARATOR)
    if first == -1:
        return None, None
    warc_block = data[:first]
    rest = data[first + len(RECORD_SEPARATOR) :]
    second = rest.find(RECORD_SEPARATOR)
    if second == -1:
        return warc_block, None
    return warc_block, rest[:second]


def _parse_block(block: bytes) -> tuple[str, dict[str, str]]:
    """Parse a header block into (first line, lowercased header dict)."""
    lines = block.split(b"\r\n")
    first_line = lines[0].decode("latin-1") if lines else ""
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if b":" in line:
            key, _, value = line.partition(b":")
            headers[key.decode("latin-1").strip().lower()] = value.decode("latin-1").strip()
    return first_line, headers


def _build_record(key: str, warc_headers: dict[str, str], http_first: str, http_headers: dict[str, str]) -> dict[str, Any]:
    rec_id = warc_headers.get("warc-record-id", "")
    warc_id = rec_id[10:-1] if rec_id.startswith("<urn:uuid:") else rec_id.strip("<>")

    http_content_length = http_headers.get("content-length")
    content_length = int(http_content_length) if http_content_length and http_content_length.isdigit() else None

    status = ""
    status_parts = http_first.split(" ")
    if len(status_parts) >= 2:  # noqa: PLR2004
        status = status_parts[1]

    return {
        "url": warc_headers.get("warc-target-uri"),
        "warc_id": warc_id,
        "source_id": key.split("/")[-1],
        "content_type": http_headers.get("content-type", "").split(";", 1)[0].strip().lower(),
        "content_length": content_length,
        "http_status": status,
        "warc_date": warc_headers.get("warc-date"),
    }


class WarcMetadataScanner:
    """Read WARC PDF metadata using range requests, skipping PDF payloads."""

    def __init__(
        self,
        reader: RangeReader,
        *,
        header_bytes: int = DEFAULT_HEADER_BYTES,
        max_header_bytes: int = MAX_HEADER_BYTES,
    ) -> None:
        self.reader = reader
        self.header_bytes = header_bytes
        self.max_header_bytes = max_header_bytes

    @property
    def bytes_read(self) -> int:
        return self.reader.bytes_read

    def iter_pdf_metadata(self, key: str) -> Iterator[dict[str, Any]]:
        """Sequentially scan an *uncompressed* WARC, yielding PDF record metadata.

        Only the leading ``header_bytes`` of each record are fetched; the body is
        skipped by advancing the byte offset using the WARC ``Content-Length``.
        """
        size = self.reader.size(key)
        offset = 0

        while offset < size:
            window = min(self.header_bytes, size - offset)
            chunk = self.reader.read(key, offset, offset + window - 1)
            warc_block, http_block = _split_header_blocks(chunk)

            # Grow the window if the WARC header block did not fit (rare).
            while warc_block is None and window < self.max_header_bytes and offset + window < size:
                window = min(window * 2, self.max_header_bytes, size - offset)
                chunk = self.reader.read(key, offset, offset + window - 1)
                warc_block, http_block = _split_header_blocks(chunk)

            if warc_block is None:
                logger.error(f"Could not parse WARC header at offset {offset} in {key}")
                return

            _, warc_headers = _parse_block(warc_block)
            try:
                warc_content_length = int(warc_headers["content-length"])
            except (KeyError, ValueError):
                logger.exception(f"Missing/invalid WARC Content-Length at offset {offset} in {key}")
                return

            warc_header_len = len(warc_block) + len(RECORD_SEPARATOR)

            if warc_headers.get("warc-type") == "response":
                record = self._maybe_build_response(key, warc_headers, http_block, chunk, offset)
                if record is not None:
                    yield record

            offset += warc_header_len + warc_content_length + len(RECORD_SEPARATOR)

    def _maybe_build_response(
        self,
        key: str,
        warc_headers: dict[str, str],
        http_block: bytes | None,
        chunk: bytes,
        offset: int,
    ) -> dict[str, Any] | None:
        # Re-fetch a larger header window if the HTTP headers were not fully captured.
        window = len(chunk)
        while http_block is None and window < self.max_header_bytes:
            window = min(window * 2, self.max_header_bytes)
            chunk = self.reader.read(key, offset, offset + window - 1)
            _, http_block = _split_header_blocks(chunk)

        if http_block is None:
            logger.warning(f"Could not parse HTTP headers at offset {offset} in {key}")
            return None

        http_first, http_headers = _parse_block(http_block)
        if PDF_CONTENT_TYPE not in http_headers.get("content-type", "").lower():
            return None

        return _build_record(key, warc_headers, http_first, http_headers)

    def read_record_metadata(self, key: str, offset: int, *, compressed: bool = True) -> dict[str, Any] | None:
        """Read a single record's metadata at byte ``offset`` (CDX-index style).

        Set ``compressed=True`` for per-record-gzipped ``.warc.gz`` (CC layout).
        Only ``header_bytes`` are fetched and (for gzip) only that prefix is
        decompressed, so the PDF body is never transferred.
        """
        window = self.header_bytes
        warc_block: bytes | None = None
        http_block: bytes | None = None

        while True:
            raw = self.reader.read(key, offset, offset + window - 1)
            data = _gunzip_prefix(raw) if compressed else raw
            warc_block, http_block = _split_header_blocks(data)
            if (warc_block is not None and http_block is not None) or window >= self.max_header_bytes:
                break
            window = min(window * 2, self.max_header_bytes)

        if warc_block is None or http_block is None:
            logger.warning(f"Could not parse headers at offset {offset} in {key}")
            return None

        _, warc_headers = _parse_block(warc_block)
        if warc_headers.get("warc-type") != "response":
            return None

        http_first, http_headers = _parse_block(http_block)
        if PDF_CONTENT_TYPE not in http_headers.get("content-type", "").lower():
            return None

        return _build_record(key, warc_headers, http_first, http_headers)


def scan_pdf_urls(
    reader: RangeReader,
    key: str,
    *,
    record_limit: int | None = None,
    header_bytes: int = DEFAULT_HEADER_BYTES,
) -> list[dict[str, Any]]:
    """Convenience: scan an uncompressed WARC and return extractor output rows."""
    scanner = WarcMetadataScanner(reader, header_bytes=header_bytes)
    rows: list[dict[str, Any]] = []
    for raw_record in scanner.iter_pdf_metadata(key):
        extracted = extract_pdf_record(raw_record)
        if extracted is None:
            continue
        rows.append(extracted)
        if record_limit is not None and len(rows) >= record_limit:
            break
    return rows
