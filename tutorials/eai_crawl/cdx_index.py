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

"""One-pass CDX-style indexing + PDF URL extraction from a WARC stream.

While iterating with ``warcio.ArchiveIterator``, record
``(offset, length)`` for every response (or every record) and also emit the
PDF URL/metadata rows used by the existing EAI pipeline.

Offsets/lengths are only useful for O(1) range-GET when the ``.warc.gz`` is
**per-record gzip** (CC layout). Whole-file gzip still yields offsets into the
compressed object, but those slices are not independently decompressible.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import IO, Any

from tutorials.eai_crawl.pdf_records import PDF_CONTENT_TYPE, extract_pdf_record, filename_from_url

logger = logging.getLogger(__name__)

CDX_COLUMNS = [
    "url",
    "warc_filename",
    "warc_record_offset",
    "warc_record_length",
    "warc_id",
    "warc_type",
    "content_mime_type",
    "http_status",
    "content_length",
    "warc_date",
    "is_pdf",
]


@dataclass
class IndexPassResult:
    """Outputs of a single WARC stream pass."""

    cdx_rows: list[dict[str, Any]] = field(default_factory=list)
    pdf_rows: list[dict[str, Any]] = field(default_factory=list)
    num_records: int = 0
    num_responses: int = 0
    gzip_layout: str = "unknown"  # "per_record", "whole_file", "uncompressed", "unknown"


def _warc_id_from_header(record: Any) -> str | None:
    warc_record_id = record.rec_headers.get_header("WARC-Record-ID")
    if warc_record_id is None:
        return None
    if warc_record_id.startswith("<urn:uuid:") and warc_record_id.endswith(">"):
        return warc_record_id[10:-1]
    return warc_record_id.strip("<>")


def _http_fields(record: Any) -> tuple[str, str | None, int | None]:
    """Return (content_type, http_status, content_length)."""
    http_headers = record.http_headers
    if http_headers is None:
        return "", None, None

    content_type = (http_headers.get_header("Content-Type") or "").lower()
    content_type = content_type.split(";", 1)[0].strip()
    status = http_headers.get_statuscode()

    content_length = record.payload_length
    if content_length is None:
        raw_length = http_headers.get_header("Content-Length")
        content_length = int(raw_length) if raw_length and raw_length.isdigit() else None

    return content_type, status, content_length


def detect_gzip_layout(stream: IO[bytes]) -> tuple[str, IO[bytes]]:
    """Peek at a stream to classify gzip layout; return (layout, rewindable_stream).

    For non-seekable streams the peek bytes are prepended via a small wrapper so
    warcio can still consume the full object.
    """
    peek = stream.read(2)
    if len(peek) < 2:
        return "unknown", _PrefixedStream(peek, stream)

    if peek != b"\x1f\x8b":
        return "uncompressed", _PrefixedStream(peek, stream)

    # Read a bit more and try to find a second gzip member after the first.
    # For per-record gzip, member 1 ends quickly (one WARC record). For
    # whole-file gzip, the first member spans the entire object.
    extra = stream.read(1 << 20)  # 1 MiB peek window
    blob = peek + extra
    layout = _classify_gzip_blob(blob)
    return layout, _PrefixedStream(blob, stream)


def _classify_gzip_blob(blob: bytes) -> str:
    import zlib

    # Prefer zlib member boundaries over GzipFile.tell(), which is unreliable
    # for concatenated members.
    decompressor = zlib.decompressobj(wbits=31)
    try:
        decompressor.decompress(blob)
    except zlib.error:
        # Truncated member in the peek window — likely whole-file gzip (member
        # continues past our peek), or a truncated download.
        return "whole_file_or_truncated"

    unused = decompressor.unused_data
    if unused.startswith(b"\x1f\x8b"):
        return "per_record"
    if not unused:
        # First member consumed the entire peek — could still be per-record if
        # the first record is huge, but more often whole-file.
        return "whole_file_or_large_first_record"
    return "whole_file"


class _PrefixedStream:
    """File-like that yields ``prefix`` then delegates to ``inner``."""

    def __init__(self, prefix: bytes, inner: IO[bytes]) -> None:
        self._prefix = prefix
        self._pos = 0
        self._inner = inner

    def read(self, size: int = -1) -> bytes:
        if self._pos < len(self._prefix):
            if size < 0:
                head = self._prefix[self._pos :]
                self._pos = len(self._prefix)
                return head + self._inner.read()
            take = min(size, len(self._prefix) - self._pos)
            head = self._prefix[self._pos : self._pos + take]
            self._pos += take
            if take < size:
                return head + self._inner.read(size - take)
            return head
        return self._inner.read(size)

    def close(self) -> None:
        close = getattr(self._inner, "close", None)
        if callable(close):
            close()

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return False


def iterate_cdx_and_pdfs(
    stream: IO[bytes],
    *,
    warc_filename: str,
    responses_only: bool = True,
    pdf_record_limit: int | None = None,
    detect_layout: bool = True,
) -> IndexPassResult:
    """Single-pass: build CDX rows for (response) records and PDF URL rows.

    ``get_record_offset`` / ``get_record_length`` are called **after** each
    record is consumed (warcio computes member bounds in ``read_to_end``).
    """
    from warcio.archiveiterator import ArchiveIterator

    result = IndexPassResult()
    if detect_layout:
        result.gzip_layout, stream = detect_gzip_layout(stream)

    archive = ArchiveIterator(stream, arc2warc=True)
    pdf_count = 0

    while True:
        try:
            record = next(archive)
        except StopIteration:
            break
        except Exception:
            logger.exception(f"Error reading record {result.num_records} in {warc_filename}")
            continue

        result.num_records += 1
        rec_type = record.rec_type or ""

        # Always drain the record so offset/length are accurate.
        try:
            _ = record.content_stream().read()
        except Exception:
            logger.exception(f"Error draining record {result.num_records} in {warc_filename}")

        offset = archive.get_record_offset()
        length = archive.get_record_length()

        if responses_only and rec_type != "response":
            continue

        result.num_responses += 1
        content_type, http_status, content_length = _http_fields(record)
        url = record.rec_headers.get_header("WARC-Target-URI")
        warc_id = _warc_id_from_header(record)
        is_pdf = PDF_CONTENT_TYPE in content_type

        cdx_row = {
            "url": url,
            "warc_filename": warc_filename,
            "warc_record_offset": offset,
            "warc_record_length": length,
            "warc_id": warc_id,
            "warc_type": rec_type,
            "content_mime_type": content_type or None,
            "http_status": http_status,
            "content_length": content_length,
            "warc_date": record.rec_headers.get_header("WARC-Date"),
            "is_pdf": is_pdf,
        }
        result.cdx_rows.append(cdx_row)

        if not is_pdf:
            continue
        if warc_id is None:
            logger.warning(f"Skipping PDF response without WARC-Record-ID in {warc_filename}")
            continue

        raw = {
            "url": url,
            "warc_id": warc_id,
            "source_id": warc_filename.split("/")[-1],
            "content_type": content_type or PDF_CONTENT_TYPE,
            "content_length": content_length,
            "http_status": http_status,
            "warc_date": record.rec_headers.get_header("WARC-Date"),
        }
        extracted = extract_pdf_record(raw)
        if extracted is None:
            continue
        # Attach index coords so PDF rows can be joined without a second scan.
        extracted["warc_filename"] = warc_filename
        extracted["warc_record_offset"] = offset
        extracted["warc_record_length"] = length
        if not extracted.get("filename"):
            extracted["filename"] = filename_from_url(url or "")
        result.pdf_rows.append(extracted)
        pdf_count += 1
        if pdf_record_limit is not None and pdf_count >= pdf_record_limit:
            break

    return result


def iter_cdx_rows(stream: IO[bytes], warc_filename: str, **kwargs: Any) -> Iterator[dict[str, Any]]:
    """Convenience generator over CDX rows only."""
    yield from iterate_cdx_and_pdfs(stream, warc_filename=warc_filename, **kwargs).cdx_rows
