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

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

logger = logging.getLogger(__name__)

PDF_CONTENT_TYPE = "application/pdf"

PDF_RECORD_COLUMNS = [
    "url",
    "warc_id",
    "source_id",
    "content_type",
    "content_length",
    "http_status",
    "warc_date",
]

PDF_OUTPUT_COLUMNS = [
    "url",
    "id",
    "warc_id",
    "source_id",
    "content_type",
    "content_length",
    "http_status",
    "warc_date",
    "filename",
]


def iterate_pdf_warc_records(file_path: str) -> Iterator[dict[str, Any]]:
    """Yield application/pdf WARC response metadata without reading PDF payloads."""
    from warcio.archiveiterator import ArchiveIterator

    filename = file_path.name if isinstance(file_path, Path) else file_path.split("/")[-1]
    num_records = 0

    with open(file_path, "rb") as file_pointer:
        archive_iterator = ArchiveIterator(file_pointer, arc2warc=True)
        while True:
            try:
                record = next(archive_iterator)
                if record.rec_type != "response":
                    continue

                http_headers = record.http_headers
                if http_headers is None:
                    continue

                content_type = (http_headers.get_header("Content-Type") or "").lower()
                if PDF_CONTENT_TYPE not in content_type:
                    continue

                warc_record_id = record.rec_headers.get_header("WARC-Record-ID")
                if warc_record_id is None:
                    logger.warning(f"Skipping PDF response without WARC-Record-ID in {filename}")
                    continue

                content_length = record.payload_length
                if content_length is None:
                    raw_length = http_headers.get_header("Content-Length")
                    content_length = int(raw_length) if raw_length and raw_length.isdigit() else None

                warc_id = warc_record_id[10:-1]
                url = record.rec_headers.get_header("WARC-Target-URI")
                yield {
                    "url": url,
                    "warc_id": warc_id,
                    "source_id": filename,
                    "content_type": content_type.split(";", 1)[0].strip(),
                    "content_length": content_length,
                    "http_status": http_headers.get_statuscode(),
                    "warc_date": record.rec_headers.get_header("WARC-Date"),
                }
                num_records += 1
            except StopIteration:
                break
            except Exception:
                logger.exception(f"Error processing record {num_records} in {filename}")
                continue


def filename_from_url(url: str) -> str:
    path = urlparse(url).path
    if not path or path.endswith("/"):
        return ""
    return unquote(path.rsplit("/", 1)[-1])


def extract_pdf_record(record: dict[str, Any]) -> dict[str, Any] | None:
    """Map a PDF WARC record to output URL/metadata fields."""
    url = record.get("url")
    if not url:
        return None

    return {
        "url": url,
        "id": record["warc_id"],
        "warc_id": record["warc_id"],
        "source_id": record["source_id"],
        "content_type": record.get("content_type", PDF_CONTENT_TYPE),
        "content_length": record.get("content_length"),
        "http_status": record.get("http_status"),
        "warc_date": record.get("warc_date"),
        "filename": filename_from_url(url),
    }
