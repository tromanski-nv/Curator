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

import gzip
import os

from tutorials.eai_crawl.s3_download import (
    BytesRangeReader,
    WarcMetadataScanner,
    scan_pdf_urls,
)


def _make_record(*, rec_type: str, content_type: str, body: bytes, record_id: str, uri: str) -> bytes:
    http = (
        b"HTTP/1.1 200 OK\r\n"
        + f"Content-Type: {content_type}\r\n".encode()
        + f"Content-Length: {len(body)}\r\n".encode()
        + b"\r\n"
        + body
    )
    warc = (
        b"WARC/1.0\r\n"
        + f"WARC-Type: {rec_type}\r\n".encode()
        + f"WARC-Record-ID: <urn:uuid:{record_id}>\r\n".encode()
        + b"WARC-Date: 2022-01-01T00:00:00Z\r\n"
        + f"WARC-Target-URI: {uri}\r\n".encode()
        + f"Content-Length: {len(http)}\r\n".encode()
        + b"\r\n"
    )
    return warc + http + b"\r\n\r\n"


class TestUncompressedSequentialScan:
    def test_yields_pdf_metadata_and_skips_bodies(self) -> None:
        big_body = os.urandom(1_000_000)
        warc = b"".join(
            [
                _make_record(
                    rec_type="response",
                    content_type="text/html",
                    body=b"<html>skip</html>",
                    record_id="html1",
                    uri="http://example.com/page.html",
                ),
                _make_record(
                    rec_type="response",
                    content_type="application/pdf",
                    body=big_body,
                    record_id="pdf1",
                    uri="http://example.com/a.pdf",
                ),
                _make_record(
                    rec_type="response",
                    content_type="application/pdf; charset=binary",
                    body=big_body,
                    record_id="pdf2",
                    uri="http://example.com/sub/b.pdf",
                ),
            ]
        )

        reader = BytesRangeReader(warc)
        scanner = WarcMetadataScanner(reader, header_bytes=8192)

        records = list(scanner.iter_pdf_metadata(""))

        assert [r["url"] for r in records] == [
            "http://example.com/a.pdf",
            "http://example.com/sub/b.pdf",
        ]
        assert records[0]["warc_id"] == "pdf1"
        assert records[0]["content_type"] == "application/pdf"
        assert records[0]["content_length"] == len(big_body)
        assert records[0]["http_status"] == "200"
        assert records[0]["warc_date"] == "2022-01-01T00:00:00Z"

        # The key property: we transferred far fewer bytes than the full file.
        assert reader.bytes_read < 64 * 1024
        assert reader.bytes_read < len(warc) // 10

    def test_scan_pdf_urls_applies_extractor_and_limit(self) -> None:
        warc = b"".join(
            _make_record(
                rec_type="response",
                content_type="application/pdf",
                body=b"%PDF-1.4 body",
                record_id=f"pdf{i}",
                uri=f"http://example.com/doc{i}.pdf",
            )
            for i in range(5)
        )

        reader = BytesRangeReader(warc)
        rows = scan_pdf_urls(reader, "", record_limit=2)

        assert len(rows) == 2
        assert rows[0]["filename"] == "doc0.pdf"
        assert rows[0]["id"] == "pdf0"
        assert "content" not in rows[0]


class TestGzipIndexedRead:
    def test_reads_truncated_gzip_prefix_only(self) -> None:
        big_body = os.urandom(500_000)
        record = _make_record(
            rec_type="response",
            content_type="application/pdf",
            body=big_body,
            record_id="pdf1",
            uri="http://example.com/a.pdf",
        )
        compressed = gzip.compress(record)
        assert len(compressed) > 16 * 1024  # incompressible body keeps it large

        reader = BytesRangeReader(compressed)
        scanner = WarcMetadataScanner(reader, header_bytes=4096)

        result = scanner.read_record_metadata("warcs/sample.warc.gz", offset=0, compressed=True)

        assert result is not None
        assert result["url"] == "http://example.com/a.pdf"
        assert result["source_id"] == "sample.warc.gz"
        assert result["content_length"] == len(big_body)
        # Only the small header prefix was fetched, not the whole compressed record.
        assert reader.bytes_read == 4096
        assert reader.bytes_read < len(compressed)
