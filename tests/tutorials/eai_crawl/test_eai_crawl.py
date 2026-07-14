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

from pathlib import Path

from tutorials.eai_crawl.pdf_records import extract_pdf_record, iterate_pdf_warc_records
from tutorials.eai_crawl.stage import EaiCrawlDownloadExtractStage
from tutorials.eai_crawl.url_generation import LocalWarcUrlGenerator


def _make_warc_record(
    *,
    rec_type: str,
    content_type: str,
    content: bytes,
    record_id: str,
    target_uri: str,
) -> bytes:
    http_response = (
        f"HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {len(content)}\r\n\r\n".encode()
        + content
        + b"\r\n"
    )
    content_length = len(http_response)
    header = (
        f"WARC/1.0\r\n"
        f"WARC-Type: {rec_type}\r\n"
        f"WARC-Record-ID: <urn:uuid:{record_id}>\r\n"
        f"WARC-Date: 2022-01-01T00:00:00Z\r\n"
        f"WARC-Target-URI: {target_uri}\r\n"
        f"Content-Length: {content_length}\r\n\r\n"
    ).encode()
    return header + http_response + b"\r\n\r\n"


class TestLocalWarcUrlGenerator:
    def test_generate_urls_from_explicit_paths(self, tmp_path: Path) -> None:
        warc_path = tmp_path / "sample.warc.gz"
        warc_path.write_bytes(b"placeholder")

        generator = LocalWarcUrlGenerator(warc_paths=[str(warc_path)])
        urls = generator.generate_urls()

        assert urls == [str(warc_path.resolve())]


class TestEaiWarcIterator:
    def test_yields_only_application_pdf_responses(self, tmp_path: Path) -> None:
        warc_path = tmp_path / "mixed.warc"
        pdf_bytes = b"%PDF-1.4 fake pdf payload"
        warc_path.write_bytes(
            b"".join(
                [
                    _make_warc_record(
                        rec_type="response",
                        content_type="text/html",
                        content=b"<html>skip me</html>",
                        record_id="html123",
                        target_uri="http://example.com/page.html",
                    ),
                    _make_warc_record(
                        rec_type="response",
                        content_type="application/pdf",
                        content=pdf_bytes,
                        record_id="pdf123",
                        target_uri="http://example.com/doc.pdf",
                    ),
                ]
            )
        )

        records = list(iterate_pdf_warc_records(str(warc_path)))

        assert len(records) == 1
        assert records[0]["url"] == "http://example.com/doc.pdf"
        assert records[0]["warc_id"] == "pdf123"
        assert records[0]["content_type"] == "application/pdf"
        assert records[0]["http_status"] == "200"
        assert records[0]["warc_date"] == "2022-01-01T00:00:00Z"
        assert "content" not in records[0]


class TestEaiPdfExtractor:
    def test_returns_url_and_metadata_without_pdf_body(self) -> None:
        record = {
            "url": "http://example.com/doc.pdf",
            "warc_id": "pdf123",
            "source_id": "sample.warc.gz",
            "content_type": "application/pdf",
            "content_length": 12345,
            "http_status": "200",
            "warc_date": "2022-01-01T00:00:00Z",
        }

        result = extract_pdf_record(record)

        assert result is not None
        assert result["url"] == "http://example.com/doc.pdf"
        assert result["warc_id"] == "pdf123"
        assert result["filename"] == "doc.pdf"
        assert result["content_length"] == 12345
        assert "id" not in result
        assert "source_id" not in result
        assert "content" not in result
        assert "text" not in result

    def test_skips_records_without_url(self) -> None:
        record = {
            "warc_id": "pdf123",
            "source_id": "sample.warc.gz",
            "content_type": "application/pdf",
        }

        assert extract_pdf_record(record) is None


class TestEaiCrawlDownloadExtractStage:
    def test_stage_decomposition(self, tmp_path: Path) -> None:
        stage = EaiCrawlDownloadExtractStage(
            download_dir=str(tmp_path / "downloads"),
            warc_paths=[str(tmp_path / "sample.warc.gz")],
        )

        stages = stage.decompose()

        assert len(stages) == 3
        assert stage.name == "eai_crawl_pdf_extract"
