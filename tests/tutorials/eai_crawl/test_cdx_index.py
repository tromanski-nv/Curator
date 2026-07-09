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
import io

from tests.tutorials.eai_crawl.test_s3_download import _make_record
from tutorials.eai_crawl.cdx_index import iterate_cdx_and_pdfs


def _two_record_warc() -> bytes:
    return b"".join(
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
                body=b"%PDF-1.4 body bytes",
                record_id="pdf1",
                uri="http://example.com/a.pdf",
            ),
        ]
    )


def _per_record_gzip(warc: bytes) -> bytes:
    """Compress each WARC record as its own gzip member (CC layout)."""
    # Split on WARC record boundaries is hard; instead gzip the whole file as
    # concatenated members by compressing each record blob separately.
    html, pdf = warc[: warc.find(b"WARC/1.0", 1)], warc[warc.find(b"WARC/1.0", 1) :]
    return gzip.compress(html) + gzip.compress(pdf)


class TestIterateCdxAndPdfs:
    def test_uncompressed_emits_cdx_and_pdf_rows(self) -> None:
        warc = _two_record_warc()
        result = iterate_cdx_and_pdfs(io.BytesIO(warc), warc_filename="sample.warc", detect_layout=True)

        assert result.gzip_layout == "uncompressed"
        assert result.num_responses == 2
        assert len(result.cdx_rows) == 2
        assert len(result.pdf_rows) == 1

        pdf_cdx = [r for r in result.cdx_rows if r["is_pdf"]][0]
        assert pdf_cdx["url"] == "http://example.com/a.pdf"
        assert pdf_cdx["warc_id"] == "pdf1"
        assert isinstance(pdf_cdx["warc_record_offset"], int)
        assert pdf_cdx["warc_record_length"] > 0

        pdf = result.pdf_rows[0]
        assert pdf["url"] == "http://example.com/a.pdf"
        assert pdf["warc_record_offset"] == pdf_cdx["warc_record_offset"]
        assert pdf["warc_record_length"] == pdf_cdx["warc_record_length"]

    def test_per_record_gzip_layout_detected(self) -> None:
        warc = _per_record_gzip(_two_record_warc())
        result = iterate_cdx_and_pdfs(io.BytesIO(warc), warc_filename="sample.warc.gz", detect_layout=True)

        assert result.gzip_layout == "per_record"
        assert len(result.cdx_rows) == 2
        assert len(result.pdf_rows) == 1
        # Offsets should be distinct compressed member starts.
        offsets = [r["warc_record_offset"] for r in result.cdx_rows]
        assert offsets[0] == 0
        assert offsets[1] > offsets[0]
