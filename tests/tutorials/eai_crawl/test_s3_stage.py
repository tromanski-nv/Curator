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

from nemo_curator.tasks import FileGroupTask
from tests.tutorials.eai_crawl.test_s3_download import _make_record
from tutorials.eai_crawl.s3_download import BytesRangeReader
from tutorials.eai_crawl.s3_stage import S3EaiCrawlStage, S3WarcMetadataStage
from tutorials.eai_crawl.s3_url_generation import S3WarcUrlGenerator


class _FakePaginator:
    def __init__(self, keys: list[str]) -> None:
        self._keys = keys

    def paginate(self, **_: object) -> list[dict]:
        # Two pages to exercise pagination handling.
        mid = len(self._keys) // 2
        return [
            {"Contents": [{"Key": k} for k in self._keys[:mid]]},
            {"Contents": [{"Key": k} for k in self._keys[mid:]]},
        ]


class _FakeS3Client:
    def __init__(self, keys: list[str]) -> None:
        self._keys = keys

    def get_paginator(self, _name: str) -> _FakePaginator:
        return _FakePaginator(self._keys)


class TestS3WarcUrlGenerator:
    def test_lists_and_filters_warc_keys(self) -> None:
        client = _FakeS3Client(
            ["crawl/b.warc.gz", "crawl/a.warc.gz", "crawl/readme.txt", "crawl/c.warc"],
        )
        generator = S3WarcUrlGenerator(bucket="bkt", prefix="crawl/", client=client)

        keys = generator.generate_urls()

        assert keys == ["crawl/a.warc.gz", "crawl/b.warc.gz", "crawl/c.warc"]

    def test_limit_applied(self) -> None:
        client = _FakeS3Client(["a.warc", "b.warc", "c.warc"])
        generator = S3WarcUrlGenerator(bucket="bkt", limit=2, client=client)

        assert generator.generate_urls() == ["a.warc", "b.warc"]


class TestS3WarcMetadataStage:
    def test_process_emits_pdf_metadata_batch(self) -> None:
        warc = b"".join(
            [
                _make_record(
                    rec_type="response",
                    content_type="text/html",
                    body=b"<html>skip</html>",
                    record_id="h1",
                    uri="http://example.com/p.html",
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

        stage = S3WarcMetadataStage(reader=BytesRangeReader(warc), header_bytes=8192)
        task = FileGroupTask(dataset_name="eai", data=["crawl/sample.warc"], _metadata={})

        batch = stage.process(task)
        df = batch.to_pandas()

        assert len(df) == 1
        row = df.iloc[0]
        assert row["url"] == "http://example.com/a.pdf"
        assert row["id"] == "pdf1"
        assert row["filename"] == "a.pdf"
        assert row["file_name"] == "sample.warc"
        assert "content" not in df.columns


class TestS3EaiCrawlStage:
    def test_decomposition(self) -> None:
        stage = S3EaiCrawlStage(bucket="bkt", prefix="crawl/")
        stages = stage.decompose()

        assert len(stages) == 2
        assert isinstance(stages[1], S3WarcMetadataStage)
        assert stage.name == "s3_eai_crawl_pdf_extract"
