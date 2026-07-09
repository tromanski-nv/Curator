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
import sys
import types
from typing import IO

import pytest

from nemo_curator.tasks import FileGroupTask
from tests.tutorials.eai_crawl.test_s3_download import _make_record
from tutorials.eai_crawl.s3_streaming import (
    ObjectStreamer,
    S3ObjectStreamer,
    S3StreamEaiCrawlStage,
    S3WarcStreamStage,
)
from tutorials.eai_crawl.s3_url_generation import S3WarcUrlGenerator


class _BytesStreamer(ObjectStreamer):
    """Return an in-memory binary stream per key (mimics S3 get_object Body)."""

    def __init__(self, data: bytes | dict[str, bytes]) -> None:
        self._single = data if isinstance(data, bytes) else None
        self._data = None if isinstance(data, bytes) else dict(data)

    def open(self, key: str) -> IO[bytes]:
        raw = self._single if self._single is not None else self._data[key]
        return io.BytesIO(raw)


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


class TestS3WarcStreamStage:
    def test_process_uncompressed_stream(self) -> None:
        stage = S3WarcStreamStage(streamer=_BytesStreamer(_two_record_warc()))
        task = FileGroupTask(dataset_name="eai", data=["eai-warc/20240814/sample.warc"], _metadata={})

        df = stage.process(task).to_pandas()

        assert len(df) == 1
        row = df.iloc[0]
        assert row["url"] == "http://example.com/a.pdf"
        assert row["warc_id"] == "pdf1"
        assert row["filename"] == "a.pdf"
        assert row["warc_filename"] == "eai-warc/20240814/sample.warc"
        assert "id" not in df.columns
        assert "source_id" not in df.columns
        assert "content_type" not in df.columns
        assert "file_name" not in df.columns
        assert "warc_record_offset" in df.columns
        assert "warc_record_length" in df.columns
        assert "content" not in df.columns

    def test_writes_cdx_parquet_when_configured(self, tmp_path) -> None:
        stage = S3WarcStreamStage(streamer=_BytesStreamer(_two_record_warc()), cdx_output_dir=str(tmp_path))
        task = FileGroupTask(dataset_name="eai", data=["eai-warc/20240814/sample.warc"], _metadata={})
        stage.process(task)

        cdx_files = list(tmp_path.glob("*.parquet"))
        assert len(cdx_files) == 1
        cdx = __import__("pandas").read_parquet(cdx_files[0])
        assert len(cdx) == 2  # html + pdf responses
        assert set(cdx["is_pdf"].tolist()) == {False, True}

    def test_process_per_record_gzip_stream(self) -> None:
        # Standard .warc.gz layout: each record is an independent gzip member.
        records = [
            _make_record(
                rec_type="response",
                content_type="application/pdf",
                body=b"%PDF-1.4 doc",
                record_id="pdf1",
                uri="http://example.com/a.pdf",
            ),
            _make_record(
                rec_type="response",
                content_type="application/pdf",
                body=b"%PDF-1.4 doc2",
                record_id="pdf2",
                uri="http://example.com/b.pdf",
            ),
        ]
        warc_gz = b"".join(gzip.compress(r) for r in records)

        stage = S3WarcStreamStage(streamer=_BytesStreamer(warc_gz))
        task = FileGroupTask(dataset_name="eai", data=["eai-warc/20240814/sample.warc.gz"], _metadata={})

        df = stage.process(task).to_pandas()

        assert list(df["url"]) == ["http://example.com/a.pdf", "http://example.com/b.pdf"]
        assert list(df["warc_filename"]) == [
            "eai-warc/20240814/sample.warc.gz",
            "eai-warc/20240814/sample.warc.gz",
        ]

    def test_record_limit(self) -> None:
        records = [
            _make_record(
                rec_type="response",
                content_type="application/pdf",
                body=b"%PDF body",
                record_id=f"pdf{i}",
                uri=f"http://example.com/doc{i}.pdf",
            )
            for i in range(4)
        ]
        stage = S3WarcStreamStage(streamer=_BytesStreamer(b"".join(records)), record_limit=2)
        task = FileGroupTask(dataset_name="eai", data=["s.warc"], _metadata={})

        assert len(stage.process(task).to_pandas()) == 2

    def test_bad_object_fails_the_task(self) -> None:
        class _Boom(ObjectStreamer):
            def open(self, key: str) -> IO[bytes]:
                msg = "network error"
                raise RuntimeError(msg)

        stage = S3WarcStreamStage(streamer=_Boom())
        task = FileGroupTask(dataset_name="eai", data=["broken.warc.gz"], _metadata={})

        with pytest.raises(RuntimeError, match="network error"):
            stage.process(task)

    def test_missing_bucket_fails_the_task(self) -> None:
        stage = S3WarcStreamStage()
        task = FileGroupTask(dataset_name="eai", data=["s.warc.gz"], _metadata={})
        with pytest.raises(ValueError, match="requires a bucket"):
            stage.process(task)


class TestS3StreamEaiCrawlStage:
    def test_decomposition_defaults_to_gz_suffix(self) -> None:
        stage = S3StreamEaiCrawlStage(
            bucket="vdi-169-essentialai-essentialai-data",
            prefix="eai-warc/20240814/",
            endpoint_url="https://pdx.s8k.io",
        )
        stages = stage.decompose()

        assert len(stages) == 2
        assert isinstance(stages[1], S3WarcStreamStage)
        assert stage.name == "s3_stream_eai_crawl_pdf_extract"
        assert stage.url_generator.suffix == ".warc.gz"
        assert stage.url_generator.endpoint_url == "https://pdx.s8k.io"


class _FakeBoto3Module(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("boto3")
        self.calls: list[dict] = []

    def client(self, service: str, **kwargs: object) -> str:  # noqa: ARG002
        # Drop Config object for easier assertions; keep endpoint/region.
        self.calls.append({k: v for k, v in kwargs.items() if k != "config"})
        return "fake-client"


class TestEndpointPlumbing:
    def test_streamer_passes_endpoint_url(self, monkeypatch) -> None:
        fake = _FakeBoto3Module()
        monkeypatch.setitem(sys.modules, "boto3", fake)
        # botocore.config.Config must import cleanly for S3ObjectStreamer.
        monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)

        streamer = S3ObjectStreamer("bkt", endpoint_url="https://pdx.s8k.io", region="us-east-1")
        client = streamer._get_client()

        assert client == "fake-client"
        assert fake.calls[0]["endpoint_url"] == "https://pdx.s8k.io"
        assert fake.calls[0]["region_name"] == "us-east-1"

    def test_streamer_endpoint_falls_back_to_env(self, monkeypatch) -> None:
        fake = _FakeBoto3Module()
        monkeypatch.setitem(sys.modules, "boto3", fake)
        monkeypatch.setenv("AWS_ENDPOINT_URL", "https://env.example.io")

        S3ObjectStreamer("bkt")._get_client()

        assert fake.calls[0]["endpoint_url"] == "https://env.example.io"

    def test_url_generator_passes_endpoint_url(self, monkeypatch) -> None:
        fake = _FakeBoto3Module()
        monkeypatch.setitem(sys.modules, "boto3", fake)
        monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)

        gen = S3WarcUrlGenerator(bucket="bkt", endpoint_url="https://pdx.s8k.io")
        gen._get_client()

        assert fake.calls[0]["endpoint_url"] == "https://pdx.s8k.io"
