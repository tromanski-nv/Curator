# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from io import BytesIO
from pathlib import Path
from typing import IO, Any
from unittest import mock

import pandas as pd
import pytest

from nemo_curator.tasks import FailedTask, FileGroupTask
from tutorials.eai_crawl.cdx_index import IndexPassResult
from tutorials.eai_crawl.s3_streaming import S3WarcStreamStage


class _Streamer:
    def open(self, _key: str) -> BytesIO:
        return BytesIO(b"warc")


def _result(key: str) -> IndexPassResult:
    return IndexPassResult(
        cdx_rows=[{"url": f"https://example.test/{key}", "warc_filename": key}],
        pdf_rows=[],
        num_responses=1,
        gzip_layout="per_record",
    )


def test_resumable_group_writes_are_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    written: list[str] = []

    def parse(
        _stream: IO[bytes],
        *,
        warc_filename: str,
        pdf_record_limit: int | None,  # noqa: ARG001
    ) -> IndexPassResult:
        return _result(warc_filename)

    def record_write(
        _df: pd.DataFrame,
        path: str,
        *,
        storage_options: dict[str, Any] | None,  # noqa: ARG001
    ) -> None:
        written.append(path)

    monkeypatch.setattr(
        "tutorials.eai_crawl.s3_streaming.iterate_cdx_and_pdfs",
        parse,
    )
    monkeypatch.setattr(
        "tutorials.eai_crawl.s3_streaming.write_parquet",
        record_write,
    )
    stage = S3WarcStreamStage(
        streamer=_Streamer(),
        pdf_output_dir=str(tmp_path / "pdf"),
        cdx_output_dir=str(tmp_path / "cdx"),
    )
    task = FileGroupTask(dataset_name="eai", data=["a.warc.gz", "b.warc.gz"])

    assert stage.process(task) is None
    first_paths = list(written)
    written.clear()
    assert stage.process(task) is None

    assert written == first_paths
    assert len(written) == 1
    assert task.get_deterministic_id() in written[0]


def test_failed_warc_keeps_group_pending(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def parse(
        _stream: IO[bytes],
        *,
        warc_filename: str,
        pdf_record_limit: int | None,  # noqa: ARG001
    ) -> IndexPassResult:
        if warc_filename == "bad.warc.gz":
            msg = "transient"
            raise OSError(msg)
        return _result(warc_filename)

    monkeypatch.setattr("tutorials.eai_crawl.s3_streaming.iterate_cdx_and_pdfs", parse)
    with mock.patch("tutorials.eai_crawl.s3_streaming.write_parquet") as write:
        stage = S3WarcStreamStage(
            streamer=_Streamer(),
            pdf_output_dir=str(tmp_path / "pdf"),
            cdx_output_dir=str(tmp_path / "cdx"),
        )
        result = stage.process(FileGroupTask(dataset_name="eai", data=["good.warc.gz", "bad.warc.gz"]))

    assert isinstance(result, FailedTask)
    write.assert_called_once()
