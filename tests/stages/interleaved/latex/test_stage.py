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

"""The Curator stages: partitioning, conversion, schema, resume."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nemo_curator.stages.interleaved.latex.arxiv.latexml.convert import LatexmlConfig
from nemo_curator.stages.interleaved.latex.arxiv.latexml.stage import (
    HTML_SCHEMA,
    ArxivLatexmlReader,
    LatexmlConvertStage,
    LatexmlTarPartitioningStage,
)
from nemo_curator.tasks import DocumentBatch, FileGroupTask

from .conftest import make_shard


def _group(shard) -> FileGroupTask:
    return FileGroupTask(dataset_name="arxiv", data=[str(shard)])


def _partition(shard, **kwargs) -> list[FileGroupTask]:
    return LatexmlTarPartitioningStage(**kwargs).process(_group(shard))


# --- partitioning ---


def test_pdf_members_are_kept(shard):
    """Dropping them changes the denominator from "of all submissions" silently."""
    entries = [json.loads(e) for t in _partition(shard) for e in t.data]
    assert sorted(e["member"] for e in entries) == ["0301/1203.5560.pdf", "0301/astro-ph0301029.gz"]


def test_descriptors_carry_a_seekable_byte_range(shard):
    entry = json.loads(_partition(shard)[0].data[0])
    assert entry.keys() == {"tar", "member", "offset", "size"}
    assert entry["offset"] > 0
    assert entry["size"] > 0


def test_submissions_are_split_into_groups(shard):
    tasks = _partition(shard, papers_per_task=1)
    assert len(tasks) == 2
    assert all(len(t.data) == 1 for t in tasks)


def test_max_papers_per_tar_caps_indexing(shard):
    entries = [e for t in _partition(shard, max_papers_per_tar=1) for e in t.data]
    assert len(entries) == 1


def test_a_corrupt_shard_does_not_end_the_run(tmp_path, shard):
    """One bad shard costs its own submissions, not the whole job."""
    broken = tmp_path / "src" / "arXiv_src_9999_001.tar"
    broken.write_bytes(b"this is not a tar")
    task = FileGroupTask(dataset_name="arxiv", data=[str(broken), str(shard)])
    entries = [e for t in LatexmlTarPartitioningStage().process(task) for e in t.data]
    assert len(entries) == 2, "the healthy shard is still indexed"


def test_papers_per_task_must_be_positive():
    with pytest.raises(ValueError, match="papers_per_task"):
        LatexmlTarPartitioningStage(papers_per_task=0)


# --- conversion ---


def _convert(shard, tmp_path, **kwargs):
    stage = LatexmlConvertStage(scratch_dir=str(tmp_path), **kwargs)
    stage.setup()
    try:
        return stage, stage.process(_partition(shard)[0])
    finally:
        stage.teardown()


def test_setup_fails_fast_when_the_converter_is_missing(tmp_path):
    stage = LatexmlConvertStage(config=LatexmlConfig(executable="no-such-latexmlc"), scratch_dir=str(tmp_path))
    with pytest.raises(RuntimeError, match="latexmlc was not found on PATH"):
        stage.setup()


def test_install_hint_names_the_container_not_pip(tmp_path):
    stage = LatexmlConvertStage(config=LatexmlConfig(executable="no-such-latexmlc"), scratch_dir=str(tmp_path))
    with pytest.raises(RuntimeError) as excinfo:
        stage.setup()
    assert "latexml/ar5ivist" in str(excinfo.value)
    assert "cannot be installed with pip" in str(excinfo.value)


def test_output_matches_the_declared_schema(shard, tmp_path, latexmlc):
    _, batch = _convert(shard, tmp_path)
    assert isinstance(batch, DocumentBatch)
    assert batch.data.schema.equals(HTML_SCHEMA)


def test_every_submission_produces_a_row(shard, tmp_path, latexmlc):
    _, batch = _convert(shard, tmp_path)
    assert batch.data.num_rows == 2


def test_provenance_columns_are_stamped(shard, tmp_path, latexmlc):
    _, batch = _convert(shard, tmp_path, snapshot="snapshot-2026-07-27", converter_id="32658425bb52")
    rows = batch.data.to_pylist()
    assert {r["snapshot"] for r in rows} == {"snapshot-2026-07-27"}
    assert {r["converter_id"] for r in rows} == {"32658425bb52"}
    assert {r["shard"] for r in rows} == {"arXiv_src_0301_001.tar"}
    assert {r["content_derivation"] for r in rows} == {"latex_latexml"}


def test_url_survives_a_projection_to_html_and_url(shard, tmp_path, latexmlc):
    _, batch = _convert(shard, tmp_path)
    by_id = {r["arxiv_id"]: r["url"] for r in batch.data.to_pylist()}
    assert by_id["astro-ph/0301029"] == "https://arxiv.org/abs/astro-ph/0301029"
    # the .pdf submission must not carry a URL that 404s
    assert by_id["1203.5560"] == "https://arxiv.org/abs/1203.5560"


def test_scratch_is_removed_on_teardown(shard, tmp_path, latexmlc):
    stage, _ = _convert(shard, tmp_path)
    assert stage._scratch is None
    assert not list(tmp_path.glob("latexml-*"))


def test_process_before_setup_is_an_error(shard, tmp_path):
    with pytest.raises(RuntimeError, match="setup"):
        LatexmlConvertStage(scratch_dir=str(tmp_path)).process(_partition(shard)[0])


# --- resume ---


def test_resume_skips_documents_already_written(shard, tmp_path, latexmlc):
    out = tmp_path / "out"
    out.mkdir()
    _, first = _convert(shard, tmp_path)
    pq.write_table(first.data, out / "part-0000.parquet")

    _, second = _convert(shard, tmp_path, resume_from=str(out))
    assert first.data.num_rows == 2
    assert second.data.num_rows == 0, "everything was already converted"


def test_resume_from_a_missing_directory_converts_everything(shard, tmp_path, latexmlc):
    _, batch = _convert(shard, tmp_path, resume_from=str(tmp_path / "never-written"))
    assert batch.data.num_rows == 2


def test_unreadable_output_does_not_silently_disable_resume(shard, tmp_path, latexmlc):
    """A half-written part must cost its own rows, not turn resume into a re-run."""
    out = tmp_path / "out"
    out.mkdir()
    (out / "part-0000.parquet").write_bytes(b"not parquet")
    _, batch = _convert(shard, tmp_path, resume_from=str(out))
    assert batch.data.num_rows == 2


# --- assets ---


def test_no_asset_tar_is_written_when_nothing_rasterizes(shard, tmp_path, latexmlc):
    latexmlc(body="<html><body><p>" + ("word " * 200) + "</p></body></html>")
    assets = tmp_path / "assets"
    _convert(shard, tmp_path, asset_dir=str(assets))
    assert not assets.exists() or not list(assets.glob("*.tar"))


# --- composite ---


def test_reader_decomposes_into_the_three_stages():
    stages = ArxivLatexmlReader(file_paths="/data").decompose()
    assert [s.__class__.__name__ for s in stages] == [
        "FilePartitioningStage",
        "LatexmlTarPartitioningStage",
        "LatexmlConvertStage",
    ]


def test_reader_forwards_its_configuration():
    reader = ArxivLatexmlReader(
        file_paths="/data", papers_per_task=7, snapshot="s", converter_id="c", resume_from="/prev"
    )
    _, partitioning, convert = reader.decompose()
    assert partitioning.papers_per_task == 7
    assert convert.snapshot == "s"
    assert convert.converter_id == "c"
    assert convert.resume_from == "/prev"


def test_empty_shard_yields_an_empty_batch_not_a_crash(tmp_path, latexmlc):
    empty = make_shard(tmp_path / "src" / "arXiv_src_0000_001.tar", {})
    stage = LatexmlConvertStage(scratch_dir=str(tmp_path))
    stage.setup()
    try:
        batch = stage.process(FileGroupTask(dataset_name="arxiv", data=[]))
    finally:
        stage.teardown()
    assert batch.data.num_rows == 0
    assert batch.data.schema.equals(HTML_SCHEMA)
    assert LatexmlTarPartitioningStage().process(_group(empty)) == []
