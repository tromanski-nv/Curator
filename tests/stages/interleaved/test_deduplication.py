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

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pytest

from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR
from nemo_curator.stages.interleaved.deduplication.pdf_sha import PdfSha256InventoryStage
from nemo_curator.stages.interleaved.deduplication.removal import (
    InterleavedSampleDuplicatesRemovalStage,
    InterleavedSampleIdRemovalStage,
)
from nemo_curator.stages.interleaved.io.readers.parquet import InterleavedParquetReaderStage
from nemo_curator.tasks import FileGroupTask, InterleavedBatch
from tutorials.interleaved.deduplication import prepare


def _interleaved_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"sample_id": "s_b", "position": -1, "modality": "metadata", "text_content": None},
            {"sample_id": "s_b", "position": 0, "modality": "text", "text_content": "duplicate text"},
            {"sample_id": "s_a", "position": -1, "modality": "metadata", "text_content": None},
            {"sample_id": "s_a", "position": 0, "modality": "text", "text_content": "keep text"},
            {"sample_id": "s_b", "position": 1, "modality": "image", "text_content": None},
        ],
    )


@pytest.mark.usefixtures("ray_client_with_id_generator")
def test_interleaved_parquet_reader_generates_and_assigns_sample_ids(tmp_path: Path) -> None:
    input_file = tmp_path / "interleaved.parquet"
    _interleaved_rows().to_parquet(input_file, index=False)
    task = FileGroupTask(dataset_name="test", data=[str(input_file)])

    generate_stage = InterleavedParquetReaderStage(_generate_ids=True)
    generate_stage.setup()
    generated = generate_stage.process(task)
    assert isinstance(generated, InterleavedBatch)
    generated_df = generated.to_pandas()

    sample_to_id = (
        generated_df[["sample_id", CURATOR_DEDUP_ID_STR]]
        .drop_duplicates()
        .sort_values("sample_id")
        .reset_index(drop=True)
    )
    assert sample_to_id["sample_id"].tolist() == ["s_a", "s_b"]
    assert sample_to_id[CURATOR_DEDUP_ID_STR].tolist() == [0, 1]
    s_b_ids = generated_df.loc[generated_df["sample_id"] == "s_b", CURATOR_DEDUP_ID_STR].tolist()
    assert len(set(s_b_ids)) == 1

    assign_stage = InterleavedParquetReaderStage(_assign_ids=True)
    assign_stage.setup()
    assigned = assign_stage.process(task)
    assert isinstance(assigned, InterleavedBatch)

    pd.testing.assert_series_equal(
        generated_df[CURATOR_DEDUP_ID_STR],
        assigned.to_pandas()[CURATOR_DEDUP_ID_STR],
        check_names=False,
    )


def test_interleaved_sample_duplicates_removal_drops_all_sample_rows(tmp_path: Path) -> None:
    df = _interleaved_rows()
    df[CURATOR_DEDUP_ID_STR] = df["sample_id"].map({"s_a": 0, "s_b": 1})
    task = InterleavedBatch(dataset_name="test", data=df)

    duplicate_dir = tmp_path / "duplicates"
    duplicate_dir.mkdir()
    pd.DataFrame({CURATOR_DEDUP_ID_STR: [1]}).to_parquet(duplicate_dir / "part.0.parquet", index=False)

    result = InterleavedSampleDuplicatesRemovalStage(ids_to_remove_path=str(duplicate_dir)).process(task)
    result_df = result.to_pandas()

    assert set(result_df["sample_id"].tolist()) == {"s_a"}
    assert CURATOR_DEDUP_ID_STR not in result_df.columns
    assert result._metadata["num_samples_removed"] == 1
    assert result._metadata["num_rows_in"] == 5
    assert result._metadata["num_rows_out"] == 2


def test_interleaved_sample_duplicates_removal_process_batch_accepts_arrow_columns(tmp_path: Path) -> None:
    df = _interleaved_rows()
    df[CURATOR_DEDUP_ID_STR] = df["sample_id"].map({"s_a": 0, "s_b": 1})
    task = InterleavedBatch(dataset_name="test", data=pa.Table.from_pandas(df, preserve_index=False))

    duplicate_dir = tmp_path / "duplicates"
    duplicate_dir.mkdir()
    pd.DataFrame({CURATOR_DEDUP_ID_STR: [1]}).to_parquet(duplicate_dir / "part.0.parquet", index=False)

    results = InterleavedSampleDuplicatesRemovalStage(ids_to_remove_path=str(duplicate_dir)).process_batch([task])
    assert len(results) == 1
    assert set(results[0].to_pandas()["sample_id"].tolist()) == {"s_a"}
    assert CURATOR_DEDUP_ID_STR not in results[0].to_pandas().columns
    assert results[0]._metadata["num_samples_removed"] == 1


def test_interleaved_sample_duplicates_removal_missing_duplicate_path_keeps_input(tmp_path: Path) -> None:
    df = _interleaved_rows()
    df[CURATOR_DEDUP_ID_STR] = df["sample_id"].map({"s_a": 0, "s_b": 1})
    task = InterleavedBatch(dataset_name="test", data=df)

    result = InterleavedSampleDuplicatesRemovalStage(
        ids_to_remove_path=str(tmp_path / "missing"),
    ).process(task)
    result_df = result.to_pandas()

    assert set(result_df["sample_id"].tolist()) == {"s_a", "s_b"}
    assert CURATOR_DEDUP_ID_STR not in result_df.columns
    assert result._metadata["num_samples_removed"] == 0
    assert result._metadata["num_rows_out"] == len(df)


def test_interleaved_sample_id_removal_drops_complete_sample(tmp_path: Path) -> None:
    duplicate_dir = tmp_path / "sample_ids"
    duplicate_dir.mkdir()
    pd.DataFrame({"sample_id": ["s_b"]}).to_parquet(duplicate_dir / "part.0.parquet", index=False)

    stage = InterleavedSampleIdRemovalStage(ids_to_remove_path=str(duplicate_dir))
    stage.setup()
    result = stage.process(InterleavedBatch(dataset_name="test", data=_interleaved_rows()))

    assert set(result.to_pandas()["sample_id"]) == {"s_a"}
    assert result._metadata["num_samples_in"] == 2
    assert result._metadata["num_samples_removed"] == 1
    assert result._metadata["num_samples_out"] == 1
    assert result._metadata["num_rows_in"] == 5
    assert result._metadata["num_rows_out"] == 2


def test_pdf_sha256_inventory_hashes_original_pdf_and_resumes(tmp_path: Path) -> None:
    pdf_root = tmp_path / "pdfs"
    pdf_root.mkdir()
    pdf_bytes = b"%PDF-1.7\noriginal bytes\n%%EOF\n"
    (pdf_root / "sample.pdf").write_bytes(pdf_bytes)

    source_parquet = tmp_path / "input.parquet"
    pd.DataFrame(
        {
            "sample_id": ["sample", "sample"],
            "pdf_name": ["sample.pdf", "sample.pdf"],
        },
    ).to_parquet(source_parquet, index=False)
    task = FileGroupTask(dataset_name="test", data=[str(source_parquet)])
    stage = PdfSha256InventoryStage(
        output_path=str(tmp_path / "inventory"),
        pdf_root=str(pdf_root),
    )

    first_result = stage.process(task)
    inventory = pd.read_parquet(first_result.data[0])
    assert inventory["sha256"].tolist() == [hashlib.sha256(pdf_bytes).hexdigest()]
    assert inventory["size_bytes"].tolist() == [len(pdf_bytes)]
    assert inventory["hash_error"].isna().all()

    second_result = stage.process(task)
    assert second_result.data == first_result.data
    assert second_result._metadata["resumed"] is True


def test_prepare_baseline_and_arxiv_version_inventory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prepare, "_code_state", dict)
    parse_root = tmp_path / "parse"
    pdf_root = tmp_path / "pdfs"
    (pdf_root / "1805").mkdir(parents=True)
    parse_root.mkdir()
    (pdf_root / "1805" / "1805.00001.pdf").write_bytes(b"pdf")
    pd.DataFrame(
        {
            "sample_id": ["1805/1805.00001", "1805/1805.00001"],
            "pdf_name": ["1805/1805.00001.pdf", "1805/1805.00001.pdf"],
        },
    ).to_parquet(parse_root / "part.parquet", index=False)

    baseline_path = tmp_path / "baseline"
    baseline_manifest = tmp_path / "baseline.json"
    prepare.build_baseline(
        SimpleNamespace(
            input_path=str(parse_root),
            pdf_root=str(pdf_root),
            output_path=str(baseline_path),
            manifest_path=str(baseline_manifest),
            overwrite=False,
            sample_id_field="sample_id",
            pdf_name_field="pdf_name",
        ),
    )
    assert json.loads(baseline_manifest.read_text())["num_unique_samples"] == 1

    metadata_path = tmp_path / "metadata.jsonl"
    metadata_path.write_text(
        json.dumps(
            {
                "id": "1805.00001",
                "update_date": "2018-10-05",
                "versions": [
                    {"version": "v1", "created": "2018-05-01"},
                    {"version": "v2", "created": "2018-10-05"},
                ],
            },
        )
        + "\n",
    )
    versions_path = tmp_path / "versions"
    removed_path = tmp_path / "removed"
    prepare.select_arxiv_versions(
        SimpleNamespace(
            inventory_path=str(baseline_path),
            metadata_path=str(metadata_path),
            metadata_sha256="test-metadata-sha256",
            metadata_source_url="https://example.test/arxiv-metadata.json",
            metadata_snapshot_date="2018-10-05",
            output_path=str(versions_path),
            removed_path=str(removed_path),
            exceptions_path=str(tmp_path / "exceptions"),
            manifest_path=str(tmp_path / "versions.json"),
            sample_id_field="sample_id",
            pdf_name_field="pdf_name",
        ),
    )

    versions = pd.read_parquet(versions_path)
    assert versions["selected"].tolist() == [True]
    assert versions["selected_version"].tolist() == ["v2"]
    assert pd.read_parquet(removed_path).empty


def test_prepare_selects_deterministic_sha_keeper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prepare, "_code_state", dict)
    inventory_path = tmp_path / "sha_inventory"
    inventory_path.mkdir()
    pd.DataFrame(
        [
            {
                "sample_id": "b",
                "size_bytes": 3,
                "sha256": "digest",
                "hash_error": None,
                "source_path": "/pdfs/b.pdf",
            },
            {
                "sample_id": "a",
                "size_bytes": 3,
                "sha256": "digest",
                "hash_error": None,
                "source_path": "/pdfs/a.pdf",
            },
        ],
    ).to_parquet(inventory_path / "part.parquet", index=False)

    output_path = tmp_path / "duplicates"
    prepare.select_sha_duplicates(
        SimpleNamespace(
            inventory_path=str(inventory_path),
            output_path=str(output_path),
            manifest_path=str(tmp_path / "sha.json"),
            sample_id_field="sample_id",
        ),
    )
    duplicates = pd.read_parquet(output_path)
    assert duplicates[["sample_id", "keeper_sample_id"]].to_dict("records") == [
        {"sample_id": "b", "keeper_sample_id": "a"},
    ]


def test_prepare_validates_end_to_end_accounting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prepare, "_code_state", dict)
    manifests = {
        "baseline": {"num_unique_samples": 10},
        "versions": {"num_input_samples": 10, "num_removed_samples": 1, "num_selected_samples": 9},
        "sha": {"num_input_samples": 9, "num_duplicate_samples": 2, "num_output_samples": 7},
        "exact": {"metadata": {"num_samples_in": 7, "num_samples_removed": 1, "num_samples_out": 6}},
        "fuzzy": {"metadata": {"num_samples_in": 6, "num_samples_removed": 2, "num_samples_out": 4}},
    }
    manifest_paths = {}
    for name, payload in manifests.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload))
        manifest_paths[name] = str(path)

    output_path = tmp_path / "accounting.json"
    prepare.validate_accounting(
        SimpleNamespace(
            baseline_manifest=manifest_paths["baseline"],
            version_manifest=manifest_paths["versions"],
            sha_manifest=manifest_paths["sha"],
            exact_removal_manifest=manifest_paths["exact"],
            fuzzy_removal_manifest=manifest_paths["fuzzy"],
            output_path=str(output_path),
        ),
    )
    accounting = json.loads(output_path.read_text())
    assert accounting["valid"] is True
    assert accounting["counts"]["final"] == 4
