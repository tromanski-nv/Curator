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

"""Tests for the stage that adds a canonical identifier column.

Driven by calling ``process()`` directly, so no Ray cluster is involved.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

from nemo_curator.stages.interleaved.identifiers import AddArxivIdStage
from nemo_curator.tasks import InterleavedBatch
from nemo_curator.tasks.interleaved import INTERLEAVED_SCHEMA


def _row(sample_id: str, position: int = 0, **extra: object) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "position": position,
        "modality": "text",
        "content_type": "text/markdown",
        "text_content": "body",
        "binary_content": None,
        "source_ref": None,
        "materialize_error": None,
        **extra,
    }


def _batch(rows: list[dict[str, Any]]) -> InterleavedBatch:
    return InterleavedBatch(dataset_name="test", data=pa.Table.from_pylist(rows))


def _run(task: InterleavedBatch, **kwargs: object) -> list[dict[str, Any]]:
    return AddArxivIdStage(**kwargs).process(task).to_pyarrow().to_pylist()


# ---- the stage contract -----------------------------------------------------


class TestContract:
    def test_it_declares_the_columns_it_reads_and_writes(self) -> None:
        stage = AddArxivIdStage()

        assert stage.inputs() == (["data"], ["sample_id"])
        assert stage.outputs() == (["data"], ["arxiv_id"])

    def test_an_empty_batch_passes_through(self) -> None:
        empty = InterleavedBatch(dataset_name="t", data=pa.Table.from_pylist([], schema=INTERLEAVED_SCHEMA))

        assert AddArxivIdStage().process(empty).to_pyarrow().num_rows == 0


# ---- what it writes ---------------------------------------------------------


class TestOutput:
    @pytest.mark.parametrize(
        ("sample_id", "expected"),
        [
            ("2410/2410.10730", "2410.10730"),
            ("cond-mat/cond-mat0410443", "cond-mat/0410443"),
            ("math/math0001001", "math/0001001"),
        ],
    )
    def test_the_column_carries_the_canonical_spelling(self, sample_id: str, expected: str) -> None:
        rows = _run(_batch([_row(sample_id)]))

        assert rows[0]["arxiv_id"] == expected
        assert rows[0]["sample_id"] == sample_id, "the original spelling is not disturbed"

    def test_every_row_of_a_document_gets_it_including_the_metadata_row(self) -> None:
        """A consumer filtering on arxiv_id must not lose the metadata row."""
        rows = _run(_batch([_row("2410/2410.10730", -1), _row("2410/2410.10730", 0), _row("2410/2410.10730", 1)]))

        assert [r["arxiv_id"] for r in rows] == ["2410.10730"] * 3

    def test_documents_do_not_bleed_into_each_other(self) -> None:
        rows = _run(_batch([_row("2410/2410.10730"), _row("math/math0001001"), _row("2308/2308.10008")]))

        assert [r["arxiv_id"] for r in rows] == ["2410.10730", "math/0001001", "2308.10008"]

    def test_the_column_names_are_configurable(self) -> None:
        rows = _run(
            _batch([_row("2410/2410.10730", doc="math/math0001001")]), source_column="doc", target_column="paper_id"
        )

        assert rows[0]["paper_id"] == "math/0001001"

    def test_running_it_twice_changes_nothing(self) -> None:
        """The column is derived, so a re-run must overwrite rather than
        duplicate, and canonicalising an already-canonical id is a no-op."""
        once = AddArxivIdStage().process(_batch([_row("2410/2410.10730")]))
        twice = AddArxivIdStage().process(once).to_pyarrow()

        assert twice.column_names.count("arxiv_id") == 1
        assert twice.to_pylist()[0]["arxiv_id"] == "2410.10730"

    def test_the_reserved_interleaved_columns_survive(self) -> None:
        table = AddArxivIdStage().process(_batch([_row("2410/2410.10730")])).to_pyarrow()

        assert set(INTERLEAVED_SCHEMA.names) <= set(table.column_names)
        assert table.schema.field("arxiv_id").type == pa.string()
