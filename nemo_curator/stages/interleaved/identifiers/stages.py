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

"""Derive a canonical identifier column from ``sample_id``."""

from __future__ import annotations

from dataclasses import dataclass, field

import pyarrow as pa

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.interleaved.identifiers import arxiv
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import InterleavedBatch

DEFAULT_SOURCE_COLUMN = "sample_id"
DEFAULT_TARGET_COLUMN = "arxiv_id"


@dataclass
class AddArxivIdStage(ProcessingStage[InterleavedBatch, InterleavedBatch]):
    """Add a canonical ``arxiv_id`` column derived from ``sample_id``.

    ``sample_id`` is whatever the corpus that produced the rows happened to
    call the document -- the Nemotron-Parse element format writes
    ``2410/2410.10730`` and ``math/math0001001``, both of which carry a
    duplicated prefix and match nothing outside this corpus.  This stage adds
    the one spelling arXiv itself uses, so a join is a column comparison rather
    than a transformation every consumer has to remember to apply.

    One column, not two.  A corpus stored in some other spelling converts *to*
    this one; carrying a second column for the other spelling is how two
    representations of the same fact drift apart.

    Unrecognised input is passed through cleaned but otherwise unchanged, which
    is what makes a bad identifier fail a join loudly instead of colliding with
    a real one.  Measured over 227,455 identifiers from the arXiv corpus:
    nothing was left unrecognised and nothing collided.

    Example::

        pipeline.add_stage(NemotronParseMarkdownPostprocessor())
        pipeline.add_stage(AddArxivIdStage())
        pipeline.add_stage(InterleavedParquetWriterStage(path=out))

    Parameters
    ----------
    source_column
        Where the identifier comes from.  Defaults to ``sample_id``, the column
        the interleaved schema uses to group rows into documents.
    target_column
        The column to write.  Defaults to ``arxiv_id``.
    """

    source_column: str = DEFAULT_SOURCE_COLUMN
    target_column: str = DEFAULT_TARGET_COLUMN
    name: str = "add_arxiv_id"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.source_column]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.target_column]

    def process(self, task: InterleavedBatch) -> InterleavedBatch:
        table = task.to_pyarrow()
        if table.num_rows == 0:
            return task

        # One canon() per distinct document rather than per row: a document is
        # a few hundred rows, and the corpus has ~500 of them per shard.
        source = table.column(self.source_column).to_pylist()
        cache: dict[str | None, str | None] = {}
        derived = [cache[s] if s in cache else cache.setdefault(s, arxiv.canon(s)) for s in source]

        column = pa.array(derived, type=pa.string())
        if self.target_column in table.column_names:
            table = table.set_column(table.schema.get_field_index(self.target_column), self.target_column, column)
        else:
            table = table.append_column(pa.field(self.target_column, pa.string()), column)

        return InterleavedBatch(
            dataset_name=task.dataset_name,
            data=table,
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )
