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

"""One Curator stage per post-processing step.

Each stage is a thin wrapper: split the batch into documents, hand each
document's rows to one function in :mod:`.steps`, write the result back as
rows.  The rules live there and are tested there, on plain Python objects; what
lives here is the Arrow plumbing and nothing else.

Splitting the steps into separate stages costs a serialisation round trip
between each pair, and buys two things worth the price.  A run can be stopped
after any step and the intermediate written out, which is how you find out
*which* rule dropped a paragraph rather than that something did.  And the
execution plan names the steps, so a run's logs read as the list of decisions
it made rather than as one opaque box.

The price, measured on a batch of 10 documents of 300 elements each with 10 MB
of picture crops between them: 74 ms per document against 16 ms for the same
work fused, so roughly 60 ms per document for the intermediate being
observable.  Set against the GPU-seconds per document the parse phase costs,
that is noise; where it is not,
:class:`~.composite.NemotronParseMarkdownPostprocessor` takes ``fuse=True`` and
runs the lot in :class:`FusedPostprocessingStage`, which produces the identical
table.

Every stage in this module takes the same :class:`~.model.Config`, so a run is
configured once and the stages agree by construction.
"""

from __future__ import annotations

import json
from abc import abstractmethod
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import pyarrow as pa
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.postprocessing import markdown, rows, steps
from nemo_curator.stages.interleaved.pdf.nemotron_parse.postprocessing.model import (
    Box,
    Config,
    Layout,
    Stats,
)
from nemo_curator.stages.interleaved.utils.schema import align_table
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import InterleavedBatch
from nemo_curator.tasks.interleaved import INTERLEAVED_SCHEMA, RESERVED_COLUMNS

if TYPE_CHECKING:
    from collections.abc import Iterable

#: Post-processing is pure Python over text: one core, and the batch size is
#: what buys throughput.  A factory, not a shared instance -- ``Resources`` is
#: mutable and six stages holding the same one would tune together.
DEFAULT_CPUS = 1.0

#: Where the metadata row records what post-processing did.
STATS_KEY = "postprocess"

#: Columns that exist only to carry state between the steps, and have no
#: meaning once the document has been assembled.
INTERNAL_COLUMNS = (rows.SLOT, rows.SPANS_PAGES)

#: Columns that only say anything when the condemned boxes are kept.  With
#: ``emit_dropped=False`` every surviving row has ``keep=True`` and an empty
#: reason, and a column with one value in it is noise.
VERDICT_COLUMNS = (rows.KEEP, rows.DROP_REASON)


def _target_schema(records: list[rows.Record], declared: tuple[str, ...] = ()) -> pa.Schema:
    """Reserved columns first, then the columns this package writes, then the rest.

    The order is canonical rather than first-seen, and the types of the columns
    this package writes come from :data:`~.rows.EXTRA_SCHEMA` rather than from
    the values.  Both matter for the same reason: the six stages and the fused
    stage build their rows from different templates, and a table whose column
    order depended on that would make two runs of the same pipeline produce two
    incompatible Parquet datasets.  Columns the caller brought along keep
    whatever Arrow infers, in the order they first appear.

    ``declared`` names the columns this stage writes whether or not any row
    happened to carry one.  Without it, a batch whose documents all came out
    empty would write a narrower Parquet file than its neighbours, and a reader
    opening the directory as one dataset would drop the columns that disagree.
    """
    seen: list[str] = list(declared)
    for record in records:
        seen.extend(k for k in record if k not in RESERVED_COLUMNS and k not in seen)
    pinned = {f.name: f for f in rows.EXTRA_SCHEMA}
    known = [pinned[name] for name in rows.EXTRA_SCHEMA.names if name in seen]
    unknown = [name for name in seen if name not in pinned]
    inferred = pa.Table.from_pylist([{k: r.get(k) for k in unknown} for r in records]) if unknown else None
    return pa.schema(
        [
            *INTERLEAVED_SCHEMA,
            *known,
            *(inferred.schema if inferred is not None else []),
        ]
    )


def _to_table(records: list[rows.Record], declared: tuple[str, ...] = ()) -> pa.Table:
    """Rows as an interleaved table with stable column order and types."""
    schema = _target_schema(records, declared)
    table = pa.Table.from_pylist([{name: record.get(name) for name in schema.names} for record in records])
    return align_table(table, schema)


@dataclass
class _DocumentStage(ProcessingStage[InterleavedBatch, InterleavedBatch]):
    """Shared plumbing: one document at a time, rows in and rows out.

    Parameters
    ----------
    config
        The post-processing tunables.  Shared by every stage in the composite.
    """

    config: Config = field(default_factory=Config)
    # Every concrete stage names itself; this only shows up if one forgets, and
    # it must not collide with ``NemotronParsePostprocessStage`` next door.
    name: str = "nemotron_parse_postprocess_step"
    resources: Resources = field(default_factory=lambda: Resources(cpus=DEFAULT_CPUS))

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    @abstractmethod
    def document(self, metadata: list[rows.Record], elements: list[rows.Record]) -> list[rows.Record]:
        """Every row this stage emits for one document, metadata row included."""

    def declared_columns(self) -> tuple[str, ...]:
        """The extra columns this stage writes, empty batch or not."""
        return (rows.ELEMENT_CLASS, rows.PAGE_NUMBER, *rows.POSTPROCESS_COLUMNS)

    def process(self, task: InterleavedBatch) -> InterleavedBatch | None:
        records: list[rows.Record] = []
        for _sample_id, metadata, elements in rows.split_documents(task.to_pyarrow().to_pylist()):
            records.extend(self.document(metadata, elements))

        if not records:
            return None

        return InterleavedBatch(
            dataset_name=task.dataset_name,
            data=_to_table(records, self.declared_columns()),
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )


@dataclass
class _LayoutStage(_DocumentStage):
    """A stage whose step is ``Layout -> Layout``."""

    @abstractmethod
    def step(self, layout: Layout) -> Layout:
        """The one function in :mod:`.steps` this stage exists to call."""

    def document(self, metadata: list[rows.Record], elements: list[rows.Record]) -> list[rows.Record]:
        if not elements:
            return list(metadata)
        layout = self.step(rows.to_layout(elements))
        return [*metadata, *rows.from_layout(layout, rows.templates(elements))]


# --------------------------------------------------------------------------
# 1. clean
# --------------------------------------------------------------------------


@dataclass
class ElementCleaningStage(_DocumentStage):
    """Order the elements and condemn the ones that never had content.

    The entry point to post-processing: it is the only stage that reads the raw
    Nemotron-Parse element format, and the one that adds the ``keep`` /
    ``drop_reason`` / ``source_positions`` columns every later stage relies on.

    See :func:`~.steps.clean`.
    """

    name: str = "nemotron_parse_clean"

    def document(self, metadata: list[rows.Record], elements: list[rows.Record]) -> list[rows.Record]:
        if not elements:
            return list(metadata)
        boxes = steps.clean(rows.to_elements(elements), self.config)
        return [*metadata, *rows.from_boxes(boxes, rows.templates(elements))]


# --------------------------------------------------------------------------
# 2-5. the layout steps
# --------------------------------------------------------------------------


@dataclass
class FloatAssignmentStage(_DocumentStage):
    """Lift floats out of the flow and match each caption to its figure.

    See :func:`~.steps.assign_floats`.  This is the stage that fills the
    ``slot`` column, after which row order -- not ``position`` -- is the reading
    order; it therefore reads a flat box stream rather than a layout, because
    the split it produces does not exist yet.
    """

    name: str = "nemotron_parse_assign_floats"

    def document(self, metadata: list[rows.Record], elements: list[rows.Record]) -> list[rows.Record]:
        if not elements:
            return list(metadata)
        layout = steps.assign_floats(rows.to_boxes(elements), self.config)
        return [*metadata, *rows.from_layout(layout, rows.templates(elements))]


@dataclass
class PageFurnitureStage(_LayoutStage):
    """Condemn running heads and page numbers.  See :func:`~.steps.mark_page_furniture`."""

    name: str = "nemotron_parse_page_furniture"

    def step(self, layout: Layout) -> Layout:
        return steps.mark_page_furniture(layout, self.config)


@dataclass
class SectionSkippingStage(_LayoutStage):
    """Condemn the table of contents and the bibliography.  See :func:`~.steps.skip_sections`."""

    name: str = "nemotron_parse_skip_sections"

    def step(self, layout: Layout) -> Layout:
        return steps.skip_sections(layout, self.config)


@dataclass
class ParagraphReconstitutionStage(_LayoutStage):
    """Rejoin sentences broken across a column or a page.

    See :func:`~.steps.reconstitute_paragraphs`.  This is the one stage that
    emits fewer rows than it read: an absorbed box lives on inside the
    ``source_positions`` of the box that took it.
    """

    name: str = "nemotron_parse_reconstitute_paragraphs"

    def step(self, layout: Layout) -> Layout:
        return steps.reconstitute_paragraphs(layout, self.config)


# --------------------------------------------------------------------------
# 6. assemble and render
# --------------------------------------------------------------------------


def _without_working_columns(record: rows.Record, *, emit_dropped: bool) -> rows.Record:
    """Drop the columns that only existed to carry state between the steps.

    Applied to every row the last stage emits, the metadata row included.  A
    column left behind on one row is a column on the whole table, and the six
    stages would then produce a wider table than the fused stage for the same
    input.
    """
    dropped = INTERNAL_COLUMNS if emit_dropped else (*INTERNAL_COLUMNS, *VERDICT_COLUMNS)
    return {k: v for k, v in record.items() if k not in dropped}


def _emitted_bbox(box: Box) -> tuple[float, float, float, float] | None:
    """The box's bounding box, or ``None`` when no single box describes it.

    A paragraph rejoined across a page boundary keeps the page it started on
    and the bounding box of the fragment that finished it -- the two do not
    describe the same piece of paper.  Rather than emit a locator that draws a
    rectangle on the wrong page, say the geometry is unknown.
    """
    return None if box.spans_pages else box.bbox


def _markdown_rows(
    boxes: Iterable[Box],
    template_by_position: dict[int, rows.Record],
    *,
    emit_dropped: bool,
) -> list[rows.Record]:
    """Boxes as interleaved markdown rows, renumbered from zero.

    An image row is emitted for its bytes, not its text, so it survives
    rendering to the empty string; a text row that renders to nothing does not.
    """
    records: list[rows.Record] = []
    emitted_at: dict[int, int] = {}
    for box in boxes:
        if not box.keep and not emit_dropped:
            continue
        rendered = markdown.render(box)
        is_image = box.has_payload or box.modality == "image"
        if not rendered and not is_image:
            continue

        record = dict(template_by_position.get(box.source_positions[0], {}))
        record.update(
            {
                rows.POSITION: len(records),
                rows.MODALITY: box.modality,
                rows.TEXT_CONTENT: rendered or None,
                rows.ELEMENT_CLASS: box.element_class,
                rows.PAGE_NUMBER: box.page,
                rows.SOURCE_REF: rows.build_source_ref(box.page, _emitted_bbox(box)),
                rows.SOURCE_POSITIONS: json.dumps(list(box.source_positions)),
                rows.MATCHED_TO: box.matched_to,
            }
        )
        if not is_image:
            record[rows.CONTENT_TYPE] = rows.MARKDOWN_CONTENT_TYPE
            record[rows.BINARY_CONTENT] = None
        if emit_dropped:
            record[rows.KEEP] = box.keep
            record[rows.DROP_REASON] = box.reason
        emitted_at[box.source_positions[0]] = len(records)
        records.append(_without_working_columns(record, emit_dropped=emit_dropped))

    # ``matched_to`` came off the box holding an *element* position; the rows
    # have just been renumbered from zero.  Left as it was it would point at
    # whatever row happens to sit at that index, so translate it -- and drop it
    # when the figure it names was not emitted at all.
    for record in records:
        matched = record.get(rows.MATCHED_TO)
        if matched is not None:
            record[rows.MATCHED_TO] = emitted_at.get(int(matched))
    return records


def _with_stats(metadata: list[rows.Record], stats: Stats) -> list[rows.Record]:
    """Record what post-processing did on the document's metadata row.

    Only an existing metadata row is amended.  A document that arrived without
    one leaves without one: inventing a row means inventing the ``url`` and
    ``pdf_name`` that belong on it.
    """
    amended: list[rows.Record] = []
    for record in metadata:
        row = dict(record)
        existing = row.get(rows.TEXT_CONTENT)
        payload: dict[str, Any] | None = {} if not existing else None
        if isinstance(existing, str) and existing:
            try:
                decoded = json.loads(existing)
            except ValueError:
                decoded = None
            if isinstance(decoded, dict):
                payload = decoded
        if payload is None:
            # Something is in there that is not the JSON object the parse phase
            # writes.  Whatever it is, it belongs to whoever put it there;
            # overwriting it to make room for statistics would be a poor trade.
            logger.debug("Metadata row is not a JSON object; leaving it alone and not recording stats on it")
            amended.append(row)
            continue
        payload[STATS_KEY] = asdict(stats)
        row[rows.TEXT_CONTENT] = json.dumps(payload)
        amended.append(row)
    return amended


@dataclass
class _FinalStage(_DocumentStage):
    """A stage that emits the finished document rather than working state.

    Parameters
    ----------
    emit_dropped
        Keep the condemned boxes in the output, marked with ``keep=False`` and
        ``drop_reason``, instead of discarding them.  Off by default, because a
        training pipeline wants the document; on, because a viewer wants to see
        what the rules threw away and why.
    """

    emit_dropped: bool = False

    def declared_columns(self) -> tuple[str, ...]:
        kept = (rows.ELEMENT_CLASS, rows.PAGE_NUMBER, rows.SOURCE_POSITIONS, rows.MATCHED_TO)
        return (*kept, *VERDICT_COLUMNS) if self.emit_dropped else kept

    def _clean_metadata(self, metadata: list[rows.Record]) -> list[rows.Record]:
        """The metadata row is not a box, so no box column belongs on it.

        It arrives carrying whatever the earlier stages wrote -- as nulls, since
        none of it applies -- and a null on one row is still a column on the
        whole table.
        """
        return [{k: v for k, v in record.items() if k not in rows.POSTPROCESS_COLUMNS} for record in metadata]


@dataclass
class MarkdownAssemblyStage(_FinalStage):
    """Put each page's flow back in front of its floats, and write markdown.

    The exit from post-processing: positions are renumbered from zero,
    ``content_type`` becomes ``text/markdown``, and the working columns that
    only existed to carry state between the steps are dropped.  Image rows keep
    their bytes and their own content type -- an interleaved document is text
    *and* pictures in one reading order, which is the whole point of the
    format.

    See :func:`~.steps.assemble` and :func:`~.markdown.render`.  Takes
    ``emit_dropped`` from :class:`_FinalStage`.
    """

    name: str = "nemotron_parse_markdown"

    def document(self, metadata: list[rows.Record], elements: list[rows.Record]) -> list[rows.Record]:
        if not elements:
            return self._clean_metadata(metadata)
        boxes = steps.assemble(rows.to_layout(elements))
        records = _markdown_rows(boxes, rows.templates(elements), emit_dropped=self.emit_dropped)
        return [*self._clean_metadata(_with_stats(metadata, steps.summarise(boxes))), *records]


# --------------------------------------------------------------------------
# all six at once
# --------------------------------------------------------------------------


@dataclass
class FusedPostprocessingStage(_FinalStage):
    """Every step in one stage, for when throughput beats inspectability.

    Produces exactly what the six stages produce -- :func:`~.steps.postprocess`
    is the same composition -- without writing the intermediate state out and
    reading it back five times.  Takes ``emit_dropped`` from
    :class:`_FinalStage`.
    """

    name: str = "nemotron_parse_postprocess_fused"

    def document(self, metadata: list[rows.Record], elements: list[rows.Record]) -> list[rows.Record]:
        if not elements:
            return self._clean_metadata(metadata)
        document = steps.postprocess(rows.to_elements(elements), self.config)
        records = _markdown_rows(document.boxes, rows.templates(elements), emit_dropped=self.emit_dropped)
        return [*self._clean_metadata(_with_stats(metadata, document.stats)), *records]
