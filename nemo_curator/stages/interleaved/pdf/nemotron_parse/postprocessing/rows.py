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

"""The bridge between interleaved rows and the pure post-processing layer.

:mod:`.steps` knows nothing about Arrow, pandas or the interleaved schema; this
module is the only place that does.  It reads a document's rows into
:class:`~.model.Element` or :class:`~.model.Layout`, and writes the result back
out as rows.

Two conventions make the round trip lossless, so every step can be a separate
pipeline stage without the intermediate state having anywhere to hide:

* A box is written back onto the row of its **first** source element.  That is
  what carries the columns the pure layer never sees -- ``url``, ``pdf_name``,
  the image bytes in ``binary_content`` -- through a step that rewrote the text.
* The flow/float split and the float ordering are held by the :data:`SLOT`
  column plus row order, both of which survive Parquet.

A note on ``source_ref``.  The interleaved schema documents it as a *storage*
locator -- ``{path, member, byte_offset, byte_size, frame_index}`` -- but the
Nemotron-Parse element format has always written ``{page, bbox}`` there, and
that is what the parquet already on disk contains.  This module reads and
writes what is actually there rather than introducing a second convention that
would split the corpus in two.  Reconciling the two meanings is worth doing,
but it is a change to the element format, not to post-processing.

Columns added to the Nemotron-Parse element format by post-processing:

    ==================  ==============  ==================================================
    Column              Type            Meaning
    ==================  ==============  ==================================================
    ``keep``            bool            Would this box reach training?
    ``drop_reason``     string          Why not.  Empty when kept.
    ``source_positions``string          JSON list of the element positions this box came
                                        from -- more than one after paragraphs are rejoined
    ``slot``            string          ``"flow"`` or ``"float"``
    ``matched_to``      int32           For a caption, the position of its figure
    ==================  ==============  ==================================================
"""

from __future__ import annotations

import json
from itertools import groupby
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

import pyarrow as pa

from nemo_curator.stages.interleaved.pdf.nemotron_parse.postprocessing.model import (
    BBox,
    Box,
    Element,
    Layout,
    Page,
)

#: One row, as :meth:`pyarrow.Table.to_pylist` gives it: plain Python values,
#: ``None`` for null.  Records rather than a DataFrame because every step walks
#: the rows one at a time, and ``DataFrame.iterrows`` over Arrow-backed columns
#: costs more than the rules it is feeding.
Record = dict[str, Any]

# -- reserved interleaved columns this module reads or writes --
SAMPLE_ID = "sample_id"
POSITION = "position"
MODALITY = "modality"
CONTENT_TYPE = "content_type"
TEXT_CONTENT = "text_content"
BINARY_CONTENT = "binary_content"
SOURCE_REF = "source_ref"

# -- user columns the Nemotron-Parse element format carries --
ELEMENT_CLASS = "element_class"
PAGE_NUMBER = "page_number"

# -- user columns post-processing adds --
KEEP = "keep"
DROP_REASON = "drop_reason"
SOURCE_POSITIONS = "source_positions"
SLOT = "slot"
MATCHED_TO = "matched_to"

SPANS_PAGES = "spans_pages"

POSTPROCESS_COLUMNS = (KEEP, DROP_REASON, SOURCE_POSITIONS, SLOT, MATCHED_TO, SPANS_PAGES)

#: Types for the columns this package writes, pinned rather than inferred.
#: Inference reads them off the values, and a batch in which every caption
#: happens to be unmatched would give ``matched_to`` the null type -- two
#: Parquet files in one dataset with incompatible schemas, and a reader that
#: fails on whichever it opens second.
EXTRA_SCHEMA = pa.schema(
    [
        pa.field(ELEMENT_CLASS, pa.string()),
        pa.field(PAGE_NUMBER, pa.int32()),
        pa.field(KEEP, pa.bool_()),
        pa.field(DROP_REASON, pa.string()),
        pa.field(SOURCE_POSITIONS, pa.string()),
        pa.field(SLOT, pa.string()),
        pa.field(MATCHED_TO, pa.int32()),
        pa.field(SPANS_PAGES, pa.bool_()),
    ]
)

#: The per-document row that carries ``{url, pdf_name, num_pages}`` and, once
#: post-processing has run, its :class:`~.model.Stats`.  It sits outside the
#: element stream and every step passes it through untouched.
METADATA_POSITION = -1
METADATA_MODALITY = "metadata"

FLOW = "flow"
FLOAT = "float"

MARKDOWN_CONTENT_TYPE = "text/markdown"

#: A bbox is four normalised coordinates or it is not a bbox.
_BBOX_LEN = 4


def _text_of(record: Record, column: str) -> str:
    value = record.get(column)
    return "" if value is None else str(value)


def _int_of(record: Record, column: str) -> int | None:
    value = record.get(column)
    return None if value is None else int(value)


def build_source_ref(page: int | None, bbox: BBox | None) -> str:
    """The locator string the element format stores in ``source_ref``."""
    return json.dumps({"page": page, "bbox": list(bbox) if bbox else None})


def parse_source_ref(value: str | None) -> tuple[int | None, BBox | None]:
    """``(page, bbox)`` from a ``source_ref``.  Unreadable means unknown."""
    if not value:
        return None, None
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return None, None
    if not isinstance(parsed, dict):
        return None, None
    page = parsed.get("page")
    bbox = parsed.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != _BBOX_LEN:
        coordinates = None
    else:
        try:
            coordinates = tuple(float(v) for v in bbox)
        except (TypeError, ValueError):
            # A locator we cannot read means unknown geometry, not a failed
            # document: the rules degrade to reading order without a bbox.
            coordinates = None
    return int(page) if isinstance(page, (int, float)) else None, coordinates


def _page_of(record: Record) -> int | None:
    """Prefer the typed column; fall back to the locator."""
    page = _int_of(record, PAGE_NUMBER)
    if page is not None:
        return page
    return parse_source_ref(record.get(SOURCE_REF))[0]


def _bbox_of(record: Record) -> BBox | None:
    return parse_source_ref(record.get(SOURCE_REF))[1]


def _has_payload(record: Record) -> bool:
    """An image element: empty text is not emptiness."""
    payload = record.get(BINARY_CONTENT)
    return payload is not None and len(payload) > 0


def split_documents(records: Sequence[Record]) -> Iterator[tuple[str, list[Record], list[Record]]]:
    """``(sample_id, metadata rows, element rows)`` per document, in input order.

    Row order is left exactly as it was found.  It has to be: from
    :func:`~.steps.assign_floats` onwards the reading order is the row order and
    nothing else -- ``position`` is by then only a join key back to the element
    a box came from, and floats deliberately sit out of position.  The first
    step sorts its own input, so nothing here needs to.

    This is the interleaved format's own convention, not a local one: a
    document's segments interleave by row order, with no ordering column.
    """
    grouped: dict[str, tuple[list[Record], list[Record]]] = {}
    for record in records:
        sample_id = str(record.get(SAMPLE_ID))
        metadata, elements = grouped.setdefault(sample_id, ([], []))
        (metadata if record.get(POSITION) == METADATA_POSITION else elements).append(record)
    for sample_id, (metadata, elements) in grouped.items():
        yield sample_id, metadata, elements


def to_elements(records: Sequence[Record]) -> list[Element]:
    """A document's element rows as :class:`~.model.Element`."""
    return [
        Element(
            position=int(record[POSITION]),
            modality=_text_of(record, MODALITY) or "text",
            element_class=_text_of(record, ELEMENT_CLASS) or "Text",
            text=_text_of(record, TEXT_CONTENT),
            page=_page_of(record),
            bbox=_bbox_of(record),
            has_payload=_has_payload(record),
        )
        for record in records
    ]


def _to_box(record: Record) -> Box:
    """One row that a previous post-processing stage already wrote."""
    positions = record.get(SOURCE_POSITIONS)
    source_positions = (
        (int(record[POSITION]),) if positions is None else tuple(int(p) for p in json.loads(str(positions)))
    )
    keep = record.get(KEEP)
    return Box(
        text=_text_of(record, TEXT_CONTENT),
        element_class=_text_of(record, ELEMENT_CLASS) or "Text",
        keep=True if keep is None else bool(keep),
        reason=_text_of(record, DROP_REASON),
        page=_page_of(record),
        bbox=_bbox_of(record),
        source_positions=source_positions,
        modality=_text_of(record, MODALITY) or "text",
        has_payload=_has_payload(record),
        spans_pages=bool(record.get(SPANS_PAGES)),
        matched_to=_int_of(record, MATCHED_TO),
    )


def to_boxes(records: Sequence[Record]) -> tuple[Box, ...]:
    """A document's rows as a flat box stream, in row order."""
    return tuple(_to_box(record) for record in records)


def to_layout(records: Sequence[Record]) -> Layout:
    """A document's rows as a :class:`~.model.Layout`.

    Rows with no :data:`SLOT` are all flow -- that is how the layout looks
    before :func:`~.steps.assign_floats` has run.  Pages are cut where the page
    number changes, so the rows must already be page-contiguous, which the
    first step's sort guarantees and no later step disturbs.
    """
    slotted = [(_to_box(record), _text_of(record, SLOT)) for record in records]
    return tuple(
        Page(
            key=key,
            flow=tuple(box for box, slot in items if slot != FLOAT),
            floats=tuple(box for box, slot in items if slot == FLOAT),
        )
        for key, items in ((key, list(group)) for key, group in groupby(slotted, key=lambda pair: pair[0].page_key))
    )


def _box_row(box: Box, template: Record, slot: str) -> Record:
    """One box written back onto the row of its first source element."""
    record = dict(template)
    record.update(
        {
            TEXT_CONTENT: box.text,
            MODALITY: box.modality,
            ELEMENT_CLASS: box.element_class,
            PAGE_NUMBER: box.page,
            SOURCE_REF: build_source_ref(box.page, box.bbox),
            KEEP: box.keep,
            DROP_REASON: box.reason,
            SOURCE_POSITIONS: json.dumps(list(box.source_positions)),
            SLOT: slot,
            MATCHED_TO: box.matched_to,
            SPANS_PAGES: box.spans_pages,
        }
    )
    return record


def templates(records: Sequence[Record]) -> dict[int, Record]:
    """Element position -> its original row, for carrying passthrough columns.

    A document with two elements at the same ``position`` is malformed; the
    later row wins, as it would anywhere else that treats position as an id.
    """
    return {int(record[POSITION]): record for record in records}


def from_boxes(
    boxes: tuple[Box, ...],
    template_by_position: dict[int, Record],
    slot: str = FLOW,
) -> list[Record]:
    """A flat box stream as rows."""
    return [_box_row(box, template_by_position.get(box.source_positions[0], {}), slot) for box in boxes]


def from_layout(layout: Layout, template_by_position: dict[int, Record]) -> list[Record]:
    """A layout as rows: each page's flow, then that page's floats.

    The same order :func:`~.steps.assemble` produces, so a layout written out
    here and read back by :func:`to_layout` is the layout that went in.
    """
    records: list[Record] = []
    for page in layout:
        records.extend(from_boxes(page.flow, template_by_position, FLOW))
        records.extend(from_boxes(page.floats, template_by_position, FLOAT))
    return records
