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

"""The post-processing steps, as pure functions.

Six steps turn a page-ordered list of Nemotron-Parse elements into a document a
reader could follow.  Each is a separate function with a single job, and each
is wrapped one-to-one by a Curator stage in :mod:`.stages`, so the pipeline you
see in ``pipeline.describe()`` is the list below:

1. :func:`clean` -- order the elements, normalise their text, and condemn the
   ones that never had content: empty, a Table that lost its ``tabular``, a
   block degenerate with one repeated word.
2. :func:`assign_floats` -- lift tables, pictures, captions and footnotes out
   of the reading flow, match each caption to the figure nearest it, and order
   them figure-then-caption for re-emission after the page.
3. :func:`mark_page_furniture` -- condemn running heads and page numbers.
4. :func:`skip_sections` -- condemn the table of contents and the bibliography,
   from their heading until prose resumes.
5. :func:`reconstitute_paragraphs` -- rejoin a sentence broken across a column
   or a page boundary.
6. :func:`assemble` -- put each page's flow back in front of its floats.

Nothing is deleted.  A box that would not reach training is marked
``keep=False`` with a reason, so a viewer can show the whole stream while a
training pipeline reads only what survived.  The one exception is a box
absorbed into the paragraph before it, which lives on inside that box's
``source_positions``.

The order matters and is not free to change: floats must leave the flow before
the flow is scanned (2 before 4 and 5), and a box's drop reason must be settled
before reconstitution reads it (1, 3, 4 before 5).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from itertools import groupby

from nemo_curator.stages.interleaved.pdf.nemotron_parse.postprocessing import geometry
from nemo_curator.stages.interleaved.pdf.nemotron_parse.postprocessing import text as _text
from nemo_curator.stages.interleaved.pdf.nemotron_parse.postprocessing.model import (
    FIGURE_CLASSES,
    FLOAT_CLASSES,
    FLUSH_CLASSES,
    NO_PAGE,
    PAGE_FURNITURE,
    SECTION_SKIP_CLASSES,
    SECTION_SKIP_REASON,
    TOC_HEADING_CLASSES,
    TOC_HEADINGS,
    Box,
    Config,
    Element,
    Layout,
    Page,
    ProcessedDocument,
    Stats,
)

#: Text holding one of these already encodes its own structure, so joining it
#: onto the paragraph before would run two blocks together.
_HARD_BREAKS = ("<br>", "\n")


def _position(box: Box) -> int:
    """The element a box came from.

    Only called before :func:`reconstitute_paragraphs`, where every box still
    stands for exactly one element.
    """
    return box.source_positions[0]


# --------------------------------------------------------------------------
# 1. clean
# --------------------------------------------------------------------------


def _clean_one(element: Element, cfg: Config) -> Box:
    """One element, normalised and judged.  An empty reason means keep."""
    body = element.text.strip()
    if cfg.strip_markdown:
        body = _text.strip_markdown(body).strip()

    reason = ""
    if not body and not (cfg.keep_images and element.has_payload):
        reason = "empty"
    elif element.element_class in cfg.drop_classes:
        reason = "dropped-class"
    elif cfg.require_tabular_in_tables and element.element_class == "Table" and not _text.contains_tabular(body):
        reason = "table-without-tabular"
    elif cfg.check_repeated_words and _text.is_degenerate(
        body, min_words=cfg.repeat_word_min_words, max_ratio=cfg.repeat_word_ratio
    ):
        reason = "repeated-words"

    return Box(
        text=body,
        element_class=element.element_class,
        keep=not reason,
        reason=reason,
        page=element.page,
        bbox=element.bbox,
        source_positions=(element.position,),
        modality=element.modality,
        has_payload=element.has_payload,
    )


def clean(elements: list[Element] | tuple[Element, ...], cfg: Config) -> tuple[Box, ...]:
    """Order the elements by ``(page, position)`` and judge each one alone.

    Ordering lives here rather than in a step of its own because every later
    step depends on it and none of them can restore it.
    """
    ordered = sorted(elements, key=lambda e: (e.page_key, e.position))
    return tuple(_clean_one(e, cfg) for e in ordered)


# --------------------------------------------------------------------------
# 2. assign_floats
# --------------------------------------------------------------------------


def _short_caption_reasons(floats: list[Box], cfg: Config) -> dict[int, str]:
    """Captions too short to be content are layout noise, not a caption."""
    reasons: dict[int, str] = {}
    for box in floats:
        if box.element_class != "Caption":
            continue
        if len(box.text) <= cfg.min_caption_chars or box.text.count(" ") <= cfg.min_caption_words:
            reasons[_position(box)] = "short-caption"
    return reasons


def _order_floats(floats: list[Box], assignment: dict[int, int], figures: list[int]) -> list[Box]:
    """Figures in reading order, each followed by its caption.

    Whatever is left follows: unmatched figures and captions first, footnotes
    last, since a footnote belongs at the foot of the page.
    """
    by_position = {_position(b): b for b in floats}
    ordered: list[Box] = []
    placed: set[int] = set()

    for figure in figures:
        ordered.append(by_position[figure])
        placed.add(figure)
        caption = assignment.get(figure)
        if caption is not None:
            ordered.append(replace(by_position[caption], matched_to=figure))
            placed.add(caption)

    for box in floats:
        if _position(box) not in placed and box.element_class != "Footnote":
            ordered.append(box)
            placed.add(_position(box))

    for box in floats:
        if _position(box) not in placed:
            ordered.append(box)

    return ordered


def assign_floats(boxes: tuple[Box, ...], cfg: Config) -> Layout:
    """Split each page into its reading flow and its floats.

    Tables, pictures, captions and footnotes are lifted out of the flow so a
    figure dropped into the middle of a column does not cut a sentence in half.
    Each caption is then matched to the figure nearest it -- an assignment, not
    a nearest-neighbour scan, so two captions cannot claim the same figure --
    and the floats are ordered figure-then-caption for re-emission after the
    page's text.
    """
    pages: list[Page] = []
    for key, group in groupby(boxes, key=lambda b: b.page_key):
        page_boxes = list(group)

        if not cfg.assign_floats:
            pages.append(Page(key=key, flow=tuple(page_boxes)))
            continue

        floats = [b for b in page_boxes if b.element_class in FLOAT_CLASSES]
        flow = [b for b in page_boxes if b.element_class not in FLOAT_CLASSES]
        if not floats:
            pages.append(Page(key=key, flow=tuple(flow)))
            continue

        caption_reasons = _short_caption_reasons(floats, cfg)
        figures = [_position(b) for b in floats if b.element_class in FIGURE_CLASSES]
        captions = [
            _position(b) for b in floats if b.element_class == "Caption" and _position(b) not in caption_reasons
        ]
        bboxes = {_position(b): b.bbox for b in floats}
        assignment = geometry.match_captions(figures, captions, bboxes)

        ordered = [
            b.discarded(caption_reasons[_position(b)]) if _position(b) in caption_reasons else b
            for b in _order_floats(floats, assignment, figures)
        ]
        pages.append(Page(key=key, flow=tuple(flow), floats=tuple(ordered)))

    return tuple(pages)


# --------------------------------------------------------------------------
# 3. mark_page_furniture
# --------------------------------------------------------------------------


def mark_page_furniture(layout: Layout, cfg: Config) -> Layout:
    """Condemn running heads and page numbers.

    They are marked here and skipped *transparently* by the two steps that
    follow: furniture does not end an open paragraph, which is what lets a
    sentence broken across a page boundary be rejoined over the page number
    sitting between its halves.
    """
    if not cfg.drop_page_furniture:
        return layout

    pages: list[Page] = []
    for page in layout:
        flow = tuple(
            box.discarded("page-furniture") if box.element_class in PAGE_FURNITURE else box for box in page.flow
        )
        pages.append(replace(page, flow=flow))
    return tuple(pages)


# --------------------------------------------------------------------------
# 4. skip_sections
# --------------------------------------------------------------------------


def skip_sections(layout: Layout, cfg: Config) -> Layout:  # noqa: C901 -- one branch per way a section starts or ends
    """Condemn the table of contents and the bibliography.

    A section is entered either by class (the model emits ``Bibliography`` or
    ``TOC`` outright) or by heading text, and left again at the next heading --
    or earlier, as soon as a block of prose is substantial enough that it
    cannot be a run of reference entries.  That escape hatch matters because
    the model labels an appendix following the references as part of them.
    """
    if not cfg.skip_toc_bib:
        return layout

    skipping = False
    pages: list[Page] = []
    for page in layout:
        flow = list(page.flow)
        for i, box in enumerate(flow):
            cls = box.element_class

            if cls in PAGE_FURNITURE:
                continue

            if cls in SECTION_SKIP_CLASSES:
                skipping = True
                flow[i] = box.discarded(SECTION_SKIP_REASON)
                continue

            if not box.keep:
                # Already condemned by an earlier step, and a box with no
                # content says nothing about where a section begins or ends.
                continue

            if cls in FLUSH_CLASSES:
                if cls in TOC_HEADING_CLASSES and box.text.lower().strip() in TOC_HEADINGS:
                    skipping = True
                    flow[i] = box.discarded(SECTION_SKIP_REASON)
                else:
                    skipping = False
                continue

            if skipping:
                # Reference entries are short and dense with punctuation;
                # discount it so a list of citations cannot reach the
                # threshold on commas and full stops alone.
                net = _text.count_words(box.text) - box.text.count(".") - box.text.count(",")
                if net >= cfg.non_bib_toc_words:
                    skipping = False
                else:
                    flow[i] = box.discarded(SECTION_SKIP_REASON)

        pages.append(replace(page, flow=tuple(flow)))
    return tuple(pages)


# --------------------------------------------------------------------------
# 5. reconstitute_paragraphs
# --------------------------------------------------------------------------


def _joinable(box: Box) -> bool:
    """Only text carrying no internal line break can be joined."""
    return not any(mark in box.text for mark in _HARD_BREAKS)


def _same_column(box: Box, previous: Box | None, previous_page: int, page: int) -> bool:
    """Do these two boxes sit in one column, on one page?

    Two stacked boxes in the same column overlap in x, so the gap between them
    is an ordinary paragraph break.  Boxes that do not overlap sit in different
    columns, so the break is a layout artefact and the text may run on.
    """
    return (
        previous is not None
        and previous_page == page
        and previous.bbox is not None
        and box.bbox is not None
        and geometry.x_overlap(box.bbox, previous.bbox)
    )


def _merge(previous: Box, box: Box, *, same_page: bool) -> Box:
    """Absorb ``box`` into the paragraph ``previous`` holds open."""
    bbox = geometry.merge(previous.bbox, box.bbox) if same_page and previous.bbox and box.bbox else box.bbox
    return replace(
        previous,
        text=f"{previous.text} {box.text}",
        bbox=bbox,
        source_positions=(*previous.source_positions, *box.source_positions),
        spans_pages=previous.spans_pages or not same_page,
    )


def reconstitute_paragraphs(layout: Layout, cfg: Config) -> Layout:  # noqa: C901, PLR0912 -- one branch per way a paragraph can end
    """Rejoin a sentence broken across a column or a page boundary.

    A box that stops mid-thought holds a paragraph *open*; the next box of
    running text closes it, unless the two sit in the same column of the same
    page, where the break between them is a real paragraph break rather than a
    layout artefact.  Page furniture between the halves is stepped over, which
    is the whole point: the page number is not the end of the sentence.

    A box condemned by an earlier step ends the open paragraph.  If it was
    condemned for *content* -- not merely because a section is being skipped --
    the half already written is condemned with it, because the continuation it
    was waiting for will never arrive.
    """
    flows = [list(page.flow) for page in layout]
    absorbed: set[tuple[int, int]] = set()
    active: tuple[int, int] | None = None
    active_page: int = NO_PAGE

    for page_index, page in enumerate(layout):
        for box_index, box in enumerate(flows[page_index]):
            cls = box.element_class

            if cls in PAGE_FURNITURE:
                continue

            if not box.keep:
                if active is not None and box.reason != SECTION_SKIP_REASON:
                    held = flows[active[0]][active[1]]
                    flows[active[0]][active[1]] = held.discarded("joined-into-previous")
                active = None
                continue

            if box.has_payload:
                # Transparent, like page furniture.  An image carries its
                # payload in a column these rules never see, so joining it into
                # the paragraph before would silently delete a picture -- the
                # text it contributes is empty, and the bytes go with the box.
                # Stepping over it also means prose still runs on across a
                # figure dropped into the middle of a column, which is what
                # happens anyway when floats are lifted out of the flow.
                continue

            if cls in FLUSH_CLASSES:
                active = None
                continue

            if not cfg.reconstitute_paragraphs:
                continue

            if cls == "List-item":
                # A list item never continues into the next box, but it can be
                # continued *by* one, so it may hold the paragraph open.
                active = (page_index, box_index) if _text.is_to_be_continued(box.text) else None
                active_page = page.key
                continue

            previous = flows[active[0]][active[1]] if active is not None else None
            if _joinable(box) and previous is not None and not _same_column(box, previous, active_page, page.key):
                flows[active[0]][active[1]] = _merge(previous, box, same_page=active_page == page.key)
                absorbed.add((page_index, box_index))
                active_page = page.key
                if not _text.is_to_be_continued(box.text):
                    active = None
                continue

            if _joinable(box) and _text.is_to_be_continued(box.text):
                active = (page_index, box_index)
                active_page = page.key
            else:
                active = None

    return tuple(
        replace(page, flow=tuple(b for i, b in enumerate(flows[pi]) if (pi, i) not in absorbed))
        for pi, page in enumerate(layout)
    )


# --------------------------------------------------------------------------
# 6. assemble
# --------------------------------------------------------------------------


def assemble(layout: Layout) -> tuple[Box, ...]:
    """Each page's flow, then that page's floats.

    Putting the floats after the text they belong to is what makes the output
    read in order: a figure lifted out of the middle of a column comes back at
    the foot of its own page, not at the end of the document.
    """
    boxes: list[Box] = []
    for page in layout:
        boxes.extend(page.flow)
        boxes.extend(page.floats)
    return tuple(boxes)


# --------------------------------------------------------------------------
# stats and the whole pipeline
# --------------------------------------------------------------------------


def summarise(boxes: tuple[Box, ...]) -> Stats:
    """Counts for comparing one run against another.

    Everything is read back off the boxes, so the numbers can be recomputed at
    any point in the pipeline -- including after the boxes have been written to
    storage and read back.
    """
    kept = [b for b in boxes if b.keep]
    body = "\n\n".join(b.text for b in kept)
    latin, nonlatin = _text.count_char_types(body)
    by_reason: dict[str, int] = defaultdict(int)
    for b in boxes:
        if not b.keep:
            by_reason[b.reason] += 1

    return Stats(
        boxes_in=sum(len(b.source_positions) for b in boxes),
        boxes_kept=len(kept),
        boxes_joined=sum(1 for b in boxes if b.is_joined),
        boxes_page_furniture=by_reason["page-furniture"],
        boxes_rejected=sum(1 for b in boxes if not b.keep),
        words_kept=_text.count_words(body),
        words_rejected=sum(_text.count_words(b.text) for b in boxes if not b.keep),
        # Every merge appends exactly one source position, so the joins a box
        # went through are recoverable from the box itself.
        paragraphs_reconstituted=sum(len(b.source_positions) - 1 for b in boxes),
        captions_matched=sum(1 for b in boxes if b.matched_to is not None),
        latin_chars=latin,
        nonlatin_chars=nonlatin,
    )


def to_layout(elements: list[Element] | tuple[Element, ...], cfg: Config) -> Layout:
    """Steps 1-2: elements in, a page-split layout out."""
    return assign_floats(clean(elements, cfg), cfg)


def postprocess(elements: list[Element] | tuple[Element, ...], cfg: Config | None = None) -> ProcessedDocument:
    """Every step, in order.

    A pure function: it reads its input and returns a new
    :class:`ProcessedDocument`.  Nothing is mutated.
    """
    cfg = cfg or Config()
    layout = to_layout(elements, cfg)
    layout = mark_page_furniture(layout, cfg)
    layout = skip_sections(layout, cfg)
    layout = reconstitute_paragraphs(layout, cfg)
    boxes = assemble(layout)
    return ProcessedDocument(boxes=boxes, stats=summarise(boxes))
