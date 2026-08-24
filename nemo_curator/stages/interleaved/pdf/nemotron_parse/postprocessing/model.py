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

"""Data model for Nemotron-Parse post-processing.

Deliberately free of any Curator, Ray, parquet or Arrow types: the caller
adapts its own storage into :class:`Element` and reads :class:`Box` back.  That
is what lets the same code run inside a Curator stage, inside a notebook, or
inside a viewer, and be tested on plain Python objects alone.

The vocabulary:

``Element``
    One row of Nemotron-Parse output.
``Box``
    One unit of post-processed output.  A Box may correspond to several input
    Elements -- that is exactly what paragraph reconstitution does.
``Page``
    One page split into its reading *flow* and its *floats* (tables, pictures,
    captions, footnotes), which are pulled out of the flow and re-emitted after
    it.
``Layout``
    The whole document as an ordered tuple of Pages.  Every post-processing
    step is a ``Layout -> Layout`` function, which is what lets each one be a
    separate pipeline stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

BBox = tuple[float, float, float, float]

#: Page furniture.  Skipped TRANSPARENTLY during paragraph reconstitution: it
#: does not end an open paragraph, which is what lets a sentence broken across
#: a page boundary be rejoined over the page number sitting between its halves.
PAGE_FURNITURE = frozenset({"Page-header", "Page-footer"})

#: Pulled out of the reading flow and re-emitted after their page.
FLOAT_CLASSES = frozenset({"Table", "Picture", "Caption", "Footnote"})

#: The float classes a caption can be matched to.
FIGURE_CLASSES = ("Table", "Picture")

#: End an open paragraph but are themselves kept.
FLUSH_CLASSES = frozenset({"Title", "Section-header", "Formula"})

#: Classes whose whole section is skipped when ``skip_toc_bib`` is on.
SECTION_SKIP_CLASSES = ("Bibliography", "TOC")

#: Headings whose section is skipped when ``skip_toc_bib`` is on.
TOC_HEADINGS = frozenset(
    {
        "table of contents",
        "bibliography",
        "index",
        "references",
        "contents",
        "list of figures",
        "list of tables",
        "list of illustrations",
    }
)

#: Headings that can *open* a skipped section (a Title or Section-header whose
#: text is one of :data:`TOC_HEADINGS`).
TOC_HEADING_CLASSES = ("Title", "Section-header")

#: ``page`` is optional on an Element; this stands in for "no page known" so
#: that ordering and grouping have a total key.
NO_PAGE = -1


@dataclass(frozen=True, slots=True)
class Element:
    """One row of Nemotron-Parse output, adapted by the caller.

    Parameters
    ----------
    position
        The element's index in the model's reading order.  Unique within a
        document; used as the join key back to the caller's own rows.
    modality
        ``"text"``, ``"image"`` or ``"table"`` -- the interleaved row modality.
        Carried through untouched; ``has_payload`` is what the rules look at.
    element_class
        The layout class the model assigned, e.g. ``"Text"``, ``"Table"``,
        ``"Section-header"``.  An unrecognised class is treated as prose.
    text
        The element's text.  Empty for a Picture, whose payload is its image.
    page
        Zero-based page number, or ``None`` when unknown.
    bbox
        Normalised ``(x1, y1, x2, y2)``, or ``None`` when the model emitted no
        box.  Column detection and caption matching both need it; without it
        they degrade rather than fail.
    has_payload
        ``True`` when the element carries non-text content -- an image.  Empty
        text is then not emptiness, so the element survives the empty check and
        reaches the interleaved output as an image row.
    """

    position: int
    modality: str
    element_class: str
    text: str
    page: int | None = None
    bbox: BBox | None = None
    has_payload: bool = False

    @property
    def page_key(self) -> int:
        """A total ordering key: ``NO_PAGE`` when the page is unknown."""
        return NO_PAGE if self.page is None else int(self.page)


@dataclass(frozen=True, slots=True)
class Box:
    """One unit of post-processed output.

    A Box may correspond to several input Elements -- that is exactly what
    paragraph reconstitution does -- so ``source_positions`` records which
    ones, in order.  A viewer can use it to show where the original segment
    boundaries were even after they have been joined.
    """

    text: str
    element_class: str
    keep: bool
    #: Empty when kept.  Otherwise why this box would not reach training.
    reason: str = ""
    page: int | None = None
    bbox: BBox | None = None
    source_positions: tuple[int, ...] = ()
    modality: str = "text"
    has_payload: bool = False
    #: Set when this box was rejoined across a page boundary, in which case
    #: ``page`` and ``bbox`` describe different pieces of paper -- the page it
    #: started on, and the fragment that finished it.  Recorded where the join
    #: happens, because by the time the document is written out the absorbed
    #: box is gone and nothing can tell after the fact.
    spans_pages: bool = False
    #: For a Caption, the position of the figure it was matched to.  ``None``
    #: on everything else, and on a caption no figure claimed.  Recording the
    #: assignment on the box is what makes it survive a round trip through
    #: storage, where the ordering alone would not: an unmatched caption can
    #: land directly after an unmatched figure.
    matched_to: int | None = None

    @property
    def is_joined(self) -> bool:
        return len(self.source_positions) > 1

    @property
    def page_key(self) -> int:
        return NO_PAGE if self.page is None else int(self.page)

    def discarded(self, reason: str) -> Box:
        return replace(self, keep=False, reason=reason)


@dataclass(frozen=True, slots=True)
class Page:
    """One page, split into reading flow and floats.

    ``flow`` is the text a reader moves through top to bottom; ``floats`` are
    the tables, pictures, captions and footnotes lifted out of it and re-emitted
    after it, each figure followed by the caption matched to it.
    """

    key: int
    flow: tuple[Box, ...] = ()
    floats: tuple[Box, ...] = ()


#: A document, as an ordered tuple of pages.  Every step is ``Layout -> Layout``.
Layout = tuple[Page, ...]


@dataclass(frozen=True, slots=True)
class Stats:
    """Counts mirroring what the predecessor pipeline reported."""

    boxes_in: int = 0
    boxes_kept: int = 0
    boxes_joined: int = 0
    boxes_page_furniture: int = 0
    boxes_rejected: int = 0
    words_kept: int = 0
    words_rejected: int = 0
    paragraphs_reconstituted: int = 0
    captions_matched: int = 0
    latin_chars: int = 0
    nonlatin_chars: int = 0


DropReason = Literal[
    "page-furniture",
    "toc-bibliography",
    "short-caption",
    "table-without-tabular",
    "repeated-words",
    "joined-into-previous",
    "dropped-class",
    "empty",
]

#: The one reason that flushes an open paragraph without condemning it.  A
#: section skip ends the paragraph before it; a box condemned for any other
#: reason breaks a continuation that was already open, so the box holding the
#: first half is condemned too.
SECTION_SKIP_REASON = "toc-bibliography"


@dataclass(frozen=True, slots=True)
class ProcessedDocument:
    """The result.

    Nothing is deleted -- every input Element is represented by some Box, kept
    or not -- so a viewer can render the full stream while a training pipeline
    reads ``.text``.
    """

    boxes: tuple[Box, ...] = ()
    stats: Stats = field(default_factory=Stats)

    @property
    def text(self) -> str:
        """What a training pipeline would consume: kept boxes only."""
        return "\n\n".join(b.text for b in self.boxes if b.keep and b.text)

    @property
    def kept(self) -> tuple[Box, ...]:
        return tuple(b for b in self.boxes if b.keep)


@dataclass(frozen=True, slots=True)
class Config:
    """Tunables, defaulted to the values the predecessor pipeline was tuned
    with, so behaviour stays comparable to that baseline -- with the
    exceptions noted where they are defined.  Every step can be switched off
    independently, which is what makes a pre/post comparison meaningful.
    """

    # A caption shorter than this is layout noise rather than content.
    min_caption_chars: int = 100
    min_caption_words: int = 10
    # Leaving TOC/bibliography once a paragraph is this substantial.
    non_bib_toc_words: int = 35
    # A long block this dominated by one word is degenerate output.
    repeat_word_ratio: float = 0.9
    repeat_word_min_words: int = 100

    # Off, unlike the predecessor pipeline.  Nemotron-Parse stores
    # ``text/markdown`` and the markup is content: ``**bold**`` table cells,
    # ``_emphasis_``, and the ``##`` a heading carries.  Stripping it also
    # damages what it does not own -- the italic rule ``_..._`` spans LaTeX
    # subscripts, so ``\(\hat{H}_a+\hat{H}_em\)`` loses its underscores.  Set
    # it True to reproduce the predecessor pipeline's plain-text baseline.
    strip_markdown: bool = False
    assign_floats: bool = True
    reconstitute_paragraphs: bool = True
    skip_toc_bib: bool = True
    drop_page_furniture: bool = True
    check_repeated_words: bool = True
    require_tabular_in_tables: bool = True

    # On, unlike the text-only baseline this was ported from.  An image element
    # carries its payload in a binary column the rules never see, so judging it
    # by its (empty) text would drop every picture in the corpus -- which is
    # exactly what interleaved output is for.  Set it False to reproduce the
    # text-only baseline.
    keep_images: bool = True

    #: Extra classes the caller does not want, e.g. ``{"Bibliography"}``.
    drop_classes: frozenset[str] = frozenset()
