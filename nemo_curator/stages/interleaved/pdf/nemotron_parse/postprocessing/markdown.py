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

r"""Render post-processed boxes as markdown.

The final step, and deliberately the only one that knows what markdown is:
:mod:`.steps` decides *what survives*, this decides *how it is written down*.
Keeping them apart is what lets the drop rules be tuned without touching the
output format, and the format be changed without re-running the rules.

The mapping is per ``element_class``, one function each, collected in
:data:`RENDERERS`.  A class with no entry falls through to a plain paragraph,
so an element_class a later Nemotron-Parse release adds degrades to readable
text rather than vanishing.

Tables are left as LaTeX.  Nemotron-Parse emits them as ``\begin{tabular}``
with ``**bold**`` and ``\(math\)`` inside the cells; there is no lossless
markdown pipe-table for ``\multicolumn``/``\multirow``, which 47% and 23% of
them use.

Math: inline gets ``$...$``, display gets ``$$...$$``.  The corpus gives no
*textual* signal for the difference -- it writes ``\(...\)`` for both -- but it
gives a structural one, which is stronger.  Inline math stays inside the string
of a ``Text``/``Caption``/``List-item`` element; display math is promoted to a
``Formula`` element of its own, with its own bbox.  Measured over 142,292
formulas, only 0.1% overlap a text column in both axes, i.e. 99.9% genuinely
occupy their own block, and 47% carry a trailing equation number.  So the
element class *is* the inline/display distinction, and this is the one place it
is known.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from nemo_curator.stages.interleaved.pdf.nemotron_parse.postprocessing.model import Box

#: Unicode bullets the corpus uses in place of a markdown marker.  ``-`` and
#: ``*`` are absent on purpose: those already are markdown and are left alone.
_BULLETS = "•‣▪●·⁃"  # noqa: RUF001 -- these are the characters the corpus uses, ambiguous or not

#: A marker markdown already understands: ``- ``, ``* ``, ``+ ``, ``1. ``, ``1) ``.
_MD_MARKER = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_LEADING_BULLET = re.compile(rf"^\s*[{_BULLETS}]\s*")
_LEADING_HASHES = re.compile(r"^(#{1,6})\s*")

#: ``\(\bullet\)`` -- the bullet swallowed into a math span, which is how a
#: sixth of list items arrive.  Two shapes, and the order they are tried
#: matters: when the span holds nothing but the bullet the whole span goes, but
#: when real math follows it only the ``\bullet`` goes and the span stays open,
#: or the ``\(`` would be left without its ``\)``.
_MATH_BULLET_ALONE = re.compile(r"^\s*\\\(\s*\\bullet\s*\\\)\s*")
_MATH_BULLET_LEADING = re.compile(r"^\s*\\\(\s*\\bullet\s*")

#: One ``\(...\)`` span.  Two properties earn their complexity, both measured:
#:
#: It may cross newlines (``\begin{array}`` inside a span -- 4.5% of formulas),
#: hence DOTALL; but it may never swallow a further ``\(``, hence the inner
#: guard.  0.3% of formulas are genuinely unbalanced -- a span that never
#: closes, then a fresh one on the next line -- and a plain non-greedy DOTALL
#: match would pair the orphan opener with the *second* span's closer and eat
#: the real delimiter between them.  Requiring a span to be free of any nested
#: opener leaves the orphan unmatched and converts the well-formed span after
#: it, which is the honest outcome: a delimiter we cannot pair is left exactly
#: as it was rather than guessed at.
#:
#: Only ``\(`` is recognised.  ``\[...\]`` is NOT display math here -- it is an
#: escaped literal bracket, i.e. a citation (``\[28\]``) or a unit (``\[K\]``),
#: and appears in 10.7% of Text elements.  Treating it as math would destroy
#: every citation in the corpus.
_MATH_SPAN = re.compile(r"(?<!\\)\\\(((?:(?!(?<!\\)\\\().)*?)(?<!\\)\\\)", re.DOTALL)

#: A fence has to be longer than the longest backtick run it contains.
_BACKTICK_RUN = re.compile(r"`+")
_MIN_FENCE = 3

TITLE_LEVEL = 1
SECTION_LEVEL = 2


def _math(text: str, delimiter: str) -> str:
    r"""Re-delimit ``\(...\)`` spans, and escape any ``$`` that was already there.

    The escape has to come first: 0.1% of Text elements contain a literal ``$``
    (papers quoting LaTeX), and left alone it would pair with a delimiter we are
    about to introduce and swallow the prose between them.

    An **empty** span is deleted rather than re-delimited.  The model emits
    ``\(\)`` when it saw mathematics it could not read -- the angle brackets of
    ``\langle\rangle`` are a common one -- and there is nothing to delimit.
    Emitting ``$$`` there would be worse than dropping it: it is not an empty
    formula to a markdown reader, it is an *opening* display-math delimiter,
    and it swallows the document down to the next one.  Measured over three
    corpus shards: 26 empty spans in 250,944, touching 0.6% of documents.

    This is a deliberate divergence from the renderer this was ported from,
    which emitted the bare delimiters.
    """
    text = text.replace("$", "\\$")
    return _MATH_SPAN.sub(lambda m: f"{delimiter}{m.group(1)}{delimiter}" if m.group(1).strip() else "", text)


def _heading(text: str, level: int) -> str:
    """Add the hashes only when they are not already there.

    Both spellings occur.  The corpus writes ``## II. MODEL`` with its own
    hashes and ``Config.strip_markdown`` would delete them, so a heading may
    arrive either way; prefixing unconditionally gives ``## ## II. MODEL``.
    When the corpus's own hashes are present its *level* is kept too -- ``###``
    under ``##`` is real nesting, and flattening it to a fixed level would lose
    the outline.

    Whitespace is collapsed because a heading must be one line: 26 of them
    carry a newline from a title set over two lines on the page, and markdown
    would end the heading there and leave the rest as a stray paragraph.
    """
    text = " ".join(text.split())
    hashes = _LEADING_HASHES.match(text)
    if hashes:
        return f"{hashes.group(1)} {text[hashes.end() :]}"
    return f"{'#' * level} {text}"


def _title(box: Box) -> str:
    return _heading(_math(box.text, "$"), TITLE_LEVEL)


def _section_header(box: Box) -> str:
    return _heading(_math(box.text, "$"), SECTION_LEVEL)


def _list_item(box: Box) -> str:
    """One markdown bullet.

    Markers arrive in six shapes and are missing outright on ~18% of items, so
    the label is never parsed or renumbered -- only enough is done to make the
    line a list.  An unrecognised label such as ``(a)`` keeps its label and gets
    a bullet in front of it, which is lossless.
    """
    text = box.text.strip()
    if _MD_MARKER.match(text):
        return _math(text, "$")
    text = _LEADING_BULLET.sub("", text)
    if _MATH_BULLET_ALONE.match(text):
        text = _MATH_BULLET_ALONE.sub("", text)
    else:
        text = _MATH_BULLET_LEADING.sub(r"\\(", text)
    return f"- {_math(text, '$')}"


def _formula(box: Box) -> str:
    """A ``Formula`` element is a block of its own -- see the module docstring."""
    return _math(box.text.strip(), "$$")


def _table(box: Box) -> str:
    """Verbatim LaTeX: no math re-delimiting, no pipe conversion."""
    return box.text.strip()


def _code(box: Box) -> str:
    r"""Fenced, with no language.

    The corpus's ``Code`` is pseudocode as often as it is a real language, and
    a wrong tag is worse than none.  Nothing inside is rewritten -- within a
    fence ``$`` and ``\(`` are already literal.
    """
    text = box.text.strip()
    longest = max((len(run) for run in _BACKTICK_RUN.findall(text)), default=0)
    fence = "`" * max(_MIN_FENCE, longest + 1)
    return f"{fence}\n{text}\n{fence}"


def _paragraph(box: Box) -> str:
    """Prose.

    A leading ``#`` is escaped -- it is not a heading, or the element would have
    been classified as one.  In practice these are code comments caught inside
    a figure (``# Calculate the projection matrix``), and left bare they would
    open a section that runs to the end of the document.
    """
    text = box.text.strip()
    if text.startswith("#"):
        text = f"\\{text}"
    return _math(text, "$")


RENDERERS: dict[str, Callable[[Box], str]] = {
    "Title": _title,
    "Section-header": _section_header,
    "List-item": _list_item,
    "Formula": _formula,
    "Table": _table,
    "Code": _code,
}


def render(box: Box) -> str:
    """One box as markdown.  Empty string means it contributes nothing.

    A ``Picture`` reaches here with empty text -- the image itself lives in a
    binary column this module never sees -- so it renders to nothing.  In
    interleaved output that is the right answer: the picture is carried by its
    own image row, not by markdown.
    """
    if not box.text.strip():
        return ""
    return RENDERERS.get(box.element_class, _paragraph)(box)


def to_markdown(boxes: tuple[Box, ...]) -> str:
    """The kept boxes as one markdown document, blank line between blocks.

    A box the pipeline rejected is no more present here than it is in
    ``ProcessedDocument.text``.  Read ``ProcessedDocument.boxes`` for the full
    stream including what was dropped.
    """
    blocks = (render(box) for box in boxes if box.keep)
    return "\n\n".join(block for block in blocks if block)
