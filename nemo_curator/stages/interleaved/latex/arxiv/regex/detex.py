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

r"""Convert LaTeX body text to readable prose.

:func:`detex` turns a LaTeX fragment into plain text suitable for language-model
training: markup is removed, text-bearing commands are unwrapped, accents and
escapes become real characters, and non-prose environments (equations, tables,
code) are dropped.

Math handling is a deliberate choice rather than a default, because deleting it
silently mangles sentences -- "we find that $M_\odot > 10$ holds" becomes "we
find that holds".  The ``math`` argument selects the behaviour:

``"strip"``
    Remove math entirely.  Cleanest prose, some sentences lose their object.
``"placeholder"``
    Replace each math span with a marker (default ``[MATH]``), keeping sentence
    structure intact and making the omission visible to a model or a filter.
``"keep"``
    Leave the math source in place, delimiters and all.

This module is intentionally heuristic.  A complete TeX implementation would
need a real macro expander; the goal here is to be right on the constructs that
actually dominate arXiv prose and to fail toward dropping markup rather than
emitting it.
"""

from __future__ import annotations

import re
from typing import Literal

MathMode = Literal["strip", "placeholder", "keep"]

DEFAULT_MATH_PLACEHOLDER = "[MATH]"

#: Environments removed wholesale -- they carry no running prose.
_DROP_ENVIRONMENTS: frozenset[str] = frozenset(
    {
        "align", "align*", "alignat", "alignat*", "aligned", "array", "cases", "displaymath",
        "eqnarray", "eqnarray*", "equation", "equation*", "flalign", "flalign*", "gather",
        "gather*", "gathered", "math", "multline", "multline*", "split", "subequations",
        "bmatrix", "pmatrix", "vmatrix", "Vmatrix", "matrix", "smallmatrix",
        "tabular", "tabular*", "tabularx", "longtable", "supertabular", "tabbing", "array*",
        "verbatim", "verbatim*", "Verbatim", "lstlisting", "minted", "alltt", "semiverbatim",
        "picture", "tikzpicture", "pspicture", "pgfpicture", "xy",
        "thebibliography", "references", "REFERENCES",
        "comment", "keywords", "PACS",
    }
)  # fmt: skip

#: Commands dropped together with their mandatory argument.
_DROP_WITH_ARG: frozenset[str] = frozenset(
    {
        "label", "ref", "eqref", "pageref", "autoref", "cref", "Cref", "nameref", "vref",
        "cite", "citep", "citet", "citealt", "citealp", "citeauthor", "citeyear", "citenum",
        "nocite", "bibliography", "bibliographystyle", "bibitem",
        "includegraphics", "includepdf", "epsfbox", "epsffile", "plotone", "plottwo",
        "plotfiddle", "psfig", "epsfig", "special", "input", "include", "usepackage",
        "documentclass", "documentstyle", "pagestyle", "thispagestyle", "setcounter",
        "addtocounter", "setlength", "addtolength", "vspace", "hspace", "rule", "index",
        "graphicspath", "markboth", "markright", "hyphenation",
        "color", "textcolor", "definecolor", "colorbox", "pagenumbering", "newcounter",
    }
)  # fmt: skip

#: Commands replaced by their (unwrapped) argument -- these wrap running text.
_UNWRAP: frozenset[str] = frozenset(
    {
        "text", "textbf", "textit", "textrm", "textsf", "texttt", "textsc", "textsl",
        "textnormal", "textup", "textmd", "emph", "underline", "uline", "mbox", "hbox",
        "makebox", "framebox", "fbox", "centerline",
        "title", "author", "abstract", "caption", "subcaption", "item", "url", "path",
        "so", "st", "lowercase", "uppercase", "MakeLowercase", "MakeUppercase",
    }
)  # fmt: skip

#: Commands whose argument becomes a parenthetical aside rather than inline text.
_PARENTHESIZE: frozenset[str] = frozenset({"footnote", "footnotetext", "thanks"})

#: Sectioning commands: the argument survives as its own paragraph.
_SECTIONING: frozenset[str] = frozenset(
    {
        "part", "chapter", "section", "subsection", "subsubsection",
        "paragraph", "subparagraph", "section*", "subsection*", "subsubsection*",
    }
)  # fmt: skip

#: Declarations with no argument at all; simply deleted.
_DROP_BARE: frozenset[str] = frozenset(
    {
        "noindent", "indent", "newpage", "clearpage", "cleardoublepage", "pagebreak",
        "linebreak", "newline", "nonumber", "notag", "bigskip", "medskip", "smallskip",
        "centering", "raggedright", "raggedleft", "maketitle", "tableofcontents",
        "listoffigures", "listoftables", "appendix", "hline", "endhead", "endfoot",
        "footnotesize", "scriptsize", "tiny", "small", "normalsize", "large", "Large",
        "LARGE", "huge", "Huge", "rm", "sf", "tt", "bf", "it", "sl", "sc", "em",
        "normalfont", "bfseries", "itshape", "rmfamily", "sffamily", "ttfamily", "upshape",
        "protect", "relax", "par", "leavevmode", "unskip", "ignorespaces", "hfill", "vfill",
        "hrulefill", "dotfill", "quad", "qquad", "smallskipamount", "columnbreak",
    }
)  # fmt: skip

#: ``\'e`` -> ``é`` and friends.  Maps accent command to a combining codepoint.
_ACCENTS: dict[str, str] = {
    "`": "\u0300", "'": "\u0301", "^": "\u0302", '"': "\u0308", "~": "\u0303",
    "=": "\u0304", ".": "\u0307", "u": "\u0306", "v": "\u030c", "H": "\u030b",
    "c": "\u0327", "k": "\u0328", "b": "\u0331", "d": "\u0323", "r": "\u030a",
}  # fmt: skip

#: Standalone symbol commands.
_SYMBOLS: dict[str, str] = {
    "ldots": "...", "dots": "...", "cdots": "...", "textellipsis": "...",
    "ss": "ß", "ae": "æ", "AE": "Æ", "oe": "œ", "OE": "Œ", "aa": "å", "AA": "Å",
    "o": "ø", "O": "Ø", "l": "ł", "L": "Ł", "i": "i", "j": "j",
    "textdegree": "°", "degree": "°", "copyright": "©", "pounds": "£",
    "textbackslash": "\\", "textasciitilde": "~", "textunderscore": "_",
    "LaTeX": "LaTeX", "TeX": "TeX", "BibTeX": "BibTeX",
    "textquotedblleft": "\u201c", "textquotedblright": "\u201d",
    "textquoteleft": "\u2018", "textquoteright": "\u2019",
    "textendash": "\u2013", "textemdash": "\u2014",
}  # fmt: skip

#: Escaped punctuation: ``\&`` -> ``&``.
_ESCAPES: frozenset[str] = frozenset({"&", "%", "$", "#", "_", "{", "}"})

_ENV_BEGIN_RE = re.compile(r"\\begin\s*\{([^{}]*)\}")
_COMMAND_RE = re.compile(r"\\([a-zA-Z@]+\*?|[^a-zA-Z\s])")
_WHITESPACE_RUN_RE = re.compile(r"[ \t]+")
_BLANK_RUN_RE = re.compile(r"\n\s*\n(\s*\n)+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?)\]])")
_SPACE_AFTER_OPEN_RE = re.compile(r"([(\[])\s+")
_ORPHAN_PUNCT_RE = re.compile(r"(?m)^[\s,;:.]+$")
_REPEATED_PUNCT_RE = re.compile(r"([,;:])\s*(?=[,;:])")


def _read_group(text: str, start: int, opener: str = "{", closer: str = "}") -> tuple[str, int] | None:
    """Read a balanced group; returns ``(content, index_after_closer)``."""
    if start >= len(text) or text[start] != opener:
        return None
    depth, index = 0, start
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start + 1 : index], index + 1
        index += 1
    return None


def _skip_optional(text: str, pos: int) -> int:
    """Skip ``[...]`` optional arguments (and the whitespace before them)."""
    while pos < len(text):
        probe = pos
        while probe < len(text) and text[probe] in " \t":
            probe += 1
        if probe >= len(text) or text[probe] != "[":
            return pos
        group = _read_group(text, probe, "[", "]")
        if group is None:
            return pos
        pos = group[1]
    return pos


def _drop_environments(text: str) -> str:
    """Remove ``\\begin{env}...\\end{env}`` for every environment in ``_DROP_ENVIRONMENTS``."""
    out: list[str] = []
    cursor = 0
    while True:
        match = _ENV_BEGIN_RE.search(text, cursor)
        if match is None:
            break
        name = match.group(1).strip()
        if name not in _DROP_ENVIRONMENTS:
            out.append(text[cursor : match.end()])
            cursor = match.end()
            continue
        out.append(text[cursor : match.start()])
        out.append(" ")
        cursor = _find_environment_end(text, match.end(), name)
    out.append(text[cursor:])
    return "".join(out)


def _find_environment_end(text: str, start: int, name: str) -> int:
    """Index just past the matching ``\\end{name}``, honouring nesting."""
    begin_re = re.compile(r"\\begin\s*\{" + re.escape(name) + r"\}")
    end_re = re.compile(r"\\end\s*\{" + re.escape(name) + r"\}")
    depth, index = 1, start
    while index < len(text):
        opened = begin_re.search(text, index)
        closed = end_re.search(text, index)
        if closed is None:
            return len(text)
        if opened is not None and opened.start() < closed.start():
            depth += 1
            index = opened.end()
            continue
        depth -= 1
        if depth == 0:
            return closed.end()
        index = closed.end()
    return len(text)


def _replace_math(text: str, math: MathMode, placeholder: str) -> str:
    r"""Handle ``$...$``, ``$$...$$``, ``\(...\)`` and ``\[...\]``."""
    if math == "keep":
        return text
    replacement = placeholder if math == "placeholder" else ""
    out: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char == "\\" and index + 1 < length:
            nxt = text[index + 1]
            if nxt in "([":
                closer = "\\)" if nxt == "(" else "\\]"
                end = text.find(closer, index + 2)
                index = length if end == -1 else end + 2
                out.append(replacement)
                continue
            out.append(text[index : index + 2])
            index += 2
            continue
        if char == "$":
            double = text.startswith("$$", index)
            token = "$$" if double else "$"
            end = text.find(token, index + len(token))
            index = length if end == -1 else end + len(token)
            out.append(replacement)
            continue
        out.append(char)
        index += 1
    return "".join(out)


def _apply_accent(accent: str, argument: str) -> str:
    """Compose ``\\'e`` style accents into precomposed characters where possible."""
    import unicodedata

    letter = argument.strip().strip("{}") or " "
    combining = _ACCENTS.get(accent)
    if combining is None or not letter:
        return letter
    return unicodedata.normalize("NFC", letter[0] + combining) + letter[1:]


def _expand_commands(text: str) -> str:  # noqa: C901, PLR0912, PLR0915
    """Single left-to-right pass turning commands into text (or nothing)."""
    out: list[str] = []
    index = 0
    length = len(text)

    while index < length:
        char = text[index]
        if char != "\\":
            # Braces that survived command removal carry no meaning in prose.
            out.append("" if char in "{}" else char)
            index += 1
            continue

        match = _COMMAND_RE.match(text, index)
        if match is None:
            index += 1
            continue
        name = match.group(1)
        pos = match.end()

        if name in _ESCAPES:
            out.append(name)
            index = pos
            continue
        if name in _ACCENTS and name != "~":
            group = _read_group(text, pos)
            if group is not None:
                out.append(_apply_accent(name, group[0]))
                index = group[1]
            else:
                argument = text[pos : pos + 1]
                out.append(_apply_accent(name, argument))
                index = pos + 1
            continue
        if name in _SYMBOLS:
            out.append(_SYMBOLS[name])
            index = pos
            continue
        if name in _DROP_BARE:
            out.append(" ")
            index = pos
            continue
        if name in _SECTIONING:
            pos = _skip_optional(text, pos)
            group = _read_group(text, pos)
            if group is None:
                index = pos
                continue
            out.append("\n\n" + _expand_commands(group[0]).strip() + "\n\n")
            index = group[1]
            continue
        if name == "href":
            first = _read_group(text, _skip_optional(text, pos))
            second = _read_group(text, first[1]) if first is not None else None
            if second is not None:
                out.append(_expand_commands(second[0]))
                index = second[1]
                continue
        if name in _DROP_WITH_ARG:
            pos = _skip_optional(text, pos)
            group = _read_group(text, pos)
            index = group[1] if group is not None else pos
            out.append(" ")
            continue
        if name in _PARENTHESIZE:
            pos = _skip_optional(text, pos)
            group = _read_group(text, pos)
            if group is not None:
                aside = _expand_commands(group[0]).strip()
                out.append(f" ({aside})" if aside else "")
                index = group[1]
                continue
            out.append(" ")
            index = pos
            continue
        if name in _UNWRAP:
            pos = _skip_optional(text, pos)
            group = _read_group(text, pos)
            if group is None:
                out.append(" ")
                index = pos
                continue
            out.append(_expand_commands(group[0]))
            index = group[1]
            continue
        if name in {"begin", "end"}:
            group = _read_group(text, pos)
            index = group[1] if group is not None else pos
            out.append("\n")
            continue

        # Unknown command.  Keep any braced argument -- most unrecognized
        # commands in running prose wrap text (a custom \newcommand for a term,
        # say) -- but drop the command token itself.
        pos = _skip_optional(text, pos)
        group = _read_group(text, pos)
        if group is None:
            out.append(" ")
            index = pos
            continue
        out.append(_expand_commands(group[0]))
        index = group[1]

    return "".join(out)


def _normalize(text: str) -> str:
    """Tidy spacing left behind by markup removal."""
    text = text.replace("~", " ").replace("\\\\", "\n")
    text = text.replace("``", "\u201c").replace("''", "\u201d").replace("`", "\u2018")
    text = text.replace("---", "\u2014").replace("--", "\u2013")
    text = _WHITESPACE_RUN_RE.sub(" ", text)
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = _SPACE_AFTER_OPEN_RE.sub(r"\1", text)
    text = _REPEATED_PUNCT_RE.sub("", text)
    text = _ORPHAN_PUNCT_RE.sub("", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _BLANK_RUN_RE.sub("\n\n", text)
    return text.strip()


def detex(
    text: str,
    *,
    math: MathMode = "strip",
    math_placeholder: str = DEFAULT_MATH_PLACEHOLDER,
) -> str:
    """Convert a LaTeX fragment to plain prose.

    Args:
        text: LaTeX source, ideally already comment-stripped.
        math: How to treat math spans -- ``"strip"``, ``"placeholder"``, ``"keep"``.
        math_placeholder: Marker used when *math* is ``"placeholder"``.

    Returns:
        Plain text with markup removed and whitespace normalized.
    """
    text = _drop_environments(text)
    text = _replace_math(text, math, math_placeholder)
    text = _expand_commands(text)
    return _normalize(text)
