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

r"""Pure-function LaTeX parsing for interleaved extraction.

Nothing here touches Ray, tar archives, or Arrow.  A LaTeX *project* -- a
``{member_name: bytes}`` mapping, typically one arXiv submission -- is turned
into an ordered list of :class:`TextSegment` and :class:`Figure` elements that
follow the document's reading order.  Keeping the tricky parts I/O-free makes
them unit-testable without any pipeline machinery.

The parsing choices below are grounded in measurements over a real arXiv source
shard (``arXiv_src_0001_001.tar``, 400 submissions):

============================  ==========================================================
Observation                   Consequence for this module
============================  ==========================================================
``\psfig`` 827, ``\plotone``  ``\includegraphics`` alone finds only ~12% of figures in
430, ``\epsfig`` 352,         pre-2005 submissions.  All legacy macros are supported --
``\includegraphics`` 332,     see ``_GRAPHICS_MACROS``.
``\epsfbox`` 258, ...
58.9% of ``\caption`` bodies  A non-greedy ``\{(.*?)\}`` regex truncates the majority of
contain nested braces         captions, so arguments are read brace-balanced instead.
Stripping ``%`` comments      Template boilerplate is usually commented out.  Comments
before scanning raises        are removed first, and verbatim-like environments are
graphics resolution from      protected from that removal.
86.6% to 95.6%
26% use ``\input``/           Included files are inlined before segmentation, otherwise
``\include``; 5% ship more    a quarter of papers lose most of their body text.
than one ``.tex``
1.8% of ``.tex`` files are    Decoding falls back to latin-1.
not valid UTF-8
============================  ==========================================================
"""

from __future__ import annotations

import mimetypes
import posixpath
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Members treated as LaTeX source when locating the root document.
TEX_EXTENSIONS: tuple[str, ...] = (".tex", ".ltx", ".latex")

#: Extensions probed when a graphics reference omits one, ordered the way TeX
#: engines resolve them: pdflatex targets first, then the classic (E)PS set.
GRAPHICS_EXTENSIONS: tuple[str, ...] = (
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".mps",
    ".eps",
    ".ps",
    ".epsi",
    ".epsf",
    ".pstex",
    ".postscript",
    ".gif",
    ".tif",
    ".tiff",
    ".bmp",
    ".svg",
    ".webp",
)

#: A graphics reference never resolves to one of these, even by basename match.
NON_GRAPHICS_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".tex",
        ".ltx",
        ".latex",
        ".sty",
        ".cls",
        ".clo",
        ".bib",
        ".bbl",
        ".bst",
        ".aux",
        ".log",
        ".toc",
        ".out",
        ".dvi",
        ".txt",
        ".tfm",
        ".fd",
    }
)

#: Float environments whose contents are treated as one figure element.
FIGURE_ENVIRONMENTS: tuple[str, ...] = (
    "figure",
    "figure*",
    "wrapfigure",
    "SCfigure",
    "SCfigure*",
    "floatingfigure",
    "sidewaysfigure",
    "sidewaysfigure*",
)

#: Environments whose bodies are copied through untouched (no comment stripping,
#: no macro scanning) because their content is literal text.
VERBATIM_ENVIRONMENTS: tuple[str, ...] = (
    "verbatim",
    "verbatim*",
    "Verbatim",
    "lstlisting",
    "alltt",
    "minted",
    "comment",
    "semiverbatim",
)

#: MIME types for figure formats :mod:`mimetypes` does not know.
_EXTENSION_CONTENT_TYPES: dict[str, str] = {
    ".eps": "application/postscript",
    ".epsi": "application/postscript",
    ".epsf": "application/postscript",
    ".ps": "application/postscript",
    ".mps": "application/postscript",
    ".pstex": "application/postscript",
    ".postscript": "application/postscript",
    ".pdf": "application/pdf",
}

#: Magic-byte prefixes, checked before falling back to the file extension.
#: arXiv submissions routinely name figures ``fig.f1`` / ``fig.0``, so the
#: extension is often useless.
_MAGIC_CONTENT_TYPES: tuple[tuple[bytes, str], ...] = (
    (b"%PDF", "application/pdf"),
    (b"%!PS", "application/postscript"),
    (b"\xc5\xd0\xd3\xc6", "application/postscript"),  # DOS binary EPS header
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
    (b"BM", "image/bmp"),
)

DEFAULT_MAX_INPUT_DEPTH = 8

# ---------------------------------------------------------------------------
# Regular expressions
# ---------------------------------------------------------------------------


def _alternation(names: Iterable[str]) -> str:
    """Build a regex alternation with longer names first so ``figure*`` wins over ``figure``."""
    return "|".join(re.escape(name) for name in sorted(names, key=len, reverse=True))


# ``(?<!\\)((?:\\\\)*)`` keeps ``\%`` (escaped percent) but allows ``\\%``
# (escaped backslash followed by a real comment).
_COMMENT_RE = re.compile(r"(?<!\\)((?:\\\\)*)%[^\n]*")
_VERBATIM_BLOCK_RE = re.compile(
    r"\\begin\s*\{(" + _alternation(VERBATIM_ENVIRONMENTS) + r")\}.*?\\end\s*\{\1\}",
    re.DOTALL,
)
_FIGURE_START_RE = re.compile(r"\\begin\s*\{(" + _alternation(FIGURE_ENVIRONMENTS) + r")\}")

_DOCUMENT_CLASS_RE = re.compile(r"\\document(?:class|style)\b")
_BEGIN_DOCUMENT_RE = re.compile(r"\\begin\s*\{document\}")
_END_DOCUMENT_RE = re.compile(r"\\end\s*\{document\}")
_APPENDIX_RE = re.compile(r"\\appendix\b|\\begin\s*\{appendix\}")
_BIBLIOGRAPHY_RE = re.compile(
    r"\\begin\s*\{thebibliography\}|\\bibliography\s*\{|\\begin\s*\{references\}", re.IGNORECASE
)

_INPUT_RE = re.compile(r"\\(?:input|include|subfile)\b\s*(?:\{([^{}]*)\}|([^\s\\{}%]+))")
_GRAPHICSPATH_RE = re.compile(r"\\graphicspath\b\s*")

# Macro definitions without arguments; mirrors the patterns used by the
# text-only arXiv extractor in nemo_curator/stages/text/download/arxiv/extract.py.
_NEWCOMMAND_RE = re.compile(r"\\newcommand\b\*?\s*\{(\\[a-zA-Z@]+)\}\s*\{(.*?)\}$", re.MULTILINE)
_DEF_RE = re.compile(r"\\def\s*(\\[a-zA-Z@]+)\s*\{(.*?)\}$", re.MULTILINE)

#: Graphics-inclusion macros, by frequency in real arXiv source.
_GRAPHICS_MACROS: tuple[str, ...] = (
    "includegraphics",
    "psfig",
    "epsfig",
    "epsfbox",
    "epsffile",
    "plotone",
    "plottwo",
    "plotfiddle",
    "includepdf",
    "BoxedEPSF",
)
#: ``\psfig``/``\epsfig`` take a key=value list rather than a bare filename.
_KEYVAL_MACROS: frozenset[str] = frozenset({"psfig", "epsfig"})
#: Number of mandatory brace arguments that are filenames.
_MACRO_FILE_ARGS: dict[str, int] = {"plottwo": 2}

_GRAPHICS_MACRO_RE = re.compile(r"\\(" + _alternation(_GRAPHICS_MACROS) + r")\b\*?")
_SPECIAL_PSFILE_RE = re.compile(r"\\special\s*\{\s*psfile\s*=\s*([^,}\s]+)")
_KEYVAL_FILE_RE = re.compile(r"(?:^|,)\s*(?:file|figure)\s*=\s*([^,]+)", re.IGNORECASE)

# ``\figcaption`` is the AASTeX spelling and is common in astro-ph submissions.
_CAPTION_RE = re.compile(r"\\(?:fig)?caption(?:of\s*\{[^{}]*\})?\b\*?")
_LABEL_RE = re.compile(r"\\label\b\*?")

_CONTROL_SEQ_RE = re.compile(r"\\(?:[a-zA-Z@]+|.)")
_DEFINITION_RE = re.compile(
    r"\\(newcommand|renewcommand|providecommand|def|edef|gdef|xdef|newenvironment|renewenvironment"
    r"|DeclareMathOperator|declaretheorem|newtheorem|newlength|setlength|setcounter|newcounter"
    r"|definecolor|newfont|newsavebox)\b\*?"
)
#: Mandatory brace groups each definition macro consumes, counting the name.
_DEFINITION_BRACES: dict[str, int] = {"newenvironment": 3, "renewenvironment": 3}
_DEFAULT_DEFINITION_BRACES = 2

_WHITESPACE_RE = re.compile(r"\s+")
_LEADING_WS_RE = re.compile(r"\s*")
_TRAILING_SPACE_RE = re.compile(r"[ \t]+\n")
_BLANK_RUN_RE = re.compile(r"\n{3,}")
# Layout-only commands dropped from text segments; they carry no information.
# Argument-taking and declaration-style commands are matched separately: a
# declaration such as ``\small`` must NOT swallow the group that follows it,
# or ``{\small {\sc Fig.}~1.---...}`` loses the "Fig." label.
_NOISE_ARG_RE = re.compile(r"\\(?:label|index|vspace|hspace|addtolength)\b\*?\s*\{[^{}]*\}")
_NOISE_DECLARATION_RE = re.compile(
    r"\\(?:nonumber|noindent|newpage|clearpage|cleardoublepage|pagebreak|linebreak|newline"
    r"|bigskip|medskip|smallskip|centering|raggedright|raggedleft"
    r"|footnotesize|scriptsize|tiny|small|normalsize|large|Large|LARGE|huge|Huge)\b\*?"
)


# ---------------------------------------------------------------------------
# Element model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TextSegment:
    """A run of body text between two figures."""

    text: str


@dataclass(frozen=True)
class Figure:
    """A single graphics inclusion, resolved against the project's members."""

    reference: str
    """The graphics argument exactly as written in the source."""

    member: str | None
    """Project member holding the image bytes, or ``None`` if unresolved."""

    caption: str | None = None
    label: str | None = None
    environment: str | None = None
    """Float environment the figure came from, or ``None`` for a bare macro."""

    group_index: int = 0
    """Index of the float this figure belongs to.

    Subfigures share one float -- and therefore one caption -- so consumers can
    use this to avoid emitting the caption once per sub-panel.
    """


Element = TextSegment | Figure


@dataclass(frozen=True)
class _FigureSpec:
    """Internal: a figure before its references are resolved to members."""

    references: tuple[str, ...]
    caption: str | None = None
    label: str | None = None
    environment: str | None = None


@dataclass
class ParsedProject:
    """Result of parsing one LaTeX project."""

    elements: list[Element] = field(default_factory=list)
    root_tex: str | None = None
    graphics_paths: tuple[str, ...] = ()
    macros: dict[str, str] = field(default_factory=dict)
    unresolved_references: tuple[str, ...] = ()
    error: str | None = None

    @property
    def figures(self) -> list[Figure]:
        return [element for element in self.elements if isinstance(element, Figure)]

    @property
    def texts(self) -> list[TextSegment]:
        return [element for element in self.elements if isinstance(element, TextSegment)]


# ---------------------------------------------------------------------------
# Low-level scanning helpers
# ---------------------------------------------------------------------------


def decode_text(payload: bytes) -> str:
    """Decode LaTeX source, falling back to latin-1 for the ~2% that are not UTF-8."""
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        return payload.decode("latin-1", errors="replace")


def strip_comments(tex: str) -> str:
    r"""Remove ``%`` comments, preserving ``\%`` and the contents of verbatim environments."""
    if "%" not in tex:
        return tex
    out: list[str] = []
    cursor = 0
    for block in _VERBATIM_BLOCK_RE.finditer(tex):
        out.append(_COMMENT_RE.sub(r"\1", tex[cursor : block.start()]))
        out.append(block.group(0))
        cursor = block.end()
    out.append(_COMMENT_RE.sub(r"\1", tex[cursor:]))
    return "".join(out)


def _read_group(text: str, start: int, opener: str = "{", closer: str = "}") -> tuple[str, int] | None:
    """Read a balanced group starting at *start*.

    Returns ``(content_without_delimiters, index_after_closer)``, or ``None`` if
    *start* does not index *opener* or the group is unterminated.  Escaped
    characters (``\\{``) do not affect nesting.
    """
    if start >= len(text) or text[start] != opener:
        return None
    depth = 0
    index = start
    length = len(text)
    while index < length:
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


def _skip_optional_args(text: str, pos: int) -> int:
    """Skip whitespace plus any ``[...]`` and ``<...>`` argument groups."""
    length = len(text)
    while pos < length:
        pos = _LEADING_WS_RE.match(text, pos).end()
        if pos >= length:
            break
        if text[pos] == "[":
            group = _read_group(text, pos, "[", "]")
        elif text[pos] == "<":
            group = _read_group(text, pos, "<", ">")
        else:
            break
        if group is None:
            break
        pos = group[1]
    return pos


@lru_cache(maxsize=64)
def _environment_patterns(env: str) -> tuple[re.Pattern[str], re.Pattern[str]]:
    return (
        re.compile(r"\\begin\s*\{" + re.escape(env) + r"\}"),
        re.compile(r"\\end\s*\{" + re.escape(env) + r"\}"),
    )


def _read_environment(text: str, start: int, env: str) -> tuple[str, int]:
    """Read *env*'s body starting just after its ``\\begin``, honouring nesting.

    Returns ``(body, index_after_end)``.  An unterminated environment consumes
    the remainder of the text -- truncated arXiv sources are common enough that
    raising here would lose otherwise good papers.
    """
    begin_re, end_re = _environment_patterns(env)
    depth = 1
    index = start
    while index < len(text):
        opened = begin_re.search(text, index)
        closed = end_re.search(text, index)
        if closed is None:
            break
        if opened is not None and opened.start() < closed.start():
            depth += 1
            index = opened.end()
            continue
        depth -= 1
        if depth == 0:
            return text[start : closed.start()], closed.end()
        index = closed.end()
    return text[start:], len(text)


# ---------------------------------------------------------------------------
# Macros
# ---------------------------------------------------------------------------


def collect_simple_macros(tex: str) -> dict[str, str]:
    r"""Collect argument-less ``\newcommand`` / ``\def`` definitions."""
    macros: dict[str, str] = {}
    for pattern in (_NEWCOMMAND_RE, _DEF_RE):
        for match in pattern.finditer(tex):
            name, value = match.group(1), match.group(2)
            if "#" in value or name in value:
                continue  # takes arguments, or is recursive
            macros[name] = value
    return macros


def expand_macros(text: str, macros: Mapping[str, str]) -> str:
    """Inline-expand argument-less macros in *text* (single pass)."""
    for name, value in macros.items():
        text = re.sub(re.escape(name) + r"(?![a-zA-Z@])", lambda _match, v=value: v, text)
    return text


def strip_definitions(text: str) -> str:
    r"""Remove macro/length/counter definitions along with their arguments.

    arXiv authors routinely put ``\def`` and ``\newcommand`` blocks *after*
    ``\begin{document}``, so definitions cannot be dropped by preamble slicing
    alone.  Removing them before :func:`expand_macros` also stops a macro from
    being expanded inside its own definition.
    """
    parts: list[str] = []
    cursor = 0
    for match in _DEFINITION_RE.finditer(text):
        if match.start() < cursor:
            continue
        pos = _LEADING_WS_RE.match(text, match.end()).end()
        wanted = _DEFINITION_BRACES.get(match.group(1), _DEFAULT_DEFINITION_BRACES)
        if pos < len(text) and text[pos] == "\\":
            named = _CONTROL_SEQ_RE.match(text, pos)
            if named is not None:
                # The name was given as a control sequence, not a brace group.
                pos = named.end()
                wanted -= 1
        taken = 0
        while taken < wanted:
            pos = _LEADING_WS_RE.match(text, pos).end()
            if pos < len(text) and text[pos] == "[":
                optional = _read_group(text, pos, "[", "]")
                if optional is None:
                    break
                pos = optional[1]
                continue
            group = _read_group(text, pos)
            if group is None:
                break
            pos = group[1]
            taken += 1
        if taken == 0:
            continue  # not a shape we recognize -- leave the text untouched
        parts.append(text[cursor : match.start()])
        cursor = pos
    parts.append(text[cursor:])
    return "".join(parts)


def extract_graphics_paths(tex: str) -> tuple[str, ...]:
    r"""Extract the directories declared by ``\graphicspath{{a/}{b/}}``."""
    paths: list[str] = []
    for match in _GRAPHICSPATH_RE.finditer(tex):
        group = _read_group(tex, _LEADING_WS_RE.match(tex, match.end()).end())
        if group is None:
            continue
        for inner in re.finditer(r"\{([^{}]*)\}", group[0]):
            value = inner.group(1).strip()
            if value and value not in paths:
                paths.append(value)
    return tuple(paths)


# ---------------------------------------------------------------------------
# Root document selection and \input expansion
# ---------------------------------------------------------------------------


def _iter_input_refs(tex: str) -> Iterable[str]:
    for match in _INPUT_RE.finditer(tex):
        ref = match.group(1) if match.group(1) is not None else match.group(2)
        if ref:
            yield ref


def _path_variants(ref: str) -> list[str]:
    """Normalized spellings of a file reference, most literal first."""
    cleaned = ref.strip().strip('"').strip().rstrip(",").replace("{", "").replace("}", "")
    variants: list[str] = []
    for value in (cleaned, cleaned.removeprefix("./"), posixpath.normpath(cleaned) if cleaned else ""):
        if value and value not in {".", ".."} and value not in variants:
            variants.append(value)
    return variants


def _resolve_tex_ref(ref: str, sources: Mapping[str, str]) -> str | None:
    if "\\" in ref or "#" in ref:
        return None
    lookup = {name.lower(): name for name in sources}
    for variant in _path_variants(ref):
        for suffix in ("", *TEX_EXTENSIONS):
            probe = f"{variant}{suffix}"
            if probe in sources:
                return probe
            hit = lookup.get(probe.lower())
            if hit is not None:
                return hit
    return None


def select_root_tex(sources: Mapping[str, str]) -> str | None:
    r"""Pick the root document of a multi-file project.

    Prefers files declaring ``\documentclass``/``\documentstyle``, then files
    containing ``\begin{document}``.  Remaining ties are broken by dropping
    candidates that another file ``\input``s, then by size (largest wins) and
    finally by name so the choice is deterministic.
    """
    if not sources:
        return None
    candidates = [name for name, text in sources.items() if _DOCUMENT_CLASS_RE.search(text)]
    if not candidates:
        candidates = [name for name, text in sources.items() if _BEGIN_DOCUMENT_RE.search(text)]
    if not candidates:
        candidates = list(sources)
    if len(candidates) == 1:
        return candidates[0]

    included: set[str] = set()
    for name, text in sources.items():
        for ref in _iter_input_refs(text):
            target = _resolve_tex_ref(ref, sources)
            if target is not None and target != name:
                included.add(target)
    roots = [name for name in candidates if name not in included] or candidates
    return min(roots, key=lambda name: (-len(sources[name]), name))


def expand_inputs(root: str, sources: Mapping[str, str], max_depth: int = DEFAULT_MAX_INPUT_DEPTH) -> str:
    r"""Inline ``\input`` / ``\include`` / ``\subfile`` starting from *root*.

    Cycles are broken by tracking the ancestor chain; unresolvable references
    are left in place (harmless -- they are dropped by text cleaning).
    """

    def _expand(name: str, depth: int, ancestors: frozenset[str]) -> str:
        text = sources.get(name, "")
        if depth >= max_depth:
            return text
        parts: list[str] = []
        cursor = 0
        for match in _INPUT_RE.finditer(text):
            ref = match.group(1) if match.group(1) is not None else match.group(2)
            target = _resolve_tex_ref(ref, sources) if ref else None
            if target is None or target in ancestors or target == name:
                continue
            parts.append(text[cursor : match.start()])
            parts.append("\n")
            parts.append(_expand(target, depth + 1, ancestors | {name, target}))
            parts.append("\n")
            cursor = match.end()
        parts.append(text[cursor:])
        return "".join(parts)

    return _expand(root, 0, frozenset())


def document_body(tex: str, *, drop_bibliography: bool = True, drop_appendix: bool = False) -> str:
    r"""Slice out the ``\begin{document}``..``\end{document}`` body.

    Documents without a ``\begin{document}`` (fragments pulled in by a driver
    file, or single-file submissions missing a preamble) are returned whole.
    """
    begin = _BEGIN_DOCUMENT_RE.search(tex)
    body = tex[begin.end() :] if begin is not None else tex
    end = _END_DOCUMENT_RE.search(body)
    if end is not None:
        body = body[: end.start()]
    if drop_appendix:
        cut = _APPENDIX_RE.search(body)
        if cut is not None:
            body = body[: cut.start()]
    if drop_bibliography:
        cut = _BIBLIOGRAPHY_RE.search(body)
        if cut is not None:
            body = body[: cut.start()]
    return body


# ---------------------------------------------------------------------------
# Graphics extraction
# ---------------------------------------------------------------------------


def _graphics_refs_in(chunk: str) -> list[tuple[int, int, str]]:
    """Find graphics references in *chunk* as ``(start, end, reference)`` in source order."""
    found: list[tuple[int, int, str]] = []
    for match in _GRAPHICS_MACRO_RE.finditer(chunk):
        macro = match.group(1)
        pos = _skip_optional_args(chunk, match.end())
        group = _read_group(chunk, pos)
        if group is None:
            continue
        value, after = group
        if macro in _KEYVAL_MACROS:
            key = _KEYVAL_FILE_RE.search(value)
            if key is not None:
                found.append((match.start(), after, key.group(1).strip()))
            continue
        references = [value]
        for _ in range(_MACRO_FILE_ARGS.get(macro, 1) - 1):
            extra = _read_group(chunk, _skip_optional_args(chunk, after))
            if extra is None:
                break
            references.append(extra[0])
            after = extra[1]
        found.extend((match.start(), after, ref.strip()) for ref in references if ref.strip())
    found.extend((m.start(), m.end(), m.group(1).strip()) for m in _SPECIAL_PSFILE_RE.finditer(chunk))
    found.sort(key=lambda item: item[0])
    return found


def _extract_braced(chunk: str, pattern: re.Pattern[str]) -> str | None:
    match = pattern.search(chunk)
    if match is None:
        return None
    group = _read_group(chunk, _skip_optional_args(chunk, match.end()))
    return group[0] if group is not None else None


def _figure_from_environment(body: str, env: str) -> _FigureSpec:
    references = tuple(dict.fromkeys(ref for _, _, ref in _graphics_refs_in(body)))
    caption = _extract_braced(body, _CAPTION_RE)
    return _FigureSpec(
        references=references,
        caption=normalize_whitespace(_strip_noise(caption)) if caption else None,
        label=_extract_braced(body, _LABEL_RE),
        environment=env,
    )


def segment_document(body: str) -> list[TextSegment | _FigureSpec]:
    """Split a document body into text runs and figures, in reading order."""
    floats: list[tuple[int, int, _FigureSpec]] = []
    pos = 0
    while True:
        match = _FIGURE_START_RE.search(body, pos)
        if match is None:
            break
        env = match.group(1)
        content, after = _read_environment(body, match.end(), env)
        floats.append((match.start(), after, _figure_from_environment(content, env)))
        pos = after

    # Graphics macros used outside any float environment still mark an image
    # position, so scan the gaps between floats for them.
    events: list[tuple[int, int, _FigureSpec]] = []
    cursor = 0
    for start, end, spec in [*floats, (len(body), len(body), None)]:
        for ref_start, ref_end, ref in _graphics_refs_in(body[cursor:start]):
            events.append((cursor + ref_start, cursor + ref_end, _FigureSpec(references=(ref,))))
        if spec is not None:
            events.append((start, end, spec))
        cursor = end
    events.sort(key=lambda item: item[0])

    segments: list[TextSegment | _FigureSpec] = []
    cursor = 0
    for start, end, spec in events:
        if start > cursor and body[cursor:start].strip():
            segments.append(TextSegment(body[cursor:start]))
        segments.append(spec)
        cursor = end
    if body[cursor:].strip():
        segments.append(TextSegment(body[cursor:]))
    return segments


# ---------------------------------------------------------------------------
# Member lookup
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProjectIndex:
    """Case- and extension-tolerant lookup over a project's member names."""

    names: tuple[str, ...]
    _exact: frozenset[str]
    _lower: dict[str, str]
    _basename: dict[str, str]

    @classmethod
    def build(cls, names: Iterable[str]) -> ProjectIndex:
        ordered = tuple(names)
        lower: dict[str, str] = {}
        basename: dict[str, str] = {}
        for name in ordered:
            lower.setdefault(name.lower(), name)
            basename.setdefault(posixpath.basename(name).lower(), name)
        return cls(names=ordered, _exact=frozenset(ordered), _lower=lower, _basename=basename)

    def resolve(self, ref: str, graphics_paths: Sequence[str] = ()) -> str | None:
        """Resolve a graphics reference to a member name, or ``None``.

        Tried in order: verbatim match, match after appending a known graphics
        extension, case-insensitive match, then basename match.  Each is also
        retried under every ``\\graphicspath`` prefix.
        """
        if "#" in ref:
            return None
        for variant in _path_variants(ref):
            for prefix in ("", *graphics_paths):
                candidate = posixpath.join(prefix, variant) if prefix else variant
                hit = self._probe(candidate)
                if hit is not None:
                    return hit
        return None

    def _probe(self, candidate: str) -> str | None:
        # A verbatim hit is authoritative: the author named this file, even if its
        # extension is something odd like ".f1" or ".0" (common on arXiv).  Only
        # after that do we probe extensions, then fall back to case-insensitive
        # and basename matching -- both of which must land on a graphics file.
        if candidate in self._exact and _is_graphics_member(candidate):
            return candidate
        for suffix in GRAPHICS_EXTENSIONS:
            if f"{candidate}{suffix}" in self._exact:
                return f"{candidate}{suffix}"
        lowered = candidate.lower()
        for lookup, key in ((self._lower, lowered), (self._basename, posixpath.basename(lowered))):
            hit = lookup.get(key)
            if hit is not None and _is_graphics_member(hit):
                return hit
            for suffix in GRAPHICS_EXTENSIONS:
                hit = lookup.get(f"{key}{suffix}")
                if hit is not None:
                    return hit
        return None


def _is_graphics_member(name: str) -> bool:
    return posixpath.splitext(name)[1].lower() not in NON_GRAPHICS_EXTENSIONS


def guess_content_type(member: str, payload: bytes | None = None) -> str:
    """Best-effort MIME type, preferring magic bytes over the file extension."""
    if payload:
        head = payload[:16]
        for magic, content_type in _MAGIC_CONTENT_TYPES:
            if head.startswith(magic):
                return content_type
        if head.startswith(b"RIFF") and payload[8:12] == b"WEBP":
            return "image/webp"
    extension = posixpath.splitext(member)[1].lower()
    if extension in _EXTENSION_CONTENT_TYPES:
        return _EXTENSION_CONTENT_TYPES[extension]
    guessed, _ = mimetypes.guess_type(member)
    return guessed or "application/octet-stream"


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------


def _strip_noise(text: str) -> str:
    """Drop layout-only commands that carry no information."""
    return _NOISE_DECLARATION_RE.sub("", _NOISE_ARG_RE.sub("", text))


def normalize_whitespace(text: str) -> str:
    """Collapse every whitespace run to a single space."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def clean_text(text: str, macros: Mapping[str, str] | None = None) -> str:
    """Tidy a body-text segment without detexing it.

    Drops in-body macro definitions, expands argument-less macros, removes
    layout-only commands, and normalizes blank runs.  Markup that carries
    meaning (math, citations, sectioning) is deliberately preserved -- consumers
    of interleaved arXiv data generally want the LaTeX, and irreversible
    detexing is better done downstream.
    """
    text = strip_definitions(text)
    if macros:
        text = expand_macros(text, macros)
    text = _strip_noise(text)
    text = _TRAILING_SPACE_RE.sub("\n", text)
    return _BLANK_RUN_RE.sub("\n\n", text).strip()


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def parse_project(  # noqa: PLR0913
    members: Mapping[str, bytes],
    *,
    drop_bibliography: bool = True,
    drop_appendix: bool = False,
    clean: bool = True,
    max_input_depth: int = DEFAULT_MAX_INPUT_DEPTH,
    min_text_chars: int = 1,
) -> ParsedProject:
    """Parse one LaTeX project into interleaved text and figure elements.

    Args:
        members: The project's files, ``{member_name: payload}``.
        drop_bibliography: Truncate the body at the bibliography.
        drop_appendix: Truncate the body at ``\\appendix``.  Off by default --
            appendices frequently carry figures worth keeping.
        clean: Apply :func:`clean_text` to text segments.
        max_input_depth: Recursion limit for ``\\input`` expansion.
        min_text_chars: Drop text segments shorter than this after cleaning.

    Returns:
        A :class:`ParsedProject`.  ``error`` is set (and ``elements`` empty)
        when no LaTeX source could be found; every other failure mode degrades
        to fewer elements rather than an exception.
    """
    sources = {
        name: strip_comments(decode_text(payload))
        for name, payload in members.items()
        if name.lower().endswith(TEX_EXTENSIONS)
    }
    if not sources:
        return ParsedProject(error="no LaTeX source found")

    root = select_root_tex(sources)
    if root is None:
        return ParsedProject(error="no root document found")

    merged = expand_inputs(root, sources, max_depth=max_input_depth)
    macros = collect_simple_macros(merged)
    graphics_paths = extract_graphics_paths(merged)
    body = document_body(merged, drop_bibliography=drop_bibliography, drop_appendix=drop_appendix)

    index = ProjectIndex.build(members)
    elements: list[Element] = []
    unresolved: list[str] = []
    group_index = 0
    for segment in segment_document(body):
        if isinstance(segment, TextSegment):
            text = clean_text(segment.text, macros) if clean else segment.text
            if len(text) >= min_text_chars and text.strip():
                elements.append(TextSegment(text))
            continue
        for reference in segment.references:
            member = index.resolve(reference, graphics_paths)
            if member is None and "\\" in reference:
                # e.g. \includegraphics{\figdir/plot.pdf}
                member = index.resolve(expand_macros(reference, macros), graphics_paths)
            if member is None:
                unresolved.append(reference)
            elements.append(
                Figure(
                    reference=reference,
                    member=member,
                    caption=segment.caption,
                    label=segment.label,
                    environment=segment.environment,
                    group_index=group_index,
                )
            )
        group_index += 1

    return ParsedProject(
        elements=elements,
        root_tex=root,
        graphics_paths=graphics_paths,
        macros=macros,
        unresolved_references=tuple(unresolved),
    )
