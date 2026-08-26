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

r"""Detect LaTeX artifacts that survived into the converted HTML.

**The converter's own log does not report these.**  Two measured examples, both
with *zero* Warning/Error/Fatal records:

``astro-ph/0301029``
    Source has ``\altaffilmark{\ref{FNAL}}``.  LaTeXML bound ``\altaffilmark``
    but emitted the nested ``\ref{FNAL}`` literally -- 28 raw control sequences
    landed in the author affiliations.

``1506.06452``
    Source has ``\eads{\mailto{a} and \mailto{b}}``.  ``\eads`` is unbound, so
    the group collapsed: one address rendered in place and the literal word
    "and" was stranded above the title.

Both would be recorded as clean conversions by any check that trusts exit codes
or log severities.  So this module reads the *output* and looks for LaTeX that
should not be there -- residual control sequences, unresolved cross-references,
stray delimiters -- which is the only way to see this class of damage.

A count of zero here is meaningful; a nonzero count localizes the damage to a
specific construct and usually to a specific missing package binding.
"""

from __future__ import annotations

import re
from collections import Counter

from nemo_curator.stages.interleaved.latex.arxiv.latexml.model import ArtifactReport

_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b.*?</\1>", re.DOTALL | re.IGNORECASE)
#: ``<math>`` legitimately carries TeX in ``alttext``; strip it before scanning
#: so real math is never counted as an artifact.
_MATH_RE = re.compile(r"<math\b.*?</math>", re.DOTALL | re.IGNORECASE)

#: A control sequence surviving into rendered text, e.g. ``\ref{FNAL}``.
_CONTROL_SEQ_RE = re.compile(r"\\[a-zA-Z@]{2,}")
#: LaTeXML renders an unresolvable cross-reference as a bare question mark.
_UNRESOLVED_REF_RE = re.compile(r"(?<![\w?])\?(?![\w?])")

#: Constructs worth naming individually, because each maps to a fixable cause.
_NAMED_ARTIFACTS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("ref", re.compile(r"\\ref\b|\\eqref\b|\\pageref\b")),
    ("cite", re.compile(r"\\cite[a-z]*\b|\\bibitem\b")),
    ("label", re.compile(r"\\label\b")),
    ("mail", re.compile(r"\\mailto\b|\\eads?\b|\\email\b")),
    ("affil", re.compile(r"\\altaffilmark\b|\\affil[a-z]*\b|\\thanks\b|\\footnotemark\b")),
    ("graphics", re.compile(r"\\includegraphics\b|\\epsfbox\b|\\psfig\b|\\plotone\b")),
    ("format", re.compile(r"\\text[a-z]+\b|\\emph\b|\\bf\b|\\it\b|\\rm\b|\\sc\b")),
    ("space", re.compile(r"\\hspace\b|\\vspace\b|\\quad\b|\\hfill\b|\\newline\b|\\\\(?![a-zA-Z])")),
    ("env", re.compile(r"\\begin\b|\\end\b|\\item\b")),
    ("size", re.compile(r"\\small\b|\\large\b|\\Large\b|\\footnotesize\b|\\tiny\b")),
)


def rendered_text(html: str) -> str:
    """Visible text of the document, with math and scripts removed."""
    without = _SCRIPT_STYLE_RE.sub(" ", html)
    without = _MATH_RE.sub(" ", without)
    return _TAG_RE.sub(" ", without)


def scan(html: str, *, max_samples: int = 6, context: int = 60) -> ArtifactReport:
    """Find LaTeX that leaked into the rendered text of *html*."""
    text = rendered_text(html)
    sequences = _CONTROL_SEQ_RE.findall(text)
    if not sequences and "\\" not in text:
        return ArtifactReport(text_chars=len(text))

    by_kind: Counter[str] = Counter()
    for kind, pattern in _NAMED_ARTIFACTS:
        by_kind[kind] = len(pattern.findall(text))
    named = sum(by_kind.values())
    by_kind["other"] = max(0, len(sequences) - named)

    samples: list[str] = []
    for match in _CONTROL_SEQ_RE.finditer(text):
        if len(samples) >= max_samples:
            break
        start, end = max(0, match.start() - context), min(len(text), match.end() + context)
        samples.append(re.sub(r"\s+", " ", text[start:end]).strip())

    return ArtifactReport(
        text_chars=len(text),
        total=len(sequences),
        by_kind={k: v for k, v in by_kind.items() if v},
        top_sequences=Counter(sequences).most_common(8),
        unresolved_refs=len(_UNRESOLVED_REF_RE.findall(text)),
        samples=samples,
    )
