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

r"""Quality gates for LaTeXML conversions.

The gates exist because **exit codes do not detect the failures that matter.**
A misordered math flag produced ``rc=0`` and 25 KB of clean, well-formed HTML
with every equation deleted; nothing in the return code, the log severities, or
the byte count distinguished it from a good conversion.  The only signal was
that a formula-dense source produced zero ``<math>`` elements.

So the design rule here is: *compare the output against what the input implies
it should contain*, rather than trusting the converter to report its own
failure.  Each gate is cheap enough to run on every document.
"""

from __future__ import annotations

import re

from nemo_curator.stages.interleaved.latex.arxiv.latexml.artifacts import scan as scan_artifacts

from nemo_curator.stages.interleaved.latex.arxiv.latexml.model import Assessment, HtmlCounts, Status, Tier

#: Source markers implying the document contains mathematics.
_SOURCE_MATH_RE = re.compile(
    r"\$|\\begin\s*\{(?:equation|align|eqnarray|displaymath|gather|multline|math)\*?\}|\\\[|\\\(",
)
#: Source markers implying the document includes a figure.
_SOURCE_FIGURE_RE = re.compile(
    r"\\includegraphics|\\epsfbox|\\epsffile|\\psfig|\\epsfig|\\plotone|\\plottwo|\\plotfiddle",
)

#: A submission whose entire LaTeX body is ``\includepdf`` of a shipped PDF.
#: These are not conversion failures -- there is no LaTeX content to convert --
#: so they belong in the PDF fallback tier rather than counting against the
#: converter.  Measured in the wild: a 145-byte paper.tex wrapping DeepMask.pdf.
_PDF_WRAPPER_RE = re.compile(r"\\includepdf\b")
_PDF_WRAPPER_MAX_BODY_CHARS = 400

_HTML_MATH_RE = re.compile(r"<math[\s>]")
_HTML_ALTTEXT_RE = re.compile(r"alttext=")
_HTML_IMG_RE = re.compile(r"<img[\s>]")
_HTML_SECTION_RE = re.compile(r"<h[1-6][\s>]", re.IGNORECASE)
_REPLACEMENT_CHAR = "�"

#: LaTeXML reports its own timeout as ``Fatal:timeout``, which is indistinguishable
#: from a genuine conversion failure if fatals are only counted.  A timeout says
#: nothing about whether the document *could* convert -- only that it did not
#: within the budget -- so it must be a separate status or it pollutes the
#: quality signal and the retry policy.
TIMEOUT_ERROR_KINDS: frozenset[str] = frozenset({"timeout"})

#: Error kinds that indicate structural damage rather than a cosmetic gap.
_STRUCTURAL_ERROR_KINDS: frozenset[str] = frozenset({"misdefined", "too_many_errors", "die", "internal"})

MIN_TEXT_CHARS = 500
MAX_REPLACEMENT_RATIO = 0.001


def count_html(html: str) -> HtmlCounts:
    """Count structural elements without a full DOM parse (cheap enough for every doc)."""
    return HtmlCounts(
        n_math=len(_HTML_MATH_RE.findall(html)),
        n_alttext=len(_HTML_ALTTEXT_RE.findall(html)),
        n_img=len(_HTML_IMG_RE.findall(html)),
        n_section=len(_HTML_SECTION_RE.findall(html)),
        n_replacement_chars=html.count(_REPLACEMENT_CHAR),
        text_chars=len(re.sub(r"<[^>]+>", " ", html)),
    )


def is_pdf_wrapper(source_text: str) -> bool:
    r"""Whether the source is just an ``\includepdf`` shell around a PDF.

    Detected on the document body: the wrapper is a near-empty document whose
    only content is the include, so a paper that merely *also* attaches a PDF
    appendix is not misclassified.
    """
    if not _PDF_WRAPPER_RE.search(source_text):
        return False
    body = source_text
    begin = re.search(r"\\begin\s*\{document\}", source_text)
    end = re.search(r"\\end\s*\{document\}", source_text)
    if begin:
        body = source_text[begin.end() : end.start() if end else len(source_text)]
    stripped = re.sub(r"\\includepdf(?:\[[^\]]*\])?\s*\{[^}]*\}", "", body)
    return len(stripped.strip()) <= _PDF_WRAPPER_MAX_BODY_CHARS


def source_expects_math(source_text: str) -> bool:
    """Whether the LaTeX source demonstrably contains mathematics."""
    return bool(_SOURCE_MATH_RE.search(source_text))


def source_expects_figures(source_text: str) -> bool:
    """Whether the LaTeX source demonstrably includes at least one figure."""
    return bool(_SOURCE_FIGURE_RE.search(source_text))


def assess(  # noqa: PLR0911, PLR0912, PLR0913, C901 -- a flat ladder of independent gates reads better than nesting
    html: str | None,
    source_text: str,
    *,
    n_error: int = 0,
    n_fatal: int = 0,
    n_warning: int = 0,
    timed_out: bool = False,
    error_kinds: tuple[str, ...] = (),
    has_source: bool = True,
) -> Assessment:
    """Grade one conversion.

    Args:
        html: Converted HTML, or ``None`` when nothing was produced.
        source_text: The (comment-stripped) LaTeX of the root document, used to
            decide what the output *should* contain.
        n_error, n_fatal, n_warning: LaTeXML log severities.
        timed_out: Whether the converter exceeded its timeout.
        error_kinds: Distinct ``Error:<kind>`` tokens from the log.
        has_source: ``False`` for PDF-only submissions with no LaTeX at all.
    """
    empty = HtmlCounts()
    if not has_source:
        return Assessment(Status.NO_SOURCE, Tier.REJECTED, empty, ("no_source",))
    # Checked before the generic fatal branch: a timed-out document is a budget
    # outcome, retryable at a higher limit, not evidence the source is bad.
    if timed_out or TIMEOUT_ERROR_KINDS.intersection(error_kinds):
        return Assessment(
            Status.TIMEOUT,
            Tier.REJECTED,
            empty,
            ("timeout",),
            ["conversion exceeded its time budget; retryable at a higher limit, not a source defect"],
        )
    if n_fatal:
        return Assessment(Status.FATAL, Tier.REJECTED, empty, ("fatal",))
    if not html:
        return Assessment(Status.EMPTY_OUTPUT, Tier.REJECTED, empty, ("empty_output",))

    if is_pdf_wrapper(source_text):
        return Assessment(
            Status.PDF_WRAPPER,
            Tier.REJECTED,
            count_html(html),
            ("pdf_wrapper",),
            ["source is an \\includepdf wrapper; route to the PDF fallback, not the converter"],
        )

    counts = count_html(html)
    failed: list[str] = []
    notes: list[str] = []

    # The gate that catches silent content loss.  A formula-dense source that
    # yields zero <math> means the math processor was misconfigured, not that
    # the paper had no equations.
    if source_expects_math(source_text) and counts.n_math == 0:
        failed.append("no_math")
        notes.append("source contains math markup but the HTML has no <math> elements")
    elif counts.n_math and counts.n_alttext == 0:
        failed.append("no_alttext")
        notes.append("<math> present but no alttext: the original TeX was not retained")

    if source_expects_figures(source_text) and counts.n_img == 0:
        failed.append("no_figures")
        notes.append("source includes graphics but the HTML has no <img> elements")

    if counts.text_chars < MIN_TEXT_CHARS:
        failed.append("too_short")
        notes.append(f"only {counts.text_chars} chars of text")

    # Residual LaTeX is invisible to the log: measured, 9% of affected documents
    # had zero errors and zero fatals.  Only reading the output finds it.
    artifacts = scan_artifacts(html)
    if artifacts.notable:
        failed.append("residual_latex")
        top = ", ".join(f"{seq} x{n}" for seq, n in artifacts.top_sequences[:3])
        notes.append(f"{artifacts.total} raw control sequences leaked into text ({top})")

    if counts.text_chars and counts.n_replacement_chars / counts.text_chars > MAX_REPLACEMENT_RATIO:
        failed.append("encoding_damage")
        notes.append(f"{counts.n_replacement_chars} U+FFFD replacement characters")

    structural = sorted(_STRUCTURAL_ERROR_KINDS.intersection(error_kinds))
    if structural:
        failed.append("structural_error")
        notes.append(f"structural error kinds in log: {', '.join(structural)}")

    if "no_math" in failed:
        return Assessment(Status.SUSPECT_NO_MATH, Tier.REJECTED, counts, tuple(failed), notes)
    if "too_short" in failed:
        return Assessment(Status.SUSPECT_TRUNCATED, Tier.REJECTED, counts, tuple(failed), notes)
    if "residual_latex" in failed:
        # Degraded, not lost: the prose is there, interspersed with markup that a
        # downstream cleaner can strip.  Worth surfacing, not worth discarding.
        return Assessment(Status.SUSPECT_ARTIFACTS, Tier.C, counts, tuple(failed), notes)
    if "no_figures" in failed:
        # Figures are recoverable downstream from the source tarball, so this is
        # degraded rather than fatal.
        return Assessment(Status.SUSPECT_NO_FIGURES, Tier.C, counts, tuple(failed), notes)
    if failed:
        return Assessment(Status.ERROR, Tier.C, counts, tuple(failed), notes)
    if n_error:
        return Assessment(Status.ERROR, Tier.C, counts, (), [f"{n_error} localized errors"])
    if n_warning:
        return Assessment(Status.WARNING, Tier.B, counts, (), [f"{n_warning} warnings"])
    return Assessment(Status.OK, Tier.A, counts, (), notes)
