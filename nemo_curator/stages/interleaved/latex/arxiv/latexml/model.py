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

"""The shapes a LaTeXML conversion deals in.

Types only -- no I/O, no subprocess, no LaTeX knowledge.  The modules that do
the work import from here, so the vocabulary of the stage can be read in one
file and a caller can type against it without importing a converter.

The flow the types describe:

    a tar member  --extract-->  ExtractedProject   (what to convert)
                  --convert-->  ConversionResult   (what LaTeXML did)
                  --assess -->  Assessment         (whether it is usable)
                  --scan   -->  ArtifactReport     (what leaked through)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

class UnsafeMemberError(Exception):
    """A tar member was rejected as unsafe to extract."""


@dataclass
class ExtractedProject:
    """One unpacked submission."""

    directory: Path
    root_tex: str | None = None
    """Project-relative path of the document to convert."""

    other_roots: tuple[str, ...] = ()
    """Additional documents that also look like roots -- separate candidates."""

    members: tuple[str, ...] = ()
    total_bytes: int = 0
    kind: str = "tar"
    """``tar``, ``single_file``, ``pdf_only`` or ``empty``."""

    warnings: list[str] = field(default_factory=list)

    @property
    def root_path(self) -> Path | None:
        return self.directory / self.root_tex if self.root_tex else None


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConversionResult:
    """Outcome of one LaTeXML invocation."""

    html: str | None
    argv: tuple[str, ...]
    """The exact ordered argv as executed -- goes verbatim into provenance."""

    returncode: int
    duration_s: float
    n_warning: int = 0
    n_error: int = 0
    n_fatal: int = 0
    log: str = ""
    timed_out: bool = False
    assets: tuple[str, ...] = ()
    """Files LaTeXML emitted alongside the HTML (rasterized figures)."""

    error_kinds: tuple[str, ...] = field(default_factory=tuple)
    """Distinct ``Error:<kind>`` tokens, for triage without re-reading logs."""


# ---------------------------------------------------------------------------
# Quality assessment
# ---------------------------------------------------------------------------

class Tier(StrEnum):
    """Usability tier for a converted document."""

    A = "A"
    """No fatals, no errors, every content gate passed."""

    B = "B"
    """Warnings only; content gates passed."""

    C = "C"
    """Localized errors; usable text but constructs may be missing."""

    REJECTED = "rejected"
    """Fatal, timeout, empty, or a failed content gate."""


class Status(StrEnum):
    """What happened to this document.

    ``SUSPECT_NO_MATH`` and ``SUSPECT_NO_FIGURES`` are the values the original
    status vocabulary lacked: the converter reported success, the HTML looks
    well-formed, and content the source demonstrably contained is simply absent.
    """

    OK = "ok"
    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"
    TIMEOUT = "timeout"
    NO_SOURCE = "no_source"
    SKIPPED = "skipped"
    EMPTY_OUTPUT = "empty_output"
    SUSPECT_NO_MATH = "suspect_no_math"
    SUSPECT_NO_FIGURES = "suspect_no_figures"
    SUSPECT_TRUNCATED = "suspect_truncated"
    SUSPECT_OVERSIZED = "suspect_oversized"
    """Output so large it can only be a converter pathology, not a paper.

    The mirror of ``SUSPECT_TRUNCATED``: that gate catches a document with too
    little text, this one catches a runaway.  Measured over 27,003 corpus
    documents the distribution is p50 0.23 MB, p99.9 4.94 MB, p99.99 9.96 MB --
    and one paper in a single test shard produced **1,444 MB**, 86% of its
    shard's entire output and ~5,700x the median, from a 72-section source.

    This is a *memory* gate as much as a quality one, which is why it lives
    beside the converter rather than in the flushing logic.  Row-count and
    byte-accumulation bounds cannot help: they are checked after a row is
    appended, so a single row larger than the whole threshold defeats them, and
    the observed peak was ~7x the document once the Python string, the Arrow
    copy and the Parquet write buffers coexisted.
    """
    SUSPECT_ARTIFACTS = "suspect_artifacts"
    """Raw LaTeX leaked into the rendered text -- usually a missing binding."""
    PDF_WRAPPER = "pdf_wrapper"
    """LaTeX source is only an ``\\includepdf`` wrapper around a shipped PDF."""


@dataclass(frozen=True)
class HtmlCounts:
    """Structural counts, used both for gating and for the stage-2 cross-check."""

    n_math: int = 0
    n_alttext: int = 0
    n_img: int = 0
    n_section: int = 0
    n_replacement_chars: int = 0
    text_chars: int = 0


@dataclass
class Assessment:
    """Gate outcome for one document."""

    status: Status
    tier: Tier
    counts: HtmlCounts
    failed_gates: tuple[str, ...] = ()
    notes: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return self.tier is not Tier.REJECTED

    def tier_c_reason(self, n_error: int) -> str | None:
        """Why this document landed in C: errors, artifacts, or both.

        ``status`` cannot answer this on its own -- a document with *both*
        localized errors and residual LaTeX reports ``suspect_artifacts``, which
        is indistinguishable from artifacts alone. The two causes have different
        remedies (a missing binding vs. a malformed source), so they are worth
        separating explicitly rather than re-deriving downstream.
        """
        if self.tier is not Tier.C:
            return None
        artifacts = "residual_latex" in self.failed_gates
        if artifacts and n_error:
            return "errors+artifacts"
        return "artifacts" if artifacts else "errors"


# ---------------------------------------------------------------------------
# Residual-LaTeX artifacts
# ---------------------------------------------------------------------------

#: Below this share of the document's characters, residual markup is noise
#: rather than a structural failure.
ARTIFACT_RATE_WARN = 0.0005


@dataclass
class ArtifactReport:
    """Residual-LaTeX findings for one converted document."""

    text_chars: int = 0
    total: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)
    top_sequences: list[tuple[str, int]] = field(default_factory=list)
    unresolved_refs: int = 0
    samples: list[str] = field(default_factory=list)
    """Short excerpts showing each artifact in context, for eyeballing."""

    @property
    def rate(self) -> float:
        """Artifacts per character of rendered text."""
        return self.total / self.text_chars if self.text_chars else 0.0

    @property
    def structural(self) -> int:
        """Artifacts that indicate a broken construct rather than stray spacing.

        A leaked ``\\ref`` in an author list is a defect at any document size,
        so these are counted regardless of rate -- unlike ``\\hbox``/``\\kern``
        noise, which only matters in volume.
        """
        return sum(self.by_kind.get(k, 0) for k in ("ref", "cite", "mail", "affil", "graphics", "env"))

    @property
    def notable(self) -> bool:
        return self.structural > 0 or (self.total > 0 and self.rate >= ARTIFACT_RATE_WARN)

# ---------------------------------------------------------------------------
# A converted document
# ---------------------------------------------------------------------------

#: Refuse HTML larger than this. A runaway document must be dropped while it is
#: still one string: measured over 27,003 documents the p99.99 is 9.96 MB, and a
#: single 72-section paper produced 1,444 MB and drove peak RSS to 10.2 GB on its
#: own. Row-count and byte-accumulation limits cannot catch it, because they are
#: tested after the row has been appended.
MAX_HTML_BYTES = 64 * 1024 * 1024


@dataclass
class ConvertedDocument:
    """The outcome of putting one submission through LaTeXML.

    Every submission produces one of these, including the ones that produced no
    HTML at all -- a PDF-only submission, a tarball with no root ``.tex``, an
    unreadable source. The denominator stays "of all submissions" rather than
    silently becoming "of submissions that converted".
    """

    arxiv_id: str
    source_sha256: str

    kind: str | None = None
    """``tar``, ``single_file``, ``pdf_only`` or ``empty``; the shape of the submission."""

    root_tex: str | None = None
    html: str | None = None

    status: Status = Status.NO_SOURCE
    tier: Tier = Tier.REJECTED
    counts: HtmlCounts = field(default_factory=HtmlCounts)
    failed_gates: tuple[str, ...] = ()

    n_warning: int = 0
    n_error: int = 0
    n_fatal: int = 0
    n_artifacts: int = 0

    source_expects_math: bool | None = None
    """``None`` means the source was never read -- distinct from ``False``.

    A re-tiering pass must be able to tell "we never looked" from "there is no
    math", or a read failure grades the document better than it is.
    """

    source_expects_figures: bool | None = None

    duration_s: float = 0.0
    log: str | None = None
    n_assets: int = 0
    """Rasterized figures LaTeXML emitted beside the HTML.

    A count, not paths: the scratch tree holding them is deleted as soon as the
    document is done, so any path handed back would already be dangling. A
    caller that wants the bytes passes an ``asset_sink`` to
    :func:`~...latexml.document.convert_submission` and receives them while
    they still exist.
    """

    @property
    def converted(self) -> bool:
        """Whether any HTML came out. Independent of whether it is usable."""
        return self.html is not None
