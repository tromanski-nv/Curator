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

r"""Invoke ar5iv-configured LaTeXML and record exactly how it was invoked.

**The argument order in :data:`MATH_ARGS` is load-bearing.**  ``latexmlc`` makes
the *first* math format the primary ``MathProcessor``, and
``LaTeXML::Post::TeXMath`` implements no ``combineParallel``.  Passing
``--mathtex`` before ``--pmml`` therefore makes every formula error out:

======================  =========================================================
``--pmml --mathtex``    rc=0, 0 errors, 305 ``<math>`` with original TeX alttext
``--mathtex --pmml``    rc=1, 117 errors, 2 fatals, **0 bytes** written
======================  =========================================================

The loud failure is the lucky one.  ``Fatal:too_many_errors`` only fires past
100 errors, so a paper with fewer formulas exits **rc=0** and writes
plausible-looking HTML with every equation silently deleted.  Measured on real
submissions: two of four papers produced 25 KB and 41 KB of clean HTML
containing zero math and would have been recorded as successful conversions.

Nothing downstream can detect that from the exit code, which is why
:func:`nemo_curator.stages.interleaved.latex.latexml.quality.assess` gates on
the math count instead, and why :func:`build_argv` returns the full ordered argv
for the provenance record rather than a summary of it.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

#: Output format.  ``html5`` is what the ar5iv bindings target.
FORMAT_ARGS: tuple[str, ...] = ("--format=html5",)

#: Math flags, in the ONLY correct order.  ``--pmml`` must come first so
#: presentation MathML is the primary processor; ``--mathtex`` then attaches the
#: original TeX as ``alttext`` on each ``<math>`` element.  Never reorder these.
MATH_ARGS: tuple[str, ...] = ("--pmml", "--mathtex")

#: ar5iv bindings shipped in the ar5ivist image.  ``bindings`` carries LaTeXML
#: implementations of packages LaTeXML does not know natively; the
#: ``supported_originals`` directory carries real journal classes (revtex.cls,
#: iopart.cls, elsart.cls, mn.sty, newpasp.sty ...) that arXiv submissions ship
#: against.  Without these, papers accumulate undefined-macro errors and trip the
#: 100-error fatal: astro-ph0001003 measured 16 errors without them and 0 with.
AR5IV_BINDINGS_ROOT = "/opt/ar5iv-bindings"
BINDING_ARGS: tuple[str, ...] = (
    "--preload=ar5iv.sty",
    f"--path={AR5IV_BINDINGS_ROOT}/bindings",
    f"--path={AR5IV_BINDINGS_ROOT}/supported_originals",
)

#: ar5iv's own ENTRYPOINT passes ``--timeout=2700``.  A shorter limit does not
#: fail loudly -- it kills convertible papers and records them as failures, so
#: the corpus silently loses its longest documents.  That is why an *arbitrary*
#: cut is dangerous: an early 300s guess killed 12 convertible papers and logged
#: them as failures, which is indistinguishable from a genuine defect.
AR5IV_TIMEOUT_S = 2700
#: Measured cut.  Over the profiled sample the completion rate at 600s and at
#: 2700s is identical (97.3%): every document that finishes at all finishes well
#: inside 600s (p99 495s), and the remainder are runaways that exhaust 2700s and
#: fail anyway.  Capping at 600s therefore changes no document's outcome and only
#: stops paying 2100s per doomed document.  Raise it back to ``AR5IV_TIMEOUT_S``
#: if a future sample shows documents completing in the 600-2700s window.
LATEXML_TIMEOUT_S = 600
#: Wall-clock guard outside LaTeXML, set above its internal timeout so LaTeXML
#: gets to report its own timeout rather than being killed mid-write.
DEFAULT_TIMEOUT_S = LATEXML_TIMEOUT_S + 120

#: Matches ar5iv's ENTRYPOINT.  ``--noinvisibletimes`` suppresses U+2062
#: INVISIBLE TIMES in the MathML: without it those invisible characters land in
#: the extracted math stream and reach the training text.
CONFORMANCE_ARGS: tuple[str, ...] = (f"--timeout={LATEXML_TIMEOUT_S}", "--noinvisibletimes")

#: Deliberately NOT passed: ``--includestyles`` forces LaTeXML to digest the
#: author's raw ``.sty``/``.cls`` files, which fights the bindings above.  It was
#: in the first trial invocation and contributed to the error storm.
#: Also not passed: ar5iv's two ``--css=`` flags reference remote stylesheets
#: that are irrelevant to text extraction and would embed a CDN dependency.
EXCLUDED_ARGS: tuple[str, ...] = ("--includestyles", "--css")


_SEVERITY_RE = re.compile(r"^(Warning|Error|Fatal):", re.MULTILINE)

#: A LaTeXML diagnostic: ``Warning:kind:object message at file; line N col M``.
#: The trailing location is what makes a record actionable -- it points at the
#: exact source construct, which a bare severity count cannot.
_RECORD_RE = re.compile(
    r"^(?P<severity>Warning|Error|Fatal):(?P<kind>[a-z_]+):(?P<object>\S*)[ \t]*(?P<message>.*?)"
    r"(?:\s+at\s+(?P<file>[^;]+);\s*line\s+(?P<line>\d+)(?:\s+col\s+(?P<col>\d+))?)?$",
    re.MULTILINE,
)


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


def build_argv(source: str, destination: str) -> tuple[str, ...]:
    """Build the full ordered ``latexmlc`` argv.

    Returned as a tuple and recorded verbatim in the dataset provenance: this
    review established that two invocations differing only in flag *order* are
    not equivalent, so a provenance record that summarizes or sorts the flags
    cannot distinguish a good corpus from a math-free one.
    """
    return (
        "latexmlc",
        *BINDING_ARGS,
        *FORMAT_ARGS,
        *MATH_ARGS,
        *CONFORMANCE_ARGS,
        f"--source={source}",
        f"--destination={destination}",
    )


def count_severities(log: str) -> tuple[int, int, int]:
    """Return ``(warnings, errors, fatals)`` from a LaTeXML log."""
    counts = {"Warning": 0, "Error": 0, "Fatal": 0}
    for match in _SEVERITY_RE.finditer(log):
        counts[match.group(1)] += 1
    return counts["Warning"], counts["Error"], counts["Fatal"]


def parse_records(log: str, limit: int = 400) -> list[dict[str, object]]:
    """Extract individual diagnostics, not just counts.

    A count says "12 errors"; a record says *which* construct failed and *where*,
    which is the difference between knowing a document is degraded and being able
    to fix the binding responsible.  Capped at *limit* records so one pathological
    document cannot bloat an export.
    """
    records: list[dict[str, object]] = []
    for match in _RECORD_RE.finditer(log):
        if len(records) >= limit:
            break
        line = match.group("line")
        records.append(
            {
                "severity": match.group("severity"),
                "kind": match.group("kind"),
                "object": (match.group("object") or "").strip(),
                "message": " ".join((match.group("message") or "").split())[:300],
                "file": (match.group("file") or "").strip(),
                "line": int(line) if line else None,
                "col": int(match.group("col")) if match.group("col") else None,
            }
        )
    return records


def error_kinds(log: str) -> tuple[str, ...]:
    """Distinct ``Error:``/``Fatal:`` kind tokens in *log*, in first-seen order.

    Fatals are included because ``Fatal:timeout`` is how LaTeXML reports hitting
    its own limit, and callers need to tell that apart from a real failure.
    """
    seen: dict[str, None] = {}
    for match in re.finditer(r"^(?:Error|Fatal):([a-z_]+)", log, re.MULTILINE):
        seen.setdefault(match.group(1), None)
    return tuple(seen)


def convert(
    project_dir: Path,
    root_tex: str,
    destination: Path,
    *,
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> ConversionResult:
    """Convert one project to HTML.

    Runs with ``cwd`` set to the root document's directory so that relative
    ``\\input`` and ``\\includegraphics`` paths resolve the way they would for a
    real LaTeX build.  Never raises for a conversion failure -- the outcome is
    reported in the result so one bad paper cannot stop a shard.
    """
    import time

    root = Path(root_tex)
    workdir = project_dir / root.parent
    destination.parent.mkdir(parents=True, exist_ok=True)
    argv = build_argv(root.name, str(destination))

    before = set(destination.parent.iterdir()) if destination.parent.exists() else set()
    started = time.monotonic()
    timed_out = False
    try:
        completed = subprocess.run(  # noqa: S603 - argv is built here, never shell-interpolated
            argv,
            cwd=workdir,
            # Without this the child inherits our stdin.  TeX prompts
            # interactively on some errors, and LaTeXML passes that through, so
            # the converter will happily *read and consume* whatever the parent
            # had on stdin.  In a shell loop of the form
            # ``while read shard; do ... done < shards.txt`` that is the shard
            # list itself: latexmlc eats pending lines, the loop hits EOF early
            # and exits reporting success.  Measured on a full-corpus run, this
            # silently dropped 78% of the work with no error anywhere.
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        log = (completed.stdout or "") + (completed.stderr or "")
        returncode = completed.returncode
    except subprocess.TimeoutExpired as exc:
        log = f"Fatal:timeout: exceeded {timeout_s}s\n{exc.stderr or ''}"
        returncode, timed_out = -1, True
    duration = time.monotonic() - started

    html = destination.read_text(encoding="utf-8", errors="replace") if destination.exists() else None
    assets = tuple(
        sorted(p.name for p in (set(destination.parent.iterdir()) - before) if p.is_file() and p != destination)
    )
    n_warning, n_error, n_fatal = count_severities(log)
    return ConversionResult(
        html=html,
        argv=argv,
        returncode=returncode,
        duration_s=duration,
        n_warning=n_warning,
        n_error=n_error,
        n_fatal=n_fatal,
        log=log,
        timed_out=timed_out,
        assets=assets,
        error_kinds=error_kinds(log),
    )


def converter_identity(image_path: str | None = None) -> dict[str, str]:
    """Describe the converter precisely enough to reproduce a conversion.

    A container *tag* is a mutable pointer -- the same tag can be re-pushed with
    a different LaTeXML underneath.  When the pinned squashfs is available its
    sha256 is recorded instead, which cannot change under a sealed dataset.
    """
    identity: dict[str, str] = {
        "argv_template": " ".join(build_argv("<source>", "<destination>")),
        "excluded_args": " ".join(EXCLUDED_ARGS),
    }
    for key, env in (
        ("latexml_version", "LATEXML_VERSION"),
        ("latexml_commit", "LATEXML_COMMIT"),
        ("ar5iv_bindings_commit", "AR5IV_BINDINGS_COMMIT"),
        ("max_errors", "MAX_ERRORS"),
        ("max_warnings", "MAX_WARNINGS"),
    ):
        value = os.environ.get(env)
        if value:
            identity[key] = value
    if image_path:
        identity["image_path"] = image_path
        path = Path(image_path)
        if path.exists():
            import hashlib

            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            identity["image_sha256"] = digest.hexdigest()
    return identity
