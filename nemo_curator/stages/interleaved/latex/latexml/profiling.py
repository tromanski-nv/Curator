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

r"""Per-phase timing for the conversion pipeline.

Total wall time per document says *which* papers are slow; it cannot say *what*
is slow, and the two have different fixes. This module records both halves:

**Wrapper phases** (ours) — reading the member out of the shard, gunzipping,
writing the project tree to disk, selecting the root. None of this was ever
measured, so its share was pure assumption.

**LaTeXML phases** (its own log) — ``latexmlc`` already prints a timing per
post-processing pass, e.g. ``(Graphics index.html 10 to process... 8.34 sec)``.
Parsing those is free and far more precise than wrapping the subprocess.

The distinction that matters for optimisation: ``Graphics`` is EPS→PNG
rasterisation through Ghostscript, which is optional work for a text corpus, and
in the sample that measured 8.34s of a 9.09s post-processing budget. If that
generalises, the cheapest large win is not a faster converter but skipping image
conversion for text-only runs.
"""

from __future__ import annotations

import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

#: ``(Graphics index.html 10 to process... 8.34 sec)`` and friends.  The pass
#: name is the first token; the trailing float is its wall time.
_PHASE_RE = re.compile(
    r"\((?P<name>[A-Za-z][\w:]*(?:\[[^\]]*\])?)[^)]*?(?P<secs>\d+\.\d+) sec\)"
)
#: ``(... 9.09 sec))`` -- the doubled paren closes the whole post-processing block.
_POST_TOTAL_RE = re.compile(r"(\d+\.\d+) sec\)\s*\)\s*$", re.MULTILINE)

#: Passes that are genuinely optional for a text-only corpus.
OPTIONAL_PASSES: frozenset[str] = frozenset({"Graphics"})


@dataclass
class Profile:
    """Phase timings for one document, in seconds."""

    wrapper: dict[str, float] = field(default_factory=dict)
    latexml: dict[str, float] = field(default_factory=dict)
    total_s: float = 0.0

    @property
    def wrapper_total(self) -> float:
        return sum(self.wrapper.values())

    @property
    def latexml_total(self) -> float:
        return sum(self.latexml.values())

    @property
    def optional_s(self) -> float:
        """Time spent in passes a text-only run could skip."""
        return sum(
            v for k, v in self.latexml.items() if k.split("[")[0] in OPTIONAL_PASSES
        )

    @property
    def unaccounted_s(self) -> float:
        """Wall time not attributed to any measured phase.

        Large values mean the digest/parse stage dominates -- LaTeXML does not
        emit a timing for it, so it can only be inferred by subtraction.
        """
        return max(0.0, self.total_s - self.wrapper_total - self.latexml_total)

    def as_dict(self) -> dict[str, object]:
        return {
            "total_s": round(self.total_s, 3),
            "wrapper": {k: round(v, 3) for k, v in self.wrapper.items()},
            "latexml": {k: round(v, 3) for k, v in self.latexml.items()},
            "wrapper_total_s": round(self.wrapper_total, 3),
            "latexml_post_total_s": round(self.latexml_total, 3),
            "optional_s": round(self.optional_s, 3),
            "unaccounted_s": round(self.unaccounted_s, 3),
        }


class PhaseTimer:
    """Accumulates wrapper-side phase timings."""

    def __init__(self) -> None:
        self.phases: dict[str, float] = {}

    @contextmanager
    def phase(self, name: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.phases[name] = self.phases.get(name, 0.0) + (
                time.perf_counter() - started
            )


_OPEN_RE = re.compile(r"\((?P<name>[A-Za-z][\w:]*(?:\[[^\]]*\])?)")
_CLOSE_RE = re.compile(r"(?P<secs>\d+\.\d+) sec\)")


def parse_phase_tree(log: str) -> tuple[dict[str, float], dict[str, float]]:
    """Split LaTeXML's timings into non-overlapping top-level stages and post passes.

    **The log is nested, and naively summing every ``... N sec)`` double-counts.**
    ``Processing`` lines sit inside ``Digesting`` (one per package), and the
    ``post-processing`` total is itself the sum of ``Scan``/``Graphics``/``XSLT``.
    Adding all levels together inflates the denominator and misattributes the cost.

    A stage's duration is printed on its *closing* paren, which for a long stage
    is many lines after the name, so this walks the text with a stack: ``(Name``
    pushes, ``N.NN sec)`` assigns to whatever is innermost and pops.

    Returns ``(top_level, post_passes)`` — the first is a genuine partition of
    conversion time, the second breaks down the ``post`` entry within it.
    """
    top: dict[str, float] = {}
    post: dict[str, float] = {}
    stack: list[str] = []
    index = 0
    while index < len(log):
        char = log[index]
        if char == "(":
            match = _OPEN_RE.match(log, index)
            stack.append(match.group("name") if match else "?")
            index += 1
            continue
        if char == ")":
            close = _CLOSE_RE.search(log, max(0, index - 14), index + 1)
            name = stack.pop() if stack else None
            if name and close and name != "Loading":
                depth = len(stack)
                if depth == 0:
                    top[name] = top.get(name, 0.0) + float(close.group("secs"))
                elif depth == 1 and stack[0] == "post":
                    post[name] = post.get(name, 0.0) + float(close.group("secs"))
            index += 1
            continue
        index += 1
    return top, post


def parse_latexml_phases(log: str) -> dict[str, float]:
    """Extract per-pass timings LaTeXML prints for its post-processing stage.

    ``Loading`` lines are excluded: they are one-off module loads shared across
    every document, not per-document work, and including them would make a fast
    conversion look post-processing-bound.
    """
    phases: dict[str, float] = {}
    for match in _PHASE_RE.finditer(log):
        name = match.group("name")
        if name == "Loading":
            continue
        phases[name] = phases.get(name, 0.0) + float(match.group("secs"))
    return phases
