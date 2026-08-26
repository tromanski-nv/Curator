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

r"""Reproducible, stratified sampling for corpus-scale conversion decisions.

This module exists because of a specific mistake.  Early measurements of this
pipeline were taken from a single shard, ``arXiv_src_0001_001.tar`` -- January
2000.  Every number derived from it (parse rate, unresolved-figure rate, "pandoc
finds zero figures") was reported as though it characterized arXiv.  It does
not.  Pre-2005 is **1.3%** of the corpus by shard count; 2019 and later is
**83%**.  The measurements came from the least representative slice available,
and the conclusion they supported -- that legacy figure macros dominate -- is
close to the opposite of what holds for the bulk of the corpus.

The fix is not "sample more".  It is to separate two jobs that a single sample
cannot do at once:

**Estimation** answers *what rate should I expect corpus-wide?*  It must be
drawn proportional to volume, and it yields rates with confidence intervals.

**Stress** answers *what breaks?*  It deliberately over-samples rare and risky
strata -- the oldest papers, the largest projects, the unusual document classes
-- because a proportional sample of 1,800 papers contains only ~23 pre-2005
papers and will not reliably surface an era-specific bug at all.

A stress sample cannot produce a rate, and an estimation sample cannot find rare
bugs.  Reporting a stress result as a rate is exactly the original error, so
:class:`Sample` records which kind it is and :func:`summarize` refuses to
extrapolate from a stress sample.

.. warning::

   **Do not truncate shards to a fixed byte budget to sample them.**  It looks
   era-neutral and is not.  Submissions have grown by more than an order of
   magnitude, so a fixed prefix yields far more old papers than new ones --
   measured on this corpus, a 39 MB head gave 16 papers from a 2024 shard and
   several times that from a 1996 shard.  Truncation therefore re-introduces
   exactly the old-paper bias this module exists to remove.  Sample *papers*
   (draw member indices across the whole shard), or read whole shards.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path  # noqa: TC003 - used at runtime by load_shards

#: ``arXiv_src_YYMM_NNN.tar`` -- ``YY`` >= 91 means 19YY, otherwise 20YY.
_SHARD_RE = re.compile(r"arXiv_src_(\d{2})(\d{2})_(\d+)\.tar$")

_CENTURY_PIVOT = 91

#: Era boundaries, chosen where the LaTeX/figure technology actually shifts.
_ERA_BOUNDS: tuple[tuple[int, str], ...] = (
    (2004, "1991-2004"),
    (2010, "2005-2010"),
    (2015, "2011-2015"),
    (2020, "2016-2020"),
)
_ERA_LATEST = "2021+"

#: 95% two-sided normal quantile, for the Wilson interval.
Z_95 = 1.959963984540054

#: Below this, a stratum's rate is too noisy to act on; reported but flagged.
MIN_STRATUM_FOR_RATE = 30


class SampleKind(StrEnum):
    """What a sample is allowed to be used for."""

    ESTIMATION = "estimation"
    """Proportional to corpus volume.  Rates generalize."""

    STRESS = "stress"
    """Deliberately skewed toward rare/risky strata.  Finds bugs, not rates."""

    GOLDEN = "golden"
    """Frozen regression set, inspected once, must never regress."""


@dataclass(frozen=True)
class Shard:
    """One source shard, with the attributes we can know without opening it."""

    name: str
    year: int
    month: int
    part: int
    size_bytes: int = 0

    @property
    def era(self) -> str:
        """Coarse stratum capturing the technology shifts that break converters.

        The boundaries are not arbitrary: LaTeX 2.09 and dvips/PostScript figures
        dominate the earliest era, pdflatex and PDF/PNG figures take over through
        the 2010s, and the 2020s add heavy TikZ and modern package churn.
        """
        for upper, label in _ERA_BOUNDS:
            if self.year <= upper:
                return label
        return _ERA_LATEST


@dataclass
class Sample:
    """A reproducible selection of shards."""

    kind: SampleKind
    seed: int
    shards: list[str]
    strata: dict[str, int] = field(default_factory=dict)
    population: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        return json.dumps(payload, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> Sample:
        payload = json.loads(text)
        payload["kind"] = SampleKind(payload["kind"])
        return cls(**payload)


def parse_shard(name: str, size_bytes: int = 0) -> Shard | None:
    """Parse ``arXiv_src_YYMM_NNN.tar``; ``None`` if the name does not match."""
    match = _SHARD_RE.search(name)
    if match is None:
        return None
    yy, mm, part = int(match.group(1)), int(match.group(2)), int(match.group(3))
    year = 1900 + yy if yy >= _CENTURY_PIVOT else 2000 + yy
    return Shard(name=match.group(0), year=year, month=mm, part=part, size_bytes=size_bytes)


def load_shards(listing: Path) -> list[Shard]:
    """Parse a shard listing (one filename per line, e.g. from ``rclone lsf``)."""
    shards = []
    for line in listing.read_text().splitlines():
        shard = parse_shard(line.strip())
        if shard is not None:
            shards.append(shard)
    return shards


#: Between-shard SD of the conversion success rate, measured over 15 shards
#: spanning 1996-2025 (~27 papers each).  Papers cluster by shard -- a shard is
#: one month of one arXiv section, sharing document classes and macro packages --
#: so this, not the per-paper binomial, governs how precise an estimate can be.
MEASURED_BETWEEN_SHARD_SD = 0.098


def required_shard_count(margin_of_error: float = 0.05, between_shard_sd: float = MEASURED_BETWEEN_SHARD_SD) -> int:
    """Shards needed for a corpus rate to within ``±margin_of_error``.

    **This, not** :func:`required_sample_size`, **is the number that matters.**
    Measured between-shard SD is 0.098 against a within-shard binomial SD of
    0.055 at 27 papers, so which shards you draw dominates how many papers you
    draw from them.  Converting more papers per shard buys almost nothing.

    Concretely: ±5% needs ~15 shards, ±3% ~42, ±2% ~93 -- at only 20-30 papers
    each.  Estimating from one shard per stratum produced per-era rates that fell
    *outside* their own binomial confidence intervals (0.978 -> 0.820 for
    1991-2004, where two shards of the same era differed by 36.7 points).
    """
    return math.ceil((Z_95 * between_shard_sd / margin_of_error) ** 2)


def required_sample_size(
    margin_of_error: float = 0.02,
    expected_rate: float = 0.75,
    confidence_z: float = Z_95,
) -> int:
    """Papers needed to estimate a rate to ``±margin_of_error``, ignoring clustering.

    .. warning::

       This assumes papers are independent draws, which they are **not** -- they
       cluster by shard.  Using it alone understates the required effort and
       produces intervals that do not cover the truth.  Size the run with
       :func:`required_shard_count` and use this only to pick how many papers to
       take *within* each chosen shard (20-30 is ample).
    """
    p = expected_rate
    return math.ceil((confidence_z**2) * p * (1 - p) / (margin_of_error**2))


def wilson_interval(successes: int, total: int, confidence_z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval -- correct near 0 and 1, unlike the normal approximation.

    That matters here: conversion success rates sit near 0.9+, where the naive
    interval can extend above 1.0 and understate uncertainty.
    """
    if total == 0:
        return (0.0, 1.0)
    p = successes / total
    z2 = confidence_z**2
    denominator = 1 + z2 / total
    center = (p + z2 / (2 * total)) / denominator
    margin = confidence_z * math.sqrt(p * (1 - p) / total + z2 / (4 * total**2)) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))


def _count_eras(names: list[str]) -> dict[str, int]:
    """Era histogram as a plain dict.

    Deliberately not a ``Counter``: ``dataclasses.asdict`` rebuilds dict
    subclasses by passing an iterable of ``(key, value)`` pairs to the
    constructor, and ``Counter`` counts those pairs as elements -- which turns
    every key into a tuple and corrupts a serialized manifest without erroring.
    """
    counts = Counter(parse_shard(name).era for name in names)
    return dict(sorted(counts.items()))


def _by_era(shards: list[Shard]) -> dict[str, list[Shard]]:
    grouped: dict[str, list[Shard]] = defaultdict(list)
    for shard in shards:
        grouped[shard.era].append(shard)
    return grouped


def stable_rank(name: str, seed: int) -> str:
    """Reproducible pseudo-random ordering key for a shard or member name.

    Ordering by this rather than drawing with :mod:`random` is what makes a
    sample *growable*: the chosen set is always a prefix of one fixed order, so
    raising the target extends it and never re-draws.  ``random.sample`` with a
    larger *n* returns an unrelated set, which would strand every document
    already converted.
    """
    return hashlib.md5(f"{seed}:{name}".encode()).hexdigest()  # noqa: S324 - ordering only, not security


def estimation_sample(shards: list[Shard], n_shards: int, seed: int = 20260728, *, growable: bool = False) -> Sample:
    """Draw shards proportional to each era's share of the corpus.

    Set *growable* to select each era's shards by :func:`stable_rank` prefix
    instead of drawing them, so a later call with a larger *n_shards* is a strict
    superset.  It is off by default because flipping it changes which shards a
    previously recorded sample resolves to, which would silently invalidate the
    provenance in an existing ``run.json``.

    Uses largest-remainder allocation so the per-era counts sum exactly to
    *n_shards* and no era is silently rounded away.

    Note that proportional *allocation* is not required for an unbiased corpus
    rate -- proportional *weighting* is, and :func:`summarize` applies it from
    ``Sample.population``.  Allocating proportionally spends most of the budget
    on the dominant era and leaves the small ones with intervals too wide to act
    on.  When a stratum is cheap to over-sample, prefer
    :func:`weighted_sample`: draw a useful number from every stratum and let the
    weighting correct for it.  That is standard disproportionate stratified
    sampling and gives tighter intervals everywhere for the same total cost.
    """
    grouped = _by_era(shards)
    total = len(shards)
    exact = {era: n_shards * len(group) / total for era, group in grouped.items()}
    allocation = {era: int(value) for era, value in exact.items()}
    remainder = n_shards - sum(allocation.values())
    for era, _ in sorted(exact.items(), key=lambda kv: kv[1] - int(kv[1]), reverse=True)[:remainder]:
        allocation[era] += 1

    rng = random.Random(seed)  # noqa: S311 - reproducible sampling, not cryptography
    chosen: list[str] = []
    for era, group in sorted(grouped.items()):
        take = min(allocation.get(era, 0), len(group))
        ordered = sorted(group, key=lambda s: s.name)
        if growable:
            ordered.sort(key=lambda s: stable_rank(s.name, seed))
            chosen.extend(shard.name for shard in ordered[:take])
        else:
            chosen.extend(shard.name for shard in rng.sample(ordered, take))

    notes = [
        "Proportional to corpus volume: rates from this sample generalize to the corpus.",
        "Rare eras are thin here by construction -- use a stress sample to hunt era-specific bugs.",
    ]
    if growable:
        notes.append("Growable: shards are a stable-rank prefix, so a larger draw is a strict superset.")
    return Sample(
        kind=SampleKind.ESTIMATION,
        seed=seed,
        shards=sorted(chosen),
        strata=_count_eras(chosen),
        population={era: len(group) for era, group in sorted(grouped.items())},
        notes=notes,
    )


def stress_sample(shards: list[Shard], per_era: int = 4, seed: int = 20260728) -> Sample:
    """Draw an equal number of shards from every era, plus the corpus extremes.

    Equal-per-era deliberately over-weights 1991-2004 by roughly two orders of
    magnitude relative to its true share.  That is the point: it is the only way
    to get statistical power on the strata where converters actually break.
    """
    grouped = _by_era(shards)
    rng = random.Random(seed)  # noqa: S311 - reproducible sampling, not cryptography
    balanced: list[str] = []
    for _era, group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda s: s.name)
        balanced.extend(shard.name for shard in rng.sample(ordered, min(per_era, len(ordered))))

    # The largest shards concentrate the biggest projects, which is where
    # timeouts and memory limits surface first.  Tracked separately so they
    # cannot disturb the equal-per-era balance that `strata` reports.
    extremes: list[str] = []
    if any(shard.size_bytes for shard in shards):
        already = set(balanced)
        extremes = [s.name for s in sorted(shards, key=lambda s: -s.size_bytes) if s.name not in already][:2]
    chosen = [*balanced, *extremes]

    notes = [
        "NOT proportional to the corpus: equal weight per era plus the largest shards.",
        "Use to FIND defects. Never quote a rate from this sample -- it is deliberately biased.",
        (
            f"strata describes the {len(balanced)} era-balanced shards; "
            f"{len(extremes)} largest-shard extreme(s) are additional: {extremes}"
        ),
    ]
    return Sample(
        kind=SampleKind.STRESS,
        seed=seed,
        shards=sorted(set(chosen)),
        strata=_count_eras(balanced),
        population={era: len(group) for era, group in sorted(grouped.items())},
        notes=notes,
    )


def weighted_sample(shards: list[Shard], per_era: int, seed: int = 20260728) -> Sample:
    """Draw *per_era* shards from every era, for post-stratified weighting.

    Unlike :func:`stress_sample` this is an ESTIMATION sample: the allocation is
    deliberately disproportionate, but :func:`summarize` re-weights each stratum
    by its true population share, so the corpus rate remains unbiased while every
    stratum gets enough observations to have a usable interval.
    """
    grouped = _by_era(shards)
    rng = random.Random(seed)  # noqa: S311 - reproducible sampling, not cryptography
    chosen: list[str] = []
    for _era, group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda s: s.name)
        chosen.extend(shard.name for shard in rng.sample(ordered, min(per_era, len(ordered))))
    return Sample(
        kind=SampleKind.ESTIMATION,
        seed=seed,
        shards=sorted(chosen),
        strata=_count_eras(chosen),
        population={era: len(group) for era, group in sorted(grouped.items())},
        notes=[
            "Disproportionate allocation, post-stratified weighting.",
            "Equal shards per era for statistical power; summarize() re-weights by population share.",
        ],
    )


@dataclass
class StratumResult:
    """Observed outcome within one stratum."""

    stratum: str
    total: int
    successes: int

    @property
    def rate(self) -> float:
        return self.successes / self.total if self.total else 0.0

    @property
    def interval(self) -> tuple[float, float]:
        return wilson_interval(self.successes, self.total)

    @property
    def underpowered(self) -> bool:
        return self.total < MIN_STRATUM_FOR_RATE


def summarize(
    sample: Sample,
    observed: dict[str, tuple[int, int]],
) -> dict[str, object]:
    """Combine per-stratum outcomes into a corpus estimate.

    Args:
        sample: The sample these results came from.
        observed: ``{stratum: (successes, total)}``.

    Returns:
        Per-stratum rates with Wilson intervals, plus -- only for an estimation
        sample -- a population-weighted corpus rate.  For a stress sample the
        corpus rate is deliberately withheld: extrapolating from a deliberately
        skewed sample is the exact error this module exists to prevent.
    """
    results = [StratumResult(stratum, total, successes) for stratum, (successes, total) in sorted(observed.items())]
    report: dict[str, object] = {
        "kind": sample.kind.value,
        "strata": [
            {
                "stratum": r.stratum,
                "observed": f"{r.successes}/{r.total}",
                "rate": round(r.rate, 4),
                "ci95": [round(r.interval[0], 4), round(r.interval[1], 4)],
                "underpowered": r.underpowered,
            }
            for r in results
        ],
    }

    if sample.kind is not SampleKind.ESTIMATION:
        report["corpus_rate"] = None
        report["corpus_rate_withheld_because"] = (
            f"{sample.kind.value} samples are deliberately not proportional to the corpus; "
            "a weighted rate from one would be misleading."
        )
        return report

    population_total = sum(sample.population.values())
    if population_total:
        weighted = sum(r.rate * sample.population.get(r.stratum, 0) for r in results) / population_total
        report["corpus_rate"] = round(weighted, 4)
        report["corpus_rate_note"] = "Weighted by each era's share of shards, not by paper count."
    underpowered = [r.stratum for r in results if r.underpowered]
    if underpowered:
        report["warning"] = (
            f"underpowered strata (n<{MIN_STRATUM_FOR_RATE}), rates are indicative only: {underpowered}"
        )
    return report
