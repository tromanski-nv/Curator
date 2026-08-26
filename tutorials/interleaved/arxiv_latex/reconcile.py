#!/usr/bin/env python
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

r"""Verify that a corpus conversion run actually converted what it claimed.

**This exists because a run lied and nothing noticed.**  A 32-node array
converted 2,871 of 12,830 shards and reported COMPLETED: every ``python3`` call
was guarded by ``|| echo SHARD_FAILED``, ``TASK_DONE`` printed unconditionally,
so the wrap exited 0 whatever happened inside it.  ``sacct`` agreed.  The
signature was visible only by re-deriving it from the preserved logs afterwards
-- 32/32 tasks printed ``TASK_DONE``, zero ``SHARD_FAILED`` anywhere, and 25 of
32 tasks stopped at exactly 89 shards against a stride share of 401.

The root causes of *that* early exit are fixed.  The detection gap is what this
script closes, and it is not specific to those causes: an OOM, a wedged mount, a
preemption, or array tasks that never launch under a QOS cap all reproduce the
incident byte for byte.  So nothing here trusts a success message.  Every check
compares two independently produced things -- part files against the worklist,
rows against the source tar, shards processed against the stride share that task
was supposed to walk -- and disagreement is the finding.

Runnable during a run as well as after it; ``--allow-partial`` is the "in
flight, coverage is expected to be short" mode.

Checks, cheapest first (the default set is all of them but the tar scan):

* **coverage** -- part files present versus the 12,830-shard worklist, missing
  shards named rather than counted, because "9,959 missing" and "9,959 missing,
  all of them the tail of every task's stride" are different incidents.
* **tasks** -- ``TASK_DONE`` count against ``ARRAY_SIZE``, ``SHARD_FAILED``
  lines, and per task the shards processed against that task's stride share.
  This last one is the check that would have caught the incident on day one.
* **integrity** -- every part file's Parquet footer parses, no ``.tmp`` strays
  from killed tasks, no part file whose rows imply an asset tar or log that is
  not there.
* **zero-row parts** -- reported as their own category, never accepted
  silently.  A shard where every member was reused from an earlier iteration
  and a shard whose source tar truncated to zero readable members produce
  *byte-identical* output; only a human with the iteration history can tell
  them apart, so this script refuses to guess.
* **rows** (``--sample-tars N``, off by default) -- rows across a shard's part
  files against ``.gz``/``.pdf`` members in its source tar.  Opening tars is the
  only expensive thing here, hence the sampling.

  This is the one check that catches a **confirmed silent failure**: a source
  tar truncated at a member boundary raises nothing at all -- ``for info in tf``
  simply stops -- so the shard writes a short part file, prints a success line,
  and is marked done forever.

Rows are summed across *all* iterations of a shard, not just the one being
checked, because reuse is by ``source_sha256``: a shard sampled in iteration 1
and completed in iteration 3 holds 24 rows in one part file and N-24 in the
other, and only the total is comparable to the tar.

Nothing here writes to the pool.  Not even the stray ``.tmp`` it reports.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow.parquet as pq

_SNAPSHOT = Path("/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/users/tromanski/containers/src-snapshot-v2")
sys.path.insert(0, str(_SNAPSHOT))
sys.path.insert(1, str(Path(__file__).resolve().parents[3]))

from nemo_curator.stages.interleaved.latex.arxiv.latexml.pool import is_readable_part

DEFAULT_POOL = Path(
    "/lustre/fsw/portfolios/nemotron/users/tromanski/data/arxiv/snapshot-2026-07-27"
    "/latexml-html/cfg-ff4b10540dde"
)
DEFAULT_SRC = Path("/lustre/fsw/portfolios/nemotron/users/tromanski/data/arxiv/snapshot-2026-07-27/src")
DEFAULT_LOGS = Path("/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/users/tromanski/corpus_logs")

#: A task that walked its whole stride lands within this fraction of its share.
#: Slack, not tolerance for loss: the last task of an array can legitimately be
#: one shard short of the first when the worklist does not divide evenly.
DEFAULT_STRIDE_TOLERANCE = 0.02

#: ``arXiv_src_0001_001.tar: 2364 converted, ...`` or ``...: already packed``.
#: Both mean the loop reached that shard, which is what is being counted.
SHARD_LINE_RE = re.compile(r"^(arXiv_src_\S+\.tar):\s")
SHARD_FAILED_RE = re.compile(r"^SHARD_FAILED\s+(\S+)")
TASK_DONE_RE = re.compile(r"^TASK_DONE\b")
TASK_SUMMARY_RE = re.compile(r"^TASK_SUMMARY\s+(.*)$")
#: ``corpus-586578_17.out`` -> job 586578, array task 17.
LOG_NAME_RE = re.compile(r"^corpus-(\d+)_(\d+)\.out$")
PART_RE = re.compile(r"^iter-(\d{3})-part-(\d{4})\.parquet$")


@dataclass
class Report:
    """Findings, and whether they are fatal.

    Warnings are separated from failures because two of them -- zero-row parts
    and a missing asset tar -- have legitimate causes that this script cannot
    distinguish from the illegitimate one.  ``--strict`` promotes them when the
    caller would rather stop and look.
    """

    strict: bool = False
    lines: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def say(self, text: str = "") -> None:
        self.lines.append(text)

    def ok(self, check: str, detail: str) -> None:
        self.say(f"  OK    {check}: {detail}")

    def warn(self, check: str, detail: str) -> None:
        self.warnings.append(f"{check}: {detail}")
        self.say(f"  WARN  {check}: {detail}")

    def fail(self, check: str, detail: str) -> None:
        self.failures.append(f"{check}: {detail}")
        self.say(f"  FAIL  {check}: {detail}")

    def listing(self, names: list[str], cap: int) -> None:
        for name in names[:cap]:
            self.say(f"          {name}")
        if len(names) > cap:
            self.say(f"          ... and {len(names) - cap} more")

    @property
    def failed(self) -> bool:
        return bool(self.failures) or (self.strict and bool(self.warnings))


@dataclass
class ShardParts:
    """What one shard directory holds, by iteration."""

    stem: str
    parts: dict[int, list[Path]] = field(default_factory=dict)
    rows: dict[Path, int] = field(default_factory=dict)
    unreadable: list[Path] = field(default_factory=list)

    def total_rows(self) -> int:
        """Rows across every iteration -- see the module docstring on reuse."""
        return sum(self.rows.values())

    def iteration_rows(self, iteration: int) -> int:
        return sum(self.rows.get(p, 0) for p in self.parts.get(iteration, []))


def read_worklist(path: Path) -> list[str]:
    if not path.is_file():
        msg = f"worklist not found: {path}"
        raise SystemExit(msg)
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def stem_of(shard: str) -> str:
    return shard.removesuffix(".tar")


def scan_pool(pool: Path) -> dict[str, ShardParts]:
    """Index every part file in the pool by shard, reading footers only.

    Footers, deliberately: ``num_rows`` comes from Parquet metadata, so this
    stays constant-memory over a pool holding terabytes of ``html``.  Loading a
    column here would OOM the login node long before it found anything.
    """
    html = pool / "html"
    if not html.is_dir():
        msg = f"pool has no html/ directory: {pool}"
        raise SystemExit(msg)
    shards: dict[str, ShardParts] = {}
    for directory in sorted(p for p in html.iterdir() if p.is_dir()):
        entry = ShardParts(stem=directory.name)
        for part in sorted(directory.glob("*.parquet")):
            match = PART_RE.match(part.name)
            if match is None:
                continue
            iteration = int(match.group(1))
            entry.parts.setdefault(iteration, []).append(part)
            if is_readable_part(part):
                entry.rows[part] = pq.ParquetFile(part).metadata.num_rows
            else:
                entry.unreadable.append(part)
        if entry.parts:
            shards[directory.name] = entry
    return shards


def read_markers(pool: Path, iteration: int) -> dict[str, dict]:
    """Per-shard completion markers, when the writer emits them.

    ``_meta/parts/<shard>-iter-NNN.json`` postdates iterations 1 and 3, so its
    absence is not a finding -- but where it exists it states how many part
    files the writer *intended*, which is the only direct way to tell a shard
    that finished in three parts from one that died after the third of four.
    """
    directory = pool / "_meta" / "parts"
    if not directory.is_dir():
        return {}
    markers: dict[str, dict] = {}
    for path in sorted(directory.glob(f"*-iter-{iteration:03d}.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict):
            markers[path.name.removesuffix(f"-iter-{iteration:03d}.json")] = payload
    return markers


def marker_part_count(marker: dict) -> int | None:
    """Pull an intended part count out of a marker without pinning its schema."""
    for key in ("n_parts", "parts", "part_count", "num_parts"):
        value = marker.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, list):
            return len(value)
    return None


def marker_row_count(marker: dict) -> int | None:
    for key in ("rows", "n_rows", "num_rows", "row_count"):
        value = marker.get(key)
        if isinstance(value, int):
            return value
    return None


@dataclass
class TaskLog:
    """One array task's own account of what it did."""

    offset: int
    path: Path
    shards: set[str] = field(default_factory=set)
    failed: list[str] = field(default_factory=list)
    done: int = 0
    summary: dict[str, str] = field(default_factory=dict)


def parse_one_log(offset: int, path: Path) -> TaskLog:
    """Everything one task said about itself, taken as claims not as facts."""
    task = TaskLog(offset=offset, path=path)
    for line in path.read_text(errors="replace").splitlines():
        shard = SHARD_LINE_RE.match(line)
        if shard is not None:
            task.shards.add(shard.group(1))
            continue
        failure = SHARD_FAILED_RE.match(line)
        if failure is not None:
            task.failed.append(failure.group(1))
            task.shards.add(failure.group(1))
            continue
        if TASK_DONE_RE.match(line):
            task.done += 1
            continue
        summary = TASK_SUMMARY_RE.match(line)
        if summary is not None:
            task.summary = dict(item.split("=", 1) for item in summary.group(1).split() if "=" in item)
    return task


def parse_task_logs(logs: Path, job: str | None) -> tuple[dict[int, TaskLog], str | None]:
    """Read ``corpus-<job>_<task>.out``, newest job unless one is named."""
    if not logs.is_dir():
        return {}, None
    candidates: dict[str, dict[int, Path]] = {}
    for path in sorted(logs.iterdir()):
        match = LOG_NAME_RE.match(path.name)
        if match is None:
            continue
        candidates.setdefault(match.group(1), {})[int(match.group(2))] = path
    if not candidates:
        return {}, None
    chosen = job or max(candidates, key=int)
    if chosen not in candidates:
        msg = f"no logs for job {chosen} under {logs}"
        raise SystemExit(msg)
    return {offset: parse_one_log(offset, path) for offset, path in sorted(candidates[chosen].items())}, chosen


def expected_share(n_shards: int, stride: int, offset: int) -> int:
    """How many worklist entries index ``offset`` under ``stride``.

    Mirrors the driver's ``(n - 1) % STRIDE == OFFSET`` exactly.  Re-deriving it
    from the same worklist the tasks read is the point: a share computed as
    ``n_shards / stride`` would be a guess, and a guess cannot be evidence.
    """
    return len(range(offset, n_shards, stride)) if 0 <= offset < stride else 0


def check_coverage(  # noqa: PLR0913
    report: Report,
    worklist: list[str],
    shards: dict[str, ShardParts],
    iteration: int,
    cap: int,
    allow_partial: bool,
) -> set[str]:
    """Part files present versus the worklist, missing shards named."""
    report.say(f"coverage (iteration {iteration} against {len(worklist)} worklist shards)")
    have = {stem for stem, entry in shards.items() if entry.parts.get(iteration)}
    wanted = [stem_of(s) for s in worklist]
    missing = [stem for stem in wanted if stem not in have]
    extra = sorted(have - set(wanted))

    detail = f"{len(have)}/{len(wanted)} shards hold an iteration-{iteration} part file"
    if missing:
        say = report.warn if allow_partial else report.fail
        say("coverage", f"{detail}; {len(missing)} missing")
        report.listing(missing, cap)
    else:
        report.ok("coverage", detail)
    if extra:
        report.warn("coverage", f"{len(extra)} part-file shards are not in the worklist")
        report.listing(extra, cap)
    return set(missing)


def check_tasks(  # noqa: PLR0913
    report: Report,
    tasks: dict[int, TaskLog],
    job: str | None,
    n_shards: int,
    array_size: int,
    tolerance: float,
    cap: int,
) -> None:
    """The check that would have caught the incident.

    ``TASK_DONE`` is counted, but it is the weakest signal here and is treated
    as such: the incident printed 32 of them.  What matters is the third block,
    comparing each task's processed-shard count against the stride share it was
    handed.  A task that ends early has no way to hide from that comparison,
    whatever it printed on the way out.
    """
    report.say(f"tasks (job {job or 'n/a'}, array size {array_size}, stride {array_size})")
    if not tasks:
        report.fail("tasks", "no per-task logs found; task accounting cannot be verified")
        return

    absent = [str(i) for i in range(array_size) if i not in tasks]
    if absent:
        report.fail("tasks", f"{len(absent)} array tasks produced no log at all (never launched?)")
        report.listing(absent, cap)

    done = sum(1 for task in tasks.values() if task.done)
    if done == array_size:
        report.ok("task_done", f"{done}/{array_size} tasks printed TASK_DONE")
    else:
        report.fail("task_done", f"{done}/{array_size} tasks printed TASK_DONE")

    failed = [(task.offset, shard) for task in tasks.values() for shard in task.failed]
    if failed:
        report.fail("shard_failed", f"{len(failed)} SHARD_FAILED lines across {len({o for o, _ in failed})} tasks")
        report.listing([f"task {offset}: {shard}" for offset, shard in failed], cap)
    else:
        report.ok("shard_failed", "no SHARD_FAILED lines")

    short: list[str] = []
    running: list[str] = []
    total_attempted = 0
    total_expected = 0
    for offset, task in sorted(tasks.items()):
        expected = expected_share(n_shards, array_size, offset)
        finished = bool(task.done or task.summary)
        # The driver's own summary is preferred when present -- it counts the
        # loop from inside, so it also sees shards that produced no output line
        # at all -- but a log without one still gets checked, which is what
        # makes this usable against the preserved logs of the failed run.
        attempted = int(task.summary["attempted"]) if "attempted" in task.summary else len(task.shards)
        total_attempted += attempted
        total_expected += expected
        if not expected or attempted >= expected * (1.0 - tolerance):
            continue
        # A task that has said nothing about finishing is presumed still
        # walking its stride, and "not done yet" is not a defect.  The
        # distinction is the whole check: the incident's tasks were short *and*
        # claimed completion, and that pair is what nothing was looking at.
        line = f"task {offset}: attempted {attempted} of {expected} ({attempted / expected:.0%})"
        (short if finished else running).append(line)
    if short:
        report.fail(
            "stride_share",
            f"{len(short)} of {len(tasks)} tasks claimed completion after processing materially fewer shards "
            f"than their stride share ({total_attempted} attempted vs {total_expected} expected overall)",
        )
        report.listing(short, cap)
    else:
        report.ok(
            "stride_share",
            f"no completed task fell short of its stride ({total_attempted}/{total_expected} shards attempted)",
        )
    if running:
        report.say(f"  ..    stride_share: {len(running)} tasks are short but have not claimed completion yet")
        report.listing(running, cap)


def check_footers(report: Report, pool: Path, shards: dict[str, ShardParts], iteration: int, cap: int) -> None:
    """Every published part file parses as Parquet.

    ``is_readable_part`` is imported from the pool module rather than
    reimplemented, because it is also what resume gates on: if this script and
    resume disagree about what "readable" means, a shard can be simultaneously
    done and broken and neither side notices.
    """
    unreadable = [str(p.relative_to(pool)) for entry in shards.values() for p in entry.unreadable]
    n_parts = sum(len(entry.parts.get(iteration, [])) for entry in shards.values())
    if unreadable:
        report.fail("parquet_footer", f"{len(unreadable)} part files do not parse as Parquet")
        report.listing(unreadable, cap)
    else:
        report.ok("parquet_footer", f"all {n_parts} iteration-{iteration} part files parse")


def check_strays(report: Report, pool: Path, cap: int, in_flight: bool) -> None:
    """Temp files a killed task left behind.

    The writers dot-prefix their temporaries precisely so that readers and
    pyarrow's dataset discovery cannot see them, which also means nobody would
    ever notice one: a 0-byte ``.tar.tmp`` can sit in the pool indefinitely as
    the only surviving trace of a task killed mid-shard.

    ``in_flight`` (``--allow-partial``) demotes this to a warning, and that is
    not a loophole -- a *running* writer always has an open ``.tmp``, and the
    file it is legitimately filling right now is indistinguishable from the one
    a dead task abandoned.  Calling a live write a failure would train the
    reader to ignore the check, which is how the original gap survived.
    """
    directories = [pool / name for name in ("html", "assets", "logs")]
    # De-duplicated: pathlib's ``*`` matches leading dots, unlike the shell, so
    # the two patterns overlap on exactly the dot-prefixed files that matter.
    strays = sorted(
        {
            str(p.relative_to(pool))
            for directory in directories
            if directory.is_dir()
            for pattern in ("*/*.tmp", "*/.*.tmp")
            for p in directory.glob(pattern)
        }
    )
    if not strays:
        report.ok("stray_tmp", "no .tmp strays")
        return
    say = report.warn if in_flight else report.fail
    qualifier = "in-flight writes or abandoned by killed tasks" if in_flight else "left behind by killed tasks"
    say("stray_tmp", f"{len(strays)} temp files, {qualifier}")
    report.listing(strays, cap)


def check_companions(report: Report, pool: Path, shards: dict[str, ShardParts], iteration: int, cap: int) -> None:
    """Shards whose asset tar or converter log is missing entirely.

    Companions are **per shard-run, not per part file**.  ``convert_shard`` emits
    many Parquet parts as it flushes, but opens exactly one asset tar and one log
    stream, named for the *first* part index that run wrote; readers join assets
    to documents by ``arxiv_id`` inside the tar, never by part index.  Checking
    per part therefore reports every part after the first as missing something it
    was never meant to have -- on a corpus of 12,830 shards flushing ~7 parts
    each that is tens of thousands of false warnings, which is worse than no
    check at all because it buries the true ones.
    """
    missing_assets: list[str] = []
    missing_logs: list[str] = []
    for stem, entry in sorted(shards.items()):
        parts = entry.parts.get(iteration, [])
        rows = entry.iteration_rows(iteration)
        if not parts or rows == 0:
            continue
        if not any((pool / "assets" / stem).glob(f"iter-{iteration:03d}-part-*.tar")):
            missing_assets.append(f"{stem} ({rows} rows, {len(parts)} parts)")
        if not any((pool / "logs" / stem).glob(f"iter-{iteration:03d}-part-*.jsonl")):
            missing_logs.append(f"{stem} ({rows} rows, {len(parts)} parts)")
    # Warnings, not failures: an asset tar is only published when a shard
    # produced a PNG and a log only when a document logged something, so both
    # can be legitimately absent.  They are still worth naming, because the
    # same absence is what an abandoned shard looks like.
    if missing_assets:
        report.warn("asset_tar", f"{len(missing_assets)} non-empty shards have no asset tar")
        report.listing(missing_assets, cap)
    else:
        report.ok("asset_tar", "every non-empty shard has an asset tar")
    if missing_logs:
        report.warn("log_jsonl", f"{len(missing_logs)} non-empty shards have no converter log")
        report.listing(missing_logs, cap)
    else:
        report.ok("log_jsonl", "every non-empty shard has a converter log")


def check_markers(
    report: Report,
    shards: dict[str, ShardParts],
    iteration: int,
    markers: dict[str, dict],
    cap: int,
) -> None:
    """Cross-check the writer's completion markers, where it wrote any.

    Used when present, never required: iterations 1 and 3 predate the marker,
    and a check that demanded one would report the entire existing pool as
    broken.  Where markers do exist they are the only statement of how many
    part files a shard was *supposed* to have, which is the difference between
    a shard that finished in three parts and one that died after its third of
    four -- indistinguishable on disk otherwise.
    """
    if not markers:
        report.ok("markers", f"no _meta/parts markers for iteration {iteration} (predates them)")
        return
    disagree: list[str] = []
    unmarked: list[str] = []
    for stem, entry in sorted(shards.items()):
        marker = markers.get(stem)
        if marker is None:
            # Warned about, not failed: a relaunch writes markers only for the
            # shards *it* converts, so every shard inherited from the pre-marker
            # run is legitimately unmarked.  Failing on those would drown the
            # one signal that matters -- a marker that disagrees with the disk.
            if entry.parts.get(iteration):
                unmarked.append(stem)
            continue
        want_parts = marker_part_count(marker)
        have_parts = len(entry.parts.get(iteration, []))
        if want_parts is not None and want_parts != have_parts:
            disagree.append(f"{stem}: marker claims {want_parts} parts, {have_parts} on disk")
        want_rows = marker_row_count(marker)
        have_rows = entry.iteration_rows(iteration)
        if want_rows is not None and want_rows != have_rows:
            disagree.append(f"{stem}: marker claims {want_rows} rows, {have_rows} on disk")
    if disagree:
        report.fail("markers", f"{len(disagree)} shards disagree with their completion marker")
        report.listing(disagree, cap)
    else:
        report.ok("markers", f"{len(markers)} completion markers agree with the parts on disk")
    if unmarked:
        report.warn("markers", f"{len(unmarked)} shards have parts but no marker (pre-marker run, or a killed writer)")
        report.listing(unmarked, cap)


def check_zero_rows(report: Report, shards: dict[str, ShardParts], iteration: int, cap: int) -> None:
    """Zero-row parts: reported always, judged never.

    A shard whose every member was reused from an earlier iteration and a shard
    whose source tar truncated to zero readable members write the *same bytes*.
    Three of these are known-legitimate; a fourth appearing is a question, not a
    regression, so this reports and does not fail.  ``--sample-tars`` including
    the shard is what settles it.
    """
    report.say(f"zero-row parts (iteration {iteration})")
    empties = [
        f"{stem}/{part.name}"
        for stem, entry in sorted(shards.items())
        for part in entry.parts.get(iteration, [])
        if entry.rows.get(part, -1) == 0
    ]
    if not empties:
        report.ok("zero_rows", "no zero-row part files")
        return
    report.warn(
        "zero_rows",
        f"{len(empties)} zero-row part files -- legitimate only if every member was reused from an "
        f"earlier iteration; identical on disk to a shard whose tar truncated to nothing",
    )
    report.listing(empties, cap)


def count_tar_members(path: Path) -> int | None:
    """``.gz``/``.pdf`` members in a source tar, or None if it cannot be read.

    No exception is raised by a tar truncated at a member boundary -- iteration
    just stops -- which is the whole reason the caller compares this against
    rows instead of trusting that the read "worked".
    """
    try:
        with tarfile.open(path) as tf:
            return sum(1 for info in tf if info.isfile() and info.name.endswith((".gz", ".pdf")))
    except (OSError, tarfile.TarError):
        return None


def check_rows_against_source(  # noqa: PLR0913
    report: Report,
    src: Path,
    shards: dict[str, ShardParts],
    iteration: int,
    sample: int,
    seed: int,
    cap: int,
) -> None:
    """Rows versus source-tar members, over a sample of completed shards."""
    completed = sorted(stem for stem, entry in shards.items() if entry.parts.get(iteration))
    if sample == 0:
        report.say("rows vs source (skipped; --sample-tars N to enable)")
        return
    chosen = (
        completed
        if sample < 0 or sample >= len(completed)
        else random.Random(seed).sample(completed, sample)  # noqa: S311 - sampling, not secrets
    )
    report.say(f"rows vs source ({len(chosen)} of {len(completed)} completed shards, seed {seed})")

    mismatches: list[str] = []
    unreadable: list[str] = []
    for stem in sorted(chosen):
        entry = shards[stem]
        members = count_tar_members(src / f"{stem}.tar")
        if members is None:
            unreadable.append(stem)
            continue
        # Summed across every iteration: reuse splits one shard's documents
        # over several part files and only the total is comparable.
        rows = entry.total_rows()
        if rows != members:
            mismatches.append(f"{stem}: {rows} rows across {len(entry.rows)} parts vs {members} tar members")
    if unreadable:
        report.fail("source_tar", f"{len(unreadable)} source tars could not be opened")
        report.listing(unreadable, cap)
    if mismatches:
        report.fail("rows_vs_source", f"{len(mismatches)} of {len(chosen)} sampled shards do not reconcile")
        report.listing(mismatches, cap)
    else:
        report.ok("rows_vs_source", f"{len(chosen)}/{len(chosen)} sampled shards reconcile exactly")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify a corpus conversion run against the pool, the worklist and the source tars.",
    )
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL, help="pool config directory (cfg-<hash>)")
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC, help="directory holding arXiv_src_*.tar")
    parser.add_argument("--logs", type=Path, default=DEFAULT_LOGS, help="directory holding corpus-<job>_<task>.out")
    parser.add_argument("--worklist", type=Path, default=None, help="shards.txt (default: <logs>/shards.txt)")
    # One iteration by default.  The pool holds three with different scopes
    # (1: 9,995-doc sample, 2: 3 smoke shards, 3: the corpus) and reporting
    # them together produces exactly the confusion this tool exists to remove.
    parser.add_argument("--iterations", default="3", help="comma-separated iterations to check (default: 3)")
    parser.add_argument("--array-size", type=int, default=32, help="expected array size, also the stride")
    parser.add_argument("--job", default=None, help="Slurm job id to read logs for (default: newest)")
    parser.add_argument("--sample-tars", type=int, default=0, metavar="N",
                        help="reconcile rows against N random source tars (-1 for all; expensive)")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed, so a finding can be re-derived")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_STRIDE_TOLERANCE,
                        help="fractional shortfall in a task's stride share tolerated before it is a failure")
    parser.add_argument("--max-list", type=int, default=20, help="cap on names printed per finding")
    parser.add_argument("--allow-partial", action="store_true",
                        help="a run still in flight: missing coverage warns instead of failing")
    parser.add_argument("--skip-tasks", action="store_true", help="skip task-log accounting")
    parser.add_argument("--strict", action="store_true", help="treat warnings as failures")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    iterations = [int(value) for value in str(args.iterations).split(",") if value.strip()]
    worklist_path = args.worklist or (args.logs / "shards.txt")
    worklist = read_worklist(worklist_path)
    shards = scan_pool(args.pool)
    tasks, job = ({}, None) if args.skip_tasks else parse_task_logs(args.logs, args.job)

    report = Report(strict=args.strict)
    report.say(f"pool     {args.pool}")
    report.say(f"worklist {worklist_path} ({len(worklist)} shards)")
    report.say(f"shards with part files: {len(shards)}")
    for iteration in iterations:
        report.say()
        report.say(f"=== iteration {iteration} ===")
        check_coverage(report, worklist, shards, iteration, args.max_list, args.allow_partial)
        report.say()
        report.say(f"integrity (iteration {iteration})")
        check_footers(report, args.pool, shards, iteration, args.max_list)
        check_strays(report, args.pool, args.max_list, args.allow_partial)
        check_companions(report, args.pool, shards, iteration, args.max_list)
        check_markers(report, shards, iteration, read_markers(args.pool, iteration), args.max_list)
        report.say()
        check_zero_rows(report, shards, iteration, args.max_list)
        report.say()
        check_rows_against_source(
            report, args.src, shards, iteration, args.sample_tars, args.seed, args.max_list,
        )
    if not args.skip_tasks:
        report.say()
        report.say("=== task accounting ===")
        check_tasks(report, tasks, job, len(worklist), args.array_size, args.tolerance, args.max_list)

    print("\n".join(report.lines))
    print()
    if report.failed:
        print(f"RECONCILE FAILED: {len(report.failures)} failures, {len(report.warnings)} warnings")
        return 1
    print(f"RECONCILE OK: 0 failures, {len(report.warnings)} warnings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
