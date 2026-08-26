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

r"""Regenerate **iteration 1** of the pool from an already-converted run directory.

This is not the corpus path.  ``convert_shard.py`` converts and packs in one
pass and is the only thing that should be pointed at new shards.  This script
exists for exactly one job: reproducing iteration 1's 9,995 rows from
``var_run/html_v2/``, which no other code can do.  Those rows are a genuine
*within-shard subsample* carrying real two-stage ``sample_weight`` values (21.0
to ~2,832); they cannot be retroactively flattened to 1.0 without destroying the
only unbiased corpus estimate that exists.

Because those rows are weighted, every row written here gets
``frame = FRAME_SAMPLE``.  That column is what keeps a weighted aggregate over
the *corpus* frame from double-counting them -- measured on the live pool, 5% of
rows carried 96% of the weight mass and summing naively overstated the corpus
22x.

**Nothing shared is defined here.**  Schema, naming, hashing and atomic writes
all come from ``latexml/pool.py``.  This file previously owned copies of them
and the copies drifted from ``convert_shard``'s -- on ``status``, on what
``duration_s`` measures, and on how identifiers were normalised -- producing a
pool whose rows meant different things depending on which script wrote them.
Importing from one place is what makes that class of bug impossible rather than
merely unlikely.

On ``duration_s``: this run never captured per-document wall clock, so the
column is written **null** here.  See :func:`converter_time_from_log`.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from nemo_curator.stages.interleaved.latex.arxiv.latexml.artifacts import scan as scan_artifacts
from nemo_curator.stages.interleaved.latex.arxiv.latexml.boilerplate import strip_boilerplate
from nemo_curator.stages.interleaved.latex.arxiv.latexml.convert import (
    build_argv,
    converter_identity,
    count_severities,
    error_kinds,
)
from nemo_curator.stages.interleaved.latex.arxiv.latexml.pool import (
    FLUSH_ROWS,
    FRAME_SAMPLE,
    SCHEMA,
    arxiv_id_from_dir,
    config_hash,
    converter_id,
    is_readable_part,
    part_name,
    pool_dir,
    shard_stem,
    write_atomic,
)
from nemo_curator.stages.interleaved.latex.arxiv.latexml.profiling import parse_phase_tree
from nemo_curator.stages.interleaved.latex.arxiv.latexml.quality import (
    assess,
    source_expects_figures,
    source_expects_math,
)
from nemo_curator.stages.interleaved.latex.arxiv.latexml.sampling import parse_shard
from nemo_curator.stages.interleaved.latex.arxiv.latexml.source_text import strip_comments

DEFAULT_SNAPSHOT = "snapshot-2026-07-27"

#: Exit codes the conversion script appended to ``log.txt`` when its outer guard
#: fired: 124 is ``timeout``'s own SIGTERM, 137 the SIGKILL escalation 30s later.
#: A hard-killed LaTeXML never gets to write ``Error:timeout`` into its log, so
#: without this the kill is indistinguishable from "produced nothing" -- which is
#: a different finding with a different fix (retry at a higher limit vs. inspect
#: the source).
_TIMEOUT_RETURN_CODES = frozenset({"rc=124", "rc=137"})


def read_source(run_dir: Path, name: str, root_tex: str | None) -> str | None:
    """The root LaTeX, or ``None`` when it is genuinely unavailable.

    Several quality gates are defined against the source: ``no_math``,
    ``no_figures`` and ``pdf_wrapper`` all ask "what should the output contain?"
    of the *source*, so with no source they never fire.  The failure mode is
    silent and one-directional -- every document looks better than it is, and the
    gate that exists to catch wholesale math deletion is exactly the one turned
    off.

    ``None`` rather than ``""`` for the unavailable case, because the two are not
    the same claim.  An empty string asserts "this source contains no math"; the
    ``source_expects_*`` columns must stay null when the truth is "we could not
    read the source", or a later re-tiering would trust a fabricated ``False``.
    """
    if not root_tex:
        return None
    path = run_dir / "projects" / name / root_tex
    if not path.exists():
        return None
    return strip_comments(path.read_text(errors="replace"))


def converter_time_from_log(log: str) -> float:
    """LaTeXML's self-reported time: the sum of its own top-level stages.

    Deliberately **not** written to ``duration_s``, and deliberately not named
    that.  ``duration_s`` in the pool schema is subprocess wall clock, which is
    what ``convert_shard`` writes; this quantity is smaller and differently
    shaped (it excludes process start-up, container overhead and our own
    extraction).  Mixing them in one column made ``duration_s > 600`` return
    different populations for reasons unrelated to the documents: iteration 1
    topped out at 543.7s, structurally unable to cross the 600s LaTeXML cap it
    was measured *inside*, while iteration 3 had 195 rows above it.  The headline
    core-hours figure was a sum over that mixture.

    Wall clock is not recoverable for this run: the only per-document timings on
    disk are 404 lines of ``html_v2/_progress.txt`` -- 4.4% of 9,165 documents,
    at one-second resolution, from a superseded script that used a 2,820s guard
    instead of 720s.  Backfilling from a 4% single-shard slice measured under a
    different timeout would be a biased column, not a recovered one.  So
    ``duration_s`` is null for iteration 1 and this value is published beside the
    logs as ``converter_time_s``, where its unit is stated and it cannot be
    summed together with wall clock by accident.
    """
    top, _ = parse_phase_tree(log)
    return round(sum(top.values()), 3)


def timed_out_from_log(log: str) -> bool:
    """Whether the outer guard killed this conversion."""
    return any(line.strip() in _TIMEOUT_RETURN_CODES for line in log.splitlines())


def compute_weights(manifest: list[dict], members: dict[str, int], listing: Path) -> dict[str, float]:
    """Weight per document: how many corpus documents it stands for.

    Two stages, because the sample is two-stage.  A shard represents its era in
    proportion to how little of that era was sampled; a document represents its
    shard in proportion to how little of that shard was taken::

        weight = (shards_in_era / shards_sampled_in_era) x (members_in_shard / taken_from_shard)

    The corpus era populations must come from the **full shard listing**, not
    from the manifest -- the manifest only knows about sampled shards, so
    deriving era sizes from it would make every era look fully sampled and
    collapse the first stage to 1.0.  That would silently reduce the corpus rate
    to an unweighted sample mean, which is the exact error this column exists to
    prevent.
    """
    corpus_era = collections.Counter()
    for line in listing.read_text().splitlines():
        shard = parse_shard(line.strip())
        if shard is not None:
            corpus_era[shard.era] += 1

    sampled_shards = {e["shard"]: e["era"] for e in manifest}
    era_sampled = collections.Counter(sampled_shards.values())
    taken = collections.Counter(e["shard"] for e in manifest)

    weights = {}
    for entry in manifest:
        shard, era = entry["shard"], entry["era"]
        stage1 = corpus_era[era] / era_sampled[era] if era_sampled[era] else 1.0
        stage2 = members.get(shard, taken[shard]) / max(1, taken[shard])
        weights[entry["dir"]] = round(stage1 * stage2, 6)
    return weights


def build_row(  # noqa: PLR0913
    entry: dict,
    *,
    run_dir: Path,
    html_subdir: str,
    shard: str,
    iteration: int,
    snapshot: str,
    image: str | None,
    digest: str | None,
    weight: float,
) -> tuple[dict, str]:
    """Grade one already-converted document.  Returns ``(row, log)``."""
    name = entry["dir"]
    doc = run_dir / html_subdir / name
    arxiv_id = arxiv_id_from_dir(name)

    html_path = doc / "index.html"
    html = None
    if html_path.exists() and html_path.stat().st_size > 0:
        html = strip_boilerplate(html_path.read_text(errors="replace"))
    # Read unconditionally.  The log used to be read only when HTML existed,
    # which meant the documents whose failure the log was the *only* record of
    # were exactly the ones whose log was discarded: 24 of the 29 no-HTML
    # documents here are timeouts, and all 24 were being filed as empty_output.
    log = (doc / "log.txt").read_text(errors="replace") if (doc / "log.txt").exists() else ""

    n_warning, n_error, n_fatal = count_severities(log)
    source = read_source(run_dir, name, entry.get("root_tex"))
    has_source = bool(entry.get("root_tex"))

    if not has_source:
        # No LaTeX ever existed, so there is nothing for assess() to grade and
        # the manifest's own classification (pdf_only / empty / tar) is the
        # finding.  ``or "no_source"`` is not cosmetic: ``kind`` was added to the
        # manifest partway through, so 42 of 9,995 entries have none, and the
        # bare ``entry.get("kind")`` this replaces wrote NULL for every one of
        # them -- which then survived every ``WHERE status <> 'no_source'``
        # filter downstream.  Matches convert_shard.py exactly.
        status, tier, gates, counts_ = entry.get("kind") or "no_source", "rejected", [], None
    else:
        # Routed through assess() even with no HTML, which is the whole point:
        # a document that fataled or timed out without writing output is a
        # ``fatal``/``timeout``, not an ``empty_output``.  Short-circuiting to
        # ``empty_output`` here made the sample's rejection-cause breakdown
        # under-count exactly the two causes that are actionable -- and the
        # sample is the population the published numbers describe.
        verdict = assess(
            html,
            source or "",
            n_error=n_error,
            n_fatal=n_fatal,
            n_warning=n_warning,
            timed_out=timed_out_from_log(log),
            error_kinds=error_kinds(log),
            has_source=True,
        )
        status, tier = verdict.status.value, verdict.tier.value
        gates, counts_ = list(verdict.failed_gates), verdict.counts

    artifacts = scan_artifacts(html) if html else None
    row = {
        "arxiv_id": arxiv_id,
        # Carries the identifier through readers that project to [html, url]
        # and drop every other column.
        "url": f"https://arxiv.org/abs/{arxiv_id}",
        "shard": shard,
        "era": entry["era"],
        "iteration": iteration,
        # Never FRAME_CENSUS from this file: every row it writes is a weighted
        # draw, and the frame is what stops it being summed as if it were not.
        "frame": FRAME_SAMPLE,
        "snapshot": snapshot,
        "converter_id": converter_id(image),
        "source_sha256": digest,
        "root_tex": entry.get("root_tex"),
        "html": html,
        "status": status,
        "tier": tier,
        "kind": entry.get("kind"),
        "n_warning": n_warning, "n_error": n_error, "n_fatal": n_fatal,
        "n_math": counts_.n_math if counts_ else 0,
        "n_alttext": counts_.n_alttext if counts_ else 0,
        "n_img": counts_.n_img if counts_ else 0,
        "n_section": counts_.n_section if counts_ else 0,
        "n_artifacts": artifacts.total if artifacts else 0,
        # Null, not False, when the source could not be read -- see read_source.
        "source_expects_math": source_expects_math(source) if source is not None else None,
        "source_expects_figures": source_expects_figures(source) if source is not None else None,
        "failed_gates": gates,
        # Wall clock was never captured for this run; see converter_time_from_log.
        "duration_s": None,
        "sample_weight": weight,
        "content_derivation": "latex_latexml",
    }
    return row, log


def publish(  # noqa: PLR0913
    pool: Path, shard: str, stem: str, rows: list[dict], logs: list[str], assets: list[tuple[str, Path]]
) -> str:
    """Write one part file plus its asset tar and log sidecar, atomically."""
    html_path = pool / "html" / shard_stem(shard) / f"{stem}.parquet"
    write_atomic(pa.Table.from_pylist(rows, schema=SCHEMA), html_path)
    if not is_readable_part(html_path):
        msg = f"part file failed its readback check: {html_path}"
        raise RuntimeError(msg)

    if assets:
        # Assets are tarred: the loose form of this sample is 81,847 PNGs across
        # 9,165 directories, and at corpus scale that is ~18M small-file inodes,
        # which is the access pattern Lustre handles worst.
        tar_path = pool / "assets" / shard_stem(shard) / f"{stem}.tar"
        tar_path.parent.mkdir(parents=True, exist_ok=True)
        # Dot-prefixed and pid-tagged for the same reason write_atomic does it:
        # a visible half-written file is worse than none, because readers that
        # glob the directory pick it up as real data.
        tmp = tar_path.with_name(f".{tar_path.name}.{os.getpid()}.tmp")
        try:
            with tarfile.open(tmp, "w") as tf:
                for arcname, src in assets:
                    tf.add(src, arcname=arcname)
            os.replace(tmp, tar_path)
        finally:
            tmp.unlink(missing_ok=True)
    if logs:
        log_path = pool / "logs" / shard_stem(shard) / f"{stem}.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("\n".join(logs))
    return str(html_path.relative_to(pool))


def merge_view(path: Path, view: str, iteration: int, documents: int) -> None:
    """Fold this iteration into a view instead of replacing it.

    This used to write ``{"iterations": [iteration]}`` outright, so re-packing
    one iteration under an existing ``--view`` name silently rewrote a
    multi-iteration view down to a single iteration -- with no error, and no way
    to tell afterwards which iterations the view had named.  A view is a union
    over time; the write has to be a union too.
    """
    existing = json.loads(path.read_text()) if path.exists() else {}
    by_iteration = {str(k): v for k, v in existing.get("documents_by_iteration", {}).items()}
    by_iteration[str(iteration)] = documents
    iterations = sorted({int(i) for i in by_iteration} | {int(i) for i in existing.get("iterations", [])})
    # Only claim a total when this file has a count for every iteration named.
    # Views written before ``documents_by_iteration`` existed carry a total whose
    # composition is unknown, and adding to it would double-count silently; a
    # null total is a question, a wrong total is an answer.
    complete = all(str(i) in by_iteration for i in iterations)
    path.write_text(
        json.dumps(
            {
                "view": view,
                "iterations": iterations,
                "documents": sum(by_iteration.values()) if complete else None,
                "documents_by_iteration": by_iteration,
            },
            indent=1,
        )
    )


def pack(  # noqa: PLR0913
    run_dir: Path,
    html_subdir: str,
    pool_root: Path,
    iteration: int,
    view: str,
    listing: Path,
    snapshot: str,
    image: str | None,
    pool_name: str | None,
) -> None:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    members = json.loads((run_dir / "shard_members.json").read_text()) if (run_dir / "shard_members.json").exists() else {}
    sha = json.loads((run_dir / "source_sha256.json").read_text()) if (run_dir / "source_sha256.json").exists() else {}
    weights = compute_weights(manifest, members, listing)

    pool = pool_dir(pool_root, pool_name, image)
    by_shard: dict[str, list[dict]] = collections.defaultdict(list)
    for entry in manifest:
        by_shard[entry["shard"]].append(entry)

    counts, part_files = collections.Counter(), []
    converter_seconds = 0.0
    for shard, entries in sorted(by_shard.items()):
        rows, assets, logs, part = [], [], [], 0

        def flush(shard: str = shard, rows: list = rows, logs: list = logs, assets: list = assets) -> None:
            nonlocal part
            if not rows:
                return
            part_files.append(publish(pool, shard, part_name(iteration, part), rows, logs, assets))
            part += 1
            rows.clear()
            logs.clear()
            assets.clear()

        for entry in entries:
            name = entry["dir"]
            row, log = build_row(
                entry, run_dir=run_dir, html_subdir=html_subdir, shard=shard, iteration=iteration,
                snapshot=snapshot, image=image, digest=sha.get(name), weight=weights.get(name, 1.0),
            )
            rows.append(row)
            counts[row["tier"]] += 1
            if log:
                # converter_time_s lives here rather than in duration_s, so the
                # two time bases are never summed together.  See
                # converter_time_from_log.
                seconds = converter_time_from_log(log)
                converter_seconds += seconds
                logs.append(json.dumps({"arxiv_id": row["arxiv_id"], "converter_time_s": seconds, "log": log}))
            for png in sorted((run_dir / html_subdir / name).glob("*.png")):
                assets.append((f"{row['arxiv_id']}/{png.name}", png))
            # Bounds peak memory to FLUSH_ROWS documents rather than to the size
            # of a shard: Table.from_pylist doubles the HTML it is handed.
            if len(rows) >= FLUSH_ROWS:
                flush()
        flush()

    meta = pool / "_meta"
    (meta / "iterations").mkdir(parents=True, exist_ok=True)
    (meta / "views").mkdir(parents=True, exist_ok=True)

    config_path = meta / "config.json"
    if not config_path.exists():
        # Written once.  With --pool-name pinning an existing pool, rewriting
        # this would overwrite the provenance of conversions this run did not
        # produce; converter_id is on every row for exactly that reason.
        config_path.write_text(
            json.dumps(
                {
                    "config_hash": config_hash(image),
                    "argv": list(build_argv("<source>", "<destination>")),
                    "converter": converter_identity(image),
                    "source_snapshot": snapshot,
                },
                indent=1,
            )
        )
    (meta / "iterations" / f"iter-{iteration:03d}.json").write_text(
        json.dumps(
            {
                "iteration": iteration,
                "frame": FRAME_SAMPLE,
                "created_utc": datetime.now(UTC).isoformat(),
                "documents_added": len(manifest),
                "tiers": dict(sorted(counts.items())),
                # Converter self-reported time, NOT wall clock, and not
                # comparable with a core-hours figure derived from duration_s.
                "converter_time_s": round(converter_seconds, 3),
                "part_files": part_files,
            },
            indent=1,
        )
    )
    merge_view(meta / "views" / f"{view}.json", view, iteration, len(manifest))

    print(f"pool: {pool}")
    print(f"  {len(manifest)} documents over {len(by_shard)} shards -> {len(part_files)} part files")
    print(f"  frame: {FRAME_SAMPLE}  snapshot: {snapshot}  converter: {converter_id(image)}")
    print(f"  tiers: {dict(sorted(counts.items()))}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--html-subdir", default="html_v2")
    parser.add_argument("--pool-root", type=Path, required=True)
    parser.add_argument("--pool-name", default=None,
                        help="Pin the pool directory (e.g. cfg-7bcf1d0b875f) instead of deriving it")
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--view", default="sample-10k")
    parser.add_argument("--listing", type=Path, required=True, help="Full shard listing, for corpus era populations")
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    parser.add_argument("--image", default=None, help="Path to the pinned converter squashfs, for converter_id")
    pack(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
