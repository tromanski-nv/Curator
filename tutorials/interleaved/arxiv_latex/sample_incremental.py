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

r"""Grow a conversion sample to size N without re-doing what is already done.

Two properties make this safe to re-run, and both need care:

**The sample is a prefix of a stable order.** Papers are ranked by
``md5(member_name + seed)``, so raising ``--target`` extends the prefix and the
previous sample is always a strict subset. Re-running with the same target is a
no-op. Sorting by tar order would also be stable but biases toward papers
submitted early in the month, since arXiv shards are ordered by id.

**Work is skipped by inspecting the output, not a ledger.** A paper is staged if
its project directory exists, and converted if its ``index.html`` is non-empty.
A ledger can disagree with reality after a crash; the filesystem cannot.

Usage::

    python sample_incremental.py --run-dir /path/to/run --target 1200
    # then convert only what it wrote to roots_todo.tsv

Writes ``manifest.json`` (merged, never truncated) and ``roots_todo.tsv``
(only papers still needing conversion).
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from nemo_curator.stages.interleaved.latex.latexml.extract import extract_submission  # noqa: E402
from nemo_curator.stages.interleaved.latex.latexml.sampling import (  # noqa: E402
    estimation_sample,
    load_shards,
    parse_shard,
    weighted_sample,
)

DEFAULT_SEED = 20260728


def rank_key(member: str, seed: int) -> str:
    """Stable pseudo-random order for a shard's members.

    Deterministic across runs and machines, and independent of how many papers
    are eventually taken -- which is what makes a larger target a superset.
    """
    return hashlib.md5(f"{seed}:{member}".encode()).hexdigest()  # noqa: S324 - ordering only, not security


def already_converted(run_dir: Path, html_subdir: str, name: str) -> bool:
    output = run_dir / html_subdir / name / "index.html"
    return output.exists() and output.stat().st_size > 0


def allocate(total: int, buckets: int) -> list[int]:
    """Spread *total* over *buckets* as evenly as possible, largest bucket first.

    Deterministic given ``(total, buckets)``, so raising the target only ever
    raises each bucket's count -- which is what keeps a grown sample a superset.
    """
    base, extra = divmod(total, buckets)
    return [base + 1] * extra + [base] * (buckets - extra)


def is_pdf_only(member: str) -> bool:
    """True for submissions arXiv stores as a bare PDF, with no LaTeX source.

    These cannot be converted, so staging them writes megabytes to disk for a
    document the converter will never look at.  They are still *counted* in the
    manifest: dropping them from the sampling frame would silently change the
    denominator from "of all submissions" to "of LaTeX submissions" and inflate
    every rate we report.
    """
    return member.endswith(".pdf")


def grow(  # noqa: PLR0913
    src: Path,
    run_dir: Path,
    target: int,
    per_era_shards: int,
    seed: int,
    html_subdir: str,
    listing: Path,
    shard_count: int = 0,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    have = {entry["dir"] for entry in manifest}
    print(f"existing manifest: {len(manifest)} papers")

    shards = load_shards(listing)
    if shard_count:
        # Proportional to each era's share of the corpus, so the pooled rate
        # generalizes.  Between-shard SD (0.098) dominates within-shard (0.055),
        # so how many shards we spread over -- not how many papers we take from
        # each -- is what sets the width of the final confidence interval.
        sample = estimation_sample(shards, n_shards=shard_count, seed=seed, growable=True)
    else:
        sample = weighted_sample(shards, per_era=per_era_shards, seed=seed)

    # Shards already represented in the manifest stay in, at whatever depth they
    # were taken.  This is what makes the grown sample a superset: the union is
    # never smaller, and no already-converted document is ever dropped.
    prior_shards = sorted({entry["shard"] for entry in manifest})
    picked = [name for name in sample.shards if (src / name).exists()]
    fresh = [name for name in picked if name not in set(prior_shards)]
    if not picked:
        msg = f"none of the sampled shards exist under {src}"
        raise SystemExit(msg)

    # Only the shards not already sampled need to absorb the shortfall.
    needed = max(0, target - len(manifest))
    quota = dict(zip(fresh, allocate(needed, len(fresh)), strict=True)) if fresh else {}
    print(f"prior: {len(manifest)} docs over {len(prior_shards)} shards")
    print(f"draw:  {needed} more over {len(fresh)} new shards -> {min(quota.values(), default=0)}-{max(quota.values(), default=0)} per shard")
    print(f"union: {len(prior_shards) + len(fresh)} shards")

    counts_path = run_dir / "shard_members.json"
    member_counts: dict[str, int] = json.loads(counts_path.read_text()) if counts_path.exists() else {}

    added = skipped_pdf = 0
    for shard_name in fresh:
        era = parse_shard(shard_name).era
        try:
            with tarfile.open(src / shard_name) as tf:
                members = [m.name for m in tf.getmembers() if m.isfile() and m.name.endswith((".gz", ".pdf"))]
                # Recorded so each document can be weighted by how many corpus
                # documents it stands for: weight = (M_shard / n_taken).
                member_counts[shard_name] = len(members)
                members.sort(key=lambda m: rank_key(m, seed))
                wanted = members[: quota.get(shard_name, 0)]
                missing = [m for m in wanted if f"{shard_name.replace('.tar', '')}__{Path(m).name.removesuffix('.gz')}" not in have]
                if not missing:
                    continue
                for member in missing:
                    stem = Path(member).name.removesuffix(".gz")
                    name = f"{shard_name.replace('.tar', '')}__{stem}"
                    if is_pdf_only(member):
                        # Counted, not staged -- see is_pdf_only.
                        manifest.append(
                            {"era": era, "shard": shard_name, "dir": name, "root_tex": None, "kind": "pdf_only"}
                        )
                        have.add(name)
                        added += 1
                        skipped_pdf += 1
                        continue
                    payload = tf.extractfile(member).read()
                    project = extract_submission(payload, member, run_dir / "projects" / name)
                    manifest.append(
                        {
                            "era": era,
                            "shard": shard_name,
                            "dir": name,
                            "root_tex": project.root_tex,
                            "kind": project.kind,
                        }
                    )
                    have.add(name)
                    added += 1
        except tarfile.TarError as exc:
            print(f"  ! {shard_name}: {exc}")

    manifest_path.write_text(json.dumps(manifest, indent=1))
    counts_path.write_text(json.dumps(member_counts, indent=1, sort_keys=True))
    todo = [e for e in manifest if e["root_tex"] and not already_converted(run_dir, html_subdir, e["dir"])]
    (run_dir / "roots_todo.tsv").write_text("\n".join(f"{e['dir']}\t{e['root_tex']}" for e in todo))

    eras = collections.Counter(e["era"] for e in manifest)
    no_source = sum(1 for e in manifest if not e["root_tex"])
    print(f"\nnewly staged: {added}  (of which {skipped_pdf} PDF-only, counted but not written to disk)")
    print(f"manifest now: {len(manifest)} papers  {dict(sorted(eras.items()))}")
    print(f"  convertible: {len(manifest) - no_source}   no LaTeX source: {no_source}")
    print(f"still to convert: {len(todo)}  -> {run_dir / 'roots_todo.tsv'}")
    print(f"already converted: {len(manifest) - len(todo) - no_source}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="Directory of arXiv_src_*.tar shards")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--target", type=int, required=True, help="Total papers wanted, old plus new")
    parser.add_argument("--per-era-shards", type=int, default=3)
    parser.add_argument(
        "--shard-count",
        type=int,
        default=0,
        help="Spread the sample over this many shards, allocated proportional to each era's "
        "share of the corpus. This -- not --target -- is what sets the confidence interval: "
        "400 shards gives about +/-1%%, 93 gives +/-2%%, 15 gives +/-5%%, at any N.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--html-subdir", default="html_v2")
    parser.add_argument("--listing", type=Path, default=Path("/tmp/arxiv_ss_listing.txt"))
    args = parser.parse_args()
    grow(
        args.src,
        args.run_dir,
        args.target,
        args.per_era_shards,
        args.seed,
        args.html_subdir,
        args.listing,
        args.shard_count,
    )


if __name__ == "__main__":
    main()
