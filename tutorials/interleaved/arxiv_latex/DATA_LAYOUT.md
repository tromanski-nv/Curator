# arXiv LaTeXML corpus — layout, iteration, and reuse

How the TeX → HTML conversion output is stored so that **growing a sample never
reconverts a document**, and the full-corpus run is the same operation as the
10k sample, just larger.

## Principles

**The HTML is the canonical retained artifact.** Markdown and any later training
view are *derived* and get their own dataset referencing this one. Re-serializing
must never require re-running LaTeXML, because LaTeXML is by far the expensive
step (~62 s/document mean, ~500 s at p99).

**Conversion output is an append-only pool, not a per-run directory.** Runs come
and go; a converted document is converted once, forever. Everything below follows
from that.

**A dataset is a view over the pool, not a copy of it.** "sample-10k",
"sample-50k" and "full" are manifests naming documents. They overlap by design,
and the overlap costs nothing because none of them own bytes.

## Root

Everything is local on Lustre:

```
/lustre/fsw/portfolios/nemotron/users/tromanski/data/arxiv/snapshot-2026-07-27/
├── src/                                 # inputs, 12,830 arXiv_src_YYMM_NNN.tar
└── latexml-html/
    └── cfg-<hash>/                      # ONE converter config == ONE pool
        ├── _meta/
        │   ├── config.json              # write-once: pinned argv, image digest, versions
        │   ├── iterations/iter-NNN.json # what this iteration added, and why
        │   ├── documents.parquet        # pool index, rebuilt from part files
        │   └── views/<name>.json        # a named dataset: which iterations it spans
        ├── html/<shard>/iter-NNN-part-MMMM.parquet
        ├── assets/<shard>/iter-NNN-part-MMMM.tar
        └── logs/<shard>/iter-NNN-part-MMMM.jsonl.zst
```

### The one decision that makes reuse work

**Part files are partitioned by `(shard, iteration)`, and are immutable once
written.**

The obvious layout — one part file per shard — breaks the iteration strategy.
Iteration 1 took 30 documents from 15 shards; iteration 2 took 23-24 documents
from 400 shards. Under one-part-per-shard, growing a shard that was already
sampled means *rewriting* that shard's part file: read it, append, write it back.
That is a rewrite of already-good data, it is not atomic, and two concurrent
iterations touching the same shard race.

Adding the iteration to the path makes every part file write-once. A shard
accumulates part files across iterations and reading it is a glob. Nothing is
ever rewritten, so nothing is ever recomputed, and concurrent iterations cannot
collide because they cannot name the same file.

This is also why **symlinks are not needed**. Symlinking presumes a document's
bytes must appear under each dataset that contains it. They do not: a dataset is
a manifest. And symlinking would not work cleanly anyway — a part file holds many
documents, so it can only be linked whole, which over-includes unless every
dataset happens to be a union of whole part files. (See "Materializing a view"
for when a symlink tree *is* useful.)

### Reuse is keyed on content, not on names

A document is already converted, and must be skipped, when the pool holds a row
with the same:

| Key | Why not something simpler |
|---|---|
| `source_sha256` | Not `arxiv_id`: a later snapshot can carry different bytes for the same id (author replacement). Name-keyed reuse would silently serve the stale conversion. |
| pool `cfg-<hash>` | Not "was it in a previous run": a converter-config change invalidates prior output. The hash is over the full argv + image digest, so this is structural rather than remembered. |

Both are recorded per row, so the check is a join against `documents.parquet`,
never a directory walk of 9,000+ directories.

## Iteration

An iteration is one append to the pool. `_meta/iterations/iter-NNN.json` records
what it added and why:

```json
{
  "iteration": 3,
  "created_utc": "...",
  "rationale": "grow estimation sample 10k -> 50k for +/-0.5% corpus rate",
  "sampling": {
    "seed": 20260728, "shard_count": 1200, "target": 50000,
    "supersedes": 2, "is_superset_of": [1, 2]
  },
  "documents_added": 40123,
  "documents_reused": 9165,
  "part_files": ["html/arXiv_src_2401_007/iter-003-part-0000.parquet", "..."]
}
```

`documents_reused` is the number that matters: it is the proof that growing did
not redo work, and it should equal the previous view's size whenever the sample
is a strict superset.

**Iterations are append-only and samples are supersets**, so a named view is a
*prefix of iterations*:

```json
{ "view": "sample-10k", "iterations": [1, 2], "documents": 9165 }
{ "view": "sample-50k", "iterations": [1, 2, 3], "documents": 49288 }
```

Reading `sample-10k` = read every part file whose iteration is in `[1, 2]`. No
filtering, no per-document lookup, and `sample-10k ⊂ sample-50k` holds by
construction rather than by assertion.

The full-corpus run is not a special case. It is iteration N with
`shard_count = 12830`, reusing every document already in the pool.

## Formats

| Path | Format | Contents |
|---|---|---|
| `html/<shard>/iter-*.parquet` | Parquet + zstd | one row per paper (schema below) |
| `assets/<shard>/iter-*.tar` | tar | `<arxiv_id>/<asset>` — figures as LaTeXML wrote them |
| `logs/<shard>/iter-*.jsonl.zst` | JSONL + zstd | one record per paper: full converter stdout/stderr |
| `_meta/*.parquet` | Parquet | queryable index; plan a run without touching the data |

Parquet for the HTML because the text stage wants column projection and
predicate pushdown on `tier`/`status` — it reads `html` for Tier A+B and never
touches the ~29% it would discard, nor the image bytes.

**Assets are tarred, not loose.** The 10k sample produced **81,847 PNG files**
across 9,165 directories. At corpus scale that is ~18M inodes of mostly <100 KB
files, which is the access pattern Lustre handles worst. One tar per
`(shard, iteration)` turns that into ~13k sequential files.

### `html/<shard>/iter-*.parquet` columns

| Column | Type | Notes |
|---|---|---|
| `arxiv_id` | string | canonical, e.g. `astro-ph/0001001`, `2401.00001` |
| `version` | int32 | arXiv version if recoverable, else null |
| `shard`, `member` | string | source shard, member path inside it |
| `source_sha256` | string | **reuse key** — ties output to exact input bytes |
| `iteration` | int32 | which append wrote this row |
| `root_tex` | string | document chosen for conversion |
| `other_roots` | list\<string\> | other genuine root candidates, not converted |
| `html` | large_string | LaTeXML output, boilerplate footer stripped; null on failure |
| `status` | string | `ok` \| `warning` \| `error` \| `fatal` \| `timeout` \| `no_source` \| `pdf_only` \| `empty_output` \| `suspect_no_math` \| `suspect_artifacts` \| `suspect_truncated` |
| `tier` | string | `A` \| `B` \| `C` \| `rejected` |
| `n_warning`, `n_error`, `n_fatal` | int32 | from the converter log |
| `n_math`, `n_alttext`, `n_img`, `n_section` | int32 | counts in the HTML, gated at write time |
| `failed_gates` | list\<string\> | which quality gates failed |
| `duration_s` | float | conversion wall time |
| `sample_weight` | double | `(shards_in_era/sampled_in_era) × (members_in_shard/taken_from_shard)` |
| `content_derivation` | string | `latex_latexml`; other values never silently mix in |

`sample_weight` is carried per row because the sample is **not** self-weighting:
shards range from 19 to 3,087 members, and iteration 1 was equal-per-era while
iteration 2 was proportional. An unweighted average over the pool is not a corpus
rate. Storing the weight next to the row is what stops that mistake being
available.

`license` is deliberately absent: arXiv source tars carry none. Join it from the
OAI metadata snapshot on `arxiv_id`. Convert everything; filter at training-set
build time.

## Materializing a view

Some tools want a plain directory rather than a manifest. Because part files are
immutable and a view is a whole number of them, a view can be materialized as a
**symlink tree** with no copying:

```
views/sample-10k/html/<shard>/iter-001-part-0000.parquet -> ../../../../cfg-<hash>/html/...
```

This is a convenience, not the source of truth, and it is safe *only* because
views are unions of whole part files. Never symlink at document granularity — it
reintroduces the millions-of-inodes problem the tarring exists to avoid.

## Rules

1. **Part files are immutable.** Never rewrite one. Correcting a document means a
   new iteration that supersedes it, with the reason recorded.
2. **Write atomically** — `os.replace` onto the final name, following
   `deduplication/pdf_sha.py`, so a killed task leaves either a complete part
   file or nothing. Never a truncated file that looks finished.
3. **Bump the pool** (`cfg-<hash>`) when the source snapshot, the converter image
   digest, or any output-affecting argv changes. A logging-only pipeline change
   does not warrant a new pool; record the commit anyway.
4. **Resume is per task and never reads a shared mutable file.** A task is done
   when its part file and its `_meta` sidecar both exist. Thousands of concurrent
   array tasks cannot append to one Parquet file, and a shared JSON counter is a
   lost-update race — so `documents.parquet` and all counts are **derived by a
   single writer** at seal time, never during a run.
5. **A conversion is not trusted because it exited zero.** Every document passes
   `latexml/quality.py` before being written. The gate that matters most: source
   contains math markup and HTML contains no `<math>` ⇒ `suspect_no_math`,
   rejected. Not hypothetical — a misordered math flag produced `rc=0` and 25 KB
   of clean HTML with every equation deleted, and no exit code, log severity, or
   byte count distinguished it from a good conversion.
6. **`content_derivation` on every row**, so a fallback-tier document can be
   excluded from a training mix by predicate.

## Known impurity in the current pool

The 10k run is **not** config-homogeneous, and this is recorded rather than
papered over. The first 402 documents were converted with `--timeout=2700`; the
remaining ~8,760 with `--timeout=600`. Strictly that is two configs and should be
two pools.

They are pooled anyway because the change is provably outcome-neutral *for these
documents*: the maximum observed completion time across the converted set is
**543.7 s** (p99 495 s), so no document in the pool could have been affected by
the lower cap. Any future document completing in the 600–2700 s window would
break that argument, which is why the cap is a named constant with the evidence
in its docstring.

## Size estimate

Sampled ar5iv HTML runs 200 KB – 1.3 MB per paper; the 10k sample is 26 GB
on disk including loose assets. At ~2.9 M papers and ~300 KB mean, roughly
**850 GB raw** and **150–250 GB** as zstd Parquet, plus assets tarred separately.

The per-paper figures are measured; the corpus totals are extrapolations from
9,165 documents. Treat them as planning numbers.
