# Recovering demoted documents

How to raise the corpus A+B rate by fixing what LaTeXML could not bind, without
reconverting 6.57M documents and without corrupting the pool that already holds
them.

## What is actually recoverable

Measured on the 9,995-document proportional sample (per-document logs retained,
weighted to corpus proportions). 41.6% of documents are demoted to tier C or
rejected; that is 36.2% of corpus weight.

| cause | docs | corpus % | recoverable by bindings? |
|---|---|---|---|
| `Error:undefined` | 2,102 | 16.99% | **yes, partially** |
| `Error:unexpected` | 804 | 6.67% | no — malformed source |
| `pdf_only` | 780 | 7.87% | no — there is no LaTeX |
| `Error:misdefined` | 516 | 3.83% | no — author TeX trickery |
| `residual_latex` gate | 501 | 4.64% | **likely, see below** |
| `Error:malformed` | 377 | 3.45% | no — illegal document tree |
| `Fatal:too_many_errors` | 242 | 1.96% | **indirectly** |
| `Fatal:timeout` | 128 | 1.01% | **no — see below** |

**Timeouts are not recoverable and must not be retried.** They look like the
cheapest possible win — `quality.py` itself calls a timeout "retryable at a
higher limit, not a source defect" — but the measured completion rate at 600 s
and at 2700 s is identical. Documents that exhaust 600 s are in unbounded loops
and exhaust 2700 s too. Retrying them costs 128 x 2700 s and recovers nothing.

## The estimate is a floor, not a point

Only **203 documents (1.43% of corpus weight, ~94,000 corpus-wide)** are
recoverable under the strict test: no blocking gate, and no error kind a binding
cannot fix. Two effects push the true number higher, and only measurement will
say by how much:

**Residual LaTeX is often the same bug seen from the other side.** An unbound
macro that reaches the output *is* the leaked control sequence the artifact
scanner flags. The strict test excludes all 501 `residual_latex` documents as
blocked; a binding that defines the macro removes the artifact with it.

**Error cascades collapse.** `Fatal:too_many_errors` fires at 100 errors, and
`undefined` is the most common error by a factor of three. Removing the undefined
errors can drop a document below the fatal threshold, moving it from *rejected*
(no HTML at all) to at least tier C.

So the plausible range is **1.43% floor to ~5% upper**, and Phase 1 measures it
directly rather than modelling it further.

## Rank targets by recovery, not by frequency

| package/class | affected | recoverable | corpus % | effort |
|---|---|---|---|---|
| `sn-jnl` (Springer Nature) | 80 | **36** | 0.21% | low — metadata stubs |
| `arydshln.sty` | 106 | **29** | 0.21% | low — ignore rules, keep cells |
| `mdframed.sty` | 104 | **27** | 0.16% | low — ignore frame, keep body |
| `mciteplus.sty` | 36 | 18 | 0.15% | medium |
| `biblatex.sty` | **186** | 15 | 0.10% | high — bibliography machinery |
| `achemso` | 31 | 15 | 0.14% | low |
| `tabu.sty` | 44 | 15 | 0.09% | medium |

**`biblatex` is first by documents affected and fifth by documents recovered.**
171 of its 186 documents have something else wrong as well, so a biblatex binding
leaves them demoted regardless. Ranking by raw frequency points at the most
expensive target with nearly the least payoff.

The macros cluster by *class*, not individually: eleven of the top-22 undefined
macros (`\affiliation`, `\orgname`, `\orgdiv`, `\orgaddress`, `\fnm`, `\sur`,
`\city`, `\postcode`, `\country`, `\state`, `\bibcommenthead`) all belong to
`sn-jnl`. There are 2,580 distinct undefined macros and the top 20 cover only 13%
of doc-macro pairs, so per-macro work is hopeless; per-class work is tractable.

Most of these are **metadata** macros. The binding does not have to render them
correctly, only absorb their arguments so they neither error nor leak into the
text. That is what makes the top three low-effort.

## Phase 0 — done: the bindings already exist, as deliberate stubs

The premise that bindings must be written from scratch was wrong, and checking
cost an hour. `/opt/ar5iv-bindings/bindings` holds **81 `.ltxml` files**,
including `biblatex`, `arydshln`, `mdframed`, `tabu`, `tabularray`, `mciteplus`,
`datetime`, `xr`, `breqn` and `pst-plot` -- all of which the logs reported as
`missing_file`. They are found and loaded; the warning is emitted *by the binding
itself*:

```
Warning:missing_file:mdframed.sty  mdframed.sty is only minimally stubbed
and will not be interpreted raw.  at mdframed.sty.ltxml; line 13
```

ar5iv ships stubs that exist so LaTeXML will not digest the raw `.sty`, announce
themselves, and define almost nothing. The macros stay undefined because the stub
never defines them -- not because the file is absent. So the work splits:

| category | packages | work |
|---|---|---|
| **stubbed but incomplete** | biblatex, arydshln, mdframed, tabu, tabularray, mciteplus, datetime, xr, breqn, pst-plot | extend an existing file |
| **genuinely absent** | sn-jnl, achemso, lipics-v2021, wlscirep, ieeecolor | write, but from the OmniBus base |

Both are far cheaper than authoring a binding from nothing.

## Loading is not the scarce resource

Bindings are **demand-loaded**: LaTeXML resolves `X.sty.ltxml` only when a
document says `\usepackage{X}`. A binding for a package the document does not use
is never loaded, so putting every binding on the path is free and cannot
conflict. There is no "enable them all" step -- availability is already
all-at-once. The only collision risks are two files with one name (resolved by
`--path` order) and a buggy binding breaking documents that currently pass.

What is scarce is **authoring effort**, and that is what the selection below
allocates.

## Which bindings to write: measured, not modelled

Recovery is *conjunctive* -- a document is recovered only when **every** package
it needs is fixed -- so this is maximum coverage with conjunctive requirements
rather than classic (disjunctive) set cover. That distinction turns out not to
matter, because the requirement sets are almost all singletons:

| document needs | share |
|---|---|
| **1 package** | **84.2%** |
| 2 packages | 14.0% |
| 3 packages | 1.4% |
| 4 packages | 0.4% |

With 84% needing exactly one package, greedy-by-frequency is near-optimal and a
smarter optimiser would buy almost nothing. Greedy over 493 eligible documents:

```
 1 sn-jnl   +33     2 arydshln +15     3 mdframed +15     4 jfm      +13
 5 cas-dc   +13     6 biblatex +13     7 spie     +10     8 lipics   + 9
 9 siamart  + 9    10 tabu     + 9    11 raa      + 8    12 interact + 8
```

**12 packages recover 155 of 493 eligible documents, ~1.02% of corpus weight,
~67,000 documents corpus-wide.** The curve flattens immediately -- 33 at the
first pick, 8 at the twelfth.

### The pattern worth exploiting

Nine of those twelve are **publisher journal classes** -- Springer (`sn-jnl`),
Elsevier (`cas-dc`), SPIE, SIAM, Dagstuhl (`lipics`), Cambridge (`jfm`), Taylor &
Francis (`interact`) -- and they fail identically: undefined *metadata* macros in
the author block (`\affiliation`, `\orgname`, `\fnm`, `\sur`, `\city`,
`\country`, `\postcode`). Fifteen macros account for essentially all of
sn-jnl's undefined errors.

So the leverage is not twelve bespoke bindings but **one shared front-matter
package** (`journal_metadata.sty.ltxml`) plus a thin class file per publisher
that loads OmniBus and requires it. That also reaches the long tail beyond the
top twelve, which bespoke work never would.

The macros do not need to render as the publisher renders them. They need to
absorb their arguments so nothing errors, and emit the words so author and
affiliation text reaches the output instead of leaking as raw control sequences.

## Phase 1 result — measured, on 80 sn-jnl documents

`journal_metadata.sty.ltxml` + `sn-jnl.cls.ltxml` were written and run against all
80 sampled sn-jnl documents, reconverted from `var_run/projects/` (11 minutes at
32-way).

| transition | docs |
|---|---|
| C -> B | 24 |
| C -> A | 1 |
| rejected -> C | 3 |
| C -> C | 47 |
| rejected -> rejected | 2 |
| **C -> rejected** | **3 (regressions)** |

**28 improved, 49 unchanged, 3 regressed.** 25 documents entered A+B from a
baseline of zero. Measured corpus weight moved into A+B: **0.149%**, against the
greedy model's 0.196% -- so the model runs about **30% optimistic**, and the
12-package projection should be read as ~0.78% (~51,000 documents), not 1.02%.

### The regressions are not a bad binding

Diagnosed on `2410.16667`:

```
baseline      8 x Error:undefined  (\sur, \orgname, \fnm, ... front matter only)
with binding  0 undefined  -- fixed
              100+ x Error:unexpected "Script _ can only appear in math mode",
                    lines 101-422 -- the document *body*
```

The binding did exactly its job. With the front matter no longer failing, LaTeXML
reaches further into the document and meets **pre-existing** source defects it
never previously got to. Those accumulate past `MAX_ERRORS=100` and trip the
`too_many_errors` fatal, so a document that was "tier C with 8 errors" becomes
"rejected with 100+". It did not get worse; the converter got far enough to see
how broken it already was. The two timeout regressions share the mechanism --
more of the document processed means more time.

**Bindings and `MAX_ERRORS` therefore cannot be tuned independently.**
`too_many_errors` is already 242 documents / 1.96% of corpus weight on its own,
and binding work *pushes documents into it*. Raising the limit is the natural
companion change: a document completing with 150 errors lands in tier C, which is
usable, instead of rejected, which is not. That should be measured on the same
loop before either is applied corpus-wide.

Net for this class: **+25 into A+B, -3 to rejected**, on a class where nothing
was passing before.

## Phase 1 — the loop itself (minutes per iteration)

The sample's extracted sources are still on disk: `var_run/projects/`, 9,215
projects, 40 GB. **No tar reading and no extraction** — run the converter
straight against them. ~200 candidates at 48-way concurrency is a ~5 minute loop.

Bindings are mounted, not baked: `BINDING_ARGS` already passes two `--path=`
arguments, so a third pointing at a host directory picks up new `.ltxml` files
with no image rebuild.

```
--path=/lustre/.../recovery-bindings
```

Loop:

1. Build the candidate set from the sample pool + logs (already scripted).
2. Extend a stub, or add a class file over `journal_metadata`.
3. Reconvert candidates from `var_run/projects/`.
4. Re-`assess()` and diff the tier against the baseline row.

**The control set is not optional.** A binding changes behaviour for *every*
document using that class, including the ones already in tier A/B. Each loop must
also reconvert a sample of currently-passing documents that use the same package
and assert that none demote. A binding that recovers 36 documents and breaks 200
is a large net loss, and nothing else in the pipeline would notice.

For `sn-jnl` specifically the control set is *empty* -- all 80 sampled documents
using it are already demoted, so the class cannot regress anything. That is a
property of this class, not a general licence: a shared `journal_metadata`
package applies to classes that do have passing documents, so the control set
must be drawn from those.

Exit criterion: measured recovery on the sample, with zero regressions on the
control set.

## Two levers, measured — and they are not the same metric

### Lever 1: bindings -> tier A+B

Greedy run to exhaustion over the eligible set, under the wider eligibility that
the measurements justified (a leaked control sequence *is* an unbound macro, and
the error-ceiling change absorbs `too_many_errors`):

| packages | docs | corpus % | after the measured 30% haircut |
|---|---|---|---|
| 5 | 132 | 0.82% | 0.57% |
| 10 | 193 | 1.32% | 0.92% |
| 20 | 274 | 1.98% | 1.38% |
| 50 | 404 | 3.05% | 2.14% |
| **218 (exhaustion)** | **634** | **5.51%** | **~3.86%** |

An earlier version of this document said "12 packages, ~1.02%". That was an
artifact of truncating the greedy loop at twelve iterations, not a ceiling.

**The tail is cheap because it is homogeneous.** Past the top twenty the list is
almost entirely journal and conference classes -- `aamas`, `jmlr`, `cvpr`,
`ceurart`, `ifacconf`, `mdpi`, `ieeeaccess`, `Interspeech`, `elife`, `jair` --
failing exactly as `sn-jnl` did. 218 packages is not 218 units of work; it is
~15-20 investigations plus ~200 five-line class files over the shared
`journal_metadata` package.

**Where the ceiling is, and how to know.** 2,102 documents show `Error:undefined`
but only ~634 are ever recoverable; the rest carry blockers a binding cannot
touch. Marginal yield is the practical signal: 33 documents at package #1, ~9 at
#12, 3 at #50, 1 at #100.

Note the greedy is not monotonic -- `achemso` contributes **+15 at step 85**,
because those documents needed a second package first. The claim that "84% need
one package so greedy is near-optimal" holds for the bulk and understates the
tail.

### Lever 2: MAX_ERRORS -> tier C

`MAX_ERRORS` is a LaTeXML state value (`Core/State.pm:96`, default 100) with no
command-line option, so it is set from a preloaded binding.

Measured on 80 sampled `too_many_errors` documents: **47 recover (59%),
rejected -> C, zero regressions**, extrapolating to ~142 of 242 documents and
**~1.15% of corpus weight**.

**Set it to 2000, not higher.** The most error-laden recovered document finishes
at 1,424 errors. Of the 33 that stay dead, 23 hit *exactly* 10,001 errors of a
single repeated kind -- a runaway loop, not a document with many distinct
problems -- and 5 more die on LaTeXML's own `if_limit`/`pushback_limit` loop
detectors. A higher ceiling recovers nothing extra and makes every runaway burn
proportionally longer.

**The two levers compose.** Binding work pushes documents past 100 errors, and
the raised ceiling catches exactly those: in the sn-jnl cohort the
binding-induced fatal regression resolved, `rejected` 5 -> 3.

### They must not be added together

Bindings move documents into **A+B** and raise the headline arXiv-strict rate.
`MAX_ERRORS` moves documents **rejected -> C** and does not change A+B at all --
it grows the usable extraction corpus, since tier C is inside the MinerU set.
Which number matters depends on which question is being asked.

## Phase 2 — extend to the corpus

**Recovery output goes to its own pool.** Adding a binding changes the converter,
which changes `config_hash`, which yields a new `cfg-<hash>` directory. That is
the invariant working as designed, not an obstacle: each pool stays
config-homogeneous, so `config_hash` keeps meaning what it claims.

The alternative — writing recovered rows back into the base pool as a new
iteration — was rejected. It breaks config-homogeneity; `already_converted()`
would skip the candidates because they already exist under the old converter; and
no reader deduplicates today, so every naive consumer (`ParquetReader` pointed at
`html/`) would silently return both the old rejected row and the new good one.

```
cfg-<base>/      6.57M rows, converter A          (untouched)
cfg-<recovery>/  ~94k rows, converter B           (bindings)
```

**The unit of recovery is a document, not a shard.** ~94k documents spread over
12,830 shards is ~7 per shard, so `convert_shard.py` needs a member filter — an
`--only` list, the inverse of the `seen` set it already maintains. Everything
else (extract, convert, flush, marker, atomic publish) is unchanged.

Cost: ~94k x 43 core-s = **~1,120 core-hours**, roughly 1.5 h at 32x48 — about
1.4% of the base run. The floor is the sequential read of the source tars, which
is IO-bound and dominates.

## Phase 3 — layered views

A dataset becomes a *layered* view: an ordered list of pools with a precedence
rule, extending the existing `_meta/views/` mechanism rather than replacing it.

```json
{
  "view": "corpus-v1",
  "layers": [
    {"pool": "cfg-<base>",     "iterations": [1]},
    {"pool": "cfg-<recovery>", "iterations": [1]}
  ],
  "precedence": "later layers win on arxiv_id"
}
```

Readers union the layers and keep the last row per `arxiv_id`. That is ~15 lines
in `reconcile.py`, `compute_analysis.py` and `run_mineru.py`, all of which we
control.

Note this **cannot** be materialised as a symlink tree. A part file contains both
superseded and surviving rows, so precedence is a row-level predicate; whole-file
links would over-include. If a single flat dataset is wanted later, seal it by
streaming both pools and writing deduplicated part files — one IO-bound pass over
~232 GB, no reconversion — and do that once at the end rather than per iteration.

## Risks

**A binding regresses passing documents.** The main one. Guarded by the control
set in Phase 1; it is the reason that control set exists.

**Binding content changes without the argv changing.** `config_hash` covers the
argv and the image digest, not the contents of a mounted bindings directory. Two
runs with the same flags and different binding files would land in the same pool
and be indistinguishable. Fold a hash of the bindings directory into
`config_hash` before Phase 2, for exactly the reason the image digest was folded
in.

**Recovery is measured on the sample and applied to the corpus.** The sample is
proportional and weighted, so the estimate generalises — but the *candidate list*
for Phase 2 must be derived from the base pool itself, not extrapolated from the
sample.

**Diminishing returns are steep.** After the top three classes the curve
flattens: 2,580 distinct undefined macros, top 20 covering 13% of pairs. Stop when
a class costs more than a day and recovers under ~0.1% of corpus weight.
