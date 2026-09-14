# The chunk orchestrator: design and failure modes

How a PDF corpus far larger than the available filesystem quota gets through
Nemotron-Parse, unattended.

Read `README.md` first for how to run a single job. This document is about the
campaign: what runs where, why, and which parts are load-bearing because
something broke.

Figures here are deliberately relative. The design depends on ratios, not on the
absolute size of any particular corpus.

---

## 1. The problem shape

Two constraints drive the entire design.

**Bytes.** Source plus output is a multiple of the filesystem quota, so the
corpus cannot be resident. Work is therefore chunked, and space is reclaimed
continuously: peak footprint is `K x per-chunk`, **independent of corpus size**.
That is the property that makes an arbitrarily large corpus fit a fixed quota.

**Inodes.** This is the one people miss. Staging one file per document costs one
inode per document, which for a large corpus exceeds the inode quota by a
multiple — before a single output file is written. No byte allocation fixes it.
Hence tar staging (§3).

A third, softer constraint: byte-exact deduplication removes a large fraction of
the corpus before any GPU work, so it is done first, from the index alone (§3).

---

## 2. The state machine

A chunk is the unit of everything: a state directory, a staging directory, a
remote prefix, and a lock. Chunks are independent — no chunk's success depends
on another's — which is what makes them safe to pipeline.

```
        fetch            process           upload+verify        reap
plan ──────────> _FETCHED ──────> _PROCESSED ──────> _UPLOADED ──────> _REAPED
  │                 │                 │                  │                │
  │            archives on        parquet on         verified in      local data
  │              scratch            scratch            object store      deleted
  └── _PLANNED records which plan dir owns this chunk id
```

Every stage is gated on its on-disk marker, so **any restart is a resume, not a
redo**. That single property is what makes the supervisor's retry loop, the
self-resubmit, and unattended operation safe at all.

Deletion is strictly last and strictly earned: `reap` runs only with a verified
`_UPLOADED`, re-checks content against the manifest, and refuses above a small
miss tolerance. Nothing is deleted on any failure path.

---

## 3. Tar staging: fetch whole archives, filter locally

A chunk is a set of **contiguous** source tars and takes every record in them.
The original design issued one HTTP range-GET per document to reassemble what is
physically a few dozen objects.

Now we download those objects and read members out of them locally by byte
range, using the offset and length already in the index. This works only because
the tars are uncompressed — a compressed member has no addressable range.

| | per-document staging | tar staging |
|---|---|---|
| inodes per chunk | one per document | one per archive |
| requests per chunk | one per document | one per archive |
| measured throughput | baseline | **~3.3x baseline** |

The inode reduction is three to four orders of magnitude, and it is what turns
the corpus from unstageable into routine.

Verified byte-identical before anything depended on it: every sampled member read
locally matched the object-store range-GET exactly, matched the index's recorded
`payload_sha256`, and began with the PDF magic bytes.

**Deduplication happens locally, at parse time, not at fetch time.** The archives
on disk contain every copy; the manifest is built from the *deduplicated* index,
so only survivors reach the GPU. Nothing about the download depends on the
keep-list — which means the keep-list can be revised without re-fetching
anything.

Parse reads these via `tar_base_dir` in `preprocess.py`, modelled on the existing
`jsonl_base_dir` + `byte_offset` path, with the same one-open-per-container
batching.

---

## 4. Where things run

The rule: **the supervisor is a poller; everything heavy leaves as its own job.**
A full chunk lifecycle measured a peak of ~14 threads in the supervisor's tree.

| component | where | how |
|---|---|---|
| campaign supervisor | CPU partition | `campaign_supervisor.sbatch`, self-resubmitting |
| fetch / verify / reap | CPU partition | `srun_self` re-executes this script |
| process | GPU partition | `sbatch` array, one node per shard |
| upload | CPU partition | DataMover job |
| log janitor | CPU partition | `log_janitor.sbatch`, loops `archive-logs` |

Nothing runs on the login node. Its `RLIMIT_NPROC` applies to the user's **entire**
set of processes there, and it has been wedged once already by a runaway pool.

---

## 5. Three separate budgets

These are different resources and conflating any two causes a distinct failure.

**K (`--max-in-flight`) — filesystem bytes.** Chunks in flight. Checked
statically before any worker starts: worst chunk x K must fit under the ceiling,
which no interleaving of per-chunk quota checks can breach. A per-chunk guard
alone cannot bound a concurrent campaign, because K workers can all read the same
free space and all conclude there is room.

**`SRUN_SLOTS` — parent-process threads.** Concurrent blocking `srun` clients,
several threads each. Deliberately *not* K: the stage that dominates wall-clock
(`process`) goes through `sbatch` and holds no client, while fetch/verify/reap
block only for minutes. Setting this equal to K tripped the thread guard.

**Inodes.** Tracked **per staging mode**. Charging one-inode-per-document while
running tar mode over-projects by orders of magnitude and starts refusing fetches
partway through the campaign — for inodes that are never created. That is not
merely pessimistic; it reintroduces, as a false alarm, the exact limit tar mode
exists to remove.

---

## 6. Failure modes that shaped the design

Each of these is a real incident. They are recorded because the code reads
"defensive" without them, and defensive code with no stated reason gets
simplified away.

### `srun` inside an allocation stalls forever, silently

The supervisor ran for **well over a day having processed nothing**, looking
perfectly healthy — `RUNNING`, quota falling, no errors. The log was one line
repeated:

```
srun: Job <id> step creation still disabled, retrying (Requested nodes are busy)
```

With `SLURM_JOB_ID` set, `srun` tries to create a *step inside the caller's
allocation* rather than requesting its own. The supervisor's allocation is a
single node it already occupies, so the step can never be scheduled — and `srun`
retries indefinitely rather than failing.

**Invariant:** `srun_self` strips the caller's `SLURM_*` so it gets its own
allocation. `stream_child` aborts after `STEP_STALL_LIMIT` such retries.

**The general lesson:** every guard in this system — the retry loop, the quota
guard, the reconcile gate, failure isolation — handles *failures*. None of them
fire on *absence of progress*. A hang is more dangerous here than a crash.

### `SLURM_CONF` must survive environment sanitising

Fixing the above by stripping all `SLURM_*` broke `sbatch` entirely:

```
sbatch: error: cli_filter/lua: Unable to stat /etc/slurm/cli_filter.lua
```

`SLURM_CONF` names the cluster's `slurm.conf`; without it the client loads a
default whose `cli_filter/lua` plugin exists on **no node**. That plugin is also
what enforces the site's "must request GPUs outside a CPU partition" policy, so
losing it bypasses a guard even where it doesn't crash.

**Invariant:** strip `SLURM_*`/`SRUN_*`, **keep `SLURM_CONF`**. Applies to both
`submit()` and `srun_self`.

### The submitter's allocation leaking into the array

`env = dict(os.environ)` was harmless while the supervisor ran on the login node.
Under Slurm it handed `SLURM_CPUS_PER_TASK` to the array job, whose own `srun`
then saw it alongside its `SLURM_TRES_PER_TASK` and died in seconds with
*"cpus-per-task set by two different environment variables"*. Every task failed
instantly, and the retry loop — unable to tell an unfixable config fault from a
transient one — burned several attempts in minutes.

### `squeue`'s exit code cannot mean "finished"

`squeue -j <id>` exits non-zero with *"Invalid job id specified"* both for a job
that has **left the queue** and for one that never existed. Keying on the exit
code makes a transient controller failure look like a completed array, after
which the retry planner resubmits shards that are **still running** — two arrays
writing the same paths on a full set of GPU nodes each. The drain loop polls
enormously often across a campaign, so this is a when, not an if.

Note the trap: the *obvious* fix (treat non-zero as unknown) is also wrong, and
would hang every chunk to its timeout. Only the error text separates the cases.

**Invariant:** absence must be confirmed by text, repeated, and corroborated by
`sacct` reaching a terminal state.

### Sequential chunks

`_run_stages` was a plain loop that blocked on each chunk's GPU array, making the
campaign's duration the sum of every chunk. Chunks are independent, so they
pipeline; K in flight divides the wall-clock by K. One chunk's `SystemExit` is a
verdict on that chunk, not the campaign — failures are recorded to
`run_report.json` and the rest continue.

### Deletion guards are per-destination, never widened

`assert_safe_remote` (output), `assert_safe_logs_remote` (logs) and
`assert_safe_local_delete` / `assert_safe_log_mirror_delete` are deliberate
near-duplicates. The source corpus shares a bucket with our destinations and we
hold DELETE on it. Widening one guard to admit a second destination is exactly
how a guard stops guarding.

### The mirror that copied a virtualenv

The Ray temp mirror — a periodic `cp -a` of Ray's session directory — accumulated
inodes at a rate that would have exhausted the inode quota on its own, before any
staging, after which `quota_guard` refuses fetches and the campaign stalls
partway with no obvious cause. Over 80% of it was `runtime_resources/`, the actor
venv, re-copied every pass for hours.

Now `rsync -a --exclude=runtime_resources/` (incremental, and without the venv),
with `archive-logs` tarring what remains to the object store before reclaiming
it. The spill directory is skipped explicitly: it matches the mirror's glob, and
enabling object spilling would otherwise stream document content into what is
supposed to be a log archive.

---

## 7. Operating it

```bash
# launch
sbatch tutorials/slurm/campaign_supervisor.sbatch
sbatch tutorials/slurm/log_janitor.sbatch

# watch
squeue --me
tail -f $WORK/logs/campaign/supervisor_<jobid>.log
$PY chunk_orchestrator.py --plan-dir $PLAN status --chunks <range>
```

**Health is not "the supervisor is RUNNING."** Check that per-chunk arrays exist
and that `_REAPED` markers are accumulating. §6's first entry is what a
healthy-looking dead campaign looks like.

`run_report.json` in the plan directory records completed and failed chunks with
reasons, continuously.

---

## 8. What is deliberately not automatic

- **Nothing deletes source data.** Ever. Guards refuse it structurally.
- **Reap refuses above its reconcile miss tolerance.** It stops rather than
  guessing.
- **Page truncation at `--max-pages` is not recorded.** The output's per-document
  `num_pages` lets you find affected documents afterwards — see `BACKFILL.md`.
- **The keep-list is not regenerated automatically.** It is an input, built once
  and verified against every record in the full index.
