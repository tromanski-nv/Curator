#!/usr/bin/env python3
"""Resumable chunk orchestrator for a large-scale Nemotron-Parse run.

The corpus does not fit on Lustre (the source exceeds the user's byte quota, and one file per
document would exceed the inode quota), so it is processed in
bounded slices -- "chunks" -- each of which is staged, parsed, uploaded,
verified, and only then deleted locally.

    PLAN -> FETCH -> PROCESS -> UPLOAD (+VERIFY) -> REAP

Every stage is guarded by a marker file under the chunk's *persistent* state
directory, so the orchestrator can be killed at any instant and restarted
without double-fetching, double-submitting, or -- the one that actually costs
money -- deleting something whose output is not durably in object storage.

=============================================================================
SAFETY MODEL (read before editing)
=============================================================================
1. Nothing on Lustre is deleted until the chunk's parquet output has been
   uploaded AND independently verified at the destination: object count plus
   per-object byte size for every file, cross-checked with `rclone check
   --size-only`, plus MD5-vs-ETag where the object was not uploaded multipart.
   A DataMover exit code is *not* verification. REAP re-lists the destination
   immediately before deleting and refuses on any discrepancy.

2. The destination bucket also holds the irreplaceable source
   corpus (pdf_corpus/v1/data|index|errors|markers). Therefore:
     * every remote path is asserted, in code, to sit under the literal prefix
       `pdf_corpus/v1/nemotron_parse_elements/` -- see assert_safe_remote();
     * copy semantics only, never `rclone sync` / `dm sync` -- see
       assert_no_destructive_verb();
     * this module contains NO S3 delete call of any kind. REAP deletes only
       from Lustre, and only paths that assert_safe_local_delete() accepts.

3. Any ambiguity is a stop, not a guess. Unverifiable upload -> no _UPLOADED
   marker -> no REAP -> the data stays where it is.

4. NO HEAVY WORK ON THE LOGIN NODE. RLIMIT_NPROC on this login node is 300
   *threads for the entire user*, shared with every shell and editor. An
   earlier run of this script opened a 64-way ProcessPoolExecutor here and
   consumed ~236 of them, after which fork() failed for everything -- even
   `echo` -- and the session had to be killed. Therefore:
     * the index scan, the fetch, and the destination verification all run
       under `srun` on cpu_datamover; the login-node process is a supervisor
       that holds a handful of threads (see MAX_LOCAL_WORKERS / cmd_run's
       thread-budget monitor);
     * `--in-process` is the only way to run any of them locally. It is
       opt-in, prints a banner, hard-caps the pool at MAX_LOCAL_WORKERS, and
       refuses outright if RLIMIT_NPROC headroom is insufficient;
     * every child is started in its own process group with PR_SET_PDEATHSIG,
       and SIGINT/SIGTERM/SIGHUP/atexit tear the whole group down, so killing
       the orchestrator cannot leave an orphaned srun behind. Slurm *batch*
       jobs (the GPU array) are deliberately NOT killed -- they are adopted by
       job id on the next run.
=============================================================================

Typical use::

    PY=/lustre/fsw/portfolios/nemotron/users/tromanski/scratch/\
nemotron_parse_production/.venv-tools/bin/python

    # 1. one-time deterministic partition of the sorted tar keys
    $PY chunk_orchestrator.py plan --target-input-gib 200

    # 2. drive chunks end to end (resumable; ^C and rerun is safe)
    $PY chunk_orchestrator.py run --chunks 0-7

    # anything else
    $PY chunk_orchestrator.py status
    $PY chunk_orchestrator.py run --chunks 0-7 --dry-run
"""

from __future__ import annotations

import argparse
import atexit
import base64
import contextlib
import ctypes
import fcntl
import glob
import hashlib
import json
import os
import pathlib
import re
import resource
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone

# --------------------------------------------------------------------------
# Paths. These are the literal layout the production run is specified against;
# changing them changes where data lands, so they are constants, not defaults.
# --------------------------------------------------------------------------
BASE = "/lustre/fsw/portfolios/nemotron/users/tromanski"
WORK = f"{BASE}/scratch/nemotron_parse_production"
INDEX_DIR = os.environ.get("CHUNK_INDEX_DIR", f"{WORK}/index_full")
# When set, the plan PROCESSES this index but SIZES staging from INDEX_DIR.
# Those are different questions in tar mode: the archive downloaded holds every
# copy of every payload, while only the deduplicated survivors are handed to the
# GPU. Sizing staging from the deduplicated index would under-project Lustre by
# the duplicate rate -- 45% on this corpus.
DEDUP_INDEX_DIR = os.environ.get("CHUNK_DEDUP_INDEX_DIR", "")
# chunks_v2 = 11,500 PDFs/shard. The original chunks/ used 17,235, which chunk_0000
# proved too large: 2 of its 8 shards hit the 9,900s timeout and needed a second
# pass. Both plans share the same tar partition (sha256 95d00540), so only the
# shard subdivision differs and chunk_0000's completed output stays valid.
PLAN_DIR_DEFAULT = f"{WORK}/chunks_v2"
STATE_ROOT = f"{WORK}/chunk_state"
LOG_ROOT = f"{WORK}/logs"
STAGE_ROOT = f"{BASE}/scratch/eai_crawl_chunks"
OUT_ROOT = f"{BASE}/data/eai-crawl/pdfs/nemotron_parse_elements"
CURATOR_DIR = f"{BASE}/github/Curator"
LAUNCHER = f"{CURATOR_DIR}/tutorials/slurm/submit_production_parse_array.sh"
PRODUCTION_YAML = f"{CURATOR_DIR}/benchmarking/production_nemotron_parse.yaml"
RETRY_ARRAY_SCRIPT = f"{CURATOR_DIR}/tutorials/slurm/retry_array.py"

# --------------------------------------------------------------------------
# Object storage. SOURCE and DESTINATION share a bucket; see SAFETY MODEL.
# --------------------------------------------------------------------------
# Bucket and prefix names identify the dataset, so they live in a site config
# OUTSIDE this repository rather than in source control. Point CHUNK_SITE_CONFIG
# at a JSON file with these keys, or place one at the default path below.
#
# Failure here is deliberately fatal and early: every destructive and every
# remote-writing guard is built from these values, so a missing or partial
# config must stop the process rather than let a guard compare against None.
SITE_CONFIG = os.environ.get(
    "CHUNK_SITE_CONFIG",
    str(pathlib.Path.home() / ".config/nemotron_parse/site.json"),
)


def _site() -> dict:
    try:
        with open(SITE_CONFIG) as fh:
            cfg = json.load(fh)
    except (OSError, ValueError) as e:
        msg = (f"cannot read site config {SITE_CONFIG!r} ({e}). It must be JSON with keys: "
               f"bucket, source_prefix, dest_prefix, dest_token, logs_prefix, logs_token, "
               f"dm_location, rclone_remote. Set CHUNK_SITE_CONFIG to override the path.")
        raise SystemExit(msg) from e
    missing = {"bucket", "source_prefix", "dest_prefix", "dest_token", "logs_prefix",
               "logs_token", "dm_location", "rclone_remote"} - set(cfg)
    if missing:
        msg = f"site config {SITE_CONFIG!r} is missing keys: {sorted(missing)}"
        raise SystemExit(msg)
    return cfg


_SITE = _site()
BUCKET = _SITE["bucket"]
SOURCE_PREFIX = _SITE["source_prefix"]  # tar_key from the index is relative to this
DEST_PREFIX = _SITE["dest_prefix"]
DEST_TOKEN = _SITE["dest_token"]  # must appear in every remote path
# Job logs get their OWN prefix and their OWN guard. Deliberately not reusing
# assert_safe_remote: that function's job is to make it impossible to write
# anywhere near the source corpus, and widening it to admit a second
# destination is exactly how such a guard stops guarding.
LOGS_PREFIX = _SITE["logs_prefix"]
LOGS_TOKEN = _SITE["logs_token"]
DM_LOCATION = _SITE["dm_location"]
RCLONE_REMOTE = _SITE["rclone_remote"]
DM_BIN = "/home/svc-datamover/bin/dm"
STORAGE_LOCATIONS = str(pathlib.Path.home() / ".config/datamover/storage_locations")

# --------------------------------------------------------------------------
# Slurm.
# --------------------------------------------------------------------------
ACCOUNT = "nemotron_n4_pre"
# Fetch and verify are CPU/network work, not DataMover-service work, so they do
# not have to run on cpu_datamover -- and they must not, at scale. The
# cpu-datamover QoS caps the user at node=2, which makes data movement a hard
# floor of ~5.6 days for the campaign's chunks (441s fetch + ~480s verify each) no matter
# how many GPUs are processing. The cpu partition has 55 nodes and the
# cpu-dataprocessing QoS imposes no node limit, taking that floor under a day.
# Verified working: srun -p cpu --qos=cpu-dataprocessing lands on cpu-000N.
# Override with CHUNK_DM_PARTITION / CHUNK_DM_QOS if the cluster changes.
DM_PARTITION = os.environ.get("CHUNK_DM_PARTITION", "cpu")
DM_QOS = os.environ.get("CHUNK_DM_QOS", "cpu-dataprocessing")

# --------------------------------------------------------------------------
# Measured constants (see submit_production_parse_array.sh's header; job
# 715690, the armed production configuration). Used for sizing only -- never
# for verification.
# --------------------------------------------------------------------------
# 6.59 pages/s/GPU * 4 GPUs * 9500s of work / 14.53 pages/PDF = 17,235. This is
# the armed number: job 715690 (tuneC2, --max-num-seqs=128) measured 145,327
# pages in 5,516s on 4 GPUs end-to-end. The superseded 5.78 pages/s/GPU figure
# gave 15,116 PDFs/shard and 8,821 shards; at 6.59 the corpus is 7,737 shards.
PDFS_PER_SHARD = 11500
OUTPUT_BYTES_PER_PDF = int(1.64 * 2**20)  # measured 1.64 MiB parquet per PDF

MARKERS = ("_PLANNED", "_FETCHED", "_PROCESSED", "_UPLOADED", "_REAPED")

# --------------------------------------------------------------------------
# Local-execution budget. See SAFETY MODEL item 4. These are not tuning knobs;
# they are the difference between a supervisor process and a wedged login node.
# --------------------------------------------------------------------------
MAX_LOCAL_WORKERS = 8  # hard cap on any pool created outside an allocation
MAX_ALLOC_WORKERS = 32  # hard cap on any pool created inside one
NPROC_HEADROOM = 80  # process slots left free for the rest of the session
LOCAL_THREAD_CEILING = 120  # cmd_run warns when the USER total passes this
ORCH_TREE_THREAD_CEILING = 64  # ...and aborts when ITS OWN tree passes this

# Concurrent blocking `srun` CLIENTS this orchestrator may hold on the login
# node. Each costs ~8 of the 300 threads RLIMIT_NPROC gives this user there
# (measured: 130 threads / 33 processes at 16 concurrent). This is deliberately
# NOT K: K budgets Lustre space, this budgets login threads, and the 3.5h
# process stage goes through sbatch so it consumes none of these.
SRUN_SLOTS = int(os.environ.get("CHUNK_SRUN_SLOTS", "6"))
_SRUN_SEM = threading.BoundedSemaphore(SRUN_SLOTS)


def orch_tree_ceiling(srun_slots: int = SRUN_SLOTS, k: int = 1) -> int:
    """Thread ceiling for our own subtree, derived rather than guessed.

    ~8 threads per concurrent srun client, one pool thread per in-flight chunk,
    plus the interpreter, the monitor and slack. Keeping this tied to what we
    actually permit means the guard still catches a runaway (the wedge that
    motivated it held 236) instead of being raised until it catches nothing."""
    return max(ORCH_TREE_THREAD_CEILING, 8 * srun_slots + k + 48)


# ==========================================================================
# small utilities
# ==========================================================================
def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


_PRINT_LOCK = threading.Lock()


def log(msg: str) -> None:
    """Thread-safe, and labelled with the chunk when K>1.

    With K chunk workers interleaving on one stdout, an unlabelled line cannot
    be attributed to a chunk, which makes an unattended 11-day log unreadable.
    Worker threads are named chunk_NNNN by _run_stages.
    """
    name = threading.current_thread().name
    tag = f"[{name}] " if name.startswith("chunk_") else ""
    with _PRINT_LOCK:
        print(f"[{utcnow()}] {tag}{msg}", flush=True)


def cid(chunk_id: int) -> str:
    """Canonical zero-padded chunk name. 4 digits covers 1,107 production
    chunks at 200 GiB and the 33,532 a 2-tar test plan produces."""
    if not isinstance(chunk_id, int) or isinstance(chunk_id, bool) or chunk_id < 0:
        msg = f"chunk id must be a non-negative int, got {chunk_id!r}"
        raise ValueError(msg)
    if chunk_id > 9999:  # noqa: PLR2004
        msg = f"chunk id {chunk_id} does not fit the chunk_NNNN naming scheme"
        raise ValueError(msg)
    return f"chunk_{chunk_id:04d}"


def state_dir(chunk_id: int) -> str:
    return f"{STATE_ROOT}/{cid(chunk_id)}"


def stage_dir(chunk_id: int) -> str:
    return f"{STAGE_ROOT}/{cid(chunk_id)}"


def out_dir(chunk_id: int) -> str:
    return f"{OUT_ROOT}/{cid(chunk_id)}"


def remote_prefix(chunk_id: int) -> str:
    """Key prefix (no bucket) for this chunk's output."""
    return f"{DEST_PREFIX}{cid(chunk_id)}/"


def marker_path(chunk_id: int, name: str) -> str:
    if name not in MARKERS:
        msg = f"unknown marker {name!r}"
        raise ValueError(msg)
    return f"{state_dir(chunk_id)}/{name}"


def has_marker(chunk_id: int, name: str) -> bool:
    return os.path.exists(marker_path(chunk_id, name))


def read_marker(chunk_id: int, name: str) -> dict:
    try:
        with open(marker_path(chunk_id, name)) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}


def write_marker(chunk_id: int, name: str, payload: dict, dry: bool = False) -> None:
    payload = {"marker": name, "chunk_id": chunk_id, "utc": utcnow(), **payload}
    path = marker_path(chunk_id, name)
    if dry:
        log(f"DRY-RUN would write {path}: {json.dumps(payload, sort_keys=True)}")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_json_atomic(path, payload)
    log(f"wrote {path}")


def write_json_atomic(path: str, payload: object) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def write_lines_atomic(path: str, lines) -> int:
    tmp = f"{path}.tmp.{os.getpid()}"
    n = 0
    with open(tmp, "w") as fh:
        for line in lines:
            fh.write(line + "\n")
            n += 1
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return n


class ChunkLock:
    """Advisory per-chunk lock so two orchestrators cannot drive one chunk."""

    def __init__(self, chunk_id: int, dry: bool = False):
        self.chunk_id = chunk_id
        self.dry = dry
        self.fh = None

    def __enter__(self):
        if self.dry:
            return self
        os.makedirs(state_dir(self.chunk_id), exist_ok=True)
        self.fh = open(f"{state_dir(self.chunk_id)}/.lock", "w")  # noqa: SIM115
        try:
            fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            self.fh.close()
            msg = f"{cid(self.chunk_id)} is locked by another orchestrator process"
            raise SystemExit(msg) from e
        self.fh.write(f"pid={os.getpid()} host={os.uname().nodename} utc={utcnow()}\n")
        self.fh.flush()
        return self

    def __exit__(self, *_exc):
        if self.fh is not None:
            fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
            self.fh.close()
        return False


# ==========================================================================
# LOCAL EXECUTION BUDGET -- see SAFETY MODEL item 4.
# ==========================================================================
def in_allocation() -> bool:
    """True when this process is inside a Slurm allocation (srun/sbatch)."""
    return bool(os.environ.get("SLURM_JOB_ID"))


def user_thread_count() -> int:
    """Threads this uid currently owns, which is what RLIMIT_NPROC counts."""
    uid = os.getuid()
    total = 0
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/status") as fh:
                mine = False
                for line in fh:
                    if line.startswith("Uid:"):
                        mine = int(line.split()[1]) == uid
                    elif line.startswith("Threads:"):
                        if mine:
                            total += int(line.split()[1])
                        break
        except (OSError, ValueError, IndexError):
            continue
    return total


def nproc_limit() -> int:
    soft, _hard = resource.getrlimit(resource.RLIMIT_NPROC)
    return soft


def _proc_table() -> dict[int, tuple[int, int]]:
    """{pid: (ppid, threads)} for this uid, read in one pass of /proc."""
    uid = os.getuid()
    table: dict[int, tuple[int, int]] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        ppid = threads = 0
        mine = False
        try:
            with open(f"/proc/{entry}/status") as fh:
                for line in fh:
                    if line.startswith("Uid:"):
                        mine = int(line.split()[1]) == uid
                    elif line.startswith("PPid:"):
                        ppid = int(line.split()[1])
                    elif line.startswith("Threads:"):
                        threads = int(line.split()[1])
                        break
        except (OSError, ValueError, IndexError):
            continue
        if mine:
            table[int(entry)] = (ppid, threads)
    return table


def own_tree_threads(root_pid: int | None = None) -> tuple[int, int]:
    """(threads, processes) held by this process and all of its descendants.

    The user's total thread count is what RLIMIT_NPROC actually governs, but it
    also counts every unrelated shell, editor and agent on the login node.
    Aborting our run because someone else is busy is not useful, so the hard
    guard is scoped to the subtree we are responsible for and the user total is
    only a warning -- except when the user total is genuinely near the limit AND
    we are a material contributor to it.
    """
    root = os.getpid() if root_pid is None else root_pid
    table = _proc_table()
    children: dict[int, list[int]] = {}
    for pid, (ppid, _n) in table.items():
        children.setdefault(ppid, []).append(pid)
    seen, stack, threads = set(), [root], 0
    while stack:
        pid = stack.pop()
        if pid in seen or pid not in table:
            continue
        seen.add(pid)
        threads += table[pid][1]
        stack.extend(children.get(pid, ()))
    return threads, len(seen)


def clamp_workers(requested: int, what: str) -> int:
    """Never let a caller-supplied worker count exceed the budget for where we
    are actually running."""
    cap = MAX_ALLOC_WORKERS if in_allocation() else MAX_LOCAL_WORKERS
    n = max(1, min(int(requested), cap))
    if n != requested:
        where = "inside a Slurm allocation" if in_allocation() else "OUTSIDE any allocation (login node)"
        log(f"WARNING: {what} asked for {requested} workers; capped to {n} because we are {where}")
    return n


def assert_nproc_headroom(workers: int, what: str) -> None:
    """Refuse to build a pool that would eat the user's whole process budget.

    This is the specific check that would have prevented the login-node wedge:
    RLIMIT_NPROC here is 300 *threads for the entire user*, not per process.
    """
    soft = nproc_limit()
    if soft in (resource.RLIM_INFINITY, -1):
        return
    used = user_thread_count()
    need = workers * 3 + NPROC_HEADROOM
    log(f"nproc budget for {what}: RLIMIT_NPROC={soft}, {used} threads already owned by this user, "
        f"{workers} workers need ~{need} slots")
    if used + need > soft:
        msg = (f"REFUSING to start {what} with {workers} workers: RLIMIT_NPROC is {soft} for the WHOLE "
               f"user, {used} threads are already in use, and this pool needs ~{need} more. "
               f"Run it under srun instead (that is the default) or lower the worker count. "
               f"Exceeding this limit makes fork() fail for every process you own, including your shell.")
        raise SystemExit(msg)


def local_execution_banner(what: str, workers: int) -> None:
    """Loud, unmissable notice that --in-process was used off-allocation."""
    if in_allocation():
        return
    bar = "!" * 76
    for line in (
        bar,
        "!! --in-process: running %s LOCALLY, outside any Slurm allocation." % what,
        "!! This is the code path that wedged the login node once already.",
        "!! RLIMIT_NPROC = %s threads for your ENTIRE user; pool capped to %d workers."
        % (nproc_limit(), workers),
        "!! The supported path is to omit --in-process and let it srun.",
        bar,
    ):
        log(line)


# ==========================================================================
# CHILD PROCESS SUPERVISION
# A killed orchestrator must not leave an srun/dm running. Children get their
# own process group (so we can signal the subtree) plus PR_SET_PDEATHSIG (so
# even SIGKILL of the orchestrator takes them down).
# ==========================================================================
_PR_SET_PDEATHSIG = 1
_CHILDREN: set[subprocess.Popen] = set()
_CHILDREN_LOCK = threading.Lock()


def _child_preexec() -> None:  # pragma: no cover -- runs post-fork in the child
    os.setsid()
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
    except Exception:  # noqa: BLE001, S110 -- best effort; setsid already helps
        pass


def popen_child(argv: list[str], **kwargs) -> subprocess.Popen:
    proc = subprocess.Popen(argv, preexec_fn=_child_preexec, **kwargs)  # noqa: S603, PLW1509
    with _CHILDREN_LOCK:
        _CHILDREN.add(proc)
    return proc


def forget_child(proc: subprocess.Popen) -> None:
    with _CHILDREN_LOCK:
        _CHILDREN.discard(proc)


def kill_children(grace_s: float = 8.0) -> int:
    """SIGTERM then SIGKILL every live child's process group."""
    with _CHILDREN_LOCK:
        kids = [p for p in _CHILDREN if p.poll() is None]
    for p in kids:
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except OSError:
            try:
                p.terminate()
            except OSError:
                pass
    deadline = time.time() + grace_s
    for p in kids:
        try:
            p.wait(timeout=max(0.1, deadline - time.time()))
        except (subprocess.TimeoutExpired, OSError):
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except OSError:
                pass
    return len(kids)


def _on_signal(signum: int, _frame) -> None:
    n = kill_children()
    log(f"received signal {signum}; terminated {n} child process group(s). "
        f"Slurm batch jobs are left running and will be adopted on the next run.")
    raise SystemExit(128 + signum)


def install_signal_handlers() -> None:
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, _on_signal)
        except (ValueError, OSError):  # not the main thread / not supported
            pass
    atexit.register(kill_children)


def run_cmd(argv: list[str], check: bool = True, capture: bool = True, timeout: int | None = None,
            env: dict | None = None) -> subprocess.CompletedProcess:
    log("$ " + " ".join(argv))
    proc = popen_child(
        argv,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        text=True,
        env=env,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            proc.kill()
        proc.communicate()
        raise
    finally:
        forget_child(proc)
    done = subprocess.CompletedProcess(argv, proc.returncode, out, err)
    if check and done.returncode != 0:
        blob = (done.stdout or "") + (done.stderr or "")
        msg = f"command failed rc={done.returncode}: {' '.join(argv)}\n{blob[-4000:]}"
        raise SystemExit(msg)
    return done


def stream_child(argv: list[str], logfile: str | None = None,
                 env: dict | None = None) -> int:
    """Run a child, mirroring its output to stdout and optionally to a log.

    Used for the srun wrappers, whose output we want to see live and keep.
    """
    log("$ " + " ".join(argv))
    stalled = 0
    fh = None
    if logfile:
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        fh = open(logfile, "a")  # noqa: SIM115
        fh.write(f"\n===== {utcnow()} =====\n$ {' '.join(argv)}\n")
        fh.flush()
    proc = popen_child(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if fh:
                fh.write(line)
                fh.flush()
            # A nested srun that cannot get a step retries forever rather than
            # failing, so an unnoticed misconfiguration looks exactly like a
            # healthy long-running job. Convert it into a loud error.
            if "step creation still disabled" in line or "step creation temporarily disabled" in line:
                stalled += 1
                if stalled >= STEP_STALL_LIMIT:
                    log(f"ABORTING child: {stalled} 'step creation disabled' retries. This srun is "
                        f"asking for a step inside the caller's allocation instead of its own; it "
                        f"will never start. See srun_self's environment sanitising.")
                    with contextlib.suppress(OSError):
                        os.killpg(proc.pid, signal.SIGKILL)
                    break
            else:
                stalled = 0
        rc = proc.wait()
    finally:
        forget_child(proc)
        if fh:
            fh.close()
    return rc


def srun_self(args, sub_argv: list[str], job_name: str, cpus: int, walltime: str,
              logfile: str | None = None) -> int:
    """Re-exec this script under srun on cpu_datamover.

    One place builds every allocation this orchestrator makes for its own
    heavy stages, so the partition/qos/account cannot drift between them.

    Throttled by _SRUN_SEM. A blocking srun CLIENT costs ~8 threads on the
    login node, where this user's RLIMIT_NPROC is 300 for everything they run.
    K=16 chunks all entering fetch at once measured 130 threads across 33
    processes, tripped ThreadBudgetMonitor's ceiling of 64, and had its clients
    killed -- which is what Slurm recorded as "CANCELLED by <uid>" on 32 chunks
    three seconds after they queued.

    So chunks-in-flight and login-side srun clients are budgeted separately: K
    is Lustre space (~405 GiB each), this is login threads. They do not fight,
    because the long stage -- process, ~3.5h -- goes through sbatch and holds no
    client here at all. Only fetch/verify/reap block, and only for minutes.
    """
    argv = ["srun", "--account", ACCOUNT, "--partition", DM_PARTITION, "--qos", DM_QOS,
            "--nodes", "1", "--ntasks", "1", "--cpus-per-task", str(cpus),
            "--time", walltime, "--job-name", job_name,
            sys.executable, os.path.abspath(__file__),
            "--plan-dir", args.plan_dir, *sub_argv]
    # Ask for a NEW allocation, never a step inside the caller's.
    #
    # When the supervisor itself runs under Slurm, SLURM_JOB_ID is set, and srun
    # then tries to create a step within that allocation instead of requesting
    # its own. The supervisor's allocation is one node it already occupies, so
    # the step can never be scheduled and srun retries FOREVER:
    #   "Job <id> step creation still disabled, retrying (Requested nodes are busy)"
    # That is worse than an error -- it never fails, so nothing reports it. The
    # first campaign launch sat in exactly this state for 1d09h having processed
    # nothing, with a healthy-looking RUNNING supervisor.
    #
    # Same rule as submit(): drop the caller's allocation, keep SLURM_CONF (which
    # names this cluster's slurm.conf; without it the client loads a default
    # whose cli_filter/lua plugin exists on no node here).
    env = {k: v for k, v in os.environ.items()
           if k == "SLURM_CONF" or not k.startswith(("SLURM_", "SLURMD_", "SRUN_"))}
    with _SRUN_SEM:
        return stream_child(argv, logfile, env=env)


# ==========================================================================
# SAFETY GUARDS -- these are code, not comments, and are called on every
# destructive or remote-writing path.
# ==========================================================================
def assert_safe_remote(bucket: str, key_prefix: str, chunk_id: int | None = None) -> None:
    """Refuse any remote path that is not strictly inside this pipeline's
    output prefix. the destination bucket also holds the source corpus and we
    hold DELETE on it, so a mistyped prefix is unrecoverable."""
    problems = []
    if bucket != BUCKET:
        problems.append(f"bucket {bucket!r} != {BUCKET!r}")
    if DEST_TOKEN not in key_prefix:
        problems.append(f"key prefix {key_prefix!r} does not contain {DEST_TOKEN!r}")
    if not key_prefix.startswith(DEST_PREFIX):
        problems.append(f"key prefix {key_prefix!r} does not start with {DEST_PREFIX!r}")
    if ".." in key_prefix or key_prefix.startswith("/") or "//" in key_prefix:
        problems.append(f"key prefix {key_prefix!r} is not a normalised relative prefix")
    for forbidden in ("pdf_corpus/v1/data/", "pdf_corpus/v1/index/",
                      "pdf_corpus/v1/errors/", "pdf_corpus/v1/markers/"):
        if key_prefix.startswith(forbidden):
            problems.append(f"key prefix {key_prefix!r} targets source corpus {forbidden!r}")
    if chunk_id is not None:
        want = remote_prefix(chunk_id)
        if not key_prefix.startswith(want):
            problems.append(f"key prefix {key_prefix!r} is not under {want!r}")
    if problems:
        msg = "REFUSING unsafe remote path:\n  " + "\n  ".join(problems)
        raise SystemExit(msg)


def assert_no_destructive_verb(argv: list[str]) -> None:
    """`rclone sync`/`dm sync`/`rclone purge`/`delete` make the destination
    match the source; against a shared bucket that is a corpus-deletion
    primitive. Copy semantics only."""
    banned = {"sync", "purge", "delete", "deletefile", "rmdir", "rmdirs", "cleanup", "move", "moveto"}
    for tok in argv[1:]:
        if tok.lower() in banned:
            msg = f"REFUSING destructive verb {tok!r} in: {' '.join(argv)}"
            raise SystemExit(msg)


def assert_safe_local_delete(path: str, chunk_id: int) -> str:
    """Only two roots are reapable, and only their own chunk_NNNN subtree."""
    real = os.path.realpath(path)
    allowed = [os.path.realpath(stage_dir(chunk_id)), os.path.realpath(out_dir(chunk_id))]
    problems = []
    if real not in allowed:
        problems.append(f"{real!r} is not one of {allowed!r}")
    if cid(chunk_id) not in real:
        problems.append(f"{real!r} does not name {cid(chunk_id)}")
    if real.count("/") < 6:  # noqa: PLR2004
        problems.append(f"{real!r} is suspiciously shallow")
    if real in ("/", BASE, STAGE_ROOT, OUT_ROOT, WORK, STATE_ROOT):
        problems.append(f"{real!r} is a root directory")
    if problems:
        msg = "REFUSING unsafe local delete:\n  " + "\n  ".join(problems)
        raise SystemExit(msg)
    return real


# ==========================================================================
# S3
# ==========================================================================
_S3_LOCAL = threading.local()


def s3_credentials() -> tuple[str, str, str]:
    """Read DataMover's storage_locations. NEVER logged or echoed."""
    txt = pathlib.Path(STORAGE_LOCATIONS).read_text()
    ak = re.search(r"access_key_id:\s*(\S+)", txt)
    sk = re.search(r"secret_access_key:\s*(\S+)", txt)
    ep = re.search(r"endpoint:\s*(\S+)", txt)
    if not (ak and sk and ep):
        msg = f"could not parse credentials from {STORAGE_LOCATIONS}"
        raise SystemExit(msg)
    return ak.group(1), sk.group(1), ep.group(1)


def s3_client(pool: int = 512):
    client = getattr(_S3_LOCAL, "client", None)
    if client is not None:
        return client
    import boto3
    from botocore.config import Config

    ak, sk, ep = s3_credentials()
    client = boto3.client(
        "s3",
        endpoint_url=ep,
        aws_access_key_id=ak,
        aws_secret_access_key=sk,
        region_name="us-east-1",
        config=Config(retries={"max_attempts": 8, "mode": "standard"},
                      max_pool_connections=pool, signature_version="s3v4"),
    )
    _S3_LOCAL.client = client
    return client


def s3_list_prefix(bucket: str, prefix: str, chunk_id: int | None = None) -> dict[str, dict]:
    """{key: {size, etag}} for every object under prefix. Read-only."""
    assert_safe_remote(bucket, prefix, chunk_id)
    client = s3_client(pool=32)
    out: dict[str, dict] = {}
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kwargs["ContinuationToken"] = token
        resp = client.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            out[obj["Key"]] = {"size": int(obj["Size"]), "etag": obj.get("ETag", "").strip('"')}
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return out


# ==========================================================================
# quota
# ==========================================================================
def lfs_quota(fs: str = "/scratch/fsw") -> dict:
    """kbytes/inodes used and limit for $USER. `lfs quota` marks over-quota
    values with a trailing '*', which int() would choke on."""
    proc = run_cmd(["lfs", "quota", "-u", os.environ.get("USER", ""), fs], check=False)
    text = (proc.stdout or "") + (proc.stderr or "")
    nums = None
    for line in text.splitlines():
        if fs in line:
            toks = line.split()
            idx = toks.index(fs) if fs in toks else 0
            nums = [t.rstrip("*") for t in toks[idx + 1:]]
            break
    if not nums or len(nums) < 7:  # noqa: PLR2004
        msg = f"could not parse `lfs quota` output for {fs}:\n{text}"
        raise SystemExit(msg)

    def num(tok: str) -> int:
        try:
            return int(tok)
        except ValueError:
            return 0

    return {
        "filesystem": fs,
        "kbytes_used": num(nums[0]),
        "kbytes_limit": num(nums[2]) or num(nums[1]),
        "inodes_used": num(nums[4]),
        "inodes_limit": num(nums[6]) or num(nums[5]),
        "raw": text.strip(),
    }


def quota_guard(chunk: dict, args, dry: bool = False) -> None:
    """Refuse to stage a new chunk if the projected footprint would cross the
    configured ceiling. Runs BEFORE any fetch.

    The inode projection MUST track the staging mode. In pdfs mode a chunk
    creates one inode per document; in tars mode it creates one per archive --
    64 rather than ~74,000. Projecting the pdfs figure while running tars mode
    is not merely pessimistic: the campaign would project far more than
    45M ceiling and start refusing fetches around chunk 580, for inodes that are
    never created. That would reintroduce, as a false alarm, exactly the limit
    tar mode exists to remove.
    """
    q = lfs_quota()
    used_b = q["kbytes_used"] * 1024
    ceiling_b = int(args.quota_ceiling_tib * 2**40)
    projected_b = used_b + chunk["input_bytes"] + chunk["n_pdfs"] * OUTPUT_BYTES_PER_PDF
    inode_ceiling = args.inode_ceiling
    parquet_inodes = max(1, chunk["n_pdfs"] // 25)  # one output file per task
    if getattr(args, "stage_mode", "pdfs") == "tars":
        staged_inodes = len(chunk.get("tars", ())) or 1
    else:
        staged_inodes = chunk["n_pdfs"]
    projected_inodes = q["inodes_used"] + staged_inodes + parquet_inodes
    log(f"quota: used {used_b / 2**40:.2f} TiB / limit {q['kbytes_limit'] * 1024 / 2**40:.2f} TiB, "
        f"inodes {q['inodes_used']:,} / {q['inodes_limit']:,}")
    log(f"quota: projected after {cid(chunk['chunk_id'])} -> {projected_b / 2**40:.2f} TiB "
        f"(ceiling {ceiling_b / 2**40:.2f} TiB), inodes {projected_inodes:,} (ceiling {inode_ceiling:,})")
    if projected_b > ceiling_b:
        msg = (f"QUOTA GUARD: projected {projected_b / 2**40:.2f} TiB exceeds ceiling "
               f"{ceiling_b / 2**40:.2f} TiB -- refusing to start {cid(chunk['chunk_id'])}. "
               f"Reap finished chunks or raise --quota-ceiling-tib.")
        raise SystemExit(msg)
    if projected_inodes > inode_ceiling:
        msg = (f"QUOTA GUARD: projected {projected_inodes:,} inodes exceeds ceiling "
               f"{inode_ceiling:,} -- refusing to start {cid(chunk['chunk_id'])}.")
        raise SystemExit(msg)
    if dry:
        log("DRY-RUN quota guard passed")


# ==========================================================================
# PLAN
# ==========================================================================
def _tame_threads() -> None:
    """One thread per worker process.

    pyarrow sizes its CPU and IO pools from the core count. On a 144-core Grace
    node that is ~300 threads per worker; times N pool workers it exhausts
    RLIMIT_NPROC for the whole user and nothing can fork any more -- which is
    exactly how the first attempt at this scan took down a login node.
    """
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "ARROW_NUM_THREADS"):
        os.environ[var] = "1"
    try:
        import pyarrow

        pyarrow.set_cpu_count(1)
        pyarrow.set_io_thread_count(1)
    except Exception:  # noqa: BLE001, S110
        pass


def _scan_index_shard(path: str) -> tuple[str, str, int, int]:
    """(tar_key, index_path, n_pdfs, staged_bytes) for one index parquet.

    `staged_bytes` always comes from the FULL index, because that is what lands
    on Lustre: a whole tar holds every copy. `n_pdfs` and the returned index
    path come from the deduplicated index when one is configured, because that
    is what gets processed. Conflating the two silently under-projects the quota
    by the duplicate rate.
    """
    _tame_threads()
    import pyarrow.parquet as pq

    tbl = pq.read_table(path, columns=["tar_key", "member_size"])
    sizes = tbl.column("member_size").to_pylist()
    tar_keys = tbl.column("tar_key")
    tar_key = tar_keys[0].as_py() if tbl.num_rows else os.path.basename(path)
    staged_bytes = sum(s for s in sizes if s and s > 0)

    if DEDUP_INDEX_DIR:
        dpath = os.path.join(DEDUP_INDEX_DIR, os.path.relpath(path, INDEX_DIR))
        if not os.path.exists(dpath):
            # Every payload in this tar survives elsewhere: nothing to process,
            # but the archive is still staged if any sibling needs it.
            return tar_key, dpath, 0, staged_bytes
        dtbl = pq.read_table(dpath, columns=["member_size"])
        return tar_key, dpath, dtbl.num_rows, staged_bytes

    return tar_key, path, tbl.num_rows, staged_bytes


def tar_stats_cache(plan_dir: str, limit: int | None) -> str:
    """A limited scan must never be mistaken for the full one, so it gets its
    own cache file name."""
    return f"{plan_dir}/tar_stats.jsonl" if limit is None else f"{plan_dir}/tar_stats_first{limit}.jsonl"


def build_tar_stats(plan_dir: str, workers: int, force: bool = False,
                    limit: int | None = None) -> list[dict]:
    """One-time scan of the local index (the parquets, one per tar) to get
    exact per-tar PDF counts and payload bytes. Cached; the cache is what makes
    replanning cheap and identical.

    `limit` scans only the first N index shards. Index file paths sort in the
    same order as the tar_keys they contain (both are .../NNNN/records-NNNNNN-NNN),
    which is asserted below, so "first N files" and "first N sorted tars" are
    the same set -- that is what makes a small test plan cheap.
    """
    cache = tar_stats_cache(plan_dir, limit)
    if os.path.exists(cache) and not force:
        with open(cache) as fh:
            stats = [json.loads(line) for line in fh if line.strip()]
        log(f"tar stats: reusing cache {cache} ({len(stats):,} tars)")
        return stats

    files = sorted(glob.glob(f"{INDEX_DIR}/**/*.parquet", recursive=True))
    if not files:
        msg = f"no index parquets under {INDEX_DIR}"
        raise SystemExit(msg)
    if limit is not None:
        files = files[:limit]
    workers = clamp_workers(workers, "index scan")
    assert_nproc_headroom(workers, "index scan")
    log(f"tar stats: scanning {len(files):,} index shards with {workers} workers "
        f"(one-time, cached to {cache})")
    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (tar_key, idx_path, n, nbytes) in enumerate(ex.map(_scan_index_shard, files, chunksize=16)):
            rows.append({"tar_key": tar_key, "index_file": idx_path, "n_pdfs": n, "bytes": nbytes})
            if (i + 1) % 10000 == 0:
                log(f"  scanned {i + 1:,}/{len(files):,} ({time.time() - t0:.0f}s)")
    by_key = sorted(rows, key=lambda r: r["tar_key"])
    if [r["index_file"] for r in by_key] != [r["index_file"] for r in rows]:
        msg = ("index file path order does not match tar_key order; --limit-tars would not select "
               "the first N sorted tars. Refusing to write a misleading cache.")
        raise SystemExit(msg)
    rows = by_key
    seen = {r["tar_key"] for r in rows}
    if len(seen) != len(rows):
        msg = f"duplicate tar_key in index: {len(rows)} shards but {len(seen)} distinct tars"
        raise SystemExit(msg)
    os.makedirs(plan_dir, exist_ok=True)
    write_lines_atomic(cache, (json.dumps(r, sort_keys=True) for r in rows))
    log(f"tar stats: {len(rows):,} tars, {sum(r['n_pdfs'] for r in rows):,} PDFs, "
        f"{sum(r['bytes'] for r in rows) / 2**40:.2f} TiB in {time.time() - t0:.0f}s")
    return rows


def partition(rows: list[dict], target_bytes: int | None, tars_per_chunk: int | None,
              max_pdfs: int | None, chunk_id_base: int) -> list[dict]:
    """Deterministic greedy partition of the sorted tar list.

    Deterministic by construction: input is sorted by tar_key, the fill rule is
    a pure function of the running totals, and no randomness or filesystem
    ordering enters. Re-running with the same parameters and the same index
    yields byte-identical plan.jsonl.
    """
    chunks: list[dict] = []
    cur: list[dict] = []

    def flush() -> None:
        if not cur:
            return
        n_pdfs = sum(r["n_pdfs"] for r in cur)
        nbytes = sum(r["bytes"] for r in cur)
        capped = False
        if max_pdfs is not None and n_pdfs > max_pdfs:
            # Scale the byte estimate; the exact figure is recomputed at FETCH
            # from the actual record list.
            nbytes = int(nbytes * max_pdfs / n_pdfs)
            n_pdfs = max_pdfs
            capped = True
        chunks.append({
            "chunk_id": chunk_id_base + len(chunks),
            "tars": [r["tar_key"] for r in cur],
            "index_files": [r["index_file"] for r in cur],
            "n_pdfs": n_pdfs,
            "input_bytes": nbytes,
            "max_pdfs": max_pdfs,
            "partial_coverage": capped,
        })
        cur.clear()

    for row in rows:
        cur.append(row)
        if tars_per_chunk is not None:
            if len(cur) >= tars_per_chunk:
                flush()
        elif sum(r["bytes"] for r in cur) >= target_bytes:
            flush()
    flush()
    return chunks


def reexec_under_srun(args, extra: list[str]) -> None:
    """Re-run this process inside a cpu_datamover allocation.

    The first `plan` reads the index parquets; that is not login-node work.
    The child is given --in-process so it does the scan where it already is
    rather than recursing, and its worker count is clamped to the CPUs we ask
    Slurm for.
    """
    dry = ["--dry-run"] if "--dry-run" in extra else []
    extra = [e for e in extra if e != "--dry-run"]
    cpus = max(1, min(args.scan_workers, MAX_ALLOC_WORKERS))
    argv = ["srun", "--account", ACCOUNT, "--partition", DM_PARTITION, "--qos", DM_QOS,
            "--nodes", "1", "--ntasks", "1", "--cpus-per-task", str(cpus),
            "--time", args.scan_time, "--job-name", "plan-scan",
            sys.executable, os.path.abspath(__file__), *dry,
            "--in-process", "--plan-dir", args.plan_dir, "plan", *extra]
    log("not inside a Slurm allocation; re-running the index scan under srun")
    rc = stream_child(argv)
    raise SystemExit(rc)


def cmd_plan(args) -> None:
    plan_dir = args.plan_dir
    plan_path = f"{plan_dir}/plan.jsonl"
    meta_path = f"{plan_dir}/plan_meta.json"
    params = {
        "target_input_gib": args.target_input_gib,
        "tars_per_chunk": args.tars_per_chunk,
        "max_pdfs_per_chunk": args.max_pdfs_per_chunk,
        "chunk_id_base": args.chunk_id_base,
        "limit_tars": args.limit_tars,
        "index_dir": INDEX_DIR,
        "dedup_index_dir": DEDUP_INDEX_DIR or None,
    }

    if os.path.exists(plan_path) and not args.force:
        with open(meta_path) as fh:
            old = json.load(fh)
        if old.get("params") != params:
            msg = (f"{plan_path} already exists with different parameters.\n"
                   f"  existing: {json.dumps(old.get('params'), sort_keys=True)}\n"
                   f"  requested: {json.dumps(params, sort_keys=True)}\n"
                   f"A plan is written once. Use --plan-dir for a separate plan, or --force.")
            raise SystemExit(msg)
        log(f"plan already exists and parameters match: {plan_path} ({old['n_chunks']:,} chunks)")
        return

    if args.tars_per_chunk is None and args.target_input_gib is None:
        msg = "pass --target-input-gib or --tars-per-chunk"
        raise SystemExit(msg)

    needs_scan = args.rescan or not os.path.exists(tar_stats_cache(plan_dir, args.limit_tars))
    if needs_scan and not args.in_process and not in_allocation():
        extra = []
        if args.dry_run:
            # The scan only reads the index and writes a cache, but it is still
            # 67k Lustre reads -- not login-node work even for a dry run.
            extra.append("--dry-run")
        for flag, val in (("--target-input-gib", args.target_input_gib),
                          ("--tars-per-chunk", args.tars_per_chunk),
                          ("--max-pdfs-per-chunk", args.max_pdfs_per_chunk),
                          ("--chunk-id-base", args.chunk_id_base),
                          ("--limit-tars", args.limit_tars),
                          ("--scan-workers", args.scan_workers)):
            if val is not None:
                extra += [flag, str(val)]
        if args.rescan:
            extra.append("--rescan")
        if args.force:
            extra.append("--force")
        reexec_under_srun(args, extra)

    if needs_scan:
        scan_workers = clamp_workers(args.scan_workers, "index scan")
        local_execution_banner("the index scan", scan_workers)
    rows = build_tar_stats(plan_dir, args.scan_workers, force=args.rescan, limit=args.limit_tars)
    if args.limit_tars is not None:
        rows = rows[: args.limit_tars]
        log(f"plan: --limit-tars {args.limit_tars} -> partitioning only the first {len(rows):,} tars")
    target_bytes = int(args.target_input_gib * 2**30) if args.target_input_gib else None
    chunks = partition(rows, target_bytes, args.tars_per_chunk, args.max_pdfs_per_chunk, args.chunk_id_base)
    over = [c["chunk_id"] for c in chunks if c["chunk_id"] > 9999]  # noqa: PLR2004
    if over:
        msg = (f"plan would create {len(over)} chunks with ids above 9999 "
               f"(first {over[0]}), which the chunk_NNNN naming scheme cannot express. "
               f"Use a larger --target-input-gib/--tars-per-chunk, a smaller --chunk-id-base, "
               f"or --limit-tars.")
        raise SystemExit(msg)

    total_pdfs = sum(c["n_pdfs"] for c in chunks)
    total_bytes = sum(c["input_bytes"] for c in chunks)
    log(f"plan: {len(chunks):,} chunks, {total_pdfs:,} PDFs, {total_bytes / 2**40:.2f} TiB input")
    log(f"plan: median chunk {sorted(c['input_bytes'] for c in chunks)[len(chunks) // 2] / 2**30:.1f} GiB, "
        f"shards/chunk ~{-(-max(c['n_pdfs'] for c in chunks) // PDFS_PER_SHARD)}")
    if args.max_pdfs_per_chunk is not None:
        log("!! PARTIAL COVERAGE: --max-pdfs-per-chunk is set, so each chunk covers only the "
            "first N record_ids of its tars. This is a TEST setting; do NOT use it for production.")

    if args.dry_run:
        for c in chunks[:5]:
            log(f"DRY-RUN {cid(c['chunk_id'])}: {len(c['tars'])} tars, {c['n_pdfs']:,} PDFs, "
                f"{c['input_bytes'] / 2**30:.1f} GiB")
        log(f"DRY-RUN would write {plan_path}")
        return

    os.makedirs(plan_dir, exist_ok=True)
    write_lines_atomic(plan_path, (json.dumps(c, sort_keys=True) for c in chunks))
    with open(plan_path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    write_json_atomic(meta_path, {
        "utc": utcnow(),
        "params": params,
        "n_chunks": len(chunks),
        "n_tars": len(rows),
        "total_pdfs": total_pdfs,
        "total_input_bytes": total_bytes,
        "plan_sha256": digest,
        "pdfs_per_shard": PDFS_PER_SHARD,
    })
    log(f"wrote {plan_path} ({len(chunks):,} chunks) sha256={digest}")


def load_plan(plan_dir: str) -> dict[int, dict]:
    plan_path = f"{plan_dir}/plan.jsonl"
    if not os.path.exists(plan_path):
        msg = f"no plan at {plan_path} -- run `plan` first"
        raise SystemExit(msg)
    out = {}
    with open(plan_path) as fh:
        for line in fh:
            if line.strip():
                c = json.loads(line)
                out[c["chunk_id"]] = c
    return out


# ==========================================================================
# FETCH
# ==========================================================================
def _read_records(path: str) -> list[tuple]:
    _tame_threads()
    import pyarrow.parquet as pq

    tbl = pq.read_table(path, columns=["record_id", "tar_key", "member_offset",
                                       "member_size", "payload_sha256"]).to_pydict()
    return list(zip(tbl["record_id"], tbl["tar_key"], tbl["member_offset"],
                    tbl["member_size"], tbl["payload_sha256"], strict=True))


def chunk_records(chunk: dict, workers: int) -> list[tuple]:
    """Deterministic record list for a chunk: every record of every tar in the
    chunk, sorted by record_id, truncated to max_pdfs if the plan sets one."""
    recs: list[tuple] = []
    files = chunk["index_files"]
    if len(files) == 1:
        recs = _read_records(files[0])
    else:
        n = clamp_workers(min(workers, len(files)), "index record read")
        assert_nproc_headroom(n, "index record read")
        with ProcessPoolExecutor(max_workers=n) as ex:
            for part in ex.map(_read_records, files):
                recs.extend(part)
    recs.sort(key=lambda r: r[0])
    if chunk.get("max_pdfs") is not None:
        recs = recs[: chunk["max_pdfs"]]
    return recs


def cmd_fetch_worker(args) -> None:
    """The real fetcher. Runs inside the srun allocation `fetch` creates.

    Idempotent: an existing file whose size matches the index is kept, so a
    restart only refetches what is missing. Writes are atomic (.part+rename),
    so a kill never leaves a short file that a later run would accept.
    """
    if not in_allocation() and not args.in_process:
        msg = ("REFUSING to run fetch-worker outside a Slurm allocation. It opens up to "
               f"{args.threads} network threads and writes tens of GiB; on the login node that is "
               "the RLIMIT_NPROC wedge. Use `fetch` (which sruns this), or pass --in-process if you "
               "really mean it.")
        raise SystemExit(msg)
    if not in_allocation():
        local_execution_banner("the fetch", clamp_workers(args.threads, "fetch"))
    plan = load_plan(args.plan_dir)
    chunk = plan[args.chunk]
    if getattr(args, "stage_mode", "pdfs") == "tars":
        fetch_tars_worker(args, chunk)
        return
    sd, gd = state_dir(args.chunk), stage_dir(args.chunk)
    pdfs = f"{gd}/pdfs"
    os.makedirs(pdfs, exist_ok=True)
    os.makedirs(sd, exist_ok=True)

    t0 = time.time()
    recs = chunk_records(chunk, args.index_workers)
    log(f"{cid(args.chunk)}: {len(recs):,} records from {len(chunk['index_files'])} index shards "
        f"({time.time() - t0:.1f}s)")
    write_lines_atomic(f"{gd}/records.jsonl", (
        json.dumps({"record_id": r[0], "tar_key": r[1], "member_offset": r[2],
                    "member_size": r[3], "payload_sha256": r[4]}, sort_keys=True) for r in recs))

    stats = {"ok": 0, "resumed": 0, "failed": 0, "zero_size": 0, "size_mismatch": 0,
             "sha_mismatch": 0, "not_pdf": 0, "bytes": 0}
    failures: list[dict] = []
    lock = threading.Lock()
    client = s3_client()
    verify_sha = not args.no_sha

    def fetch_one(rec) -> str | None:
        rid, tar_key, off, size, sha = rec
        dest = f"{pdfs}/{rid}.pdf"
        if size is None or size <= 0:
            with lock:
                stats["zero_size"] += 1
            return None
        try:
            if os.path.getsize(dest) == size:
                with lock:
                    stats["resumed"] += 1
                    stats["bytes"] += size
                return rid
        except OSError:
            pass
        key = SOURCE_PREFIX + tar_key
        body = None
        for attempt in range(3):
            try:
                body = client.get_object(Bucket=BUCKET, Key=key,
                                         Range=f"bytes={off}-{off + size - 1}")["Body"].read()
                break
            except Exception as e:  # noqa: BLE001
                err = repr(e)
                if attempt == 2:  # noqa: PLR2004
                    with lock:
                        stats["failed"] += 1
                        if len(failures) < 200:  # noqa: PLR2004
                            failures.append({"record_id": rid, "tar_key": tar_key, "error": err[:300]})
                    return None
                time.sleep(1.5 * (attempt + 1))
        if len(body) != size:
            with lock:
                stats["size_mismatch"] += 1
                stats["failed"] += 1
                if len(failures) < 200:  # noqa: PLR2004
                    failures.append({"record_id": rid, "reason": "size", "want": size, "got": len(body)})
            return None
        if verify_sha and sha:
            got = hashlib.sha256(body).hexdigest()
            if got != sha:
                with lock:
                    stats["sha_mismatch"] += 1
                    stats["failed"] += 1
                    if len(failures) < 200:  # noqa: PLR2004
                        failures.append({"record_id": rid, "reason": "sha256"})
                return None
        tmp = dest + ".part"
        with open(tmp, "wb") as fh:
            fh.write(body)
        os.replace(tmp, dest)
        with lock:
            stats["ok"] += 1
            stats["bytes"] += len(body)
            if body[:4] != b"%PDF":
                stats["not_pdf"] += 1
            done = stats["ok"] + stats["resumed"]
            if done % 5000 == 0:
                el = time.time() - t1
                log(f"  {done:,}/{len(recs):,}  {stats['bytes'] / 2**30:.2f} GiB  "
                    f"{stats['bytes'] / 2**20 / max(el, 1e-6):.0f} MiB/s")
        return rid

    t1 = time.time()
    got: list[str] = []
    fetch_threads = args.threads if in_allocation() else clamp_workers(args.threads, "fetch")
    assert_nproc_headroom(fetch_threads, "fetch")
    with ThreadPoolExecutor(max_workers=fetch_threads) as ex:
        for rid in ex.map(fetch_one, recs):
            if rid is not None:
                got.append(rid)
    elapsed = time.time() - t1

    got.sort()
    n_manifest = write_lines_atomic(f"{gd}/manifest.jsonl",
                                    (json.dumps({"file_name": f"{r}.pdf"}) for r in got))
    on_disk = len(glob.glob(f"{pdfs}/*.pdf"))
    log(f"{cid(args.chunk)} fetch: ok={stats['ok']:,} resumed={stats['resumed']:,} "
        f"failed={stats['failed']:,} zero_size={stats['zero_size']:,} "
        f"{stats['bytes'] / 2**30:.2f} GiB in {elapsed:.0f}s "
        f"({stats['bytes'] / 2**20 / max(elapsed, 1e-6):.0f} MiB/s); manifest={n_manifest:,} on_disk={on_disk:,}")
    if failures:
        write_json_atomic(f"{sd}/fetch_failures.json", failures)

    tolerated = int(len(recs) * args.fetch_failure_tolerance)
    if stats["failed"] > tolerated:
        msg = (f"FETCH FAILED for {cid(args.chunk)}: {stats['failed']:,} of {len(recs):,} records could not be "
               f"fetched/verified (tolerance {tolerated:,}). Data left in place; see {sd}/fetch_failures.json. "
               f"Rerun to retry only the missing ones, or raise --fetch-failure-tolerance.")
        raise SystemExit(msg)
    if n_manifest == 0:
        msg = f"FETCH FAILED for {cid(args.chunk)}: manifest is empty"
        raise SystemExit(msg)

    write_marker(args.chunk, "_FETCHED", {
        "records_planned": len(recs),
        "fetched": stats["ok"],
        "resumed": stats["resumed"],
        "failed": stats["failed"],
        "zero_size_in_index": stats["zero_size"],
        "size_mismatch": stats["size_mismatch"],
        "sha_mismatch": stats["sha_mismatch"],
        "non_pdf_magic": stats["not_pdf"],
        "manifest_entries": n_manifest,
        "pdfs_on_disk": on_disk,
        "bytes": stats["bytes"],
        "sha256_verified": verify_sha,
        "elapsed_s": round(elapsed, 1),
        "tars": len(chunk["tars"]),
        "partial_coverage": bool(chunk.get("partial_coverage")),
    })


def fetch_tars_worker(args, chunk: dict) -> None:
    """Stage whole tar archives instead of one file per PDF.

    A chunk is 64 CONTIGUOUS source tars and takes every record in them, so the
    per-record path issues ~130,000 range-GETs to reassemble bytes that are 64
    whole objects. Downloading the objects costs the same bytes, ~2,000x fewer
    requests, and -- the reason this exists -- 64 inodes instead of 130,000.

    Inodes, not bytes, are what bound this corpus: the corpus documents against a
    <inode quota> inode quota is 2.5x over before a single parquet is written. Tar
    staging turns that into ~67,000 inodes for the whole corpus.

    Deduplication then happens LOCALLY. The manifest is built from the chunk's
    index shards -- which after re-planning are the deduplicated ones -- so the
    archives on disk hold every copy, while the only records handed to the GPU
    are the survivors. Nothing about the download depends on the dedup, which
    means the keep-list can be revised without re-fetching anything.
    """
    sd, gd = state_dir(args.chunk), stage_dir(args.chunk)
    tars_dir = f"{gd}/tars"
    os.makedirs(tars_dir, exist_ok=True)
    os.makedirs(sd, exist_ok=True)

    client = s3_client()
    stats = {"downloaded": 0, "resumed": 0, "failed": 0, "bytes": 0}
    failures: list[dict] = []
    lock = threading.Lock()

    def one_tar(tar_key: str) -> str | None:
        dest = os.path.join(tars_dir, tar_key)
        key = SOURCE_PREFIX + tar_key
        try:
            head = client.head_object(Bucket=BUCKET, Key=key)
            want = head["ContentLength"]
        except Exception as e:  # noqa: BLE001
            with lock:
                stats["failed"] += 1
                failures.append({"tar_key": tar_key, "stage": "head", "error": repr(e)})
            return None
        # Resume: a complete archive is one whose size matches the object.
        # Partial files are re-downloaded rather than appended to -- a truncated
        # tar yields silently wrong byte ranges, which is the worst outcome here.
        try:
            if os.path.getsize(dest) == want:
                with lock:
                    stats["resumed"] += 1
                    stats["bytes"] += want
                return tar_key
        except OSError:
            pass
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        tmp = f"{dest}.part"
        for attempt in range(3):
            try:
                client.download_file(BUCKET, key, tmp)
                got = os.path.getsize(tmp)
                if got != want:
                    msg = f"short download {got} != {want}"
                    raise OSError(msg)  # noqa: TRY301
                os.replace(tmp, dest)
                with lock:
                    stats["downloaded"] += 1
                    stats["bytes"] += want
                return tar_key
            except Exception as e:  # noqa: BLE001
                if attempt == 2:  # noqa: PLR2004
                    with contextlib.suppress(OSError):
                        os.unlink(tmp)
                    with lock:
                        stats["failed"] += 1
                        failures.append({"tar_key": tar_key, "stage": "download", "error": repr(e)})
                    return None
                time.sleep(2 ** attempt)
        return None

    tars = chunk["tars"]
    log(f"{cid(args.chunk)}: staging {len(tars)} tar archive(s) -> {tars_dir}")
    t1 = time.time()
    workers = min(args.threads, 16)  # 4 GiB objects; concurrency is bandwidth, not IOPS
    assert_nproc_headroom(workers, "fetch-tars")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one_tar, tars))
    elapsed = time.time() - t1

    if stats["failed"]:
        write_json_atomic(f"{sd}/fetch_failures.json", failures)
        msg = (f"FETCH FAILED for {cid(args.chunk)}: {stats['failed']} of {len(tars)} archives could not "
               f"be staged. Data left in place; see {sd}/fetch_failures.json. Rerun to retry only the "
               f"missing ones -- complete archives are detected by size and skipped.")
        raise SystemExit(msg)

    # Manifest from the chunk's index shards: these are the records to PROCESS,
    # which post-dedup is a subset of what the archives physically contain.
    recs = chunk_records(chunk, args.index_workers)
    n_manifest = write_lines_atomic(f"{gd}/manifest.jsonl", (
        json.dumps({"file_name": f"{r[0]}.pdf", "tar_file": r[1],
                    "byte_offset": r[2], "size": r[3]}, sort_keys=True) for r in recs))
    if n_manifest == 0:
        msg = f"FETCH FAILED for {cid(args.chunk)}: manifest is empty"
        raise SystemExit(msg)

    log(f"{cid(args.chunk)} fetch: {stats['downloaded']} downloaded, {stats['resumed']} resumed, "
        f"{stats['bytes'] / 2**30:.2f} GiB in {elapsed:.0f}s "
        f"({stats['bytes'] / 2**20 / max(elapsed, 1e-6):.0f} MiB/s); manifest={n_manifest:,} records "
        f"across {len(tars)} archive(s) = {len(tars)} inodes")

    write_marker(args.chunk, "_FETCHED", {
        "stage_mode": "tars",
        "tars": len(tars),
        "tars_downloaded": stats["downloaded"],
        "tars_resumed": stats["resumed"],
        "bytes": stats["bytes"],
        "manifest_entries": n_manifest,
        "records_planned": len(recs),
        "inodes_used": len(tars),
        "elapsed_s": round(elapsed, 1),
        "partial_coverage": bool(chunk.get("partial_coverage")),
    })


def cmd_fetch(args) -> None:
    """Driver side: quota-guard, then run the fetch worker under srun on the
    data-movement partition (never on the login node)."""
    plan = load_plan(args.plan_dir)
    chunk = plan[args.chunk]
    if has_marker(args.chunk, "_FETCHED"):
        log(f"{cid(args.chunk)}: _FETCHED present, skipping")
        return
    if has_marker(args.chunk, "_REAPED"):
        msg = f"{cid(args.chunk)} is already _REAPED; refusing to re-stage it"
        raise SystemExit(msg)

    quota_guard(chunk, args, dry=args.dry_run)

    sub_argv = [
        "fetch-worker",
        "--chunk", str(args.chunk),
        "--threads", str(args.threads), "--index-workers", str(args.index_workers),
        "--fetch-failure-tolerance", str(args.fetch_failure_tolerance),
        "--stage-mode", getattr(args, "stage_mode", "pdfs"),
    ]
    if args.no_sha:
        sub_argv.append("--no-sha")
    if args.dry_run:
        log("DRY-RUN would srun: " + " ".join(sub_argv))
        if getattr(args, "stage_mode", "pdfs") == "tars":
            log(f"DRY-RUN would stage {len(chunk['tars'])} tar archive(s) "
                f"({chunk['input_bytes'] / 2**30:.1f} GiB, {len(chunk['tars'])} inodes) into "
                f"{stage_dir(args.chunk)}/tars and process {chunk['n_pdfs']:,} deduplicated records")
            return
        log(f"DRY-RUN would stage {chunk['n_pdfs']:,} PDFs "
            f"({chunk['input_bytes'] / 2**30:.1f} GiB) into {stage_dir(args.chunk)}/pdfs")
        return

    os.makedirs(state_dir(args.chunk), exist_ok=True)
    rc = srun_self(args, sub_argv, f"fetch-{cid(args.chunk)}",
                   cpus=args.fetch_cpus, walltime=args.fetch_time,
                   logfile=f"{state_dir(args.chunk)}/fetch.log")
    if rc != 0 or not has_marker(args.chunk, "_FETCHED"):
        msg = f"fetch of {cid(args.chunk)} failed (rc={rc}); see {state_dir(args.chunk)}/fetch.log"
        raise SystemExit(msg)


# ==========================================================================
# PROCESS
# ==========================================================================
def make_chunk_yaml(chunk_id: int, dry: bool = False) -> str:
    """Derive a per-chunk config from the committed production YAML.

    The production file is never edited: it is loaded, four paths are
    retargeted at this chunk, and the result is written into the chunk's own
    state directory. Every inference/tuning argument is inherited verbatim, so
    the per-chunk run is the production run.
    """
    import yaml

    with open(PRODUCTION_YAML) as fh:
        cfg = yaml.safe_load(fh)

    sd, gd = state_dir(chunk_id), stage_dir(chunk_id)
    retarget = {
        "results_path": f"{sd}/benchmark_results",
        "output_path": f"{out_dir(chunk_id)}/shard_${{SHARD_INDEX_PADDED}}",
        "checkpoint_path": f"{sd}/checkpoint",
    }
    names = {p["name"] for p in cfg["paths"]}
    missing = set(retarget) - names
    if missing:
        msg = f"{PRODUCTION_YAML} has no paths entries named {sorted(missing)}"
        raise SystemExit(msg)
    for p in cfg["paths"]:
        if p["name"] in retarget:
            p["host_path"] = retarget[p["name"]]

    hit = False
    for ds in cfg["datasets"]:
        if ds["name"] == "nemotron_parse_pdf_production":
            for fmt in ds["formats"]:
                if fmt["type"] == "manifest":
                    fmt["path"] = f"{gd}/manifest.jsonl"
                    hit = True
                elif fmt["type"] == "pdf_dir":
                    fmt["path"] = f"{gd}/pdfs"
                elif fmt["type"] == "tar_dir":
                    # Tar staging: the config's --tar-base-dir resolves here and
                    # the manifest's tar_file entries are relative to it.
                    fmt["path"] = f"{gd}/tars"
    if not hit:
        msg = f"{PRODUCTION_YAML} has no dataset nemotron_parse_pdf_production with a manifest format"
        raise SystemExit(msg)

    path = f"{sd}/config.yaml"
    header = (f"# GENERATED by chunk_orchestrator.py for {cid(chunk_id)} at {utcnow()}.\n"
              f"# Derived from {PRODUCTION_YAML}; do not edit by hand -- it is regenerated.\n"
              f"# Only results_path/output_path/checkpoint_path and the dataset paths differ.\n")
    if dry:
        log(f"DRY-RUN would write {path} (manifest={gd}/manifest.jsonl, output={retarget['output_path']})")
        return path
    os.makedirs(sd, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        fh.write(header)
        yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=False, width=100000)
    os.replace(tmp, path)
    return path


def slurm_array_retry_plan(checkpoint_path: str) -> dict | None:
    """Which logical shards have no completion manifest.

    Mirrors nemo_curator.backends.slurm_array.find_slurm_array_retries /
    build_slurm_array_retry_submissions, which tutorials/slurm/retry_array.py
    wraps. It is reimplemented here (over the same on-disk JSON) only because
    importing nemo_curator needs cosmos_xenna, i.e. the container -- and the
    orchestrator polls this every minute from outside it. `retry-check` runs
    the real script in the container to confirm the two agree.
    """
    cdir = os.path.join(checkpoint_path, ".nemo_curator_metadata", ".slurm_array_completion")
    run_json = os.path.join(cdir, "run.json")
    if not os.path.isfile(run_json):
        return None
    with open(run_json) as fh:
        cfgp = json.load(fh)
    total = int(cfgp["total_shards"])
    minimum = int(cfgp["minimum_shard_index"])
    done = set()
    for f in sorted(glob.glob(os.path.join(cdir, "completed_slurm_array_*.json"))):
        with open(f) as fh:
            payload = json.load(fh)
        if payload.get("status") != "completed":
            msg = f"completion manifest {f} has status {payload.get('status')!r}"
            raise SystemExit(msg)
        if int(payload["total_shards"]) != total or int(payload["minimum_shard_index"]) != minimum:
            msg = f"completion manifest {f} does not match run.json"
            raise SystemExit(msg)
        done.add(int(payload["shard_index"]))
    expected = set(range(minimum, minimum + total))
    return {"missing": sorted(expected - done), "total_shards": total,
            "minimum_shard_index": minimum, "completed": len(done & expected)}


def format_array_indices(indices) -> str:
    """Compact Slurm --array expression; mirrors format_slurm_array_indices."""
    idx = sorted(set(indices))
    if not idx:
        return ""
    parts, start, end = [], idx[0], idx[0]
    for i in idx[1:]:
        if i == end + 1:
            end = i
            continue
        parts.append(str(start) if start == end else f"{start}-{end}")
        start = end = i
    parts.append(str(start) if start == end else f"{start}-{end}")
    return ",".join(parts)


STEP_STALL_LIMIT = 20  # srun retries ~every 5s, so ~100s of proof
SQUEUE_TRIES = 5
SQUEUE_RETRY_S = 15
SQUEUE_TIMEOUT_S = 120
DRAIN_CONFIRMATIONS = 2
SACCT_CONFIRM_TRIES = 3


# `squeue -j <id>` exits 1 for a job that has LEFT THE QUEUE exactly as it does
# for a job id that never existed -- both say "Invalid job id specified". So the
# return code cannot distinguish "finished" from "slurm is broken"; only the
# error text can. Verified on this cluster: finished array 721182 and bogus id
# 999999999 both give rc=1 with that message, while a live job gives rc=0.
# Anything else on stderr ("Socket timed out on send/recv operation", "Unable to
# contact slurm controller") is a transport failure and must NOT be read as
# "the job is done".
SQUEUE_ABSENT_MARKERS = ("invalid job id", "invalid user id")


def _squeue_lines(argv: list[str]) -> list[str] | None:
    """Run an squeue query, retrying transient failures.

    Returns the non-empty output lines ([] meaning "slurm says this is not in
    the queue"), or None if squeue could not be made to answer at all. None is
    NOT the same as [] and callers must not conflate them: the caller resubmits
    an 8-node GPU array on [].
    """
    last = ""
    for attempt in range(1, SQUEUE_TRIES + 1):
        proc = run_cmd(argv, check=False, timeout=SQUEUE_TIMEOUT_S)
        if proc.returncode == 0:
            return [ln for ln in (proc.stdout or "").splitlines() if ln.strip()]
        err = (proc.stderr or "").strip()
        if any(m in err.lower() for m in SQUEUE_ABSENT_MARKERS):
            return []  # definitive: slurm answered, and the job is not queued
        last = err.splitlines()[-1] if err else f"rc={proc.returncode}"
        if attempt < SQUEUE_TRIES:
            log(f"squeue failed ({last}); retry {attempt}/{SQUEUE_TRIES - 1} in {SQUEUE_RETRY_S}s")
            time.sleep(SQUEUE_RETRY_S)
    log(f"squeue UNAVAILABLE after {SQUEUE_TRIES} tries: {last}")
    return None


def squeue_alive(job_id: str) -> list[str] | None:
    """Queue entries for a job. [] = not queued, None = squeue did not answer."""
    try:
        return _squeue_lines(["squeue", "-j", str(job_id), "-h", "-o", "%i %T"])
    except subprocess.TimeoutExpired:
        log(f"squeue -j {job_id} timed out after {SQUEUE_TIMEOUT_S}s")
        return None


def squeue_by_name(job_name: str) -> list[str] | None:
    """Array job ids currently queued/running under a given --job-name.

    Belt and braces for the resume path: if the orchestrator is killed in the
    window between `sbatch` returning and process.json being written, the job
    id is not in any file, and resubmitting would double-run the shard. The
    job name is derived from the chunk, so it is enough to find it again.

    Returns None if squeue did not answer -- the caller must not read that as
    "no orphans", because that is precisely the double-submit this exists to
    prevent.
    """
    try:
        lines = _squeue_lines(["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%F %j %T"])
    except subprocess.TimeoutExpired:
        log(f"squeue -u timed out after {SQUEUE_TIMEOUT_S}s")
        return None
    if lines is None:
        return None
    ids = []
    for ln in lines:
        parts = ln.split()
        if len(parts) >= 2 and parts[1] == job_name and parts[0] not in ids:  # noqa: PLR2004
            ids.append(parts[0])
    return ids


# States sacct reports for work that has not finished. Anything else -- and
# sacct keeps rows after squeue has dropped them -- means the task is over.
SACCT_LIVE_STATES = frozenset({
    "PENDING", "RUNNING", "SUSPENDED", "COMPLETING", "CONFIGURING",
    "REQUEUED", "REQUEUE_HOLD", "REQUEUE_FED", "RESIZING", "SIGNALING",
    "STAGE_OUT", "STOPPED", "RESV_DEL_HOLD",
})


def sacct_live_tasks(job_id: str) -> list[str] | None:
    """Array tasks of job_id that sacct still considers unfinished.

    The second opinion behind squeue. squeue drops a job the moment it leaves
    the queue, so "absent from squeue" is ambiguous between finished and
    never-answered; sacct retains the row with a terminal State, so it can tell
    the two apart. Returns None if sacct did not answer either.
    """
    try:
        proc = run_cmd(["sacct", "-j", str(job_id), "-X", "--parsable2", "-n", "-o", "JobID,State"],
                       check=False, timeout=SQUEUE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        log(f"sacct -j {job_id} timed out after {SQUEUE_TIMEOUT_S}s")
        return None
    if proc.returncode != 0:
        log(f"sacct -j {job_id} failed: rc={proc.returncode}")
        return None
    rows = [ln for ln in (proc.stdout or "").splitlines() if ln.strip()]
    if not rows:
        # No accounting rows at all: the job is unknown to slurmdbd. Treat as
        # unknown rather than finished -- a purged or not-yet-flushed record
        # must not be read as a completed one.
        return None
    live = []
    for ln in rows:
        parts = ln.split("|")
        if len(parts) < 2:  # noqa: PLR2004
            continue
        # "CANCELLED by 12345" -> "CANCELLED"
        if parts[1].split()[0].upper() in SACCT_LIVE_STATES:
            live.append(parts[0])
    return live


def sacct_summary(job_id: str) -> list[str]:
    proc = run_cmd(["sacct", "-j", str(job_id), "-X", "--parsable2", "-n",
                    "-o", "JobID,State,ExitCode,Elapsed"], check=False)
    return [ln for ln in (proc.stdout or "").splitlines() if ln.strip()]


def count_outputs(chunk_id: int) -> tuple[int, int]:
    n, nbytes = 0, 0
    for root, _dirs, files in os.walk(out_dir(chunk_id)):
        for f in files:
            if f.endswith(".parquet"):
                n += 1
                nbytes += os.path.getsize(os.path.join(root, f))
    return n, nbytes


def cmd_process(args) -> None:
    plan = load_plan(args.plan_dir)
    chunk = plan[args.chunk]
    sd = state_dir(args.chunk)
    if has_marker(args.chunk, "_PROCESSED"):
        log(f"{cid(args.chunk)}: _PROCESSED present, skipping")
        return
    if not has_marker(args.chunk, "_FETCHED"):
        msg = f"{cid(args.chunk)}: refusing to process without _FETCHED"
        raise SystemExit(msg)

    fetched = read_marker(args.chunk, "_FETCHED")
    n_pdfs = int(fetched.get("manifest_entries") or chunk["n_pdfs"])
    shards = max(1, -(-n_pdfs // args.pdfs_per_shard))
    cfg_path = make_chunk_yaml(args.chunk, dry=args.dry_run)
    logdir = f"{LOG_ROOT}/{cid(args.chunk)}"
    ckpt = f"{sd}/checkpoint"
    state_file = f"{sd}/process.json"

    log(f"{cid(args.chunk)}: {n_pdfs:,} PDFs -> {shards} shard(s) "
        f"(@{args.pdfs_per_shard:,} PDFs/shard)")

    def submit(array_expr: str, offset: int, minimum: int, total: int) -> str:
        # Strip the SUBMITTER's allocation out of the child's environment.
        # sbatch inherits os.environ, so when the orchestrator itself runs under
        # srun -- which it now does, to keep work off the login node -- its
        # SLURM_* vars are handed to the array job. The array's internal srun
        # then sees SLURM_CPUS_PER_TASK from our allocation alongside
        # SLURM_TRES_PER_TASK from its own and dies in 13s with
        # "cpus-per-task set by two different environment variables".
        #
        # That failure is fast and self-inflicted: the retry loop resubmitted
        # three identical arrays in four minutes before anyone could look. The
        # launcher derives everything it needs from its OWN allocation, so
        # dropping these is not just safe, it is the only correct thing.
        # SLURM_CONF is the exception and must survive: it names this cluster's
        # slurm.conf (/cm/shared/apps/slurm/etc/oci-aga-slurm-1/slurm.conf).
        # Without it sbatch falls back to a default config whose cli_filter/lua
        # plugin references /etc/slurm/cli_filter.lua, which exists on NO node
        # here, and dies before submitting anything. Verified on cpu-0014: full
        # env submits, all-SLURM_*-stripped fails on cli_filter, stripped-but-
        # SLURM_CONF-kept submits.
        env = {k: v for k, v in os.environ.items()
               if k == "SLURM_CONF" or not k.startswith(("SLURM_", "SLURMD_", "SRUN_"))}
        env.update({
            "CONFIG": cfg_path,
            "TOTAL_SHARDS": str(total),
            "SESSION_NAME": f"nemotron-parse-{cid(args.chunk)}",
            "SHARD_INDEX_OFFSET": str(offset),
            "MINIMUM_SHARD_INDEX": str(minimum),
            "SBATCH_ACCOUNT": ACCOUNT,
            # Pin the image explicitly so a chunk's shards and its retries all
            # run the same container even if the default is rebuilt mid-run.
            "CONTAINER_IMAGE": args.container_image,
        })
        argv = ["sbatch", "--parsable",
                f"--array={array_expr}",
                f"--job-name=parse-{cid(args.chunk)}",
                f"--output={logdir}/parse_%A_%a.log",
                LAUNCHER]
        if args.dry_run:
            log("DRY-RUN would submit: CONFIG=%s TOTAL_SHARDS=%s SESSION_NAME=%s "
                "SHARD_INDEX_OFFSET=%s MINIMUM_SHARD_INDEX=%s %s"
                % (cfg_path, total, env["SESSION_NAME"], offset, minimum, " ".join(argv)))
            return "DRYRUN"
        os.makedirs(logdir, exist_ok=True)
        proc = run_cmd(argv, env=env)
        job_id = (proc.stdout or "").strip().split(";")[0]
        if not job_id.isdigit():
            msg = f"could not parse sbatch job id from {proc.stdout!r}"
            raise SystemExit(msg)
        rec = {"utc": utcnow(), "job_id": job_id, "array": array_expr, "offset": offset,
               "minimum": minimum, "total_shards": total, "config": cfg_path}
        hist = json.load(open(state_file)) if os.path.exists(state_file) else []  # noqa: SIM115
        hist.append(rec)
        write_json_atomic(state_file, hist)
        log(f"{cid(args.chunk)}: submitted array job {job_id} ({array_expr}) "
            f"TOTAL_SHARDS={total} offset={offset}")
        return job_id

    if args.dry_run:
        submit(f"0-{shards - 1}" if shards > 1 else "0", 0, 0, shards)
        log(f"DRY-RUN would wait for {shards} shard(s), then write _PROCESSED")
        return

    os.makedirs(ckpt, exist_ok=True)
    hist = json.load(open(state_file)) if os.path.exists(state_file) else []  # noqa: SIM115

    # Resume: adopt an already-submitted array rather than submitting again.
    active = None
    if hist:
        last = hist[-1]
        alive = squeue_alive(last["job_id"])
        if alive is None:
            msg = (f"{cid(args.chunk)}: squeue will not say whether array job {last['job_id']} is still "
                   f"running, and resubmitting on a guess would double-run its shards on {shards} nodes. "
                   f"Nothing submitted, nothing deleted; retry when slurm is answering.")
            raise SystemExit(msg)
        if alive:
            active = last["job_id"]
            log(f"{cid(args.chunk)}: adopting in-flight array job {active} (not resubmitting)")
    if active is None:
        # Nothing recorded, but a job with this chunk's name may still be in the
        # queue -- e.g. we were killed between sbatch and the state write.
        known = {h["job_id"] for h in hist}
        by_name = squeue_by_name(f"parse-{cid(args.chunk)}")
        if by_name is None:
            msg = (f"{cid(args.chunk)}: squeue will not answer, so the orphan check that stops this "
                   f"chunk being submitted twice cannot run. Nothing submitted, nothing deleted.")
            raise SystemExit(msg)
        orphans = [j for j in by_name if j not in known]
        if orphans:
            active = orphans[-1]
            log(f"{cid(args.chunk)}: adopting UNRECORDED in-flight array job {active} found by job name "
                f"(orchestrator was probably killed just after sbatch); recording it now")
            hist.append({"utc": utcnow(), "job_id": active, "array": "adopted", "offset": 0,
                         "minimum": 0, "total_shards": shards, "config": cfg_path,
                         "adopted_by_name": True})
            write_json_atomic(state_file, hist)


    attempts = len(hist)
    while True:
        retry = slurm_array_retry_plan(ckpt)
        if active is None:
            if retry is None:
                if attempts >= args.max_process_attempts:
                    msg = f"{cid(args.chunk)}: exhausted {attempts} process attempts with no checkpoint state"
                    raise SystemExit(msg)
                active = submit(f"0-{shards - 1}" if shards > 1 else "0", 0, 0, shards)
                attempts += 1
            elif retry["missing"]:
                if attempts >= args.max_process_attempts:
                    msg = (f"{cid(args.chunk)}: {len(retry['missing'])} shard(s) still incomplete after "
                           f"{attempts} attempts: {format_array_indices(retry['missing'])}. Stopping; "
                           f"nothing deleted.")
                    raise SystemExit(msg)
                log(f"{cid(args.chunk)}: {len(retry['missing'])} incomplete shard(s) -> "
                    f"{format_array_indices(retry['missing'])}")
                subs: dict[int, list[int]] = {}
                cap = args.max_array_size
                for s in retry["missing"]:
                    off = (s // cap) * cap
                    subs.setdefault(off, []).append(s - off)
                last_job = None
                for off, idxs in sorted(subs.items()):
                    last_job = submit(format_array_indices(idxs), off,
                                      retry["minimum_shard_index"], retry["total_shards"])
                active = last_job
                attempts += 1
            else:
                break

        # Concluding "drained" too early is expensive and silent: the code below
        # rebuilds the retry plan from on-disk manifests, finds the shards the
        # still-running array has not written yet, and resubmits them -- so two
        # arrays write the same output paths on 8 GPU nodes each. Over ~1,049
        # chunks this loop polls ~200,000 times, so a transient slurm failure is
        # not hypothetical. Require squeue to say "gone" twice AND sacct to agree
        # the tasks reached a terminal state before believing it.
        deadline = time.time() + args.process_timeout_s
        gone = 0
        sacct_mute = 0
        while True:
            alive = squeue_alive(active)
            if alive is None:
                # Unknown, not finished. Keep waiting; the deadline still applies.
                gone = 0
            elif alive:
                gone = 0
            else:
                gone += 1
                if gone >= DRAIN_CONFIRMATIONS:
                    live = sacct_live_tasks(active)
                    if live is None:
                        # Don't let a sick slurmdbd hold a chunk hostage: squeue
                        # already answered definitively twice, so after a few
                        # silent accounting polls, take its word and say so.
                        sacct_mute += 1
                        if sacct_mute >= SACCT_CONFIRM_TRIES:
                            log(f"{cid(args.chunk)}: sacct has not confirmed {active} in "
                                f"{sacct_mute} polls; proceeding on squeue's verdict alone")
                            break
                        log(f"{cid(args.chunk)}: squeue says {active} is gone but sacct will not "
                            f"confirm ({sacct_mute}/{SACCT_CONFIRM_TRIES}); still waiting")
                        gone = 0
                    elif live:
                        sacct_mute = 0
                        log(f"{cid(args.chunk)}: squeue says {active} is gone but sacct still lists "
                            f"{len(live)} unfinished task(s) ({', '.join(live[:4])}); still waiting")
                        gone = 0
                    else:
                        break
            if time.time() > deadline:
                msg = (f"{cid(args.chunk)}: array job {active} still running after "
                       f"{args.process_timeout_s}s; giving up on waiting (job left running, nothing deleted)")
                raise SystemExit(msg)
            time.sleep(args.poll_s)
        log(f"{cid(args.chunk)}: array job {active} drained:")
        for line in sacct_summary(active):
            log(f"    {line}")
        active = None

    n_parquet, out_bytes = count_outputs(args.chunk)
    retry = slurm_array_retry_plan(ckpt)
    if retry is None or retry["missing"]:
        msg = f"{cid(args.chunk)}: refusing _PROCESSED -- retry plan is {retry}"
        raise SystemExit(msg)
    if n_parquet == 0:
        msg = f"{cid(args.chunk)}: refusing _PROCESSED -- no parquet under {out_dir(args.chunk)}"
        raise SystemExit(msg)
    write_marker(args.chunk, "_PROCESSED", {
        "shards": retry["total_shards"],
        "completed_shards": retry["completed"],
        "parquet_files": n_parquet,
        "parquet_bytes": out_bytes,
        "manifest_entries": n_pdfs,
        "array_jobs": [h["job_id"] for h in (json.load(open(state_file)) if os.path.exists(state_file) else [])],  # noqa: SIM115
        "config": cfg_path,
    })


def cmd_retry_check(args) -> None:
    """Cross-check slurm_array_retry_plan() against the real retry_array.py by
    running the latter inside the Curator container."""
    ckpt = f"{state_dir(args.chunk)}/checkpoint"
    mine = slurm_array_retry_plan(ckpt)
    expect = ""
    if mine and mine["missing"]:
        subs: dict[int, list[int]] = {}
        for s in mine["missing"]:
            off = (s // args.max_array_size) * args.max_array_size
            subs.setdefault(off, []).append(s - off)
        expect = "\n".join(
            f"{format_array_indices(i)} {off} {mine['minimum_shard_index']} {mine['total_shards']}"
            for off, i in sorted(subs.items()))
    log(f"pure-python retry plan: {mine}")
    log(f"pure-python `--format fields` equivalent:\n{expect or '(empty)'}")

    image = args.container_image
    argv = ["srun", "--account", ACCOUNT, "--partition", DM_PARTITION, "--qos", DM_QOS,
            "--nodes", "1", "--ntasks", "1", "--time", "00:20:00",
            f"--container-image={image}", f"--container-mounts={BASE}:{BASE}",
            f"--container-workdir={CURATOR_DIR}", "--no-container-mount-home",
            f"--export=ALL,PYTHONPATH={CURATOR_DIR}",
            "/opt/venv/bin/python", RETRY_ARRAY_SCRIPT,
            "--checkpoint-path", ckpt, "--format", "fields"]
    if args.max_array_size:
        argv += ["--max-array-size", str(args.max_array_size)]
    proc = run_cmd(argv, check=False, timeout=1800)
    real = (proc.stdout or "").strip()
    log(f"retry_array.py rc={proc.returncode} stdout={real!r}")
    if proc.stderr:
        log(f"retry_array.py stderr tail: {proc.stderr[-1500:]}")
    if proc.returncode == 0:
        if real == expect.strip():
            log("AGREE: retry_array.py and the orchestrator's pure-python mirror match")
        else:
            log(f"DISAGREE: script={real!r} mirror={expect.strip()!r}")
            raise SystemExit(1)


# ==========================================================================
# UPLOAD + VERIFY
# ==========================================================================
def local_inventory(chunk_id: int) -> dict[str, int]:
    """{relpath: size} for every file under the chunk's output tree."""
    root = out_dir(chunk_id)
    inv = {}
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            p = os.path.join(dirpath, f)
            if os.path.islink(p):
                continue
            inv[os.path.relpath(p, root)] = os.path.getsize(p)
    return inv


def strip_dir_markers(remote: dict[str, dict]) -> tuple[dict[str, dict], list[str]]:
    """Split a listing into real objects and DataMover's directory markers.

    DataMover materialises each directory as a zero-byte object whose key ends
    in "/" (chunk_NNNN/, chunk_NNNN/shard_MMMM/). They have no local
    counterpart, so counting them as "unexpected at the destination" would make
    every upload fail verification forever -- and nothing would ever be reaped.
    An output file's key can never end in "/", so this cannot mask a real
    discrepancy; a zero-byte *file* still shows up as a normal object.
    """
    markers = sorted(k for k, v in remote.items() if k.endswith("/") and v["size"] == 0)
    marker_set = set(markers)
    return {k: v for k, v in remote.items() if k not in marker_set}, markers


def md5_b64_and_hex(path: str) -> tuple[str, str]:
    h = hashlib.md5()  # noqa: S324 -- S3 ETag is MD5; not used as a security primitive
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(8 << 20), b""):
            h.update(block)
    return base64.b64encode(h.digest()).decode(), h.hexdigest()


def dm_env() -> dict:
    """Environment for the local `dm` client.

    `dm` is a Go binary and sizes its thread pool from the core count: on this
    144-core login node one `job status` poll spawns 28 OS threads, and the
    orchestrator polls it once per --poll-s for the life of every upload.
    Against a 300-thread RLIMIT_NPROC that is a large recurring bite for a
    process that only submits a Slurm job and reads JSON back. Measured:
    GOMAXPROCS=4 takes the same poll from 28 threads to 6, with identical
    output.
    """
    env = dict(os.environ)
    env["GOMAXPROCS"] = "4"
    return env


def dm_status(job_id: str) -> dict:
    """`dm` writes progress chatter before the JSON and exits non-zero without
    a TTY even on success, so read the JSON, not the exit code."""
    proc = run_cmd([DM_BIN, "--format", "json", "job", "status", job_id], check=False, timeout=600,
                   env=dm_env())
    text = (proc.stdout or "")
    i = text.find("{")
    if i < 0:
        msg = f"no JSON in `dm job status {job_id}` output:\n{text[:2000]}\n{(proc.stderr or '')[:2000]}"
        raise SystemExit(msg)
    try:
        return json.loads(text[i:])
    except json.JSONDecodeError:
        # multi-stage jobs print several documents; take the first complete one
        dec = json.JSONDecoder()
        obj, _end = dec.raw_decode(text[i:])
        return obj


def cmd_upload(args) -> None:
    chunk_id = args.chunk
    sd = state_dir(chunk_id)
    if has_marker(chunk_id, "_UPLOADED"):
        log(f"{cid(chunk_id)}: _UPLOADED present, skipping")
        return
    if not has_marker(chunk_id, "_PROCESSED"):
        msg = f"{cid(chunk_id)}: refusing to upload without _PROCESSED"
        raise SystemExit(msg)

    src = out_dir(chunk_id) + "/"
    key_prefix = remote_prefix(chunk_id)
    assert_safe_remote(BUCKET, key_prefix, chunk_id)
    dst_dm = f"{DM_LOCATION}:{BUCKET}/{key_prefix}"
    inv = local_inventory(chunk_id)
    if not inv:
        msg = f"{cid(chunk_id)}: nothing to upload under {src}"
        raise SystemExit(msg)
    log(f"{cid(chunk_id)}: uploading {len(inv):,} files "
        f"({sum(inv.values()) / 2**30:.2f} GiB) {src} -> {dst_dm}")

    logdir = f"{LOG_ROOT}/{cid(chunk_id)}"
    argv = [DM_BIN, "job", "copy", src, dst_dm,
            "--runtime", "slurm",
            "--slurm-account", ACCOUNT,
            "--slurm-partition", DM_PARTITION,
            "--slurm-qos", DM_QOS,
            "--slurm-time", args.dm_time,
            "--slurm-nodes", str(args.dm_nodes),
            "--slurm-log-dir", logdir,
            "-y"]
    assert_no_destructive_verb(argv)

    if args.dry_run:
        log("DRY-RUN would run: " + " ".join(argv))
        log(f"DRY-RUN would then verify {len(inv):,} objects under s3://{BUCKET}/{key_prefix}")
        return

    os.makedirs(logdir, exist_ok=True)
    job_id = None
    state = json.load(open(f"{sd}/upload.json")) if os.path.exists(f"{sd}/upload.json") else []  # noqa: SIM115
    if state and args.reuse_dm_job:
        job_id = state[-1]["job_id"]
        log(f"{cid(chunk_id)}: reusing DataMover job {job_id} from upload.json")

    if job_id is None:
        with open(f"{sd}/upload.log", "a") as fh:
            fh.write(f"\n===== upload attempt {utcnow()} =====\n$ {' '.join(argv)}\n")
            proc = run_cmd(argv, check=False, timeout=3600, env=dm_env())
            fh.write((proc.stdout or "") + (proc.stderr or ""))
        blob = (proc.stdout or "") + (proc.stderr or "")
        log(f"dm rc={proc.returncode} (non-zero without a TTY is expected; the JSON status is authoritative)")
        m = re.findall(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", blob)
        if not m:
            msg = f"could not find a DataMover job id in dm output:\n{blob[-4000:]}"
            raise SystemExit(msg)
        job_id = m[0]
        state.append({"utc": utcnow(), "job_id": job_id, "src": src, "dst": dst_dm, "files": len(inv)})
        write_json_atomic(f"{sd}/upload.json", state)
        log(f"{cid(chunk_id)}: DataMover job {job_id}")

    deadline = time.time() + args.upload_timeout_s
    terminal = {"finished", "failed", "cancelled", "startup error", "error", "terminated"}
    while True:
        st = dm_status(job_id)
        s = str(st.get("state", "")).lower()
        log(f"  dm {job_id}: state={s} files={st.get('files_copied')}/{st.get('files_found')} "
            f"bytes={st.get('bytes_copied')}/{st.get('bytes_found')} errors={st.get('errors')}")
        if s in terminal:
            break
        if time.time() > deadline:
            msg = f"{cid(chunk_id)}: DataMover job {job_id} did not reach a terminal state in time"
            raise SystemExit(msg)
        time.sleep(args.poll_s)
    if s != "finished":
        msg = (f"{cid(chunk_id)}: DataMover job {job_id} ended in state {s!r}; NOT verifying, NOT deleting. "
               f"Rerun `upload` to retry (copy semantics, so re-copying is safe).")
        raise SystemExit(msg)

    cmd_verify(args)


def cmd_verify(args) -> None:
    """Dispatcher: verification reads every local parquet (MD5) and runs
    `rclone check`, i.e. hundreds of GiB of IO for a production chunk. That is
    not login-node work, so unless we are already inside an allocation it is
    re-run under srun. The marker is written by the worker; we re-check it."""
    chunk_id = args.chunk
    if args.dry_run:
        inv = local_inventory(chunk_id)
        log(f"DRY-RUN would verify {len(inv):,} local files against "
            f"s3://{BUCKET}/{remote_prefix(chunk_id)}")
        return
    if in_allocation() or args.in_process:
        if not in_allocation():
            local_execution_banner("destination verification", 1)
        cmd_verify_worker(args)
        return

    sub_argv = ["verify-worker", "--chunk", str(chunk_id),
                "--rclone-timeout-s", str(args.rclone_timeout_s),
                "--rclone-checkers", str(args.rclone_checkers)]
    if args.no_checksum:
        sub_argv.append("--no-checksum")
    # Verify, and REPAIR if the only defect is absent objects.
    #
    # Two independent facts make a single-shot verify wrong. (1) S3 listing here
    # is eventually consistent: chunk_0000 verified as 5 missing, and 3 of those
    # appeared unprompted within ~70min. (2) DataMover genuinely drops objects
    # while reporting success -- that same chunk logged files=5206/5206 errors=0
    # with 2 objects truly absent. So failing terminally would stall on transient
    # lag, and succeeding on dm's word would lose data. Retry with backoff to let
    # listings settle, re-copy anything still absent, and re-verify.
    #
    # Repair is deliberately narrow: ONLY when every defect is a missing object.
    # A size mismatch, an MD5 failure or an unexpected key means something is
    # wrong that re-copying should not paper over, so those still fail hard.
    attempts = max(1, int(getattr(args, "verify_attempts", 4)))
    backoff = [0, 60, 300, 900]
    for attempt in range(1, attempts + 1):
        rc = srun_self(args, sub_argv, f"verify-{cid(chunk_id)}",
                       cpus=args.verify_cpus, walltime=args.verify_time,
                       logfile=f"{state_dir(chunk_id)}/verify.log")
        if rc == 0 and has_marker(chunk_id, "_UPLOADED"):
            if attempt > 1:
                log(f"{cid(chunk_id)}: verification succeeded on attempt {attempt}")
            return
        if attempt == attempts:
            break
        try:
            with open(f"{state_dir(chunk_id)}/verify.json") as fh:
                rep = json.load(fh)
        except OSError:
            rep = {}
        n_missing = int(rep.get("n_missing_remote") or 0)
        repairable = (n_missing > 0
                      and not rep.get("size_mismatch")
                      and not rep.get("unexpected_remote")
                      and int(rep.get("md5_mismatch") or 0) == 0
                      and int(rep.get("md5_failures") and len(rep["md5_failures"]) or 0) == 0)
        if not repairable:
            log(f"{cid(chunk_id)}: verification failed with a defect that re-copying "
                f"cannot fix (missing={n_missing}, "
                f"size_mismatch={len(rep.get('size_mismatch') or [])}, "
                f"unexpected={len(rep.get('unexpected_remote') or [])}, "
                f"md5_mismatch={rep.get('md5_mismatch')}) -- not retrying")
            break
        wait = backoff[min(attempt, len(backoff) - 1)]
        log(f"{cid(chunk_id)}: {n_missing} object(s) absent at the destination; "
            f"waiting {wait}s for listing to settle, then re-copying (attempt "
            f"{attempt + 1}/{attempts})")
        if wait:
            time.sleep(wait)
        key_prefix = remote_prefix(chunk_id)
        assert_safe_remote(BUCKET, key_prefix, chunk_id)
        # rclone copy, not sync: it transfers only what is absent or differs and
        # can never delete at the destination.
        rep_argv = ["rclone", "copy", out_dir(chunk_id),
                    f"{RCLONE_REMOTE}:{BUCKET}/{key_prefix}",
                    "--transfers", str(clamp_workers(8, "rclone repair")),
                    "--checkers", str(clamp_workers(8, "rclone repair")),
                    "--retries", "3", "--stats-one-line"]
        assert_no_destructive_verb(rep_argv)
        rrc = run_cmd(rep_argv, check=False, timeout=args.rclone_timeout_s)
        log(f"{cid(chunk_id)}: repair copy rc={rrc.returncode}")

    msg = (f"{cid(chunk_id)}: VERIFICATION FAILED after {attempts} attempt(s); see "
           f"{state_dir(chunk_id)}/verify.json and {state_dir(chunk_id)}/verify.log. "
           f"No _UPLOADED marker, so REAP will refuse. Nothing deleted.")
    raise SystemExit(msg)


def cmd_verify_worker(args) -> None:
    """Independent destination verification. This -- not dm's exit code -- is
    what authorises deletion."""
    chunk_id = args.chunk
    sd = state_dir(chunk_id)
    key_prefix = remote_prefix(chunk_id)
    assert_safe_remote(BUCKET, key_prefix, chunk_id)
    inv = local_inventory(chunk_id)
    if not inv:
        msg = f"{cid(chunk_id)}: no local output to verify against ({out_dir(chunk_id)})"
        raise SystemExit(msg)

    remote, dir_markers = strip_dir_markers(s3_list_prefix(BUCKET, key_prefix, chunk_id))
    want = {key_prefix + rel: size for rel, size in inv.items()}
    missing = sorted(k for k in want if k not in remote)
    extra = sorted(k for k in remote if k not in want)
    size_mismatch = [{"key": k, "local": want[k], "remote": remote[k]["size"]}
                     for k in sorted(want) if k in remote and remote[k]["size"] != want[k]]

    # MD5-vs-ETag where the object is not multipart (ETag has no "-N" suffix).
    checked = md5_ok = md5_bad = multipart = 0
    md5_failures = []
    if not args.no_checksum:
        root = out_dir(chunk_id)
        for rel in sorted(inv):
            key = key_prefix + rel
            if key not in remote:
                continue
            etag = remote[key]["etag"]
            if "-" in etag:
                multipart += 1
                continue
            checked += 1
            _b64, hexd = md5_b64_and_hex(os.path.join(root, rel))
            if hexd == etag:
                md5_ok += 1
            else:
                md5_bad += 1
                md5_failures.append({"key": key, "local_md5": hexd, "etag": etag})

    # Independent cross-check with a different client and different code path.
    # --checkers is bounded: rclone's default (8, plus its own runtime threads)
    # is harmless here but this code path must stay cheap wherever it lands.
    rc_argv = ["rclone", "check", out_dir(chunk_id),
               f"{RCLONE_REMOTE}:{BUCKET}/{key_prefix}", "--size-only",
               "--checkers", str(clamp_workers(args.rclone_checkers, "rclone check")),
               "--transfers", "1", "--retries", "3"]
    assert_no_destructive_verb(rc_argv)
    rc = run_cmd(rc_argv, check=False, timeout=args.rclone_timeout_s)
    rc_out = ((rc.stdout or "") + (rc.stderr or "")).strip()
    rclone_ok = rc.returncode == 0

    ok = (not missing and not size_mismatch and not extra and md5_bad == 0 and rclone_ok
          and len(remote) == len(want))
    report = {
        "utc": utcnow(),
        "chunk": cid(chunk_id),
        "local_root": out_dir(chunk_id),
        "remote": f"s3://{BUCKET}/{key_prefix}",
        "local_files": len(want),
        "local_bytes": sum(want.values()),
        "remote_objects": len(remote),
        "remote_bytes": sum(v["size"] for v in remote.values()),
        "remote_dir_markers": dir_markers,
        "n_remote_dir_markers": len(dir_markers),
        "missing_remote": missing[:50],
        "n_missing_remote": len(missing),
        "unexpected_remote": extra[:50],
        "n_unexpected_remote": len(extra),
        "size_mismatch": size_mismatch[:50],
        "n_size_mismatch": len(size_mismatch),
        "md5_checked": checked,
        "md5_ok": md5_ok,
        "md5_mismatch": md5_bad,
        "md5_failures": md5_failures[:20],
        "multipart_etag_skipped": multipart,
        "rclone_check_rc": rc.returncode,
        "rclone_check_tail": rc_out[-2000:],
        "verified": ok,
    }
    os.makedirs(sd, exist_ok=True)
    write_json_atomic(f"{sd}/verify.json", report)
    log(f"{cid(chunk_id)} verify: local={len(want):,} remote={len(remote):,} missing={len(missing)} "
        f"extra={len(extra)} size_mismatch={len(size_mismatch)} md5 {md5_ok}/{checked} ok "
        f"(multipart skipped {multipart}) rclone_rc={rc.returncode} -> verified={ok}")
    if not ok:
        msg = (f"{cid(chunk_id)}: VERIFICATION FAILED -- see {sd}/verify.json. "
               f"No _UPLOADED marker, so REAP will refuse. Nothing deleted.")
        raise SystemExit(msg)
    write_marker(chunk_id, "_UPLOADED", {
        "remote": f"s3://{BUCKET}/{key_prefix}",
        "objects": len(remote),
        "bytes": sum(v["size"] for v in remote.values()),
        "local_files": len(want),
        "local_bytes": sum(want.values()),
        "md5_checked": checked,
        "rclone_check_rc": rc.returncode,
    })


# ==========================================================================
# REAP
# ==========================================================================
RECONCILE_TOLERANCE_DEFAULT = 0.001   # 0.1%; measured unexplained rate is 0.021%


def reconcile_chunk(chunk_id: int) -> dict:
    """Compare sample_ids actually present in the chunk's output against the
    manifest it was told to process, and name every PDF that produced nothing.

    This exists because nothing else catches silent loss. chunk_9000 passed
    upload verification, _PROCESSED and _UPLOADED while dropping a PDF, and
    chunk_0000 dropped 94 of 130,139 -- 67 unparseable input, 27 unexplained.
    Object-count and size checks cannot see this: every parquet was present and
    the right size, the documents simply were not in them.

    Reads the LOCAL parquet, not S3: verification has already established that
    local and remote agree, and re-downloading 200+ GiB per chunk to count ids
    would cost more than the parse.
    """
    import pyarrow.parquet as _pq
    man_path = f"{stage_dir(chunk_id)}/manifest.jsonl"
    expected = []
    with open(man_path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                expected.append(json.loads(line)["file_name"])
    want = {n[:-4] if n.endswith(".pdf") else n for n in expected}

    got: set[str] = set()
    rows = 0
    root = out_dir(chunk_id)
    for dirpath, _d, files in os.walk(root):
        for f in sorted(files):
            if not f.endswith(".parquet"):
                continue
            t = _pq.read_table(os.path.join(dirpath, f), columns=["sample_id"])
            rows += t.num_rows
            got.update(t.column("sample_id").to_pylist())

    missing = sorted(want - got)
    frac = (len(missing) / len(want)) if want else 0.0
    return {
        "utc": utcnow(),
        "chunk": cid(chunk_id),
        "manifest_entries": len(want),
        "unique_sample_ids": len(got),
        "output_rows": rows,
        "n_missing": len(missing),
        "missing_fraction": round(frac, 6),
        # Every id, not a sample: the point is to be able to re-derive or audit
        # exactly which documents produced nothing, after the inputs are gone.
        "missing_sample_ids": missing,
    }


def cmd_reap(args) -> None:
    """Dispatcher. Reaping is NOT cheap bookkeeping: reconciliation reads the
    sample_id column of every parquet in the chunk (~27s on a compute node, and
    it took 2min on the login node for chunk_0000), then the deletes unlink
    ~135,000 files. Doing that on a shared login node with RLIMIT_NPROC=300 for
    the whole user is exactly the kind of load that wedged this machine earlier,
    and at the campaign's chunks it would happen 1,049 times. So unless we are already
    inside an allocation, re-run under srun like fetch and verify do."""
    if args.dry_run or in_allocation() or getattr(args, "in_process", False):
        if args.dry_run:
            pass
        elif not in_allocation():
            local_execution_banner("chunk reap (reconcile + delete)", 1)
        cmd_reap_worker(args)
        return
    sub_argv = ["reap-worker", "--chunk", str(args.chunk),
                "--reconcile-tolerance", str(getattr(args, "reconcile_tolerance",
                                                     RECONCILE_TOLERANCE_DEFAULT))]
    if getattr(args, "skip_reconcile", False):
        sub_argv.append("--skip-reconcile")
    rc = srun_self(args, sub_argv, f"reap-{cid(args.chunk)}",
                   cpus=getattr(args, "verify_cpus", 8),
                   walltime=getattr(args, "verify_time", "02:00:00"),
                   logfile=f"{state_dir(args.chunk)}/reap.log")
    if rc != 0:
        msg = (f"{cid(args.chunk)}: REAP FAILED (rc={rc}); see "
               f"{state_dir(args.chunk)}/reap.log. Local data left in place.")
        raise SystemExit(msg)


def cmd_reap_worker(args) -> None:
    chunk_id = args.chunk
    sd = state_dir(chunk_id)
    if has_marker(chunk_id, "_REAPED"):
        log(f"{cid(chunk_id)}: _REAPED present, skipping")
        return
    if not has_marker(chunk_id, "_UPLOADED"):
        msg = (f"{cid(chunk_id)}: REFUSING to reap -- no _UPLOADED marker. "
               f"Local data left in place.")
        raise SystemExit(msg)
    try:
        with open(f"{sd}/verify.json") as fh:
            report = json.load(fh)
    except OSError as e:
        msg = f"{cid(chunk_id)}: REFUSING to reap -- {sd}/verify.json unreadable ({e})"
        raise SystemExit(msg) from e
    if not report.get("verified"):
        msg = f"{cid(chunk_id)}: REFUSING to reap -- verify.json says verified={report.get('verified')!r}"
        raise SystemExit(msg)

    # Content reconciliation. Upload verification proves the FILES arrived; this
    # proves the DOCUMENTS did. They are different questions and only the second
    # catches a shard that silently produced nothing for part of its input.
    tol = getattr(args, "reconcile_tolerance", RECONCILE_TOLERANCE_DEFAULT)
    if getattr(args, "skip_reconcile", False):
        log(f"{cid(chunk_id)}: reconciliation SKIPPED by request -- "
            f"no record of which documents produced no output")
        recon = None
    else:
        recon = reconcile_chunk(chunk_id)
        with open(f"{sd}/reconcile.json", "w") as fh:
            json.dump(recon, fh, indent=1)
        log(f"{cid(chunk_id)}: reconcile {recon['unique_sample_ids']:,}/"
            f"{recon['manifest_entries']:,} documents present, "
            f"{recon['n_missing']:,} missing ({100*recon['missing_fraction']:.3f}%)")
        if recon["missing_fraction"] > tol:
            msg = (f"{cid(chunk_id)}: REFUSING to reap -- {recon['n_missing']:,} of "
                   f"{recon['manifest_entries']:,} documents ({100*recon['missing_fraction']:.3f}%) "
                   f"produced NO output, above the {100*tol:.3f}% tolerance. Ids are listed in "
                   f"{sd}/reconcile.json. Nothing deleted.")
            raise SystemExit(msg)

    key_prefix = remote_prefix(chunk_id)
    assert_safe_remote(BUCKET, key_prefix, chunk_id)
    targets = [stage_dir(chunk_id), out_dir(chunk_id)]

    if args.dry_run:
        log(f"DRY-RUN would re-verify s3://{BUCKET}/{key_prefix} then delete: {targets}")
        return

    # Re-verify at the moment of deletion. Cheap insurance against a stale
    # marker, a partially-rerun upload, or someone clearing the prefix.
    inv = local_inventory(chunk_id)
    remote, _markers = strip_dir_markers(s3_list_prefix(BUCKET, key_prefix, chunk_id))
    want = {key_prefix + rel: size for rel, size in inv.items()}
    bad = [k for k in want if k not in remote or remote[k]["size"] != want[k]]
    if inv and bad:
        msg = (f"{cid(chunk_id)}: REFUSING to reap -- {len(bad)} of {len(want)} objects are missing or "
               f"the wrong size at the destination RIGHT NOW. Nothing deleted.")
        raise SystemExit(msg)
    if not inv:
        # Output already gone (a previous reap died between the two deletes).
        # The marker plus verify.json are the record; fall through to clean the
        # staging dir and mark _REAPED.
        log(f"{cid(chunk_id)}: local output already absent; relying on _UPLOADED + verify.json")
        if int(report.get("remote_objects") or 0) != len(remote):
            msg = (f"{cid(chunk_id)}: REFUSING to reap -- destination now has {len(remote)} objects but "
                   f"verify.json recorded {report.get('remote_objects')}")
            raise SystemExit(msg)

    freed = 0
    removed = []
    for target in targets:
        if not os.path.exists(target):
            continue
        real = assert_safe_local_delete(target, chunk_id)
        n = b = 0
        for dirpath, _d, files in os.walk(real):
            for f in files:
                try:
                    b += os.path.getsize(os.path.join(dirpath, f))
                    n += 1
                except OSError:
                    pass
        log(f"{cid(chunk_id)}: deleting {real} ({n:,} files, {b / 2**30:.2f} GiB)")
        shutil.rmtree(real)
        freed += b
        removed.append({"path": real, "files": n, "bytes": b})

    write_marker(chunk_id, "_REAPED", {
        "removed": removed,
        "freed_bytes": freed,
        "verified_objects": len(remote),
        "remote": f"s3://{BUCKET}/{key_prefix}",
        "note": "state dir and checkpoint retained; nothing deleted from S3",
    })


# ==========================================================================
# STATUS / RUN
# ==========================================================================
def chunk_state(chunk_id: int) -> str:
    for name in reversed(MARKERS):
        if has_marker(chunk_id, name):
            return name
    return "-"


def in_flight(plan: dict) -> list[int]:
    """Chunks that hold Lustre space: staged or processed but not yet reaped."""
    return [c for c in sorted(plan)
            if has_marker(c, "_FETCHED") and not has_marker(c, "_REAPED")]


def assert_safe_logs_remote(bucket: str, key: str) -> None:
    """Refuse any logs destination not strictly inside LOGS_PREFIX.

    A deliberate near-duplicate of assert_safe_remote rather than a widening of
    it. Both destinations live in the bucket that also holds the the source corpus
    source corpus, which we hold DELETE on, so each gets its own narrow guard
    and neither can be relaxed by a change aimed at the other.
    """
    problems = []
    if bucket != BUCKET:
        problems.append(f"bucket {bucket!r} != {BUCKET!r}")
    if LOGS_TOKEN not in key:
        problems.append(f"key {key!r} does not contain {LOGS_TOKEN!r}")
    if not key.startswith(LOGS_PREFIX):
        problems.append(f"key {key!r} does not start with {LOGS_PREFIX!r}")
    if ".." in key or key.startswith("/") or "//" in key:
        problems.append(f"key {key!r} is not a normalised relative key")
    for forbidden in ("pdf_corpus/v1/data/", "pdf_corpus/v1/index/",
                      "pdf_corpus/v1/errors/", "pdf_corpus/v1/markers/", DEST_PREFIX):
        if key.startswith(forbidden):
            problems.append(f"key {key!r} targets {forbidden!r}")
    if problems:
        msg = "UNSAFE LOGS DESTINATION -- refusing:\n  " + "\n  ".join(problems)
        raise SystemExit(msg)


def assert_safe_log_mirror_delete(path: str) -> str:
    """Only a single shard's mirror directory may be deleted, never the root.

    assert_safe_local_delete is chunk-scoped and cannot express this, and
    loosening it to fit would weaken the guard that protects staged corpus data.
    """
    real = os.path.realpath(path)
    root = os.path.realpath(f"{WORK}/raytmp_mirrors")
    problems = []
    if not real.startswith(root + os.sep):
        problems.append(f"{real!r} is not inside {root!r}")
    if real == root:
        problems.append(f"{real!r} is the mirror root, not one shard")
    if os.path.dirname(real) != root:
        problems.append(f"{real!r} is not a direct child of {root!r}")
    if real in ("/", BASE, STAGE_ROOT, OUT_ROOT, WORK, STATE_ROOT):
        problems.append(f"{real!r} is a root directory")
    if problems:
        msg = "REFUSING unsafe log-mirror delete:\n  " + "\n  ".join(problems)
        raise SystemExit(msg)
    return real


def mirror_is_quiet(path: str, quiet_s: int) -> bool:
    """True when nothing under `path` has been written for `quiet_s` seconds.

    The mirror is refreshed every 15s for as long as its shard runs, so an idle
    directory belongs to a finished shard. Checking mtimes beats asking squeue:
    the job id is in the directory name, but a requeued or renumbered job makes
    that lookup quietly wrong, whereas idle is idle whatever the scheduler says.
    """
    now = time.time()
    newest = 0.0
    for root, _dirs, files in os.walk(path, onerror=lambda _e: None):
        with contextlib.suppress(OSError):
            newest = max(newest, os.stat(root).st_mtime)
        for f in files:
            with contextlib.suppress(OSError):
                newest = max(newest, os.lstat(os.path.join(root, f)).st_mtime)
        if now - newest < quiet_s:
            return False
    return newest > 0 and (now - newest) >= quiet_s


def cmd_archive_logs(args) -> None:
    """Tar each finished shard's Ray/Dynamo log mirror, upload it, then delete it.

    raytmp_mirrors is a 15s `cp -a` of Ray's whole session directory, one per
    shard (submit_production_parse_array.sh:223). Nothing reclaims it: cmd_reap
    deletes only stage_dir and out_dir. Measured at 1,459,669 inodes from
    roughly 18 chunks -- about 81k per chunk, which across a 1,049-chunk
    campaign projects tens of millions against a <inode quota> inode quota. It can
    exhaust the quota on its own, before any staging, after which quota_guard
    refuses fetches and the campaign stalls past the middle with no obvious
    cause.

    These are the logs of every Ray and Dynamo process that touched the corpus,
    so they are preserved rather than deleted: one compressed tar per shard in
    S3, which also collapses ~81k inodes into 1.

    Deletion follows the same rule as chunk output -- upload, confirm the object
    exists at the expected size, and only then unlink. A tar that fails to
    upload is left exactly where it was.
    """
    import tarfile

    root = f"{WORK}/raytmp_mirrors"
    if not os.path.isdir(root):
        log(f"archive-logs: no {root}; nothing to do")
        return
    if not args.dry_run and not in_allocation() and not args.in_process:
        msg = ("REFUSING to archive logs outside a Slurm allocation: it walks and "
               "compresses hundreds of thousands of files. Use `archive-logs` via srun, "
               "or pass --in-process if you really mean it.")
        raise SystemExit(msg)

    s3 = s3_client()
    staging = f"{WORK}/log_archives"
    os.makedirs(staging, exist_ok=True)
    mirrors = sorted(d for d in glob.glob(f"{root}/*") if os.path.isdir(d))
    log(f"archive-logs: {len(mirrors)} mirror dir(s) under {root}")

    archived = skipped = failed = 0
    freed_inodes = 0
    for d in mirrors:
        name = os.path.basename(d)
        if not mirror_is_quiet(d, args.quiet_s):
            log(f"  {name}: still being written (shard running) -- skipping")
            skipped += 1
            continue
        key = f"{LOGS_PREFIX}{name}.tar.gz"
        assert_safe_logs_remote(BUCKET, key)
        tarpath = f"{staging}/{name}.tar.gz"
        n_inodes = sum(len(dd) + len(ff) for _r, dd, ff in os.walk(d, onerror=lambda _e: None))

        if args.dry_run:
            log(f"  DRY-RUN would tar {name} ({n_inodes:,} inodes) -> s3://{BUCKET}/{key}, then delete")
            continue

        try:
            with tarfile.open(tarpath, "w:gz") as tf:
                tf.add(d, arcname=name)
            size = os.path.getsize(tarpath)
            s3.upload_file(tarpath, BUCKET, key)
            head = s3.head_object(Bucket=BUCKET, Key=key)
            if head["ContentLength"] != size:
                msg = f"uploaded size {head['ContentLength']} != local {size}"
                raise OSError(msg)
        except Exception as e:  # noqa: BLE001
            log(f"  {name}: FAILED to archive ({e!r}); left in place, nothing deleted")
            failed += 1
            with contextlib.suppress(OSError):
                os.unlink(tarpath)
            continue

        assert_safe_log_mirror_delete(d)
        shutil.rmtree(d, ignore_errors=True)
        with contextlib.suppress(OSError):
            os.unlink(tarpath)
        archived += 1
        freed_inodes += n_inodes
        log(f"  {name}: {n_inodes:,} inodes -> s3://{BUCKET}/{key} "
            f"({size / 2**20:.1f} MiB), local deleted")

    log(f"archive-logs: archived={archived} skipped={skipped} failed={failed} "
        f"inodes_freed={freed_inodes:,}")


def cmd_status(args) -> None:
    plan = load_plan(args.plan_dir)
    ids = parse_chunk_selector(args.chunks, plan) if args.chunks else sorted(plan)
    q = lfs_quota()
    log(f"quota: {q['kbytes_used'] * 1024 / 2**40:.2f} TiB used of "
        f"{q['kbytes_limit'] * 1024 / 2**40:.2f} TiB, inodes {q['inodes_used']:,}/{q['inodes_limit']:,}")
    print(f"{'chunk':<12} {'state':<12} {'PDFs':>10} {'in GiB':>9} {'parquet':>8} {'out GiB':>9}  remote")
    for c in ids:
        st = chunk_state(c)
        pl = plan[c]
        f = read_marker(c, "_FETCHED")
        p = read_marker(c, "_PROCESSED")
        u = read_marker(c, "_UPLOADED")
        print(f"{cid(c):<12} {st:<12} "
              f"{f.get('manifest_entries', pl['n_pdfs']):>10,} "
              f"{pl['input_bytes'] / 2**30:>9.1f} "
              f"{p.get('parquet_files', 0):>8,} "
              f"{p.get('parquet_bytes', 0) / 2**30:>9.2f}  "
              f"{u.get('objects', '-')} obj")
    fl = in_flight(plan)
    print(f"\nin flight (staged, not reaped): {len(fl)} -> {[cid(c) for c in fl]}")


def parse_chunk_selector(spec: str, plan: dict) -> list[int]:
    if spec.strip().lower() == "all":
        return sorted(plan)
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    missing = [c for c in out if c not in plan]
    if missing:
        msg = f"chunks not in plan: {missing}"
        raise SystemExit(msg)
    return sorted(dict.fromkeys(out))


STAGE_ORDER = ["fetch", "process", "upload", "reap"]
STAGE_MARKER = {"fetch": "_FETCHED", "process": "_PROCESSED", "upload": "_UPLOADED", "reap": "_REAPED"}
STAGE_FN = {"fetch": cmd_fetch, "process": cmd_process, "upload": cmd_upload, "reap": cmd_reap}


class ThreadBudgetMonitor:
    """Regression guard for the login-node wedge.

    `run` is a supervisor: it should hold a handful of threads and let Slurm do
    the work. This samples two numbers while it drives chunks:

      * OUR subtree's thread count -- the thing this program is responsible
        for. Crossing `tree_hard` means the bug is back, so we tear the
        children down and stop.
      * the user's total -- what RLIMIT_NPROC actually governs, but it also
        counts every unrelated shell and agent on the node. That is a WARNING
        only; aborting because someone else is busy just loses work. (An
        earlier version aborted on the total and killed a run seconds before it
        would have written _PROCESSED, when our own subtree held ~3 threads.)
        It escalates to an abort only when the total is inside `reserve` of the
        limit AND our subtree is a material part of it, where stopping actually
        helps.
    """

    def __init__(self, tree_hard: int, user_warn: int, period_s: float = 5.0,
                 reserve: int = 40, material: int = 16, statefile: str | None = None):
        self.tree_hard, self.user_warn, self.period = tree_hard, user_warn, period_s
        self.reserve, self.material = reserve, material
        self.statefile = statefile
        self.peak_tree = self.peak_user = 0
        self.warned = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _abort(self, why: str) -> None:
        log(f"THREAD BUDGET: {why}; killing children and aborting before fork() starts failing.")
        kill_children()
        os.kill(os.getpid(), signal.SIGTERM)

    def _loop(self) -> None:
        limit = nproc_limit()
        unlimited = limit in (resource.RLIM_INFINITY, -1)
        while not self._stop.wait(self.period):
            tree, nproc = own_tree_threads()
            total = user_thread_count()
            self.peak_tree = max(self.peak_tree, tree)
            self.peak_user = max(self.peak_user, total)
            if tree > self.tree_hard:
                self._abort(f"this orchestrator's own process tree holds {tree} threads across "
                            f"{nproc} processes, over its ceiling of {self.tree_hard}")
                return
            if not unlimited and total > limit - self.reserve and tree > self.material:
                self._abort(f"the user total is {total} of RLIMIT_NPROC {limit} and this "
                            f"orchestrator holds {tree} of them")
                return
            if total > self.user_warn and not self.warned:
                self.warned = True
                log(f"THREAD BUDGET WARNING: {total} threads owned by this user "
                    f"(warn at {self.user_warn}, RLIMIT_NPROC {limit}); this orchestrator's own "
                    f"tree is {tree}. Not aborting -- the rest is other processes.")

    def __enter__(self) -> "ThreadBudgetMonitor":
        self.peak_tree, _n = own_tree_threads()
        self.peak_user = user_thread_count()
        log(f"thread budget: starting at {self.peak_tree} threads in this orchestrator's tree, "
            f"{self.peak_user} for the whole user (RLIMIT_NPROC={nproc_limit()}, "
            f"tree ceiling {self.tree_hard}, user warn {self.user_warn})")
        self._thread = threading.Thread(target=self._loop, daemon=True, name="thread-budget")
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> bool:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.period + 1)
        log(f"thread budget: peak {self.peak_tree} threads in this orchestrator's tree, "
            f"{self.peak_user} for the whole user")
        if self.statefile:
            try:
                write_json_atomic(self.statefile, {
                    "utc": utcnow(), "peak_orchestrator_tree_threads": self.peak_tree,
                    "peak_user_threads": self.peak_user, "rlimit_nproc": nproc_limit(),
                    "tree_hard": self.tree_hard, "user_warn": self.user_warn})
            except OSError:
                pass
        return False


def cmd_run(args) -> None:
    plan = load_plan(args.plan_dir)
    ids = parse_chunk_selector(args.chunks, plan)
    stages = [s for s in STAGE_ORDER if s in {x.strip() for x in args.stages.split(",")}]
    shown = f"{cid(ids[0])}..{cid(ids[-1])} ({len(ids)})" if len(ids) > 8 else [cid(c) for c in ids]  # noqa: PLR2004
    log(f"run: chunks {shown} stages {stages} "
        f"(K={args.max_in_flight}, quota ceiling {args.quota_ceiling_tib} TiB, dry_run={args.dry_run})")
    # The 64 default was calibrated when this loop was sequential and held ~3
    # threads. Pipelined, the tree legitimately holds ~8 per concurrent srun
    # client, so leave the default derived from the limits actually in force and
    # only honour an explicit override.
    tree_hard = args.thread_tree_hard
    if tree_hard == ORCH_TREE_THREAD_CEILING:
        tree_hard = orch_tree_ceiling(SRUN_SLOTS, args.max_in_flight)
    log(f"run: {SRUN_SLOTS} concurrent srun clients max, own-tree thread ceiling {tree_hard} "
        f"(RLIMIT_NPROC {nproc_limit()}, user currently at {user_thread_count()})")
    monitor = ThreadBudgetMonitor(tree_hard, args.thread_user_warn,
                                  statefile=None if args.dry_run else f"{LOG_ROOT}/thread_budget.json")
    with monitor:
        _run_stages(args, plan, ids, stages)
    log("run: done")


_LOG_LOCK = threading.Lock()
_REPORT_LOCK = threading.Lock()


def assert_k_fits_quota(plan: dict, ids: list[int], args) -> None:
    """Refuse a K whose worst case cannot fit under the quota ceiling.

    quota_guard reads live `lfs quota` per chunk, so with K workers all K can
    read the same free space and all conclude there is room -- it cannot bound
    a concurrent campaign on its own. Serialising the whole fetch would fix
    that, but fetch is ~7.5min per chunk and serialising 1,049 of them puts 5.5
    days of dead time on the critical path.

    So bound it statically instead: if the largest chunk's footprint times K
    still fits under the ceiling, no interleaving of K quota checks can breach
    it, and quota_guard stays a per-chunk backstop. A chunk holds its staged
    PDFs AND its parquet output until it is reaped, so both count.
    """
    k = max(1, int(args.max_in_flight))
    worst = max((plan[c]["input_bytes"] + plan[c]["n_pdfs"] * OUTPUT_BYTES_PER_PDF) for c in ids)
    used_b = lfs_quota()["kbytes_used"] * 1024
    ceiling_b = int(args.quota_ceiling_tib * 2**40)
    projected_b = used_b + k * worst
    log(f"run: K={k} x worst chunk {worst / 2**40:.2f} TiB + {used_b / 2**40:.2f} TiB used "
        f"= {projected_b / 2**40:.2f} TiB against a {ceiling_b / 2**40:.2f} TiB ceiling")
    if projected_b > ceiling_b:
        affordable = max(1, int((ceiling_b - used_b) // worst))
        msg = (f"K={k} cannot fit: {k} chunks x {worst / 2**40:.2f} TiB on top of "
               f"{used_b / 2**40:.2f} TiB used projects to {projected_b / 2**40:.2f} TiB, over the "
               f"{ceiling_b / 2**40:.2f} TiB ceiling. Use --max-in-flight {affordable} or raise "
               f"--quota-ceiling-tib.")
        raise SystemExit(msg)


def _run_one_chunk(args, plan: dict, c: int, stages: list[str]) -> None:
    """Drive one chunk through every stage. Raises on failure."""
    with ChunkLock(c, dry=args.dry_run):
        # A chunk id owns a state dir, a staging dir AND a remote prefix.
        # Two plans that both number a chunk 0 would silently share all
        # three, so the first plan to touch a chunk id keeps it.
        owner = read_marker(c, "_PLANNED").get("plan_dir")
        if owner and os.path.realpath(owner) != os.path.realpath(args.plan_dir):
            msg = (f"{cid(c)} was planned from {owner}, not {args.plan_dir}. A chunk id owns "
                   f"{state_dir(c)}, {stage_dir(c)} and s3://{BUCKET}/{remote_prefix(c)}; reusing it "
                   f"across plans would mix two runs' data. Use --chunk-id-base to give this plan "
                   f"its own id range.")
            raise SystemExit(msg)
        if not has_marker(c, "_PLANNED"):
            write_marker(c, "_PLANNED", {
                "tars": len(plan[c]["tars"]),
                "n_pdfs": plan[c]["n_pdfs"],
                "input_bytes": plan[c]["input_bytes"],
                "max_pdfs": plan[c].get("max_pdfs"),
                "partial_coverage": bool(plan[c].get("partial_coverage")),
                "plan_dir": args.plan_dir,
            }, dry=args.dry_run)
        for stage in stages:
            if has_marker(c, STAGE_MARKER[stage]):
                log(f"{cid(c)}: {stage} already done ({STAGE_MARKER[stage]})")
                continue
            log(f"=== {cid(c)} :: {stage} ===")
            sub = argparse.Namespace(**vars(args))
            sub.chunk = c
            STAGE_FN[stage](sub)


def _run_stages(args, plan: dict, ids: list[int], stages: list[str]) -> None:
    """Drive `ids` through `stages`, up to K chunks at a time.

    This used to be `for c in ids: for stage in stages: ...`, which blocked the
    whole campaign on each chunk's ~3.5h GPU array: the campaign's chunks x ~4h is 175
    days. Nothing about a chunk depends on any other chunk -- separate staging
    dirs, separate array jobs, separate S3 prefixes -- so they pipeline, and K
    chunks in flight turns 175 days into roughly 175/K.

    K is bounded by Lustre (~405 GiB per chunk in flight against the quota
    ceiling), not by the scheduler: every heavy stage runs under srun/sbatch, so
    a worker here is a thread blocked on a subprocess, and the GPU cost is K x 8
    nodes against a 2,000-node QoS limit on a 3,633-node cluster.

    One chunk's failure must not stop the campaign -- the point is unattended
    operation -- so failures are recorded and the remaining chunks continue.
    """
    report_path = f"{args.plan_dir}/run_report.json"
    done: list[int] = []
    failed: dict[int, str] = {}

    def record() -> None:
        if args.dry_run:
            return
        with _REPORT_LOCK, contextlib.suppress(OSError):
            write_json_atomic(report_path, {
                "utc": utcnow(), "pid": os.getpid(), "stages": stages,
                "k": args.max_in_flight, "requested": len(ids),
                "completed": sorted(done), "n_completed": len(done),
                "failed": {cid(k): v for k, v in sorted(failed.items())},
                "n_failed": len(failed),
            })

    def worker(c: int) -> None:
        # log() reads the thread name to attribute interleaved lines to a chunk.
        prev = threading.current_thread().name
        threading.current_thread().name = cid(c)
        try:
            _run_one_chunk(args, plan, c, stages)
        except (SystemExit, Exception) as e:  # noqa: BLE001
            # SystemExit is how every guard in this file refuses; it is a
            # per-chunk verdict, not a reason to abandon the other 1,048.
            reason = str(e) or type(e).__name__
            with _LOG_LOCK:
                log(f"!!! {cid(c)}: FAILED -- {reason}")
                log(f"!!! {cid(c)}: left in place, nothing deleted; campaign continues")
            failed[c] = reason
        else:
            done.append(c)
            with _LOG_LOCK:
                log(f"*** {cid(c)}: complete ({len(done)}/{len(ids)} done, {len(failed)} failed)")
        finally:
            threading.current_thread().name = prev
        record()

    k = max(1, int(args.max_in_flight))
    if not args.dry_run and k > 1:
        assert_k_fits_quota(plan, ids, args)
    if k == 1 or len(ids) == 1:
        for c in ids:
            worker(c)
    else:
        pre = in_flight(plan)
        if pre:
            log(f"run: {len(pre)} chunk(s) already hold Lustre space and will be resumed: "
                f"{[cid(x) for x in pre][:12]}")
        with ThreadPoolExecutor(max_workers=k) as ex:
            list(ex.map(worker, ids))

    record()
    log(f"run: {len(done)} completed, {len(failed)} failed of {len(ids)} requested")
    if failed:
        for c, reason in sorted(failed.items()):
            log(f"  FAILED {cid(c)}: {reason.splitlines()[0][:160]}")
        log(f"run: failure detail in {report_path}")


# ==========================================================================
# CLI
# ==========================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan-dir", default=PLAN_DIR_DEFAULT,
                   help="directory holding plan.jsonl + tar_stats.jsonl")
    p.add_argument("--dry-run", action="store_true", help="print actions, change nothing")
    p.add_argument("--in-process", action="store_true",
                   help="DANGEROUS off-allocation: run heavy stages in THIS process instead of "
                        "re-running them under srun. Prints a banner, hard-caps every pool at "
                        f"{MAX_LOCAL_WORKERS} workers, and refuses without RLIMIT_NPROC headroom. "
                        "Running the index scan or a fetch on the login node without this is not "
                        "possible, by design -- it wedged the node once.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, chunk: bool = True):
        if chunk:
            sp.add_argument("--chunk", type=int, required=True)
        sp.add_argument("--poll-s", type=int, default=60)
        sp.add_argument("--quota-ceiling-tib", type=float, default=60.0)
        sp.add_argument("--inode-ceiling", type=int, default=45_000_000)
        sp.add_argument("--threads", type=int, default=128)
        sp.add_argument("--index-workers", type=int, default=16)
        sp.add_argument("--no-sha", action="store_true",
                        help="skip sha256 verification of fetched PDFs (size-only)")
        sp.add_argument("--fetch-failure-tolerance", type=float, default=0.0,
                        help="fraction of a chunk's records allowed to fail fetch")
        sp.add_argument("--quiet-s", type=int, default=1800,
                        help="A Ray log mirror with no write for this long belongs to a "
                             "finished shard. The mirror refreshes every 15s while its shard "
                             "runs, so this is generous by two orders of magnitude.")
        sp.add_argument("--stage-mode", choices=("pdfs", "tars"), default="pdfs",
                        help="pdfs: one file per document (1 inode each, the corpus for the corpus "
                             "against a the quota). tars: whole uncompressed archives, read by "
                             "byte range at parse time (~67k inodes for the corpus).")
        sp.add_argument("--fetch-cpus", type=int, default=48)
        sp.add_argument("--fetch-time", default="04:00:00")
        sp.add_argument("--pdfs-per-shard", type=int, default=PDFS_PER_SHARD)
        sp.add_argument("--max-array-size", type=int, default=2000)
        sp.add_argument("--max-process-attempts", type=int, default=8)
        sp.add_argument("--verify-attempts", type=int, default=4,
                        help="Verify passes before giving up. Between passes, objects "
                             "absent at the destination are re-copied and the listing is "
                             "given time to settle (0/60/300/900s).")
        sp.add_argument("--reconcile-tolerance", type=float, default=RECONCILE_TOLERANCE_DEFAULT,
                        help="Max fraction of documents allowed to produce no output before "
                             "reap refuses. Measured rate is 0.021%%.")
        sp.add_argument("--skip-reconcile", action="store_true")
        sp.add_argument("--process-timeout-s", type=int, default=172800)
        sp.add_argument("--dm-time", default="04:00:00")
        sp.add_argument("--dm-nodes", type=int, default=2)
        sp.add_argument("--upload-timeout-s", type=int, default=43200)
        sp.add_argument("--reuse-dm-job", action="store_true",
                        help="poll the last recorded DataMover job instead of starting a new copy")
        sp.add_argument("--no-checksum", action="store_true",
                        help="skip MD5-vs-ETag (size + count + rclone check still required)")
        sp.add_argument("--rclone-timeout-s", type=int, default=3600)
        sp.add_argument("--rclone-checkers", type=int, default=4)
        sp.add_argument("--verify-cpus", type=int, default=8)
        sp.add_argument("--verify-time", default="02:00:00")
        sp.add_argument("--thread-user-warn", type=int, default=LOCAL_THREAD_CEILING,
                        help="log loudly if this user's TOTAL thread count passes this (warn only)")
        sp.add_argument("--thread-tree-hard", type=int, default=ORCH_TREE_THREAD_CEILING,
                        help="kill children and abort if THIS orchestrator's own process tree "
                             "passes this; a supervisor should never come close")
        sp.add_argument("--container-image",
                        default=f"{BASE}/containers/nemo-curator-nightly-with-dynamo-pdf-parse-venv-20260910.sqsh",
                        help="Container image. MUST contain the venv named by the config's "
                             "--server-py-executable (/opt/dynamo-pdf/bin/python). The plain "
                             "nightly does NOT: pairing it with that flag fails every shard with "
                             "'bash: /opt/dynamo-pdf/bin/python: No such file or directory' after "
                             "~7min, because Ray retries worker startup 5 times before giving up.")

    sp = sub.add_parser("plan", help="partition the sorted tar list into chunks (written once)")
    sp.add_argument("--target-input-gib", type=float, default=None)
    sp.add_argument("--tars-per-chunk", type=int, default=None)
    sp.add_argument("--max-pdfs-per-chunk", type=int, default=None,
                    help="TEST ONLY: cap each chunk at N record_ids (partial coverage)")
    sp.add_argument("--chunk-id-base", type=int, default=0)
    sp.add_argument("--limit-tars", type=int, default=None,
                    help="TEST ONLY: partition only the first N sorted tars")
    sp.add_argument("--scan-workers", type=int, default=16,
                    help=f"index-scan processes (each pinned to one thread); capped at "
                         f"{MAX_ALLOC_WORKERS} under srun and {MAX_LOCAL_WORKERS} locally")
    sp.add_argument("--scan-time", default="02:00:00", help="walltime for the srun'd index scan")
    sp.add_argument("--rescan", action="store_true", help="rebuild tar_stats.jsonl")
    sp.add_argument("--force", action="store_true", help="overwrite an existing plan")
    sp.set_defaults(func=cmd_plan)

    sp = sub.add_parser("status", help="one line per chunk")
    sp.add_argument("--chunks", default=None)
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("run", help="drive chunks through the stages")
    sp.add_argument("--chunks", required=True, help="e.g. 0-7 or 0,3,9 or all")
    sp.add_argument("--stages", default=",".join(STAGE_ORDER))
    sp.add_argument("--max-in-flight", type=int, default=8,
                    help="K: chunks staged on Lustre but not yet reaped")
    common(sp, chunk=False)
    sp.set_defaults(func=cmd_run)

    for name, fn, helptext in (
        ("fetch", cmd_fetch, "stage a chunk's PDFs (srun on cpu_datamover)"),
        ("fetch-worker", cmd_fetch_worker, "internal: the fetch body, run inside the allocation"),
        ("process", cmd_process, "submit + wait for the Slurm array"),
        ("upload", cmd_upload, "DataMover copy to SwiftStack, then verify"),
        ("verify", cmd_verify, "verify the destination against local output (sruns itself)"),
        ("verify-worker", cmd_verify_worker, "internal: the verify body, run inside the allocation"),
        ("reap-worker", cmd_reap_worker, "reap internals (runs inside an allocation)"),
        ("reap", cmd_reap, "delete local data (only with a verified _UPLOADED)"),
        ("archive-logs", cmd_archive_logs,
         "tar finished shards' Ray log mirrors to S3, then reclaim their inodes"),
        ("retry-check", cmd_retry_check, "cross-check the retry plan against retry_array.py"),
    ):
        sp = sub.add_parser(name, help=helptext)
        # archive-logs sweeps every finished shard's mirror across the whole
        # campaign, so a required --chunk would be meaningless.
        common(sp, chunk=(name != "archive-logs"))
        sp.add_argument("--max-in-flight", type=int, default=8)
        sp.set_defaults(func=fn)
    return p


def main() -> None:
    install_signal_handlers()
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
