#!/bin/bash
# =============================================================================
# Nemotron-Parse PDF production run -- Slurm array driven by benchmarking/run.py.
# Retargeted from the H100 cluster to the GB300 cluster (oci-aga-slurm-1).
#
# One array task = one node = one shard. Shard membership is chosen by
# nemo_curator.backends.slurm_array's SHA-256 hash of each source's
# deterministic id, so shards are disjoint without any coordination.
#
# ── CONFIG selects what gets run ─────────────────────────────────────────────
#   CONFIG=benchmarking/production_nemotron_parse.yaml   (default, production corpus)
#   CONFIG=benchmarking/gb300_nemotron_parse_10k.yaml    (10k validation run)
#
# ── Array size: see ARRAY SIZE DERIVATION below ──────────────────────────────
#
# ── Rollout: phased, NOT all shards at once ──────────────────────────────────
#   TOTAL_SHARDS must be IDENTICAL across every submission, even when a given
#   submission covers only a few indices. It overrides the raw
#   SLURM_ARRAY_TASK_COUNT that SlurmArrayConfig would otherwise read
#   (_get_int_env_var precedence in slurm_array.py). Without it, --array=0-1
#   today and --array=2-1999 later would see counts of 2 and 1998 and hash PDFs
#   to different shards -- silent gaps and overlaps.
#
#   The `normal` QoS caps submitted jobs per user at 2000 and array tasks count
#   individually, so 7,737 shards must go out in chunks. TOTAL_SHARDS stays 7737
#   for every one of them.
#
#     # Phase 1 -- first two shards together, sharing one checkpoint_path:
#     sbatch --array=0-1 tutorials/slurm/submit_production_parse_array.sh
#
#     # Phase 2+ -- only after phase 1 is verified, <=2000 indices at a time,
#     # each chunk submitted as the previous one drains:
#     sbatch --array=2-1999    tutorials/slurm/submit_production_parse_array.sh
#     sbatch --array=2000-3999 tutorials/slurm/submit_production_parse_array.sh
#     sbatch --array=4000-5999 tutorials/slurm/submit_production_parse_array.sh
#     sbatch --array=6000-7736 tutorials/slurm/submit_production_parse_array.sh
#
#   Failed/timed-out shards: tutorials/slurm/retry_array.py --checkpoint-path
#   <checkpoint_path> --format fields, then resubmit just those indices. A retry
#   gets a new SLURM_JOB_ID so it lands in its own parse_<idx>_<jobid> entry dir
#   under the same session, while the shared checkpoint_path lets it resume
#   mid-shard rather than reprocessing.
#
# ── Required before submitting ───────────────────────────────────────────────
#   production_nemotron_parse.yaml's manifest/pdf_dir are still placeholders --
#   the corpus is not staged on this cluster. Do not submit the
#   production config until those are real paths. The 10k config is runnable now.
# =============================================================================

#SBATCH --job-name=nemotron-parse-prod
#SBATCH --account=nemotron_n4_pre
# Hard 3h cap per task. `batch` (4:00:00 limit) accommodates it with an hour to
# spare and is the default partition, so it backfills more readily than
# batch_long.
#SBATCH --partition=batch
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
# GB300 NVL72 tray = 4 GPUs + 2 Grace CPUs (144 cores), NOT the 8-GPU DGX layout
# this script originally assumed. Every GPU QoS here sets MinTRES gres/gpu=4, so
# 4 is both the node size and the minimum allocatable request.
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --output=/lustre/fsw/portfolios/nemotron/users/tromanski/scratch/nemotron_parse_production/logs/parse_%A_%a.log

set -euo pipefail

LUSTRE=/lustre/fsw/portfolios/nemotron/users/tromanski
CURATOR_DIR="${CURATOR_DIR:-$LUSTRE/github/Curator}"
# Prebaked-venv image: /opt/dynamo-pdf/bin/python carries vLLM 0.23.0+cu129 so
# Ray skips the actor-venv build entirely (239.9s -> 107.4s startup, job 718150).
# Pair with --server-py-executable in the config; without that flag this image
# behaves exactly like the plain nightly.
IMAGE="${CONTAINER_IMAGE:-$LUSTRE/containers/nemo-curator-nightly-with-dynamo-pdf-parse-venv-20260910.sqsh}"
CONFIG="${CONFIG:-$CURATOR_DIR/benchmarking/production_nemotron_parse.yaml}"
WORK="$LUSTRE/scratch/nemotron_parse_production"
CHECKPOINT_PATH="$WORK/checkpoint"
SESSION_NAME="${SESSION_NAME:-nemotron-parse-production}"
# Exported so the containerised `bash -c` block can reference them directly
# rather than splicing them in through nested quoting.
export CURATOR_DIR CONFIG SESSION_NAME

# ── ARRAY SIZE DERIVATION ────────────────────────────────────────────────────
#   Target: no task exceeds 3h of Slurm wall clock.
#   ALL INPUTS BELOW ARE MEASURED on this cluster by job 715690 (tuneC2), the
#   fastest 10,000-PDF run. Nothing here is extrapolated from the H100 handoff.
#
#   r = 6.59 pages/s/GPU   end-to-end, 145,327 pages / 5,516s on 4 GPUs (job
#                          715690). The ONLY difference from the 5.78 run was
#                          --max-num-seqs=128: capping vLLM's concurrent
#                          sequences cut per-replica generation throughput ~13%
#                          (6,250 -> 5,430 tok/s) but raised END-TO-END by 14%.
#                          Running pinned at exactly 128/replica (vs 279 avg
#                          uncapped) and Waiting rose 1.6 -> 180, i.e. the queue
#                          moved off the GPU and into the scheduler. With 32
#                          client slots each blocked on a semaphore, tail
#                          latency dominates: fewer faster requests beat more
#                          slower ones. This also explains why tuneB (16 client
#                          workers) failed -- we were pushing the wrong knob.
#                          Use END-TO-END only; the inference-stage metric is
#                          inflated when tasks < configured parallelism
#                          (benchmark:139).
#   g = 4 GPUs/node
#   N = <corpus document count>
#   p = 14.53 pages/PDF    MEASURED: 145,327/10,000 at --max-pages=645. Matches
#                          the 14.54 predicted from the page-count distribution.
#   P = N * p = 1,937,491,345 pages
#
#   STARTUP_S = 400        measured 242s (server ready, CUDA graphs ON) plus
#                          margin. The handoff's ~35min graph-capture figure is
#                          an H100 number; on sm_103 capture costs ~80s.
#   TIMEOUT_S = 9900       run.py internal timeout, 15min inside the 3h wall so
#                          the entry exits its try/finally cleanly.
#   T_work = 9900 - 400 = 9500s
#
#   pages/shard = g * r * T_work = 4 * 6.59 * 9500 = 250,420
#   N_shards    = ceil(P / pages_per_shard) = 7,737 at zero margin
#
#   SHIPPED VALUE IS 8,900, +15%. Simulating 200 real chunks (120,471 PDFs, 8
#   shards, measured page distribution, SHA-256 assignment) the worst shard drew
#   247,875 pages = 156.7 min against a 158 min budget -- 1.0% headroom. Spread
#   max/mean was 1.069 mean / 1.111 p95 / 1.145 max. Assignment is per TASK (25
#   PDFs) and page counts are heavy-tailed, so imbalance eats the margin even
#   though disjointness is exact. At 7,737 a large fraction of shards would
#   overrun the 3h wall and need a second pass; 8,900 restores ~14% headroom for
#   ~3% more startup overhead.
#   -> ~14,983 PDFs/shard, ~25 GB source PDFs, ~24 GB parquet out per shard.
#
#   Total ~23,211 node-hours including the 3h-granularity overhead.
#   At the 2000-node QoS cap that is ~12h of wall clock; at the 693 nodes
#   typically idle, ~34h.
#
#   Undersizing is not catastrophic: a shard that overruns resumes from
#   checkpoint_path on the next attempt, so the cost is an extra pass.
#
#   HEADROOM: at --max-num-seqs=128 the server runs Running=128/replica with
#   Waiting~180 and KV at only 7.9%, so the GPU is no longer the queue -- the
#   scheduler is. Further gains are likely to come from tuning max_num_seqs
#   (64? 192?) rather than from offering MORE client concurrency, which is what
#   tuneB tried and failed at. Startup is also still ~240s/shard = ~515 node-h
#   across 7,737 shards; a container with a prebaked venv (py_executable) and a
#   persistent CUDA_CACHE_PATH on Lustre would cut most of that.
#
#   Slurm caps: MaxArraySize=32768 is fine, but the `normal` QoS allows only
#   2000 SUBMITTED JOBS PER USER and array tasks count individually, so 7,737
#   must go out in chunks -- see the rollout above.
PAGES_PER_PDF="${PAGES_PER_PDF:-14.53}"
TOTAL_SHARDS="${TOTAL_SHARDS:-8900}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: submit with sbatch --array=<range>, e.g. sbatch --array=0-1 $0" >&2
    exit 2
fi
[ -f "$IMAGE" ] || { echo "container image missing: $IMAGE"; exit 1; }
[ -f "$CONFIG" ] || { echo "config missing: $CONFIG"; exit 1; }

SHARD_INDEX_OFFSET="${SHARD_INDEX_OFFSET:-0}"
SHARD_INDEX="$((SLURM_ARRAY_TASK_ID + SHARD_INDEX_OFFSET))"
# 4 digits: 7,737 shards -> max index 7736. Widening later would rename every
# output directory, so this is padded for the largest plausible count now.
SHARD_INDEX_PADDED="$(printf '%04d' "$SHARD_INDEX")"
export SHARD_INDEX_PADDED
export NEMO_CURATOR_SLURM_ARRAY_SHARD_INDEX="$SHARD_INDEX"
export NEMO_CURATOR_SLURM_ARRAY_TOTAL_SHARDS="$TOTAL_SHARDS"
export NEMO_CURATOR_SLURM_ARRAY_MINIMUM_SHARD_INDEX="${MINIMUM_SHARD_INDEX:-0}"

# Persistent CUDA JIT kernel cache. Without it every shard recompiles kernels on
# startup; with a warm cache readiness drops sharply (125s -> 61s and first
# request 121s -> 6.3s in Praateek's measurement). It lives under $LUSTRE, which
# is already bind-mounted into the container, so no extra --container-mounts.
# CUDA_CACHE_MAXSIZE is raised to 4 GiB because the default (~1 GiB on recent
# CUDA) evicts entries and defeats the point across a fleet this size.
#
# CONCURRENCY CAVEAT: Praateek's numbers are for a warm cache and a SINGLE job.
# 7,737 shards writing one shared directory is a different regime -- CUDA writes
# cache entries atomically, but contention and eviction churn are unmeasured at
# this scale. Populate it with a couple of shards first, and if it misbehaves
# either give each shard its own (losing the benefit) or make it read-only once
# warm.
CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$WORK/cuda_cache}"
# VLLM_CACHE_ROOT is a SEPARATE layer from CUDA_CACHE_PATH: the latter is the
# CUDA driver's JIT kernel cache, this is vLLM's own artefact cache
# (torch.compile output, graph captures). Both default somewhere ephemeral, so
# both recompile on every shard unless pointed at persistent storage.
VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$WORK/vllm_cache}"
# Two more compile caches, per upstream PR #2395 (GB200 PDF sweep, same node
# shape as ours). TRITON_CACHE_DIR holds Triton JIT kernels -- a third distinct
# layer from the CUDA driver cache and vLLM's own artefacts. UV_CACHE_DIR only
# matters while the actor venv is still built at runtime; once py_executable
# points at a prebaked venv it is inert, but it costs nothing and makes the
# fallback path much cheaper.
# HF_MODULES_CACHE needs no entry: transformers derives it from HF_HOME, and the
# Nemotron-Parse vision encoder's auto_map remote code already lands under
# $HF_HOME/modules/transformers_modules/.
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$WORK/triton_cache}"
UV_CACHE_DIR="${UV_CACHE_DIR:-$WORK/uv_cache}"
export CUDA_CACHE_PATH VLLM_CACHE_ROOT TRITON_CACHE_DIR UV_CACHE_DIR

mkdir -p "$WORK/logs" "$WORK/benchmark_results" "$CHECKPOINT_PATH" "$CUDA_CACHE_PATH" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$UV_CACHE_DIR"

echo "=================================================="
echo "  Nemotron-Parse production -- Slurm array (GB300)"
echo "=================================================="
echo "  Array job ID    : ${SLURM_ARRAY_JOB_ID:-manual}"
echo "  Array task ID   : ${SLURM_ARRAY_TASK_ID}"
echo "  Job ID (attempt): ${SLURM_JOB_ID}"
echo "  Shard index     : ${SHARD_INDEX} / ${TOTAL_SHARDS} (padded: ${SHARD_INDEX_PADDED})"
echo "  pages/PDF assumed: ${PAGES_PER_PDF}"
echo "  Config          : ${CONFIG}"
echo "  Session name    : ${SESSION_NAME}"
echo "  Checkpoint path : ${CHECKPOINT_PATH}"
echo "  CUDA cache      : ${CUDA_CACHE_PATH}"
echo "  vLLM cache      : ${VLLM_CACHE_ROOT}"
echo "  triton/uv cache : ${TRITON_CACHE_DIR} | ${UV_CACHE_DIR}"
echo "  Node            : $(hostname)"
echo "=================================================="

# run.py's ray_cluster.py makes its OWN short temp dir (/tmp/ray_<uuid8>) to dodge
# the AF_UNIX 107-byte sun_path limit, and ignores RAY_TMPDIR. So mirror by glob
# rather than a name we predict -- the previous /tmp/ray_${SLURM_JOB_ID} guess
# never matched and silently mirrored nothing.
RAYMIRROR="$WORK/raytmp_mirrors/shard${SHARD_INDEX}_${SLURM_JOB_ID}"
mkdir -p "$RAYMIRROR"
# rsync, not cp -a, and NOT the whole session dir.
#
# The point of this mirror is Ray/Dynamo logs. But a Ray session dir also holds
# runtime_resources/ -- the ~247-package actor venv uv builds -- and `cp -a`
# copied all of it, every 15s, for the whole 2-3h shard. Measured: 1,459,669
# inodes across roughly 18 chunks, ~81k per shard, of which the logs are a
# rounding error. A magic-byte census of 398 sampled files found ZERO document
# content, and a file-name census found 83,322 fmhaSm100fKernel_Qkv kernels and
# 58,092 __init__.py. Nothing reclaims any of it, and across the campaign it
# projects tens of millions of inodes against the quota -- enough to
# stall the campaign on its own once quota_guard starts refusing fetches.
#
# Two fixes in one line. --exclude drops the venv, which is reproducible from
# the image and worthless in a log archive. rsync is incremental, so each pass
# ships only what changed rather than re-copying ~81k files every 15 seconds,
# which was itself a standing metadata load on Lustre for the whole run.
#
# /tmp/ray_spill is skipped explicitly. nemo_curator/core/utils.py:176 hardcodes
# the object-spilling directory to exactly that path, and it MATCHES the
# /tmp/ray_* glob below. Spilling is off by default (enable_object_spilling
# defaults to False), which is the only reason this mirror holds no document
# content today -- with it on, spilled page images would stream onto Lustre
# through this loop with no other change anywhere. Skipping it here means the
# no-document-content property survives someone flipping that flag.
RAY_MIRROR_ARGS=(-a --exclude=runtime_resources/ --exclude='*.sock' --exclude=sockets/)
( while true; do
    for d in /tmp/ray_*; do
      [ "$d" = "/tmp/ray_spill" ] && continue
      [ -d "$d" ] && rsync "${RAY_MIRROR_ARGS[@]}" "$d" "$RAYMIRROR"/ 2>/dev/null
    done
    sleep 15
  done ) &
MIRROR_PID=$!
# `|| true` on every cleanup step: a non-zero rc from the trap body becomes the
# job's exit status under `set -e` and shows as FAILED in sacct even on success.
trap 'kill "$MIRROR_PID" 2>/dev/null || true;
      for d in /tmp/ray_*; do [ "$d" = "/tmp/ray_spill" ] && continue; [ -d "$d" ] && rsync "${RAY_MIRROR_ARGS[@]}" "$d" "$RAYMIRROR"/ 2>/dev/null; done;
      true' EXIT

srun --container-image="$IMAGE" \
     --container-mounts="$LUSTRE:$LUSTRE,/tmp:/tmp" \
     --container-workdir="$CURATOR_DIR" \
     --no-container-mount-home \
     --export=ALL,HF_HOME="$LUSTRE/hf_cache",RAY_MAX_LIMIT_FROM_API_SERVER=200000,PYTHONPATH="$CURATOR_DIR:$WORK/pydeps",CUDA_CACHE_PATH="$CUDA_CACHE_PATH",CUDA_CACHE_MAXSIZE=4294967296,VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT",TRITON_CACHE_DIR="$TRITON_CACHE_DIR",UV_CACHE_DIR="$UV_CACHE_DIR",VLLM_USE_FLASHINFER_SAMPLER=0 \
     bash -c '
set -euo pipefail
# NOTE: plain `bash -c`, never `bash -lc`. A login shell sources ~/.bashrc, which
# prepends the Lustre micromamba env (python 3.14) ahead of the container venv.
PY=/opt/venv/bin/python
echo "[$(hostname)] python=$("$PY" --version 2>&1)"

# nemo_curator must resolve to the worktree, not the image site-packages.
# entry.py derives the SCRIPT path from run.py __file__, but "import
# nemo_curator" still follows sys.path -- without CURATOR_DIR on PYTHONPATH the
# image frozen copy wins and local fixes (e.g. the actor-venv quack pin)
# silently do not apply.
NC_PATH=$("$PY" -c "import nemo_curator; print(nemo_curator.__file__)")
echo "[$(hostname)] nemo_curator -> $NC_PATH"
case "$NC_PATH" in
    "$CURATOR_DIR"/*) : ;;
    *) echo "FATAL: nemo_curator resolved outside $CURATOR_DIR"; exit 4 ;;
esac

# cv2 is required by PDFPreprocessStage but the nightly image omits the
# nemo_curator[cv2] extra; it is supplied via PYTHONPATH ($WORK/pydeps).
"$PY" -c "import cv2; print(\"[cv2]\", cv2.__version__)" \
    || { echo "FATAL: cv2 missing -- see WORK/pydeps"; exit 5; }

nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv,noheader 2>/dev/null \
    | sed "s/^/  [$(hostname)] GPU /" || echo "  [$(hostname)] no GPUs"

# run.py must be invoked by its live-worktree path: runner/entry.py derives
# _curator_repo_path from __file__, so invoking the image-frozen copy would
# silently run the image copy of nemotron_parse_pdf_benchmark.py, not ours.
"$PY" "$CURATOR_DIR/benchmarking/run.py" \
    --config "$CONFIG" \
    --session-name "$SESSION_NAME"
'
rc=$?
echo "=== shard ${SHARD_INDEX} finished rc=$rc $(date -Is) ==="
echo "-- Ray/Dynamo logs mirrored to: $RAYMIRROR"
exit "$rc"
