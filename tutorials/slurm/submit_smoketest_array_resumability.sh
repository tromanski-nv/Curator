#!/bin/bash
# =============================================================================
# Infra smoke test: does the Slurm array + --checkpoint-dir machinery in
# production_nemotron_parse.yaml / submit_production_parse_array.sh actually
# work? Runs benchmarking/smoketest_array_resumability.yaml against the
# existing 500-PDF sample (nemotron-parse-pr2349-smoke/), 4 total shards.
#
# Test plan (see the YAML's header for what each step checks):
#   1. sbatch --array=0-1 tutorials/slurm/submit_smoketest_array_resumability.sh
#      -> shards 0 and 1 of 4, run together. Check their output_path dirs
#      (.../output/shard_000/, shard_001/) for disjoint sample_ids -- confirms
#      Slurm-array shard assignment actually splits work.
#   2. sbatch --array=0 tutorials/slurm/submit_smoketest_array_resumability.sh
#      -> re-run shard 0 alone. Should finish much faster than step 1's shard 0
#      and its stdouterr.log should show sources being skipped as already
#      complete -- confirms --checkpoint-dir resumability.
#   3. (optional, completes coverage) sbatch --array=2-3 ... for the rest.
#
# TOTAL_SHARDS is fixed at 4 regardless of which of the above you submit, same
# reasoning as submit_production_parse_array.sh (NEMO_CURATOR_SLURM_ARRAY_TOTAL_SHARDS
# overrides the raw Slurm-provided count so shard assignment stays consistent
# across separate sbatch calls).
# =============================================================================

#SBATCH --job-name=nemotron-parse-smoketest
#SBATCH --account=nemotron_n4_pre
#SBATCH --partition=batch
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --cpus-per-task=128
#SBATCH --output=/lustre/fsw/portfolios/nemotron/users/tromanski/scratch/nemotron_parse_production_smoketest/logs/smoketest_%A_%a.log

set -euo pipefail

LUSTRE=/lustre/fsw/portfolios/nemotron/users/tromanski
CURATOR_DIR="${CURATOR_DIR:-$LUSTRE/scratch/curator-pr-2349}"
IMAGE="${CONTAINER_IMAGE:-$LUSTRE/containers/arxiv-nemotron-parse-dynamo-20260909.sqsh}"
WORK="$LUSTRE/scratch/nemotron_parse_production_smoketest"
CHECKPOINT_PATH="$WORK/checkpoint"
SESSION_NAME="${SESSION_NAME:-nemotron-parse-smoketest}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: submit with sbatch --array=<range>, e.g. sbatch --array=0-1 $0" >&2
    exit 2
fi
[ -f "$IMAGE" ] || { echo "container image missing: $IMAGE"; exit 1; }

TOTAL_SHARDS="${TOTAL_SHARDS:-4}"
SHARD_INDEX_OFFSET="${SHARD_INDEX_OFFSET:-0}"
SHARD_INDEX="$((SLURM_ARRAY_TASK_ID + SHARD_INDEX_OFFSET))"
SHARD_INDEX_PADDED="$(printf '%03d' "$SHARD_INDEX")"
export SHARD_INDEX_PADDED
export NEMO_CURATOR_SLURM_ARRAY_SHARD_INDEX="$SHARD_INDEX"
export NEMO_CURATOR_SLURM_ARRAY_TOTAL_SHARDS="$TOTAL_SHARDS"
export NEMO_CURATOR_SLURM_ARRAY_MINIMUM_SHARD_INDEX="${MINIMUM_SHARD_INDEX:-0}"

mkdir -p "$WORK/logs" "$WORK/benchmark_results" "$CHECKPOINT_PATH"

echo "=================================================="
echo "  Nemotron-Parse array+resumability SMOKE TEST"
echo "=================================================="
echo "  Array job ID    : ${SLURM_ARRAY_JOB_ID:-manual}"
echo "  Array task ID   : ${SLURM_ARRAY_TASK_ID}"
echo "  Job ID (attempt): ${SLURM_JOB_ID}"
echo "  Shard index     : ${SHARD_INDEX} / ${TOTAL_SHARDS} (padded: ${SHARD_INDEX_PADDED})"
echo "  Checkpoint path : ${CHECKPOINT_PATH}"
echo "  Node            : $(hostname)"
echo "=================================================="

RAYTMP="/tmp/ray_${SLURM_JOB_ID}"
RAYMIRROR="$WORK/raytmp_mirrors/shard${SHARD_INDEX}_${SLURM_JOB_ID}"
mkdir -p "$RAYMIRROR"
( while true; do cp -a "$RAYTMP"/. "$RAYMIRROR"/ 2>/dev/null; sleep 5; done ) &
MIRROR_PID=$!
trap 'kill "$MIRROR_PID" 2>/dev/null || true; cp -a "$RAYTMP"/. "$RAYMIRROR"/ 2>/dev/null || true' EXIT

srun --container-image="$IMAGE" \
     --container-mounts="$LUSTRE:$LUSTRE,/tmp:/tmp" \
     --container-workdir="$CURATOR_DIR" \
     --no-container-mount-home \
     --export=ALL,HF_HOME="$LUSTRE/hf_cache",HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,RAY_TMPDIR="$RAYTMP" \
     bash -c '
set -euo pipefail
source /opt/curator/.venv/bin/activate
echo "[$(hostname)] python=$(python --version 2>&1)"

python "'"$CURATOR_DIR"'/benchmarking/run.py" \
    --config "'"$CURATOR_DIR"'/benchmarking/smoketest_array_resumability.yaml" \
    --session-name "'"$SESSION_NAME"'"
'
rc=$?
echo "=== shard ${SHARD_INDEX} finished rc=$rc $(date -Is) ==="
exit "$rc"
