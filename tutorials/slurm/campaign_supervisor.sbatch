#!/bin/bash
# =============================================================================
# Unattended supervisor for the deduplicated, tar-staged Nemotron-Parse campaign.
#
#   sbatch tutorials/slurm/campaign_supervisor.sbatch
#
# Runs on a CPU node, never the login node. What stays here is a poller: every
# heavy stage goes out as its own job (fetch/verify/reap via srun, process via
# an sbatch array), so this holds a handful of threads -- measured peak 14
# in-tree for a full chunk lifecycle.
#
# ── Why an sbatch job and not nohup on the login node ────────────────────────
# The login node's RLIMIT_NPROC is 300 for this user's ENTIRE set of processes
# and it has been wedged once already. A 6-day supervisor belongs in the
# scheduler. This does mean the orchestrator submits sbatch from inside an
# allocation, which needs two things that are now handled in
# chunk_orchestrator.py's submit(): the submitter's SLURM_* must NOT be
# inherited by the array (it breaks the array's own srun with "cpus-per-task set
# by two different environment variables"), and SLURM_CONF MUST be kept (without
# it sbatch loads a default config whose cli_filter/lua plugin does not exist on
# this cluster).
#
# ── Resumability ─────────────────────────────────────────────────────────────
# Every stage is gated on an on-disk marker and in-flight arrays are adopted by
# job id and by job name, so a restart resumes rather than redoes. That is what
# makes both the inner retry loop and the self-resubmit below safe.
# =============================================================================
#SBATCH --job-name=parse-campaign
#SBATCH --account=nemotron_n4_pre
#SBATCH --partition=cpu
#SBATCH --qos=cpu-dataprocessing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=/lustre/fsw/portfolios/nemotron/users/tromanski/scratch/nemotron_parse_production/logs/campaign/supervisor_%j.log

set -uo pipefail

CURATOR=/lustre/fsw/portfolios/nemotron/users/tromanski/github/Curator
WORK=/lustre/fsw/portfolios/nemotron/users/tromanski/scratch/nemotron_parse_production
PY="$WORK/.venv-tools/bin/python"
ORCH="$CURATOR/tutorials/slurm/chunk_orchestrator.py"

PLAN_DIR="${PLAN_DIR:-$WORK/chunks_v3}"
CHUNKS="${CHUNKS:-2001-3048}"
K="${K:-16}"
MAX_RESTARTS="${MAX_RESTARTS:-20}"
RESTART_SLEEP="${RESTART_SLEEP:-300}"

mkdir -p "$WORK/logs/campaign"

echo "=============================================================="
echo "campaign start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  slurm job   : ${SLURM_JOB_ID:-?} on $(hostname)"
echo "  plan        : $PLAN_DIR"
echo "  chunks      : $CHUNKS   K=$K   stage-mode=tars"
echo "  commit      : $(cd "$CURATOR" && git rev-parse --short HEAD)"
echo "=============================================================="

attempt=0
rc=1
while [ $attempt -lt "$MAX_RESTARTS" ]; do
    attempt=$((attempt + 1))
    echo "--- orchestrator attempt $attempt/$MAX_RESTARTS $(date -u +%Y-%m-%dT%H:%M:%SZ) ---"
    "$PY" "$ORCH" --plan-dir "$PLAN_DIR" run \
        --chunks "$CHUNKS" --max-in-flight "$K" --stage-mode tars
    rc=$?
    if [ $rc -eq 0 ]; then
        echo "orchestrator finished cleanly (rc=0) $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        break
    fi
    echo "orchestrator exited rc=$rc; resuming from markers in ${RESTART_SLEEP}s"
    sleep "$RESTART_SLEEP"
done

# Self-resubmit if work remains. The 7-day wall is shorter than the projected
# campaign, and markers make a fresh allocation a resume, not a redo.
REMAINING=$("$PY" - <<'EOF'
import json, os, sys
sys.path.insert(0, "/lustre/fsw/portfolios/nemotron/users/tromanski/github/Curator/tutorials/slurm")
import chunk_orchestrator as co
co.log = lambda *a, **k: None
plan = co.load_plan(os.environ.get("PLAN_DIR", f"{co.WORK}/chunks_v3"))
lo, hi = (int(x) for x in os.environ.get("CHUNKS", "2001-3048").split("-"))
print(sum(1 for c in plan if lo <= c <= hi and not co.has_marker(c, "_REAPED")))
EOF
)
echo "chunks still unreaped: ${REMAINING:-unknown}"
if [ "${REMAINING:-0}" -gt 0 ] 2>/dev/null; then
    echo "resubmitting supervisor for the remainder"
    cd "$CURATOR" && sbatch tutorials/slurm/campaign_supervisor.sbatch
fi
echo "campaign supervisor end $(date -u +%Y-%m-%dT%H:%M:%SZ)"
