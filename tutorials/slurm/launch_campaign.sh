#!/bin/bash
# Launch the full 1,048-chunk Nemotron-Parse campaign, unattended.
#
#   setsid nohup bash tutorials/slurm/launch_campaign.sh >/dev/null 2>&1 &
#
# setsid is load-bearing: without it the supervisor is in the login shell's
# session and dies with the terminal. Everything heavy already runs under
# srun/sbatch, so what stays here is K threads blocked on subprocesses --
# measured peak 1 thread in-tree, against the 300 RLIMIT_NPROC this user has
# for the WHOLE login node.
#
# The supervisor is restartable by construction: every stage is gated on an
# on-disk marker (_FETCHED/_PROCESSED/_UPLOADED/_REAPED) and adopts in-flight
# array jobs by id and by job name, so a restart resumes rather than redoes.
# That is what makes the retry loop below safe -- it is for the supervisor
# being killed (node reboot, OOM), not for masking a broken config.
set -uo pipefail

CURATOR=/lustre/fsw/portfolios/nemotron/users/tromanski/github/Curator
WORK=/lustre/fsw/portfolios/nemotron/users/tromanski/scratch/nemotron_parse_production
PY="$WORK/.venv-tools/bin/python"
ORCH="$CURATOR/tutorials/slurm/chunk_orchestrator.py"

CHUNKS="${CHUNKS:-1-1048}"
K="${K:-16}"
MAX_RESTARTS="${MAX_RESTARTS:-20}"
RESTART_SLEEP="${RESTART_SLEEP:-300}"

LOGDIR="$WORK/logs/campaign"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/campaign.log"
echo $$ > "$LOGDIR/supervisor.pid"

exec >>"$LOG" 2>&1

echo "=============================================================="
echo "campaign start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  chunks=$CHUNKS K=$K pid=$$ host=$(hostname)"
echo "  commit=$(cd "$CURATOR" && git rev-parse --short HEAD)"
echo "=============================================================="

attempt=0
while :; do
    attempt=$((attempt + 1))
    echo "--- supervisor attempt $attempt/$MAX_RESTARTS $(date -u +%Y-%m-%dT%H:%M:%SZ) ---"
    "$PY" "$ORCH" run --chunks "$CHUNKS" --max-in-flight "$K"
    rc=$?
    if [ $rc -eq 0 ]; then
        echo "campaign finished cleanly (rc=0) $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        break
    fi
    if [ $attempt -ge "$MAX_RESTARTS" ]; then
        echo "GIVING UP after $attempt attempts (last rc=$rc). Nothing deleted."
        break
    fi
    echo "supervisor exited rc=$rc; resuming from markers in ${RESTART_SLEEP}s"
    sleep "$RESTART_SLEEP"
done

echo "campaign end $(date -u +%Y-%m-%dT%H:%M:%SZ)"
rm -f "$LOGDIR/supervisor.pid"
