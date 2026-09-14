#!/bin/bash
# =============================================================================
# Periodic log janitor: tar each finished shard's Ray log mirror to SwiftStack
# and reclaim its inodes.
#
#   sbatch tutorials/slurm/log_janitor.sbatch
#
# Runs alongside the campaign supervisor. Without it the campaign stalls on
# inodes rather than bytes: raytmp_mirrors is never reclaimed by cmd_reap (which
# only deletes stage_dir and out_dir), and it was measured at 1,459,669 inodes
# from roughly 18 chunks.
#
# The launcher no longer mirrors the actor venv (submit_production_parse_array.sh
# excludes runtime_resources/, which was 81.5% of inodes and 70.5% of bytes), so
# new mirrors are far smaller than the ones this was written for. It still runs:
# the remaining logs are real, ~234k files across the measured sample, and one
# tar per shard is one inode instead of thousands.
#
# archive-logs uploads, confirms the object exists at the expected size, and only
# then deletes -- the same order used for chunk output. A shard still being
# written is skipped on an mtime check, so this is safe to run continuously
# against a live campaign.
# =============================================================================
#SBATCH --job-name=parse-log-janitor
#SBATCH --account=nemotron_n4_pre
#SBATCH --partition=cpu
#SBATCH --qos=cpu-dataprocessing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=7-00:00:00
#SBATCH --output=/lustre/fsw/portfolios/nemotron/users/tromanski/scratch/nemotron_parse_production/logs/campaign/janitor_%j.log

set -uo pipefail

CURATOR=/lustre/fsw/portfolios/nemotron/users/tromanski/github/Curator
WORK=/lustre/fsw/portfolios/nemotron/users/tromanski/scratch/nemotron_parse_production
PY="$WORK/.venv-tools/bin/python"

INTERVAL="${INTERVAL:-1800}"
QUIET_S="${QUIET_S:-1800}"

echo "log janitor start $(date -u +%Y-%m-%dT%H:%M:%SZ) job=${SLURM_JOB_ID:-?} interval=${INTERVAL}s"
while :; do
    "$PY" "$CURATOR/tutorials/slurm/chunk_orchestrator.py" \
        --plan-dir "$WORK/chunks_v3" archive-logs --quiet-s "$QUIET_S"
    echo "-- sweep done $(date -u +%Y-%m-%dT%H:%M:%SZ); sleeping ${INTERVAL}s --"
    sleep "$INTERVAL"
done
