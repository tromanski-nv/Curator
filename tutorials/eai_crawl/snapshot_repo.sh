#!/bin/bash
# =============================================================================
# Freeze a copy of the Curator repo so running/queued SLURM jobs reference
# immutable code while you keep editing the live checkout.
#
# Why this works without rebuilding the venv: run_slurm.py inserts its own
# REPO_ROOT (= Path(__file__).resolve().parents[2]) at sys.path[0]. So when a
# job runs the SNAPSHOT's copy of run_slurm.py, both `nemo_curator` and
# `tutorials` resolve to the snapshot tree (sys.path[0] is searched by
# PathFinder before the venv's editable-install finder). The venv is reused
# only for third-party dependencies.
# =============================================================================
set -euo pipefail

SRC="${REPO_ROOT:-$(pwd)}"
[[ -f "$SRC/pyproject.toml" ]] || {
    echo "ERROR: '$SRC' is not a repo root (no pyproject.toml). cd to the repo or set REPO_ROOT." >&2
    exit 1
}
command -v rsync >/dev/null || { echo "ERROR: rsync not found" >&2; exit 1; }

stamp="$(date +%Y%m%d-%H%M%S)"
sha="$(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo nogit)"
dirty=""
git -C "$SRC" diff --quiet 2>/dev/null || dirty="-dirty"
DEST="${DEST:-${SNAPSHOT_ROOT:-${SRC}/../curator-snapshots}/curator-${stamp}-${sha}${dirty}}"

mkdir -p "$DEST"
SNAPSHOT_MAX_FILE_SIZE="${SNAPSHOT_MAX_FILE_SIZE:-50M}"
rsync -a \
    --max-size="$SNAPSHOT_MAX_FILE_SIZE" \
    --exclude '.git/' \
    --exclude '.venv/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.pytest_cache/' \
    --exclude '.ruff_cache/' \
    --exclude 'logs/' \
    --exclude 'output/' \
    --exclude 'curator-snapshots/' \
    "$SRC"/ "$DEST"/

mkdir -p "$DEST/logs"
{
    echo "source_repo: $SRC"
    echo "git_sha: $sha"
    echo "created: $stamp"
    echo "created_by: ${USER:-unknown}@$(hostname)"
} > "$DEST/SNAPSHOT_INFO.txt"

echo "Snapshot created: $DEST" >&2
echo "$DEST"
