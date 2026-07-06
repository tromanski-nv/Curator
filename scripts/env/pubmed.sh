# Nemotron-Parse pubmed run — environment (BASH / Slurm compute node).
# Usage:  source scripts/env/pubmed.sh
# Scale knobs below are calibrated (~4.16 s/PDF) and overflow-safe (100 PDFs/task).

export CURATOR_DIR=/home/tromanski/github/Curator-fork
export HF_HOME=/lustre/fsw/portfolios/nemotron/users/tromanski/hf_cache

# Dataset + workspace
export PDF_ROOT=/lustre/fsw/portfolios/nemotron/users/tromanski/data/pubmed
export WORK=/lustre/fsw/portfolios/nemotron/users/tromanski/workspace/pubmed_nemotron_parse_run

# Chain submitter inputs
export SOURCE_MODE=pdf
export DATASET=$PDF_ROOT
export MANIFEST_DIR=$WORK/manifests
export OUTPUT=$WORK/out
export ACCOUNT=nemotron_n4_pre
export PARTITION=batch

# Scale / tuning — CALIBRATED for 2,121,576 PDFs @ ~4.16 s/PDF (measured 2026-07-02)
export NODES=16
export GPUS_PER_NODE=8
export TIME_LIMIT=4:00:00
export NUM_JOBS=8
# PDFS_PER_TASK 100 not 300: at 300 pubmed's per-task page-image binaries exceeded
# pyarrow's 2 GB `binary` offset limit in postprocess ("ArrowInvalid: offset
# overflow"). 100 keeps each task's table well under 2 GB.
export PDFS_PER_TASK=100
export SHARD_SIZE=64000   # generate_pdf_manifest.py --shard-size

echo "[pubmed/bash] WORK=$WORK NODES=$NODES PDFS_PER_TASK=$PDFS_PER_TASK SHARD_SIZE=$SHARD_SIZE"
