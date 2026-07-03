# Nemotron-Parse arxiv run — environment (BASH / Slurm compute node).
# Usage:  source scripts/env/arxiv.sh
# Same vars as arxiv.fish, for on-node manifest generation / calibration.

export CURATOR_DIR=/lustre/fsw/portfolios/nemotron/users/tromanski/github/Curator
export HF_HOME=/lustre/fsw/portfolios/nemotron/users/tromanski/hf_cache

# Dataset + workspace
export PDF_ROOT=/lustre/fsw/portfolios/nemotron/users/tromanski/data/arxiv-pdfs
export WORK=/lustre/fsw/portfolios/nemotron/users/tromanski/workspace/arxiv_nemotron_parse_run

# Chain submitter inputs
export SOURCE_MODE=pdf
export DATASET=$PDF_ROOT
export MANIFEST_DIR=$WORK/manifests
export OUTPUT=$WORK/out
export ACCOUNT=nemotron_n4_pre
export PARTITION=batch

# Scale / tuning — Option A: 16 nodes, ~2 days for 2,904,491 PDFs @ 6.77 s/PDF
export NODES=16
export GPUS_PER_NODE=8
export TIME_LIMIT=4:00:00
export NUM_JOBS=15
# PDFS_PER_TASK 100 not 200: at 200 arxiv's per-task page-image binaries exceeded
# pyarrow's 2 GB `binary` offset limit in postprocess ("ArrowInvalid: offset
# overflow"), stalling the GPU stage until jobs were idle-reaped. 100 stays < 2 GB.
export PDFS_PER_TASK=100
export SHARD_SIZE=64000   # generate_pdf_manifest.py --shard-size

echo "[arxiv/bash] WORK=$WORK NODES=$NODES PDFS_PER_TASK=$PDFS_PER_TASK SHARD_SIZE=$SHARD_SIZE"
