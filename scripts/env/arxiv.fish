# Nemotron-Parse arxiv run — environment (FISH / local + login shell).
# Usage:  source scripts/env/arxiv.fish
# Sets everything submit_nemotron_parse_pdf_chain.sh reads, plus PDF_ROOT/WORK
# used by the manifest-gen and monitoring commands.

set -gx CURATOR_DIR /home/tromanski/github/Curator-fork
set -gx HF_HOME     /lustre/fsw/portfolios/nemotron/users/tromanski/hf_cache

# Dataset + workspace
set -gx PDF_ROOT /lustre/fsw/portfolios/nemotron/users/tromanski/data/arxiv-pdfs
set -gx WORK     /lustre/fsw/portfolios/nemotron/users/tromanski/workspace/arxiv_nemotron_parse_run

# Chain submitter inputs
set -gx SOURCE_MODE  pdf
set -gx DATASET      $PDF_ROOT
set -gx MANIFEST_DIR $WORK/manifests
set -gx OUTPUT       $WORK/out
set -gx ACCOUNT      nemotron_n4_pre
set -gx PARTITION    batch

# Scale / tuning — Option A: 16 nodes, ~2 days for 2,904,491 PDFs @ 6.77 s/PDF
set -gx NODES         16
set -gx GPUS_PER_NODE 8
set -gx TIME_LIMIT    4:00:00
set -gx NUM_JOBS      15
# PDFS_PER_TASK 100 not 200: at 200 arxiv's per-task page-image binaries exceeded
# pyarrow's 2 GB `binary` offset limit in postprocess ("ArrowInvalid: offset
# overflow"), stalling the GPU stage until jobs were idle-reaped. 100 stays < 2 GB.
set -gx PDFS_PER_TASK 100
set -gx SHARD_SIZE    64000   # generate_pdf_manifest.py --shard-size

echo "[arxiv/fish] WORK=$WORK NODES=$NODES PDFS_PER_TASK=$PDFS_PER_TASK SHARD_SIZE=$SHARD_SIZE"
