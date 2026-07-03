# Nemotron-Parse pubmed run — environment (FISH / local + login shell).
# Usage:  source scripts/env/pubmed.fish
# Scale knobs below are calibrated (~4.16 s/PDF) and overflow-safe (100 PDFs/task).

set -gx CURATOR_DIR /lustre/fsw/portfolios/nemotron/users/tromanski/github/Curator
set -gx HF_HOME     /lustre/fsw/portfolios/nemotron/users/tromanski/hf_cache

# Dataset + workspace
set -gx PDF_ROOT /lustre/fsw/portfolios/nemotron/users/tromanski/data/pubmed
set -gx WORK     /lustre/fsw/portfolios/nemotron/users/tromanski/workspace/pubmed_nemotron_parse_run

# Chain submitter inputs
set -gx SOURCE_MODE  pdf
set -gx DATASET      $PDF_ROOT
set -gx MANIFEST_DIR $WORK/manifests
set -gx OUTPUT       $WORK/out
set -gx ACCOUNT      nemotron_n4_pre
set -gx PARTITION    batch

# Scale / tuning — CALIBRATED for 2,121,576 PDFs @ ~4.16 s/PDF (measured 2026-07-02)
set -gx NODES         16
set -gx GPUS_PER_NODE 8
set -gx TIME_LIMIT    4:00:00
set -gx NUM_JOBS      8
# PDFS_PER_TASK 100 not 300: at 300 pubmed's per-task page-image binaries exceeded
# pyarrow's 2 GB `binary` offset limit in postprocess ("ArrowInvalid: offset
# overflow"). 100 keeps each task's table well under 2 GB.
set -gx PDFS_PER_TASK 100
set -gx SHARD_SIZE    64000   # generate_pdf_manifest.py --shard-size

echo "[pubmed/fish] WORK=$WORK NODES=$NODES PDFS_PER_TASK=$PDFS_PER_TASK SHARD_SIZE=$SHARD_SIZE"
