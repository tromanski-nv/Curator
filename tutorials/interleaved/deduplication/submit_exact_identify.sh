#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Single-node GPU driver for exact-text duplicate identification.
#
# This stage is NOT a Slurm array: exact dedup builds one global ID generator
# and must see the whole dataset in a single Ray job. Submit as a plain sbatch
# (no --array). Node-local Ray uses every GPU in the allocation.
#
#   mkdir -p "$DEDUP/logs"
#   sbatch tutorials/interleaved/deduplication/submit_exact_identify.sh
# gpu:8 grabs a full node: the shuffle runs one actor per GPU, so the insert/hash
# phase reads and hashes the 2.2TB with 8-way parallelism. May queue behind a
# fully-idle node; drop to --gres=gpu:2 to backfill onto a partially-free node.

#SBATCH --job-name=exact_identify
#SBATCH --account=nemotron_n4_pre
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fsw/portfolios/nemotron/users/tromanski/workspace/arxiv_nemotron_parse_run/dedup/logs/exact_identify-%j.out

set -euo pipefail

export REPO=/lustre/fsw/portfolios/nemotron/users/tromanski/github/Curator
export CURATOR_ENV="$REPO/.venv"
export PARSE=/lustre/fsw/portfolios/nemotron/users/tromanski/workspace/arxiv_nemotron_parse_run
export DEDUP="$PARSE/dedup"

unset RAY_ADDRESS || true
export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}"
mkdir -p "$RAY_TMPDIR"

source "$CURATOR_ENV/bin/activate"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"

echo "exact-identify on $(hostname); GPUs=${CUDA_VISIBLE_DEVICES:-unset}"

# --total-nparts caps the number of shuffle output partitions. The output is just
# a small list of duplicate IDs, so the default (num_input_tasks // 3 ~= 1646) makes
# the finalize phase write ~1646 mostly-empty files with heavy per-partition overhead
# on few GPUs (hours). 128 larger partitions finish in minutes with identical results.
python tutorials/interleaved/deduplication/run.py exact-identify \
  --input-path "$DEDUP/sha_pdf/removal/deduplicated" \
  --output-path "$DEDUP/exact_text/identification" \
  --manifest-path "$DEDUP/manifests/exact_text_identification.json" \
  --input-blocksize 512MiB \
  --total-nparts 128
