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
# Slurm-array driver for the resumable sample-id removal stage. It writes a new
# interleaved dataset with the given sample_ids dropped (e.g. the SHA-256 PDF
# duplicates). Like sha-inventory, Curator's Ray executor auto-detects the array
# (SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT) and shards the file groups by
# task-id hash, so do NOT pre-shard the file list. Resumable: the writer uses
# mode="ignore", so already-written output parts are skipped on resubmit.
#
# Submit (contiguous array 0-(N-1)); INPUT/IDS/OUTPUT are overridable via env:
#   mkdir -p "$DEDUP/logs"
#   sbatch --array=0-63%56 tutorials/interleaved/deduplication/submit_sample_id_remove.sh

#SBATCH --job-name=sample_id_remove
#SBATCH --account=nemotron_n4_pre
#SBATCH --partition=cpu_dataprocessing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=/lustre/fsw/portfolios/nemotron/users/tromanski/workspace/arxiv_nemotron_parse_run/dedup/logs/sample_id_remove-%A_%a.out

set -euo pipefail

export REPO=/lustre/fsw/portfolios/nemotron/users/tromanski/github/Curator
export CURATOR_ENV="$REPO/.venv"
export PARSE=/lustre/fsw/portfolios/nemotron/users/tromanski/workspace/arxiv_nemotron_parse_run
export DEDUP="$PARSE/dedup"

# Defaults perform the SHA-duplicate removal; override to reuse for other passes.
export INPUT_PATH="${INPUT_PATH:-$PARSE/out}"
export IDS_TO_REMOVE_PATH="${IDS_TO_REMOVE_PATH:-$DEDUP/sha_pdf/identification/duplicate_sample_ids}"
export OUTPUT_PATH="${OUTPUT_PATH:-$DEDUP/sha_pdf/removal/deduplicated}"

unset RAY_ADDRESS || true
export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$RAY_TMPDIR"

source "$CURATOR_ENV/bin/activate"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"

echo "shard ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT} on $(hostname) with ${SLURM_CPUS_ON_NODE} CPUs"

python tutorials/interleaved/deduplication/run.py sample-id-remove \
  --input-path "$INPUT_PATH" \
  --ids-to-remove-path "$IDS_TO_REMOVE_PATH" \
  --output-path "$OUTPUT_PATH" \
  --manifest-path "$DEDUP/manifests/sha_pdf_removal.shard-${SLURM_ARRAY_TASK_ID}-of-${SLURM_ARRAY_TASK_COUNT}.json" \
  --input-blocksize 512MiB
