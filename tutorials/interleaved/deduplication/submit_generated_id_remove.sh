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
# Slurm-array driver for the resumable generated-id removal stage (exact-text and
# fuzzy passes). It reads the duplicate integer IDs (_curator_dedup_id) plus the
# id_generator.json produced by the identification stage, reconstructs the same
# IDs for each file group, drops the duplicate samples, and materializes a new
# interleaved dataset. Curator's Ray executor auto-detects the array
# (SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT) and shards the file groups by
# task-id hash, so do NOT pre-shard the file list. Resumable: the writer uses
# mode="ignore", so already-written output parts are skipped on resubmit.
#
# CRITICAL: --input-path and --input-blocksize MUST match the identification run
# (sha_pdf/removal/deduplicated + 512MiB). The id_generator maps file groups to
# integer-ID ranges by hashing each group's file list, so a different input set
# or blocksize would reconstruct different IDs and remove the wrong samples.
#
# Submit (contiguous array 0-(N-1)); paths are overridable via env:
#   mkdir -p "$DEDUP/logs"
#   sbatch --array=0-63%56 tutorials/interleaved/deduplication/submit_generated_id_remove.sh

#SBATCH --job-name=generated_id_remove
#SBATCH --account=nemotron_n4_pre
#SBATCH --partition=cpu_dataprocessing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=/lustre/fsw/portfolios/nemotron/users/tromanski/workspace/arxiv_nemotron_parse_run/dedup/logs/generated_id_remove-%A_%a.out

set -euo pipefail

export REPO=/lustre/fsw/portfolios/nemotron/users/tromanski/github/Curator
export CURATOR_ENV="$REPO/.venv"
export PARSE=/lustre/fsw/portfolios/nemotron/users/tromanski/workspace/arxiv_nemotron_parse_run
export DEDUP="$PARSE/dedup"

# Defaults perform the exact-text removal pass; override for the fuzzy pass by
# pointing INPUT_PATH/IDS_TO_REMOVE_PATH/ID_GENERATOR_PATH/OUTPUT_PATH at the
# fuzzy identification artifacts.
export INPUT_PATH="${INPUT_PATH:-$DEDUP/sha_pdf/removal/deduplicated}"
export IDS_TO_REMOVE_PATH="${IDS_TO_REMOVE_PATH:-$DEDUP/exact_text/identification/ExactDuplicateIds}"
export ID_GENERATOR_PATH="${ID_GENERATOR_PATH:-$DEDUP/exact_text/identification/exact_id_generator.json}"
export OUTPUT_PATH="${OUTPUT_PATH:-$DEDUP/exact_text/removal/deduplicated}"
export MANIFEST_PREFIX="${MANIFEST_PREFIX:-exact_text_removal}"
export INPUT_BLOCKSIZE="${INPUT_BLOCKSIZE:-512MiB}"

unset RAY_ADDRESS || true
export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$RAY_TMPDIR"

source "$CURATOR_ENV/bin/activate"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"

echo "shard ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT} on $(hostname) with ${SLURM_CPUS_ON_NODE} CPUs"

python tutorials/interleaved/deduplication/run.py generated-id-remove \
  --input-path "$INPUT_PATH" \
  --ids-to-remove-path "$IDS_TO_REMOVE_PATH" \
  --id-generator-path "$ID_GENERATOR_PATH" \
  --output-path "$OUTPUT_PATH" \
  --manifest-path "$DEDUP/manifests/${MANIFEST_PREFIX}.shard-${SLURM_ARRAY_TASK_ID}-of-${SLURM_ARRAY_TASK_COUNT}.json" \
  --input-blocksize "$INPUT_BLOCKSIZE"
