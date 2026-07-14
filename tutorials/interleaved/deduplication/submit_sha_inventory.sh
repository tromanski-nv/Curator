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
# Slurm-array driver for the resumable PDF SHA-256 inventory stage.
#
# Each array task owns one whole CPU node. Curator's Ray executor detects the
# Slurm array automatically (via SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT)
# and shards the FilePartitioningStage source tasks across the array by hashing
# each file group's task ID -- so every array task lists the full input set but
# only hashes its assigned ~1/N of the file groups. Do NOT pre-shard the file
# list yourself; that double-shards and each task ends up processing ~1/N^2.
# The stage is resumable and writes atomically, so a failed/preempted shard can
# simply be resubmitted.
#
# Submit with the array range choosing how many nodes to use, e.g. 64 nodes:
#   mkdir -p "$DEDUP/logs"
#   sbatch --array=0-63 tutorials/interleaved/deduplication/submit_sha_inventory.sh
# Use "%K" to cap concurrency (e.g. --array=0-80%40 runs 81 shards, 40 at a time).
# The array MUST be contiguous starting at 0 (0-(N-1)) so the built-in sharding
# sees a shard index in [0, SLURM_ARRAY_TASK_COUNT).

#SBATCH --job-name=sha_inventory
#SBATCH --account=nemotron_n4_pre
# cpu_dataprocessing shares the same 108-node CPU pool as `cpu` but has no
# per-user node cap (the `cpu`/`cpu_interactive`/`cpu_long` QOS `p_cpu` caps you
# at 2 nodes; `cpu_short` at 10). Override with `sbatch --partition=...` if needed.
#SBATCH --partition=cpu_dataprocessing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=/lustre/fsw/portfolios/nemotron/users/tromanski/workspace/arxiv_nemotron_parse_run/dedup/logs/sha_inventory-%A_%a.out

set -euo pipefail

export REPO=/lustre/fsw/portfolios/nemotron/users/tromanski/github/Curator
export CURATOR_ENV="$REPO/.venv"
export PARSE=/lustre/fsw/portfolios/nemotron/users/tromanski/workspace/arxiv_nemotron_parse_run
export PDF_ROOT=/lustre/fsw/portfolios/nemotron/users/tromanski/data/arxiv-pdfs
export DEDUP="$PARSE/dedup"

# Ray must run node-local: do not attach to an external cluster, and keep the
# per-node temp/session dir off the shared filesystem to avoid collisions.
unset RAY_ADDRESS || true
export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$RAY_TMPDIR"

source "$CURATOR_ENV/bin/activate"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"

echo "shard ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT} on $(hostname) with ${SLURM_CPUS_ON_NODE} CPUs"

python tutorials/interleaved/deduplication/run.py sha-inventory \
  --input-path "$PARSE/out" \
  --pdf-root "$PDF_ROOT" \
  --output-path "$DEDUP/sha_pdf/identification/inventory" \
  --manifest-path "$DEDUP/manifests/sha_pdf_inventory.shard-${SLURM_ARRAY_TASK_ID}-of-${SLURM_ARRAY_TASK_COUNT}.json" \
  --input-blocksize 512MiB
