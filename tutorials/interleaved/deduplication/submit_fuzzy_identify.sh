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
# Single-node GPU driver for fuzzy (MinHash + LSH + connected components)
# duplicate identification on the exact-deduplicated interleaved corpus.
#
# Like exact-identify, this is NOT a Slurm array: fuzzy dedup builds one global
# ID generator and connected-components graph across the whole dataset in a single
# Ray job. Submit as a plain sbatch (no --array); node-local Ray uses every GPU.
#
#   mkdir -p "$DEDUP/logs"
#   sbatch tutorials/interleaved/deduplication/submit_fuzzy_identify.sh
#
# Outputs are namespaced by CONFIG so different LSH settings can be compared
# side by side without clobbering: dedup/fuzzy/$CONFIG/{cache,identification}.

#SBATCH --job-name=fuzzy_identify
#SBATCH --account=nemotron_n4_pre
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fsw/portfolios/nemotron/users/tromanski/workspace/arxiv_nemotron_parse_run/dedup/logs/fuzzy_identify-%j.out

set -euo pipefail

export REPO=/lustre/fsw/portfolios/nemotron/users/tromanski/github/Curator
export CURATOR_ENV="$REPO/.venv"
export PARSE=/lustre/fsw/portfolios/nemotron/users/tromanski/workspace/arxiv_nemotron_parse_run
export DEDUP="$PARSE/dedup"

# LSH configuration. num-bands x minhashes-per-band = total minhashes (260 here).
# The similarity threshold is ~ (1/num_bands)^(1/minhashes_per_band); with 20/13
# that is ~0.79 Jaccard over char-ngram shingles. Raise the threshold (more bands
# or fewer minhashes-per-band) to catch only very close dupes; lower it to be
# more aggressive. CONFIG names the output namespace so re-runs don't collide.
export CONFIG="${CONFIG:-b20r13}"
export CHAR_NGRAMS="${CHAR_NGRAMS:-24}"
export NUM_BANDS="${NUM_BANDS:-20}"
export MINHASHES_PER_BAND="${MINHASHES_PER_BAND:-13}"
export BANDS_PER_ITERATION="${BANDS_PER_ITERATION:-5}"
export SEED="${SEED:-42}"
export INPUT_BLOCKSIZE="${INPUT_BLOCKSIZE:-512MiB}"

export INPUT_PATH="${INPUT_PATH:-$DEDUP/exact_text/removal/deduplicated}"
export CACHE_PATH="${CACHE_PATH:-$DEDUP/fuzzy/$CONFIG/cache}"
export OUTPUT_PATH="${OUTPUT_PATH:-$DEDUP/fuzzy/$CONFIG/identification}"

unset RAY_ADDRESS || true
export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}"
mkdir -p "$RAY_TMPDIR" "$CACHE_PATH" "$OUTPUT_PATH"

source "$CURATOR_ENV/bin/activate"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"

echo "fuzzy-identify ($CONFIG) on $(hostname); GPUs=${CUDA_VISIBLE_DEVICES:-unset}"

python tutorials/interleaved/deduplication/run.py fuzzy-identify \
  --input-path "$INPUT_PATH" \
  --cache-path "$CACHE_PATH" \
  --output-path "$OUTPUT_PATH" \
  --manifest-path "$DEDUP/manifests/fuzzy_identification.$CONFIG.json" \
  --input-blocksize "$INPUT_BLOCKSIZE" \
  --seed "$SEED" \
  --char-ngrams "$CHAR_NGRAMS" \
  --num-bands "$NUM_BANDS" \
  --minhashes-per-band "$MINHASHES_PER_BAND" \
  --bands-per-iteration "$BANDS_PER_ITERATION"
