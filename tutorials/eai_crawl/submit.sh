#!/bin/bash
# =============================================================================
# EAI WARC -> PDF URL pipeline — SLURM submit script (bare-metal, using uv)
#
# CPU-only metadata extraction across multiple nodes via SlurmRayClient.
# Mirrors tutorials/slurm/submit.sh; see that file for Pyxis/enroot container
# alternatives.
#
# Prerequisites:
#   - uv sync --group eai-warcs --extra text_cpu already done on the shared checkout;
#     this script activates .venv rather than `uv run` (Ray workers + uv rebuild = no ray).
#   - NeMo Curator checked out on a SHARED filesystem (Lustre/NFS)
#   - For S3 mode: AWS_* = read creds (team-vendor-data); output uses
#     --output-rclone-remote eai-data (or EAI_OUT_AWS_*)
#
# Usage (S3/SwiftStack -> eai-data outputs):
#   EAI_S3_BUCKET=vdi-169-essentialai-essentialai-data \
#   EAI_S3_PREFIX=eai-warc/20240814/ \
#   EAI_S3_ENDPOINT_URL=https://pdx.s8k.io \
#   EAI_STREAM=1 \
#   EAI_OUTPUT_DIR=s3://eai-warcs/pdf_url_idx/crawl_date=20240814/ \
#   EAI_CDX_OUTPUT_DIR=s3://eai-warcs/cdx/crawl_date=20240814/ \
#   EAI_OUTPUT_RCLONE_REMOTE=eai-data \
#   AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... \
#       sbatch --nodes=2 --time=04:00:00 tutorials/eai_crawl/submit.sh
#
# Tip: one job per day-prefix. One Ray task = one WARC.
# =============================================================================

#SBATCH --job-name=eai-warc-pdf
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/eai_warc_%j.log
#SBATCH --error=logs/eai_warc_%j.log

set -euo pipefail

# Under sbatch, $0 / BASH_SOURCE often point at a spool copy under
# /cm/local/apps/slurm/var — not the repo. Prefer an explicit CURATOR_DIR, else
# SLURM_SUBMIT_DIR (cwd when you ran sbatch; submit from the repo root).
if [[ -z "${CURATOR_DIR:-}" ]]; then
    if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/pyproject.toml" ]]; then
        CURATOR_DIR="${SLURM_SUBMIT_DIR}"
    elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/../../pyproject.toml" ]]; then
        CURATOR_DIR="$(cd "${SLURM_SUBMIT_DIR}/../.." && pwd)"
    else
        CURATOR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    fi
fi
VENV_PATH="${VENV_PATH:-${CURATOR_DIR}/.venv}"
if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
    echo "ERROR: missing venv at ${VENV_PATH}" >&2
    echo "  Set CURATOR_DIR to the shared Curator checkout, or sbatch from the repo root." >&2
    echo "  Then: uv sync --group eai-warcs --extra text_cpu" >&2
    exit 1
fi

# Shared dir for Ray port broadcast — must be visible to ALL nodes. When
# CURATOR_DIR is a frozen snapshot this may not exist yet, so create it.
export RAY_PORT_BROADCAST_DIR="${CURATOR_DIR}/logs"
mkdir -p "${RAY_PORT_BROADCAST_DIR}"
export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}"
# Do NOT use `uv run` under Ray — workers relaunch with a bare `uv run` that can
# rebuild an empty .venv (ModuleNotFoundError: ray). Activate the synced venv instead.

# SLURM job-array mode: one array task == one chunk. Derive the per-chunk keys
# file and output prefixes from SLURM_ARRAY_TASK_ID + templates, since every
# array task shares the same submission environment (run_all_days.sh can't pass
# distinct EAI_OUTPUT_DIR per task). Only engages when both are present, so the
# single-job path (explicit EAI_* below) is unaffected.
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && -n "${EAI_KEYS_DIR:-}" ]]; then
    _cid="$(printf '%04d' "${SLURM_ARRAY_TASK_ID}")"
    : "${EAI_OUT_BUCKET:?array mode needs EAI_OUT_BUCKET}"
    EAI_S3_KEYS_FILE="${EAI_KEYS_DIR}/chunk_${_cid}.keys"
    EAI_OUTPUT_DIR="s3://${EAI_OUT_BUCKET}/${EAI_PDF_PREFIX:-pdf_url_idx}/chunk=${_cid}/"
    EAI_CDX_OUTPUT_DIR="s3://${EAI_OUT_BUCKET}/${EAI_CDX_PREFIX:-cdx}/chunk=${_cid}/"
    : "${EAI_CHECKPOINT_ROOT:?array mode needs EAI_CHECKPOINT_ROOT}"
    EAI_CHECKPOINT_PATH="${EAI_CHECKPOINT_ROOT}/chunk_${_cid}"
    export EAI_S3_KEYS_FILE EAI_OUTPUT_DIR EAI_CDX_OUTPUT_DIR EAI_CHECKPOINT_PATH
fi

EAI_OUTPUT_DIR="${EAI_OUTPUT_DIR:?Set EAI_OUTPUT_DIR (local path or s3://eai-warcs/pdf_url_idx/<day>/)}"
EAI_WARC_DIR="${EAI_WARC_DIR:-}"
EAI_S3_BUCKET="${EAI_S3_BUCKET:-}"
EAI_S3_PREFIX="${EAI_S3_PREFIX:-}"
EAI_S3_ENDPOINT_URL="${EAI_S3_ENDPOINT_URL:-}"
EAI_STREAM="${EAI_STREAM:-}"
EAI_URL_LIMIT="${EAI_URL_LIMIT:-}"
EAI_CDX_OUTPUT_DIR="${EAI_CDX_OUTPUT_DIR:-}"
EAI_OUTPUT_RCLONE_REMOTE="${EAI_OUTPUT_RCLONE_REMOTE:-}"
# Optional byte-chunk manifest: file of WARC keys (one per line) spanning any days.
# Must live on a shared FS (Lustre) so the head node can read it at run time.
EAI_S3_KEYS_FILE="${EAI_S3_KEYS_FILE:-}"
# Ray execution backend. Default ray_actor_pool.
EAI_BACKEND="${EAI_BACKEND:-}"
# Target CDX rows per consolidated Parquet part (~2M rows ~= 250 MiB).
EAI_CDX_ROWS_PER_FILE="${EAI_CDX_ROWS_PER_FILE:-}"
# Target PDF-index rows per deterministic source-group part.
EAI_PDF_ROWS_PER_FILE="${EAI_PDF_ROWS_PER_FILE:-}"
EAI_WARCS_PER_TASK="${EAI_WARCS_PER_TASK:-}"
EAI_CHECKPOINT_PATH="${EAI_CHECKPOINT_PATH:-}"

# Build source-specific args.
if [[ -n "${EAI_S3_BUCKET}" ]]; then
    SOURCE_ARGS="--s3-bucket ${EAI_S3_BUCKET}"
    # Only pass --s3-prefix when non-empty: in byte-chunk (keys-file) mode the
    # prefix is empty, and an empty unquoted value collapses to nothing, leaving
    # a bare "--s3-prefix" that argparse rejects ("expected one argument").
    [[ -n "${EAI_S3_PREFIX}" ]] && SOURCE_ARGS="${SOURCE_ARGS} --s3-prefix ${EAI_S3_PREFIX}"
    [[ -n "${EAI_S3_ENDPOINT_URL}" ]] && SOURCE_ARGS="${SOURCE_ARGS} --s3-endpoint-url ${EAI_S3_ENDPOINT_URL}"
    # Streaming is required for compressed .warc.gz (e.g. the EAI crawl).
    [[ -n "${EAI_STREAM}" ]] && SOURCE_ARGS="${SOURCE_ARGS} --stream"
    [[ -n "${EAI_S3_KEYS_FILE}" ]] && SOURCE_ARGS="${SOURCE_ARGS} --s3-keys-file ${EAI_S3_KEYS_FILE}"
elif [[ -n "${EAI_WARC_DIR}" ]]; then
    SOURCE_ARGS="--warc-dir ${EAI_WARC_DIR}"
else
    echo "ERROR: set either EAI_WARC_DIR or EAI_S3_BUCKET" >&2
    exit 1
fi
[[ -n "${EAI_URL_LIMIT}" ]] && SOURCE_ARGS="${SOURCE_ARGS} --url-limit ${EAI_URL_LIMIT}"
[[ -n "${EAI_STREAM_CPUS:-}" ]] && SOURCE_ARGS="${SOURCE_ARGS} --stream-cpus ${EAI_STREAM_CPUS}"
[[ -n "${EAI_BACKEND}" ]] && SOURCE_ARGS="${SOURCE_ARGS} --backend ${EAI_BACKEND}"
[[ -n "${EAI_CDX_OUTPUT_DIR}" ]] && SOURCE_ARGS="${SOURCE_ARGS} --cdx-output-dir ${EAI_CDX_OUTPUT_DIR}"
[[ -n "${EAI_CDX_ROWS_PER_FILE}" ]] && SOURCE_ARGS="${SOURCE_ARGS} --cdx-rows-per-file ${EAI_CDX_ROWS_PER_FILE}"
[[ -n "${EAI_PDF_ROWS_PER_FILE}" ]] && SOURCE_ARGS="${SOURCE_ARGS} --pdf-rows-per-file ${EAI_PDF_ROWS_PER_FILE}"
[[ -n "${EAI_WARCS_PER_TASK}" ]] && SOURCE_ARGS="${SOURCE_ARGS} --warcs-per-task ${EAI_WARCS_PER_TASK}"
[[ -n "${EAI_CHECKPOINT_PATH}" ]] && SOURCE_ARGS="${SOURCE_ARGS} --checkpoint-path ${EAI_CHECKPOINT_PATH}"
[[ -n "${EAI_OUTPUT_RCLONE_REMOTE}" ]] && SOURCE_ARGS="${SOURCE_ARGS} --output-rclone-remote ${EAI_OUTPUT_RCLONE_REMOTE}"

# Fail fast if S3 read creds are missing: without this the whole allocation boots
# Ray and only dies ~1 min in with botocore NoCredentialsError (wasted node-time).
if [[ -n "${EAI_S3_BUCKET}" && -z "${AWS_ACCESS_KEY_ID:-}" ]]; then
    echo "ERROR: EAI_S3_BUCKET is set but AWS_ACCESS_KEY_ID is empty." >&2
    echo "  Export read creds in the shell you sbatch from, e.g. from rclone:" >&2
    echo "    export AWS_ACCESS_KEY_ID=\$(rclone config show team-vendor-data | sed -n 's/^access_key_id = //p')" >&2
    echo "    export AWS_SECRET_ACCESS_KEY=\$(rclone config show team-vendor-data | sed -n 's/^secret_access_key = //p')" >&2
    exit 1
fi
if [[ -n "${EAI_S3_KEYS_FILE}" && ! -f "${EAI_S3_KEYS_FILE}" ]]; then
    echo "ERROR: EAI_S3_KEYS_FILE=${EAI_S3_KEYS_FILE} not found (must be on a shared FS)." >&2
    exit 1
fi

echo "=================================================="
echo "  EAI WARC -> PDF URL pipeline (SLURM)"
echo "  Job ID : ${SLURM_JOB_ID}"
echo "  Nodes  : ${SLURM_JOB_NODELIST} (${SLURM_JOB_NUM_NODES} nodes)"
echo "  Dir    : ${CURATOR_DIR}"
echo "  Venv   : ${VENV_PATH}"
echo "  Source : ${SOURCE_ARGS}"
echo "  Keys   : ${EAI_S3_KEYS_FILE:-"(prefix listing)"}"
echo "  Output : ${EAI_OUTPUT_DIR}"
echo "  CDX    : ${EAI_CDX_OUTPUT_DIR:-"(disabled)"}"
echo "  Checkpt: ${EAI_CHECKPOINT_PATH:-"(disabled)"}"
echo "  Backend: ${EAI_BACKEND:-"ray_actor_pool (default)"}"
echo "=================================================="

mkdir -p logs
# Only mkdir local output dirs (s3:// paths are created on write).
if [[ "${EAI_OUTPUT_DIR}" != s3://* ]]; then
    mkdir -p "${EAI_OUTPUT_DIR}"
fi
if [[ -n "${EAI_CDX_OUTPUT_DIR}" && "${EAI_CDX_OUTPUT_DIR}" != s3://* ]]; then
    mkdir -p "${EAI_CDX_OUTPUT_DIR}"
fi

srun \
    --ntasks-per-node=1 \
    bash -c "
cd '${CURATOR_DIR}'
source '${VENV_PATH}/bin/activate'
export RAY_TMPDIR=/tmp/ray_\${SLURM_JOB_ID}
export RAY_PORT_BROADCAST_DIR='${CURATOR_DIR}/logs'
# EAI manifests are already pre-sharded by the outer SLURM array. Present each
# chunk to Curator as one logical shard so native source filtering does not
# partition it a second time, while retaining FailedTask/completion manifests.
export NEMO_CURATOR_SLURM_ARRAY_ENABLED=1
export NEMO_CURATOR_SLURM_ARRAY_SHARD_INDEX=0
export NEMO_CURATOR_SLURM_ARRAY_TOTAL_SHARDS=1
export NEMO_CURATOR_SLURM_ARRAY_MINIMUM_SHARD_INDEX=0
# Read creds (team-vendor-data) for WARC streaming.
export AWS_ACCESS_KEY_ID='${AWS_ACCESS_KEY_ID:-}'
export AWS_SECRET_ACCESS_KEY='${AWS_SECRET_ACCESS_KEY:-}'
export AWS_SESSION_TOKEN='${AWS_SESSION_TOKEN:-}'
export AWS_DEFAULT_REGION='${AWS_DEFAULT_REGION:-}'
export AWS_ENDPOINT_URL='${EAI_S3_ENDPOINT_URL:-${AWS_ENDPOINT_URL:-}}'
# Optional explicit write creds (else --output-rclone-remote loads eai-data from rclone.conf).
export EAI_OUT_AWS_ACCESS_KEY_ID='${EAI_OUT_AWS_ACCESS_KEY_ID:-}'
export EAI_OUT_AWS_SECRET_ACCESS_KEY='${EAI_OUT_AWS_SECRET_ACCESS_KEY:-}'
export EAI_OUT_AWS_SESSION_TOKEN='${EAI_OUT_AWS_SESSION_TOKEN:-}'
export EAI_OUT_AWS_ENDPOINT_URL='${EAI_OUT_AWS_ENDPOINT_URL:-${EAI_S3_ENDPOINT_URL:-}}'
export EAI_OUT_AWS_DEFAULT_REGION='${EAI_OUT_AWS_DEFAULT_REGION:-${AWS_DEFAULT_REGION:-}}'
echo \"[\$(hostname)] SLURM_NODEID=\${SLURM_NODEID} python=\$(python --version 2>&1) which=\$(which python)\"
python '${CURATOR_DIR}/tutorials/eai_crawl/run_slurm.py' \
    --slurm \
    ${SOURCE_ARGS} \
    --output-dir '${EAI_OUTPUT_DIR}'
"

echo "=================================================="
echo "  DONE"
echo "=================================================="
