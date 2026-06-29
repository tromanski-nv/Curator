#!/bin/bash
# =============================================================================
# EAI WARC -> PDF URL pipeline — SLURM submit script (bare-metal, using uv)
#
# CPU-only metadata extraction across multiple nodes via SlurmRayClient.
# Mirrors tutorials/slurm/submit.sh; see that file for Pyxis/enroot container
# alternatives.
#
# Prerequisites:
#   - uv installed
#   - NeMo Curator checked out on a SHARED filesystem (Lustre/NFS)
#   - Input WARCs and --output-dir on the shared filesystem (local mode)
#   - For S3 mode: AWS creds available (env vars below, ~/.aws, or IAM role)
#
# Usage (local WARC dir):
#   EAI_WARC_DIR=/shared/warcs EAI_OUTPUT_DIR=/shared/out \
#       sbatch --nodes=4 tutorials/eai_crawl/submit.sh
#
# Usage (S3 source):
#   EAI_S3_BUCKET=my-bucket EAI_S3_PREFIX=crawl/warcs/ EAI_OUTPUT_DIR=/shared/out \
#       sbatch --nodes=4 tutorials/eai_crawl/submit.sh
# =============================================================================

#SBATCH --job-name=eai-warc-pdf
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=logs/eai_warc_%j.log
#SBATCH --error=logs/eai_warc_%j.log

set -euo pipefail

CURATOR_DIR="${CURATOR_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"

# Shared dir for Ray port broadcast — must be visible to ALL nodes.
export RAY_PORT_BROADCAST_DIR="${CURATOR_DIR}/logs"
export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${HOME}/.cache/uv}"

EAI_OUTPUT_DIR="${EAI_OUTPUT_DIR:?Set EAI_OUTPUT_DIR to a shared output directory}"
EAI_WARC_DIR="${EAI_WARC_DIR:-}"
EAI_S3_BUCKET="${EAI_S3_BUCKET:-}"
EAI_S3_PREFIX="${EAI_S3_PREFIX:-}"
EAI_URL_LIMIT="${EAI_URL_LIMIT:-}"

# Build source-specific args.
if [[ -n "${EAI_S3_BUCKET}" ]]; then
    SOURCE_ARGS="--s3-bucket ${EAI_S3_BUCKET} --s3-prefix ${EAI_S3_PREFIX}"
elif [[ -n "${EAI_WARC_DIR}" ]]; then
    SOURCE_ARGS="--warc-dir ${EAI_WARC_DIR}"
else
    echo "ERROR: set either EAI_WARC_DIR or EAI_S3_BUCKET" >&2
    exit 1
fi
[[ -n "${EAI_URL_LIMIT}" ]] && SOURCE_ARGS="${SOURCE_ARGS} --url-limit ${EAI_URL_LIMIT}"

echo "=================================================="
echo "  EAI WARC -> PDF URL pipeline (SLURM)"
echo "  Job ID : ${SLURM_JOB_ID}"
echo "  Nodes  : ${SLURM_JOB_NODELIST} (${SLURM_JOB_NUM_NODES} nodes)"
echo "  Source : ${SOURCE_ARGS}"
echo "  Output : ${EAI_OUTPUT_DIR}"
echo "=================================================="

mkdir -p logs "${EAI_OUTPUT_DIR}"

srun \
    --ntasks-per-node=1 \
    bash -c "
cd '${CURATOR_DIR}'
export RAY_TMPDIR=/tmp/ray_\${SLURM_JOB_ID}
export RAY_PORT_BROADCAST_DIR='${CURATOR_DIR}/logs'
# Forward AWS credentials to workers for S3 mode (no-op if unset).
export AWS_ACCESS_KEY_ID='${AWS_ACCESS_KEY_ID:-}'
export AWS_SECRET_ACCESS_KEY='${AWS_SECRET_ACCESS_KEY:-}'
export AWS_SESSION_TOKEN='${AWS_SESSION_TOKEN:-}'
export AWS_DEFAULT_REGION='${AWS_DEFAULT_REGION:-}'
echo \"[\$(hostname)] SLURM_NODEID=\${SLURM_NODEID} python=\$(uv run python --version 2>&1)\"
uv run --extra text_cpu python '${CURATOR_DIR}/tutorials/eai_crawl/run_slurm.py' \
    --slurm \
    ${SOURCE_ARGS} \
    --output-dir '${EAI_OUTPUT_DIR}'
"

echo "=================================================="
echo "  DONE"
echo "=================================================="
