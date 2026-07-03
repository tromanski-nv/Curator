#!/bin/bash
#SBATCH --job-name=rclone-verify
#SBATCH --account=nemotron_n4_pre
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Verify an rclone-downloaded dataset against its S3 source (count + size).
# Mirrors the download's `--size-only` semantics (S3 multipart etags are not
# plain MD5, so hash checks would false-positive; size-only is the right check).
#
# Usage (bash, from the data dir so logs/ lands there):
#   SRC=eai-data:vdi-169-essentialai-essentialai-data/pubmed \
#   DST=/lustre/fsw/portfolios/nemotron/users/tromanski/data/pubmed \
#     sbatch /lustre/fsw/portfolios/nemotron/users/tromanski/github/Curator/scripts/verify_download.sbatch

set -euo pipefail

REMOTE="${REMOTE:-eai-data}"
RCLONE_BIN="${RCLONE_BIN:-/lustre/fsw/portfolios/nemotron/users/tromanski/bin/rclone}"
SRC="${SRC:?set SRC=eai-data:bucket/prefix}"
DST="${DST:?set DST=/local/path}"
OUT="${OUT:-verify_$(basename "${DST}")_${SLURM_JOB_ID}}"

mkdir -p logs "${OUT}"

echo "============================================================"
echo "  rclone verify"
echo "  SRC=${SRC}"
echo "  DST=${DST}"
echo "  OUT=${OUT}"
echo "  Node=$(hostname)  CPUs=${SLURM_CPUS_PER_TASK}  Start=$(date)"
echo "============================================================"
"${RCLONE_BIN}" version | head -1

# 1) Object count + total bytes on the S3 source.
echo "== [1/3] rclone size (S3 source) =="
"${RCLONE_BIN}" size "${SRC}" --json | tee "${OUT}/s3_size.json"; echo

# 2) Local count + total bytes.
echo "== [2/3] local counts =="
LOCAL_ALL=$(find "${DST}" -type f | wc -l)
LOCAL_PDF=$(find "${DST}" -type f -iname '*.pdf' | wc -l)
LOCAL_BYTES=$(du -sb "${DST}" | cut -f1)
echo "local_all_files=${LOCAL_ALL}"  | tee "${OUT}/local_counts.txt"
echo "local_pdf_files=${LOCAL_PDF}"  | tee -a "${OUT}/local_counts.txt"
echo "local_bytes=${LOCAL_BYTES}"    | tee -a "${OUT}/local_counts.txt"
echo

# 3) The authoritative check: every S3 object must exist in DST with same size.
#    --one-way so extra local files (manifests, etc.) are not flagged.
echo "== [3/3] rclone check (one-way, size-only) =="
"${RCLONE_BIN}" check "${SRC}" "${DST}" \
  --size-only --one-way \
  --checkers 128 \
  --missing-on-dst "${OUT}/missing_on_dst.txt" \
  --differ "${OUT}/differ.txt" \
  --error "${OUT}/error.txt" \
  --log-level INFO --log-file "${OUT}/check.log" || true

MISSING=$(wc -l < "${OUT}/missing_on_dst.txt" 2>/dev/null || echo 0)
DIFFER=$(wc -l < "${OUT}/differ.txt" 2>/dev/null || echo 0)
ERRORS=$(wc -l < "${OUT}/error.txt" 2>/dev/null || echo 0)
S3_COUNT=$(grep -o '"count":[0-9]*' "${OUT}/s3_size.json" | head -1 | cut -d: -f2)

echo "============================================================"
echo "  SUMMARY"
echo "    s3_objects      = ${S3_COUNT:-?}"
echo "    local_all_files = ${LOCAL_ALL}"
echo "    local_pdf_files = ${LOCAL_PDF}"
echo "    missing_on_dst  = ${MISSING}   (S3 files NOT downloaded)"
echo "    differ (size)   = ${DIFFER}    (downloaded but wrong size)"
echo "    check_errors    = ${ERRORS}"
echo "    details in: ${OUT}/"
echo "  End=$(date)"
echo "============================================================"

if [[ "${MISSING}" -eq 0 && "${DIFFER}" -eq 0 && "${ERRORS}" -eq 0 ]]; then
  echo "RESULT: OK — all S3 objects present locally with matching size."
else
  echo "RESULT: MISMATCH — see ${OUT}/missing_on_dst.txt / differ.txt"
fi
