#!/bin/bash
# =============================================================================
# Submit one crawl day as a fixed-WARC-count Slurm array.
#
# Each array element gets one exclusive node and one exact key file. The login
# process lists the selected day once; compute jobs do not list the prefix.
#
# Usage (from the repository root):
#   DRY_RUN=1 bash tutorials/eai_crawl/run_day_array.sh 20240814
#   EXISTING_WORKLIST_DIR=$PWD/logs/eai_array_worklists/20240814.active \
#     bash tutorials/eai_crawl/run_day_array.sh 20240814
# Or omit the dry run and submit directly with the first command minus DRY_RUN.
#
# Runtime sizing is deliberately visible. The default 240 WARCs/hour * 3 hours
# is an initial estimate, not a measurement. After a representative one-node
# canary, either set ESTIMATED_WARCS_PER_HOUR or WARCS_PER_ARRAY_TASK directly.
# A four-hour Slurm limit leaves startup and throughput-variance margin.
# =============================================================================
set -euo pipefail

if [[ "$#" -ne 1 || ! "$1" =~ ^[0-9]{8}$ ]]; then
    echo "Usage: $0 YYYYMMDD" >&2
    exit 2
fi
DAY="$1"

CURATOR_DIR="${CURATOR_DIR:-$(pwd)}"
[[ -f "${CURATOR_DIR}/pyproject.toml" ]] || {
    echo "ERROR: CURATOR_DIR must point to the shared Curator checkout" >&2
    exit 1
}
CURATOR_DIR="$(cd "${CURATOR_DIR}" && pwd -P)"

# --- Slurm / grouping ---
ACCOUNT="${ACCOUNT:-nemotron_n4_pre}"
PARTITION="${PARTITION:-cpu_dataprocessing}"
CPUS_PER_TASK="${CPUS_PER_TASK:-64}"
ARRAY_MAX_CONCURRENT="${ARRAY_MAX_CONCURRENT:-108}"
TARGET_HOURS="${TARGET_HOURS:-3}"
ESTIMATED_WARCS_PER_HOUR="${ESTIMATED_WARCS_PER_HOUR:-240}"
WARCS_PER_ARRAY_TASK="${WARCS_PER_ARRAY_TASK:-$((TARGET_HOURS * ESTIMATED_WARCS_PER_HOUR))}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
DRY_RUN="${DRY_RUN:-0}"
EXISTING_WORKLIST_DIR="${EXISTING_WORKLIST_DIR:-}"

for value_name in CPUS_PER_TASK ARRAY_MAX_CONCURRENT TARGET_HOURS ESTIMATED_WARCS_PER_HOUR WARCS_PER_ARRAY_TASK; do
    value="${!value_name}"
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: ${value_name} must be a positive integer (got ${value})" >&2
        exit 2
    fi
done

# --- source (read) ---
SRC_REMOTE="${SRC_REMOTE:-team-vendor-data}"
SRC_BUCKET="${SRC_BUCKET:-vdi-169-essentialai-essentialai-data}"
SRC_ROOT="${SRC_ROOT:-eai-warc}"
ENDPOINT="${ENDPOINT:-https://pdx.s8k.io}"

# --- output (write through rclone remote) ---
OUT_REMOTE="${OUT_REMOTE:-eai-data}"
OUT_BUCKET="${OUT_BUCKET:-eai-warcs}"
PDF_PREFIX="${PDF_PREFIX:-pdf_url_idx}"
CDX_PREFIX="${CDX_PREFIX:-cdx}"

LOG_ROOT="${LOG_ROOT:-${CURATOR_DIR}/logs}"
WORKLIST_ROOT="${WORKLIST_ROOT:-${LOG_ROOT}/eai_array_worklists}"
mkdir -p "$LOG_ROOT" "$WORKLIST_ROOT"
LOG_ROOT="$(cd "$LOG_ROOT" && pwd -P)"
WORKLIST_ROOT="$(cd "$WORKLIST_ROOT" && pwd -P)"

: "${AWS_ACCESS_KEY_ID:?export source READ credentials before submitting}"
: "${AWS_SECRET_ACCESS_KEY:?export source READ credentials before submitting}"
command -v rclone >/dev/null || { echo "ERROR: rclone is required on the login node" >&2; exit 1; }
command -v split >/dev/null || { echo "ERROR: GNU split is required on the login node" >&2; exit 1; }
[[ -f "${CURATOR_DIR}/tutorials/eai_crawl/submit.sh" ]] || {
    echo "ERROR: missing tutorials/eai_crawl/submit.sh under CURATOR_DIR" >&2
    exit 1
}

source_path="${SRC_REMOTE}:${SRC_BUCKET}/${SRC_ROOT}/${DAY}/"
key_prefix="${SRC_ROOT}/${DAY}/"
created_worklist=0
if [[ -n "$EXISTING_WORKLIST_DIR" ]]; then
    if [[ "$EXISTING_WORKLIST_DIR" != /* || ! -d "$EXISTING_WORKLIST_DIR" ]]; then
        echo "ERROR: EXISTING_WORKLIST_DIR must be an absolute existing directory" >&2
        exit 1
    fi
    worklist_dir="$(cd "$EXISTING_WORKLIST_DIR" && pwd -P)"
    if [[ "$worklist_dir" != "${WORKLIST_ROOT}/${DAY}.active" ]]; then
        echo "ERROR: existing worklist must match this day's campaign claim: ${WORKLIST_ROOT}/${DAY}.active" >&2
        exit 1
    fi
else
    worklist_dir="${WORKLIST_ROOT}/${DAY}.active"
    if ! mkdir "$worklist_dir"; then
        echo "ERROR: campaign claim already exists: ${worklist_dir}" >&2
        echo "Reuse it with EXISTING_WORKLIST_DIR=${worklist_dir}, or remove it only after the prior campaign has stopped." >&2
        exit 1
    fi
    created_worklist=1
fi

pdf_output="${OUT_REMOTE}:${OUT_BUCKET}/${PDF_PREFIX}/crawl_date=${DAY}/"
cdx_output="${OUT_REMOTE}:${OUT_BUCKET}/${CDX_PREFIX}/crawl_date=${DAY}/"
for output in "$pdf_output" "$cdx_output"; do
    existing="$(rclone lsf --max-depth 1 "$output")"
    if [[ -n "$existing" ]]; then
        if [[ "$created_worklist" == "1" ]]; then
            rmdir "$worklist_dir"
        fi
        echo "ERROR: output already exists at ${output}" >&2
        echo "Use a fresh prefix or purge the incomplete day before creating a new grouping." >&2
        exit 1
    fi
done

if [[ "$created_worklist" == "1" ]]; then
    # This is the one source listing needed to construct the scheduling
    # worklist. A later real submission can reuse a dry-run's exact files.
    if ! rclone lsf --recursive --files-only "$source_path" \
        | LC_ALL=C sort \
        | awk -v prefix="$key_prefix" '$0 ~ /\.warc\.gz$/ { print prefix $0 }' \
        | split -l "$WARCS_PER_ARRAY_TASK" -d -a 5 --additional-suffix=.txt - "${worklist_dir}/group_"; then
        rm -rf "$worklist_dir"
        echo "ERROR: failed to list and group WARCs under ${source_path}" >&2
        exit 1
    fi
fi

shopt -s nullglob
groups=("${worklist_dir}"/group_*.txt)
shopt -u nullglob
if [[ "${#groups[@]}" -eq 0 ]]; then
    if [[ "$created_worklist" == "1" ]]; then
        rmdir "$worklist_dir"
    fi
    echo "ERROR: no .warc.gz objects found under ${source_path}" >&2
    exit 1
fi

warc_count="$(awk 'END { print NR }' "${groups[@]}")"
group_capacity="$(wc -l < "${groups[0]}")"
for group_index in "${!groups[@]}"; do
    group="${groups[$group_index]}"
    expected_group="$(printf 'group_%05d.txt' "$group_index")"
    if [[ "$(basename "$group")" != "$expected_group" ]]; then
        echo "ERROR: worklist groups must be contiguous from group_00000.txt (got ${group})" >&2
        exit 1
    fi
    group_count="$(wc -l < "$group")"
    if (( group_count < 1 || group_count > group_capacity )); then
        echo "ERROR: invalid worklist size ${group_count} in ${group}" >&2
        exit 1
    fi
    if (( group_index < ${#groups[@]} - 1 && group_count != group_capacity )); then
        echo "ERROR: non-final worklist has ${group_count} keys; expected ${group_capacity}: ${group}" >&2
        exit 1
    fi
done
chmod 0444 "${groups[@]}"
worklist_dir="$(cd "$worklist_dir" && pwd -P)"

group_total="${#groups[@]}"
array_last=$((group_total - 1))
if command -v scontrol >/dev/null; then
    max_array_size="$(scontrol show config 2>/dev/null | awk '$1 == "MaxArraySize" { print $3; exit }' || true)"
    if [[ "$max_array_size" =~ ^[1-9][0-9]*$ ]] && (( group_total > max_array_size )); then
        echo "ERROR: ${group_total} groups exceed Slurm MaxArraySize=${max_array_size}" >&2
        echo "Increase WARCS_PER_ARRAY_TASK; this launcher intentionally submits only one capped array." >&2
        exit 1
    fi
fi

# submit.sh resolves group_<array-index>.txt and appends an isolated output
# directory for that group. Do not enable Curator's native array sharding: this
# 1.2 pipeline already receives an externally disjoint exact key set.
export EAI_WARC_KEY_DIR="$worklist_dir"
export EAI_S3_BUCKET="$SRC_BUCKET"
export EAI_S3_PREFIX="$key_prefix"
export EAI_S3_ENDPOINT_URL="$ENDPOINT"
export EAI_STREAM=1
export EAI_OUTPUT_DIR="s3://${OUT_BUCKET}/${PDF_PREFIX}/crawl_date=${DAY}/"
export EAI_CDX_OUTPUT_DIR="s3://${OUT_BUCKET}/${CDX_PREFIX}/crawl_date=${DAY}/"
export EAI_OUTPUT_RCLONE_REMOTE="$OUT_REMOTE"
unset EAI_URL_LIMIT
unset NEMO_CURATOR_SLURM_ARRAY_ENABLED
unset NEMO_CURATOR_SLURM_ARRAY_SHARD_INDEX
unset NEMO_CURATOR_SLURM_ARRAY_TOTAL_SHARDS
unset NEMO_CURATOR_SLURM_ARRAY_MINIMUM_SHARD_INDEX

sbatch_args=(
    sbatch
    --account="$ACCOUNT"
    --partition="$PARTITION"
    --nodes=1
    --ntasks-per-node=1
    --cpus-per-task="$CPUS_PER_TASK"
    --exclusive
    --time="$TIME_LIMIT"
    --array="0-${array_last}%${ARRAY_MAX_CONCURRENT}"
    --no-requeue
    --chdir="$CURATOR_DIR"
    --job-name="eai-${DAY}"
    --output="${LOG_ROOT}/eai_${DAY}_%A_%a.log"
    --error="${LOG_ROOT}/eai_${DAY}_%A_%a.log"
    "${CURATOR_DIR}/tutorials/eai_crawl/submit.sh"
)

echo "=================================================="
echo "  EAI one-day WARC array  (DRY_RUN=${DRY_RUN})"
echo "  Day       : ${DAY}"
echo "  Source    : ${source_path}"
echo "  WARCs     : ${warc_count}"
echo "  Groups    : ${group_total} (<= ${group_capacity} WARCs each)"
echo "  Array     : 0-${array_last}%${ARRAY_MAX_CONCURRENT}"
echo "  Allocation: 1 exclusive node/group, ${CPUS_PER_TASK} CPUs, limit=${TIME_LIMIT}"
echo "  Worklists : ${worklist_dir}"
echo "  PDF output: ${EAI_OUTPUT_DIR}warc_group=<index>/"
echo "  CDX output: ${EAI_CDX_OUTPUT_DIR}warc_group=<index>/"
echo "=================================================="

if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY_RUN command:'
    printf ' %q' "${sbatch_args[@]}"
    printf '\n'
    printf 'Submit this exact worklist with: EXISTING_WORKLIST_DIR=%q bash %q %q\n' \
        "$worklist_dir" "${CURATOR_DIR}/tutorials/eai_crawl/run_day_array.sh" "$DAY"
else
    if [[ -e "${worklist_dir}/.submitted" || ! -d "$worklist_dir" ]]; then
        echo "ERROR: worklist has already been submitted or disappeared: ${worklist_dir}" >&2
        exit 1
    fi
    if ! mkdir "${worklist_dir}/.submitting"; then
        echo "ERROR: another launcher is already submitting this worklist: ${worklist_dir}" >&2
        exit 1
    fi
    if submission="$("${sbatch_args[@]}" 2>&1)"; then
        printf '%s\n' "$submission" > "${worklist_dir}/.submitting/sbatch.out"
        mv "${worklist_dir}/.submitting" "${worklist_dir}/.submitted"
        printf '%s\n' "$submission"
    else
        rmdir "${worklist_dir}/.submitting"
        printf '%s\n' "$submission" >&2
        exit 1
    fi
fi
