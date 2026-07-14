#!/bin/bash
# =============================================================================
# EAI WARC -> PDF URL pipeline — full-crawl driver (set-and-forget)
#
# Enumerates every crawl day under the source bucket and submits ONE SLURM job
# per day (via submit.sh). Designed to be launched once and left alone:
#   - skips empty days and days already completed (safe to re-run == resume)
#   - sizes --nodes per day by WARC count (never over-allocates tiny days)
#   - --requeue so preempted/failed nodes auto-restart
#   - overwrite output mode, so a re-run of a day cleanly replaces partials
#
# Run from the repo root on a LOGIN node (needs rclone + network):
#   # 1) export READ creds (team-vendor-data) so submit.sh's guard passes
#   export AWS_ACCESS_KEY_ID=$(rclone config show team-vendor-data | sed -n 's/^access_key_id = //p')
#   export AWS_SECRET_ACCESS_KEY=$(rclone config show team-vendor-data | sed -n 's/^secret_access_key = //p')
#   # 2) preview what WOULD be submitted (no jobs created)
#   DRY_RUN=1 bash tutorials/eai_crawl/run_all_days.sh
#   # 3) submit for real
#   bash tutorials/eai_crawl/run_all_days.sh
#
# Common overrides (env):
#   NODES_MAX=16 MAX_PER_CORE_MB_S=1.2   # planning knobs
#   ONLY_DAYS="20251015 20260502"        # restrict to specific days
#   DRY_RUN=1                            # preview only
# =============================================================================
set -euo pipefail

# --- SLURM / cluster ---
ACCOUNT="${ACCOUNT:-nemotron_n4_pre}"
PARTITION="${PARTITION:-cpu_dataprocessing}"       # infinite walltime; account allowed
CPUS_PER_TASK="${CPUS_PER_TASK:-64}"               # whole node (nodes have 64 cores)
WORKERS_PER_NODE="${WORKERS_PER_NODE:-62}"         # usable after Ray reserves ~2
NODES_MAX="${NODES_MAX:-16}"                        # cap per-day node request
TIME_LIMIT="${TIME_LIMIT:-4-00:00:00}"             # MUST be explicit (partition default is 31m)

# --- source (READ) ---
SRC_REMOTE="${SRC_REMOTE:-team-vendor-data}"
SRC_BUCKET="${SRC_BUCKET:-vdi-169-essentialai-essentialai-data}"
SRC_ROOT="${SRC_ROOT:-eai-warc}"
ENDPOINT="${ENDPOINT:-https://pdx.s8k.io}"

# --- output (WRITE via rclone remote) ---
OUT_REMOTE="${OUT_REMOTE:-eai-data}"
OUT_BUCKET="${OUT_BUCKET:-eai-warcs}"
PDF_PREFIX="${PDF_PREFIX:-pdf_url_idx}"
CDX_PREFIX="${CDX_PREFIX:-cdx}"

DRY_RUN="${DRY_RUN:-0}"
ONLY_DAYS="${ONLY_DAYS:-}"

# --- preflight ---
: "${AWS_ACCESS_KEY_ID:?export READ creds first (team-vendor-data access_key_id)}"
: "${AWS_SECRET_ACCESS_KEY:?export READ creds first (team-vendor-data secret_access_key)}"
command -v rclone >/dev/null || { echo "ERROR: rclone not found (run on a login node)" >&2; exit 1; }
[[ -f tutorials/eai_crawl/submit.sh ]] || { echo "ERROR: run from the Curator repo root" >&2; exit 1; }

# Static env consumed by submit.sh (exported to each sbatch job).
export EAI_S3_BUCKET="$SRC_BUCKET"
export EAI_S3_ENDPOINT_URL="$ENDPOINT"
export EAI_STREAM=1
export EAI_OUTPUT_RCLONE_REMOTE="$OUT_REMOTE"
unset EAI_URL_LIMIT 2>/dev/null || true   # process ALL WARCs per day

# Enumerate day directories (keep only YYYYMMDD).
if [[ -n "$ONLY_DAYS" ]]; then
    days="$ONLY_DAYS"
else
    days=$(rclone lsf --dirs-only "${SRC_REMOTE}:${SRC_BUCKET}/${SRC_ROOT}/" \
           | sed 's#/$##' | grep -E '^[0-9]{8}$' || true)
fi
[[ -n "$days" ]] || { echo "No day directories found under ${SRC_REMOTE}:${SRC_BUCKET}/${SRC_ROOT}/" >&2; exit 1; }

total_days=0 submitted=0 skipped=0 total_bytes=0
echo "=================================================="
echo "  EAI full-crawl driver  (DRY_RUN=${DRY_RUN})"
echo "  Source : ${SRC_REMOTE}:${SRC_BUCKET}/${SRC_ROOT}/"
echo "  Output : ${OUT_REMOTE}:${OUT_BUCKET}/{${PDF_PREFIX},${CDX_PREFIX}}/crawl_date=<day>/"
echo "  Sizing : <= ${NODES_MAX} nodes/day, ${CPUS_PER_TASK} cpus/node, time=${TIME_LIMIT}"
echo "=================================================="

for day in $days; do
    total_days=$((total_days + 1))

    # rclone size --json -> {"count":N,"bytes":B,...}
    stats=$(rclone size --json "${SRC_REMOTE}:${SRC_BUCKET}/${SRC_ROOT}/${day}/" 2>/dev/null || echo '{}')
    bytes=$(sed -n 's/.*"bytes":\([0-9]*\).*/\1/p' <<<"$stats"); bytes="${bytes:-0}"
    count=$(sed -n 's/.*"count":\([0-9]*\).*/\1/p' <<<"$stats"); count="${count:-0}"

    if [[ "$bytes" -eq 0 || "$count" -eq 0 ]]; then
        echo "skip  $day  (empty)"; skipped=$((skipped + 1)); continue
    fi

    # Resume: skip if this day's PDF output already has files.
    if [[ -n "$(rclone lsf "${OUT_REMOTE}:${OUT_BUCKET}/${PDF_PREFIX}/crawl_date=${day}/" 2>/dev/null || true)" ]]; then
        echo "skip  $day  (output exists)"; skipped=$((skipped + 1)); continue
    fi

    # Nodes = enough for ~one wave, capped. ceil(count / WORKERS_PER_NODE).
    nodes=$(( (count + WORKERS_PER_NODE - 1) / WORKERS_PER_NODE ))
    (( nodes < 1 )) && nodes=1
    (( nodes > NODES_MAX )) && nodes=$NODES_MAX

    gib=$(awk "BEGIN{printf \"%.1f\", ${bytes}/1073741824}")
    total_bytes=$((total_bytes + bytes))

    export EAI_S3_PREFIX="${SRC_ROOT}/${day}/"
    export EAI_OUTPUT_DIR="s3://${OUT_BUCKET}/${PDF_PREFIX}/crawl_date=${day}/"
    export EAI_CDX_OUTPUT_DIR="s3://${OUT_BUCKET}/${CDX_PREFIX}/crawl_date=${day}/"

    echo "submit $day  (${count} WARCs, ${gib} GiB) -> nodes=${nodes}"
    if [[ "$DRY_RUN" == "0" ]]; then
        sbatch -A "$ACCOUNT" -p "$PARTITION" \
            --nodes="$nodes" --cpus-per-task="$CPUS_PER_TASK" --time="$TIME_LIMIT" \
            --requeue --job-name="eai-${day}" \
            tutorials/eai_crawl/submit.sh
        submitted=$((submitted + 1))
    fi
done

tib=$(awk "BEGIN{printf \"%.2f\", ${total_bytes}/1099511627776}")
echo "=================================================="
echo "  Days seen: ${total_days}  submitted: ${submitted}  skipped: ${skipped}"
echo "  Bytes queued this run: ${tib} TiB"
[[ "$DRY_RUN" != "0" ]] && echo "  (DRY_RUN — nothing submitted; unset DRY_RUN to launch)"
echo "=================================================="
