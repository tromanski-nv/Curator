#!/bin/bash
# =============================================================================
# EAI WARC -> PDF URL pipeline — full-crawl driver (set-and-forget)
#
# Two grouping modes (GROUP_BY):
#   bytes (default) — flatten ALL WARCs (across every day) into fixed byte-size
#                     chunks so each job runs ~TARGET_HOURS, regardless of how
#                     big/small individual days are. Chunk size is derived from a
#                     configurable observed throughput (PER_NODE_MBPS). One SLURM
#                     job per chunk; a chunk may span multiple days.
#   day             — one SLURM job per crawl day (legacy; runtime varies with
#                     day size).
#
# Both modes are safe to re-run == resume: a unit whose output already exists is
# skipped. Byte-chunk boundaries are deterministic (sorted keys + fixed chunk
# size), so re-running after new days appear only submits the new tail chunks.
#
# Run from the repo root on a LOGIN node (needs rclone + network):
#   export AWS_ACCESS_KEY_ID=$(rclone config show team-vendor-data | sed -n 's/^access_key_id = //p')
#   export AWS_SECRET_ACCESS_KEY=$(rclone config show team-vendor-data | sed -n 's/^secret_access_key = //p')
#   # preview (no jobs created):
#   DRY_RUN=1 bash tutorials/eai_crawl/run_all_days.sh
#   # submit for real:
#   bash tutorials/eai_crawl/run_all_days.sh
#
# Sizing knobs (bytes mode):
#   TARGET_HOURS=3          # desired wall time per job
#   PER_NODE_MBPS=135       # observed sustained throughput per node (MB/s)
#   NODES_PER_JOB=4         # nodes each chunk job requests
#   => bytes/job = PER_NODE_MBPS * 1e6 * 3600 * TARGET_HOURS * NODES_PER_JOB
#
# Common overrides:
#   GROUP_BY=day            # switch back to one-job-per-day
#   ONLY_DAYS="20251015 20260502"   # restrict to specific days
#   EXCLUSIVE=0             # drop --exclusive (default is exclusive nodes)
#   DRY_RUN=1               # preview only
#
# WARC listing is cached (source bucket is immutable): the full recursive S3
# listing runs once into $ENUM_CACHE (default logs/eai_warc_enum.txt) and every
# later run reuses it. Set REFRESH_ENUM=1 to re-list (e.g. after new days land).
# =============================================================================
set -euo pipefail

# --- grouping ---
GROUP_BY="${GROUP_BY:-bytes}"                       # bytes | day
TARGET_HOURS="${TARGET_HOURS:-3}"                  # desired wall time per job
PER_NODE_MBPS="${PER_NODE_MBPS:-135}"              # observed per-node MB/s (measured ~135)
NODES_PER_JOB="${NODES_PER_JOB:-4}"                # nodes per chunk job (bytes mode)

# --- SLURM / cluster ---
ACCOUNT="${ACCOUNT:-nemotron_n4_pre}"
PARTITION="${PARTITION:-cpu_dataprocessing}"       # infinite walltime; account allowed
CPUS_PER_TASK="${CPUS_PER_TASK:-64}"               # whole node (nodes have 64 cores)
WORKERS_PER_NODE="${WORKERS_PER_NODE:-62}"         # usable after Ray reserves ~2 (day mode sizing)
NODES_MAX="${NODES_MAX:-16}"                        # cap per-day node request (day mode)
EXCLUSIVE="${EXCLUSIVE:-1}"                          # 1 => add --exclusive
# Give Slurm a limit a bit above target (partition default is only 31m). Slurm
# accepts hours>24 in H:MM:SS form.
TIME_LIMIT="${TIME_LIMIT:-$((TARGET_HOURS + 1)):00:00}"

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

# --- misc ---
DRY_RUN="${DRY_RUN:-0}"
ONLY_DAYS="${ONLY_DAYS:-}"
CHUNK_DIR="${CHUNK_DIR:-logs/chunks}"               # where per-chunk key manifests are written (Lustre)
# The source bucket is immutable, so the full "size;key" enumeration is cached
# here and reused on every subsequent run. Set REFRESH_ENUM=1 to force a re-list
# (e.g. if new days were added).
ENUM_CACHE="${ENUM_CACHE:-logs/eai_warc_enum.txt}"
REFRESH_ENUM="${REFRESH_ENUM:-0}"

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
unset EAI_URL_LIMIT 2>/dev/null || true   # process ALL WARCs per unit

REPO_ROOT="$(pwd)"

# Build the common sbatch resource flags shared by both modes.
sbatch_common=(-A "$ACCOUNT" -p "$PARTITION" --cpus-per-task="$CPUS_PER_TASK" --time="$TIME_LIMIT" --requeue)
[[ "$EXCLUSIVE" == "1" ]] && sbatch_common+=(--exclusive)

# =============================================================================
# BYTES MODE — flatten all WARCs into fixed-size chunks
# =============================================================================
run_bytes_mode() {
    # bytes/job = rate(MB/s) * 1e6 * seconds * nodes
    local bytes_per_job
    bytes_per_job=$(awk -v r="$PER_NODE_MBPS" -v h="$TARGET_HOURS" -v n="$NODES_PER_JOB" \
        'BEGIN{printf "%d", r*1000000*3600*h*n}')
    local tib_per_job
    tib_per_job=$(awk -v b="$bytes_per_job" 'BEGIN{printf "%.2f", b/1099511627776}')

    echo "=================================================="
    echo "  EAI full-crawl driver — BYTES mode (DRY_RUN=${DRY_RUN})"
    echo "  Source : ${SRC_REMOTE}:${SRC_BUCKET}/${SRC_ROOT}/"
    echo "  Output : ${OUT_REMOTE}:${OUT_BUCKET}/{${PDF_PREFIX},${CDX_PREFIX}}/chunk=<id>/"
    echo "  Target : ~${TARGET_HOURS}h/job @ ${PER_NODE_MBPS} MB/s/node x ${NODES_PER_JOB} nodes"
    echo "           => ${tib_per_job} TiB/chunk, ${CPUS_PER_TASK} cpus/node, time=${TIME_LIMIT}, exclusive=${EXCLUSIVE}"
    echo "  Chunks : ${CHUNK_DIR}/chunk_*.keys"
    echo "=================================================="

    mkdir -p "$CHUNK_DIR"

    # Enumerate every WARC across all days as "size;key", deterministically sorted.
    # The source bucket is immutable, so this full listing is done ONCE and cached
    # to $ENUM_CACHE; later runs reuse it (REFRESH_ENUM=1 forces a re-list).
    # rclone --format "sp" => "<size>;<relpath>"; relpath is under SRC_ROOT/.
    if [[ "$REFRESH_ENUM" == "1" || ! -s "$ENUM_CACHE" ]]; then
        echo "  Listing WARCs (one-time) -> ${ENUM_CACHE} ..."
        mkdir -p "$(dirname "$ENUM_CACHE")"
        local enum_build="${ENUM_CACHE}.tmp.$$"
        rclone lsf -R --files-only --format "sp" "${SRC_REMOTE}:${SRC_BUCKET}/${SRC_ROOT}/" \
            | grep '\.warc\.gz$' \
            | awk -F';' -v root="$SRC_ROOT" '{print $1";"root"/"$2}' \
            | sort -t';' -k2,2 > "$enum_build"
        mv -f "$enum_build" "$ENUM_CACHE"   # atomic: partial listings never poison the cache
    else
        echo "  Reusing cached WARC listing: ${ENUM_CACHE} ($(wc -l < "$ENUM_CACHE") WARCs)"
    fi

    # Per-run view: optionally restrict the cached listing to specific days.
    local enum_tmp
    enum_tmp="$(mktemp)"
    if [[ -n "$ONLY_DAYS" ]]; then
        grep -E "/($(echo "$ONLY_DAYS" | tr ' ' '|'))/" "$ENUM_CACHE" > "$enum_tmp" || true
    else
        cp "$ENUM_CACHE" "$enum_tmp"
    fi

    local total_warcs total_bytes
    total_warcs=$(wc -l < "$enum_tmp")
    total_bytes=$(awk -F';' '{s+=$1} END{printf "%d", s}' "$enum_tmp")
    [[ "$total_warcs" -gt 0 ]] || { echo "No WARCs found." >&2; rm -f "$enum_tmp"; exit 1; }

    # Greedy pack into chunks (pass 1): write one manifest per chunk.
    local chunk_idx=0 cur_bytes=0 cur_n=0 manifest=""
    declare -a M_FILE M_BYTES M_N
    while IFS=';' read -r sz key; do
        [[ -z "$key" ]] && continue
        if (( cur_n == 0 )); then
            manifest="${CHUNK_DIR}/chunk_$(printf '%04d' "$chunk_idx").keys"
            : > "$manifest"
        fi
        printf '%s\n' "$key" >> "$manifest"
        cur_bytes=$((cur_bytes + sz)); cur_n=$((cur_n + 1))
        if (( cur_bytes >= bytes_per_job )); then
            M_FILE[chunk_idx]="$manifest"; M_BYTES[chunk_idx]=$cur_bytes; M_N[chunk_idx]=$cur_n
            chunk_idx=$((chunk_idx + 1)); cur_bytes=0; cur_n=0
        fi
    done < "$enum_tmp"
    if (( cur_n > 0 )); then
        M_FILE[chunk_idx]="$manifest"; M_BYTES[chunk_idx]=$cur_bytes; M_N[chunk_idx]=$cur_n
        chunk_idx=$((chunk_idx + 1))
    fi
    rm -f "$enum_tmp"

    local tib_total
    tib_total=$(awk -v b="$total_bytes" 'BEGIN{printf "%.2f", b/1099511627776}')
    echo "  Planned: ${total_warcs} WARCs, ${tib_total} TiB -> ${chunk_idx} chunk(s)"
    echo "--------------------------------------------------"

    # Pass 2: resume-skip + submit.
    local submitted=0 skipped=0 i
    for (( i = 0; i < chunk_idx; i++ )); do
        local id gib out_pdf out_cdx
        id=$(printf '%04d' "$i")
        gib=$(awk -v b="${M_BYTES[i]}" 'BEGIN{printf "%.1f", b/1073741824}')
        out_pdf="s3://${OUT_BUCKET}/${PDF_PREFIX}/chunk=${id}/"
        out_cdx="s3://${OUT_BUCKET}/${CDX_PREFIX}/chunk=${id}/"

        if [[ -n "$(rclone lsf "${OUT_REMOTE}:${OUT_BUCKET}/${PDF_PREFIX}/chunk=${id}/" 2>/dev/null || true)" ]]; then
            echo "skip  chunk=${id}  (output exists)"; skipped=$((skipped + 1)); continue
        fi

        echo "submit chunk=${id}  (${M_N[i]} WARCs, ${gib} GiB) -> nodes=${NODES_PER_JOB}  keys=${M_FILE[i]}"
        if [[ "$DRY_RUN" == "0" ]]; then
            EAI_S3_PREFIX="" \
            EAI_S3_KEYS_FILE="${REPO_ROOT}/${M_FILE[i]}" \
            EAI_OUTPUT_DIR="$out_pdf" \
            EAI_CDX_OUTPUT_DIR="$out_cdx" \
            sbatch "${sbatch_common[@]}" \
                --nodes="$NODES_PER_JOB" --job-name="eai-chunk-${id}" \
                tutorials/eai_crawl/submit.sh
            submitted=$((submitted + 1))
        fi
    done

    echo "=================================================="
    echo "  Chunks: ${chunk_idx}  submitted: ${submitted}  skipped: ${skipped}"
    [[ "$DRY_RUN" != "0" ]] && echo "  (DRY_RUN — nothing submitted; unset DRY_RUN to launch)"
    echo "=================================================="
}

# =============================================================================
# DAY MODE — one job per crawl day (legacy)
# =============================================================================
run_day_mode() {
    local days
    if [[ -n "$ONLY_DAYS" ]]; then
        days="$ONLY_DAYS"
    else
        days=$(rclone lsf --dirs-only "${SRC_REMOTE}:${SRC_BUCKET}/${SRC_ROOT}/" \
               | sed 's#/$##' | grep -E '^[0-9]{8}$' || true)
    fi
    [[ -n "$days" ]] || { echo "No day directories found under ${SRC_REMOTE}:${SRC_BUCKET}/${SRC_ROOT}/" >&2; exit 1; }

    local total_days=0 submitted=0 skipped=0 total_bytes=0
    echo "=================================================="
    echo "  EAI full-crawl driver — DAY mode (DRY_RUN=${DRY_RUN})"
    echo "  Source : ${SRC_REMOTE}:${SRC_BUCKET}/${SRC_ROOT}/"
    echo "  Output : ${OUT_REMOTE}:${OUT_BUCKET}/{${PDF_PREFIX},${CDX_PREFIX}}/crawl_date=<day>/"
    echo "  Sizing : <= ${NODES_MAX} nodes/day, ${CPUS_PER_TASK} cpus/node, time=${TIME_LIMIT}, exclusive=${EXCLUSIVE}"
    echo "=================================================="

    local day
    for day in $days; do
        total_days=$((total_days + 1))

        local stats bytes count
        stats=$(rclone size --json "${SRC_REMOTE}:${SRC_BUCKET}/${SRC_ROOT}/${day}/" 2>/dev/null || echo '{}')
        bytes=$(sed -n 's/.*"bytes":\([0-9]*\).*/\1/p' <<<"$stats"); bytes="${bytes:-0}"
        count=$(sed -n 's/.*"count":\([0-9]*\).*/\1/p' <<<"$stats"); count="${count:-0}"

        if [[ "$bytes" -eq 0 || "$count" -eq 0 ]]; then
            echo "skip  $day  (empty)"; skipped=$((skipped + 1)); continue
        fi
        if [[ -n "$(rclone lsf "${OUT_REMOTE}:${OUT_BUCKET}/${PDF_PREFIX}/crawl_date=${day}/" 2>/dev/null || true)" ]]; then
            echo "skip  $day  (output exists)"; skipped=$((skipped + 1)); continue
        fi

        local nodes
        nodes=$(( (count + WORKERS_PER_NODE - 1) / WORKERS_PER_NODE ))
        (( nodes < 1 )) && nodes=1
        (( nodes > NODES_MAX )) && nodes=$NODES_MAX

        local gib
        gib=$(awk "BEGIN{printf \"%.1f\", ${bytes}/1073741824}")
        total_bytes=$((total_bytes + bytes))

        echo "submit $day  (${count} WARCs, ${gib} GiB) -> nodes=${nodes}"
        if [[ "$DRY_RUN" == "0" ]]; then
            EAI_S3_PREFIX="${SRC_ROOT}/${day}/" \
            EAI_OUTPUT_DIR="s3://${OUT_BUCKET}/${PDF_PREFIX}/crawl_date=${day}/" \
            EAI_CDX_OUTPUT_DIR="s3://${OUT_BUCKET}/${CDX_PREFIX}/crawl_date=${day}/" \
            sbatch "${sbatch_common[@]}" \
                --nodes="$nodes" --job-name="eai-${day}" \
                tutorials/eai_crawl/submit.sh
            submitted=$((submitted + 1))
        fi
    done

    local tib
    tib=$(awk "BEGIN{printf \"%.2f\", ${total_bytes}/1099511627776}")
    echo "=================================================="
    echo "  Days seen: ${total_days}  submitted: ${submitted}  skipped: ${skipped}"
    echo "  Bytes queued this run: ${tib} TiB"
    [[ "$DRY_RUN" != "0" ]] && echo "  (DRY_RUN — nothing submitted; unset DRY_RUN to launch)"
    echo "=================================================="
}

case "$GROUP_BY" in
    bytes) run_bytes_mode ;;
    day)   run_day_mode ;;
    *)     echo "ERROR: GROUP_BY must be 'bytes' or 'day' (got '${GROUP_BY}')" >&2; exit 1 ;;
esac
