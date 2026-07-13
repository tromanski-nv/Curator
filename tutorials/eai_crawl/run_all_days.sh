#!/bin/bash
# =============================================================================
# EAI WARC -> PDF URL pipeline — full-crawl driver (set-and-forget)
#
# Two grouping modes (GROUP_BY):
#   bytes (default) — flatten ALL WARCs (across every day) into TIME-balanced
#                     chunks so each job runs ~TARGET_HOURS, regardless of how
#                     big/small individual days (or individual WARCs) are. Each
#                     WARC's cost is modelled as bytes/bandwidth + a fixed
#                     per-WARC overhead, so chunks full of tiny WARCs get FEWER
#                     WARCs (overhead-bound) and chunks of large WARCs get more
#                     bytes (bandwidth-bound). One SLURM job per chunk; a chunk
#                     may span multiple days.
#   day             — one SLURM job per crawl day (legacy; runtime varies with
#                     day size).
#
# Byte mode is safe to re-run: only a digest-matching `_SUCCESS` marker skips a
# chunk, while unmarked partial output resumes through native checkpoints.
# Byte-chunk boundaries are deterministic for fixed sizing parameters.
#
# Run from the repo root on a LOGIN node (needs rclone + network):
#   export AWS_ACCESS_KEY_ID=$(rclone config show team-vendor-data | sed -n 's/^access_key_id = //p')
#   export AWS_SECRET_ACCESS_KEY=$(rclone config show team-vendor-data | sed -n 's/^secret_access_key = //p')
#   # preview (no jobs created):
#   DRY_RUN=1 bash tutorials/eai_crawl/run_all_days.sh
#   # submit for real:
#   bash tutorials/eai_crawl/run_all_days.sh
#
# Sizing knobs (bytes mode) — TIME-balanced cost model:
#   TARGET_HOURS=3            # desired wall time per job
#   BW_PER_NODE_MBPS=2500     # per-node streaming bandwidth (MB/s), bytes term
#   PER_WARC_OVERHEAD_S=0.05  # amortized per-WARC cost (node-seconds), overhead term
#   NODES_PER_JOB=4           # nodes each chunk job requests
#   => cost(warc) [node-s] = size / (BW_PER_NODE_MBPS*1e6) + PER_WARC_OVERHEAD_S
#      chunk closes when sum(cost) >= TARGET_HOURS*3600*NODES_PER_JOB
#      (i.e. wall time on NODES_PER_JOB nodes ~= TARGET_HOURS)
#   Recalibrate BW/overhead from a real run: pick BW so a large-WARC chunk hits
#   ~TARGET_HOURS, then raise PER_WARC_OVERHEAD_S until tiny-WARC chunks match.
#
# Snapshot (code freeze) knobs:
#   SNAPSHOT=1              # (default) copy repo code to a frozen dir; all jobs
#                          # run from it so you can keep editing the live checkout
#   SNAPSHOT=0             # submit against the live repo (no freeze)
#   SNAPSHOT_ROOT=<dir>    # where snapshots live (default <repo>/../curator-snapshots)
#   VENV_PATH=<dir>        # venv to reuse (default <repo>/.venv)
#
# Common overrides:
#   STREAM_CPUS=0.25        # CPU per WARC stream (lower => more concurrency; default 0.25)
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
# Time-balanced cost model: cost(warc) = size/BW + per-WARC overhead (node-seconds).
BW_PER_NODE_MBPS="${BW_PER_NODE_MBPS:-2500}"       # per-node streaming bandwidth (MB/s)
PER_WARC_OVERHEAD_S="${PER_WARC_OVERHEAD_S:-0.05}" # amortized per-WARC overhead (node-seconds)
NODES_PER_JOB="${NODES_PER_JOB:-4}"                # nodes per chunk job (bytes mode)

# --- SLURM / cluster ---
ACCOUNT="${ACCOUNT:-nemotron_n4_pre}"
PARTITION="${PARTITION:-cpu_dataprocessing}"       # infinite walltime; account allowed
CPUS_PER_TASK="${CPUS_PER_TASK:-64}"               # whole node (nodes have 64 cores)
WORKERS_PER_NODE="${WORKERS_PER_NODE:-62}"         # usable after Ray reserves ~2 (day mode sizing)
NODES_MAX="${NODES_MAX:-16}"                        # cap per-day node request (day mode)
EXCLUSIVE="${EXCLUSIVE:-1}"                          # 1 => add --exclusive
# Submit one SLURM job ARRAY (one task per chunk) instead of N independent jobs:
# collapses monitoring to a single JOBID_[...] and one scancel. Works because the
# time-balanced chunker makes every chunk homogeneous (same nodes/time). Set
# USE_ARRAY=0 for the legacy one-sbatch-per-chunk path. ARRAY_THROTTLE caps how
# many tasks run at once (appended as %N) — useful to avoid overrunning the
# SwiftStack source; empty => unlimited. (bytes mode only.)
USE_ARRAY="${USE_ARRAY:-1}"
ARRAY_THROTTLE="${ARRAY_THROTTLE:-}"
# Give Slurm a limit well above the target so a few stragglers (or SwiftStack
# throttling) don't get the job killed mid-flush. Default = TARGET_HOURS + 90min
# of slack (e.g. 3h target => 4:30:00). Override TIME_BUFFER_MIN or TIME_LIMIT.
TIME_BUFFER_MIN="${TIME_BUFFER_MIN:-90}"
if [[ -z "${TIME_LIMIT:-}" ]]; then
    _tot_min=$(( TARGET_HOURS * 60 + TIME_BUFFER_MIN ))
    TIME_LIMIT="$(printf '%d:%02d:00' $(( _tot_min / 60 )) $(( _tot_min % 60 )))"
fi

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

# --- code snapshot (freeze) ---
# When SNAPSHOT=1 the repo code is copied to a frozen dir and every submitted
# job runs from it (CURATOR_DIR=<snapshot>), so editing the live checkout can't
# affect in-flight jobs. The venv is reused for third-party deps.
SNAPSHOT="${SNAPSHOT:-1}"
SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-}"                  # default set after REPO_ROOT is known
VENV_PATH="${VENV_PATH:-}"                          # default <repo>/.venv

# --- preflight ---
: "${AWS_ACCESS_KEY_ID:?export READ creds first (team-vendor-data access_key_id)}"
: "${AWS_SECRET_ACCESS_KEY:?export READ creds first (team-vendor-data secret_access_key)}"
command -v rclone >/dev/null || { echo "ERROR: rclone not found (run on a login node)" >&2; exit 1; }
command -v sha256sum >/dev/null || { echo "ERROR: sha256sum not found" >&2; exit 1; }
command -v python >/dev/null || { echo "ERROR: python not found" >&2; exit 1; }
[[ -f tutorials/eai_crawl/submit.sh ]] || { echo "ERROR: run from the Curator repo root" >&2; exit 1; }

# CPU reservation per WARC stream task. Streaming is I/O-bound, so reserving a
# full core per stream (1.0) under-subscribes the node (~64 streams) and leaves
# CPUs idle. 0.25 packs ~256 concurrent streams/node to saturate the NIC. Tune
# lower (0.125) for very small WARCs; raise if the source, not concurrency, caps.
STREAM_CPUS="${STREAM_CPUS:-0.25}"
WARCS_PER_TASK="${WARCS_PER_TASK:-32}"

# Static env consumed by submit.sh (exported to each sbatch job).
export EAI_S3_BUCKET="$SRC_BUCKET"
export EAI_S3_ENDPOINT_URL="$ENDPOINT"
export EAI_STREAM=1
export EAI_STREAM_CPUS="$STREAM_CPUS"
export EAI_WARCS_PER_TASK="$WARCS_PER_TASK"
export EAI_OUTPUT_RCLONE_REMOTE="$OUT_REMOTE"
unset EAI_URL_LIMIT 2>/dev/null || true   # process ALL WARCs per unit

REPO_ROOT="$(pwd)"
SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-${REPO_ROOT}/../curator-snapshots}"
VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv}"
_checkpoint_name="${OUT_BUCKET}-${PDF_PREFIX//\//_}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${REPO_ROOT}/logs/eai_checkpoints/${_checkpoint_name}}"
mkdir -p "$CHECKPOINT_ROOT"
export EAI_CHECKPOINT_ROOT="$CHECKPOINT_ROOT"

# Where submitted jobs load code + key manifests from. Overridden to the frozen
# snapshot dir by make_snapshot() when SNAPSHOT=1.
CODE_DIR="$REPO_ROOT"
KEYS_BASE="$REPO_ROOT"

# Freeze the repo so in-flight jobs are immune to edits of the live checkout.
# Copies code + the freshly built chunk manifests into the snapshot, then points
# submitted jobs at it via CURATOR_DIR/VENV_PATH (exported for sbatch).
make_snapshot() {
    command -v rsync >/dev/null || { echo "ERROR: rsync not found (needed for SNAPSHOT=1)" >&2; exit 1; }
    [[ -f "${VENV_PATH}/bin/activate" ]] || {
        echo "ERROR: VENV_PATH=${VENV_PATH} has no venv; set VENV_PATH or run 'uv sync ...'." >&2
        exit 1
    }
    echo "  Freezing code snapshot ..." >&2
    CODE_DIR="$(SNAPSHOT_ROOT="$SNAPSHOT_ROOT" REPO_ROOT="$REPO_ROOT" \
        bash tutorials/eai_crawl/snapshot_repo.sh)"
    # Include the freshly generated chunk manifests so jobs read immutable keys.
    mkdir -p "${CODE_DIR}/${CHUNK_DIR}"
    cp "${CHUNK_DIR}"/*.keys "${CODE_DIR}/${CHUNK_DIR}"/ 2>/dev/null || true
    KEYS_BASE="$CODE_DIR"
    # submit.sh reads these from the environment (sbatch --export=ALL default).
    export CURATOR_DIR="$CODE_DIR"
    export VENV_PATH
    echo "  Snapshot : ${CODE_DIR}" >&2
    echo "  Venv     : ${VENV_PATH} (reused)" >&2
}

# Build the common sbatch resource flags shared by both modes.
sbatch_common=(-A "$ACCOUNT" -p "$PARTITION" --cpus-per-task="$CPUS_PER_TASK" --time="$TIME_LIMIT" --requeue)
[[ "$EXCLUSIVE" == "1" ]] && sbatch_common+=(--exclusive)

# =============================================================================
# BYTES MODE — flatten all WARCs into fixed-size chunks
# =============================================================================
run_bytes_mode() {
    # Time-balanced packing: a chunk closes when the summed per-WARC cost
    #   cost(warc) = size / (BW_PER_NODE_MBPS*1e6) + PER_WARC_OVERHEAD_S   [node-seconds]
    # reaches cap_node_seconds = TARGET_HOURS * 3600 * NODES_PER_JOB, i.e. the
    # chunk takes ~TARGET_HOURS wall time on NODES_PER_JOB nodes.
    local bw_bytes_per_s cap_node_seconds
    bw_bytes_per_s=$(awk -v m="$BW_PER_NODE_MBPS" 'BEGIN{printf "%d", m*1000000}')
    cap_node_seconds=$(awk -v h="$TARGET_HOURS" -v n="$NODES_PER_JOB" 'BEGIN{printf "%d", h*3600*n}')

    echo "=================================================="
    echo "  EAI full-crawl driver — BYTES mode / time-balanced (DRY_RUN=${DRY_RUN})"
    echo "  Source : ${SRC_REMOTE}:${SRC_BUCKET}/${SRC_ROOT}/"
    echo "  Output : ${OUT_REMOTE}:${OUT_BUCKET}/{${PDF_PREFIX},${CDX_PREFIX}}/chunk=<id>/"
    echo "  Target : ~${TARGET_HOURS}h/job on ${NODES_PER_JOB} nodes"
    echo "  Cost   : size/(${BW_PER_NODE_MBPS} MB/s) + ${PER_WARC_OVERHEAD_S}s/WARC"
    echo "           => cap ${cap_node_seconds} node-s/chunk, ${CPUS_PER_TASK} cpus/node, time=${TIME_LIMIT}, exclusive=${EXCLUSIVE}"
    echo "  Chunks : ${CHUNK_DIR}/chunk_*.keys"
    echo "  State  : ${CHECKPOINT_ROOT}/chunk_<id>"
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

    # Greedy pack into chunks (pass 1). Done in a single awk pass: bash while-read
    # over millions of lines (reopening a manifest per line) took ~1h; awk keeps
    # each manifest open and closes it when the chunk fills, so this is seconds.
    # Packing is TIME-balanced: accumulate per-WARC cost (node-seconds) and close
    # the chunk when it reaches cap_node_seconds. awk writes the manifests and a
    # meta file of "idx;path;bytes;nwarcs;cost_node_seconds" per chunk.
    local meta_file
    meta_file="$(mktemp)"
    awk -F';' -v dir="$CHUNK_DIR" -v cap="$cap_node_seconds" -v bw="$bw_bytes_per_s" \
        -v ovh="$PER_WARC_OVERHEAD_S" -v meta="$meta_file" '
        BEGIN { idx = 0; cost = 0; bytes = 0; n = 0; man = "" }
        $2 == "" { next }
        {
            if (n == 0) { man = sprintf("%s/chunk_%04d.keys", dir, idx) }
            print $2 > man          # first write truncates; awk appends after
            cost += ($1 / bw) + ovh
            bytes += $1; n++
            if (cost >= cap) {
                close(man)
                printf "%d;%s;%d;%d;%.1f\n", idx, man, bytes, n, cost >> meta
                idx++; cost = 0; bytes = 0; n = 0
            }
        }
        END {
            if (n > 0) { close(man); printf "%d;%s;%d;%d;%.1f\n", idx, man, bytes, n, cost >> meta }
        }
    ' "$enum_tmp"
    rm -f "$enum_tmp"

    # Load chunk metadata back into arrays.
    local chunk_idx=0
    declare -a M_FILE M_BYTES M_N M_COST
    local _idx _man _bytes _n _cost
    while IFS=';' read -r _idx _man _bytes _n _cost; do
        M_FILE[_idx]="$_man"; M_BYTES[_idx]="$_bytes"; M_N[_idx]="$_n"; M_COST[_idx]="$_cost"
        chunk_idx=$((_idx + 1))
    done < "$meta_file"
    rm -f "$meta_file"

    local tib_total
    tib_total=$(awk -v b="$total_bytes" 'BEGIN{printf "%.2f", b/1099511627776}')
    echo "  Planned: ${total_warcs} WARCs, ${tib_total} TiB -> ${chunk_idx} chunk(s)"
    echo "--------------------------------------------------"

    # Pass 2a: resume-skip. Only a success marker whose manifest digest matches
    # this exact chunk proves completion. Unmarked legacy/partial Parquet is
    # queued and purged exactly once by run_slurm.py before native checkpointing.
    local skipped=0 i
    declare -a RUN_IDX=()
    for (( i = 0; i < chunk_idx; i++ )); do
        local id gib est_h manifest_sha marker_sha marker_path
        id=$(printf '%04d' "$i")
        gib=$(awk -v b="${M_BYTES[i]}" 'BEGIN{printf "%.1f", b/1073741824}')
        # est wall hours on NODES_PER_JOB nodes = cost_node_seconds / nodes / 3600
        est_h=$(awk -v c="${M_COST[i]}" -v n="$NODES_PER_JOB" 'BEGIN{printf "%.1f", c/n/3600}')
        manifest_sha="$(sha256sum "${M_FILE[i]}" | awk '{print $1}')"
        marker_path="${OUT_REMOTE}:${OUT_BUCKET}/${PDF_PREFIX}/chunk=${id}/_SUCCESS"
        marker_sha="$(
            rclone cat "$marker_path" 2>/dev/null \
                | python -c 'import json,sys; print(json.load(sys.stdin).get("manifest_sha256", ""))' \
                2>/dev/null || true
        )"
        if [[ -n "$marker_sha" ]]; then
            if [[ "$marker_sha" != "$manifest_sha" ]]; then
                echo "ERROR: ${marker_path} belongs to a different chunk manifest." >&2
                echo "  Existing: ${marker_sha}" >&2
                echo "  Current : ${manifest_sha}" >&2
                echo "  Use a fresh output/checkpoint namespace when rechunking." >&2
                exit 1
            fi
            echo "skip  chunk=${id}  (_SUCCESS matches manifest)"; skipped=$((skipped + 1)); continue
        fi
        echo "queue chunk=${id}  (${M_N[i]} WARCs, ${gib} GiB, ~${est_h}h)"
        RUN_IDX+=("$i")
    done

    local n_run=${#RUN_IDX[@]}
    echo "--------------------------------------------------"
    if [[ "$n_run" -eq 0 ]]; then
        echo "  Nothing to submit (${chunk_idx} chunks, ${skipped} already done)."
        return 0
    fi

    # Freeze code (+ these manifests) so live edits can't affect queued jobs.
    # Skipped on DRY_RUN (nothing is submitted, so no snapshot is needed).
    if [[ "$SNAPSHOT" == "1" && "$DRY_RUN" == "0" ]]; then
        make_snapshot
        echo "--------------------------------------------------"
    fi

    local submitted=0
    if [[ "$USE_ARRAY" == "1" ]]; then
        # Pass 2b (array): a single `sbatch --array=<indices>[%throttle]`. Each
        # array task derives its own chunk keys/outputs from SLURM_ARRAY_TASK_ID
        # (see submit.sh), so we pass only static templates here.
        local spec max_idx max_arr
        max_idx=${RUN_IDX[$((n_run - 1))]}
        if [[ "$skipped" -eq 0 ]]; then
            spec="0-${max_idx}"                      # contiguous fresh run: compact range
        else
            spec="$(IFS=,; echo "${RUN_IDX[*]}")"    # gaps from resume: explicit list
        fi
        [[ -n "$ARRAY_THROTTLE" ]] && spec="${spec}%${ARRAY_THROTTLE}"

        # Guard against MaxArraySize (largest index must be < the cluster limit).
        max_arr="$(scontrol show config 2>/dev/null | awk '/^MaxArraySize/{print $3}')"
        if [[ -n "$max_arr" && "$max_idx" -ge "$max_arr" ]]; then
            echo "ERROR: chunk index ${max_idx} >= MaxArraySize ${max_arr}." >&2
            echo "  Split the run (e.g. ONLY_DAYS=...) or use USE_ARRAY=0." >&2
            exit 1
        fi

        echo "submit ARRAY  tasks=${n_run}  indices=[${RUN_IDX[0]}..${max_idx}]${ARRAY_THROTTLE:+ throttle=%${ARRAY_THROTTLE}}  nodes/task=${NODES_PER_JOB}"
        if [[ "$DRY_RUN" == "0" ]]; then
            EAI_S3_PREFIX="" \
            EAI_KEYS_DIR="${KEYS_BASE}/${CHUNK_DIR}" \
            EAI_OUT_BUCKET="${OUT_BUCKET}" \
            EAI_PDF_PREFIX="${PDF_PREFIX}" \
            EAI_CDX_PREFIX="${CDX_PREFIX}" \
            sbatch "${sbatch_common[@]}" \
                --array="${spec}" \
                --nodes="$NODES_PER_JOB" --job-name="eai" \
                --output="logs/eai_warc_%A_%a.log" --error="logs/eai_warc_%A_%a.log" \
                tutorials/eai_crawl/submit.sh
            submitted=$n_run
        fi
    else
        # Pass 2b (legacy): one independent sbatch per chunk.
        for i in "${RUN_IDX[@]}"; do
            local id out_pdf out_cdx
            id=$(printf '%04d' "$i")
            out_pdf="s3://${OUT_BUCKET}/${PDF_PREFIX}/chunk=${id}/"
            out_cdx="s3://${OUT_BUCKET}/${CDX_PREFIX}/chunk=${id}/"
            echo "submit chunk=${id}  keys=${KEYS_BASE}/${M_FILE[i]}"
            if [[ "$DRY_RUN" == "0" ]]; then
                EAI_S3_PREFIX="" \
                EAI_S3_KEYS_FILE="${KEYS_BASE}/${M_FILE[i]}" \
                EAI_OUTPUT_DIR="$out_pdf" \
                EAI_CDX_OUTPUT_DIR="$out_cdx" \
                EAI_CHECKPOINT_PATH="${CHECKPOINT_ROOT}/chunk_${id}" \
                sbatch "${sbatch_common[@]}" \
                    --nodes="$NODES_PER_JOB" --job-name="eai-chunk-${id}" \
                    tutorials/eai_crawl/submit.sh
                submitted=$((submitted + 1))
            fi
        done
    fi

    echo "=================================================="
    echo "  Chunks: ${chunk_idx}  to-run: ${n_run}  submitted: ${submitted}  skipped: ${skipped}  mode: $([[ "$USE_ARRAY" == "1" ]] && echo array || echo per-job)"
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

    if [[ "$REFRESH_ENUM" == "1" || ! -s "$ENUM_CACHE" ]]; then
        echo "  Listing WARCs (one-time) -> ${ENUM_CACHE} ..."
        mkdir -p "$(dirname "$ENUM_CACHE")"
        local enum_build="${ENUM_CACHE}.tmp.$$"
        rclone lsf -R --files-only --format "sp" "${SRC_REMOTE}:${SRC_BUCKET}/${SRC_ROOT}/" \
            | grep '\.warc\.gz$' \
            | awk -F';' -v root="$SRC_ROOT" '{print $1";"root"/"$2}' \
            | sort -t';' -k2,2 > "$enum_build"
        mv -f "$enum_build" "$ENUM_CACHE"
    fi
    mkdir -p "$CHUNK_DIR"
    local day day_manifest
    for day in $days; do
        day_manifest="${CHUNK_DIR}/day_${day}.keys"
        awk -F';' -v needle="/${day}/" 'index($2, needle) {print $2}' "$ENUM_CACHE" > "$day_manifest"
    done

    local total_days=0 submitted=0 skipped=0 total_bytes=0
    echo "=================================================="
    echo "  EAI full-crawl driver — DAY mode (DRY_RUN=${DRY_RUN})"
    echo "  Source : ${SRC_REMOTE}:${SRC_BUCKET}/${SRC_ROOT}/"
    echo "  Output : ${OUT_REMOTE}:${OUT_BUCKET}/{${PDF_PREFIX},${CDX_PREFIX}}/crawl_date=<day>/"
    echo "  Sizing : <= ${NODES_MAX} nodes/day, ${CPUS_PER_TASK} cpus/node, time=${TIME_LIMIT}, exclusive=${EXCLUSIVE}"
    echo "=================================================="

    # Freeze code and the day manifests before any jobs are submitted.
    if [[ "$SNAPSHOT" == "1" && "$DRY_RUN" == "0" ]]; then
        make_snapshot
        echo "--------------------------------------------------"
    fi

    for day in $days; do
        total_days=$((total_days + 1))

        local stats bytes count manifest_sha marker_sha marker_path
        day_manifest="${CHUNK_DIR}/day_${day}.keys"
        stats=$(rclone size --json "${SRC_REMOTE}:${SRC_BUCKET}/${SRC_ROOT}/${day}/" 2>/dev/null || echo '{}')
        bytes=$(sed -n 's/.*"bytes":\([0-9]*\).*/\1/p' <<<"$stats"); bytes="${bytes:-0}"
        count=$(sed -n 's/.*"count":\([0-9]*\).*/\1/p' <<<"$stats"); count="${count:-0}"

        if [[ "$bytes" -eq 0 || "$count" -eq 0 || ! -s "$day_manifest" ]]; then
            echo "skip  $day  (empty)"; skipped=$((skipped + 1)); continue
        fi
        manifest_sha="$(sha256sum "$day_manifest" | awk '{print $1}')"
        marker_path="${OUT_REMOTE}:${OUT_BUCKET}/${PDF_PREFIX}/crawl_date=${day}/_SUCCESS"
        marker_sha="$(
            rclone cat "$marker_path" 2>/dev/null \
                | python -c 'import json,sys; print(json.load(sys.stdin).get("manifest_sha256", ""))' \
                2>/dev/null || true
        )"
        if [[ -n "$marker_sha" ]]; then
            [[ "$marker_sha" == "$manifest_sha" ]] || {
                echo "ERROR: ${marker_path} belongs to a different day manifest." >&2
                exit 1
            }
            echo "skip  $day  (_SUCCESS matches manifest)"; skipped=$((skipped + 1)); continue
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
            EAI_S3_PREFIX="" \
            EAI_S3_KEYS_FILE="${KEYS_BASE}/${day_manifest}" \
            EAI_OUTPUT_DIR="s3://${OUT_BUCKET}/${PDF_PREFIX}/crawl_date=${day}/" \
            EAI_CDX_OUTPUT_DIR="s3://${OUT_BUCKET}/${CDX_PREFIX}/crawl_date=${day}/" \
            EAI_CHECKPOINT_PATH="${CHECKPOINT_ROOT}/day_${day}" \
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
