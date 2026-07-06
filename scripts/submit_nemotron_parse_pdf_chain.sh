#!/usr/bin/env bash
# Submit a chain of SLURM jobs that run Nemotron-Parse PDF inference across
# multiple manifests, resuming via --skip-existing when a time-limited
# allocation ends.
#
# Each chain job allocates N nodes via srun; one manifest is processed per srun
# so SlurmRayClient gets a clean Ray cluster bring-up/teardown cycle.
#
# Usage:
#   DATASET=/path/to/data OUTPUT=/path/to/out ACCOUNT=<your_slurm_account> \
#     bash scripts/submit_nemotron_parse_pdf_chain.sh
#
# Continue the chain even when a prior job failed (default here is afterany):
#   DEP_TYPE=afterok bash scripts/submit_nemotron_parse_pdf_chain.sh
#
# Queue a new chain behind an existing job (first job waits afterany:<id>):
#   START_AFTER=344048 NUM_JOBS=10 bash scripts/submit_nemotron_parse_pdf_chain.sh
#
# Two source modes (set SOURCE_MODE, default pdf):
#
#   SOURCE_MODE=pdf   (default) DATASET is a directory tree of PDFs. Build
#     sharded manifests of relative PDF paths first:
#       python scripts/generate_pdf_manifest.py \
#         --pdf-dir "$DATASET" -o "$MANIFEST_DIR" --shard-size 2000
#
#   SOURCE_MODE=jsonl  DATASET holds JSONL shards (base64 PDF content). Build
#     byte-offset manifests first:
#       python scripts/generate_jsonl_manifest.py -i <shard>.jsonl -o "$DATASET" \
#         --manifest-output "$MANIFEST_DIR/<shard>.manifest.jsonl"
#
# MANIFEST_DIR (default $DATASET/manifests) is where *.manifest.jsonl are read
# from; keep it out of a read-only source tree if needed (e.g. under $OUTPUT).

set -euo pipefail

# ---- Paths (override via env) ------------------------------------------------
# SOURCE_MODE=pdf: DATASET is the root of a (nested) PDF tree, passed as --pdf-dir.
# SOURCE_MODE=jsonl: DATASET is the JSONL dataset root, passed as --jsonl-base-dir.
# MANIFEST_DIR holds *.manifest.jsonl. OUTPUT collects one parquet per completed task.
SOURCE_MODE="${SOURCE_MODE:-pdf}"
DATASET="${DATASET:?set DATASET=/path/to/source_root (pdf tree for SOURCE_MODE=pdf)}"
OUTPUT="${OUTPUT:?set OUTPUT=/path/to/parquet_output}"
MANIFEST_DIR="${MANIFEST_DIR:-${DATASET}/manifests}"
CURATOR_DIR="${CURATOR_DIR:-/lustre/fsw/portfolios/nemotron/users/tromanski/github/Curator}"

case "${SOURCE_MODE}" in
  pdf) SOURCE_FLAG="--pdf-dir" ;;
  jsonl) SOURCE_FLAG="--jsonl-base-dir" ;;
  *) echo "Error: SOURCE_MODE must be 'pdf' or 'jsonl' (got '${SOURCE_MODE}')" >&2; exit 1 ;;
esac

# Resolve to absolute paths. Each per-manifest srun below cd's into CURATOR_DIR,
# so a relative DATASET/OUTPUT (and the manifest paths globbed from MANIFEST_DIR)
# would fail to resolve once inside it. readlink -m canonicalizes even when the
# target does not exist yet (e.g. a not-yet-created OUTPUT dir).
DATASET="$(readlink -m "${DATASET}")"
OUTPUT="$(readlink -m "${OUTPUT}")"
MANIFEST_DIR="$(readlink -m "${MANIFEST_DIR}")"
CURATOR_DIR="$(readlink -m "${CURATOR_DIR}")"

if [[ ! -d "${DATASET}" ]]; then
  echo "Error: DATASET is not a directory: ${DATASET}" >&2
  exit 1
fi

if [[ ! -d "${MANIFEST_DIR}" ]]; then
  echo "Error: MANIFEST_DIR is not a directory: ${MANIFEST_DIR}" >&2
  echo "Build manifests first (see header of this script)." >&2
  exit 1
fi

# ---- Pre-flight: self-heal repo + venv before queueing any jobs --------------
# The custom scripts/main.py live only on a feature branch and have twice been
# lost to filesystem purges. Before submitting, guarantee a known-good state so a
# chain can never silently run against missing or wrong code again:
#   1. enforce the expected branch (EXPECT_BRANCH; set to "" to skip),
#   2. restore any tracked files a purge deleted from the work tree (from git),
#   3. require the venv to exist (rebuilding it is a deliberate `uv sync`),
#   4. ensure vLLM's nemotron_parse.py is present + patched -- the patch script
#      self-restores it from .bak, and flock serializes the shared-venv write.
# Any unrecoverable condition aborts here with actionable guidance.
EXPECT_BRANCH="${EXPECT_BRANCH:-tim/pdf-to-interleaved-custom-scripts}"

preflight() {
  local dir="${CURATOR_DIR}"

  if ! git -C "${dir}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "PREFLIGHT ERROR: ${dir} is not a git work tree." >&2
    exit 1
  fi

  # Enforce the branch (checkout if needed; fails safely if the tree is dirty).
  if [[ -n "${EXPECT_BRANCH}" ]]; then
    local cur
    cur="$(git -C "${dir}" symbolic-ref --short -q HEAD || echo DETACHED)"
    if [[ "${cur}" != "${EXPECT_BRANCH}" ]]; then
      echo "PREFLIGHT: on '${cur}', switching to '${EXPECT_BRANCH}'..." >&2
      git -C "${dir}" checkout "${EXPECT_BRANCH}" || {
        echo "PREFLIGHT ERROR: cannot checkout ${EXPECT_BRANCH}; commit/stash local changes first." >&2
        exit 1
      }
    fi
  fi

  # Restore only DELETED tracked files (purge recovery) without clobbering
  # intentional local edits to modified-but-present files.
  local deleted
  deleted="$(git -C "${dir}" ls-files --deleted)"
  if [[ -n "${deleted}" ]]; then
    echo "PREFLIGHT: restoring $(printf '%s\n' "${deleted}" | grep -c .) purged tracked file(s) from git..." >&2
    # shellcheck disable=SC2086
    ( cd "${dir}" && git checkout -- ${deleted} )
  fi

  local f
  for f in "tutorials/interleaved/nemotron_parse_pdf/main.py" "scripts/patch_vllm_nemotron_parse.py"; do
    [[ -f "${dir}/${f}" ]] || { echo "PREFLIGHT ERROR: ${dir}/${f} missing and git could not restore it." >&2; exit 1; }
  done

  if [[ ! -f "${dir}/.venv/bin/activate" ]]; then
    echo "PREFLIGHT ERROR: no venv at ${dir}/.venv -- run: (cd '${dir}' && uv sync --extra interleaved_cuda12)" >&2
    exit 1
  fi

  # Patch vLLM once here (self-restores nemotron_parse.py from .bak if a purge
  # removed it) so the per-manifest srun patch is just a no-op confirmation.
  ( cd "${dir}" && source .venv/bin/activate \
      && flock "${dir}/.venv/.nemotron_parse_patch.lock" python scripts/patch_vllm_nemotron_parse.py ) || {
    echo "PREFLIGHT ERROR: vLLM Nemotron-Parse patch failed (see message above)." >&2
    exit 1
  }

  echo "PREFLIGHT OK: branch=${EXPECT_BRANCH:-<unchecked>}, code present, venv ready, vLLM patched."
}

preflight

# ---- Slurm / scale knobs (override via env) ----------------------------------
NUM_JOBS="${NUM_JOBS:-10}"
MAX_RETRIES="${MAX_RETRIES:-3}"
NODES="${NODES:-2}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
TIME_LIMIT="${TIME_LIMIT:-4:00:00}"
# Pipeline tuning: PDFs per task (sets ~task duration; size from your calibrated
# s/PDF: PDFS_PER_TASK ~= 1800 / s_per_pdf for ~30 min tasks). EXTRA_ARGS is
# appended verbatim to main.py (e.g. EXTRA_ARGS="--max-pages 30 --dpi 200").
PDFS_PER_TASK="${PDFS_PER_TASK:-50}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
# EDIT THESE for your cluster: your Slurm account and partition.
ACCOUNT="${ACCOUNT:?set ACCOUNT=<your_slurm_account>}"
PARTITION="${PARTITION:-batch}"
JOB_PREFIX="${JOB_PREFIX:-sdg_nemotron_parse}"
# afterok: next job runs only if the previous one succeeded; afterany: always continue
DEP_TYPE="${DEP_TYPE:-afterany}"
# Optional: seed the FIRST job of this chain with a dependency on an existing
# Slurm job id (uses DEP_TYPE). Lets a new chain queue behind work already in
# the queue. e.g. START_AFTER=344048 -> job 1 gets --dependency=afterany:344048
START_AFTER="${START_AFTER:-}"

# ---- GPU idle-time exemption (OccupiedIdleGPUsJobReaper) ---------------------
# This pipeline is render/IO-bound: each manifest reads PDFs off Lustre and
# rasterizes pages at high DPI (CPU) before the GPU inference stage is fed, so a
# fraction of GPUs sit idle during per-manifest ramp and can trip the job-reaper.
# Per the cluster's "GPU Idle Time Exemption Guide", we attach an exemption JSON
# to the JOB-level --comment (NOT to inner srun steps). Tune the grace period via
# IDLE_EXEMPT_MINS; set IDLE_EXEMPTION_COMMENT="" to disable entirely.
IDLE_EXEMPT_MINS="${IDLE_EXEMPT_MINS:-60}"
IDLE_EXEMPT_REASON="${IDLE_EXEMPT_REASON:-data_loading}"
IDLE_EXEMPT_DESC="${IDLE_EXEMPT_DESC:-PDF read + high-DPI page rendering (CPU/IO) preprocessing feeds the GPU inference stage; GPUs idle during per-manifest ramp.}"
# If the caller did not set IDLE_EXEMPTION_COMMENT at all, build the default JSON.
# (An explicitly empty value disables the exemption.)
if [[ -z "${IDLE_EXEMPTION_COMMENT+x}" ]]; then
  IDLE_EXEMPTION_COMMENT="{\"OccupiedIdleGPUsJobReaper\":{\"exemptIdleTimeMins\":\"${IDLE_EXEMPT_MINS}\",\"reason\":\"${IDLE_EXEMPT_REASON}\",\"description\":\"${IDLE_EXEMPT_DESC}\"}}"
fi
COMMENT_ARGS=()
if [[ -n "${IDLE_EXEMPTION_COMMENT}" ]]; then
  COMMENT_ARGS=(--comment="${IDLE_EXEMPTION_COMMENT}")
fi

# Shared HuggingFace cache. Point this at a cache that already has
# nvidia/NVIDIA-Nemotron-Parse-v1.2 (or one you can write to for the download).
HF_HOME="${HF_HOME:-/lustre/fsw/portfolios/nemotron/users/tromanski/hf_cache}"
# Absolutize for the same reason as DATASET/OUTPUT: HF_HOME is exported inside the
# srun that cd's into CURATOR_DIR, so a relative value would resolve incorrectly.
HF_HOME="$(readlink -m "${HF_HOME}")"

# Prefer all CPUs on each exclusive node. Use an explicit CPUS_PER_TASK if the
# caller set one, else query sinfo (stripping any trailing '+' on mixed nodes).
# If neither yields a value, omit --cpus-per-task entirely and let --exclusive
# grant all cores rather than passing an unsupported sentinel like -1.
CPUS_ARGS=()
if [[ -n "${CPUS_PER_TASK:-}" ]]; then
  CPUS_ARGS=(--cpus-per-task="${CPUS_PER_TASK}")
elif CPUS_PER_TASK="$(sinfo -h -p "${PARTITION}" -o %c 2>/dev/null | tr -d '+' | head -1)" \
  && [[ -n "${CPUS_PER_TASK}" ]]; then
  CPUS_ARGS=(--cpus-per-task="${CPUS_PER_TASK}")
else
  echo "Note: could not determine CPUs per node; omitting --cpus-per-task (relying on --exclusive)" >&2
fi

mkdir -p "${OUTPUT}" "${OUTPUT}/slurm_logs" "${OUTPUT}/ray_ports"

if [[ -n "${START_AFTER}" && ! "${START_AFTER}" =~ ^[0-9]+(_[0-9]+)?$ ]]; then
  echo "Error: START_AFTER must be a numeric Slurm job id (got '${START_AFTER}')" >&2
  exit 1
fi

# Seeding PREV_JOB_ID makes the first loop iteration attach --dependency just
# like any later link in the chain.
PREV_JOB_ID="${START_AFTER}"
JOB_IDS=()

if [[ -n "${PREV_JOB_ID}" ]]; then
  echo "Seeding chain: job 1 will wait for ${DEP_TYPE}:${PREV_JOB_ID}"
fi

for CHAIN_IDX in $(seq 1 "${NUM_JOBS}"); do
  DEP_ARGS=()
  if [[ -n "${PREV_JOB_ID}" ]]; then
    DEP_ARGS=(--dependency="${DEP_TYPE}:${PREV_JOB_ID}")
  fi

  PREV_JOB_ID="$(
    sbatch --parsable "${DEP_ARGS[@]}" \
      --job-name="${JOB_PREFIX}:${CHAIN_IDX}" \
      --account="${ACCOUNT}" \
      --partition="${PARTITION}" \
      "${COMMENT_ARGS[@]}" \
      --nodes="${NODES}" \
      --ntasks-per-node=1 \
      "${CPUS_ARGS[@]}" \
      --gpus-per-node="${GPUS_PER_NODE}" \
      --time="${TIME_LIMIT}" \
      --exclusive \
      --output="${OUTPUT}/slurm_logs/sdg_%j_${CHAIN_IDX}.out" \
      --error="${OUTPUT}/slurm_logs/sdg_%j_${CHAIN_IDX}.err" \
      <<EOF
#!/bin/bash
set -euo pipefail

echo "============================================================"
echo "  Nemotron-Parse chain job ${CHAIN_IDX}/${NUM_JOBS}"
echo "  SLURM_JOB_ID=\${SLURM_JOB_ID}"
echo "  DATASET=${DATASET}"
echo "  OUTPUT=${OUTPUT}"
echo "============================================================"

# Created here so the per-manifest srun below (which exports the same path as
# RAY_PORT_BROADCAST_DIR) can rely on it existing.
RAY_PORT_BROADCAST_DIR="${OUTPUT}/ray_ports/\${SLURM_JOB_ID}"
mkdir -p "\${RAY_PORT_BROADCAST_DIR}"

# Per-manifest .done markers let chained jobs skip finished work without paying
# a fresh Ray/vLLM bring-up just to find --skip-existing has nothing to do.
DONE_DIR='${OUTPUT}/done_markers'
mkdir -p "\${DONE_DIR}"

MANIFESTS_OK=0
MANIFESTS_FAILED=0
MANIFESTS_SKIPPED=0

shopt -s nullglob
MANIFESTS=( '${MANIFEST_DIR}'/*.manifest.jsonl )
shopt -u nullglob

if [[ \${#MANIFESTS[@]} -eq 0 ]]; then
  echo "ERROR: no manifests found under ${MANIFEST_DIR}/*.manifest.jsonl"
  exit 1
fi

for MANIFEST in "\${MANIFESTS[@]}"; do
  echo "------------------------------------------------------------"

  MANIFEST_DONE="\${DONE_DIR}/\$(basename "\${MANIFEST}").done"
  if [[ -f "\${MANIFEST_DONE}" ]]; then
    echo "Skipping \${MANIFEST} (already completed: \${MANIFEST_DONE})"
    MANIFESTS_SKIPPED=\$((MANIFESTS_SKIPPED + 1))
    continue
  fi

  # Auto-retire manifests whose remaining PDFs are permanently unprocessable
  # (e.g. every PDF times out or produces 0 output, so srun always exits non-zero
  # even though there is genuinely nothing left to do).
  MANIFEST_FAILS_FILE="\${DONE_DIR}/\$(basename "\${MANIFEST}").fails"
  MANIFEST_FAIL_COUNT=\$(cat "\${MANIFEST_FAILS_FILE}" 2>/dev/null || echo 0)
  MAX_RETRIES="${MAX_RETRIES}"
  if [[ \${MANIFEST_FAIL_COUNT} -ge \${MAX_RETRIES} ]]; then
    echo "WARNING: \${MANIFEST} has failed \${MANIFEST_FAIL_COUNT}/${MAX_RETRIES} times in a row; marking done and skipping (remaining PDFs are likely permanently unprocessable)"
    touch "\${MANIFEST_DONE}"
    rm -f "\${MANIFEST_FAILS_FILE}"
    MANIFESTS_SKIPPED=\$((MANIFESTS_SKIPPED + 1))
    continue
  fi

  echo "Processing \${MANIFEST} (consecutive failures so far: \${MANIFEST_FAIL_COUNT})"

  set +e
  srun --ntasks-per-node=1 bash -c "
    set -euo pipefail
    cd '${CURATOR_DIR}' && source .venv/bin/activate

    # Re-apply the vLLM Nemotron-Parse shard-id fix in case the venv was
    # rebuilt by 'uv sync' since the last run (idempotent; no-op if patched).
    # The venv is a SINGLE shared copy, so letting every node patch concurrently
    # races and truncates nemotron_parse.py (breaks model inspection on all
    # ranks). Serialize with a shared-FS lock: exactly one writer at a time; the
    # rest then observe 'already patched' and no-op. Pre-patch once before submit
    # so even the first holder just confirms the fix.
    flock '${CURATOR_DIR}/.venv/.nemotron_parse_patch.lock' python scripts/patch_vllm_nemotron_parse.py || true

    export TMPDIR=/tmp/vllm_\${USER}
    export RAY_TMPDIR=/tmp/ray_\${SLURM_JOB_ID}
    mkdir -p /tmp/vllm_\${USER} /tmp/ray_\${SLURM_JOB_ID}
    export HF_HOME='${HF_HOME}'
    export HF_HUB_CACHE=\${HF_HOME}/hub
    export RAY_PORT_BROADCAST_DIR='${OUTPUT}/ray_ports/\${SLURM_JOB_ID}'

    echo \"[\$(hostname)] SLURM_JOB_ID=\${SLURM_JOB_ID} SLURM_NODEID=\${SLURM_NODEID} SLURM_NNODES=\${SLURM_NNODES} SLURM_CPUS_ON_NODE=\${SLURM_CPUS_ON_NODE}\"

    python tutorials/interleaved/nemotron_parse_pdf/main.py \\
      --manifest \"\${MANIFEST}\" \\
      ${SOURCE_FLAG} '${DATASET}' \\
      --output-dir '${OUTPUT}' \\
      --backend vllm \\
      --enforce-eager \\
      --skip-existing \\
      --pdfs-per-task ${PDFS_PER_TASK} ${EXTRA_ARGS}
  "
  RC=\$?
  set -e

  if [[ \${RC} -eq 0 ]]; then
    MANIFESTS_OK=\$((MANIFESTS_OK + 1))
    touch "\${MANIFEST_DONE}"
    rm -f "\${MANIFEST_FAILS_FILE}"
  else
    MANIFEST_FAIL_COUNT=\$((MANIFEST_FAIL_COUNT + 1))
    echo "\${MANIFEST_FAIL_COUNT}" > "\${MANIFEST_FAILS_FILE}"
    MANIFESTS_FAILED=\$((MANIFESTS_FAILED + 1))
    echo "WARN: \${MANIFEST} exited with status \${RC} (consecutive failure \${MANIFEST_FAIL_COUNT}/${MAX_RETRIES})"
  fi
done

echo "============================================================"
echo "  Job ${CHAIN_IDX}/${NUM_JOBS} summary: ok=\${MANIFESTS_OK} failed=\${MANIFESTS_FAILED} skipped=\${MANIFESTS_SKIPPED}"
echo "============================================================"

# Fail the Slurm job when any manifest failed so afterok stops the chain.
if [[ \${MANIFESTS_FAILED} -gt 0 ]]; then
  exit 1
fi
EOF
  )"

  JOB_IDS+=("${PREV_JOB_ID}")
  echo "Submitted ${CHAIN_IDX}/${NUM_JOBS}: ${PREV_JOB_ID}  (depends: ${DEP_ARGS[*]:-none})"
done

echo
echo "Chain (${NUM_JOBS} jobs, dependency=${DEP_TYPE}): ${JOB_IDS[*]}"
echo "Logs:  ${OUTPUT}/slurm_logs/"
echo "Cancel entire chain: scancel ${JOB_IDS[*]}"
