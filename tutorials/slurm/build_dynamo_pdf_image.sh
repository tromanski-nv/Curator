#!/bin/bash
# Build an arm64 nemo-curator image carrying a PREBAKED Dynamo/vLLM 0.23 venv at
# /opt/dynamo-pdf, so Ray's `py_executable` runtime_env can point at it and the
# per-job actor-venv `uv pip install` becomes a no-op.
#
# The venv is built exactly the way Ray builds the actor venv
# (ray/_private/runtime_env/{virtualenv_utils,uv}.py):
#   1. clone the driver venv (/opt/venv) with ray's vendored _clonevirtualenv.py
#   2. `uv pip install -r <reqs>` with the same --override / --torch-backend /
#      --index-strategy / --extra-index-url options Curator's
#      DYNAMO_VLLM_RUNTIME_ENV passes.
# The requirements are the MERGED uv package list Curator produces:
#   ai-dynamo[vllm]==1.3.1  +  albumentations==2.0.8
#
# Runs on a cpu_datamover node -- they are aarch64 like the GB300 trays, so the
# arm64 wheels resolve natively with no emulation. NEVER on the login node: it
# has a 300-thread user cap and heavy local work wedges it.
#
# enroot/mksquashfs CANNOT use Lustre for scratch -- overlayfs refuses a Lustre
# lowerdir and you get "enroot-mksquashovlfs: failed to mount overlay: Invalid
# argument". So ENROOT_{DATA,TEMP,CACHE,RUNTIME}_PATH all point at node-local
# disk. On cpu-dm that is /tmp (492G, ~450G free); /raid exists there too but is
# not user-writable. GPU nodes have a large writable /raid instead.
#
# ~4 minutes end to end. Output is ~50GB against the 30GB base: /opt/dynamo-pdf
# is ~20GB and enroot writes squashfs uncompressed by default. That is +20GB of
# Lustre read per job; measure before assuming it is free at 2000-way fan-out.
#
# Verified on a GB300 (sm_103) after building -- see the header of
# benchmarking/gb300_nemotron_parse_10k_tuned.yaml for why 0.23 specifically:
# vLLM 0.22 cannot run on sm_103 at all.
#SBATCH --job-name=dynamo-pdf-image
#SBATCH --account=nemotron_n4_pre
#SBATCH --partition=cpu_datamover
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --output=/lustre/fsw/portfolios/nemotron/users/tromanski/scratch/imgbuild/build_%j.log

set -euo pipefail

LUSTRE=/lustre/fsw/portfolios/nemotron/users/tromanski
SRC_SQSH="$LUSTRE/containers/nemo-curator-nightly-2026-09-10-arm64.sqsh"
OUT_SQSH="$LUSTRE/containers/nemo-curator-nightly-2026-09-10-arm64-dynamo-pdf.sqsh"
STAGE="/tmp/tromanski/dynamo-pdf-build"
CNAME="dynamopdf$$"

export ENROOT_DATA_PATH="$STAGE/data"
export ENROOT_TEMP_PATH="$STAGE/temp"
export ENROOT_CACHE_PATH="$STAGE/cache"
export ENROOT_RUNTIME_PATH="$STAGE/runtime"
mkdir -p "$ENROOT_DATA_PATH" "$ENROOT_TEMP_PATH" "$ENROOT_CACHE_PATH" "$ENROOT_RUNTIME_PATH" "$STAGE/uvcache" "$STAGE/out"

echo "=== node=$(hostname) $(date -Is)"
df -h /tmp

cleanup() { enroot remove -f "$CNAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "=== [1/4] enroot create from $SRC_SQSH"
time enroot create --name "$CNAME" "$SRC_SQSH"
df -h /tmp

cat > "$STAGE/inside.sh" <<'INNER'
set -euo pipefail
export PATH=/root/.local/bin:$PATH
export UV_CACHE_DIR=/uvcache
export HOME=/root

BASE=/opt/venv
TARGET=/opt/dynamo-pdf
PY="$BASE/bin/python"
RAYV=$("$PY" -c 'import ray; print(ray.__version__)')
SITE=$("$PY" -c 'import ray,os; print(os.path.dirname(ray.__file__))')
CLONE="$SITE/_private/runtime_env/_clonevirtualenv.py"
echo "[inside] ray=$RAYV clone=$CLONE uv=$(uv --version)"

# Same --override file ensure_actor_overrides_on_all_nodes() writes:
#  * ray pinned to the driver's patch (Ray rejects a mismatch)
#  * nixl-cu13 excluded so the cu12 NIXL backend is kept
#  * quack-kernels>=0.4.1: vLLM pins nvidia-cutlass-dsl==4.5.2 but accepts
#    quack-kernels>=0.3.3, and quack 0.4.0 does `from cutlass.base_dsl import
#    Arch`, which 4.5.2 does not export -> EngineCore dies on the ViT
#    flash-attn path. (upstream PR #2391)
cat > /tmp/overrides.txt <<EOF
ray==$RAYV
nixl-cu13 ; sys_platform == 'never'
quack-kernels>=0.4.1
EOF
cat /tmp/overrides.txt

# Merged uv package list from DYNAMO_VLLM_RUNTIME_ENV + the Nemotron-Parse
# model runtime_env.
cat > /tmp/requirements.txt <<'EOF'
ai-dynamo[vllm]==1.3.1
albumentations==2.0.8
EOF

echo "[inside] cloning $BASE -> $TARGET"
"$PY" "$CLONE" "$BASE" "$TARGET"
"$TARGET/bin/python" -c 'import sys; print("[inside] clone python", sys.executable, sys.version)'

echo "[inside] uv pip install"
uv pip install --python "$TARGET/bin/python" \
    -r /tmp/requirements.txt \
    --override /tmp/overrides.txt \
    --torch-backend cu129 \
    --index-strategy unsafe-best-match \
    --extra-index-url https://wheels.vllm.ai/0.23.0/cu129

echo "[inside] === resulting versions ==="
"$TARGET/bin/python" - <<'EOF'
import importlib.metadata as m
for p in ("vllm","torch","transformers","quack-kernels","nvidia-cutlass-dsl",
          "flashinfer-python","ai-dynamo","ray","albumentations"):
    try:
        print(f"  {p}=={m.version(p)}")
    except Exception as e:
        print(f"  {p}: MISSING ({e})")
EOF

echo "[inside] import smoke (CPU-only; GPU checks happen on a GB300)"
"$TARGET/bin/python" -c 'import vllm; print("vllm", vllm.__version__)'
"$TARGET/bin/python" -c 'import quack; print("quack ok", quack.__file__)'
"$TARGET/bin/python" -c 'import ray; print("ray", ray.__version__)'
"$TARGET/bin/python" -c 'import albumentations; print("albumentations", albumentations.__version__)'

# Base venv must be untouched.
"$BASE/bin/python" -c 'import importlib.metadata as m; print("[inside] base venv vllm ==", m.version("vllm"))'

du -sh "$TARGET"
INNER

echo "=== [2/4] building /opt/dynamo-pdf inside the container"
time enroot start --root --rw \
    --mount "$STAGE/uvcache:/uvcache" \
    --mount "$STAGE/inside.sh:/inside.sh" \
    "$CNAME" bash /inside.sh

echo "=== [3/4] strip uv cache bind + export"
df -h /tmp
time enroot export --output "$STAGE/out/$(basename "$OUT_SQSH")" "$CNAME"
ls -la "$STAGE/out"

echo "=== [4/4] copy to Lustre"
time cp "$STAGE/out/$(basename "$OUT_SQSH")" "$OUT_SQSH.tmp"
mv "$OUT_SQSH.tmp" "$OUT_SQSH"
ls -la "$OUT_SQSH"
echo "=== DONE $(date -Is)"
