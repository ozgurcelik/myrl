#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Reproducible "Option A" environment setup for the GRPO (TRL + vLLM) pipeline.
#
# Builds an isolated uv venv using a single CUDA 12.8 stack that runs across
# every RunPod GPU you rent:
#   Ampere   (sm_86):  A40, A6000, A5000, A4500, A4000
#   Ada      (sm_89):  L4, L40, L40S, RTX 2000/4000/6000 Ada
#   Blackwell(sm_120): RTX PRO 4000/4500/6000   <-- requires this cu128 stack
#
# STORAGE LAYOUT (important on RunPod):
#   - The venv (~15GB) is built on the container ROOT disk (ephemeral). This is
#     intentional: re-run this script on each fresh pod. requirements.lock (in
#     git) is the durable, reproducible artifact -- not the venv.
#   - The uv wheel cache lives on /dev/shm (fast tmpfs) and is cleared at the
#     end to free RAM.
#   - /workspace is a small (~20GB) PERSISTENT network volume with a hard quota,
#     so we only put the HuggingFace model cache there (not the venv/cache).
#
# Idempotent: if requirements.lock exists it installs the exact pinned versions;
# otherwise it resolves requirements.in and freezes the lock for next time.
#
# Usage:
#   bash setup.sh
#   RELOCK=1 bash setup.sh           # ignore existing lock and re-resolve
#   ENV_DIR=/workspace/envs/grpo bash setup.sh   # only if you enlarged /workspace
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Config (override via env) ---------------------------------------------
WORKSPACE="${WORKSPACE:-/workspace}"
ENV_DIR="${ENV_DIR:-/root/grpo-venv}"          # ephemeral, on root disk
PY_VERSION="${PY_VERSION:-3.12}"
TORCH_BACKEND="${TORCH_BACKEND:-cu128}"
REQ_FILE="${REQ_FILE:-$SCRIPT_DIR/requirements.in}"
LOCK_FILE="${LOCK_FILE:-$SCRIPT_DIR/requirements.lock}"
MIN_ROOT_GB="${MIN_ROOT_GB:-17}"               # rough venv footprint guard

# --- Caches: wheels on tmpfs (fast, ephemeral); models on persistent volume -
export UV_CACHE_DIR="${UV_CACHE_DIR:-/dev/shm/uv-cache}"
export TMPDIR="${TMPDIR:-/dev/shm/tmp}"        # keep build temp off the small root
export HF_HOME="${HF_HOME:-$WORKSPACE/.cache/huggingface}"
mkdir -p "$UV_CACHE_DIR" "$TMPDIR" "$HF_HOME" "$(dirname "$ENV_DIR")"
# cache (shm) and venv (root) are different filesystems -> copy, not hardlink
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

echo "==> Config"
echo "    ENV_DIR        = $ENV_DIR        (ephemeral, root disk)"
echo "    PY_VERSION     = $PY_VERSION"
echo "    TORCH_BACKEND  = $TORCH_BACKEND"
echo "    UV_CACHE_DIR   = $UV_CACHE_DIR   (tmpfs, cleared at end)"
echo "    HF_HOME        = $HF_HOME        (persistent)"

# --- 1. Ensure uv -----------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  echo "==> Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

# --- 2. Preflight: driver + root disk space ---------------------------------
echo "==> GPU / driver preflight"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
  DRV_MAJOR="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1 || echo 0)"
  if [ "${DRV_MAJOR:-0}" -lt 570 ]; then
    echo "    WARNING: driver major $DRV_MAJOR < 570; CUDA 12.8 / Blackwell may not work."
    echo "             Prefer pods advertised with CUDA >= 12.8."
  fi
else
  echo "    WARNING: nvidia-smi not found."
fi

ROOT_FREE_GB="$(df -BG --output=avail / | tail -1 | tr -dc '0-9')"
echo "==> Root disk free: ${ROOT_FREE_GB}G (need ~${MIN_ROOT_GB}G for the venv)"
if [ "${ROOT_FREE_GB:-0}" -lt "$MIN_ROOT_GB" ]; then
  echo "    WARNING: low root space. If install fails, point ENV_DIR at a bigger disk."
fi

# --- 3. Create venv ---------------------------------------------------------
echo "==> Creating venv at $ENV_DIR (python $PY_VERSION)"
rm -rf "$ENV_DIR"
uv venv --python "$PY_VERSION" --seed "$ENV_DIR"
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

# --- 4. Install the pinned stack -------------------------------------------
if [ -f "$LOCK_FILE" ] && [ "${RELOCK:-0}" != "1" ]; then
  echo "==> Installing exact pins from $LOCK_FILE"
  uv pip install -r "$LOCK_FILE" --torch-backend="$TORCH_BACKEND"
else
  echo "==> Resolving from $REQ_FILE (will freeze a lock afterwards)"
  uv pip install -r "$REQ_FILE" --torch-backend="$TORCH_BACKEND"
  echo "==> Freezing exact versions -> $LOCK_FILE"
  uv pip freeze > "$LOCK_FILE"
fi

# --- 5. Verify torch supports THIS pod's GPU architecture -------------------
echo "==> Verifying GPU compatibility"
python - <<'PY'
import sys, torch
print("torch", torch.__version__, "| built for CUDA", torch.version.cuda)
archs = torch.cuda.get_arch_list()
print("torch arch list:", archs)
if not torch.cuda.is_available():
    print("WARNING: CUDA not available in this process."); sys.exit(0)
cap = torch.cuda.get_device_capability()
sm = f"sm_{cap[0]}{cap[1]}"
name = torch.cuda.get_device_name(0)
ok = sm in archs
print(f"GPU: {name}  capability={sm}  supported_by_torch={ok}")
if not ok:
    print(f"ERROR: this torch build cannot run {sm}. Use a newer --torch-backend.", file=sys.stderr)
    sys.exit(1)
x = torch.randn(1024, device="cuda")
print("CUDA sanity check OK, sum =", float((x * 2).sum()))
PY

# --- 6. Free the tmpfs wheel cache (reclaim RAM) ----------------------------
echo "==> Clearing tmpfs caches ($UV_CACHE_DIR, $TMPDIR)"
rm -rf "$UV_CACHE_DIR" "$TMPDIR"

echo ""
echo "==> Setup complete."
echo "    Activate:  source $ENV_DIR/bin/activate"
echo "    Models ->  $HF_HOME (persistent, but ~20GB quota: watch model sizes)"
