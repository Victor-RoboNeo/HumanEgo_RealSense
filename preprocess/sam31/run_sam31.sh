#!/usr/bin/env bash

set -euo pipefail


THIS_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")" &&
    pwd
)"

HUMANEGO_ROOT="$(
    cd "$THIS_DIR/../.." &&
    pwd
)"

RUNTIME="$HUMANEGO_ROOT/third_party/sam31_runtime"


die() {
    echo "[SAM31 ERROR] $*" >&2
    exit 1
}


# ============================================================
# Basic checks
# ============================================================

UV_BIN="$(command -v uv || true)"

[[ -n "$UV_BIN" ]] || \
    die "uv is not installed or not on PATH"

[[ -L "$RUNTIME" || -d "$RUNTIME" ]] || \
    die "runtime missing: $RUNTIME"

[[ -d "$RUNTIME/.venv" ]] || \
    die "runtime .venv missing: $RUNTIME/.venv"

[[ -f "$RUNTIME/scripts/humanego_sam31_v2.py" ]] || \
    die "runtime script missing"

[[ -d "$RUNTIME/third_party/sam3" ]] || \
    die "SAM3 source missing"

[[ -f "$RUNTIME/checkpoints/sam3.1/sam3.1_multiplex.pt" ]] || \
    die "SAM3.1 checkpoint missing"


# ============================================================
# IMPORTANT:
#
# HumanEgo runs inside a conda environment.
#
# We MUST NOT let the parent's:
#
#   LD_LIBRARY_PATH
#   LD_PRELOAD
#   PYTHONPATH
#   PYTHONHOME
#
# leak into the SAM3.1 uv process.
#
# Otherwise CUDA/cuDNN from:
#
#   miniconda3/envs/humanego/...
#
# may be loaded together with CUDA/cuDNN from:
#
#   sam31_humanego_smoketest/.venv/...
#
# resulting in ABI/symbol conflicts such as:
#
#   cudnnGetLibConfig
#
# CUDA_VISIBLE_DEVICES is intentionally preserved.
# ============================================================


echo "║ [SAM31-UV] runtime:"
echo "║   $RUNTIME"

echo "║ [SAM31-UV] isolated native environment"
echo "║   LD_LIBRARY_PATH : cleared"
echo "║   LD_PRELOAD      : cleared"
echo "║   PYTHONPATH      : cleared"
echo "║   PYTHONHOME      : cleared"


exec env \
    -u LD_LIBRARY_PATH \
    -u LD_PRELOAD \
    -u PYTHONPATH \
    -u PYTHONHOME \
    VIRTUAL_ENV="$RUNTIME/.venv" \
    "$UV_BIN" run \
        --directory "$RUNTIME" \
        --no-project \
        --no-sync \
        --offline \
        --no-python-downloads \
        --no-config \
        scripts/humanego_sam31_v2.py \
        "$@"