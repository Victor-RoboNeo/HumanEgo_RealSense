#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$ROOT/.venv}"
SAM3_ROOT="${SAM3_ROOT:-$ROOT/third_party/sam3}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

if ! command -v uv >/dev/null 2>&1; then
  echo "[ERROR] uv not found."
  echo "Install uv first, then rerun this script."
  exit 1
fi

echo "[1/5] Creating Python ${PYTHON_VERSION} venv: $VENV"
uv venv "$VENV" --python "$PYTHON_VERSION"
# shellcheck disable=SC1090
source "$VENV/bin/activate"

echo "[2/5] Installing PyTorch (official SAM3 README currently uses torch 2.10 + cu128)"
uv pip install torch==2.10.0 torchvision --index-url "$TORCH_INDEX_URL"

echo "[3/5] Cloning/updating official facebookresearch/sam3"
mkdir -p "$(dirname "$SAM3_ROOT")"
if [[ -d "$SAM3_ROOT/.git" ]]; then
  git -C "$SAM3_ROOT" pull --ff-only
else
  git clone https://github.com/facebookresearch/sam3.git "$SAM3_ROOT"
fi

echo "[4/5] Installing SAM3 and smoke-test deps"
uv pip install -e "$SAM3_ROOT"
uv pip install opencv-python pillow numpy tqdm huggingface_hub modelscope

echo "[5/5] Environment check"
python "$ROOT/scripts/check_env.py"

cat <<EOF

[OK] Environment created.

Activate it with:
  source "$VENV/bin/activate"

Next:
  1) Request/accept access to facebook/sam3.1 on Hugging Face.
  2) Run: hf auth login
  3) Download checkpoint:
       python "$ROOT/scripts/download_checkpoint.py" --source hf

SAM3 repo:
  $SAM3_ROOT
EOF
