#!/usr/bin/env bash
# Set up SGLang with TRAIL:
#   1. clone SGLang at the commit sglang.patch was written against,
#   2. apply sglang.patch,
#   3. install the patched SGLang into a Python 3.12 virtual environment.
#
# Usage: bash setup.sh
# Env:   SGLANG_DIR (default ./sglang), VENV (default ./.venv)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_REPO="https://github.com/sgl-project/sglang.git"
SGLANG_COMMIT="2b47bd3a348bf5ca52a3a4910b2e22c851798576"
SGLANG_DIR="${SGLANG_DIR:-$ROOT/sglang}"
VENV="${VENV:-$ROOT/.venv}"

# The predictor weights are stored with Git LFS. Fail early if they are still pointers.
for f in "$ROOT/predictors/initial_distilbert/model.safetensors" \
         "$ROOT/predictors/mlp_layer11/model.pt"; do
  if [ ! -f "$f" ] || head -c 100 "$f" | grep -q "git-lfs.github.com"; then
    echo "error: $f is missing or is a Git LFS pointer." >&2
    echo "       Install git-lfs and run 'git lfs pull' in $ROOT first." >&2
    exit 1
  fi
done

if ! command -v uv >/dev/null 2>&1; then
  echo "==> Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "==> SGLang @ ${SGLANG_COMMIT:0:10} in $SGLANG_DIR"
if [ ! -d "$SGLANG_DIR/.git" ]; then
  git clone --filter=blob:none "$SGLANG_REPO" "$SGLANG_DIR"
fi
cd "$SGLANG_DIR"
if git apply --reverse --check "$ROOT/sglang.patch" >/dev/null 2>&1; then
  echo "    sglang.patch is already applied"
else
  if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "error: $SGLANG_DIR has local changes; use a clean checkout." >&2
    exit 1
  fi
  git checkout --quiet "$SGLANG_COMMIT"
  git apply "$ROOT/sglang.patch"
  echo "    applied sglang.patch"
fi

echo "==> Python environment in $VENV"
if [ ! -x "$VENV/bin/python" ]; then
  uv venv --python 3.12 --python-preference only-managed "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
uv pip install -e "$SGLANG_DIR/python"
uv pip install ninja aiohttp datasets

python - <<'EOF'
import sglang, torch
print(f"    sglang {sglang.__version__}, torch {torch.__version__} (CUDA {torch.version.cuda})")
from sglang.srt.managers.trail_predictor import TrailPredictorRuntime  # noqa: F401
print("    TRAIL modules import OK")
EOF

echo "==> Done. Activate the environment with: source $VENV/bin/activate"
