#!/usr/bin/env bash
# Setup script: clones LivePortrait, installs deps, downloads pretrained weights.
# Run from the app/ directory: bash setup.sh

set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR"

echo "[1/5] Cloning LivePortrait..."
if [ ! -d "liveportrait" ]; then
  git clone https://github.com/KwaiVGI/LivePortrait.git liveportrait
else
  echo "    liveportrait/ already exists, skipping clone."
fi

echo "[2/5] Creating virtualenv (venv)..."
# Prefer Python 3.10 or 3.11 — LivePortrait's pinned deps don't have wheels
# for Python 3.12+ on all platforms, which forces source builds (scipy, etc.).
PYBIN=""
for cand in python3.11 python3.10 python3; do
  if command -v "$cand" >/dev/null 2>&1; then
    PYBIN="$cand"; break
  fi
done
if [ -z "$PYBIN" ]; then
  echo "ERROR: no python3 found on PATH" >&2; exit 1
fi
echo "    using $($PYBIN --version) at $(command -v $PYBIN)"

if [ ! -d "venv" ]; then
  "$PYBIN" -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate

echo "[3/5] Installing Python dependencies..."
pip install --upgrade pip wheel setuptools

# Install LivePortrait's own deps FIRST so its numpy 1.x + torch pins win.
# LivePortrait ships a macOS variant (CPU PyTorch + onnxruntime-silicon) —
# use that on Darwin. Run pip from inside liveportrait/ so the `-r` include
# resolves relative paths correctly.
if [ -d liveportrait ]; then
  case "$(uname -s)" in
    Darwin) LP_REQ="requirements_macOS.txt" ;;
    *)      LP_REQ="requirements.txt" ;;
  esac
  echo "    installing LivePortrait requirements: liveportrait/$LP_REQ"
  ( cd liveportrait && pip install --prefer-binary -r "$LP_REQ" )
fi

# Now install the app's own deps. Pins are compatible (numpy<2, opencv<4.11)
# so nothing LivePortrait installed gets clobbered.
pip install --prefer-binary -r requirements.txt

echo "[4/5] Downloading LivePortrait pretrained weights from HuggingFace..."
pip install --upgrade "huggingface_hub[cli]"
mkdir -p liveportrait/pretrained_weights
# `hf` is the new CLI (huggingface_hub >=1.x). The old `huggingface-cli`
# is deprecated. Both `--include`/`--exclude` quirks interpret trailing
# positional args as filenames, so download everything (~2 GB) and let
# the runtime pick what it needs.
if command -v hf >/dev/null 2>&1; then
  hf download KwaiVGI/LivePortrait --local-dir liveportrait/pretrained_weights
else
  huggingface-cli download KwaiVGI/LivePortrait --local-dir liveportrait/pretrained_weights
fi

echo "[5/5] Preparing static directories..."
mkdir -p static/uploads static/outputs

echo ""
echo "Setup complete."
echo "Activate the venv:  source venv/bin/activate"
echo "Run the app:        python app.py"
