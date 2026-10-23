#!/usr/bin/env bash
# One-shot setup for the PoneTTS server (Linux + NVIDIA GPU).
#
# Creates a venv, installs deps, downloads the model (~12 GB) from HuggingFace,
# and applies the v2.5 voice patch (pronunciation fix). Run once, then start with:
#     ./tts.py start      (or:  ./run_server.sh --host 0.0.0.0 --port 8000)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY_BIN="${PYTHON:-python3}"
echo "[setup] using $($PY_BIN --version) at $(command -v "$PY_BIN")"

# 1) venv
if [ ! -d venv ]; then
  echo "[setup] creating venv..."
  "$PY_BIN" -m venv venv
fi
PY="$HERE/venv/bin/python"
"$PY" -m pip install --upgrade pip wheel

# 2) torch/torchaudio from the CUDA 12.8 wheel index, then the rest
echo "[setup] installing PyTorch (CUDA 12.8 build)..."
"$PY" -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.11.0" "torchaudio==2.11.0"
echo "[setup] installing server deps..."
"$PY" -m pip install -r requirements.txt

# 3) download the model weights from HuggingFace (public repo)
echo "[setup] downloading model (~12 GB, one time)..."
"$PY" download_model.py

# 4) sanity-check the model bundle downloaded (server imports the runtime as-is)
BUNDLE="release/moss_tts_torchopt_runner_bundle"
if [ ! -f "$BUNDLE/portable_tts_runtime.py" ]; then
  echo "[setup] WARNING: bundle runtime not found at $BUNDLE — download may have failed."
  exit 1
fi

# 5) apply the v2.5 voice patch (pronunciation fine-tune, merged into the model once)
echo "[setup] applying the v2.5 voice patch (a few minutes)..."
"$PY" apply_voice_patch.py

# 6) record a performance preset if one was requested (MOSS_PERF_PRESET=low_vram ./setup.sh)
if [ -n "${MOSS_PERF_PRESET:-}" ]; then
  echo "${MOSS_PERF_PRESET}" > preset.conf
  echo "[setup] performance preset set to: ${MOSS_PERF_PRESET}"
fi

# 7) mark setup COMPLETE — only now is the model fully downloaded + patched. The
# launcher checks this marker (not just the venv) so it never tries to start the
# server while a download is still in progress.
: > "$HERE/.setup_done"

echo
echo "[setup] done."
echo
echo "Pick a performance preset to match your GPU's VRAM (default: quality):"
echo "    ./tts.py preset quality    # ~16 GB cards  (fastest)"
echo "    ./tts.py preset balanced   # ~10-12 GB"
echo "    ./tts.py preset low_vram   # ~8 GB"
echo "    ./tts.py preset potato     # ~6 GB  (as low_vram, shortest line cap)"
echo
echo "Then start the server:"
echo "    ./tts.py start"
