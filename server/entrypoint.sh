#!/usr/bin/env bash
# Container entrypoint: mirrors what setup.sh did on bare metal, minus the
# venv (the image's system python already has everything installed). Runs
# once per fresh volume — the .setup_done marker lives inside release/ (the
# model volume), so it survives container restarts/recreates but re-runs if
# you ever wipe that volume.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

MARKER="release/.setup_done"
BUNDLE="release/moss_tts_torchopt_runner_bundle"

# Explain a failed step instead of exiting silently (a process killed by a
# signal, e.g. the out-of-memory killer, prints nothing of its own).
explain_exit() {
    local what="$1" rc="$2" why=""
    case "$rc" in
        137) why=" — killed (SIGKILL). Usually out of memory: check 'free -h' / 'dmesg | grep -i oom' on the host." ;;
        139) why=" — crashed (segmentation fault)." ;;
        132) why=" — crashed (illegal instruction: a library needs a CPU feature this machine lacks)." ;;
        134) why=" — aborted (SIGABRT)." ;;
        143) why=" — terminated (SIGTERM)." ;;
        *)   (( rc > 128 )) && why=" — killed by signal $(( rc - 128 ))." ;;
    esac
    echo "[entrypoint] ERROR: $what failed with exit code ${rc}${why}" >&2
    echo "[entrypoint] Disk space for the model volume:" >&2
    df -h release 2>&1 | sed 's/^/[entrypoint]   /' >&2 || true
}

if [ ! -f "$MARKER" ]; then
    echo "[entrypoint] first run on this volume: downloading model (~12 GB)..."
    rc=0; python3 download_model.py || rc=$?
    if [ "$rc" -ne 0 ]; then
        explain_exit "model download" "$rc"
        # Pause before exiting so the restart policy doesn't hammer
        # HuggingFace (and fill the log) in a tight loop.
        sleep 60
        exit "$rc"
    fi

    if [ ! -f "$BUNDLE/portable_tts_runtime.py" ]; then
        echo "[entrypoint] ERROR: model bundle not found at $BUNDLE — download failed."
        exit 1
    fi

    : > "$MARKER"
    echo "[entrypoint] first-run setup complete."
fi

if [ -n "${MOSS_PERF_PRESET:-}" ] && [ ! -f preset.conf ]; then
    echo "${MOSS_PERF_PRESET}" > preset.conf
    echo "[entrypoint] performance preset set to: ${MOSS_PERF_PRESET}"
fi

# --- Reapply the voice patch on every start, not just first-run. It's
# already internally idempotent (checks its own marker inside the checkpoint
# dir) — running it unconditionally means dropping adapter_model.safetensors
# into the (bind-mounted) voice_patch/ folder later and restarting actually
# picks it up, rather than being silently skipped forever because
# $MARKER already exists from an earlier run where the file was missing.
echo "[entrypoint] checking voice patch..."
python3 apply_voice_patch.py

# --- Reapply local fixes to the downloaded torchopt bundle. Runs every start
# (not just first-run) — it's cheap and idempotent, and this way it's not
# silently skipped if the bundle is ever re-downloaded or reset later without
# also clearing $MARKER. See patch_torchopt_bundle.py for what/why.
echo "[entrypoint] checking torchopt bundle patches..."
python3 patch_torchopt_bundle.py

# --- Pre-set LD_LIBRARY_PATH to pip's bundled NVIDIA libs (CUDA build only;
# no-op on ROCm, where no `nvidia` pip packages exist to find). -------------
# The runtime otherwise self-restarts itself via os.execv the first time it
# needs to fix this up, which server.py's own comments warn is unsafe after
# a chdir — and entrypoint.sh chdir's into /app above. Setting this here
# avoids that restart entirely, exactly like bare-metal run_server.sh did.
# It's also what lets onnxruntime-gpu's CUDA execution provider (used by the
# "quality"/"balanced" presets' GPU vocoder) find cuDNN/cuBLAS — those .so
# files ship inside the same pip nvidia-* packages torch already installed,
# so this one lookup covers both needs.
NV_LIBS="$(python3 -c "
import site
from pathlib import Path
dirs=[]
for root in dict.fromkeys(site.getsitepackages()+[site.getusersitepackages()]):
    nv=Path(root)/'nvidia'
    if nv.exists():
        for d in nv.glob('*/lib'):
            if d.is_dir(): dirs.append(str(d))
print(':'.join(dirs))
")"
if [ -n "$NV_LIBS" ]; then
    export LD_LIBRARY_PATH="${NV_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export MOSS_TTS_NVIDIA_LD_LIBRARY_PATH_READY=1
    echo "[entrypoint] LD_LIBRARY_PATH set from pip nvidia-* packages ($(echo "$NV_LIBS" | tr ':' '\n' | wc -l) dirs)"
fi

# --- Fail fast with a clear one-liner instead of a 60-line traceback three
# layers deep in model warmup, if the GPU backend isn't actually usable. ---
python3 - <<'PYEOF'
import sys
import torch
if not torch.cuda.is_available():
    print("[entrypoint] ERROR: torch.cuda.is_available() is False — the GPU "
          "isn't visible to this container. Check that you passed the right "
          "GPU device flags for this backend (--gpus all / nvidia deploy "
          "reservation for CUDA, --device=/dev/kfd --device=/dev/dri for "
          "ROCm) and that the host driver is installed and working outside "
          "Docker first (nvidia-smi / rocm-smi).", file=sys.stderr)
    sys.exit(1)
print(f"[entrypoint] GPU visible: {torch.cuda.get_device_name(0)}")
PYEOF

echo "[entrypoint] starting server on 0.0.0.0:${PORT:-8800}"
exec python3 server.py --host 0.0.0.0 --port "${PORT:-8800}"
