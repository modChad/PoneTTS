#!/usr/bin/env bash
# Container entrypoint: runs prepare.py (the same setup steps setup.sh uses on
# bare metal), checks the GPU, then starts the server. The model download only
# happens once per volume: prepare.py's .setup_done marker lives in release/.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"


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

# Download the model (first start on this volume only), apply the voice patch
# and fix the torchopt bundle. prepare.py skips whatever is already done, so
# dropping adapter_model.safetensors into voice_patch/ later and restarting
# still picks it up.
rc=0; python3 prepare.py || rc=$?
if [ "$rc" -ne 0 ]; then
    explain_exit "setup (prepare.py)" "$rc"
    # Pause before exiting so the restart policy doesn't hammer
    # HuggingFace (and fill the log) in a tight loop.
    sleep 60
    exit "$rc"
fi

DATA_DIR="${MOSS_DATA_DIR:-${MOSS_STATE_DIR:-data}}"
mkdir -p "$DATA_DIR"
if [ -n "${MOSS_PERF_PRESET:-}" ] && [ ! -f "$DATA_DIR/preset.conf" ]; then
    echo "${MOSS_PERF_PRESET}" > "$DATA_DIR/preset.conf"
    echo "[entrypoint] performance preset set to: ${MOSS_PERF_PRESET}"
fi

# --- Pre-set LD_LIBRARY_PATH to pip's bundled NVIDIA libs. -----------------
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
          "isn't visible to this container. Check that the GPU is passed "
          "through (--gpus all, or the nvidia deploy reservation in compose), "
          "that the NVIDIA Container Toolkit is installed, and that "
          "nvidia-smi works on the host outside Docker.", file=sys.stderr)
    sys.exit(1)
print(f"[entrypoint] GPU visible: {torch.cuda.get_device_name(0)}")
PYEOF

echo "[entrypoint] starting server on 0.0.0.0:${PORT:-8800}"
exec python3 server.py --host 0.0.0.0 --port "${PORT:-8800}"
