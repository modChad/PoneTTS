#!/usr/bin/env bash
# Launch the MOSS-TTS-PNY LAN server.
# Pre-sets LD_LIBRARY_PATH to the venv's bundled NVIDIA libs so the runtime does
# not need to os.execv-restart the process (which breaks under chdir).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HERE/venv/bin/python"

NV_LIBS="$("$PY" -c "
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

export LD_LIBRARY_PATH="${NV_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Belt-and-suspenders: tell the runtime the lib path is already set up.
export MOSS_TTS_NVIDIA_LD_LIBRARY_PATH_READY=1

# Let PyTorch's CUDA allocator hand memory back to the driver instead of holding
# fragmented cached segments. This is what lets the ONNX vocoder (a separate CUDA
# allocator) grab its ~1GB Conv buffers on long lines instead of OOMing (the 500s).
# Pure allocator behavior; does not change any audio output.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

exec "$PY" "$HERE/server.py" "$@"
