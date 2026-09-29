# PoneTTS server — dual GPU-backend build.
#
# Two independent, self-contained stages below: pick one explicitly at build
# time with --target. Nothing is auto-detected — you choose per host.
#
#   NVIDIA host:  docker build --target runtime-cuda -t pone-ttsmod:cuda .
#   AMD host:     docker build --target runtime-rocm -t pone-ttsmod:rocm .
#
# In docker-compose, set `target: runtime-cuda` or `target: runtime-rocm`
# under this service's `build:` block (see docker-compose.snippet.yml).
#
# Both base images (nvidia/cuda:...-ubuntu22.04, rocm/dev-ubuntu-22.04) ship
# Python 3.10 by default, but numpy==2.5.0 (and onnxruntime-gpu's own numpy
# dependency) require Python >=3.12 — so both stages install 3.12 via
# deadsnakes rather than relying on the base image's system python.
#
# Torch/CUDA/ROCm versions default to what the original project's setup.sh
# pinned (torch==2.11.0). Override per-build if your host has a different
# confirmed-working combination:
#   --build-arg TORCH_VERSION=... --build-arg TORCHAUDIO_VERSION=... \
#   --build-arg TORCH_INDEX_URL=...

########################################
# NVIDIA / CUDA variant
########################################
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04 AS runtime-cuda

ARG TORCH_VERSION=2.11.0
ARG TORCHAUDIO_VERSION=2.11.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # hf_transfer / hf-xet (the fast parallel downloaders) got the container
    # killed mid-download on real hardware (AMD laptop); the plain downloader
    # finished the ~12 GB in under 10 minutes. Override with -e to re-enable.
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    HF_HUB_DISABLE_XET=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ffmpeg git curl ca-certificates build-essential \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
    && python3.12 /tmp/get-pip.py && rm /tmp/get-pip.py \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python

WORKDIR /app
COPY server/requirements.txt .
RUN python3 -m pip install --upgrade pip wheel \
    && python3 -m pip install --index-url "${TORCH_INDEX_URL}" \
         "torch==${TORCH_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}" \
    && python3 -m pip install -r requirements.txt

# faster-whisper depends on the CPU "onnxruntime" package, which installs into
# the same module folder as onnxruntime-gpu and can overwrite it — leaving no
# CUDA vocoder for the quality preset. Remove both and reinstall only the GPU
# build, then fail the build here if its CUDA provider isn't there.
RUN ORT_VER="$(grep -E '^onnxruntime-gpu==' requirements.txt | cut -d= -f3)" \
    && python3 -m pip uninstall -y onnxruntime onnxruntime-gpu \
    && python3 -m pip install --no-deps "onnxruntime-gpu==${ORT_VER}" \
    && python3 -c "import onnxruntime as o; p=o.get_available_providers(); print('onnxruntime providers:', p); assert 'CUDAExecutionProvider' in p, 'onnxruntime-gpu has no CUDA provider'"

COPY server/ .
RUN chmod +x entrypoint.sh

EXPOSE 8800
# Generous start-period: first run downloads a ~12GB model + applies the
# voice patch + does a compile warm-up before the server is even up.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20m --retries=3 \
    CMD curl -fsS "http://localhost:${PORT:-8800}/health" || exit 1

ENTRYPOINT ["./entrypoint.sh"]


########################################
# AMD / ROCm variant
########################################
FROM rocm/dev-ubuntu-22.04:6.4-complete AS runtime-rocm

# NOTE: torch's ROCm wheel builds trail its CUDA builds — 2.11.0 (used in the
# CUDA stage above) has no rocm6.4 wheel yet. 2.9.1 is the latest actually
# published for this index as of this writing (confirmed via
# https://pytorch.org/get-started/previous-versions/). Keep this matched to
# whatever ROCM_WHEEL_INDEX below targets — a wheel built for a different
# ROCm version than what's actually in this base image can fail at runtime
# in less obvious ways than a pip install error.
ARG TORCH_VERSION=2.9.1
ARG TORCHAUDIO_VERSION=2.9.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/rocm6.4

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # hf_transfer / hf-xet (the fast parallel downloaders) got the container
    # killed mid-download on real hardware (AMD laptop); the plain downloader
    # finished the ~12 GB in under 10 minutes. Override with -e to re-enable.
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    HF_HUB_DISABLE_XET=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    # RX 7700S is gfx1102 (RDNA3 mobile, not officially supported by ROCm) —
    # override to the nearest supported target. 11.0.2 maps directly to
    # gfx1102; 11.0.0 (gfx1100) is a DIFFERENT, desktop chip (7900XT) with
    # different specs and has been reported to error on gfx1102 hardware.
    # Change this if your host uses a different GPU/override value.
    HSA_OVERRIDE_GFX_VERSION=11.0.2

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ffmpeg git curl ca-certificates build-essential \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
    && python3.12 /tmp/get-pip.py && rm /tmp/get-pip.py \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python

WORKDIR /app
COPY server/requirements.txt .
RUN python3 -m pip install --upgrade pip wheel \
    && python3 -m pip install --index-url "${TORCH_INDEX_URL}" \
         "torch==${TORCH_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}" \
    && python3 -m pip install -r requirements.txt

# onnxruntime-gpu is CUDA-only, so it's no use on ROCm, and faster-whisper's
# CPU "onnxruntime" can end up mixed with it in the same module folder.
# Remove both and install just the CPU build (same version). The vocoder runs
# on the CPU on AMD whatever the preset.
RUN ORT_VER="$(grep -E '^onnxruntime-gpu==' requirements.txt | cut -d= -f3)" \
    && python3 -m pip uninstall -y onnxruntime onnxruntime-gpu \
    && python3 -m pip install --no-deps "onnxruntime==${ORT_VER}" \
    && python3 -c "import onnxruntime as o; p=o.get_available_providers(); print('onnxruntime providers:', p); assert 'CPUExecutionProvider' in p"

COPY server/ .
RUN chmod +x entrypoint.sh

EXPOSE 8800
# Generous start-period: first run downloads a ~12GB model + applies the
# voice patch + does a compile warm-up before the server is even up.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20m --retries=3 \
    CMD curl -fsS "http://localhost:${PORT:-8800}/health" || exit 1

ENTRYPOINT ["./entrypoint.sh"]
