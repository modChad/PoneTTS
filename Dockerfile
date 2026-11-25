# PoneTTS server (NVIDIA / CUDA).
#
#   docker build --target runtime-cuda -t pone-ttsmod:cuda .
#
# The base image ships Python 3.10, but numpy==2.5.0 (and onnxruntime-gpu's own
# numpy dependency) need Python >=3.12, so 3.12 is installed via deadsnakes.
#
# torch defaults to 2.11.0 (CUDA 12.8). Override per build if needed:
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
    # killed mid-download on real hardware; the plain downloader
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
