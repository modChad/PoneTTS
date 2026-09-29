"""Download the MOSS-TTS-PNY model into ./release (run by setup.sh).

Pulls only the files the server needs from the public HuggingFace repo. The
weights are NOT bundled in this mod — they live on HuggingFace and are fetched here.
"""
import os
from pathlib import Path

# Plain downloader by default: the fast ones (hf_transfer, hf-xet) got the
# container killed mid-download on real hardware. Set these env vars to
# re-enable them if your machine has RAM to spare.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import snapshot_download

REPO_ID = "ZDisket/MOSS-TTS-PNY"
RELEASE_DIR = Path(__file__).resolve().parent / "release"
RELEASE_DIR.mkdir(parents=True, exist_ok=True)

PATTERNS = [
    "README.md",
    "moss_tts_local_clipper_checkpoint/**",
    "moss_audio_tokenizer/**",
    "istftnet2_decoder4_50hz/istftnet2_decoder.onnx",
    "istftnet2_decoder4_50hz/export_manifest.json",
    "moss_tts_torchopt_runner_bundle/**",
]
path = snapshot_download(
    repo_id=REPO_ID,
    repo_type="model",
    local_dir=str(RELEASE_DIR),
    allow_patterns=PATTERNS,
)
print("done:", path)
