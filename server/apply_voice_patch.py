#!/usr/bin/env python
"""Apply the v2.5 voice patch (run automatically by setup, once).

The base model on HuggingFace slurs uncommon words for characters that had
little screen time (their share of training audio was tiny). This patch is a
LoRA fine-tuned on a speaker-rebalanced mix of the same public show audio; it
fixes pronunciation for the data-starved voices while leaving voice identity
and the emotion/energy controls untouched.

This script merges voice_patch/ into the downloaded checkpoint IN PLACE:
  release/moss_tts_local_clipper_checkpoint          <- becomes the patched model
  release/moss_tts_local_clipper_checkpoint_prepatch <- untouched original (safe to
                                                        delete to reclaim ~6 GB)
Safe to re-run: a marker file makes it a no-op once applied.
Needs ~8 GB free RAM (or a GPU) and ~6 GB free disk for a few minutes.

The adapter (voice_patch/adapter_model.safetensors, ~100 MB) is not stored in
the repo. If it's missing, this script downloads it once from ADAPTER_URL and
checks its SHA-256. Override with MOSS_VOICE_PATCH_URL / MOSS_VOICE_PATCH_SHA256,
or set MOSS_VOICE_PATCH_URL=0 to never download (drop the file in yourself).
"""
import hashlib
import os
import shutil
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CKPT = HERE / "release" / "moss_tts_local_clipper_checkpoint"
PATCH = HERE / "voice_patch"
BACKUP = CKPT.with_name(CKPT.name + "_prepatch")
TMP = CKPT.with_name(CKPT.name + "_patched.tmp")
MARKER = ".voice_patch_v25"
ADAPTER = PATCH / "adapter_model.safetensors"
ADAPTER_URL = os.environ.get("MOSS_VOICE_PATCH_URL", "https://u.pone.rs/qkzcavpd.safetensors").strip()
ADAPTER_SHA256 = os.environ.get(
    "MOSS_VOICE_PATCH_SHA256", "a1847c97bb26c51cba6df1a37e7fe0c5a18659a5125d274daed43f1dc428bbb8"
).strip().lower()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ensure_adapter() -> bool:
    """Make sure the adapter is the real file, downloading it if it's missing.
    The server also reads it at runtime for per-voice blending, so this runs even
    when the patch is already merged."""
    if ADAPTER.exists() and ADAPTER.stat().st_size > 1_000_000:
        return True  # a real-sized file (possibly your own adapter): leave it alone
    if ADAPTER_URL in ("", "0"):
        print("[voice-patch] adapter missing and downloading is off (MOSS_VOICE_PATCH_URL=0).")
        return False
    print(f"[voice-patch] downloading the voice patch (~100 MB) from {ADAPTER_URL} ...", flush=True)
    PATCH.mkdir(parents=True, exist_ok=True)
    part = ADAPTER.with_name(ADAPTER.name + ".part")
    try:
        req = urllib.request.Request(ADAPTER_URL, headers={"User-Agent": "PoneTTS"})
        with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            done, mark = 0, 10
            while True:
                block = r.read(1 << 20)
                if not block:
                    break
                f.write(block)
                done += len(block)
                while total and mark <= 100 and done * 100 >= mark * total:
                    print(f"[voice-patch]   {mark}%  ({done / 1024**2:.0f} MB)", flush=True)
                    mark += 10
        got = _sha256(part)
        if ADAPTER_SHA256 and got != ADAPTER_SHA256:
            raise ValueError(f"checksum mismatch: got {got}, expected {ADAPTER_SHA256}")
        part.replace(ADAPTER)
        print("[voice-patch] download OK, checksum matches.")
        return True
    except Exception as exc:  # noqa: BLE001
        part.unlink(missing_ok=True)
        print(f"[voice-patch] could not download the voice patch: {exc}")
        return False


def main() -> int:
    have_adapter = ensure_adapter()
    if (CKPT / MARKER).exists():
        print("[voice-patch] already applied — nothing to do.")
        return 0
    if not (CKPT / "config.json").exists():
        print("[voice-patch] ERROR: model checkpoint not found — run the model download first.")
        return 1
    if not have_adapter:
        print("[voice-patch] no voice patch — skipping. The server will run on the base")
        print("[voice-patch] (unpatched) model. Drop adapter_model.safetensors into")
        print("[voice-patch] voice_patch/ (or fix the network) and restart to apply it later.")
        return 0

    import torch
    from transformers import AutoModel
    from peft import PeftModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[voice-patch] loading base model on {device} (a few minutes on CPU)...", flush=True)
    model = AutoModel.from_pretrained(CKPT, trust_remote_code=True, torch_dtype=torch.bfloat16)
    if device == "cuda":
        model = model.to(device)
    print("[voice-patch] merging patch into weights...", flush=True)
    model = PeftModel.from_pretrained(model, PATCH)
    model = model.merge_and_unload()

    print("[voice-patch] saving patched checkpoint...", flush=True)
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    model.save_pretrained(TMP, safe_serialization=False)  # lm_head is tied to embeddings
    del model

    # carry over the checkpoint's non-weight files (custom code, tokenizer, configs)
    for f in CKPT.iterdir():
        if f.name.startswith("pytorch_model") or f.name == "__pycache__":
            continue
        dest = TMP / f.name
        if not dest.exists() and f.is_file():
            shutil.copy2(f, dest)
    (TMP / MARKER).write_text("applied\n")

    # atomic-ish swap; keep the original so this is reversible
    if BACKUP.exists():
        shutil.rmtree(BACKUP)
    CKPT.rename(BACKUP)
    TMP.rename(CKPT)
    print("[voice-patch] done. Original kept at:")
    print(f"[voice-patch]   {BACKUP}")
    print("[voice-patch]   (delete that folder to reclaim ~6 GB, or swap it back to unpatch)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
