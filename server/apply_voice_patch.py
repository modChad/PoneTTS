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
"""
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CKPT = HERE / "release" / "moss_tts_local_clipper_checkpoint"
PATCH = HERE / "voice_patch"
BACKUP = CKPT.with_name(CKPT.name + "_prepatch")
TMP = CKPT.with_name(CKPT.name + "_patched.tmp")
MARKER = ".voice_patch_v25"


def main() -> int:
    if (CKPT / MARKER).exists():
        print("[voice-patch] already applied — nothing to do.")
        return 0
    if not (CKPT / "config.json").exists():
        print("[voice-patch] ERROR: model checkpoint not found — run the model download first.")
        return 1
    if not (PATCH / "adapter_model.safetensors").exists():
        print("[voice-patch] voice_patch/adapter_model.safetensors not found — skipping.")
        print("[voice-patch] Server will run on the base (unpatched) model. Drop the file")
        print("[voice-patch] into voice_patch/ and restart the container to apply it later.")
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
