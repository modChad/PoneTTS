#!/usr/bin/env python3
"""Get the server ready to run. Safe to run on every start: each step skips
itself once it's done.

  1. download the model into release/ (~12 GB, first time only)
  2. apply the voice patch (downloads the adapter if needed; see apply_voice_patch.py)
  3. fix the downloaded torchopt bundle (see patch_torchopt_bundle.py)
  4. download the Whisper model the speech checks use (first time only)

Used by entrypoint.sh (Docker), setup.sh / setup.bat (bare metal) and the Colab
notebook, so they can't drift apart. Exit code is non-zero if a step failed.
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RELEASE = HERE / "release"
MARKER = RELEASE / ".setup_done"   # written once the model download has finished
BUNDLE_RUNTIME = RELEASE / "moss_tts_torchopt_runner_bundle" / "portable_tts_runtime.py"
WHISPER_DIR = RELEASE / "whisper"   # server.py loads it from here too
WHISPER_MODEL = os.environ.get("MOSS_WHISPER_MODEL", "base.en")


def run(script: str) -> int:
    return subprocess.call([sys.executable, str(HERE / script)], cwd=HERE)


def download_whisper() -> None:
    """Fetch Whisper now so the first checked sentence doesn't wait for it.
    Not fatal: the server can still download it on first use."""
    marker = WHISPER_DIR / f".{WHISPER_MODEL.replace('/', '_')}.ok"
    if marker.exists():
        print(f"[prepare] Whisper ({WHISPER_MODEL}) already downloaded.")
        return
    print(f"[prepare] downloading Whisper ({WHISPER_MODEL}, first time only)...", flush=True)
    try:
        from faster_whisper.utils import download_model
        download_model(WHISPER_MODEL, cache_dir=str(WHISPER_DIR))
        marker.touch()
        print("[prepare] Whisper downloaded.", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[prepare] WARNING: could not download Whisper ({exc}); the server will try on first use.", flush=True)


def main() -> int:
    if MARKER.exists():
        print("[prepare] model already downloaded.")
    else:
        print("[prepare] downloading the model (~12 GB, first time only)...", flush=True)
        rc = run("download_model.py")
        if rc != 0:
            print(f"[prepare] ERROR: model download failed (exit {rc}).", flush=True)
            return rc
        if not BUNDLE_RUNTIME.exists():
            print(f"[prepare] ERROR: model bundle not found at {BUNDLE_RUNTIME.parent} — download failed.", flush=True)
            return 1
        MARKER.touch()

    for script in ("apply_voice_patch.py", "patch_torchopt_bundle.py"):
        print(f"[prepare] running {script}...", flush=True)
        rc = run(script)
        if rc != 0:
            print(f"[prepare] ERROR: {script} failed (exit {rc}).", flush=True)
            return rc
    download_whisper()
    print("[prepare] ready.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
