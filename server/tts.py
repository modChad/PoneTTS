#!/usr/bin/env python3
"""One-command control for the PONE-TTS LAN server.

Usage:
    ./tts.py start      # launch the server (detached, survives this shell)
    ./tts.py stop       # stop the server and free the GPU
    ./tts.py restart    # stop, wait for VRAM to clear, start again
    ./tts.py status     # is it running? health + GPU memory
    ./tts.py clear      # kill the server AND free the GPU (same as stop, explicit)
    ./tts.py logs       # tail the server log
    ./tts.py preset X   # set performance preset: quality | balanced | low_vram
    ./tts.py blend            # show voice-patch blend (default + per-speaker)
    ./tts.py blend 0.6        # default strength 0..1 (0 = original voices, 1 = full patch)
    ./tts.py blend rainbow 0.5   # per-speaker override (name or id; -1 clears it)

No arguments => status.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAUNCHER = HERE / "run_server.sh"
LOG = HERE / "tts_server.log"
PIDFILE = HERE / ".tts_server.pid"
HOST = "0.0.0.0"
PORT = 8000
URL = f"http://127.0.0.1:{PORT}"
PRESET_FILE = HERE / "preset.conf"
PRESETS = ("quality", "balanced", "low_vram", "potato")


SERVICE = "moss-tts.service"


def _preset() -> str:
    """Chosen performance/VRAM preset: env > preset.conf > 'quality'."""
    val = os.environ.get("MOSS_PERF_PRESET", "").strip()
    if not val and PRESET_FILE.exists():
        val = PRESET_FILE.read_text(encoding="utf-8").strip()
    return val if val in PRESETS else "quality"


# --- helpers ---------------------------------------------------------------
def _has_service() -> bool:
    """True if the systemd user service is installed (preferred control path)."""
    r = subprocess.run(
        ["systemctl", "--user", "list-unit-files", SERVICE],
        capture_output=True, text=True,
    )
    return SERVICE in r.stdout


def _svc(*args: str) -> None:
    subprocess.run(["systemctl", "--user", *args, SERVICE])


def _server_pids() -> list[int]:
    """All PIDs running our server.py (robust even if the pidfile is stale)."""
    out = subprocess.run(
        ["pgrep", "-f", "moss-tts/server.py"],
        capture_output=True, text=True,
    ).stdout
    return [int(p) for p in out.split() if p.isdigit()]


def _gpu_used_mb() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()
        return int(out[0])
    except Exception:
        return None


def _http(path: str, timeout: float = 4.0):
    with urllib.request.urlopen(URL + path, timeout=timeout) as r:
        return json.load(r)


def _is_up() -> bool:
    try:
        return _http("/health").get("model_loaded", False)
    except Exception:
        return False


# --- commands --------------------------------------------------------------
def stop() -> None:
    if _has_service():
        # clean systemd stop => no auto-restart (Restart=on-failure only)
        print(f"stopping {SERVICE}...")
        _svc("stop")
    pids = _server_pids()
    if not pids:
        print("server not running.")
    else:
        print(f"stopping server (pids: {', '.join(map(str, pids))})...")
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        # give it a moment, then force-kill any survivors
        time.sleep(3)
        for pid in _server_pids():
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    PIDFILE.unlink(missing_ok=True)

    # wait for the GPU to actually release
    print("waiting for GPU to clear...", end="", flush=True)
    for _ in range(20):
        used = _gpu_used_mb()
        if used is None or used < 2000:
            break
        print(".", end="", flush=True)
        time.sleep(1)
    used = _gpu_used_mb()
    print(f" done. GPU used: {used} MiB" if used is not None else " done.")


def start() -> None:
    if _is_up():
        print(f"server already running and healthy at {URL}")
        return

    if _has_service():
        print(f"starting {SERVICE}...")
        _svc("start")
    else:
        if _server_pids():
            print("a server process exists but isn't healthy; stopping it first.")
            stop()
        preset = _preset()
        print(f"starting server (preset={preset})...")
        with open(LOG, "wb") as logf:
            # setsid => own session so it survives this script exiting
            subprocess.Popen(
                [str(LAUNCHER), "--host", HOST, "--port", str(PORT), "--preset", preset],
                stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                start_new_session=True, cwd=str(HERE),
            )

    print("loading model + warming up (first start ~30-90s)", end="", flush=True)
    for _ in range(90):
        if _is_up():
            print("\nREADY.")
            status()
            return
        if LOG.exists() and any(
            kw in LOG.read_text(errors="ignore")
            for kw in ("Traceback", "Error:", "allocate memory")
        ):
            print(f"\nstartup FAILED — see: {LOG}")
            os.system(f"tail -n 20 {LOG}")
            return
        print(".", end="", flush=True)
        time.sleep(2)
    where = "journalctl --user -u " + SERVICE if _has_service() else str(LOG)
    print(f"\ntimed out waiting for readiness; check: {where}")


def restart() -> None:
    if _has_service():
        print(f"restarting {SERVICE}...")
        _svc("restart")
        print("loading model + warming up", end="", flush=True)
        for _ in range(90):
            if _is_up():
                print("\nREADY.")
                status()
                return
            print(".", end="", flush=True)
            time.sleep(2)
        print("\ntimed out; check: journalctl --user -u " + SERVICE)
        return
    stop()
    start()


def status() -> None:
    pids = _server_pids()
    running = bool(pids)
    healthy = _is_up()
    used = _gpu_used_mb()
    if _has_service():
        active = subprocess.run(
            ["systemctl", "--user", "is-active", SERVICE],
            capture_output=True, text=True,
        ).stdout.strip()
        enabled = subprocess.run(
            ["systemctl", "--user", "is-enabled", SERVICE],
            capture_output=True, text=True,
        ).stdout.strip()
        print(f"service : {active} (boot: {enabled}, auto-restart on crash)")
    print(f"process : {'running ' + str(pids) if running else 'not running'}")
    print(f"health  : {'OK (' + URL + ')' if healthy else 'not responding'}")
    if healthy:
        try:
            g = _http("/gpu")
            print(f"speakers: {_http('/health').get('num_speakers')}")
            print(f"GPU     : {g['free_mb']} MB free / {g['total_mb']} MB "
                  f"(torch reserves {g['torch_reserved_mb']} MB)")
            return
        except Exception:
            pass
    print(f"GPU     : {used} MiB used" if used is not None else "GPU     : n/a")


def set_preset() -> None:
    name = sys.argv[2] if len(sys.argv) > 2 else ""
    if name not in PRESETS:
        print(f"usage: ./tts.py preset <{' | '.join(PRESETS)}>")
        print(f"current: {_preset()}")
        print("  quality  : ~16 GB cards  (fastest, vocoder on GPU)")
        print("  balanced : ~10-12 GB     (vocoder on CPU)")
        print("  low_vram : ~8 GB         (no CUDA-graph, vocoder on CPU, shorter cap)")
        print("  potato   : ~6 GB         (as low_vram, shortest line cap; not quantized)")
        return
    PRESET_FILE.write_text(name + "\n", encoding="utf-8")
    print(f"preset set to '{name}'. Apply with: ./tts.py restart")


def logs() -> None:
    if _has_service():
        os.system(f"journalctl --user -u {SERVICE} -n 40 -f")
        return
    if not LOG.exists():
        print("no log yet.")
        return
    os.system(f"tail -n 40 -f {LOG}")


def _speakers() -> list:
    with urllib.request.urlopen(f"{URL}/speakers", timeout=10) as r:
        return json.load(r)


def _print_blend(cur: dict, speakers: list) -> None:
    names = {s["id"]: s["name"] for s in speakers}
    print(f"default blend : {cur['default']:g}   (0 = original voices, 1 = full pronunciation patch)")
    if cur.get("per_speaker"):
        print("per-speaker   :")
        for k, v in cur["per_speaker"].items():
            print(f"    {names.get(int(k), '?'):15s} ({k}) -> {v:g}")
    print(f"live weights  : {cur['live_scale']:g}x  (re-syncs automatically per line)")


def blend() -> None:
    """Get/set the voice-patch blend on the RUNNING server (live, no restart)."""
    args = sys.argv[2:]
    try:
        speakers = _speakers()
        if not args:
            with urllib.request.urlopen(f"{URL}/voice_blend", timeout=10) as r:
                _print_blend(json.load(r), speakers)
            return
        body: dict = {}
        if len(args) == 1:
            body["scale"] = float(args[0])
        else:
            who, scale = args[0], float(args[1])
            match = [s for s in speakers
                     if str(s["id"]) == who or who.lower() in s["name"].lower()]
            if len(match) != 1:
                opts = ", ".join(s["name"] for s in match) or "none"
                print(f"speaker {who!r} is ambiguous or unknown (matches: {opts})")
                return
            body = {"scale": scale, "speaker_id": match[0]["id"]}
            print(f"-> {match[0]['name']} (id {match[0]['id']})")
        req = urllib.request.Request(f"{URL}/voice_blend", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            _print_blend(json.load(r), speakers)
        print("saved; per-speaker blends apply live as each character speaks.")
    except ValueError:
        print(f"usage: ./tts.py blend [scale] | blend <speaker> <scale>")
    except Exception as exc:
        print(f"blend failed ({exc}). Is the server running? ./tts.py status")


COMMANDS = {
    "start": start, "stop": stop, "restart": restart,
    "status": status, "clear": stop, "logs": logs, "preset": set_preset,
    "blend": blend,
}


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    fn = COMMANDS.get(cmd)
    if fn is None:
        print(__doc__)
        sys.exit(1)
    fn()


if __name__ == "__main__":
    main()
