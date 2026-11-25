# Before you build this

> **AMD (ROCm) support is highly experimental.** NVIDIA is the supported
> path. The AMD image has only been run on one machine (an RX 7700S laptop),
> and it needed manual workarounds to get there. Expect problems on other
> AMD cards.

## The easy way (Linux, or Windows via WSL2)
Run `./run.sh` (inside Ubuntu on WSL2 if you're on Windows). It checks Docker/buildx are working, detects your GPU
vendor and VRAM, picks a matching performance preset, generates a
`docker-compose.yml` for your hardware, and (with your confirmation) builds
and starts it. Re-running it is safe — an existing `docker-compose.yml` gets
backed up, not overwritten.

On WSL2, `run.sh` also installs the NVIDIA Container Toolkit from NVIDIA's apt repo, starts Docker with `sudo service docker start` if systemd is off, and prints LAN-access steps. The voice patch is downloaded automatically on the first start. Keep the project on the WSL filesystem, not `/mnt/c`.

Mac, or want to understand/customize what it's doing? Read on.

## 1. GPU driver, outside Docker, first
Docker does not provide a GPU driver — it only passes through to whatever's
already installed and working on the host. Confirm this in a plain terminal
(no container) before touching this project at all:

- **NVIDIA:** `nvidia-smi` should print your GPU and driver version. Under WSL2 the driver is installed on Windows; never install a Linux driver inside WSL.
- **AMD:** `rocm-smi` should print your GPU.

If neither works, install/fix the host driver first — nothing here can fix
that for you, and the container will fail in confusing ways if you skip this.

## 2. Pick your build target
This Dockerfile has two independent targets — you build exactly one, matching
your actual GPU vendor:

```
docker build --target runtime-cuda -t pone-ttsmod:cuda .   # NVIDIA
docker build --target runtime-rocm -t pone-ttsmod:rocm .   # AMD (highly experimental)
```

In `docker-compose.snippet.yml`, use Option A (NVIDIA) or Option B (AMD) —
whichever matches. Don't mix; there's no auto-detection by design.

## 3. Pick a performance preset for YOUR card's VRAM
Set `MOSS_PERF_PRESET` in the compose file's `environment:` block:

| Preset     | VRAM tier    | Vocoder location |
|------------|--------------|-------------------|
| `quality`  | ~14-16 GB    | GPU (`onnx_cuda`) |
| `balanced` | ~10-12 GB    | CPU (`onnx_cpu`)  |
| `low_vram` | ~7-8 GB      | CPU (`onnx_cpu`)  |
| `potato`   | weakest cards| CPU (`onnx_cpu`)  |

If you have more VRAM than the person who originally set this up, you
probably want `quality` or `balanced` rather than the `low_vram` default —
that's the whole point of a bigger card. Only `quality` uses the GPU vocoder
path; if you hit an error specifically mentioning `onnxruntime`, `cuDNN`, or
CUDA execution providers on that preset, that's the first place to look —
this path is untested by us beyond static code review, since our own
hardware only exercises `low_vram`.

## 4. First start takes a while — that's expected
The first container start downloads the ~12GB model and applies a voice
patch before the server is reachable at all (up to the ~20 minute healthcheck
grace period). `docker compose logs -f pone-ttsmod` to watch progress. Subsequent
restarts skip this (it's cached in the `pone-ttsmod_release` volume) unless that
volume is deleted.

## 5. If the container crash-loops on first start
Check `docker compose logs pone-ttsmod`. The entrypoint now fails fast with a
one-line message if the GPU isn't visible to the container at all — if you
see something else (a deep Python traceback), that's a real bug and worth
reporting back with the full log, not just the last few lines.
