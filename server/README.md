# PoneTTS voice server

Local text-to-speech server for the **MOSS-TTS-PNY** model (My Little Pony + Team
Fortress 2 character voices). It runs on your own NVIDIA GPU and serves audio over HTTP
to the game (or any device on your LAN). This is the voice backend for the
PonyPonyParadise TTS mod.

- 43 speakers (ids 0–32 = MLP, 33–42 = TF2). See `GET /speakers`.
- Output: 48 kHz mono WAV.
- Default address: `http://127.0.0.1:8000` (bind `0.0.0.0` to reach it from other
  devices on your network).

## Setup (one time)
Easiest is the launcher in the parent folder (`START-WINDOWS.bat`, or on Linux run
`bash START-LINUX.sh` from a terminal) — it runs setup automatically the first time.
To do it by hand:
```bash
./setup.sh           # Linux/macOS
```
```bat
setup.bat            REM Windows (native, no WSL needed)
```
Creates a venv, installs PyTorch (CUDA 12.8) + deps, then runs `prepare.py`: downloads the
model (~12 GB) from HuggingFace, applies the voice patch and fixes the downloaded bundle.

**Requires an NVIDIA GPU.** The model's codec runs CUDA-only (there is no CPU build),
so a discrete NVIDIA card is mandatory. ~6 GB VRAM minimum on the `potato` preset,
~8 GB on `low_vram`, ~16 GB for full quality (see presets below). Linux; Windows works
under WSL2.

## Performance presets (run on smaller cards)
Pick a preset to trade speed/quality for VRAM:

| preset | VRAM (approx) | notes |
|--------|---------------|-------|
| `quality` | ~14–16 GB | fastest; vocoder on GPU |
| `balanced` **(default)** | ~9–12 GB | LM on GPU (fast), vocoder on CPU |
| `low_vram` | ~7–8 GB | also drops CUDA-graph capture + shorter line cap |
| `potato` | ~6 GB | same as `low_vram` with the shortest line cap (512 tokens). Not quantized — the model bundle has no int8 support. |

**Easiest: the dropdown.** Open the server's page at `http://127.0.0.1:8000` and pick a
preset from the **Quality / VRAM** dropdown at the top. It reloads the model in the
background (~30–90 s, the page shows progress) and remembers your choice across restarts.

From the command line instead:
```bash
./tts.py preset low_vram     # set it (persists)
./tts.py restart             # apply
curl http://127.0.0.1:8000/health   # shows "preset": ...
```
Or per launch: `MOSS_PERF_PRESET=low_vram ./tts.py start`, or directly
`./run_server.sh --preset low_vram`. (API: `POST /preset?name=low_vram` triggers the same
background swap; `GET /preset` lists the options.) The presets only change VRAM/speed —
the **voices and quality are the same model**; lower presets are just slower and cap very
long lines a bit shorter. (No GPU at all is not supported by this model.)

## Start / stop
```bash
./tts.py start      # load model, warm up, run in the background
./tts.py status     # running? health + free VRAM
./tts.py stop       # stop and free the GPU
./tts.py restart
./tts.py logs       # tail the log
```
`start` blocks until the model is warmed up (first start ~30–90 s) then keeps running.
Manual alternative:
```bash
./run_server.sh --host 0.0.0.0 --port 8000 --max-new-tokens 1024
```

## API
- `GET  /health`   — readiness check (use this to gate before playing audio)
- `GET  /speakers` — list voices `[{id,name,label,language,utterances}]`
- `GET  /gpu`      — VRAM diagnostics
- `POST /tts`      — body `{"text","speaker_id",...}` → `audio/wav`
                     (or `"format":"base64"` → JSON `{audio_b64,sample_rate,duration_sec}`)
- `GET  /tts?text=...&speaker_id=31` — quick browser/curl test → `audio/wav`
- `GET  /normalize?text=...` — preview the cleaned text the model will speak (no audio)
- `GET  /`         — tiny HTML test page (pick a voice, type, hear it)

```bash
curl http://127.0.0.1:8000/speakers
curl -X POST http://127.0.0.1:8000/tts -H "Content-Type: application/json" \
  -d '{"text":"Hello from Ponyville!","speaker_id":31}' --output out.wav
```
`POST /tts` optional params (all defaulted): `audio_temperature` (0.0 deterministic …
0.7–0.9 expressive), `audio_top_p`, `audio_top_k`, `audio_repetition_penalty`,
`max_new_tokens`, `trim_silence`, `silence_max_gap_ms`, `normalize`, `expand_numbers`.

## Pronunciation (fixing wrong words / screeching)
The model has no phonemizer — it reads raw text — so exotic punctuation (`…`, `—`, `--`,
`*actions*`) can make it screech or hiss, and some words come out wrong ("beg" → "borg").

The server normalizes every line automatically (default on): **pauses and stops are
preserved** (rewritten to commas/periods), screech-triggering symbols are cleaned, and
numbers/symbols are expanded (`$5` → "five dollars", `50%` → "fifty percent").

For specific words it mispronounces, edit **`data/pronunciations.txt`** (`word = respelling`,
one per line). Spell the word how it should *sound* (`beg = behg`, `colonel = ker nel`).
It **hot-reloads** — edit and the next request picks it up. Preview with:
```bash
curl "http://127.0.0.1:8000/normalize?text=I+beg+you"
```

## Speed vs reliability (vocoder mode)
Set by `MOSS_DECODER_RUNTIME`:

| mode | notes |
|------|-------|
| `onnx_hybrid` **(default)** | short/normal lines decode on GPU (fast); very long lines fall back to CPU, and a GPU-OOM auto-retries on CPU — so it can't run out of VRAM. Best all-round. |
| `onnx_cpu` | vocoder always on system RAM. Slower on long lines, never touches VRAM. |
| `onnx_cuda` | everything on GPU; fastest, but can fail on very long lines on smaller cards. |

```bash
MOSS_DECODER_RUNTIME=onnx_cpu ./run_server.sh --host 0.0.0.0 --port 8000
```
Tuning: `MOSS_VOCODER_GPU_MAX_FRAMES` (hybrid GPU/CPU split, default 600 ≈ ~12 s),
`MOSS_VOCODER_CPU_THREADS` (default = all cores), `MOSS_VOCODER_GPU_MEM_GB` (GPU arena
cap, default 5). Each `/tts` reply includes an `X-Timing` header (`gen=`/`dec=`/`audio=`).

## Troubleshooting
- **Game can't reach the server from another PC:** start with `--host 0.0.0.0` and open
  port 8000 in your firewall **to your LAN only** (e.g. `sudo ufw allow from
  192.168.0.0/16 to any port 8000`). Do not expose it to the internet.
- **503 on a line:** VRAM was momentarily tight on a very long line — it's retryable;
  the hybrid mode normally avoids this by falling back to CPU.
- **First request slow:** that's CUDA warmup; subsequent lines are fast.
- **Out of VRAM at load:** close other GPU apps, or use `MOSS_DECODER_RUNTIME=onnx_cpu`.

## Files
- `server.py` — the HTTP server
- `tts.py` — start/stop/status helper
- `run_server.sh` — launcher (sets up the bundled CUDA libs)
- `prepare.py` — setup steps (model download, voice patch, bundle fix); run by setup and Docker
- `download_model.py` — fetches the model from HuggingFace (run by prepare.py)
- `defaults/` — the starting `pronunciations.txt` and `voice_blend.json`
- `data/` — your settings, history, caches and `pronunciations.txt` (created on first start; `MOSS_DATA_DIR` moves it)
- `release/`, `venv/` — created by setup (model weights + Python env; not shipped)

Model: MOSS-TTS-PNY by ZDisket — https://huggingface.co/ZDisket/MOSS-TTS-PNY
