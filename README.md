# PoneTTS

Self-hosted text-to-speech server for the **[MOSS-TTS-PNY](https://huggingface.co/ZDisket/MOSS-TTS-PNY)** model, packaged for Docker.

- 43 character voices: IDs 0–32 are My Little Pony, IDs 33–42 are Team Fortress 2
- 48 kHz mono WAV output
- Voice backend for **Pony Pony Paradise (PPP)** and **SillyTavern**
- Browser test page, OpenAI-compatible and XTTSv2-compatible endpoints

Needs an **NVIDIA** GPU. No local GPU? Run it on Google Colab instead: see [Colab](#colab).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/modChad/PoneTTS/blob/main/colab/pone_tts_colab.ipynb)

## Requirements

| | |
|---|---|
| GPU | NVIDIA, ~6 GB VRAM minimum, see [presets](#performance-presets) |
| OS | Linux, or Windows 11/10 via WSL2 |
| Host driver | `nvidia-smi` lists your card |
| Extra | [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |
| Docker | Engine, Compose plugin, buildx |
| Disk | ~12 GB for the model, plus the image |

The GPU driver must work **outside** Docker first. Docker only passes the GPU through; it can't fix a broken host driver.

**Voice patch:** the voice patch (`server/voice_patch/adapter_model.safetensors`, ~100 MB) isn't in the repo. It's downloaded automatically on the first start and checked against its SHA-256.

## Quick start

```bash
git clone <this repo> PoneTTS
cd PoneTTS
./run.sh 2>&1 | tee run.log
```

`run.sh`:

0. Detects WSL (see below).
1. Checks for the Docker CLI, Engine, Compose and buildx, and offers to install anything missing (pacman, apt, dnf or zypper). It always asks before installing.
2. Detects your GPU vendor and VRAM, and on NVIDIA checks for the NVIDIA Container Toolkit.
3. Suggests a performance preset. You can accept it or pick your own.
4. Asks for a port (default `8800`) and whether `/sfx` may fetch sound effects from the internet.
5. Writes `docker-compose.yml` for your hardware, including a RAM cap so running out of memory restarts the server instead of your desktop.
6. Asks whether to start the container right away or only build the image, then builds (and starts) it.

**First start takes a while.** The container downloads the ~12 GB model into a Docker volume, which can take up to ~20 minutes. Later starts skip this.

Watch it come up:

```bash
docker compose logs -f
```

It's ready when you see:

```text
[entrypoint] GPU visible: <your GPU>
[server] ready. device=cuda ...
```

Then open `http://localhost:8800/` for the test page.

### Windows (WSL2)

1. In PowerShell (admin): `wsl --install -d Ubuntu-22.04`, then reboot.
2. Install the latest NVIDIA driver **on Windows**. Do not install a Linux driver inside WSL. `nvidia-smi` inside Ubuntu must list your card.
3. Inside Ubuntu, clone the repo into the WSL filesystem (e.g. `~/PoneTTS`, **not** `/mnt/c/...`) and run `./run.sh`.

`run.sh` detects WSL and then: offers to install Docker (starting it with `sudo service docker start` if systemd is off), installs the NVIDIA Container Toolkit from NVIDIA's apt repo (pinned to `1.20.1-1`, override with `NVIDIA_CONTAINER_TOOLKIT_VERSION`), warns about `/mnt/c` paths and CRLF line endings, and prints the steps for reaching the server from other devices.

- **Memory:** the RAM limit is based on WSL's allowance. Raise it with `memory=<N>GB` under `[wsl2]` in `%UserProfile%\.wslconfig`, then `wsl --shutdown`.
- **Start Docker at boot:** add `[boot]` / `systemd=true` to `/etc/wsl.conf`, then `wsl --shutdown`.
- **Other devices on your LAN:** `localhost:8800` works from Windows. For other devices, either set `networkingMode=mirrored` under `[wsl2]` in `.wslconfig`, or run (admin PowerShell) `netsh interface portproxy add v4tov4 listenport=8800 listenaddress=0.0.0.0 connectport=8800 connectaddress=<WSL IP>` (get the IP with `hostname -I`; it changes on restart), plus an inbound firewall rule for the port.

## Adding it to an existing Compose setup

To run PoneTTS alongside other services (e.g. SillyTavern), add a service block like this to your own `docker-compose.yml`:

```yaml
services:
  pone-ttsmod:
    build:
      context: /path/to/PoneTTS      # the folder, not a file inside it
      dockerfile: Dockerfile
      target: runtime-cuda
    image: pone-ttsmod:cuda
    container_name: pone-ttsmod
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    ports:
      - "8800:8800"
    volumes:
      - pone-ttsmod_release:/app/release         # the model (~12 GB)
      - ./server/voice_patch:/app/voice_patch    # the voice patch is downloaded here
      - ./server/data:/app/data                  # settings, history, caches, pronunciations.txt
    environment:
      - MOSS_PERF_PRESET=balanced
      - MOSS_DATA_DIR=/app/data

volumes:
  pone-ttsmod_release:
```

- `context` must point at the PoneTTS **folder**. If it's missing, Docker builds from the wrong folder and fails with `"/server": not found`.
- If you already built the image with `run.sh`, you can drop the `build:` block and keep `image: pone-ttsmod:cuda`.
- Behind a VPN container (e.g. `network_mode: "service:gluetun"`), remove `ports:` and publish `8800` on the VPN container instead. The server uses port `8800` rather than `8000` so it doesn't clash with gluetun's control server.

`docker-compose.snippet.yml` has a full example with every setting.

To build the image by hand:

```bash
docker build --target runtime-cuda -t pone-ttsmod:cuda .
```

## Performance presets

Set with `MOSS_PERF_PRESET`, or switch live from the dropdown on the test page (the choice persists across restarts).

| Preset | VRAM | Notes |
|---|---|---|
| `quality` | ~14–16 GB | Fastest. Vocoder on the GPU |
| `balanced` | ~9–12 GB | **Server default.** Model on the GPU, vocoder on the CPU |
| `low_vram` | ~7–8 GB | Also skips CUDA-graph capture; shorter line cap |
| `potato` | ~6 GB | As `low_vram` with the shortest line cap. Not quantized |

The presets only trade speed and VRAM. Every preset uses the same voices and quality.

`run.sh` suggests: under 7 GB `potato`, 7–9 GB `low_vram`, 9–12.5 GB `balanced`, above that `quality`.

## Configuration

Environment variables for the container:

| Variable | Default | What it does |
|---|---|---|
| `MOSS_PERF_PRESET` | `balanced` | Performance preset (see above) |
| `PORT` | `8800` | Port the server listens on inside the container |
| `MOSS_AUDIO_TEMPERATURE` | `0.85` | Default sampling temperature. Lower is steadier, higher is more expressive |
| `MOSS_SFX_ALLOW_NETWORK` | `1` | `0` makes `/sfx` serve only from its cache, with no outgoing requests |
| `MOSS_FIX_STUCK_WORDS` | `1` | Detect and regenerate words that get stuck droning. Runs Whisper on every sentence; `0` is much faster |
| `MOSS_STUCK_WHISPER_GATE` | `1.3` | Stuck-word check: Whisper only listens to sentences this many times longer than expected. Lower catches more, `0` listens to every sentence (slowest) |
| `MOSS_LOOP_GUARD_REL_MULT` | `1.6` | A sentence this many times longer than expected (or one that hits its length limit) is checked with Whisper for repeats and failures |
| `MOSS_LOOP_GUARD_MIN_HEARD` | `0.5` | If Whisper hears less than this share of a checked sentence's words, the sentence counts as failed and is regenerated |
| `MOSS_FAILED_CLAUSE_RETRIES` | `2` | How many times a failed sentence is regenerated before it's sent as-is |
| `MOSS_COMMA_SPLIT_MIN_WORDS` | `20` | Sentences longer than this are also split at commas (lists are kept together). `0` turns comma splitting off |
| `MOSS_SENTENCE_PAUSE_SEC` | `0.12` | Silence inserted between sentences, in seconds. Lower is snappier |
| `MOSS_COMMA_PAUSE_SEC` | `0.08` | Silence between pieces of a sentence split at commas |
| `MOSS_STREAM_BUFFER_SEC` | `3` | Streaming only: seconds of audio generated before playback starts (`0` = off, large = wait for the whole line) |
| `MOSS_PAUSE_SEC` | `0.35` | Silence at an em-dash or ellipsis (a deliberate pause) |
| `MOSS_WHISPER_MODEL` | `base.en` | Whisper model the speech checks use (`tiny.en` is faster, `small.en` more accurate). Downloaded during setup into the model volume, and loaded at start-up when `MOSS_FIX_STUCK_WORDS` or `MOSS_FIX_PRON` is on |
| `MOSS_WHISPER_THREADS` | all cores minus 2 | CPU threads Whisper uses |
| `MOSS_FIX_PRON` | `0` | Check rare words with Whisper and fix their pronunciation (slower) |
| `MOSS_AUTO_RESPELL` | `0` | Automatically respell rare words phonetically |
| `MOSS_DECODER_RUNTIME` | from preset | Override the vocoder: `onnx_cuda`, `onnx_cpu` or `onnx_hybrid` |
| `MOSS_HISTORY_MAX` | `500` | How many generated lines to keep in the history (oldest deleted first). `0` turns history off |
| `MOSS_VOICE_PATCH_URL` / `MOSS_VOICE_PATCH_SHA256` | built-in link / checksum | Where the voice patch is downloaded from on first start. `MOSS_VOICE_PATCH_URL=0` turns the download off |
| `MOSS_DATA_DIR` | `server/data` | The [data folder](#data-folder). Docker uses `/app/data`, mounted from `./server/data` |
| `HF_HUB_ENABLE_HF_TRANSFER` / `HF_HUB_DISABLE_XET` | `0` / `1` | The fast model downloaders are off by default; they got the container killed on test hardware |

## API

All endpoints are on `http://<host>:8800`.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Browser test page: generate, stop generations, browse and play the history |
| `GET` | `/health` | Readiness check (`status`: `ok` or `loading`) |
| `GET` | `/speakers` | List voices: `id`, `name`, `label`, `language` |
| `GET` | `/gpu` | VRAM diagnostics |
| `POST` | `/cancel?scope=mine` | Stop the caller's own generations (matched by `X-Client-App`). `scope=all` stops everyone's |
| `GET` | `/history` | Saved generations, newest first. Filters: `voice`, `client`, `limit` |
| `GET` | `/history/files/<id>.wav` | Play or download a saved generation |
| `DELETE` | `/history/<id>` | Delete one saved generation (`DELETE /history` deletes all) |
| `POST` | `/tts` | Main synthesis endpoint (see below) |
| `GET` | `/tts?text=...&speaker_id=31` | Quick test from a browser or curl |
| `GET` | `/normalize?text=...` | Preview the cleaned-up text the model will actually read |
| `GET` / `POST` | `/preset` | Show or switch the performance preset (`POST /preset?name=low_vram`) |
| `GET` / `POST` | `/voice_blend` | Voice-patch blend settings |
| `GET` | `/sfx?query=...` | Sound effect for a short description (Freesound / BBC, cached) |
| `POST` | `/v1/audio/speech` | OpenAI-compatible (SillyTavern). Always returns WAV |
| `POST` | `/tts_to_audio/` | XTTSv2-compatible (SillyTavern) |
| `GET` | `/tts_stream/` | XTTSv2-compatible streaming URL |
| `POST` | `/set_tts_settings` | XTTSv2-compatible settings (accepted, ignored) |
| `POST` | `/st_cache/generate` | Generate and cache a clip for a chat message |
| `GET` | `/st_cache/...` | Download cached clips |

### `POST /tts`

```bash
curl -X POST http://localhost:8800/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello from Ponyville!", "speaker_id": 31}' \
  --output hello.wav
```

Only `text` is required. Useful optional fields:

| Field | Default | Notes |
|---|---|---|
| `speaker_id` | `31` | See `GET /speakers` |
| `format` | `wav` | `base64` returns JSON (`audio_b64`, `sample_rate`, `duration_sec`) |
| `audio_temperature` | server default | `0` is deterministic; `0.7`–`0.9` is expressive |
| `style_emotion_id` | none | `0`–`11`. If unset, a `[tag]` in the text is used, else Neutral |
| `style_energy` | none | `0`–`1`. If unset, a `[tag]` in the text is used, else `0.5` |
| `style_strength` | none | `1.0` is normal; higher exaggerates the emotion (max ~2.3) |
| `max_new_tokens` | preset | Upper limit on generated length |
| `trim_silence` | `true` | Shorten long pauses at punctuation |
| `normalize` | `true` | Clean punctuation and symbols, apply respellings |
| `expand_numbers` | `true` | `100` → "one hundred" |
| `client_pronunciations` | `{}` | Per-request respellings, e.g. `{"Zecora": "Zeh core ah"}` |
| `fix_stuck_words` | server default | Regenerate if a word gets stuck droning |

Long text is split into clauses at `. ! ? : ;` and em-dashes, so paragraphs and poems don't get cut off. Sentences over 20 words are also split at commas, packed into pieces of about 5–15 words; commas beside short list items (including an Oxford comma) are never split.

### Stopping generations

`POST /cancel` stops generations **between sentences**: the sentence being generated when you cancel still finishes (the model's own code can't be interrupted mid-sentence), then that request stops, and queued requests never start. Cancelled requests get HTTP `409` with `"detail": "cancelled"`. Requests that arrive after the cancel are unaffected.

- `scope=mine` only stops requests from the same client, going by the `X-Client-App` header.
- `scope=all` stops every client's requests.

The test page has **Stop mine** and **Stop all** buttons, and a live line showing what the server is generating and how many requests are queued (also in `GET /health` under `activity`).

### History

Every successfully generated line is saved to `server/data/history/` as a WAV plus a `.json` with its details (full text, voice, client, duration, temperature, preset). File names sort by time and are readable:

```text
20260930-200612-345_twilight_so-a-brilliant-mind-could-forge.wav
```

The newest `MOSS_HISTORY_MAX` (default 500, roughly 500 MB) are kept. The test page lists them with play, download and delete, filterable by voice and client. They live in the [data folder](#data-folder), so they survive container rebuilds.

> Anyone who can reach the server can list and play the history, since there's no login.

### Identifying your client

Send an `X-Client-App` header (e.g. `X-Client-App: ppp`) so your requests show up by name in the server log:

```text
[req] 2026-09-28T22:03:58 client=ppp ip=172.18.0.1 POST /tts 200 1.84s
```

## SillyTavern

Use either TTS provider:

- **XTTSv2:** server URL `http://pone-ttsmod:8800`. SillyTavern fetches the voice list from `/speakers` automatically.
- **OpenAI Compatible:** endpoint `http://pone-ttsmod:8800/v1/audio/speech`. This provider doesn't fetch voices, so type the speaker names you want (from `/speakers`) into its comma-separated voice list.

Use the container name (`pone-ttsmod`) when SillyTavern runs in a container on the same Docker network. Otherwise use the host's IP address.

## Moods

Put a tag in the text to set the mood: `[mood]`, `[mood:energy]` or `[mood:energy:strength]`, e.g. `[angry:0.8]`. Moods: neutral, happy, sad, angry, annoyed, fearful, surprised, calm, disgusted, shouting (plus aliases such as irritated, scared, joyful). Energy is `0`–`1` (default `0.5`); strength is `1.0`–`2.3`.

A tag works anywhere in the text and applies until the next tag, so one request can change mood mid-way:

```text
Welcome to Ponyville. [irritated:1] I am here to tell you there is an update. [calm] Please read the instructions.
```

Text before the first tag uses the request's `style_*` fields, or Neutral.

## Voice effects

Each voice can have its own audio effect: `robot`, `radio`, `cavern`, `helium` or `changeling` (the original voice plus two extra tracks pitched -2 and -4 semitones). Choose it in the **Voice effect** box under the voice selector on the web page. The setting belongs to the voice, so it applies to every caller (web page, SillyTavern, Pony Pony Paradise). Settings are stored in `voice_fx.json` in the [data folder](#data-folder), so they survive container rebuilds. Scripts can use `GET/POST /voice_fx`.

## Streaming

`/tts_stream/` (the SillyTavern XTTSv2 streaming endpoint) sends audio sentence by sentence, so playback starts before a long paragraph finishes. Enable streaming in SillyTavern's XTTSv2 settings.

## Pronunciation

The model reads raw text with no phonetic dictionary, so some words come out wrong. Fix them in `pronunciations.txt` in the [data folder](#data-folder) (`server/data/`), one per line, or with the web page's **Pronunciations** button:

```text
beg = behg
Zecora = Zeh core ah
```

- Matching is whole-word and case-insensitive.
- The file reloads automatically; no restart needed.
- Check a respelling with `GET /normalize?text=I+beg+you`.

**Voice patch:** `adapter_model.safetensors` is downloaded into `server/voice_patch/` on the first start (set `MOSS_VOICE_PATCH_URL=0` to skip it, or drop in your own copy). It improves pronunciation for low-screen-time characters (Maud, Zecora, Cheerilee, Celestia, ...). Without it the server runs on the base model.

## Data folder

Everything the server writes, or you edit, lives in one folder: `server/data/` (Docker mounts it at `/app/data`).

| In `server/data/` | What it is |
|---|---|
| `pronunciations.txt` | Your respellings (starts as a copy of `server/defaults/pronunciations.txt`) |
| `preset.conf`, `voice_fx.json`, `voice_blend.json` | Settings changed from the web page |
| `history/` | Saved generations |
| `st_cache/` | Clips cached by `/st_cache/generate` |
| `sfx_cache/` | Downloaded sound effects |
| `pronunciation_cache.json`, `freesound_key.txt` | Optional pronunciation-check cache, your Freesound API key |

Older versions kept these in `server/`, `server/state/`, `server/history/` and `server/st_cache/`. The server (and `run.sh`) move them into `server/data/` on the first start. Re-run `./run.sh` once after updating, so the compose file mounts the new folder. Sound effects cached in the old `pone-ttsmod_sfx_cache` volume are downloaded again as needed; you can delete that volume with `docker volume rm pone-ttsmod_sfx_cache`.

## Colab

No NVIDIA GPU? `colab/pone_tts_colab.ipynb` runs the server on a free Google Colab GPU and gives you a public link for the game, SillyTavern or a browser. Click the **Open in Colab** badge at the top, then *Runtime → Run all*. The server settings from [Configuration](#configuration) (pauses, stream buffer, respelling, quality checks and so on) are a form in step 1. The link stops working when the Colab session ends.

## Project layout

```text
.
├── run.sh                      # Linux/WSL2 setup: prerequisites, GPU detection, compose file, build
├── Dockerfile                  # The NVIDIA image (target: runtime-cuda)
├── docker-compose.snippet.yml  # Example service block with every setting
├── colab/                      # Google Colab notebook
├── DISTRIBUTING.md             # Build notes for other machines
├── VERSION.txt                 # Release notes and known issues
└── server/
    ├── server.py               # The HTTP server (FastAPI)
    ├── entrypoint.sh           # Container start: prepare.py, GPU check, launch
    ├── prepare.py              # Setup steps shared by Docker, setup.sh/.bat and Colab
    ├── download_model.py       # Fetches the model from HuggingFace
    ├── apply_voice_patch.py    # Downloads and merges the voice patch
    ├── patch_torchopt_bundle.py# Fixes a bug in the downloaded model runtime
    ├── defaults/               # Starting pronunciations.txt and voice_blend.json
    ├── voice_patch/            # Pronunciation LoRA (downloaded on first start)
    └── data/                   # Your settings, history and caches (created at runtime)
```

`server/setup.sh`, `server/run_server.sh` and `server/tts.py` are the original bare-metal (non-Docker) launchers. See `server/README.md` for how to use them.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `unknown flag: --no-cache` | The Docker Compose plugin is missing. `run.sh` offers to install it |
| Arch: `docker ... and docker-cli-bin ... are in conflict` | `docker` includes the CLI and replaces `docker-cli-bin`. Answer **y** when pacman asks |
| `failed to get console: provided file is not a console` | Older `run.sh` piped to `tee`. Current `run.sh` handles this; by hand, set `COMPOSE_PROGRESS=plain` |
| `fork/exec /usr/bin/docker-proxy: no such file or directory` | Reinstall Docker and restart the service, or set `{"userland-proxy": false}` in `/etc/docker/daemon.json` |
| Container restarts during the model download | Usually memory. Check `docker compose logs`: the entrypoint prints the exit code and free disk space |
| `torch.cuda.is_available() is False` | GPU not passed through: check the NVIDIA Container Toolkit |
| `[voice-patch] could not download the voice patch` | No network, or the download link is down. The server still runs on the base model. Put `adapter_model.safetensors` into `server/voice_patch/` and restart |
| `$'\r': command not found` (WSL) | CRLF line endings. `sed -i 's/\r$//' run.sh server/*.sh`; the repo's `.gitattributes` forces LF on new clones |
| `could not select device driver "nvidia"` (WSL) | NVIDIA Container Toolkit missing or Docker not restarted: re-run `./run.sh`, or `sudo nvidia-ctk runtime configure --runtime=docker && sudo service docker restart` |
| Can't reach the server from the browser | Check `docker compose ps` and your port mapping. After a kernel update on Arch, reboot so Docker's networking modules load |
| `Temporary failure in name resolution` in the log | The container has no network. Same fix as above |

## Known issues

- The NVIDIA image hasn't yet been confirmed running on real hardware.

## Credits

- Model: **MOSS-TTS-PNY** by ZDisket, <https://huggingface.co/ZDisket/MOSS-TTS-PNY>
- This is an unofficial fan project. It is not affiliated with or endorsed by Hasbro or Valve. My Little Pony and Team Fortress 2 are trademarks of their respective owners.

## License

PoneTTS's own code (the Dockerfile, scripts and server) is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE), the same licence as Pony Pony Paradise.

In short: you can use, copy, modify and share it freely for any **noncommercial** purpose. You may not use it in a commercial product. This is a "source-available" licence rather than an OSI-approved open-source one, because of the noncommercial restriction.

Not covered by this licence:

- **The MOSS-TTS-PNY model.** It's downloaded from HuggingFace at first start, not included here, and has its own terms. See the [model card](https://huggingface.co/ZDisket/MOSS-TTS-PNY).
- **Sound effects in `server/data/sfx_cache/`.** These come from Freesound and the BBC Sound Effects archive and stay under their original licences.
- **Characters and voices.** My Little Pony and Team Fortress 2 characters belong to their respective owners.
