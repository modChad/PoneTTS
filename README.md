# PoneTTS

Self-hosted text-to-speech server for the **[MOSS-TTS-PNY](https://huggingface.co/ZDisket/MOSS-TTS-PNY)** model, packaged for Docker.

- 43 character voices: IDs 0–32 are My Little Pony, IDs 33–42 are Team Fortress 2
- 48 kHz mono WAV output
- Voice backend for **Pony Pony Paradise (PPP)** and **SillyTavern**
- Browser test page, OpenAI-compatible and XTTSv2-compatible endpoints

> **AMD (ROCm) support is highly experimental.** NVIDIA is the supported path. The AMD image has run on exactly one machine (an RX 7700S laptop), and it needed manual workarounds to get there. Expect problems on other AMD cards.

[[_TOC_]]

## Requirements

| | NVIDIA (supported) | AMD (experimental) |
|---|---|---|
| OS | Linux | Linux |
| GPU memory | ~6 GB minimum, see [presets](#performance-presets) | ~8 GB tested |
| Host driver | `nvidia-smi` lists your card | `rocm-smi` lists your card |
| Extra | [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) | User in the `video` and `render` groups |
| Docker | Engine, Compose plugin, buildx | Engine, Compose plugin, buildx |
| Disk | ~12 GB for the model, plus the image | ~12 GB for the model, plus a larger image |

The GPU driver must work **outside** Docker first. Docker only passes the GPU through; it can't fix a broken host driver.

## Quick start

```bash
git clone <this repo> PoneTTS
cd PoneTTS
./run.sh 2>&1 | tee run.log
```

`run.sh`:

1. Checks for the Docker CLI, Engine, Compose and buildx, and offers to install anything missing (pacman, apt, dnf or zypper). It always asks before installing.
2. Detects your GPU vendor and VRAM, and on NVIDIA checks for the NVIDIA Container Toolkit.
3. Suggests a performance preset. You can accept it or pick your own.
4. Asks for a port (default `8800`) and whether `/sfx` may fetch sound effects from the internet.
5. Writes `docker-compose.yml` for your hardware, including a RAM cap so running out of memory restarts the server instead of your desktop.
6. Builds the image and starts it.

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

## Adding it to an existing Compose setup

To run PoneTTS alongside other services (e.g. SillyTavern), add a service block like this to your own `docker-compose.yml`:

```yaml
services:
  pone-ttsmod:
    build:
      context: /path/to/PoneTTS      # the folder, not a file inside it
      dockerfile: Dockerfile
      target: runtime-cuda           # runtime-rocm for AMD (experimental)
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
      - pone-ttsmod_sfx_cache:/app/sfx_cache
      - ./server/voice_patch:/app/voice_patch    # optional
      - ./server/st_cache:/app/st_cache          # optional
      - ./server/history:/app/history            # optional: keeps saved clips on the host
    environment:
      - MOSS_PERF_PRESET=balanced

volumes:
  pone-ttsmod_release:
  pone-ttsmod_sfx_cache:
```

- `context` must point at the PoneTTS **folder**. If it's missing, Docker builds from the wrong folder and fails with `"/server": not found`.
- If you already built the image with `run.sh`, you can drop the `build:` block and keep `image: pone-ttsmod:cuda`.
- Behind a VPN container (e.g. `network_mode: "service:gluetun"`), remove `ports:` and publish `8800` on the VPN container instead. The server uses port `8800` rather than `8000` so it doesn't clash with gluetun's control server.

`docker-compose.snippet.yml` has full NVIDIA and AMD examples.

To build the image by hand:

```bash
docker build --target runtime-cuda -t pone-ttsmod:cuda .   # NVIDIA
docker build --target runtime-rocm -t pone-ttsmod:rocm .   # AMD (experimental)
```

## Performance presets

Set with `MOSS_PERF_PRESET`, or switch live from the dropdown on the test page (the choice persists across restarts).

| Preset | VRAM | Notes |
|---|---|---|
| `quality` | ~14–16 GB | Fastest. Vocoder on the GPU (NVIDIA only; on AMD it runs on the CPU, same as `balanced`) |
| `balanced` | ~9–12 GB | **Server default.** Model on the GPU, vocoder on the CPU |
| `low_vram` | ~7–8 GB | Also skips CUDA-graph capture; shorter line cap |
| `potato` | ~6 GB | As `low_vram` with the shortest line cap. Not quantized |

The presets only trade speed and VRAM. Every preset uses the same voices and quality.

`run.sh` suggests: under 7 GB `potato`, 7–9 GB `low_vram`, 9–12.5 GB `balanced`, above that `quality` (`balanced` on AMD).

## Configuration

Environment variables for the container:

| Variable | Default | What it does |
|---|---|---|
| `MOSS_PERF_PRESET` | `balanced` | Performance preset (see above) |
| `PORT` | `8800` | Port the server listens on inside the container |
| `MOSS_AUDIO_TEMPERATURE` | `0.85` | Default sampling temperature. Lower is steadier, higher is more expressive |
| `MOSS_SFX_ALLOW_NETWORK` | `1` | `0` makes `/sfx` serve only from its cache, with no outgoing requests |
| `MOSS_FIX_STUCK_WORDS` | `1` | Detect and regenerate words that get stuck droning. Runs Whisper on every sentence; `0` is much faster |
| `MOSS_LOOP_GUARD_REL_MULT` | `1.6` | A sentence this many times longer than expected (or one that hits its length limit) is checked with Whisper for repeats and failures |
| `MOSS_LOOP_GUARD_MIN_HEARD` | `0.5` | If Whisper hears less than this share of a checked sentence's words, the sentence counts as failed and is regenerated |
| `MOSS_FAILED_CLAUSE_RETRIES` | `2` | How many times a failed sentence is regenerated before it's sent as-is |
| `MOSS_COMMA_SPLIT_MIN_WORDS` | `20` | Sentences longer than this are also split at commas (lists are kept together). `0` turns comma splitting off |
| `MOSS_FIX_PRON` | `0` | Check rare words with Whisper and fix their pronunciation (slower) |
| `MOSS_AUTO_RESPELL` | `0` | Automatically respell rare words phonetically |
| `MOSS_DECODER_RUNTIME` | from preset | Override the vocoder: `onnx_cuda`, `onnx_cpu` or `onnx_hybrid` |
| `MOSS_HISTORY_MAX` | `500` | How many generated lines to keep in the history (oldest deleted first). `0` turns history off |
| `HSA_OVERRIDE_GFX_VERSION` | `11.0.2` (AMD image) | AMD only: GPU architecture override. See [known issues](#known-issues) |
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

Every successfully generated line is saved to `server/history/` as a WAV plus a `.json` with its details (full text, voice, client, duration, temperature, preset). File names sort by time and are readable:

```text
20260930-200612-345_twilight_so-a-brilliant-mind-could-forge.wav
```

The newest `MOSS_HISTORY_MAX` (default 500, roughly 500 MB) are kept. The test page lists them with play, download and delete, filterable by voice and client. Mount `./server/history` to keep them across container rebuilds.

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

## Pronunciation

The model reads raw text with no phonetic dictionary, so some words come out wrong. Fix them in `server/pronunciations.txt`, one per line:

```text
beg = behg
Zecora = Zeh core ah
```

- Matching is whole-word and case-insensitive.
- The file reloads automatically; no restart needed.
- Check a respelling with `GET /normalize?text=I+beg+you`.

**Voice patch (optional):** drop `adapter_model.safetensors` into `server/voice_patch/` and restart. It improves pronunciation for low-screen-time characters (Maud, Zecora, Cheerilee, Celestia, ...). Without it the server runs on the base model.

## Project layout

```text
.
├── run.sh                      # Linux setup: prerequisites, GPU detection, compose file, build
├── Dockerfile                  # Two build targets: runtime-cuda, runtime-rocm
├── docker-compose.snippet.yml  # Example service blocks (NVIDIA and AMD)
├── DISTRIBUTING.md             # Build notes for other machines
├── VERSION.txt                 # Release notes and known issues
└── server/
    ├── server.py               # The HTTP server (FastAPI)
    ├── entrypoint.sh           # Container start: download, patches, GPU check, launch
    ├── download_model.py       # Fetches the model from HuggingFace
    ├── apply_voice_patch.py    # Merges the optional voice patch
    ├── patch_torchopt_bundle.py# Fixes a bug in the downloaded model runtime
    ├── pronunciations.txt      # Your respellings
    ├── voice_patch/            # Optional pronunciation LoRA
    ├── history/                # Saved generations (created at runtime)
    └── sfx_cache/              # Cached sound effects
```

`server/setup.sh`, `run_server.sh` and `tts.py` are the original bare-metal (non-Docker) launchers. See `server/README.md` for how to use them.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `unknown flag: --no-cache` | The Docker Compose plugin is missing. `run.sh` offers to install it |
| Arch: `docker ... and docker-cli-bin ... are in conflict` | `docker` includes the CLI and replaces `docker-cli-bin`. Answer **y** when pacman asks |
| `failed to get console: provided file is not a console` | Older `run.sh` piped to `tee`. Current `run.sh` handles this; by hand, set `COMPOSE_PROGRESS=plain` |
| `fork/exec /usr/bin/docker-proxy: no such file or directory` | Reinstall Docker and restart the service, or set `{"userland-proxy": false}` in `/etc/docker/daemon.json` |
| Container restarts during the model download | Usually memory. Check `docker compose logs`: the entrypoint prints the exit code and free disk space |
| `torch.cuda.is_available() is False` | GPU not passed through: check the NVIDIA Container Toolkit (NVIDIA), or the `/dev/kfd` and `/dev/dri` devices (AMD) |
| Can't reach the server from the browser | Check `docker compose ps` and your port mapping. After a kernel update on Arch, reboot so Docker's networking modules load |
| `Temporary failure in name resolution` in the log | The container has no network. Same fix as above |

## Known issues

- **AMD is highly experimental** (see the note at the top).
- The AMD image sets `HSA_OVERRIDE_GFX_VERSION=11.0.2` (RX 7700S) for every AMD card. Other cards may need a different value or none at all.
- The NVIDIA image hasn't yet been confirmed running on real hardware.

## Credits

- Model: **MOSS-TTS-PNY** by ZDisket, <https://huggingface.co/ZDisket/MOSS-TTS-PNY>
- This is an unofficial fan project. It is not affiliated with or endorsed by Hasbro or Valve. My Little Pony and Team Fortress 2 are trademarks of their respective owners.

## License

PoneTTS's own code (the Dockerfile, scripts and server) is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE), the same licence as Pony Pony Paradise.

In short: you can use, copy, modify and share it freely for any **noncommercial** purpose. You may not use it in a commercial product. This is a "source-available" licence rather than an OSI-approved open-source one, because of the noncommercial restriction.

Not covered by this licence:

- **The MOSS-TTS-PNY model.** It's downloaded from HuggingFace at first start, not included here, and has its own terms. See the [model card](https://huggingface.co/ZDisket/MOSS-TTS-PNY).
- **Sound effects in `server/sfx_cache/`.** These come from Freesound and the BBC Sound Effects archive and stay under their original licences.
- **Characters and voices.** My Little Pony and Team Fortress 2 characters belong to their respective owners.
