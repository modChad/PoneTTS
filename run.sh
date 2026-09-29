#!/usr/bin/env bash
# PoneTTS setup — Linux only.
#
# Detects your GPU vendor (NVIDIA/AMD) and VRAM, generates a docker-compose.yml
# tailored to your hardware (right build target, right device-passthrough
# flags, right performance preset), and optionally builds + starts it.
#
# Does NOT install Docker, the NVIDIA Container Toolkit, or ROCm itself —
# those are prerequisites. See DISTRIBUTING.md: confirm your GPU driver works
# OUTSIDE Docker first (nvidia-smi / rocm-smi), or nothing here will help.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BOLD=$'\033[1m'; RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RESET=$'\033[0m'
# When output is piped (e.g. "./run.sh | tee build.log") there's no terminal
# for Compose/BuildKit's animated progress display, and some Compose versions
# abort with "failed to get console: provided file is not a console". Use
# plain text progress instead (also far more useful in a log file).
if [[ ! -t 1 ]]; then
    export COMPOSE_PROGRESS="${COMPOSE_PROGRESS:-plain}"
    export BUILDKIT_PROGRESS="${BUILDKIT_PROGRESS:-plain}"
    export COMPOSE_ANSI="${COMPOSE_ANSI:-never}"
fi

info()  { echo "${BOLD}==>${RESET} $*"; }
warn()  { echo "${YELLOW}${BOLD}WARNING:${RESET} $*"; }
error() { echo "${RED}${BOLD}ERROR:${RESET} $*" >&2; }

# -----------------------------------------------------------------------------
# 1. OS check
# -----------------------------------------------------------------------------
if [[ "$(uname -s)" != "Linux" ]]; then
    error "This setup script is Linux-only (detected: $(uname -s))."
    error "Windows/Mac users: build/run the Dockerfile manually per DISTRIBUTING.md."
    exit 1
fi

# -----------------------------------------------------------------------------
# 2. Docker + buildx prerequisite check
# -----------------------------------------------------------------------------
info "Checking Docker prerequisites..."

# Which Compose command works: the v2 plugin ("docker compose") or the
# standalone binary ("docker-compose"). Empty if neither.
find_compose() {
    if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
        echo "docker compose"
    elif command -v docker-compose >/dev/null 2>&1 && docker-compose version >/dev/null 2>&1; then
        echo "docker-compose"
    fi
}

NEED_CLI=0; NEED_ENGINE=0; NEED_COMPOSE=0; NEED_BUILDX=0
command -v docker  >/dev/null 2>&1 || NEED_CLI=1
# The engine counts as present if dockerd is installed OR the CLI can already
# reach a daemon (Docker Desktop runs its engine in a VM, so there's no local
# dockerd binary — don't try to install a second engine next to it).
if ! command -v dockerd >/dev/null 2>&1; then
    if [[ "$NEED_CLI" -eq 1 ]] || ! docker info >/dev/null 2>&1; then
        NEED_ENGINE=1
    fi
fi
[[ -n "$(find_compose)" ]]         || NEED_COMPOSE=1
if [[ "$NEED_CLI" -eq 1 ]] || ! docker buildx version >/dev/null 2>&1; then NEED_BUILDX=1; fi

if (( NEED_CLI + NEED_ENGINE + NEED_COMPOSE + NEED_BUILDX > 0 )); then
    MISSING=()
    [[ "$NEED_CLI"     -eq 1 ]] && MISSING+=("Docker CLI")
    [[ "$NEED_ENGINE"  -eq 1 ]] && MISSING+=("Docker Engine")
    [[ "$NEED_COMPOSE" -eq 1 ]] && MISSING+=("Docker Compose")
    [[ "$NEED_BUILDX"  -eq 1 ]] && MISSING+=("Docker buildx")
    warn "Missing: $(IFS=,; echo "${MISSING[*]}" | sed 's/,/, /g')"

    # Package names per package manager. If Docker's own repo
    # (download.docker.com) is set up, use its package names — mixing them
    # with the distro's docker packages causes file conflicts.
    PM=""; PKG_CLI=""; PKG_ENGINE=""; PKG_COMPOSE=""; PKG_BUILDX=""
    if command -v pacman >/dev/null 2>&1; then
        PM="pacman"
        PKG_CLI=docker; PKG_ENGINE=docker; PKG_COMPOSE=docker-compose; PKG_BUILDX=docker-buildx
    elif command -v apt-get >/dev/null 2>&1; then
        PM="apt"
        if apt-cache show docker-ce-cli >/dev/null 2>&1; then
            PKG_CLI=docker-ce-cli; PKG_ENGINE=docker-ce
            PKG_COMPOSE=docker-compose-plugin; PKG_BUILDX=docker-buildx-plugin
        else
            PKG_CLI=docker.io; PKG_ENGINE=docker.io; PKG_BUILDX=docker-buildx
            if apt-cache show docker-compose-v2 >/dev/null 2>&1; then
                PKG_COMPOSE=docker-compose-v2
            else
                PKG_COMPOSE=docker-compose
            fi
        fi
    elif command -v dnf >/dev/null 2>&1; then
        PM="dnf"
        if dnf -q list docker-ce-cli >/dev/null 2>&1; then
            PKG_CLI=docker-ce-cli; PKG_ENGINE=docker-ce
            PKG_COMPOSE=docker-compose-plugin; PKG_BUILDX=docker-buildx-plugin
        else
            PKG_CLI=docker-cli; PKG_ENGINE=moby-engine
            PKG_COMPOSE=docker-compose; PKG_BUILDX=docker-buildx
        fi
    elif command -v zypper >/dev/null 2>&1; then
        PM="zypper"
        PKG_CLI=docker; PKG_ENGINE=docker; PKG_COMPOSE=docker-compose; PKG_BUILDX=docker-buildx
    fi

    if [[ -z "$PM" ]]; then
        error "Couldn't find a supported package manager (pacman, apt, dnf, zypper)."
        error "Install Docker Engine, the Compose plugin and buildx yourself, then re-run."
        exit 1
    fi

    PKGS=()
    add_pkg() { local p; for p in "${PKGS[@]}"; do [[ "$p" == "$1" ]] && return; done; PKGS+=("$1"); }
    [[ "$NEED_CLI"     -eq 1 ]] && add_pkg "$PKG_CLI"
    [[ "$NEED_ENGINE"  -eq 1 ]] && add_pkg "$PKG_ENGINE"
    [[ "$NEED_COMPOSE" -eq 1 ]] && add_pkg "$PKG_COMPOSE"
    [[ "$NEED_BUILDX"  -eq 1 ]] && add_pkg "$PKG_BUILDX"

    # Arch: a CLI-only package (e.g. docker-cli-bin) conflicts with the full
    # "docker" package, which ships its own CLI. pacman has to replace it, and
    # --noconfirm would answer "No" to that — so let pacman ask the user.
    PACMAN_CONFIRM=(--noconfirm)
    if [[ "$PM" == pacman && "$NEED_ENGINE" -eq 1 ]]; then
        CONFLICTS="$(pacman -Qq 2>/dev/null | grep -E '^docker-cli' || true)"
        if [[ -n "$CONFLICTS" ]]; then
            warn "You have $(echo "$CONFLICTS" | tr '\n' ' ' | sed 's/ $//') installed (Docker CLI only)."
            warn "Arch's 'docker' package includes the CLI as well, so it replaces that package."
            warn "pacman will ask 'Remove ...? [y/N]' — answer y to continue."
            PACMAN_CONFIRM=()
        fi
    fi

    case "$PM" in
        pacman) INSTALL_CMD=(sudo pacman -S --needed "${PACMAN_CONFIRM[@]}" "${PKGS[@]}") ;;
        apt)    INSTALL_CMD=(sudo apt-get install -y "${PKGS[@]}") ;;
        dnf)    INSTALL_CMD=(sudo dnf install -y "${PKGS[@]}") ;;
        zypper) INSTALL_CMD=(sudo zypper --non-interactive install "${PKGS[@]}") ;;
    esac

    echo "  This will run: ${INSTALL_CMD[*]}"
    read -rp "  Install them now? [Y/n]: " do_install
    do_install="${do_install:-Y}"
    if [[ ! "$do_install" =~ ^[Yy] ]]; then
        error "Docker prerequisites are missing — install them, then re-run this script."
        exit 1
    fi
    [[ "$PM" == apt ]] && sudo apt-get update
    if ! "${INSTALL_CMD[@]}"; then
        error "Package install failed — see the output above."
        exit 1
    fi

    if [[ "$NEED_ENGINE" -eq 1 ]] && command -v systemctl >/dev/null 2>&1; then
        info "Enabling and starting the Docker service..."
        sudo systemctl enable --now docker || warn "Couldn't start docker.service — start it yourself."
    fi
fi

# Re-check everything after any install.
if ! command -v docker >/dev/null 2>&1; then
    error "docker still not found after install. Install Docker Engine for your distro, then re-run."
    exit 1
fi

if ! docker buildx version >/dev/null 2>&1; then
    error "docker buildx plugin still not found or not working."
    error "Install it — e.g. 'sudo apt install docker-buildx-plugin' (Debian/Ubuntu),"
    error "'sudo dnf install docker-buildx-plugin' (Fedora/RHEL), or"
    error "'sudo pacman -S docker-buildx' (Arch) — then re-run this script."
    exit 1
fi

COMPOSE="$(find_compose)"
if [[ -z "$COMPOSE" ]]; then
    error "Docker Compose still not found after install."
    error "Install it — e.g. 'sudo apt install docker-compose-plugin' (Debian/Ubuntu),"
    error "'sudo dnf install docker-compose-plugin' (Fedora/RHEL), or"
    error "'sudo pacman -S docker-compose' (Arch) — then re-run this script."
    exit 1
fi
read -r -a COMPOSE_CMD <<< "$COMPOSE"

if ! docker info >/dev/null 2>&1; then
    error "Can't talk to the Docker daemon. Is it running? Are you in the 'docker' group?"
    error "Try: sudo systemctl enable --now docker"
    if ! id -nG 2>/dev/null | grep -qw docker; then
        error "You're not in the 'docker' group. Add yourself with:"
        error "  sudo usermod -aG docker \$USER"
        error "then log out and back in (or run 'newgrp docker'), and re-run this script."
    fi
    exit 1
fi

echo "  Docker + Compose ($COMPOSE) + buildx OK."

# -----------------------------------------------------------------------------
# 3. GPU vendor + VRAM detection
# -----------------------------------------------------------------------------
info "Detecting GPU..."

NVIDIA_OK=0
NVIDIA_VRAM_MB=0
AMD_OK=0
AMD_VRAM_MB=0

if command -v nvidia-smi >/dev/null 2>&1; then
    # First GPU only — multi-GPU setups should edit the generated compose file.
    if line="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1)" \
        && [[ "$line" =~ ^[0-9]+$ ]]; then
        NVIDIA_OK=1
        NVIDIA_VRAM_MB="$line"
        echo "  NVIDIA GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1) (${NVIDIA_VRAM_MB} MiB VRAM)"
    else
        warn "nvidia-smi is present but didn't return usable output — driver may be broken."
    fi
fi

if command -v rocm-smi >/dev/null 2>&1; then
    # rocm-smi's output format has genuinely changed across versions. Try
    # --json first (most reliable when supported); fall back to text parsing.
    json_out="$(rocm-smi --showmeminfo vram --json 2>/dev/null || true)"
    if [[ -n "$json_out" ]] && command -v python3 >/dev/null 2>&1; then
        tmpfile="$(mktemp)"
        printf '%s' "$json_out" > "$tmpfile"
        AMD_VRAM_MB="$(python3 - "$tmpfile" <<'PYEOF'
import json, sys
try:
    with open(sys.argv[1]) as f:
        data = json.load(f)
    for card in data.values():
        for k, v in card.items():
            if "VRAM Total Memory" in k or "vram_total" in k.lower():
                # value may be bytes as a string
                print(int(str(v).strip()) // (1024 * 1024))
                sys.exit(0)
except Exception:
    pass
print(0)
PYEOF
)"
        rm -f "$tmpfile"
    fi
    if [[ "${AMD_VRAM_MB:-0}" -eq 0 ]]; then
        # Text fallback: look for a "VRAM Total Memory (B): <bytes>" style line
        # specifically — NOT just "the last number in the output", since a
        # "VRAM Total Used Memory" line can appear after it and would
        # otherwise get picked up instead, silently under-reporting VRAM.
        text_out="$(rocm-smi --showmeminfo vram 2>/dev/null || true)"
        bytes_val="$(echo "$text_out" | grep -i "VRAM Total Memory" | grep -oE '[0-9]+' | tail -n1 || true)"
        if [[ -n "$bytes_val" && "$bytes_val" =~ ^[0-9]+$ && "$bytes_val" -gt 1048576 ]]; then
            AMD_VRAM_MB=$(( bytes_val / 1024 / 1024 ))
        fi
    fi

    if command -v rocminfo >/dev/null 2>&1 || rocm-smi --showproductname >/dev/null 2>&1; then
        AMD_OK=1
        if [[ "${AMD_VRAM_MB:-0}" -gt 0 ]]; then
            echo "  AMD GPU detected (${AMD_VRAM_MB} MiB VRAM)"
        else
            warn "AMD GPU detected via rocm-smi, but couldn't parse VRAM size from its output"
            warn "(this varies across ROCm versions). Defaulting to the safe low_vram preset."
        fi
    fi
fi

if [[ "$NVIDIA_OK" -eq 0 && "$AMD_OK" -eq 0 ]]; then
    error "No working GPU detected — neither nvidia-smi nor rocm-smi returned usable output."
    error "Confirm your GPU driver works OUTSIDE Docker first: run 'nvidia-smi' or 'rocm-smi'"
    error "directly in this terminal. If neither prints your GPU, fix that before re-running."
    exit 1
fi

VENDOR=""
VRAM_MB=0

if [[ "$NVIDIA_OK" -eq 1 && "$AMD_OK" -eq 1 ]]; then
    warn "Both an NVIDIA and an AMD GPU look usable on this system."
    echo "  1) NVIDIA"
    echo "  2) AMD"
    read -rp "  Which one should PoneTTS use? [1/2]: " choice
    case "$choice" in
        1) VENDOR=nvidia; VRAM_MB="$NVIDIA_VRAM_MB" ;;
        2) VENDOR=amd; VRAM_MB="$AMD_VRAM_MB" ;;
        *) error "Not a valid choice."; exit 1 ;;
    esac
elif [[ "$NVIDIA_OK" -eq 1 ]]; then
    VENDOR=nvidia
    VRAM_MB="$NVIDIA_VRAM_MB"
else
    VENDOR=amd
    VRAM_MB="$AMD_VRAM_MB"
fi

if [[ "$VENDOR" == nvidia ]]; then
    IMAGE_TAG="pone-ttsmod:cuda"
    BUILD_TARGET="runtime-cuda"
else
    IMAGE_TAG="pone-ttsmod:rocm"
    BUILD_TARGET="runtime-rocm"
fi

# -----------------------------------------------------------------------------
# 3b. NVIDIA Container Toolkit — without it Docker can't hand the GPU to the
# container, even though nvidia-smi works fine on the host.
# -----------------------------------------------------------------------------
if [[ "$VENDOR" == nvidia ]]; then
    if command -v nvidia-container-runtime-hook >/dev/null 2>&1 \
        || command -v nvidia-ctk >/dev/null 2>&1; then
        echo "  NVIDIA Container Toolkit OK."
    else
        warn "NVIDIA Container Toolkit not found — Docker can't pass your GPU to the container without it."
        TK_CMD=()
        if command -v pacman >/dev/null 2>&1; then
            TK_CMD=(sudo pacman -S --needed --noconfirm nvidia-container-toolkit)
        elif command -v apt-get >/dev/null 2>&1 && apt-cache show nvidia-container-toolkit >/dev/null 2>&1; then
            TK_CMD=(sudo apt-get install -y nvidia-container-toolkit)
        elif command -v dnf >/dev/null 2>&1 && dnf -q list nvidia-container-toolkit >/dev/null 2>&1; then
            TK_CMD=(sudo dnf install -y nvidia-container-toolkit)
        elif command -v zypper >/dev/null 2>&1 && zypper -q info nvidia-container-toolkit 2>/dev/null | grep -q '^Name'; then
            TK_CMD=(sudo zypper --non-interactive install nvidia-container-toolkit)
        fi

        if [[ "${#TK_CMD[@]}" -eq 0 ]]; then
            error "It isn't in your configured package repos. Add NVIDIA's repo and install it:"
            error "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
            error "then re-run this script."
            exit 1
        fi

        echo "  This will run: ${TK_CMD[*]}"
        echo "  and then restart Docker so it picks the toolkit up."
        read -rp "  Install it now? [Y/n]: " do_tk
        do_tk="${do_tk:-Y}"
        if [[ ! "$do_tk" =~ ^[Yy] ]]; then
            error "The NVIDIA Container Toolkit is required — install it, then re-run this script."
            exit 1
        fi
        if ! "${TK_CMD[@]}"; then
            error "Toolkit install failed — see the output above."
            exit 1
        fi
        if command -v systemctl >/dev/null 2>&1; then
            sudo systemctl restart docker || warn "Couldn't restart Docker — restart it yourself before continuing."
        fi
        echo "  NVIDIA Container Toolkit installed."
    fi
fi

# -----------------------------------------------------------------------------
# 4. Pick a performance preset from VRAM — auto, or manual per server/README.md
# -----------------------------------------------------------------------------
# Rounded to the nearest GB for display (cards report slightly under their
# marketed size, e.g. an 8 GB card shows ~8188 MiB). Thresholds below use MiB.
VRAM_GB=$(( (VRAM_MB + 512) / 1024 ))

if [[ "$VRAM_MB" -eq 0 ]]; then
    # Detection failed to produce a number at all (AMD text-parsing miss, etc.)
    # — default to the safest tier rather than guess.
    SUGGESTED_PRESET=low_vram
    SUGGESTED_REASON="VRAM size undetermined, defaulting to the safe tier"
elif (( VRAM_MB < 7168 )); then          # under 7 GB: 6 GB cards and smaller
    SUGGESTED_PRESET=potato
    SUGGESTED_REASON="${VRAM_GB} GB VRAM"
    if (( VRAM_MB < 5632 )); then        # under ~5.5 GB
        warn "${VRAM_GB} GB VRAM is below the ~6 GB minimum — the model will probably not fit."
    fi
elif (( VRAM_MB <= 9216 )); then         # 7-9 GB
    SUGGESTED_PRESET=low_vram
    SUGGESTED_REASON="${VRAM_GB} GB VRAM"
elif (( VRAM_MB <= 12800 )); then        # 9-12.5 GB
    SUGGESTED_PRESET=balanced
    SUGGESTED_REASON="${VRAM_GB} GB VRAM"
else
    SUGGESTED_PRESET=quality
    SUGGESTED_REASON="${VRAM_GB} GB VRAM"
fi

# quality runs the vocoder on onnxruntime's CUDA path, which doesn't exist on
# ROCm — on AMD the server would quietly run it on the CPU anyway, i.e. it
# would just be balanced. Suggest balanced directly.
if [[ "$VENDOR" == amd && "$SUGGESTED_PRESET" == quality ]]; then
    SUGGESTED_PRESET=balanced
    SUGGESTED_REASON="${VRAM_GB} GB VRAM; quality's GPU vocoder is NVIDIA-only"
fi

echo
echo "  Auto-detected suggestion: ${BOLD}${SUGGESTED_PRESET}${RESET} (${SUGGESTED_REASON})"
read -rp "  Use this automatically, or pick a preset yourself? [A]uto / [M]anual: " preset_mode
preset_mode="${preset_mode:-A}"

if [[ "$preset_mode" =~ ^[Mm] ]]; then
    # Table quoted directly from server/README.md so this can't drift from
    # what the original project itself documents.
    echo
    echo "  preset               VRAM (approx)   notes"
    echo "  quality              ~14-16 GB       fastest; vocoder on GPU (NVIDIA only)"
    echo "  balanced             ~9-12 GB        LM on GPU (fast), vocoder on CPU"
    echo "  low_vram             ~7-8 GB         also drops CUDA-graph capture + shorter line cap"
    echo "  potato               ~6 GB           as low_vram with the shortest line cap; not quantized"
    echo
    while true; do
        read -rp "  Enter preset [quality/balanced/low_vram/potato]: " manual_preset
        case "$manual_preset" in
            quality|balanced|low_vram|potato)
                PRESET="$manual_preset"
                PRESET_REASON="manually selected"
                if [[ "$VENDOR" == amd && "$PRESET" == quality ]]; then
                    warn "quality's GPU vocoder is NVIDIA-only — on AMD it runs on the CPU, same as balanced."
                fi
                break
                ;;
            *)
                echo "  Not a valid preset name — must be exactly one of: quality, balanced, low_vram, potato"
                ;;
        esac
    done
else
    PRESET="$SUGGESTED_PRESET"
    PRESET_REASON="$SUGGESTED_REASON"
fi

echo "  Selected preset: ${BOLD}${PRESET}${RESET} (${PRESET_REASON})"

# -----------------------------------------------------------------------------
# 4b. Pick a host port
# -----------------------------------------------------------------------------
DEFAULT_PORT=8800
echo
read -rp "  Which port should the server listen on? [${DEFAULT_PORT}]: " HOST_PORT
HOST_PORT="${HOST_PORT:-$DEFAULT_PORT}"

if ! [[ "$HOST_PORT" =~ ^[0-9]+$ ]] || (( HOST_PORT < 1 || HOST_PORT > 65535 )); then
    error "'$HOST_PORT' isn't a valid port number (must be 1-65535)."
    exit 1
fi
if (( HOST_PORT < 1024 )); then
    warn "Port $HOST_PORT is a privileged port (<1024) — Docker may need root to bind it."
fi

# Soft check only — a false positive here (e.g. an old pone-ttsmod container
# already sitting on this exact port, which is fine) shouldn't block anyone,
# just prompt them to double check.
if (exec 3<>"/dev/tcp/127.0.0.1/$HOST_PORT") 2>/dev/null; then
    exec 3>&- 3<&-
    warn "Something is already listening on port $HOST_PORT."
    read -rp "  Use it anyway? [y/N]: " use_anyway
    if [[ ! "$use_anyway" =~ ^[Yy] ]]; then
        error "Aborting — re-run and pick a different port."
        exit 1
    fi
fi

echo "  Using port: ${BOLD}${HOST_PORT}${RESET}"

# -----------------------------------------------------------------------------
# 4c. Sound effects: outgoing internet access
# -----------------------------------------------------------------------------
echo
echo "  The /sfx feature fetches sound effects from Freesound and the BBC Sound"
echo "  Effects Archive on demand, caching each result so it's never re-fetched."
read -rp "  Allow outgoing internet calls for this? [Y/n]: " allow_sfx_net
SFX_ALLOW_NETWORK="1"
if [[ "$allow_sfx_net" =~ ^[Nn] ]]; then
    SFX_ALLOW_NETWORK="0"
    echo "  Disabled — /sfx will only ever serve from its cache, never reach out itself."
fi

# -----------------------------------------------------------------------------
# 4d. System RAM limit for the container. Without one, if the server runs the
# machine out of RAM the kernel may kill your desktop session instead of the
# server. With a limit, only the container gets killed (and restarted).
# Keeps back a quarter of RAM (at least 3 GB) for everything else. Swap is
# disabled for the container so it fails fast instead of freezing the machine.
# -----------------------------------------------------------------------------
RAM_MB="$(awk '/^MemTotal:/ {print int($2 / 1024)}' /proc/meminfo 2>/dev/null || echo 0)"
MEM_LIMIT_LINES=""
SHM_SIZE="8g"   # AMD only; shrunk below to fit inside the memory limit
if [[ "${RAM_MB:-0}" -gt 0 ]]; then
    RESERVE_MB=$(( RAM_MB / 4 ))
    (( RESERVE_MB < 3072 )) && RESERVE_MB=3072
    LIMIT_MB=$(( RAM_MB - RESERVE_MB ))
    echo
    if (( LIMIT_MB < 4096 )); then
        warn "Only $(( RAM_MB / 1024 )) GB of system RAM — not enough to safely cap the server's memory."
        warn "Skipping the memory limit; close other apps while the server is running."
    else
        (( LIMIT_MB / 2 < 8192 )) && SHM_SIZE="$(( LIMIT_MB / 2 ))m"
        MEM_LIMIT_LINES="    mem_limit: ${LIMIT_MB}m
    memswap_limit: ${LIMIT_MB}m
"
        echo "  Memory limit: ${BOLD}$(( (LIMIT_MB + 512) / 1024 )) GB${RESET} of $(( (RAM_MB + 512) / 1024 )) GB RAM"
        echo "  (if the server hits it, only the server is stopped and restarted — not your desktop)"
    fi
fi

# -----------------------------------------------------------------------------
# 5. Generate docker-compose.yml
# -----------------------------------------------------------------------------
COMPOSE_FILE="docker-compose.yml"

if [[ -f "$COMPOSE_FILE" ]]; then
    backup="docker-compose.yml.bak.$(date +%Y%m%d%H%M%S)"
    warn "$COMPOSE_FILE already exists — backing it up to $backup rather than overwriting blind."
    mv "$COMPOSE_FILE" "$backup"
fi

# sfx_cache only ever gets populated by a successful outgoing fetch — with
# network access declined there's nothing that would ever be written to it,
# so skip provisioning the volume entirely rather than leave an always-empty
# one lying around.
SFX_VOLUME_LINE=""
SFX_VOLUME_DECL=""
if [[ "$SFX_ALLOW_NETWORK" == "1" ]]; then
    SFX_VOLUME_LINE="      - pone-ttsmod_sfx_cache:/app/sfx_cache
"
    SFX_VOLUME_DECL="  pone-ttsmod_sfx_cache:
"
fi

if [[ "$VENDOR" == nvidia ]]; then
    cat > "$COMPOSE_FILE" <<EOF
# Generated by run.sh — NVIDIA build, preset: ${PRESET} (${PRESET_REASON})
services:
  pone-ttsmod:
    build:
      context: .
      dockerfile: Dockerfile
      target: runtime-cuda
    image: pone-ttsmod:cuda
    container_name: pone-ttsmod
    restart: unless-stopped
${MEM_LIMIT_LINES}    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    ports:
      - "${HOST_PORT}:${HOST_PORT}"
    volumes:
      - pone-ttsmod_release:/app/release
${SFX_VOLUME_LINE}      - ./server/voice_patch:/app/voice_patch   # drop adapter_model.safetensors here if missing — no rebuild needed
      - ./server/st_cache:/app/st_cache             # cache for the /st_cache/generate endpoint — visible/durable on the host
    environment:
      - MOSS_PERF_PRESET=${PRESET}
      - MOSS_SFX_ALLOW_NETWORK=${SFX_ALLOW_NETWORK}
      - PORT=${HOST_PORT}

volumes:
  pone-ttsmod_release:
${SFX_VOLUME_DECL}
EOF
else
    cat > "$COMPOSE_FILE" <<EOF
# Generated by run.sh — AMD/ROCm build, preset: ${PRESET} (${PRESET_REASON})
#
# NOTE: some AMD GPUs aren't in ROCm's officially supported list and need
# HSA_OVERRIDE_GFX_VERSION set to the nearest supported architecture to
# actually initialize (e.g. a RX 7700S/gfx1102 needs 11.0.2 — map your gfxNNNN
# to MAJOR.MINOR.PATCH directly, e.g. gfx1102 -> 11.0.2; don't assume gfx1100's
# override value works for a different chip, even in the same family). This
# script can't reliably auto-detect whether that's needed for your specific
# card, or its correct value — if the container starts but
# torch.cuda.is_available() comes back false, or you see HSA/HIP errors
# (e.g. HSA_STATUS_ERROR_INVALID_ISA) in the logs, that's the first thing to
# check. Add it under environment: below and rebuild.
services:
  pone-ttsmod:
    build:
      context: .
      dockerfile: Dockerfile
      target: runtime-rocm
    image: pone-ttsmod:rocm
    container_name: pone-ttsmod
    restart: unless-stopped
${MEM_LIMIT_LINES}    devices:
      - /dev/kfd
      - /dev/dri
    group_add:
      - video
      - render
    security_opt:
      - seccomp=unconfined
    cap_add:
      - SYS_PTRACE
    ipc: host
    shm_size: "${SHM_SIZE}"
    ports:
      - "${HOST_PORT}:${HOST_PORT}"
    volumes:
      - pone-ttsmod_release:/app/release
${SFX_VOLUME_LINE}      - ./server/voice_patch:/app/voice_patch   # drop adapter_model.safetensors here if missing — no rebuild needed
      - ./server/st_cache:/app/st_cache             # cache for the /st_cache/generate endpoint — visible/durable on the host
    environment:
      - MOSS_PERF_PRESET=${PRESET}
      - MOSS_SFX_ALLOW_NETWORK=${SFX_ALLOW_NETWORK}
      - PORT=${HOST_PORT}
      # - HSA_OVERRIDE_GFX_VERSION=11.0.2   # uncomment + adjust if needed, see note above

volumes:
  pone-ttsmod_release:
${SFX_VOLUME_DECL}
EOF
fi

echo "  Wrote $COMPOSE_FILE"

if [[ ! -f "server/voice_patch/adapter_model.safetensors" ]]; then
    warn "server/voice_patch/adapter_model.safetensors is missing."
    warn "The server will still run without it, but without the v2.5 pronunciation"
    warn "fix for low-screen-time characters (see server/voice_patch/README.md). If you have"
    warn "this file, just drop it into ./server/voice_patch/ — it's bind-mounted, so no"
    warn "rebuild is needed, even after the container's already running."
fi

# -----------------------------------------------------------------------------
# 6. Build (or reuse an existing image) and start
# -----------------------------------------------------------------------------
IMAGE_EXISTS=0
if docker image inspect "$IMAGE_TAG" >/dev/null 2>&1; then
    IMAGE_EXISTS=1
fi

DO_BUILD=1
if [[ "$IMAGE_EXISTS" -eq 1 ]]; then
    echo
    info "An existing ${IMAGE_TAG} image was found."
    read -rp "  Rebuild it, or reuse it as-is? [R]ebuild / [U]se existing: " rebuild_choice
    rebuild_choice="${rebuild_choice:-R}"
    if [[ "$rebuild_choice" =~ ^[Uu] ]]; then
        DO_BUILD=0
    fi
fi

echo
info "Ready to go. This will:"
if [[ "$DO_BUILD" -eq 1 ]]; then
    echo "    - Build the ${VENDOR} image (several minutes, larger for ROCm)"
else
    echo "    - Reuse the existing ${IMAGE_TAG} image (no build)"
fi
echo "    - On first start on a fresh volume, download the ~12 GB model + apply the voice patch"
echo "    - Expose the server on host port ${HOST_PORT}"
echo
read -rp "Proceed now? [Y/n]: " proceed
proceed="${proceed:-Y}"
if [[ ! "$proceed" =~ ^[Yy] ]]; then
    echo "Skipping. Review/edit $COMPOSE_FILE, then run:"
    echo "  ${COMPOSE} up -d --build"
    exit 0
fi

if [[ "$DO_BUILD" -eq 1 ]]; then
    info "Building (${COMPOSE} build --no-cache)..."
    "${COMPOSE_CMD[@]}" build --no-cache
fi

info "Starting..."
"${COMPOSE_CMD[@]}" up -d

echo
info "Done. Following logs (Ctrl+C to stop watching — the container keeps running):"
"${COMPOSE_CMD[@]}" logs -f
