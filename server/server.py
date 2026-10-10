#!/usr/bin/env python
"""LAN TTS server for MOSS-TTS-PNY.

Exposes an HTTP API so any device on the local network can request voiced lines
with a chosen speaker. Designed as the TTS backend for an interactive visual novel.

Endpoints:
  GET  /health           -> {"status","model_loaded","device","sample_rate"}
  GET  /speakers         -> [{"id","name","label","language"}, ...]
  POST /tts              -> audio/wav  (or JSON base64 if format=="base64")
  GET  /tts?text=..&speaker_id=31  -> audio/wav  (browser-friendly quick test)
  GET  /                 -> tiny HTML test page

Run:
  venv/bin/python server.py --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import re
import sys
import threading
import time
import unicodedata
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional

# The runtime may os.execv-restart this process (to set LD_LIBRARY_PATH) using
# sys.argv. We chdir below, so argv[0] must be absolute or the re-exec can't find
# this file. Make it absolute before anything else.
sys.argv[0] = str(Path(__file__).resolve())

# --- locate the model bundle and make its modules importable -----------------
RELEASE_DIR = Path(__file__).resolve().parent / "release"
BUNDLE_DIR = RELEASE_DIR / "moss_tts_torchopt_runner_bundle"
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))
# run from the bundle dir so relative asset paths resolve
os.chdir(BUNDLE_DIR)

import base64

import numpy as np
import torch
from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# imported from the bundle (added to sys.path above)
from portable_tts_runtime import OptimizedTTSConfig, OptimizedTTSRunner  # noqa: E402

SPEAKER_CSV = BUNDLE_DIR / "speaker_id_map.csv"

# ----------------------------------------------------------------------------
# Global runner + a lock. model.generate() uses CUDA graphs and is stateful, so
# only ONE synthesis may run at a time. Requests serialize on this lock.
# ----------------------------------------------------------------------------
_runner: Optional[OptimizedTTSRunner] = None
_lock = threading.Lock()
_default_max_new_tokens = 256
# Default audio sampling temperature. Configurable (env MOSS_AUDIO_TEMPERATURE or
# --temperature); a request that doesn't set audio_temperature falls back to this.
_default_audio_temperature = float(os.environ.get("MOSS_AUDIO_TEMPERATURE", "0.85"))
# Preset hot-swap state (the web-page dropdown reloads the model in the background).
_reload_lock = threading.Lock()
_reload_state = {"busy": False, "target": None, "error": None}
PRESET_FILE = Path(__file__).resolve().parent / "preset.conf"


def load_speakers() -> list[dict]:
    speakers: list[dict] = []
    with open(SPEAKER_CSV, newline="") as f:
        for row in csv.DictReader(f):
            speakers.append(
                {
                    "id": int(row["speaker_id"]),
                    # SillyTavern's built-in XTTSv2 TTS provider reads voice.voice_id
                    # specifically (not voice.id) after a successful name match, and
                    # throws "Unable to attain voiceId" if it's missing — see
                    # index.js's processTtsQueue(). Alias of "id"; purely additive,
                    # nothing else here keys off "voice_id".
                    "voice_id": int(row["speaker_id"]),
                    "name": row.get("display_name") or row.get("name") or "",
                    "label": row.get("label") or row.get("display_name") or "",
                    "language": "en",
                    "utterances": int(row.get("utterance_count", 0) or 0),
                }
            )
    speakers.sort(key=lambda s: s["id"])
    return speakers


SPEAKERS = load_speakers()
SPEAKER_IDS = {s["id"] for s in SPEAKERS}

# ----------------------------------------------------------------------------
# Text normalization
# ----------------------------------------------------------------------------
# The model has no phonemizer: the text string is tokenized and fed straight to
# the LM, which learned pronunciation from raw text. Two consequences this fixes:
#   1. Out-of-distribution punctuation (ellipses, em/en-dashes, asterisks, stray
#      symbols) pushes it off the rails -> screeching, or a 5-second hiss that
#      fades on a trailing "s". Mapping those to plain commas/periods stops it.
#   2. Some words it just says wrong (e.g. "beg" -> "borg"). There's no way to
#      fix that except spelling the word differently. That's the job of the
#      user-editable override dictionary in pronunciations.txt.
PROJECT_DIR = RELEASE_DIR.parent  # computed before the chdir into the bundle
PRONUNCIATION_FILE = PROJECT_DIR / "pronunciations.txt"

_SMART = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",  # single quotes
    "“": '"', "”": '"', "„": '"', "‟": '"',  # double quotes
    "–": "—", "—": "—", "―": "—", "−": "-",  # en/em/bar dashes -> pause mark, minus

    " ": " ", " ": " ", " ": " ", " ": " ",  # odd spaces
})

_ONES = ("zero one two three four five six seven eight nine ten eleven twelve "
         "thirteen fourteen fifteen sixteen seventeen eighteen nineteen").split()
_TENS = ("_ _ twenty thirty forty fifty sixty seventy eighty ninety").split()
_SCALES = ("_ thousand million billion trillion").split()


def _under_1000(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + _under_1000(n % 100))


def _int_to_words(n: int) -> str:
    if n == 0:
        return "zero"
    if n < 0:
        return "minus " + _int_to_words(-n)
    groups = []
    while n > 0:
        n, r = divmod(n, 1000)
        groups.append(r)
    if len(groups) > len(_SCALES):
        return None  # too big to bother — leave the digits alone
    out = []
    for i in range(len(groups) - 1, -1, -1):
        if groups[i] == 0:
            continue
        out.append(_under_1000(groups[i]))
        if i > 0:
            out.append(_SCALES[i])
    return " ".join(out)


def _year_to_words(n: int) -> Optional[str]:
    """1996 -> 'nineteen ninety six', 2007 -> 'twenty oh seven', 2000 -> 'two thousand'."""
    if not (1000 <= n <= 2999):
        return None
    hi, lo = divmod(n, 100)
    if lo == 0:
        return _int_to_words(hi) + " hundred" if n % 1000 else _int_to_words(n)
    if n % 1000 < 10:  # 2007 -> twenty oh seven
        return _int_to_words(hi) + " oh " + _ONES[lo]
    return _int_to_words(hi) + " " + _under_1000(lo)


def _expand_number(m: re.Match) -> str:
    whole = m.group(0)
    # decimals: read digit-by-digit after the point
    if "." in whole:
        intpart, frac = whole.split(".", 1)
        words = _int_to_words(int(intpart)) if intpart else "zero"
        if words is None:
            return whole
        return words + " point " + " ".join(_ONES[int(d)] for d in frac)
    n = int(whole)
    if len(whole) == 4 and not whole.startswith("0"):
        y = _year_to_words(n)  # treat bare 4-digit numbers as years (common in dialogue)
        if y:
            return y
    w = _int_to_words(n)
    return w if w is not None else whole


def load_pronunciations() -> dict[str, str]:
    """Parse pronunciations.txt: `word = respelling`, one per line, # comments."""
    mapping: dict[str, str] = {}
    if not PRONUNCIATION_FILE.exists():
        return mapping
    for raw in PRONUNCIATION_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if key:
            mapping[key.lower()] = val
    return mapping


# hot-reload cache: re-read the file only when it changes (so you can tune
# pronunciations live without restarting the server)
_pron_cache: tuple[float, dict[str, str], Optional[re.Pattern]] = (0.0, {}, None)


def _get_pronunciations() -> tuple[dict[str, str], Optional[re.Pattern]]:
    global _pron_cache
    try:
        mtime = PRONUNCIATION_FILE.stat().st_mtime if PRONUNCIATION_FILE.exists() else 0.0
    except OSError:
        mtime = 0.0
    if mtime != _pron_cache[0]:
        mapping = load_pronunciations()
        pattern = None
        if mapping:
            keys = sorted(mapping, key=len, reverse=True)
            pattern = re.compile(
                r"(?<!\w)(" + "|".join(re.escape(k) for k in keys) + r")(?!\w)",
                re.IGNORECASE,
            )
        _pron_cache = (mtime, mapping, pattern)
    return _pron_cache[1], _pron_cache[2]


def _apply_respellings(text: str) -> str:
    mapping, pattern = _get_pronunciations()
    if not pattern:
        return text

    def repl(m: re.Match) -> str:
        orig = m.group(0)
        sub = mapping[orig.lower()]
        # preserve a leading capital so sentence starts stay capitalized
        if orig[:1].isupper() and sub[:1].islower():
            sub = sub[:1].upper() + sub[1:]
        return sub

    return pattern.sub(repl, text)


# Per-install client-side overrides (v2.8): a player's own pronunciations.txt,
# kept and edited entirely on their machine (see api_config_menu.gd's
# "Pronunciations" editor) and sent with every request rather than requiring
# SSH access to whatever server they've pointed the game at. Applied BEFORE
# the server's own pronunciations.txt/auto-respell pass, not after — running
# first means the substituted text no longer contains the original word, so
# the server-side dictionary can't re-match and clobber it. Same whole-word,
# case-preserving matching as _apply_respellings, just built fresh per
# request from the caller-supplied dict instead of the cached server file.
_CLIENT_PRON_MAX_ENTRIES = 300  # sanity cap — a real hand-edited list is dozens of lines, not thousands


def _apply_client_respellings(text: str, mapping: dict) -> str:
    if not mapping:
        return text
    lower_map = {str(k).strip().lower(): str(v) for k, v in mapping.items() if str(k).strip()}
    if not lower_map:
        return text
    if len(lower_map) > _CLIENT_PRON_MAX_ENTRIES:
        lower_map = dict(list(lower_map.items())[:_CLIENT_PRON_MAX_ENTRIES])
    keys = sorted(lower_map, key=len, reverse=True)
    pattern = re.compile(
        r"(?<!\w)(" + "|".join(re.escape(k) for k in keys) + r")(?!\w)",
        re.IGNORECASE,
    )

    def repl(m: re.Match) -> str:
        orig = m.group(0)
        sub = lower_map[orig.lower()]
        if orig[:1].isupper() and sub[:1].islower():
            sub = sub[:1].upper() + sub[1:]
        return sub

    return pattern.sub(repl, text)


# ----------------------------------------------------------------------------
# Auto-respell: a phonemizer IN FRONT of the model. The model reads graphemes,
# so we convert hard words to phonemes (CMU dict via `pronouncing`) and write
# the phonemes back out as spellings it reliably reads ("erosion" -> "ih roh
# zhuhn"). Applied only to words that are (a) rare enough that the show-dialogue
# finetune likely eroded them (wordfreq Zipf below MOSS_RESPELL_ZIPF) and
# (b) actually in the CMU dictionary — unknown words (pony names) are left
# alone, and pronunciations.txt overrides always win (they run first).
# ----------------------------------------------------------------------------
try:
    import pronouncing
    from wordfreq import zipf_frequency
    _RESPELL_AVAILABLE = True
except ImportError:  # optional: server still runs without the two packages
    _RESPELL_AVAILABLE = False

# Ear-test verdict (2026-07-04): spaced-syllable respellings come out CHOPPY — the
# model reads the gaps as word boundaries. And the slurring turned out to be
# per-SPEAKER (Twilight fine, Maud/Celestia slur), so a global text rewrite is
# the wrong knob. OFF by default; kept for per-request experiments.
AUTO_RESPELL_DEFAULT = os.environ.get("MOSS_AUTO_RESPELL", "0") != "0"
# Zipf scale: ~7 = "the", ~4 = everyday word, <4 = uncommon. Words BELOW the
# threshold get respelled. Raise it to respell more aggressively.
RESPELL_ZIPF_THRESHOLD = float(os.environ.get("MOSS_RESPELL_ZIPF", "4.2"))

# Show vocabulary is rare in general English (zipf gates it in) but is exactly
# what the finetune knows best — never respell it. Speaker names + world words.
_RESPELL_SKIP: set[str] = {
    "pony", "ponies", "everypony", "anypony", "somepony", "nopony", "filly",
    "fillies", "colt", "colts", "foal", "foals", "mare", "mares", "stallion",
    "stallions", "pegasus", "pegasi", "unicorn", "unicorns", "alicorn",
    "griffon", "griffons", "equestria", "equestrian", "canterlot", "ponyville",
    "cloudsdale", "manehattan", "appleloosa", "harmony", "cutie", "wonderbolt",
    "wonderbolts", "changeling", "changelings", "timberwolf", "timberwolves",
}
for _s in SPEAKERS:
    _RESPELL_SKIP.update(re.findall(r"[a-z]+", _s["name"].lower()))

# ARPAbet -> "spell what you hear". Vowel spellings chosen to be unambiguous
# English (igh = "eye" sound, oh, ee, ...) once glued to their syllable.
_ARPA_SPELL = {
    "AA": "ah", "AE": "a", "AH": "uh", "AO": "aw", "AW": "ow", "AY": "igh",
    "EH": "eh", "ER": "er", "EY": "ay", "IH": "ih", "IY": "ee", "OW": "oh",
    "OY": "oy", "UH": "oo", "UW": "oo",
    "B": "b", "CH": "ch", "D": "d", "DH": "th", "F": "f", "G": "g", "HH": "h",
    "JH": "j", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ng", "P": "p",
    "R": "r", "S": "s", "SH": "sh", "T": "t", "TH": "th", "V": "v", "W": "w",
    "Y": "y", "Z": "z", "ZH": "zh",
}
_ARPA_VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH",
                "IY", "OW", "OY", "UH", "UW"}


def _phones_to_respelling(phones: str, sep: str = " ") -> str:
    """'IH0 R OW1 ZH AH0 N' -> 'ih roh zhuhn' (syllable chunks, stress dropped).
    Each vowel takes the single consonant before it as its onset; remaining
    consonants trail the previous syllable. sep='' joins into one word — the
    ear-test showed spaced syllables come out CHOPPY, joined reads fluent."""
    toks = [p.rstrip("012") for p in phones.split()]
    if not all(t in _ARPA_SPELL for t in toks):
        return ""  # unknown phone — bail, leave the word as written
    sylls: list[list[str]] = []
    cur: list[str] = []
    for i, t in enumerate(toks):
        if t in _ARPA_VOWELS and any(x in _ARPA_VOWELS for x in cur):
            # new syllable: steal at most one preceding consonant as onset
            onset = [cur.pop()] if cur and cur[-1] not in _ARPA_VOWELS else []
            sylls.append(cur)
            cur = onset
        cur.append(t)
    sylls.append(cur)
    return sep.join("".join(_ARPA_SPELL[t] for t in s) for s in sylls if s)


_VOICELESS = {"P", "T", "K", "F", "TH", "S", "SH", "CH", "HH"}
_SIBILANT = {"S", "Z", "SH", "ZH", "CH", "JH"}
_PREFIXES = (("mis", "M IH0 S"), ("over", "OW1 V ER0"), ("out", "AW1 T"),
             ("pre", "P R IY0"), ("re", "R IY0"), ("un", "AH0 N"))


def _phones_for(word: str, depth: int = 0) -> Optional[str]:
    """CMU phones for `word`, deriving inflections (ed/s/ing/ly) and common
    prefixes from a known stem when the exact form isn't in the dictionary."""
    hits = pronouncing.phones_for_word(word)
    if hits:
        return hits[0]
    if depth >= 2 or len(word) < 5:
        return None
    if word.endswith("ing"):
        for stem in (word[:-3], word[:-3] + "e"):
            p = _phones_for(stem, depth + 1)
            if p:
                return p + " IH0 NG"
    if word.endswith("ed"):
        for stem in (word[:-1], word[:-2]):
            p = _phones_for(stem, depth + 1)
            if p:
                last = p.split()[-1].rstrip("012")
                if last in ("T", "D"):
                    return p + " IH0 D"
                return p + (" T" if last in _VOICELESS else " D")
    if word.endswith("s") and not word.endswith("ss"):
        for stem in (word[:-1], word[:-2] if word.endswith("es") else word[:-1]):
            p = _phones_for(stem, depth + 1)
            if p:
                last = p.split()[-1].rstrip("012")
                if last in _SIBILANT:
                    return p + " IH0 Z"
                return p + (" S" if last in _VOICELESS else " Z")
    if word.endswith("ly"):
        p = _phones_for(word[:-2], depth + 1)
        if p:
            return p + " L IY0"
    for pre, pre_phones in _PREFIXES:
        if word.startswith(pre):
            p = _phones_for(word[len(pre):], depth + 1)
            if p:
                return pre_phones + " " + p
    return None


def _respell_word(word: str) -> str:
    """Respelling for one lowercase word, or '' to leave it untouched."""
    if len(word) < 4 or not word.isalpha() or word in _RESPELL_SKIP:
        return ""
    if zipf_frequency(word, "en") >= RESPELL_ZIPF_THRESHOLD:
        return ""  # common enough that the model says it fine
    phones = _phones_for(word)
    if not phones:
        return ""  # not in CMU dict even via stem (names etc.) — don't guess
    return _phones_to_respelling(phones)


_respell_cache: dict[str, str] = {}
_WORD_RE = re.compile(r"(?<![\w'])[A-Za-z]{4,}(?![\w'])")


def _auto_respell(text: str) -> tuple[str, dict[str, str]]:
    """Respell rare dictionary words in `text`. Returns (new_text, {word: respelling})."""
    applied: dict[str, str] = {}
    if not _RESPELL_AVAILABLE:
        return text, applied

    def repl(m: re.Match) -> str:
        orig = m.group(0)
        key = orig.lower()
        if key in _get_pronunciations()[0]:
            return orig  # word has a pronunciations.txt entry — the user owns it
        if key not in _respell_cache:
            _respell_cache[key] = _respell_word(key)
        sub = _respell_cache[key]
        if not sub:
            return orig
        applied[key] = sub
        if orig[:1].isupper():
            sub = sub[:1].upper() + sub[1:]
        return sub

    return _WORD_RE.sub(repl, text), applied


# ----------------------------------------------------------------------------
# Verified pronunciation cache + solver (v2.5).
# Measured facts this design rests on (2026-07-04, per-speaker Whisper matrix):
#   - Mispronunciation is PER-SPEAKER (data-starved voices slur — Maud has 177
#     training clips vs Twilight's 4628) and DETERMINISTIC (0/10 clean retries).
#   - JOINED respellings fix it ("precippitation", "erosean" -> 3/3 verified),
#     but the winning spelling differs per word AND per speaker.
# So: keep a per-(word, speaker) cache of Whisper-VERIFIED respellings. Cached
# winners substitute before synthesis; risky words with no verdict yet are
# transcribed back after synthesis (faster-whisper, CPU) and failures get a
# bounded in-line search over candidate respellings. Verified winners persist
# in pronunciation_cache.json (human-editable; delete an entry to re-solve).
# ----------------------------------------------------------------------------
import difflib
import json

PRON_CACHE_FILE = PROJECT_DIR / "pronunciation_cache.json"
# OFF by user decision (2026-07-04): after the LoRA fine-tune, the model's raw
# output is what plays — no hidden verification/retries/respellings. Manual
# pronunciations.txt entries still apply. Set MOSS_FIX_PRON=1 to re-enable.
FIX_PRON_DEFAULT = os.environ.get("MOSS_FIX_PRON", "0") != "0"
PRON_OK_RATIO = 0.8            # Whisper fuzzy-match ratio counted as "said it right"
PRON_TRIES_PER_LINE = int(os.environ.get("MOSS_PRON_TRIES", "3"))

# ON by default (unlike MOSS_FIX_PRON above) — this catches a different,
# more clearly-a-bug glitch: a single word occasionally getting stuck in what
# looks like a low-level generation loop and dragging on for several
# seconds, unrelated to word rarity (observed on "YES!" and "you" — neither
# a rare word _risky_words would ever flag) and NOT requiring the model hit
# its token cap (unlike _loop_guard_trim's whole-line-repeat case; this can
# happen mid-generation, well within budget). Simple regenerate-and-retry,
# not a respelling search — the fix isn't about how the word's spelled.
FIX_STUCK_WORDS_DEFAULT = os.environ.get("MOSS_FIX_STUCK_WORDS", "1") != "0"
STUCK_WORD_RETRIES = int(os.environ.get("MOSS_STUCK_WORD_RETRIES", "2"))
# A word is flagged if its actual spoken duration blows past EITHER bar —
# relative (some multiple of a rough expected duration for its length) OR
# absolute (long regardless of length) — so a naturally slow long word and a
# short word stuck for several seconds both get caught by the bar suited to
# them. First-pass numbers, calibrated against two real observed glitches
# (a stuck "YES!" at ~5s, a stuck "you" at ~3.3s) rather than verified
# end-to-end — expect to retune once this sees more real traffic.
STUCK_WORD_REL_MULT = float(os.environ.get("MOSS_STUCK_WORD_REL_MULT", "2.5"))
STUCK_WORD_ABS_CEIL_SEC = float(os.environ.get("MOSS_STUCK_WORD_ABS_CEIL", "1.5"))

_pron_lock = threading.Lock()


def _pron_load() -> dict:
    try:
        return json.loads(PRON_CACHE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


# {word: {str(speaker_id): {"use": respelling ("" = plain verified), "tried": [...]}}}
_pron: dict = _pron_load()


def _pron_save() -> None:
    tmp = PRON_CACHE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(_pron, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(PRON_CACHE_FILE)


def _pron_entry(word: str, sid: int) -> dict:
    return _pron.setdefault(word, {}).setdefault(str(sid), {})


_whisper = None
_whisper_lock = threading.Lock()


def _get_whisper():
    """faster-whisper on CPU int8 — verifies words without touching the GPU."""
    global _whisper
    with _whisper_lock:
        if _whisper is None:
            from faster_whisper import WhisperModel
            _whisper = WhisperModel("small.en", device="cpu", compute_type="int8")
    return _whisper


def _transcribed_words(wav_bytes: bytes) -> list[str]:
    segs, _ = _get_whisper().transcribe(io.BytesIO(wav_bytes), beam_size=5)
    return re.findall(r"[a-z']+", " ".join(s.text for s in segs).lower())


def _heard_ratio(target: str, heard: list[str]) -> float:
    """Best fuzzy match of `target` against heard words (and adjacent pairs,
    since slurred audio often transcribes as two fragments)."""
    cands = heard + [a + b for a, b in zip(heard, heard[1:])]
    return max((difflib.SequenceMatcher(None, target, c).ratio() for c in cands), default=0.0)


def _heard_ok(target: str, heard: list[str]) -> bool:
    # long words need a stricter bar: at 0.8, "miscommuted" passed for
    # "miscommunicated" (a whole syllable swallowed) and got wrongly cached
    thr = 0.87 if len(target) >= 10 else PRON_OK_RATIO
    return _heard_ratio(target, heard) >= thr


def _risky_words(text: str) -> list[str]:
    """Rare words worth verifying: same gate as auto-respell, minus the CMU-dict
    requirement (the solver's mutations don't need phonemes to exist)."""
    if not _RESPELL_AVAILABLE:
        return []
    seen: set[str] = set()
    out: list[str] = []
    user_map = _get_pronunciations()[0]
    for m in _WORD_RE.finditer(text):
        w = m.group(0).lower()
        if w in seen or w in _RESPELL_SKIP or w in user_map:
            continue
        seen.add(w)
        if zipf_frequency(w, "en") < RESPELL_ZIPF_THRESHOLD:
            out.append(w)
    return out[:8]


# alternate vowel spellings for extra phoneme-respelling variants
_ARPA_ALT1 = {**_ARPA_SPELL, "AY": "y", "EY": "ai", "AH": "ah", "OW": "oe"}
_ARPA_ALT2 = {**_ARPA_SPELL, "AY": "eye", "IY": "ea", "UW": "ew", "ER": "ur"}


def _pron_candidates(word: str) -> list[str]:
    """Ranked respelling candidates, best rules first — each rule produced a
    Whisper-verified winner in testing: suffix swaps (erosion->erosean),
    consonant doubling (precippitation), joined phoneme respelling (plus vowel
    alternates), silent-letter simplifications, leading-vowel swaps."""
    out: list[str] = []
    for suf, reps in (("sion", ("sean", "shun", "zhun")), ("tion", ("shun", "tayshun")),
                      ("sure", ("zher",)), ("ture", ("cher",))):
        if word.endswith(suf):
            out += [word[: -len(suf)] + r for r in reps]
    for m in re.finditer(r"(?<=[aeiou])([bdfgklmnprstz])(?=[aeiouy])", word):
        i = m.start(1)
        out.append(word[:i] + word[i] + word[i:])
    for cluster, rep in (("gn", "n"), ("mb", "m"), ("bt", "t"), ("stle", "sle")):
        if cluster in word:
            out.append(word.replace(cluster, rep))
    p = _phones_for(word)
    if p:
        for table in (None, _ARPA_ALT1, _ARPA_ALT2):
            out.append(_phones_to_respelling(p, sep="") if table is None
                       else _phones_to_respelling_with(p, table))
    if word[0] in "aeiou":
        out.append({"a": "uh", "e": "i", "i": "ee", "o": "oh", "u": "oo"}[word[0]] + word[1:])
    seen: set[str] = set()
    res: list[str] = []
    for c in out:
        if c and c != word and c not in seen:
            seen.add(c)
            res.append(c)
    return res[:14]


def _phones_to_respelling_with(phones: str, table: dict[str, str]) -> str:
    toks = [p.rstrip("012") for p in phones.split()]
    if not all(t in table for t in toks):
        return ""
    return "".join(table[t] for t in toks)


def _sub_word(text: str, word: str, repl: str) -> str:
    """Whole-word, case-preserving substitution of one word."""
    def r(m: re.Match) -> str:
        s = repl
        if m.group(0)[:1].isupper() and s[:1].islower():
            s = s[:1].upper() + s[1:]
        return s
    return re.sub(rf"(?<![\w']){re.escape(word)}(?![\w'])", r, text, flags=re.IGNORECASE)


def normalize_text(text: str, expand_numbers: bool = True, auto_respell: bool = True,
                   client_pronunciations: Optional[dict] = None) -> str:
    """Make text easier for the model to pronounce. Punctuation/symbol cleanup is
    always applied; word respellings come from the caller's own client_pronunciations
    (if any) first, then this server's pronunciations.txt, then rare words are
    auto-respelled from their CMU-dict phonemes (auto_respell)."""
    if not text:
        return text
    t = unicodedata.normalize("NFKC", text)
    t = t.translate(_SMART)

    # ALL-CAPS words (shouting / emphasis) make the model screech or drag out
    # the vowels. Fold them to a normal Titlecase word so it reads them plainly:
    #   "STOP" -> "Stop", "DON'T" -> "Don't", "WHAT" -> "What".
    # 1-letter words ("I", "A") don't match; acronyms get Titlecased too, which
    # the model pronounces far more reliably than a string of capitals.
    t = re.sub(r"\b[A-Z][A-Z']*[A-Z]\b", lambda m: m.group(0).capitalize(), t)

    # Quote marks around words/speech aren't spoken — strip them. Keep apostrophes
    # INSIDE a word (contractions: don't, o'clock, y'all) which the model needs;
    # only drop apostrophes at a word edge (i.e. used as quotes: 'allowed').
    t = t.replace('"', " ")
    t = re.sub(r"(?<![A-Za-z])'|'(?![A-Za-z])", "", t)

    if client_pronunciations:
        t = _apply_client_respellings(t, client_pronunciations)  # caller's own local overrides — applied first, see above
    t = _apply_respellings(t)                       # user overrides (beg -> behg, etc.)
    if auto_respell:
        t, _ = _auto_respell(t)                     # rare words -> phonetic respelling

    t = t.replace("…", "...")                  # unicode ellipsis -> dots
    t = re.sub(r"(?<=\w)-(?=\w)", " ", t)           # hard-to-pronounce -> hard to pronounce
    t = re.sub(r"\s*-{2,}\s*", "—", t)              # -- pause dash -> pause mark
    t = re.sub(r"\s+-\s+", "—", t)                  # spaced hyphen pause -> pause mark

    if expand_numbers:
        t = re.sub(r"\$\s?(\d[\d,]*)", lambda m: _int_to_words(int(m.group(1).replace(",", ""))) + " dollars", t)
        t = t.replace(",", "\0")                    # protect existing commas from the digit-group strip
        t = re.sub(r"(?<=\d)\0(?=\d{3}\b)", "", t)  # 1,000 -> 1000 (drop thousands separators)
        t = t.replace("\0", ",")
        t = re.sub(r"\d+\.\d+|\d+", _expand_number, t)

    t = t.replace("&", " and ").replace("%", " percent ").replace("@", " at ")
    t = t.replace("°", " degrees ").replace("+", " plus ").replace("=", " equals ")

    # strip symbols the model never saw cleanly (asterisks/markdown/brackets/etc.)
    t = re.sub(r"[\(\)\[\]\{\}]", ", ", t)          # parentheticals -> comma pauses
    t = re.sub(r"[*_~`#^|<>\\/]", " ", t)           # stray markup/symbols -> space

    # ellipses: drop leading, soften trailing to a period, internal -> pause mark
    t = re.sub(r"^\s*\.{2,}\s*", "", t)
    t = re.sub(r"\s*\.{2,}\s*$", ". ", t)
    t = re.sub(r"\s*\.{2,}\s*", "—", t)

    # pause marks (—): the model never saw them, so they are NOT sent to it —
    # the /tts endpoint splits the text here and inserts real silence between
    # the pieces. Tidy them: collapse runs, drop at the edges (trailing -> .)
    t = re.sub(r"(\s*—\s*)+", "—", t)
    t = re.sub(r"^—", "", t)
    t = re.sub(r"—(?=[.!?,;:])", "", t)
    t = re.sub(r"—\s*$", ".", t)

    # collapse runaway terminal punctuation: "?!?!" / "!!!" -> one mark
    t = re.sub(r"[!?]{2,}", lambda m: "?" if "?" in m.group(0) else "!", t)

    t = re.sub(r"\s+([,.!?;:])", r"\1", t)          # no space before punctuation
    t = re.sub(r"([,;:])(?=\S)", r"\1 ", t)          # one space after a comma
    t = re.sub(r",\s*(?=[,.!?;:])", "", t)           # drop comma stacked on other punct
    t = re.sub(r"\s+", " ", t).strip()               # collapse ALL whitespace, incl. newlines —
                                                       # raw \n reaching the model untouched causes
                                                       # it to produce no usable output on multi-line
                                                       # input (paragraphs, poetry with line breaks)

    if t and t[-1] not in ".!?,;:\"'":               # ensure it ends on a sentence mark
        t += "."
    return t


# ----------------------------------------------------------------------------
# Emotion / energy control (MOSS-TTS-PNY style_feature_dim=2 checkpoint).
# The model takes an emotion class id (0-11) plus an energy scalar in [0,1]. The
# game's AI writes a leading tag like "[happy:0.8]" (or just "[happy]") on each
# line; we parse it here, map the name -> id, clamp energy, and strip the tag
# before synthesis. Keeping this server-side means the name->id map is one place
# and the emotion logic is shared by every client.
# ----------------------------------------------------------------------------
EMOTION_IDS = {
    "neutral": 0, "happy": 1, "sad": 2, "angry": 3, "annoyed": 4,
    "fearful": 5, "surprised": 6, "calm": 7, "disgusted": 8,
    "whispering": 9, "nonverbal": 10, "shouting": 11,
}
# id 10 ("nonverbal") is the reserved NSFW placeholder and does nothing yet —
# never send it to the model; fold it to neutral.
FORBIDDEN_EMOTION_IDS = {10}
# Classes the model renders badly get remapped rather than sent as-is.
# 9 ("whispering") doesn't produce a whisper (user ear-test 2026-07-04) —
# fold it to calm, keeping the tag's energy.
EMOTION_REMAP = {9: 7}
DEFAULT_EMOTION_ID = 0            # neutral
DEFAULT_ENERGY = 0.5

# words the AI might use -> one of the 11 usable classes (nonverbal excluded)
_EMOTION_ALIASES = {
    "neutral": "neutral", "flat": "neutral", "normal": "neutral",
    "happy": "happy", "joy": "happy", "joyful": "happy", "cheerful": "happy", "excited": "happy",
    "sad": "sad", "sorrow": "sad", "sorrowful": "sad", "depressed": "sad", "down": "sad",
    "angry": "angry", "mad": "angry", "furious": "angry", "enraged": "angry",
    "annoyed": "annoyed", "irritated": "annoyed", "frustrated": "annoyed",
    "fearful": "fearful", "afraid": "fearful", "scared": "fearful", "nervous": "fearful", "anxious": "fearful",
    "surprised": "surprised", "shocked": "surprised", "astonished": "surprised",
    "calm": "calm", "relaxed": "calm", "content": "calm", "gentle": "calm", "soft": "calm",
    "disgusted": "disgusted", "disgust": "disgusted", "grossed": "disgusted",
    "whispering": "whispering", "whisper": "whispering", "hushed": "whispering",
    "shouting": "shouting", "shout": "shouting", "yelling": "shouting", "screaming": "shouting",
}

# leading "[emotion]" or "[emotion:energy]" (energy may be 0-1 or 0-100)
_EMOTION_TAG = re.compile(
    r"^\s*\[\s*([A-Za-z]+)\s*(?::\s*([0-9]*\.?[0-9]+)\s*(?::\s*([0-9]*\.?[0-9]+)\s*)?)?\]\s*"
)


def parse_emotion_tag(
    text: str,
    default_id: int = DEFAULT_EMOTION_ID,
    default_energy: float = DEFAULT_ENERGY,
) -> tuple[int, float, Optional[float], str]:
    """Pull a leading [emotion:energy] or [emotion:energy:strength] tag off `text`.
    Returns (emotion_id, energy, strength_or_None, remaining_text). Unknown or
    forbidden emotion -> neutral; missing/garbage energy -> default. Energy accepts
    0..1 or 0..100. Strength (optional 3rd slot) over-drives the emotion conditioning
    — 1.0 normal, up to 3.0 for characters whose personality resists the tag."""
    m = _EMOTION_TAG.match(text or "")
    if not m:
        return default_id, default_energy, None, text
    canon = _EMOTION_ALIASES.get(m.group(1).lower())
    eid = EMOTION_IDS.get(canon, default_id) if canon else default_id
    if eid in FORBIDDEN_EMOTION_IDS:
        eid = DEFAULT_EMOTION_ID
    energy = default_energy
    if m.group(2):
        try:
            energy = float(m.group(2))
            if energy > 1.0:            # tolerate "80" meaning 0.8
                energy = energy / 100.0
            energy = max(0.0, min(1.0, energy))
        except ValueError:
            energy = default_energy
    strength: Optional[float] = None
    if m.group(3):
        try:
            strength = max(0.0, min(2.3, float(m.group(3))))
        except ValueError:
            strength = None
    return eid, energy, strength, text[m.end():]


def shorten_silences(
    audio: torch.Tensor,
    sample_rate: int,
    max_gap_ms: int = 300,
    edge_ms: int = 40,
    thresh_db: float = -40.0,
    min_run_ms: int = 120,
) -> torch.Tensor:
    """Collapse the long dead-air gaps the model emits at commas/ellipses/sentence
    breaks. Internal silences longer than `max_gap_ms` are shrunk to `max_gap_ms`;
    leading/trailing silence is trimmed to `edge_ms`. Speech is untouched — a run
    must be quieter than `thresh_db` (relative to the clip's peak) AND longer than
    `min_run_ms` to count as a gap, so natural short pauses inside words survive.
    """
    x = audio.detach().cpu().numpy().astype(np.float32).reshape(-1)
    n = x.size
    if n == 0:
        return audio
    peak = float(np.max(np.abs(x)))
    if peak <= 0.0:
        return audio
    thresh = peak * (10.0 ** (thresh_db / 20.0))
    silent = (np.abs(x) < thresh).astype(np.int8)

    max_gap = int(sample_rate * max_gap_ms / 1000)
    edge = int(sample_rate * edge_ms / 1000)
    min_run = int(sample_rate * min_run_ms / 1000)

    # boundaries of each contiguous silent run
    edges = np.flatnonzero(np.diff(np.concatenate(([0], silent, [0]))))
    starts, ends = edges[0::2], edges[1::2]  # ends are exclusive

    keep = np.ones(n, dtype=bool)
    for s, e in zip(starts, ends):
        run = e - s
        at_start, at_end = (s == 0), (e == n)
        if run < min_run and not (at_start or at_end):
            continue  # short internal pause — leave it
        target = edge if (at_start or at_end) else max_gap
        if run <= target:
            continue
        if at_start:
            keep[s : e - target] = False          # keep the tail of leading silence
        elif at_end:
            keep[s + target : e] = False           # keep the head of trailing silence
        else:
            half = target // 2
            keep[s + half : e - half] = False      # keep a short pause, drop the middle

    if keep.all():
        return audio
    return torch.from_numpy(x[keep])


def audio_to_wav_bytes(audio: torch.Tensor, sample_rate: int) -> bytes:
    pcm = (audio.clamp(-1.0, 1.0) * 32767.0).to(torch.int16).cpu().numpy()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return buf.getvalue()


# ----------------------------------------------------------------------------
class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    speaker_id: int = 31
    language: str = "en"
    format: str = "wav"  # "wav" | "base64"
    audio_temperature: Optional[float] = None  # None => server default (_default_audio_temperature)
    audio_top_p: float = 0.95
    audio_top_k: int = 50
    audio_repetition_penalty: float = 1.05  # model authors' default; tames the drone/hiss. 1.1+ destabilizes -> cut-off sentences
    max_new_tokens: Optional[int] = None
    trim_silence: bool = True          # collapse the long pauses at punctuation
    silence_max_gap_ms: int = 300      # longest internal pause to keep
    normalize: bool = True             # clean punctuation/symbols + apply respellings
    expand_numbers: bool = True        # 100 -> "one hundred" (only if normalize)
    auto_respell: bool = AUTO_RESPELL_DEFAULT  # rare words -> phonetic respelling (only if normalize)
    client_pronunciations: dict[str, str] = Field(default_factory=dict)  # caller's own local pronunciations.txt overrides — see _apply_client_respellings
    fix_pronunciation: bool = FIX_PRON_DEFAULT  # verify rare words via Whisper; solve + cache respellings
    fix_stuck_words: bool = FIX_STUCK_WORDS_DEFAULT  # regenerate if a word's duration looks like a stuck-loop glitch
    style_emotion_id: Optional[int] = None   # 0-11; None => use [tag] in text, else Neutral
    style_energy: Optional[float] = None      # 0..1; None => use [tag] in text, else 0.5
    style_strength: Optional[float] = None    # emotion over-drive: 1.0 normal, >1 exaggerates (clamped 0..2.3; past ~2.3 the model destabilizes)


app = FastAPI(title="MOSS-TTS-PNY LAN server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    # Custom response headers are invisible to browser fetch() by default
    # unless explicitly exposed — X-Cache-Path (from /st_cache/generate)
    # needs this or client-side code silently can't read it.
    expose_headers=["X-Cache-Path"],
)


# ----------------------------------------------------------------------------
# Caller identification + request logging (v2.8). This server has several
# genuinely different callers hitting it at once — the built-in web test
# page, SillyTavern (via the XTTS/OpenAI-compat shims below), and a visual
# novel client (PPP) — and uvicorn's bare access log ("GET /tts HTTP/1.1"
# 200 OK) can't tell any of them apart. Identification sources, in order:
#   1. An explicit X-Client-App header, set by whichever client can set its
#      own request headers (the web page, PPP's Godot client).
#   2. Path alone, for SillyTavern: its XTTS/OpenAI-compat routes are fixed
#      contracts nothing else calls, so the route IS the identity — no
#      header needed and none is expected from ST itself.
#   3. "unspecified" otherwise — logged, not silently dropped, so a caller
#      that hasn't been updated to send the header is still visible, just
#      unattributed.
# ----------------------------------------------------------------------------
_ST_COMPAT_PATHS = {"/tts_to_audio/", "/tts_stream/", "/v1/audio/speech", "/set_tts_settings"}


def _client_app(request: Request) -> str:
    tag = request.headers.get("x-client-app", "").strip()
    if tag:
        return tag
    if request.url.path in _ST_COMPAT_PATHS:
        return "sillytavern"
    return "unspecified"


@app.middleware("http")
async def _request_log_middleware(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    dt = time.time() - t0
    client_ip = request.client.host if request.client else "-"
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    print(f"[req] {ts} client={_client_app(request)} ip={client_ip} "
          f"{request.method} {request.url.path} {response.status_code} {dt:.2f}s",
          flush=True)
    return response


# ----------------------------------------------------------------------------
# Cancelling generations (POST /cancel). Generation can't be interrupted
# mid-sentence (that happens inside the model's own code), so a cancel stops a
# request before its NEXT sentence: running requests stop after the current
# sentence, queued ones give up as soon as they reach the model. Cancels are
# counters: a request remembers the counters it started with, and is cancelled
# if either has moved on ("all", or its own client's) - so a cancel never
# affects requests that arrive after it.
# ----------------------------------------------------------------------------
_cancel_lock = threading.Lock()
_cancel_all_gen = 0
_cancel_client_gen: dict[str, int] = {}
_active_tickets: list = []
_tl = threading.local()           # the ticket of the request this thread is serving


class _Ticket:
    __slots__ = ("client", "text", "started", "all_gen", "client_gen", "state")

    def __init__(self, client: str, text: str):
        self.client = client
        self.text = text
        self.started = time.time()
        self.all_gen = _cancel_all_gen
        self.client_gen = _cancel_client_gen.get(client, 0)
        self.state = "queued"     # queued -> generating


def _ticket_open(client: str, text: str) -> "_Ticket":
    with _cancel_lock:
        t = _Ticket(client, text)
        _active_tickets.append(t)
        return t


def _ticket_close(t: "_Ticket") -> None:
    with _cancel_lock:
        if t in _active_tickets:
            _active_tickets.remove(t)


def _check_cancelled() -> None:
    """Raise if the current request has been cancelled. Called right before
    each sentence is generated (after waiting for the model), so it covers
    queued requests, the gap between sentences, and retries."""
    t = getattr(_tl, "ticket", None)
    if t is None:
        return
    if t.all_gen != _cancel_all_gen or t.client_gen != _cancel_client_gen.get(t.client, 0):
        raise HTTPException(status_code=409, detail="cancelled")
    t.state = "generating"


def _activity() -> list[dict]:
    now = time.time()
    with _cancel_lock:
        return [{"client": t.client, "text": t.text[:80], "state": t.state,
                 "seconds": round(now - t.started, 1)} for t in _active_tickets]


# ----------------------------------------------------------------------------
# Generation history: every successfully generated line is saved to
# history/ as <time>_<voice>_<first-words>.wav plus a matching .json with the
# details, newest MOSS_HISTORY_MAX kept (0 = off). Browsable and playable from
# the web page; the files are plain WAVs on disk if the folder is mounted.
# ----------------------------------------------------------------------------
HISTORY_DIR = PROJECT_DIR / "history"
HISTORY_MAX = int(os.environ.get("MOSS_HISTORY_MAX", "500"))
_history_lock = threading.Lock()
_HISTORY_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
HISTORY_DIR.mkdir(parents=True, exist_ok=True)   # exists even when off, for the static mount


def _slug(s: str, max_len: int) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")
    return s[:max_len].rstrip("-") or "x"


def _speaker_name(speaker_id: int) -> str:
    for sp in SPEAKERS:
        if sp["id"] == speaker_id:
            return sp.get("name") or str(speaker_id)
    return str(speaker_id)


def _history_add(out: dict, req: "TTSRequest", client: str) -> None:
    if HISTORY_MAX <= 0:
        return
    try:
        now = datetime.now()
        voice = _speaker_name(req.speaker_id)
        words = " ".join(re.sub(r"^\s*\[[^\]]*\]\s*", "", req.text).split()[:6])
        hid = f"{now.strftime('%Y%m%d-%H%M%S')}-{now.microsecond // 1000:03d}_{_slug(voice, 20)}_{_slug(words, 40)}"
        meta = {
            "id": hid,
            "time": now.isoformat(timespec="seconds"),
            "speaker_id": req.speaker_id,
            "voice": voice,
            "client": client,
            "text": req.text,
            "duration_sec": round(float(out.get("duration_sec", 0.0)), 2),
            "temperature": req.audio_temperature if req.audio_temperature is not None
                           else _default_audio_temperature,
            "preset": _active_preset,
        }
        with _history_lock:
            (HISTORY_DIR / f"{hid}.wav").write_bytes(out["wav"])
            (HISTORY_DIR / f"{hid}.json").write_text(json.dumps(meta), encoding="utf-8")
            # prune: names start with the timestamp, so sorting = oldest first
            wavs = sorted(HISTORY_DIR.glob("*.wav"))
            for old in wavs[:max(0, len(wavs) - HISTORY_MAX)]:
                old.unlink(missing_ok=True)
                old.with_suffix(".json").unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001 - history must never break a request
        print(f"[history] not saved ({exc})", flush=True)


def _synthesize(req: TTSRequest, client: str = "unspecified") -> dict:
    """Entry point for every generating endpoint: tracks the request so it can
    be cancelled (and shown as active), then saves the result to history."""
    ticket = _ticket_open(client, req.text)
    _tl.ticket = ticket
    try:
        out = _synthesize_inner(req)
    finally:
        _tl.ticket = None
        _ticket_close(ticket)
    out = _fx_apply_to_out(out, req.speaker_id)     # the voice's own effect, if one is set
    _history_add(out, req, client)
    return out


def _prepare_synth(req: TTSRequest):
    """Validate + normalize a request; returns (text, mnt, temp, emo_id, energy,
    strength). Shared by the one-shot and streaming paths so they behave alike."""
    if req.speaker_id not in SPEAKER_IDS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown speaker_id {req.speaker_id}; see GET /speakers",
        )
    assert _runner is not None, "model not loaded"
    mnt = req.max_new_tokens or _default_max_new_tokens
    temp = req.audio_temperature if req.audio_temperature is not None else _default_audio_temperature
    # Pull a leading [emotion:energy(:strength)] tag off the text (AI-written, per
    # line). A tag in the text wins; otherwise explicit request fields, then defaults.
    emo_id, energy, tag_strength, raw = parse_emotion_tag(
        req.text,
        default_id=req.style_emotion_id if req.style_emotion_id is not None else DEFAULT_EMOTION_ID,
        default_energy=req.style_energy if req.style_energy is not None else DEFAULT_ENERGY,
    )
    if emo_id in FORBIDDEN_EMOTION_IDS:      # belt-and-suspenders nonverbal guard
        emo_id = DEFAULT_EMOTION_ID
    emo_id = EMOTION_REMAP.get(emo_id, emo_id)
    strength = tag_strength if tag_strength is not None else req.style_strength
    text = (normalize_text(raw, expand_numbers=req.expand_numbers, auto_respell=req.auto_respell,
                           client_pronunciations=req.client_pronunciations)
            if req.normalize else raw)
    if not text.strip():
        raise HTTPException(status_code=400, detail="text is empty after normalization")
    return text, mnt, temp, emo_id, energy, strength


def _synthesize_inner(req: TTSRequest) -> dict:
    text, mnt, temp, emo_id, energy, strength = _prepare_synth(req)
    return _synth_pauses(text, req, mnt, temp, emo_id, energy, strength)


PAUSE_SEC = 0.45        # real silence inserted per pause mark (em-dash/ellipsis)
_PAUSE_MAX_SEGS = 6

# Natural gap for a split clause boundary (sentences, and now colon/
# semicolon breaks too — see _SENTENCE_SPLIT_RE) — shorter than PAUSE_SEC,
# which represents a deliberate dramatic pause (an em-dash), not an ordinary
# clause boundary. Using PAUSE_SEC's 0.45s between every clause of normal
# dialogue would sound choppy/robotic.
SENTENCE_PAUSE_SEC = 0.25
# Safety cap on total clause count after BOTH splits combined (em-dash parts
# further split into sentences) — same spirit as _PAUSE_MAX_SEGS, preventing
# pathological input from generating an unbounded number of requests.
# Safety cap on total clause count after BOTH splits combined (em-dash parts
# further split into sentences) — prevents truly pathological input (e.g.
# thousands of one-character fragments) from generating an unbounded number
# of requests. NOT meant to bound legitimate long-form content: two stanzas
# of The Raven alone produce 6 clauses, and the full 18-stanza poem lands
# around 50+ — an earlier value of 12 here caused exactly the cutoff this
# splitting feature was built to prevent, just for the back three-quarters
# of long text instead of the whole thing (everything past the cap gets
# crudely rejoined into one oversized, unsplit blob). 200 comfortably
# covers a full poem, a long monologue, or several pages of prose.
_MAX_TOTAL_CLAUSES = 200

# Splits on . ! ? : ; followed by whitespace. Colon/semicolon were added
# after The Raven (dense with semicolon-joined clauses) hit the same
# mid-generation cutoff on long unsplit lines that sentence-splitting was
# built to fix — a colon/semicolon-joined clause is just as long-generation-
# risky as a plain sentence, it just wasn't being split at all. Deliberately
# simple regex (matches this file's existing em-dash splitting, which
# doesn't try to be perfect either) — doesn't know about abbreviations
# ("Dr. Smith" mis-splits into "Dr." and "Smith..."), but by the time text
# reaches this point normalize_text has already expanded numbers, so
# "$3.14" doesn't false-trigger (no whitespace between the digits and the
# decimal point).
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?:;])\s+")

# Long sentences with no . ! ? : ; inside them are also split at commas, since
# one long generation is where the model most often loses its place (repeats
# its ending, or trails off into silence). Comma-separated parts are PACKED
# into pieces rather than split at every comma, so lists and Oxford commas
# ("apples, pears, and bananas") stay together and no piece is a fragment.
COMMA_SPLIT_MIN_WORDS = int(os.environ.get("MOSS_COMMA_SPLIT_MIN_WORDS", "20"))
COMMA_PIECE_MAX_WORDS = 15   # grow a piece until the next part would pass this
COMMA_PIECE_MIN_WORDS = 5    # never leave a piece shorter than this
COMMA_LIST_ITEM_MAX_WORDS = 2  # a comma part this short is a list item: never split beside it
# Gap after a comma piece - short, since it's a mid-sentence breath, not a stop.
COMMA_PAUSE_SEC = 0.12
_COMMA_SPLIT_RE = re.compile(r"(?<=,)\s+")


def _pack_commas(sentence: str) -> list[str]:
    """Split one long sentence into comma-bounded pieces of a sensible size.
    Short sentences (and anything without commas) come back unchanged."""
    def nwords(t: str) -> int:
        return len(t.split())
    if COMMA_SPLIT_MIN_WORDS <= 0 or nwords(sentence) <= COMMA_SPLIT_MIN_WORDS:
        return [sentence]
    parts = [p for p in _COMMA_SPLIT_RE.split(sentence) if p.strip()]
    if len(parts) < 2:
        return [sentence]
    # A comma is only a usable split point if the parts on BOTH sides are
    # clause-sized. Next to a short part we're inside a list ("apples, pears,
    # oranges, and bananas...") - splitting there would break the list,
    # including just before an Oxford-comma "and"/"or".
    def splittable(i: int) -> bool:   # boundary between parts[i] and parts[i+1]
        return (nwords(parts[i]) >= COMMA_LIST_ITEM_MAX_WORDS + 1
                and nwords(parts[i + 1]) >= COMMA_LIST_ITEM_MAX_WORDS + 1)
    pieces: list[str] = []
    cur = ""
    for i, part in enumerate(parts):
        if not cur:
            cur = part
        elif (nwords(cur) + nwords(part) > COMMA_PIECE_MAX_WORDS
              and nwords(cur) >= COMMA_PIECE_MIN_WORDS
              and splittable(i - 1)):
            pieces.append(cur)
            cur = part
        else:
            cur = f"{cur} {part}"
    if cur:
        if pieces and nwords(cur) < COMMA_PIECE_MIN_WORDS:
            pieces[-1] = f"{pieces[-1]} {cur}"   # short leftover joins its neighbour
        else:
            pieces.append(cur)
    return pieces


def _split_clauses(text: str) -> list[tuple[str, float]]:
    """Split text into (segment, gap_after_sec) pairs for independent
    synthesis. First splits on em-dash (a deliberate dramatic pause, 0.45s
    gap) exactly as before; each resulting piece is then further split on
    sentence-ending punctuation (a natural gap, 0.15s) if it contains more
    than one sentence. This gives long paragraphs — not just em-dash text —
    the same per-segment token budget and independent loop-guard protection,
    which is what actually prevents mid-sentence cutoffs on long lines."""
    dash_parts = [p.strip() for p in text.split("—") if p.strip()]
    if len(dash_parts) > _PAUSE_MAX_SEGS:
        dash_parts = dash_parts[:_PAUSE_MAX_SEGS - 1] + [", ".join(dash_parts[_PAUSE_MAX_SEGS - 1:])]

    # (text, ends_a_dash_part, ends_a_sentence_group) for each raw clause,
    # before final punctuation-appending and gap assignment.
    raw = []
    for i, part in enumerate(dash_parts):
        is_last_dash_part = i == len(dash_parts) - 1
        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(part) if s.strip()]
        for j, s in enumerate(sentences):
            is_last_sentence = j == len(sentences) - 1
            pieces = _pack_commas(s)
            for k, piece in enumerate(pieces):
                if k < len(pieces) - 1:
                    # mid-sentence comma piece: short gap, never a sentence end
                    raw.append((piece, False, False, True))
                else:
                    raw.append((piece, is_last_dash_part and is_last_sentence,
                                is_last_sentence, False))

    if len(raw) > _MAX_TOTAL_CLAUSES:
        head = raw[:_MAX_TOTAL_CLAUSES - 1]
        tail_texts = [c[0] for c in raw[_MAX_TOTAL_CLAUSES - 1:]]
        head.append((" ".join(tail_texts), raw[-1][1], True, False))
        raw = head

    out: list[tuple[str, float]] = []
    n = len(raw)
    for idx, (s, is_true_last, ends_sentence_group, comma_piece) in enumerate(raw):
        is_final = idx == n - 1
        if s[-1] in ":;":
            # Don't leave a raw colon/semicolon as the literal last character
            # sent to the model — same soft, well-understood mark the
            # em-dash pieces already get, not an ending the model may never
            # have seen a request actually stop on.
            s = s[:-1].rstrip() + ("." if is_final else ",")
        elif s[-1] not in ".!?,;:":
            # continuation intonation mid-line, sentence-final on the true last piece
            s += "," if not is_final else "."
        if is_final:
            gap = 0.0
        elif comma_piece:
            gap = COMMA_PAUSE_SEC     # mid-sentence comma break
        elif ends_sentence_group and not is_true_last:
            gap = PAUSE_SEC          # ends an em-dash-separated part
        else:
            gap = SENTENCE_PAUSE_SEC  # ordinary sentence boundary
        out.append((s, gap))
    return out


def _synthesize_stream(req: TTSRequest, client: str = "unspecified"):
    """Streaming variant of _synthesize for /tts_stream/: returns a generator of
    WAV bytes (header with unknown length, then each sentence's PCM + its gap as
    soon as it is generated). Validation runs here, eagerly, so a bad request
    still gets a proper HTTP error. The finished line is saved to history, and
    the request shows up in (and can be cancelled from) the activity list like
    any other. The ticket is re-bound on every chunk because the framework may
    run successive chunks on different worker threads."""
    text, mnt, temp, emo_id, energy, strength = _prepare_synth(req)
    clauses = _split_clauses(text)

    def gen():
        ticket = _ticket_open(client, req.text)
        pieces = []
        sr = 0
        done = False
        fx = None
        try:
            for p, gap_sec in clauses:
                _tl.ticket = ticket
                try:
                    seg_mnt = min(mnt, max(96, len(p) * 3 + 64))
                    o = _synth_no_stuck_words(p, req, seg_mnt, temp, emo_id, energy, strength)
                except HTTPException as exc:   # cancelled / failed mid-stream: end the stream
                    print(f"[stream] stopped: {exc.detail}", flush=True)
                    return
                finally:
                    _tl.ticket = None
                sr = o["sample_rate"]
                with wave.open(io.BytesIO(o["wav"]), "rb") as h:
                    pcm = h.readframes(h.getnframes())
                if not pieces:
                    # sizes unknown up front, so 0xFFFFFFFF in both length fields
                    yield (b"RIFF" + (0xFFFFFFFF).to_bytes(4, "little") + b"WAVEfmt "
                           + (16).to_bytes(4, "little") + (1).to_bytes(2, "little")
                           + (1).to_bytes(2, "little") + sr.to_bytes(4, "little")
                           + (sr * 2).to_bytes(4, "little") + (2).to_bytes(2, "little")
                           + (16).to_bytes(2, "little") + b"data"
                           + (0xFFFFFFFF).to_bytes(4, "little"))
                if fx is None and len(pieces) == 0:
                    fx = _fx_for(req.speaker_id, sr)      # the voice's own effect, if one is set
                gap_n = int(sr * gap_sec) if gap_sec > 0 else 0
                if fx is not None:
                    # effect runs over the sentence plus its trailing gap, so reverb can ring into it
                    x = np.concatenate([_pcm16_to_f32(pcm), np.zeros(gap_n, dtype=np.float32)])
                    chunk = _f32_to_pcm16(fx.process(x))
                    pieces.append(chunk)
                    yield chunk
                else:
                    pieces.append(pcm)
                    yield pcm
                    if gap_n:
                        silence = bytes(gap_n * 2)
                        pieces.append(silence)
                        yield silence
            if fx is not None:
                tail = fx.flush()                            # whatever is still ringing (reverb)
                if len(tail):
                    chunk = _f32_to_pcm16(tail)
                    pieces.append(chunk)
                    yield chunk
            done = True
        finally:
            _ticket_close(ticket)
            if done and pieces:
                try:
                    raw = b"".join(pieces)
                    buf = io.BytesIO()
                    with wave.open(buf, "wb") as h:
                        h.setnchannels(1)
                        h.setsampwidth(2)
                        h.setframerate(sr)
                        h.writeframes(raw)
                    _history_add({"wav": buf.getvalue(), "duration_sec": len(raw) / 2 / sr},
                                 req, client)
                except Exception as exc:  # noqa: BLE001 - history must never break a request
                    print(f"[history] not saved ({exc})", flush=True)
    return gen()


def _synth_pauses(text: str, req: TTSRequest, mnt: int, temp: float,
                  emo_id: int, energy: float, strength: Optional[float]) -> dict:
    """Text is split into independently-synthesized clauses (see
    _split_clauses) and the pieces are joined with real silence. Splitting
    isn't just about honoring written pause marks anymore — short
    independent generations are inherently less prone to the mid-generation
    cutoffs/drift that long single generations are, so this is now also the
    main stability mechanism for long lines."""
    clauses = _split_clauses(text)
    outs = []
    for p, _gap in clauses:
        # loop guard, part 1: budget tokens to the piece length (~14 tok/audio-
        # sec measured). When the model misses end-of-speech and repeats the
        # sentence, this stops it within seconds instead of at the service-wide
        # cap (1024 tokens = ~75s of "You feel so good. You feel so good. ...").
        seg_mnt = min(mnt, max(96, len(p) * 3 + 64))
        outs.append(_synth_no_stuck_words(p, req, seg_mnt, temp, emo_id, energy, strength))
    if len(outs) == 1:
        return outs[0]
    sr = outs[0]["sample_rate"]
    pieces = []
    for i, o in enumerate(outs):
        with wave.open(io.BytesIO(o["wav"]), "rb") as h:
            pieces.append(np.frombuffer(h.readframes(h.getnframes()), dtype=np.int16))
        gap_sec = clauses[i][1]
        if gap_sec > 0:
            pieces.append(np.zeros(int(sr * gap_sec), dtype=np.int16))
    full = np.concatenate(pieces)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as h:
        h.setnchannels(1)
        h.setsampwidth(2)
        h.setframerate(sr)
        h.writeframes(full.tobytes())
    return {"wav": buf.getvalue(), "sample_rate": sr,
            "duration_sec": float(len(full)) / sr,
            "normalized_text": " -- ".join(o["normalized_text"] for o in outs),
            "timing": " | ".join(o.get("timing", "") for o in outs)}


# The loop guard also runs when a clause comes out this many times longer
# than its text should take, not only when it hits the token cap. A clean
# double read ("It waits. It waits.") ends on its own well inside the cap, so
# the cap alone never catches it. Duration is free to check; Whisper only
# runs on clauses that trip this.
LOOP_GUARD_REL_MULT = float(os.environ.get("MOSS_LOOP_GUARD_REL_MULT", "1.6"))


def _loop_guard_suspect(text: str, dur_sec: float, capped: bool) -> bool:
    """Is this clause worth the Whisper repeat check?"""
    return capped or dur_sec > _expected_clause_dur(text) * LOOP_GUARD_REL_MULT


# A suspect clause where Whisper hears fewer than this share of the text's
# words is treated as a failed generation (silence, hiss or gibberish instead
# of the line) and regenerated, rather than passed on.
LOOP_GUARD_MIN_HEARD = float(os.environ.get("MOSS_LOOP_GUARD_MIN_HEARD", "0.5"))
# How many times a failed clause is regenerated before giving up.
FAILED_CLAUSE_RETRIES = int(os.environ.get("MOSS_FAILED_CLAUSE_RETRIES", "2"))


def _norm_word(w: str) -> str:
    return re.sub(r"[^a-z']", "", w.lower())


def _first_read_end(heard: list[str], target: list[str]) -> Optional[int]:
    """Index in `heard` of the last word of the FIRST complete reading of
    `target`, found by matching the text's final few words. Only accepted if
    that match sits roughly where one reading should end (not suspiciously
    early, e.g. a phrase that also appears mid-sentence) and more speech
    follows it (otherwise there's nothing to cut). None if no clean match."""
    n = len(target)
    for k in (3, 2):
        if n < k + 1:
            continue
        tail = target[-k:]
        for i in range(len(heard) - k + 1):
            if heard[i:i + k] == tail:
                end = i + k - 1
                if end + 1 >= int(n * 0.6) and end < len(heard) - 1:
                    return end
                break  # first occurrence decides; don't hunt for later ones
    return None


def _loop_guard_trim(audio, text: str, sr: int, capped: bool):
    """Loop guard, part 2. Returns (audio, failed).

    The model sometimes misses end-of-speech and repeats the line, in full
    ("You feel so good. You feel so good.") or just its tail ("...until the
    work is complete? ...is complete?"); or it produces silence/hiss instead
    of the words. A runaway runs into the token budget (`capped`); a clean
    repeat stops on its own but runs past the text's expected length. Only
    then is the (CPU-expensive) whisper pass worth it. Honest lines - even
    slow, breathy, over-driven ones - stop on their own and never pay it.

    With the transcript:
      - far fewer words than the text  -> failed=True (caller regenerates)
      - the text's ending heard, then more speech -> cut after the first
        reading's ending (catches whole and partial repeats)
      - no clean ending match, but the text heard well over once -> cut
        after the expected word count (the original fallback)
    """
    dur = float(audio.numel()) / sr
    if not _loop_guard_suspect(text, dur, capped):
        return audio, False
    try:
        target = [_norm_word(w) for w in re.findall(r"[A-Za-z']+", text)]
        target = [w for w in target if w]
        target_n = len(target)
        if target_n == 0:
            return audio, False
        wav = audio_to_wav_bytes(audio, sr)
        segs, _ = _get_whisper().transcribe(io.BytesIO(wav), beam_size=5,
                                            word_timestamps=True)
        words = [w for s in segs for w in (s.words or [])]
        heard = [_norm_word(w.word) for w in words]

        if len(words) < max(1, int(target_n * LOOP_GUARD_MIN_HEARD)):
            print(f"[loop-guard] heard {len(words)} of {target_n} words in "
                  f"{dur:.1f}s; generation failed", flush=True)
            return audio, True

        end = _first_read_end(heard, target)
        if end is not None:
            how = "repeated ending"
        elif len(words) > int(target_n * 1.5) + 2:
            end, how = target_n - 1, "heard well over once"
        else:
            return audio, False
        cut = min(dur, float(words[end].end) + 0.35)
        n_samp = min(audio.numel(), int(cut * sr))
        print(f"[loop-guard] {how}: heard {len(words)} words for a {target_n}-word "
              f"line; trimmed {dur:.1f}s -> {cut:.1f}s", flush=True)
        return (audio[..., :n_samp] if audio.dim() > 1 else audio[:n_samp]), False
    except Exception as exc:  # noqa: BLE001 - guard must never break synthesis
        print(f"[loop-guard] skipped ({exc})", flush=True)
        return audio, False


def _synth_once(text: str, req: TTSRequest, mnt: int, temp: float,
                emo_id: int, energy: float, strength: Optional[float] = None) -> dict:
    text = text.replace("—", ", ")   # pause marks are for _synth_pauses, never the model
    with _lock:
        _check_cancelled()   # queued, between sentences, or before a retry
        _blend_sync_for_speaker(req.speaker_id)
        # emotion over-drive: the model's conditioner takes a strength multiplier
        # (config.style_feature_scale); it's read per forward pass, so set it per
        # request and always reset — 1.0 when the request doesn't ask.
        strength = 1.0 if strength is None else max(0.0, min(2.3, float(strength)))
        inner = getattr(getattr(_runner, "model", None), "model", None)
        if inner is not None and hasattr(inner, "style_feature_scale"):
            inner.style_feature_scale = strength
        try:
            result = _runner.synthesize(
                text=text,
                language=req.language,
                speaker_id=req.speaker_id,
                max_new_tokens=mnt,
                audio_temperature=temp,
                audio_top_p=req.audio_top_p,
                audio_top_k=req.audio_top_k,
                audio_repetition_penalty=req.audio_repetition_penalty,
                style_emotion_id=emo_id,
                style_energy=energy,
            )
            audio = result["audio"]
            _gen_tok = int(result.get("generated_tokens", 0) or 0)
            audio, gen_failed = _loop_guard_trim(audio, text, _runner.sample_rate,
                                                 _gen_tok >= mnt - 2)
            if req.trim_silence:
                audio = shorten_silences(
                    audio, _runner.sample_rate, max_gap_ms=req.silence_max_gap_ms
                )
            duration = float(audio.numel() / _runner.sample_rate)
            wav = audio_to_wav_bytes(audio, _runner.sample_rate)
            timing = (f"gen={result.get('generate_sec', 0):.2f} "
                      f"dec={result.get('decode_sec', 0):.2f} "
                      f"audio={result.get('audio_sec', 0):.2f} "
                      f"tok={result.get('generated_tokens', 0)}")
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            # Recover the GPU so one bad request doesn't poison later ones, and
            # return a clean, retryable error instead of an opaque 500.
            _recover_gpu()
            msg = str(exc)
            if _is_oom(msg):
                raise HTTPException(
                    status_code=503,
                    detail="GPU out of memory (line too long for free VRAM). "
                    "Retry, shorten the line, or lower max_new_tokens.",
                ) from exc
            raise HTTPException(status_code=500, detail=f"synthesis failed: {msg}") from exc
    return {"wav": wav, "sample_rate": _runner.sample_rate, "duration_sec": duration,
            "normalized_text": text, "timing": timing, "failed": gen_failed}


def _synth_verified(text: str, req: TTSRequest, mnt: int, temp: float,
                    emo_id: int, energy: float, strength: Optional[float] = None) -> dict:
    """Synthesize with the verified-pronunciation loop: substitute cached
    respelling winners, Whisper-verify risky words with no verdict yet, and run
    a bounded respelling search on failures (see the pron-cache section)."""
    fix = req.fix_pronunciation and _RESPELL_AVAILABLE
    risky = _risky_words(text) if fix else []
    spelling: dict[str, str] = {}          # word -> spelling currently in the text
    for w in risky:
        e = _pron.get(w, {}).get(str(req.speaker_id))
        spelling[w] = e["use"] if (e and e.get("use")) else w
        if spelling[w] != w:
            text = _sub_word(text, w, spelling[w])
    out = _synth_once(text, req, mnt, temp, emo_id, energy, strength)
    if not risky:
        return out
    # ALWAYS verify lines with risky words — a respelling verified in one
    # sentence can still come out slurred in another (measured, not theory).
    try:
        heard = _transcribed_words(out["wav"])
    except Exception as exc:  # noqa: BLE001 — no whisper => serve unverified
        print(f"[pron] verify unavailable ({exc}); serving unverified", flush=True)
        return out
    solved: dict[str, str] = {}
    with _pron_lock:
        failed = [w for w in risky if not _heard_ok(w, heard)]
        for w in risky:
            if w not in failed and spelling[w] == w:
                _pron_entry(w, req.speaker_id).setdefault("use", "")  # plain verified
        if not failed:
            _pron_save()
            return out
        cands: dict[str, list[str]] = {}
        for w in failed:
            tried = _pron_entry(w, req.speaker_id).setdefault("tried", [])
            pool = _pron_candidates(w)
            if spelling[w] != w:
                pool = [w] + pool  # cached variant flopped here — plain gets a shot too
            cands[w] = [c for c in pool if c not in tried and c != spelling[w]]
        best, best_score = out, sum(_heard_ratio(w, heard) for w in failed)
        cur = text
        for _ in range(PRON_TRIES_PER_LINE):
            attempt = {w: cands[w].pop(0) for w in failed if w not in solved and cands[w]}
            if not attempt:
                break
            for w, c in attempt.items():
                cur = _sub_word(cur, spelling[w], c)
                spelling[w] = c
            res = _synth_once(cur, req, mnt, temp, emo_id, energy, strength)
            h2 = _transcribed_words(res["wav"])
            score = 0.0
            for w in failed:
                r = _heard_ratio(w, h2)
                score += r
                if w in attempt:
                    _pron_entry(w, req.speaker_id)["tried"].append(attempt[w])
                    if _heard_ok(w, h2):
                        solved[w] = attempt[w]
                        _pron_entry(w, req.speaker_id)["use"] = attempt[w]
            if score >= best_score:
                best, best_score = res, score
            if len(solved) == len(failed):
                best = res
                break
        _pron_save()
        print(f"[pron] spk={req.speaker_id} failed={failed} solved={solved}", flush=True)
    return best


def _expected_word_dur(word: str) -> float:
    """Rough expected spoken duration for a word — same style of simple
    linear heuristic already used elsewhere in this file for the
    per-segment token budget (len(text)*3+64); not meant to be precise,
    just a sanity ceiling to compare actual duration against."""
    n = len(re.sub(r"[^a-zA-Z']", "", word))
    return 0.15 + n * 0.05


# Coarse, Whisper-independent companion to the per-word check below. Added
# after a real observed glitch ("HOSE?!" droning for ~2.9s) slipped past
# the per-word check entirely — Whisper apparently didn't attribute the
# drone to one long word, likely chopping the repetitive audio into several
# short garbled fragments instead, each individually too short to trip a
# per-word threshold even though the whole clause was obviously too long
# for two words. This check doesn't need Whisper to correctly hear the
# glitch at all — it only needs the clip's actual duration (already
# computed by the pipeline) and the input text, so it catches exactly the
# case that got through, and works even if Whisper itself is unavailable.
STUCK_CLAUSE_REL_MULT = float(os.environ.get("MOSS_STUCK_CLAUSE_REL_MULT", "2.0"))
STUCK_CLAUSE_BASE_SEC = 0.3  # fixed leading/trailing-pause allowance so short clauses don't false-positive on ordinary padding


def _expected_clause_dur(text: str) -> float:
    words = re.findall(r"[a-zA-Z']+", text)
    return STUCK_CLAUSE_BASE_SEC + sum(_expected_word_dur(w) for w in words)


def _find_stuck_word(wav_bytes: bytes, text: str, actual_dur: float) -> Optional[str]:
    """Returns a description of the problem if this clause's audio looks
    like the model got stuck in a low-level generation loop — the
    signature is audio running much longer than its text should ever
    take. Two complementary checks:
      1. Whole-clause duration vs. a rough expected total for the text
         (_expected_clause_dur) — no Whisper needed, so it's immune to
         Whisper mis-hearing a droning glitch as short garbled fragments.
      2. Per-word duration via Whisper, when available — catches a
         proportionally smaller anomaly inside an otherwise-normal longer
         clause, and names the specific word for a clearer log.
    Distinct from _loop_guard_trim's whole-line repetition (that requires
    hitting the token cap; this can happen well within budget) and from
    mispronunciation (the word sounds fine, it's just stretched out).
    Returns None if nothing looks wrong — this must never block synthesis,
    so a broken/unavailable Whisper degrades to just the coarse check
    rather than failing closed."""
    expected_total = _expected_clause_dur(text)
    clause_anomaly = actual_dur > expected_total * STUCK_CLAUSE_REL_MULT

    culprit_word = None
    try:
        segs, _ = _get_whisper().transcribe(io.BytesIO(wav_bytes), beam_size=5,
                                            word_timestamps=True)
        for s in segs:
            for w in (s.words or []):
                dur = float(w.end) - float(w.start)
                expected_w = _expected_word_dur(w.word)
                if dur > STUCK_WORD_ABS_CEIL_SEC or dur > expected_w * STUCK_WORD_REL_MULT:
                    culprit_word = w.word.strip()
                    break
            if culprit_word:
                break
    except Exception as exc:  # noqa: BLE001 — verification must never break synthesis
        print(f"[stuck-word] whisper check skipped ({exc})", flush=True)

    if culprit_word:
        return f"'{culprit_word}'"
    if clause_anomaly:
        return f"(clause ran {actual_dur:.1f}s, expected ~{expected_total:.1f}s)"
    return None


def _synth_no_stuck_words(text: str, req: TTSRequest, mnt: int, temp: float,
                          emo_id: int, energy: float, strength: Optional[float]) -> dict:
    """Wraps _synth_verified with a bounded plain-regenerate retry when this
    clause's audio looks stuck (see _find_stuck_word) — a distinct glitch
    from both wrong-word mispronunciation (fixed by respelling) and
    whole-line repetition (fixed by _loop_guard_trim's trim). Regenerating
    from the same text is the right fix here, not a respelling search:
    sampling is stochastic, so the same text is unlikely to get stuck the
    same way twice. Ships the flawed audio, with a log line, if it's still
    stuck after exhausting retries — never blocks the response
    indefinitely."""
    out = _synth_verified(text, req, mnt, temp, emo_id, energy, strength)
    # Failed generation (loop guard heard silence/hiss/gibberish instead of the
    # words): regenerate. Always on - independent of MOSS_FIX_STUCK_WORDS.
    for attempt in range(1, FAILED_CLAUSE_RETRIES + 1):
        if not out.get("failed"):
            break
        print(f"[loop-guard] regenerating failed clause "
              f"({attempt}/{FAILED_CLAUSE_RETRIES}): {text[:60]!r}", flush=True)
        out = _synth_verified(text, req, mnt, temp, emo_id, energy, strength)
    else:
        if out.get("failed"):
            print(f"[loop-guard] clause still failed after {FAILED_CLAUSE_RETRIES} "
                  f"retries; shipping as-is: {text[:60]!r}", flush=True)
    if not req.fix_stuck_words:
        return out
    for attempt in range(1, STUCK_WORD_RETRIES + 1):
        stuck = _find_stuck_word(out["wav"], text, out["duration_sec"])
        if stuck is None:
            return out
        if attempt == STUCK_WORD_RETRIES:
            print(f"[stuck-word] {stuck} still anomalous after {attempt} "
                  f"retries; shipping as-is", flush=True)
            return out
        print(f"[stuck-word] {stuck} ran anomalously long, retrying "
              f"({attempt}/{STUCK_WORD_RETRIES})", flush=True)
        out = _synth_verified(text, req, mnt, temp, emo_id, energy, strength)
    return out


def _is_oom(msg: str) -> bool:
    m = msg.lower()
    return "out of memory" in m or "failed to allocate" in m or "bfc_arena" in m


def _recover_gpu() -> None:
    try:
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


@app.get("/health")
def health():
    return {
        "status": "ok" if _runner is not None else "loading",
        "model_loaded": _runner is not None,
        "device": str(_runner.device) if _runner else None,
        "sample_rate": _runner.sample_rate if _runner else None,
        "num_speakers": len(SPEAKERS),
        "preset": _active_preset,
        "presets": list(PRESETS),
        "max_new_tokens": _default_max_new_tokens,
        "audio_temperature": _default_audio_temperature,
        "reloading": _reload_state["busy"],
        "reload_target": _reload_state["target"],
        "reload_error": _reload_state["error"],
        "activity": _activity(),
        "history": HISTORY_MAX > 0,
    }


@app.get("/gpu")
def gpu():
    if not torch.cuda.is_available():
        return {"cuda": False}
    free, total = torch.cuda.mem_get_info()
    return {
        "cuda": True,
        "name": torch.cuda.get_device_name(0),
        "total_mb": round(total / 1024**2),
        "free_mb": round(free / 1024**2),
        "torch_reserved_mb": round(torch.cuda.memory_reserved() / 1024**2),
        "torch_allocated_mb": round(torch.cuda.memory_allocated() / 1024**2),
    }


@app.get("/speakers")
def speakers():
    return SPEAKERS


PRESET_INFO = {
    "quality":  "~16 GB VRAM — fastest, GPU vocoder (NVIDIA only)",
    "balanced": "~10-12 GB — fast, vocoder on CPU",
    "low_vram": "~8 GB — vocoder on CPU, no CUDA-graph",
    "potato":   "~6 GB — as low_vram, shortest line cap",
}


@app.get("/preset")
def preset_get():
    """Current preset + the menu for the web-page dropdown."""
    return {
        "active": _active_preset,
        "reloading": _reload_state["busy"],
        "target": _reload_state["target"],
        "error": _reload_state["error"],
        "options": [{"id": k, "label": f"{k} — {PRESET_INFO.get(k, '')}"} for k in PRESETS],
    }


@app.post("/preset")
def preset_set(name: str = Query(..., description="quality|balanced|low_vram|potato")):
    """Hot-swap the performance/VRAM preset: reloads the model in the background
    (~30-90s). The page polls /health until the new preset is live."""
    if name not in PRESETS:
        raise HTTPException(status_code=400, detail=f"unknown preset '{name}'; pick one of {list(PRESETS)}")
    if name == _active_preset and not _reload_state["busy"]:
        return {"status": "unchanged", "active": _active_preset}
    with _reload_lock:
        if _reload_state["busy"]:
            raise HTTPException(status_code=409, detail=f"already reloading to '{_reload_state['target']}'")
        _reload_state["busy"] = True
        _reload_state["target"] = name
        _reload_state["error"] = None
    threading.Thread(target=_reload_worker, args=(name,), daemon=True).start()
    return {"status": "reloading", "target": name, "from": _active_preset}


# ----------------------------------------------------------------------------
# Sound effects (v2.6+): the game's AI writes [sfx: description] stage directions;
# the game asks this endpoint for a matching sound, served as MP3 (Godot plays
# AudioStreamMP3 natively — no transcode dependency). Sources, in order:
#   1. disk cache (each distinct query is fetched exactly once)
#   2. Freesound.org API — only if a free API key is present
#      (https://freesound.org/apiv2/apply -> freesound_key.txt or FREESOUND_API_KEY)
#   3. Freesound.org public search page — NO key needed; the same preview MP3s
#      the site plays in your browser, parsed straight off the search results
#   4. BBC Sound Effects archive (~33k pro recordings, public API, NO key needed)
#      https://sound-effects.bbcrewind.co.uk — RemArc licence, personal use
#   5. fuzzy match against the shipped/pre-seeded sound pack (offline fallback)
# So a plain drag-and-drop install gets dynamic, on-demand sounds with zero setup.
# ----------------------------------------------------------------------------
import hashlib
import urllib.parse
import urllib.request as _urlreq

SFX_CACHE_DIR = PROJECT_DIR / "sfx_cache"
SFX_KEY_FILE = PROJECT_DIR / "freesound_key.txt"
SFX_MAX_DURATION = 12.0   # seconds; filter applied at search time
# On by default (existing behavior). Set MOSS_SFX_ALLOW_NETWORK=0 to disable
# every outgoing call this feature makes (Freesound API/scrape, BBC archive)
# for privacy — /sfx then only ever serves from what's already cached on
# disk (fuzzy-matched against manifest.json), never reaching out itself.
SFX_ALLOW_NETWORK = os.environ.get("MOSS_SFX_ALLOW_NETWORK", "1") != "0"


def _freesound_key() -> str:
    key = os.environ.get("FREESOUND_API_KEY", "").strip()
    if not key and SFX_KEY_FILE.exists():
        key = SFX_KEY_FILE.read_text(encoding="utf-8").strip()
    return key


def _sfx_cache_path(query: str) -> Path:
    digest = hashlib.sha1(query.strip().lower().encode()).hexdigest()[:20]
    return SFX_CACHE_DIR / f"{digest}.mp3"


# manifest maps cache filename -> original query text, so no-key installs can
# fuzzy-match new descriptions onto the shipped, pre-seeded sound pack
SFX_MANIFEST = SFX_CACHE_DIR / "manifest.json"
_sfx_manifest_lock = threading.Lock()


def _sfx_manifest_load() -> dict:
    try:
        return json.loads(SFX_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _sfx_manifest_add(query: str, filename: str) -> None:
    with _sfx_manifest_lock:
        m = _sfx_manifest_load()
        m[filename] = query.strip().lower()
        SFX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        SFX_MANIFEST.write_text(json.dumps(m, indent=1, sort_keys=True), encoding="utf-8")


def _sfx_fuzzy_match(query: str) -> Optional[Path]:
    """Closest already-cached sound for `query` (token overlap + string ratio).
    A near-miss sound beats silence — thresholds are deliberately forgiving."""
    m = _sfx_manifest_load()
    if not m:
        return None
    q = query.strip().lower()
    q_tokens = set(re.findall(r"[a-z]+", q))
    best_name, best_score = None, 0.0
    for filename, text in m.items():
        t_tokens = set(re.findall(r"[a-z]+", text))
        jacc = len(q_tokens & t_tokens) / max(len(q_tokens | t_tokens), 1)
        ratio = difflib.SequenceMatcher(None, q, text).ratio()
        score = max(jacc, ratio)
        if score > best_score:
            best_name, best_score = filename, score
    if best_name and best_score >= 0.3:
        p = SFX_CACHE_DIR / best_name
        if p.exists():
            return p
    return None


def _is_plausible_mp3(data: bytes) -> bool:
    """Loose validity check on fetched sfx bytes — catches truncated/corrupt
    downloads (a slow connection or network hiccup can return a partial read
    without raising an exception) that would otherwise get cached and served
    as if they were valid audio. Not a real MP3 parser, just enough to
    reject obvious garbage: too small to be a real one-shot sound effect, or
    missing both a legitimate ID3 tag and a valid MPEG frame sync at the
    very start of the data."""
    if len(data) < 1024:
        return False
    if data[:3] == b"ID3":
        return True
    return len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0


_FS_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
          "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")


# the site's full-text search ANDs every word, so flavour words ("comedic",
# "angrily") zero out results — the retry ladder strips them progressively
_FS_STOPWORDS = {"a", "an", "the", "of", "in", "on", "at", "with", "and",
                 "sound", "sounds", "sfx", "effect", "noise", "comedic",
                 "cartoonish", "dramatic"}


def _fs_web_search(query: str) -> list:
    """One search-page fetch -> scored [(score, preview_url), ...]."""
    url = ("https://freesound.org/search/?q=" + urllib.parse.quote_plus(query)
           + "&f=" + urllib.parse.quote(f"duration:[0.3 TO {SFX_MAX_DURATION:g}]"))
    req = _urlreq.Request(url, headers={"User-Agent": _FS_UA})
    with _urlreq.urlopen(req, timeout=15) as r:
        html = r.read().decode("utf-8", "replace")
    q_tokens = set(re.findall(r"[a-z]+", query.lower()))
    cands = []
    for m in re.finditer(r'class="bw-player"', html):
        tag = html[m.start():html.find(">", m.start())]
        mp3 = re.search(r'data-mp3="([^"]+)"', tag)
        if not mp3:
            continue
        title = re.search(r'data-title="([^"]*)"', tag)
        dls = re.search(r'data-num-downloads="(\d+)"', tag)
        t_tokens = set(re.findall(r"[a-z]+", (title.group(1) if title else "").lower()))
        jacc = len(q_tokens & t_tokens) / max(len(q_tokens | t_tokens), 1)
        # title relevance, nudged by popularity, minus a small page-order decay
        score = (jacc + min(int(dls.group(1)) if dls else 0, 5000) / 50000.0
                 - len(cands) * 0.01)
        cands.append((score, mp3.group(1)))
    cands.sort(key=lambda c: -c[0])
    return cands


def _sfx_fetch_web(query: str) -> bytes:
    """No-key Freesound: parse the public search-results page and download the
    best hit's preview MP3 (the same file the in-browser player streams)."""
    words = re.findall(r"[A-Za-z]+", query.lower())
    core = [w for w in words if w not in _FS_STOPWORDS and not
            (w.endswith("ly") and len(w) > 4)]
    variants = [query]
    if core and core != words:
        variants.append(" ".join(core))
    if len(core) > 2:
        variants.append(" ".join(core[:2]))
    if core and len(core[0]) >= 3:
        variants.append(core[0])
    cands = []
    for v in dict.fromkeys(variants):  # dedupe, keep order
        cands = _fs_web_search(v)
        if cands:
            break
    if not cands:
        raise HTTPException(status_code=404, detail=f"no freesound match for {query!r}")
    lq = cands[0][1]
    for u in (lq.replace("-lq.mp3", "-hq.mp3"), lq):  # HQ preview first
        try:
            with _urlreq.urlopen(_urlreq.Request(u, headers={"User-Agent": _FS_UA}),
                                 timeout=20) as r:
                data = r.read()
            if _is_plausible_mp3(data):
                return data
        except Exception:  # noqa: BLE001
            continue
    raise HTTPException(status_code=502, detail=f"freesound preview download failed for {query!r}")


_BBC_SEARCH = "https://sound-effects-api.bbcrewind.co.uk/api/sfx/search"
_BBC_MEDIA = "https://sound-effects-media.bbcrewind.co.uk/mp3/{sid}.mp3"


def _sfx_fetch_bbc(query: str) -> bytes:
    """Search the BBC Sound Effects archive (no key needed), download best MP3."""
    body = json.dumps({"criteria": {"from": 0, "size": 30, "query": query}}).encode()
    req = _urlreq.Request(_BBC_SEARCH, data=body,
                          headers={"Content-Type": "application/json",
                                   "User-Agent": "ppp-ttsmod/2.7"})
    with _urlreq.urlopen(req, timeout=15) as r:
        results = json.loads(r.read()).get("results", [])
    q_tokens = set(re.findall(r"[a-z]+", query.lower()))

    def _pick(max_dur: float):
        best_id, best_key = None, None
        for s in results:
            dur = float(s.get("duration") or 0) / 1000.0  # BBC reports milliseconds
            if not 0.3 <= dur <= max_dur:
                continue  # skip multi-minute ambience beds — this is a one-shot cue
            text = (s.get("description") or "") + " " + " ".join(s.get("tags") or [])
            t_tokens = set(re.findall(r"[a-z]+", text.lower()))
            jacc = len(q_tokens & t_tokens) / max(len(q_tokens | t_tokens), 1)
            # relevance first, then a shorter clip on ties (snappier stage cue)
            key = (-jacc, dur)
            if best_key is None or key < best_key:
                best_id, best_key = s.get("id"), key
        return best_id

    # BBC has lots of longer recordings (clocks, crowds, weather) — if nothing
    # fits the snappy one-shot window, accept a medium-length clip over silence
    best_id = _pick(SFX_MAX_DURATION) or _pick(30.0)
    if not best_id:
        raise HTTPException(status_code=404, detail=f"no BBC sound found for {query!r}")
    with _urlreq.urlopen(_BBC_MEDIA.format(sid=best_id), timeout=25) as r:
        data = r.read()
    if not _is_plausible_mp3(data):
        raise HTTPException(status_code=502, detail=f"BBC preview download failed validation for {query!r}")
    return data


def _sfx_fetch(query: str, key: str) -> bytes:
    """Search Freesound, download the best short clip's HQ preview."""
    params = urllib.parse.urlencode({
        "query": query,
        "token": key,
        "fields": "id,name,previews,duration,avg_rating,num_ratings",
        "filter": f"duration:[0.3 TO {SFX_MAX_DURATION}]",
        "sort": "score",
        "page_size": "8",
    })
    with _urlreq.urlopen(f"https://freesound.org/apiv2/search/text/?{params}", timeout=15) as r:
        results = json.loads(r.read()).get("results", [])
    if not results:
        raise HTTPException(status_code=404, detail=f"no sound found for {query!r}")
    # prefer decently-rated hits, otherwise take the top relevance match
    rated = [s for s in results if float(s.get("avg_rating") or 0) >= 3.5
             and int(s.get("num_ratings") or 0) >= 2]
    best = (rated or results)[0]
    url = best["previews"].get("preview-hq-mp3") or best["previews"].get("preview-lq-mp3")
    if not url:
        raise HTTPException(status_code=404, detail=f"no preview for {query!r}")
    with _urlreq.urlopen(url, timeout=20) as r:
        data = r.read()
    if not _is_plausible_mp3(data):
        raise HTTPException(status_code=502, detail=f"Freesound preview download failed validation for {query!r}")
    return data


@app.get("/sfx")
def sfx(query: str = Query(..., min_length=2, max_length=120)):
    """Sound effect for a short description. Cached on disk: each distinct query
    is fetched exactly once, so gameplay replays are instant and offline.
    Fetch order (skipped entirely if MOSS_SFX_ALLOW_NETWORK=0): Freesound API
    (if a key is configured) -> Freesound public search page (no key) -> BBC
    archive (no key) -> fuzzy match against whatever has already been cached
    from a previous successful fetch (there's no bundled sound library —
    with network calls disabled from a fresh install, nothing has ever been
    cached yet, so this returns 404 until/unless network access is enabled)."""
    cache = _sfx_cache_path(query)
    if cache.exists():
        return Response(cache.read_bytes(), media_type="audio/mpeg")
    data: Optional[bytes] = None
    if SFX_ALLOW_NETWORK:
        key = _freesound_key()
        if key:
            try:
                data = _sfx_fetch(query, key)
            except Exception:  # noqa: BLE001 — no match / API down: try keyless paths
                data = None
        if data is None:
            try:
                data = _sfx_fetch_web(query)
            except Exception:  # noqa: BLE001 — no hit on the site: try the BBC archive
                data = None
        if data is None:
            try:
                data = _sfx_fetch_bbc(query)
            except Exception:  # noqa: BLE001 — offline or no hit: fall back to cache
                data = None
    if data is None:
        near = _sfx_fuzzy_match(query)
        if near is not None:
            return Response(near.read_bytes(), media_type="audio/mpeg")
        raise HTTPException(status_code=404,
                            detail=f"no sound found for {query!r} (Freesound/BBC/cache)")
    SFX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(data)
    _sfx_manifest_add(query, cache.name)
    return Response(data, media_type="audio/mpeg")


@app.get("/normalize")
def normalize_preview(text: str = Query(..., min_length=1), expand_numbers: bool = True,
                      auto_respell: bool = True):
    """See exactly what text the model will receive (no synthesis). Handy for
    tuning pronunciations.txt: GET /normalize?text=I%20beg%20you..."""
    mapping, _ = _get_pronunciations()
    # run the respell pass separately so the preview can report what it did
    _, respelled = _auto_respell(_apply_respellings(text)) if auto_respell else (text, {})
    return {
        "input": text,
        "normalized": normalize_text(text, expand_numbers=expand_numbers, auto_respell=auto_respell),
        "auto_respelled": respelled,
        "pronunciation_overrides": len(mapping),
    }


def _audio_response(out: dict, fmt: str) -> Response:
    if fmt == "base64":
        return JSONResponse(
            {
                "audio_b64": base64.b64encode(out["wav"]).decode("ascii"),
                "sample_rate": out["sample_rate"],
                "duration_sec": out["duration_sec"],
                "normalized_text": out.get("normalized_text", ""),
                "format": "wav",
            }
        )
    return Response(
        content=out["wav"],
        media_type="audio/wav",
        headers={
            "X-Duration-Sec": f"{out['duration_sec']:.3f}",
            "X-Sample-Rate": str(out["sample_rate"]),
            # ASCII-safe so it's always a legal HTTP header value
            "X-Normalized-Text": out.get("normalized_text", "").encode("ascii", "replace").decode(),
            "X-Timing": out.get("timing", ""),
        },
    )


@app.post("/tts")
def tts_post(req: TTSRequest, request: Request):
    return _audio_response(_synthesize(req, _client_app(request)), req.format)


@app.get("/tts")
def tts_get(
    request: Request,
    text: str = Query(..., min_length=1),
    speaker_id: int = 31,
    language: str = "en",
    format: str = "wav",
    audio_temperature: Optional[float] = None,
    max_new_tokens: Optional[int] = None,
    trim_silence: bool = True,
    silence_max_gap_ms: int = 300,
    normalize: bool = True,
):
    req = TTSRequest(
        text=text,
        speaker_id=speaker_id,
        language=language,
        format=format,
        audio_temperature=audio_temperature,
        max_new_tokens=max_new_tokens,
        trim_silence=trim_silence,
        silence_max_gap_ms=silence_max_gap_ms,
        normalize=normalize,
    )
    return _audio_response(_synthesize(req, _client_app(request)), req.format)


# ----------------------------------------------------------------------------
# SillyTavern compatibility shim
# ----------------------------------------------------------------------------
# SillyTavern's built-in "XTTSv2" TTS provider talks to a fixed API surface
# (see https://github.com/daswer123/xtts-api-server). These three routes
# translate that contract onto the existing /tts pipeline so no separate
# server or SillyTavern extension is needed — just point the provider's
# "Provider Endpoint" field at this server's base URL.
#
# ST's XTTSTtsProvider.getVoice() matches voices by exact `name`, and
# GET /speakers already returns objects with a "name" field, so voice
# lookup works unmodified in the common case. It sends the chosen voice
# as `speaker_wav` (that field is normally a path to a reference clip
# for real XTTS; here we repurpose it to carry the speaker name/id
# instead). Some ST voice-picker UIs render/store disambiguated labels
# like "Celestia (31)" instead of the bare name, so the lookup below
# tolerates that shape too.

_TRAILING_ID_RE = re.compile(r"^(?P<base>.*?)\s*\((?P<id>\d+)\)\s*$")


def _speaker_id_from_name(name: str) -> int:
    name = name.strip()

    # Exact match against the raw "name" field (the common case).
    for s in SPEAKERS:
        if s["name"] == name:
            return s["id"]

    # "Name (id)" style label some voice-picker UIs render/store instead
    # of the bare name — trust the embedded id first (unambiguous even
    # if names collide), then fall back to matching the stripped base
    # name.
    match = _TRAILING_ID_RE.match(name)
    if match:
        embedded_id = int(match.group("id"))
        if embedded_id in SPEAKER_IDS:
            return embedded_id
        base_name = match.group("base")
        for s in SPEAKERS:
            if s["name"] == base_name:
                return s["id"]

    # Bare numeric id/string, e.g. "31".
    try:
        speaker_id = int(name)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"unknown speaker '{name}'; see GET /speakers",
        )
    if speaker_id not in SPEAKER_IDS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown speaker '{name}'; see GET /speakers",
        )
    return speaker_id


class XTTSCompatRequest(BaseModel):
    text: str
    speaker_wav: str
    language: str = "en"


@app.post("/tts_to_audio/")
def xtts_compat_generate(req: XTTSCompatRequest, request: Request):
    """POST body: {"text", "speaker_wav", "language"} -> raw audio/wav bytes."""
    tts_req = TTSRequest(
        text=req.text,
        speaker_id=_speaker_id_from_name(req.speaker_wav),
        language=req.language,
        format="wav",
    )
    out = _synthesize(tts_req, _client_app(request))
    return Response(content=out["wav"], media_type="audio/wav")


@app.get("/tts_stream/")
def xtts_compat_stream(
    request: Request,
    text: str = Query(..., min_length=1),
    speaker_wav: str = Query(...),
    language: str = "en",
):
    """GET for ST's streaming-mode URL. Streams the WAV one sentence at a time so
    playback can start after the first sentence rather than the whole paragraph.
    (/tts_to_audio/ stays one-shot.)"""
    tts_req = TTSRequest(
        text=text,
        speaker_id=_speaker_id_from_name(speaker_wav),
        language=language,
        format="wav",
    )
    return StreamingResponse(_synthesize_stream(tts_req, _client_app(request)),
                             media_type="audio/wav")


@app.post("/set_tts_settings")
def xtts_compat_settings(_payload: dict = Body(default={})):
    """ST pushes its temperature/speed/etc sliders here on every settings
    change. This mod's runner is tuned via /preset and TTSRequest fields
    instead, so we just accept and discard the ST-side sliders."""
    return {"status": "ok"}


# ----------------------------------------------------------------------------
# SillyTavern "OpenAI Compatible" provider
# ----------------------------------------------------------------------------
# Unlike the XTTS provider above, this one never calls GET /speakers at all —
# ST's OpenAICompatibleTtsProvider builds its voice list locally from a
# comma-separated text field the user types into its settings UI, and routes
# generation through ST's own backend rather than the browser directly. It
# expects a plain OpenAI /v1/audio/speech-shaped endpoint: POST
# {model, input, voice, response_format, speed} -> raw audio bytes.
#
# `response_format` is accepted but ignored — we always return WAV with a
# matching Content-Type header, which browsers happily decode regardless of
# what format the client originally asked for (playback goes through a Blob,
# not a file extension). `speed` is likewise accepted and ignored, since this
# mod's runner has no equivalent knob; wire it up to TTSRequest if/when one
# is added.
class OpenAISpeechRequest(BaseModel):
    model: str = "tts-1"
    input: str
    voice: str
    response_format: str = "mp3"
    speed: float = 1.0


@app.post("/v1/audio/speech")
def openai_compat_generate(req: OpenAISpeechRequest, request: Request):
    tts_req = TTSRequest(
        text=req.input,
        speaker_id=_speaker_id_from_name(req.voice),
        format="wav",
    )
    out = _synthesize(tts_req, _client_app(request))
    return Response(content=out["wav"], media_type="audio/wav")


# ----------------------------------------------------------------------------
# SillyTavern "TTSMOD Clip Downloader" extension — server-side cache
# ----------------------------------------------------------------------------
# Generates a clip the same way as everything else, but additionally writes
# it to disk at st_cache/{chat}/{character}-{message_num}.wav, overwriting
# any previous take for that exact slot (regenerating means the old attempt
# wasn't wanted — no point accumulating every retry). Files are then served
# directly as static downloads from /st_cache/, so the extension can offer a
# plain clickable link rather than needing blob/IndexedDB machinery.
ST_CACHE_DIR = Path(__file__).resolve().parent / "st_cache"
ST_CACHE_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/st_cache", StaticFiles(directory=str(ST_CACHE_DIR)), name="st_cache")
app.mount("/history/files", StaticFiles(directory=str(HISTORY_DIR)), name="history_files")


@app.post("/cancel")
def cancel(request: Request, scope: str = Query("mine", description="mine|all")):
    """Cancel generations: scope=mine cancels the caller's own client's
    requests (by X-Client-App), scope=all cancels everyone's. Running requests
    stop after their current sentence; queued ones never start."""
    global _cancel_all_gen
    if scope not in ("mine", "all"):
        raise HTTPException(status_code=400, detail="scope must be 'mine' or 'all'")
    client = _client_app(request)
    with _cancel_lock:
        if scope == "all":
            _cancel_all_gen += 1
            hit = len(_active_tickets)
        else:
            _cancel_client_gen[client] = _cancel_client_gen.get(client, 0) + 1
            hit = sum(1 for t in _active_tickets if t.client == client)
    print(f"[cancel] scope={scope} by client={client}: {hit} request(s) cancelled", flush=True)
    return {"scope": scope, "client": client, "cancelled": hit}


@app.get("/history")
def history_list(limit: int = Query(200, ge=1, le=5000),
                 voice: Optional[str] = None, client: Optional[str] = None):
    """Saved generations, newest first. Optional filters: voice, client."""
    items = []
    with _history_lock:
        metas = sorted(HISTORY_DIR.glob("*.json"), reverse=True)
    for m in metas:
        try:
            meta = json.loads(m.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if voice and meta.get("voice") != voice:
            continue
        if client and meta.get("client") != client:
            continue
        meta["url"] = f"/history/files/{meta['id']}.wav"
        items.append(meta)
        if len(items) >= limit:
            break
    return {"enabled": HISTORY_MAX > 0, "max": HISTORY_MAX, "items": items}


@app.delete("/history/{hid}")
def history_delete(hid: str):
    if not _HISTORY_ID_RE.match(hid):
        raise HTTPException(status_code=400, detail="bad id")
    with _history_lock:
        wav = HISTORY_DIR / f"{hid}.wav"
        if not wav.exists():
            raise HTTPException(status_code=404, detail="not found")
        wav.unlink(missing_ok=True)
        wav.with_suffix(".json").unlink(missing_ok=True)
    return {"deleted": hid}


@app.delete("/history")
def history_clear():
    n = 0
    with _history_lock:
        for f in HISTORY_DIR.glob("*.wav"):
            f.unlink(missing_ok=True)
            f.with_suffix(".json").unlink(missing_ok=True)
            n += 1
    return {"deleted": n}


def _safe_path_component(s: str, fallback: str = "untitled", max_len: int = 80) -> str:
    """Sanitize a client-supplied string for use as a directory/file name
    component. Strips anything that isn't alphanumeric/space/hyphen/
    underscore (blocks path traversal regardless of what's sent), collapses
    whitespace to underscores, and falls back to a safe default if nothing
    usable survives."""
    s = re.sub(r"[^a-zA-Z0-9 _-]", "", s or "").strip()
    s = re.sub(r"\s+", "_", s)
    return s[:max_len] or fallback


class STCacheGenerateRequest(BaseModel):
    text: str = Field(..., min_length=1)
    character: str          # speaker's display name — used for both voice
                             # lookup (via _speaker_id_from_name) and the
                             # output filename
    chat_name: str
    message_num: int
    speaker_id: Optional[int] = None  # explicit override; else resolved from `character`
    audio_temperature: Optional[float] = None
    audio_top_p: float = 0.95
    audio_top_k: int = 50
    audio_repetition_penalty: float = 1.05
    max_new_tokens: Optional[int] = None
    style_emotion_id: Optional[int] = None
    style_energy: Optional[float] = None
    style_strength: Optional[float] = None


@app.post("/st_cache/generate")
def st_cache_generate(req: STCacheGenerateRequest, request: Request):
    speaker_id = req.speaker_id if req.speaker_id is not None else _speaker_id_from_name(req.character)
    tts_req = TTSRequest(
        text=req.text,
        speaker_id=speaker_id,
        format="wav",
        audio_temperature=req.audio_temperature,
        audio_top_p=req.audio_top_p,
        audio_top_k=req.audio_top_k,
        audio_repetition_penalty=req.audio_repetition_penalty,
        max_new_tokens=req.max_new_tokens,
        style_emotion_id=req.style_emotion_id,
        style_energy=req.style_energy,
        style_strength=req.style_strength,
    )
    out = _synthesize(tts_req, _client_app(request))

    chat_slug = _safe_path_component(req.chat_name, "chat")
    character_slug = _safe_path_component(req.character, "speaker")
    chat_dir = ST_CACHE_DIR / chat_slug
    chat_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{character_slug}-{req.message_num}.wav"
    (chat_dir / filename).write_bytes(out["wav"])

    cache_path = f"/st_cache/{chat_slug}/{filename}"
    return Response(
        content=out["wav"],
        media_type="audio/wav",
        headers={"X-Cache-Path": cache_path},
    )


# Stop buttons, live activity line and the History list for the web page.
# Plain string (not an f-string) so the JavaScript can use normal braces.
# Every history value is inserted with textContent / attributes, never as HTML:
# the texts come from any client.
_FX_PAGE_JS = """<script>
let fxData=null;
function fxShowStr(){ document.getElementById('fxstrv').textContent=(+document.getElementById('fxstr').value).toFixed(2); }
async function fxInit(){
  try{
    fxData=await (await fetch('/voice_fx')).json();
    const sel=document.getElementById('fxsel');
    sel.innerHTML='<option value="off">No effect</option>'+fxData.effects.map(e=>`<option value="${e.id}">${e.label}</option>`).join('');
    fxLoadForVoice();
  }catch(e){ document.getElementById('fxstat').textContent='effects unavailable: '+e; }
}
function fxLoadForVoice(){
  if(!fxData) return;
  const id=document.getElementById('s').value, cur=fxData.per_speaker[id];
  document.getElementById('fxsel').value=cur?cur.fx:'off';
  document.getElementById('fxstr').value=cur?cur.strength:0.7;
  fxShowStr();
  const n=Object.keys(fxData.per_speaker).length;
  document.getElementById('fxstat').textContent=cur?'saved: '+cur.fx+' at '+cur.strength.toFixed(2)
    :'no effect saved for this voice ('+n+' voice'+(n==1?'':'s')+' have one)';
}
async function fxSave(preview){
  const body={speaker_id:+document.getElementById('s').value,fx:document.getElementById('fxsel').value,
              strength:+document.getElementById('fxstr').value};
  const r=await fetch('/voice_fx',{method:'POST',headers:{'Content-Type':'application/json','X-Client-App':'webui'},body:JSON.stringify(body)});
  if(!r.ok){ document.getElementById('fxstat').textContent='error: '+(await r.json()).detail; return; }
  fxData=await r.json(); fxLoadForVoice();
  document.getElementById('fxstat').textContent='saved - applies to this voice for every caller';
  if(preview) go();
}
fxInit();
</script>"""


_PAGE_EXTRA_JS = """<script>
const $ = id => document.getElementById(id);
async function stopGen(scope){
  if(scope==='all' && !confirm('Stop every generation, including the VN and SillyTavern?')) return;
  try{
    const r=await fetch('/cancel?scope='+scope,{method:'POST',headers:{'X-Client-App':'webui'}});
    const j=await r.json();
    $('st').textContent='stop sent: '+j.cancelled+' request(s) cancelled (current sentence finishes first)';
  }catch(e){ $('st').textContent='stop failed: '+e; }
  if(scope==='mine' && genCtl) genCtl.abort();
}
async function pollActivity(){
  try{
    const h=await (await fetch('/health')).json();
    const a=h.activity||[];
    if(!a.length){ $('act').textContent='server idle'; }
    else{
      const g=a.filter(x=>x.state==='generating'), q=a.length-g.length;
      $('act').textContent='busy: '+(g.length?('generating for '+g[0].client+' ('+g[0].seconds+'s) - "'+g[0].text+'"'):'starting')+
        (q?(' | '+q+' queued'):'');
    }
  }catch(e){ $('act').textContent=''; }
  setTimeout(pollActivity,2000);
}
function el(tag, cls, text){ const e=document.createElement(tag); if(cls) e.className=cls; if(text!==undefined) e.textContent=text; return e; }
function fillFilter(sel, values){
  const cur=sel.value, first=sel.options[0];
  sel.replaceChildren(first);
  for(const v of values){ const o=el('option','',v); o.value=v; sel.appendChild(o); }
  sel.value=values.includes(cur)?cur:'';
}
let histAll=[];
async function loadHistory(){
  const q=new URLSearchParams({limit:'500'});
  if($('hvoice').value) q.set('voice',$('hvoice').value);
  if($('hclient').value) q.set('client',$('hclient').value);
  let j;
  try{ j=await (await fetch('/history?'+q)).json(); }catch(e){ $('hstat').textContent='could not load history'; return; }
  if(!j.enabled){ $('hstat').textContent='History is off (MOSS_HISTORY_MAX=0).'; $('hist').replaceChildren(); return; }
  if(!$('hvoice').value && !$('hclient').value){
    histAll=j.items;
    fillFilter($('hvoice'), [...new Set(histAll.map(x=>x.voice))].sort());
    fillFilter($('hclient'), [...new Set(histAll.map(x=>x.client))].sort());
  }
  $('hstat').textContent=j.items.length+' shown (keeps the newest '+j.max+')';
  const list=$('hist'); list.replaceChildren();
  for(const it of j.items){
    const row=el('div','h');
    row.appendChild(el('div','meta', it.time.replace('T',' ')+'  ·  '+it.voice+'  ·  '+it.client+'  ·  '+it.duration_sec+'s  ·  temp '+it.temperature));
    const t=el('div','txt', it.text); t.title='click to show all'; t.onclick=()=>t.classList.toggle('open');
    row.appendChild(t);
    const au=el('audio'); au.controls=true; au.preload='none'; au.src=it.url; row.appendChild(au);
    const acts=el('div','acts');
    const dl=el('a','', 'download'); dl.href=it.url; dl.download=it.id+'.wav'; acts.appendChild(dl);
    const del=el('button','', 'delete');
    del.onclick=async()=>{ if(!confirm('Delete this clip?')) return;
      const r=await fetch('/history/'+encodeURIComponent(it.id),{method:'DELETE'}); if(r.ok) row.remove(); };
    acts.appendChild(del); row.appendChild(acts);
    list.appendChild(row);
  }
}
async function clearHistory(){
  if(!confirm('Delete ALL saved clips?')) return;
  await fetch('/history',{method:'DELETE'}); loadHistory();
}
pollActivity(); loadHistory();
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    opts = "".join(
        f'<option value="{s["id"]}">{s["label"]}</option>' for s in SPEAKERS
    )
    return f"""<!doctype html><html><head><meta charset=utf-8>
<title>PoneTTS Management</title>
<style>body{{font-family:system-ui;max-width:640px;margin:40px auto;padding:0 16px}}
textarea,select,button{{font-size:16px;width:100%;margin:6px 0;padding:8px}}</style></head>
<style>.box{{border:1px solid #ccc;border-radius:8px;padding:8px 12px;margin:10px 0;background:#f7f7f7}}
.box label{{font-weight:600}} #pstat{{font-size:14px;color:#555}}
.row{{display:flex;gap:8px;align-items:center}} .row button,.row select{{width:auto;flex:1}}
button.stop{{background:#fbe9e9;border:1px solid #d99}} button.small{{flex:0 0 auto;font-size:14px;padding:4px 10px}}
#act,#hstat{{font-size:14px;color:#555}}
.h{{border-top:1px solid #ddd;padding:8px 0}} .h .meta{{font-size:13px;color:#555}}
.h .txt{{margin:4px 0;cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.h .txt.open{{white-space:normal}} .h audio{{width:100%;height:36px}}
.h .acts{{font-size:13px}} .h .acts a,.h .acts button{{margin-right:10px;font-size:13px;width:auto;padding:2px 8px}}</style>
<body><h2>PoneTTS Management</h2>
<div class=box>
<label for=preset>Quality / VRAM</label>
<select id=preset onchange=setPreset()></select>
<p id=pstat>&nbsp;</p>
</div>
<textarea id=t rows=3>Hello there! Welcome to Ponyville.</textarea>
<select id=s onchange=fxLoadForVoice()>{opts}</select>
<div class=box id=fxbox><div class=row><strong style="flex:1">Voice effect</strong>
<select id=fxsel></select>
<input id=fxstr type=range min=0 max=1 step=0.05 value=0.7 style="flex:1" oninput="fxShowStr()">
<span id=fxstrv style="width:2.5em;text-align:right">0.70</span></div>
<div class=row><button class=small onclick="fxSave(false)" title="Save this effect for the voice selected above. It then applies for every caller: SillyTavern, the VN and this page.">Save for this voice</button>
<button class=small onclick="fxSave(true)">Save &amp; preview</button></div>
<p id=fxstat></p></div>
<label>temperature <input id=temp type=number step=0.05 value={_default_audio_temperature}></label>
<div class=row><button id=speak onclick=go()>Speak</button>
<button id=stopmine class=stop onclick="stopGen('mine')" hidden>Stop mine</button>
<button id=stopall class=stop onclick="stopGen('all')" title="Cancel every client's generations (VN, SillyTavern, this page)">Stop all</button></div>
<p id=st></p><audio id=a controls></audio>
<p id=act></p>
<div class=box id=histbox><div class=row><strong style="flex:1">History</strong>
<select id=hvoice onchange=loadHistory()><option value="">all voices</option></select>
<select id=hclient onchange=loadHistory()><option value="">all clients</option></select>
<button class=small onclick=loadHistory()>Refresh</button>
<button class="small stop" onclick=clearHistory()>Clear all</button></div>
<p id=hstat></p><div id=hist></div></div>
<script>
const presetSel=document.getElementById('preset'),pstat=document.getElementById('pstat'),
      speakBtn=document.getElementById('speak');
async function loadPresets(){{
 const p=await (await fetch('/preset')).json();
 presetSel.innerHTML=p.options.map(o=>`<option value="${{o.id}}">${{o.label}}</option>`).join('');
 presetSel.value=p.active;
}}
async function setPreset(){{
 const name=presetSel.value;
 const r=await fetch('/preset?name='+encodeURIComponent(name),{{method:'POST'}});
 if(!r.ok){{pstat.textContent='error: '+(await r.json()).detail;return;}}
 const j=await r.json();
 if(j.status==='unchanged'){{pstat.textContent='already on '+name;return;}}
 presetSel.disabled=true;speakBtn.disabled=true;
 pstat.textContent='reloading model into '+name+' (~30-90s)...';
 poll();
}}
async function poll(){{
 try{{
  const h=await (await fetch('/health')).json();
  if(h.reload_error){{pstat.textContent='reload failed: '+h.reload_error;}}
  else if(h.reloading||!h.model_loaded){{pstat.textContent='reloading into '+(h.reload_target||'...')+' (~30-90s)...';return setTimeout(poll,2000);}}
  else{{pstat.textContent='ready on '+h.preset;presetSel.value=h.preset;}}
 }}catch(e){{return setTimeout(poll,2000);}}
 presetSel.disabled=false;speakBtn.disabled=false;
}}
let genCtl=null;
async function go(){{
 const st=document.getElementById('st');st.textContent='generating...';
 const body={{text:document.getElementById('t').value,
   speaker_id:+document.getElementById('s').value,
   audio_temperature:+document.getElementById('temp').value}};
 genCtl=new AbortController();
 speakBtn.disabled=true;document.getElementById('stopmine').hidden=false;
 try{{
  const r=await fetch('/tts',{{method:'POST',headers:{{'Content-Type':'application/json','X-Client-App':'webui'}},
    body:JSON.stringify(body),signal:genCtl.signal}});
  if(r.status===409){{st.textContent='stopped';return;}}
  if(!r.ok){{st.textContent='error '+r.status;return;}}
  const b=await r.blob();document.getElementById('a').src=URL.createObjectURL(b);
  st.textContent='ok ('+(r.headers.get('X-Duration-Sec')||'?')+'s)';
  document.getElementById('a').play();
  loadHistory();
 }}catch(e){{st.textContent=(e.name==='AbortError')?'stopped':'error: '+e;}}
 finally{{genCtl=null;speakBtn.disabled=false;document.getElementById('stopmine').hidden=true;}}
}}
loadPresets();
</script>""" + _PAGE_EXTRA_JS + _FX_PAGE_JS


# ----------------------------------------------------------------------------
# Performance presets — trade quality/speed for VRAM so smaller NVIDIA cards work.
# NOTE: the model needs an NVIDIA GPU (the codec/decoder4 is CUDA-fp16 only — there
# is no CPU build), so these tune VRAM *within* CUDA; they don't enable CPU-only.
#   opt      = torch sampler mode (cudagraph = fastest + most VRAM; fixed-full = less)
#   vocoder  = where the iSTFTNet2 vocoder runs (onnx_hybrid/onnx_cuda use VRAM; onnx_cpu frees ~5GB)
#   tokens   = default max_new_tokens ceiling (lower = lower peak memory on long lines)
# ----------------------------------------------------------------------------
PRESETS: dict[str, dict] = {
    # ~14-16 GB cards: fastest, best quality headroom. GPU vocoder (onnx_cuda).
    "quality":  {"opt": "fixed-full-cudagraph", "compile": "max-autotune-no-cudagraphs",
                 "vocoder": "onnx_cuda", "tokens": 1024, "quant": "none"},
    # ~9-12 GB cards: fast LM on GPU, vocoder on CPU (frees the ~5GB vocoder arena so
    # sustained game load can't OOM the GPU). Same voices/quality as quality.
    "balanced": {"opt": "fixed-full-cudagraph", "compile": "max-autotune-no-cudagraphs",
                 "vocoder": "onnx_cpu", "tokens": 1024, "quant": "none"},
    # ~7-8 GB cards: drop CUDA-graph capture too, vocoder on CPU, shorter cap
    "low_vram": {"opt": "fixed-full", "compile": "default",
                 "vocoder": "onnx_cpu", "tokens": 768, "quant": "none"},
    # weakest (~6 GB) cards: same as low_vram with the smallest token cap.
    # Not quantized: the bundle only supports "none" or torchao fp8
    # (MOSS_QUANT), and fp8 needs an RTX 40-series or newer.
    "potato":   {"opt": "fixed-full", "compile": "default",
                 "vocoder": "onnx_cpu", "tokens": 512, "quant": "none"},
}
# Default to "balanced": same voices/quality as "quality" but the vocoder runs on
# the CPU, so it CANNOT exhaust GPU VRAM under sustained game load (the OOM crash).
# Strong-card users can switch to "quality" in the web dropdown for max speed.
DEFAULT_PRESET = "balanced"
_active_preset = DEFAULT_PRESET


def _resolve_vocoder(requested: str) -> str:
    """Return the vocoder runtime to actually use.

    onnx_cuda needs onnxruntime's CUDA provider on an NVIDIA GPU. It isn't
    available on ROCm (AMD), or if the CPU onnxruntime package overwrote the
    GPU one. The bundle would quietly fall back to the CPU in that case; do
    it here instead so the log says so."""
    if requested != "onnx_cuda":
        return requested
    reason = None
    if getattr(torch.version, "hip", None):
        reason = "AMD/ROCm GPU (onnxruntime's CUDA provider is NVIDIA-only)"
    else:
        try:
            import onnxruntime as ort
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                reason = "onnxruntime has no CUDA provider (is onnxruntime-gpu installed?)"
        except Exception as exc:  # noqa: BLE001
            reason = f"onnxruntime failed to import ({exc})"
    if reason:
        print(f"[server] WARNING: GPU vocoder unavailable: {reason}. "
              f"Running the vocoder on the CPU (onnx_cpu) instead.", flush=True)
        return "onnx_cpu"
    return requested


def build_runner(preset: str, max_new_tokens: Optional[int] = None) -> None:
    global _runner, _default_max_new_tokens, _active_preset
    preset = preset if preset in PRESETS else DEFAULT_PRESET
    _active_preset = preset
    cfg_p = PRESETS[preset]
    # env still wins (back-compat / power users): MOSS_DECODER_RUNTIME overrides vocoder
    decoder_runtime = _resolve_vocoder(os.environ.get("MOSS_DECODER_RUNTIME", cfg_p["vocoder"]))
    _default_max_new_tokens = max_new_tokens or cfg_p["tokens"]
    print(f"[server] loading MOSS-TTS-PNY (preset={preset}, first call compiles ~1-2 min)...", flush=True)
    quant = os.environ.get("MOSS_QUANT", cfg_p.get("quant", "none"))
    cfg = OptimizedTTSConfig(
        decoder_runtime=decoder_runtime,
        torch_opt_mode=cfg_p["opt"],
        compile_mode=cfg_p["compile"],
        tts_quantization=quant,
    )
    print(f"[server] vocoder={decoder_runtime} opt={cfg_p['opt']} quant={quant} "
          f"max_new_tokens={_default_max_new_tokens}", flush=True)
    _runner = OptimizedTTSRunner(cfg)
    # warm up so the first real request is fast
    print("[server] warming up...", flush=True)
    with _lock:
        _runner.synthesize(text="Warm up.", speaker_id=31, max_new_tokens=_default_max_new_tokens,
                           style_emotion_id=DEFAULT_EMOTION_ID, style_energy=DEFAULT_ENERGY)
    if _RESPELL_AVAILABLE and FIX_PRON_DEFAULT:
        # preload whisper (CPU) in the background so the first verified line is fast
        threading.Thread(target=_get_whisper, daemon=True).start()
    print(f"[server] ready. device={_runner.device} sr={_runner.sample_rate} "
          f"speakers={len(SPEAKERS)} temp={_default_audio_temperature}", flush=True)


# ----------------------------------------------------------------------------
# Voice-patch blend (dynamic): the deployed model is base + 1.0x the LoRA delta.
# Because delta = (alpha/r) * B@A from the 101MB adapter, the live weights can be
# shifted to ANY blend s in seconds: W += (s_new - s_cur) * delta. s=0 is the
# original base voices, s=1 the full pronunciation fine-tune. Mid values trade a
# little of the fine-tune's pronunciation for more of the base model's prosody on
# data-rich speakers (Dash, Twilight, ...).
# ----------------------------------------------------------------------------
VOICE_BLEND_FILE = PROJECT_DIR / "voice_blend.json"
_VOICE_PATCH_DIRS = [
    Path(os.environ.get("MOSS_VOICE_PATCH_DIR", "")),
    PROJECT_DIR / "voice_patch",
]
_blend_scale: float = 1.0            # what the live weights currently contain
_blend_factors = None                # name -> (A, B) on CPU, loaded lazily
_blend_scaling: float = 2.0          # alpha/r from adapter_config.json
_blend_default: float = 1.0          # config: scale for speakers with no override
# config: per-speaker overrides. Weak-data speakers NEED the full patch (measured:
# Maud at 0.6 says "urchin pests" for "erosion deposits"); data-rich speakers keep
# perfect pronunciation at partial blend and regain the base model's prosody.
_blend_per_speaker: dict[int, float] = {}


def _blend_desired(speaker_id: int) -> float:
    return _blend_per_speaker.get(speaker_id, _blend_default)


def _blend_save() -> None:
    try:
        VOICE_BLEND_FILE.write_text(json.dumps(
            {"default": _blend_default,
             "per_speaker": {str(k): v for k, v in sorted(_blend_per_speaker.items())}},
            indent=1), encoding="utf-8")
    except OSError:
        pass


def _blend_load_config() -> None:
    global _blend_default, _blend_per_speaker
    try:
        cfg = json.loads(VOICE_BLEND_FILE.read_text())
    except (OSError, ValueError):
        return
    if "scale" in cfg and "default" not in cfg:   # migrate the old single-scale format
        cfg = {"default": cfg["scale"], "per_speaker": {}}
    try:
        _blend_default = max(0.0, min(1.5, float(cfg.get("default", 1.0))))
        _blend_per_speaker = {int(k): max(0.0, min(1.5, float(v)))
                              for k, v in (cfg.get("per_speaker") or {}).items()}
    except (TypeError, ValueError):
        pass


def _voice_patch_dir() -> Optional[Path]:
    for d in _VOICE_PATCH_DIRS:
        if d and (d / "adapter_model.safetensors").exists():
            return d
    return None


def _blend_load_factors() -> dict:
    """name (module path in the loaded model) -> (lora_A, lora_B), CPU fp32."""
    global _blend_factors, _blend_scaling
    if _blend_factors is not None:
        return _blend_factors
    d = _voice_patch_dir()
    if d is None:
        raise HTTPException(status_code=503, detail="voice patch adapter not found; blend unavailable")
    cfg = json.loads((d / "adapter_config.json").read_text())
    _blend_scaling = float(cfg.get("lora_alpha", 32)) / float(cfg.get("r", 16))
    from safetensors import safe_open
    pairs: dict[str, dict] = {}
    with safe_open(str(d / "adapter_model.safetensors"), framework="pt") as f:
        for k in f.keys():
            # base_model.model.<path>.lora_A.weight -> <path>
            m = re.match(r"base_model\.model\.(.+)\.lora_([AB])\.weight$", k)
            if not m:
                continue
            pairs.setdefault(m.group(1), {})[m.group(2)] = f.get_tensor(k).float()
    _blend_factors = {n: (ab["A"], ab["B"]) for n, ab in pairs.items()
                      if "A" in ab and "B" in ab}
    print(f"[blend] adapter loaded: {len(_blend_factors)} layers, scaling={_blend_scaling}", flush=True)
    return _blend_factors


_BLEND_CHUNK_ROWS = 64   # rows of each weight matrix updated per step — keeps
                         # the transient delta tensor small even on a nearly-
                         # full GPU, instead of one big allocation per layer
_BLEND_RETRIES = 3       # the OOM that motivated this can be transient (co-
                         # occurring with another request's peak usage), so
                         # a short bounded retry is worth it before giving up


def _blend_shift_locked(target: float) -> int:
    """Shift live weights by (target - current) * delta. CALLER HOLDS _lock.
    Returns the number of layers updated (0 if already at target).

    Applied in small row-chunks per layer to keep the transient scratch
    allocation small (this used to compute one full-size delta tensor per
    layer, which could be tens of MB — enough to OOM on an 8GB card already
    near its ceiling from the base model + KV cache alone). If ANY chunk
    fails after retries, every change made so far in this call is rolled
    back (delta subtracted back out) before re-raising, so a failed shift
    always leaves the live weights exactly as they were — never partially
    blended between two different states with _blend_scale reporting a
    third, stale value."""
    global _blend_scale
    target = max(0.0, min(1.5, float(target)))
    factors = _blend_load_factors()
    assert _runner is not None, "model not loaded"
    diff = target - _blend_scale
    if abs(diff) < 1e-6:
        return 0
    # The runner nests modules differently than the PEFT training wrap did:
    # torch.compile inserts "._orig_mod" into paths (language_model), and the
    # top-level wrapper may add/omit a leading "model.". Index the live tree
    # under de-noised names and resolve each adapter name across variants.
    modules: dict = {}
    for n, m in _runner.model.named_modules():
        modules.setdefault(n, m)
        modules.setdefault(n.replace("._orig_mod", ""), m)

    def _resolve(n: str):
        for cand in (n, n.removeprefix("model."), "model." + n):
            m = modules.get(cand)
            if m is not None and hasattr(m, "weight"):
                return m
        return None

    resolved = {n: _resolve(n) for n in factors}
    missing = [n for n, m in resolved.items() if m is None]
    if missing:
        raise HTTPException(status_code=500,
                            detail=f"blend name mismatch: {len(missing)} adapter layers "
                                   f"not found in model (e.g. {missing[0]})")
    # pre-flight shape check: abort BEFORE touching any weight, never mid-loop
    for n, (A, B) in factors.items():
        if tuple(resolved[n].weight.shape) != (B.shape[0], A.shape[1]):
            raise HTTPException(status_code=500,
                                detail=f"blend shape mismatch at {n}: weight "
                                       f"{tuple(resolved[n].weight.shape)} vs delta "
                                       f"{(B.shape[0], A.shape[1])}")

    t0 = time.time()
    # Everything needed to undo a chunk without re-deriving anything —
    # (weight_tensor, A_on_device, B_full, row_start, row_end) for every
    # chunk actually applied so far in this call.
    applied: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]] = []

    def _apply_chunk(w, A_dev, B, start, end, sign: float) -> None:
        """Apply (sign=+1) or undo (sign=-1) one chunk's delta, with the
        same bounded retry either direction — if we're rolling back BECAUSE
        of memory pressure, the rollback's own fresh delta computation can
        hit that exact same pressure, so it needs the same resilience the
        forward pass has, not just the forward pass."""
        attempt = 0
        while True:
            try:
                B_chunk = B[start:end].to(w.device)
                delta_chunk = (B_chunk @ A_dev) * (_blend_scaling * diff * sign)
                w[start:end].add_(delta_chunk.to(w.dtype))
                return
            except torch.cuda.OutOfMemoryError:
                attempt += 1
                if attempt >= _BLEND_RETRIES:
                    raise
                torch.cuda.empty_cache()
                time.sleep(0.05)

    try:
        with torch.no_grad():
            for name, (A, B) in factors.items():
                w = resolved[name].weight
                out_rows = B.shape[0]
                A_dev = A.to(w.device)
                for start in range(0, out_rows, _BLEND_CHUNK_ROWS):
                    end = min(start + _BLEND_CHUNK_ROWS, out_rows)
                    _apply_chunk(w, A_dev, B, start, end, sign=1.0)
                    applied.append((w, A_dev, B, start, end))
    except Exception:
        # Undo every chunk applied so far in THIS call, restoring the exact
        # pre-attempt weights, so a failed shift never leaves the model in a
        # half-blended, inconsistent state.
        with torch.no_grad():
            for w, A_dev, B, start, end in reversed(applied):
                try:
                    _apply_chunk(w, A_dev, B, start, end, sign=-1.0)
                except torch.cuda.OutOfMemoryError:
                    # Retries exhausted on the rollback itself too — memory
                    # pressure severe enough that even undoing is failing.
                    # Don't pretend this is a clean failure: say plainly
                    # that live weights may now be inconsistently blended.
                    print("[blend] CRITICAL: rollback could not complete — "
                          "live weights may be partially blended. Restart "
                          "the server or free up GPU memory before relying "
                          "on further generations.", flush=True)
                    raise
        raise

    _blend_scale = target
    print(f"[blend] weights now at {_blend_scale:g}x voice patch "
          f"({len(factors)} layers, {time.time() - t0:.1f}s)", flush=True)
    return len(factors)


def _blend_sync_for_speaker(speaker_id: int) -> None:
    """Called inside _synth_once (lock held): make the live weights match this
    speaker's configured blend. No-op when nothing needs to change or when no
    adapter is available. Never breaks synthesis."""
    if _voice_patch_dir() is None:
        return
    try:
        want = _blend_desired(speaker_id)
        if abs(want - _blend_scale) >= 1e-6:
            _blend_shift_locked(want)
    except Exception as exc:  # noqa: BLE001
        print(f"[blend] per-speaker sync failed ({exc}); staying at {_blend_scale:g}", flush=True)


class BlendRequest(BaseModel):
    scale: float = Field(..., ge=-1.0, le=1.5)   # -1 clears a per-speaker override
    speaker_id: Optional[int] = None


@app.get("/voice_blend")
def voice_blend_get():
    return {
        "default": _blend_default,
        "per_speaker": {str(k): v for k, v in sorted(_blend_per_speaker.items())},
        "live_scale": _blend_scale,
        "available": _voice_patch_dir() is not None,
        "hint": "0 = original voices, 1 = full pronunciation patch. POST "
                "{\"scale\": 0.6} sets the default; add \"speaker_id\" for one "
                "speaker; scale -1 clears that speaker's override.",
    }


@app.post("/voice_blend")
def voice_blend_set(req: BlendRequest):
    global _blend_default
    if req.speaker_id is not None:
        if req.speaker_id not in SPEAKER_IDS:
            raise HTTPException(status_code=400, detail=f"unknown speaker_id {req.speaker_id}")
        if req.scale < 0:
            _blend_per_speaker.pop(req.speaker_id, None)
        else:
            _blend_per_speaker[req.speaker_id] = req.scale
    else:
        if req.scale < 0:
            raise HTTPException(status_code=400, detail="default scale must be 0..1.5")
        _blend_default = req.scale
    _blend_save()
    # weights re-sync lazily per request; nudge them now so a curl test hears it
    if _voice_patch_dir() is not None:
        with _lock:
            try:
                _blend_shift_locked(req.scale if (req.speaker_id is None and req.scale >= 0)
                                    else _blend_default)
            except HTTPException:
                raise
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(status_code=500, detail=f"blend failed: {exc}") from exc
    return voice_blend_get()


# ----------------------------------------------------------------------------
# Voice effects (post-processing). Applied to the finished audio of a voice, so
# the effect belongs to the VOICE and is identical for every caller (SillyTavern,
# the VN, this web page). The web page is the control centre: POST /voice_fx
# saves {speaker_id -> effect + strength} to voice_fx.json. Pure numpy, no new
# dependencies. Streaming-safe: each effect is a small stateful object that
# processes one chunk at a time and carries what it needs (oscillator phase,
# reverb tail) across chunk boundaries so sentence joins don't click.
# ----------------------------------------------------------------------------
VOICE_FX_FILE = PROJECT_DIR / "voice_fx.json"
FX_NAMES = {
    "robot": "Robot",
    "radio": "Radio",
    "cavern": "Cavern",
    "helium": "Helium",
    "changeling": "Changeling",
}
_fx_per_speaker: dict[int, dict] = {}   # speaker_id -> {"fx": name, "strength": 0..1}
_fx_lock = threading.Lock()


def _fx_load_config() -> None:
    global _fx_per_speaker
    try:
        cfg = json.loads(VOICE_FX_FILE.read_text())
    except (OSError, ValueError):
        return
    out = {}
    try:
        for k, v in (cfg.get("per_speaker") or {}).items():
            name = str(v.get("fx", ""))
            if name in FX_NAMES:
                out[int(k)] = {"fx": name, "strength": max(0.0, min(1.0, float(v.get("strength", 0.7))))}
    except (TypeError, ValueError, AttributeError):
        return
    _fx_per_speaker = out


def _fx_save() -> None:
    try:
        VOICE_FX_FILE.write_text(json.dumps(
            {"per_speaker": {str(k): v for k, v in sorted(_fx_per_speaker.items())}},
            indent=1), encoding="utf-8")
    except OSError:
        pass


def _stft(x: np.ndarray, n: int = 1024, hop: int = 256) -> np.ndarray:
    pad = np.zeros(n, dtype=np.float32)
    x = np.concatenate([pad, x, pad])
    win = np.hanning(n + 1)[:-1].astype(np.float32)
    n_frames = 1 + (len(x) - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(n_frames)[:, None]
    return np.fft.rfft(x[idx] * win, axis=1)


def _istft(S: np.ndarray, start: int, length: int, n: int = 1024, hop: int = 256) -> np.ndarray:
    win = np.hanning(n + 1)[:-1].astype(np.float32)
    frames = np.fft.irfft(S, n=n, axis=1).astype(np.float32) * win
    total = hop * (len(frames) - 1) + n
    y = np.zeros(total, dtype=np.float32)
    wsum = np.zeros(total, dtype=np.float32)
    for i, f in enumerate(frames):
        y[i * hop:i * hop + n] += f
        wsum[i * hop:i * hop + n] += win * win
    y /= np.maximum(wsum, 1e-3)
    y = y[start:start + length]
    if len(y) < length:
        y = np.pad(y, (0, length - len(y)))
    return y


def _pitch_shift(x: np.ndarray, semitones: float) -> np.ndarray:
    """Pitch shift keeping the duration (phase-vocoder time-stretch, then
    resample back to the original length). Same length in as out, so shifted
    layers stay aligned with the original."""
    if len(x) < 2048 or abs(semitones) < 1e-3:
        return x
    n, hop = 1024, 256
    ratio = 2.0 ** (semitones / 12.0)          # >1 = higher
    S = _stft(x, n, hop)
    rate = 1.0 / ratio                          # stretch by `ratio`, then play back faster/slower
    steps = np.arange(0, S.shape[0] - 1, rate)
    omega = 2 * np.pi * hop * np.arange(S.shape[1]) / n
    phase = np.angle(S[0])
    out = np.empty((len(steps), S.shape[1]), dtype=np.complex64)
    for i, t in enumerate(steps):
        k = int(t)
        frac = t - k
        a, b = S[k], S[k + 1]
        mag = (1 - frac) * np.abs(a) + frac * np.abs(b)
        out[i] = mag * np.exp(1j * phase)
        dphi = np.angle(b) - np.angle(a) - omega
        dphi -= 2 * np.pi * np.round(dphi / (2 * np.pi))
        phase = phase + omega + dphi
    # drop the input's leading pad (it was stretched too), keep exactly len*ratio samples
    start = int(round(n * ratio))
    stretched = _istft(out, start, int(round(len(x) * ratio)), n, hop)
    # resample the stretched signal back to the original length
    src = np.linspace(0, len(stretched) - 1, num=len(x))
    y = np.interp(src, np.arange(len(stretched)), stretched).astype(np.float32)
    # the vocoder sheds some energy on big upward shifts; match the input's level
    r_in = float(np.sqrt(np.mean(x * x)))
    r_out = float(np.sqrt(np.mean(y * y)))
    if r_out > 1e-6:
        y *= min(4.0, max(0.5, r_in / r_out))
    return y


def _fft_filter(x: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    n = 1 << int(np.ceil(np.log2(max(len(x), 2))))
    X = np.fft.rfft(x, n)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    mask = ((f >= lo) & (f <= hi)).astype(np.float32)
    # soften the band edges a little so it doesn't ring
    k = max(1, int(150 / (sr / n)))
    mask = np.convolve(mask, np.hanning(2 * k + 1) / np.hanning(2 * k + 1).sum(), mode="same")
    return np.fft.irfft(X * mask, n)[:len(x)].astype(np.float32)


class _Fx:
    """One effect instance for one request. process() returns exactly as many
    samples as it was given; anything that rings on past the end (reverb) is
    held back and added to the start of the next chunk, or returned by flush()."""

    def __init__(self, name: str, strength: float, sr: int):
        self.name, self.s, self.sr = name, float(strength), sr
        self.pos = 0                                   # samples seen (oscillator phase)
        self.carry = np.zeros(0, dtype=np.float32)
        self.hist = np.zeros(int(sr * 0.003), dtype=np.float32)
        if name == "cavern":
            rng = np.random.default_rng(7)
            L = int(sr * 1.8)
            t = np.arange(L) / sr
            self.ir = (rng.standard_normal(L) * np.exp(-t * 3.2)).astype(np.float32)
            self.ir[0] = 0.0
            self.ir /= np.sqrt(np.sum(self.ir ** 2)) + 1e-9

    def process(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        if n == 0:
            return x
        s, sr = self.s, self.sr
        if self.name == "robot":
            t = (self.pos + np.arange(n)) / sr
            ring = x * np.sin(2 * np.pi * 75.0 * t).astype(np.float32)
            d = int(sr * 0.003)                         # 3 ms comb = metallic resonance
            ext = np.concatenate([self.hist, ring])     # hist = last d samples of the previous chunk
            comb = (ring + 0.55 * ext[:n]) * 0.75
            self.hist = ext[-d:]
            y = (1 - s) * x + s * comb
        elif self.name == "radio":
            band = _fft_filter(x, sr, 450.0, 3300.0)
            driven = np.tanh(band * (1.0 + 4.0 * s)) / np.tanh(1.0 + 4.0 * s)
            y = (1 - s) * x + s * driven
        elif self.name == "helium":
            y = (1 - s) * x + s * _pitch_shift(x, 3.0 + 4.0 * s)   # +3 .. +7 semitones
        elif self.name == "changeling":
            low2 = _pitch_shift(x, -2.0)
            low4 = _pitch_shift(x, -4.0)
            y = x + s * (0.8 * low2 + 0.65 * low4)
            y /= (1.0 + s * 1.45) ** 0.6                # keep the stack from getting louder than one voice
        elif self.name == "cavern":
            m = len(self.ir)
            size = 1 << int(np.ceil(np.log2(n + m)))
            wet = np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(self.ir, size), size)[:n + m - 1].astype(np.float32)
            wet *= 0.9 * s
            dry = x * (1.0 - 0.35 * s)
            full = np.concatenate([dry, np.zeros(m - 1, dtype=np.float32)]) + wet
            if len(self.carry):
                full[:len(self.carry)] += self.carry
            self.carry = full[n:].copy()
            y = full[:n]
        else:
            y = x
        self.pos += n
        return np.clip(y, -1.0, 1.0).astype(np.float32)

    def flush(self) -> np.ndarray:
        """Whatever is still ringing after the last chunk (cavern reverb tail)."""
        out, self.carry = self.carry, np.zeros(0, dtype=np.float32)
        if len(out) and self.name == "cavern":
            return np.clip(out, -1.0, 1.0).astype(np.float32)
        return np.zeros(0, dtype=np.float32)


def _fx_for(speaker_id: int, sr: int) -> Optional[_Fx]:
    cfg = _fx_per_speaker.get(speaker_id)
    if not cfg:
        return None
    return _Fx(cfg["fx"], cfg["strength"], sr)


def _pcm16_to_f32(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.int16).astype(np.float32) / 32768.0


def _f32_to_pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


def _fx_apply_to_out(out: dict, speaker_id: int) -> dict:
    """One-shot path: run the voice's effect over a finished WAV result."""
    try:
        sr = int(out["sample_rate"])
        fx = _fx_for(speaker_id, sr)
        if fx is None:
            return out
        with wave.open(io.BytesIO(out["wav"]), "rb") as h:
            raw = h.readframes(h.getnframes())
        y = fx.process(_pcm16_to_f32(raw))
        y = np.concatenate([y, fx.flush()])
        buf = io.BytesIO()
        with wave.open(buf, "wb") as h:
            h.setnchannels(1)
            h.setsampwidth(2)
            h.setframerate(sr)
            h.writeframes(_f32_to_pcm16(y))
        out = dict(out)
        out["wav"] = buf.getvalue()
        out["duration_sec"] = float(len(y)) / sr
    except Exception as exc:  # noqa: BLE001 - an effect must never break synthesis
        print(f"[fx] skipped ({exc})", flush=True)
    return out


class FxRequest(BaseModel):
    speaker_id: int
    fx: str = "off"                      # off / robot / radio / cavern / helium / changeling
    strength: float = Field(0.7, ge=0.0, le=1.0)


@app.get("/voice_fx")
def voice_fx_get():
    return {
        "effects": [{"id": k, "label": v} for k, v in FX_NAMES.items()],
        "per_speaker": {str(k): v for k, v in sorted(_fx_per_speaker.items())},
        "hint": "POST {\"speaker_id\": N, \"fx\": \"robot\", \"strength\": 0.7}; fx \"off\" clears it. "
                "The effect belongs to the voice and applies for every caller.",
    }


@app.post("/voice_fx")
def voice_fx_set(req: FxRequest):
    if req.speaker_id not in SPEAKER_IDS:
        raise HTTPException(status_code=400, detail=f"unknown speaker_id {req.speaker_id}")
    if req.fx != "off" and req.fx not in FX_NAMES:
        raise HTTPException(status_code=400, detail=f"unknown effect '{req.fx}'; one of: off, {', '.join(FX_NAMES)}")
    with _fx_lock:
        if req.fx == "off":
            _fx_per_speaker.pop(req.speaker_id, None)
        else:
            _fx_per_speaker[req.speaker_id] = {"fx": req.fx, "strength": req.strength}
        _fx_save()
    return voice_fx_get()


def _persisted_preset() -> Optional[str]:
    """Preset saved by the web-page dropdown, if any (survives restarts)."""
    try:
        if PRESET_FILE.exists():
            val = PRESET_FILE.read_text(encoding="utf-8").strip()
            return val if val in PRESETS else None
    except OSError:
        pass
    return None


def _reload_worker(preset: str) -> None:
    """Drop the current model, free VRAM, and load a new preset. Runs in a thread
    so the HTTP request that triggered it returns immediately; the web page polls
    /health to see when the new preset is ready."""
    global _runner
    try:
        # Take the synthesis lock so no request runs mid-swap, then free the old model.
        with _lock:
            old, _runner = _runner, None
            del old
            _recover_gpu()
        build_runner(preset)             # sets _runner + _active_preset, warms up
        global _blend_scale
        _blend_scale = 1.0               # fresh load = the on-disk (merged, 1.0x) weights
        try:
            PRESET_FILE.write_text(preset + "\n", encoding="utf-8")  # persist across restarts
        except OSError:
            pass
        _reload_state["error"] = None
    except Exception as exc:  # noqa: BLE001
        _reload_state["error"] = str(exc)
        print(f"[server] preset reload FAILED: {exc}", flush=True)
    finally:
        _reload_state["busy"] = False
        _reload_state["target"] = None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-new-tokens", type=int, default=None,
                   help="override the preset's token ceiling")
    p.add_argument("--temperature", type=float, default=None,
                   help="default audio sampling temperature for requests that don't set one "
                        "(default: env MOSS_AUDIO_TEMPERATURE, then 0.85)")
    p.add_argument("--preset", default=None, choices=list(PRESETS),
                   help="performance/VRAM preset (default: env MOSS_PERF_PRESET, "
                        "then preset.conf, then 'quality')")
    args = p.parse_args()

    # Resolve preset: CLI flag > env > persisted preset.conf (set by the dropdown) > default
    if args.preset is None:
        args.preset = os.environ.get("MOSS_PERF_PRESET") or _persisted_preset() or DEFAULT_PRESET

    # CLI --temperature overrides the env/hardcoded default
    if args.temperature is not None:
        global _default_audio_temperature
        _default_audio_temperature = args.temperature

    # build model BEFORE uvicorn binds the socket (the runtime may os.execv-restart
    # the process to set LD_LIBRARY_PATH; doing it first keeps that clean).
    build_runner(args.preset, args.max_new_tokens)
    _blend_load_config()                 # per-speaker blends re-sync lazily per request
    _fx_load_config()                    # per-voice effects (voice_fx.json)

    import uvicorn

    print(f"[server] listening on http://{args.host}:{args.port}", flush=True)
    # access_log=False: the _request_log_middleware above replaces uvicorn's bare
    # "GET /tts HTTP/1.1" 200 OK line with one that also names the caller (client_app)
    # and timing — keeping both would just double every log line.
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", access_log=False)


if __name__ == "__main__":
    main()
