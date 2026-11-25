# v2.5 voice patch

A LoRA adapter for MOSS-TTS-PNY that fixes slurred pronunciation of uncommon
words for the low-screen-time characters (Maud, Zecora, Cheerilee, Celestia,
Mrs. Cake, ...). Trained on a speaker-rebalanced mix of the same public show
audio the base model used, so every character gets equal learning exposure.
Voice identity and the emotion/energy controls are untouched.

Applied automatically during setup by `apply_voice_patch.py` (merged into the
downloaded checkpoint; the unpatched original is kept next to it).

The adapter (`adapter_model.safetensors`, ~100 MB) isn't stored in the repo.
`apply_voice_patch.py` downloads it here on the first start and checks its
SHA-256. To use your own copy, drop it in this folder; set
`MOSS_VOICE_PATCH_URL=0` to turn the download off.
