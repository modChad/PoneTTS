# v2.5 voice patch

A LoRA adapter for MOSS-TTS-PNY that fixes slurred pronunciation of uncommon
words for the low-screen-time characters (Maud, Zecora, Cheerilee, Celestia,
Mrs. Cake, ...). Trained on a speaker-rebalanced mix of the same public show
audio the base model used, so every character gets equal learning exposure.
Voice identity and the emotion/energy controls are untouched.

**Not included in this repo:** only `adapter_config.json` ships. Put
`adapter_model.safetensors` in this folder and it's applied automatically by
`apply_voice_patch.py` on the next setup or container start (merged into the
downloaded checkpoint; the unpatched original is kept next to it). Without it
the server runs on the base model.
