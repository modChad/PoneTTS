# v2.5 voice patch

A LoRA adapter for MOSS-TTS-PNY that fixes slurred pronunciation of uncommon
words for the low-screen-time characters (Maud, Zecora, Cheerilee, Celestia,
Mrs. Cake, ...). Trained on a speaker-rebalanced mix of the same public show
audio the base model used, so every character gets equal learning exposure.
Voice identity and the emotion/energy controls are untouched.

Applied automatically during setup by `apply_voice_patch.py` (merged into the
downloaded checkpoint; the unpatched original is kept next to it).
