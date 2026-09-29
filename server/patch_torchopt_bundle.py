#!/usr/bin/env python3
"""Reapply the feedback_lookup_tables fix to the downloaded torchopt bundle.

release/moss_tts_torchopt_runner_bundle/torch_hf_optimizations.py is NOT part
of this project's source — download_model.py fetches it fresh from
HuggingFace on every clean volume. As shipped, two functions
(_sample_frame_fixed_full_impl and _sample_frame_greedy_full_impl) reference
a `feedback_lookup_tables` local that no caller ever supplies, which crashes
with a NameError the first time the model actually generates anything (see
the sibling cudagraph/static-local-cache variants, which already carry this
same parameter with a `= None` default — this patch brings these two in
line with that existing pattern).

Safe to run every container start: it's idempotent (skips any function that
already has the parameter) and a no-op if the bundle isn't there yet.
"""
import re
import sys
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent / "release" / "moss_tts_torchopt_runner_bundle"
TARGET = BUNDLE / "torch_hf_optimizations.py"

# (function name, parameter to add, regex to find its signature block)
PATCHES = [
    (
        "_sample_frame_fixed_full_impl",
        "    feedback_lookup_tables: list[torch.Tensor | None] | None = None,\n",
    ),
    (
        "_sample_frame_greedy_full_impl",
        "    feedback_lookup_tables: list[torch.Tensor | None] | None = None,\n",
    ),
]


def patch_function(src: str, func_name: str, param_line: str) -> tuple[str, bool]:
    """Insert param_line just before the `) -> torch.Tensor:` that closes
    func_name's signature, unless it's already present in that signature."""
    sig_pattern = re.compile(
        rf"(def {re.escape(func_name)}\s*\([^)]*?)(\n\)\s*->\s*torch\.Tensor:)",
        re.DOTALL,
    )
    match = sig_pattern.search(src)
    if not match:
        print(f"[patch] WARNING: could not locate signature for {func_name} — "
              f"upstream file may have changed shape; skipping.", file=sys.stderr)
        return src, False

    signature_body = match.group(1)
    if "feedback_lookup_tables" in signature_body:
        print(f"[patch] {func_name}: already patched, skipping.")
        return src, False

    new_signature = signature_body.rstrip("\n") + "\n" + param_line.rstrip("\n") + match.group(2)
    src = src[:match.start()] + new_signature + src[match.end():]
    print(f"[patch] {func_name}: added feedback_lookup_tables parameter.")
    return src, True


def main() -> int:
    if not TARGET.exists():
        print(f"[patch] {TARGET} not found — nothing to patch yet.")
        return 0

    src = TARGET.read_text()
    original = src
    for func_name, param_line in PATCHES:
        src, _ = patch_function(src, func_name, param_line)

    if src != original:
        TARGET.write_text(src)
        print(f"[patch] wrote changes to {TARGET}")
    else:
        print("[patch] no changes needed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
