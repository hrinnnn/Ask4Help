#!/usr/bin/env python3
"""Make the official Robo-Dopamine LoRA target list configurable and auditable."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    source = args.repo / "train/qwenvl/train/train_qwen.py"
    arguments = args.repo / "train/qwenvl/train/argument.py"
    original_source = source.read_text(encoding="utf-8")
    original_arguments = arguments.read_text(encoding="utf-8")
    target_line = 'target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen 的 attention 线性层'
    replacement = 'target_modules=[item.strip() for item in training_args.lora_target_modules.split(",") if item.strip()],'
    if target_line in original_source:
        source.write_text(original_source.replace(target_line, replacement), encoding="utf-8")
    elif replacement not in original_source:
        raise RuntimeError("official train_qwen.py no longer has the expected LoRA target line")
    field = '    lora_target_modules: str = field(default="q_proj,k_proj,v_proj,o_proj")\n'
    marker = '    lora_dropout: float = field(default=0.0)\n'
    if field not in original_arguments:
        if marker not in original_arguments:
            raise RuntimeError("official argument.py no longer has the expected LoRA argument block")
        arguments.write_text(original_arguments.replace(marker, marker + field), encoding="utf-8")
    manifest = {
        "repo": str(args.repo.resolve()),
        "patched_files": {str(source.relative_to(args.repo)): sha256(source), str(arguments.relative_to(args.repo)): sha256(arguments)},
        "purpose": "Expose official trainer PEFT target_modules for StackCube one-shot LoRA; all preprocessing/pair generation/trainer code stays upstream.",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
