"""Dataset, tokenization, LoRA scope, and checkpoint reload smoke gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoProcessor

from prismatic.vla.action_tokenizer import ActionTokenizer

from .dataset import (
    AirplaneDataset,
    action_token_round_trip,
    compute_action_stats,
    normalize_action,
    validate_lerobot_dataset,
)
from .model import load_openvla, predict_action


def dataset_and_token_smoke(data_dir: Path, base_path: str) -> dict:
    validation = validate_lerobot_dataset(data_dir)
    dataset = AirplaneDataset(data_dir)
    sample = dataset[0]
    processor = AutoProcessor.from_pretrained(base_path, trust_remote_code=True)
    stats = compute_action_stats(data_dir)
    normalized = normalize_action(sample["action"], stats)
    round_trip = action_token_round_trip(normalized, processor.tokenizer, ActionTokenizer(processor.tokenizer))
    if round_trip["max_abs_error"] > 0.02:
        raise ValueError(f"Action-token round-trip error is too large: {round_trip['max_abs_error']}")
    return {"dataset": validation, "round_trip": round_trip}


def reload_forward_smoke(data_dir: Path, base_path: str, checkpoint: Path, device: int) -> dict:
    model, processor = load_openvla(base_path, checkpoint, device)
    if not isinstance(model, PeftModel):
        raise TypeError("Reloaded checkpoint is not a PEFT model")
    unexpected_trainable = [name for name, value in model.named_parameters() if value.requires_grad and "lora_" not in name]
    if unexpected_trainable:
        raise ValueError(f"Non-LoRA parameters are trainable after reload: {unexpected_trainable[:8]}")
    sample = AirplaneDataset(data_dir)[0]
    action, inputs = predict_action(model, processor, sample["image"], device)
    action = np.asarray(action)
    if action.shape != (8,) or not np.isfinite(action).all():
        raise ValueError(f"Expected one finite 8D action, got {action.shape}: {action}")
    return {
        "checkpoint": str(checkpoint),
        "action_shape": list(action.shape),
        "action": action.tolist(),
        "pixel_value_type": type(inputs["pixel_values"]).__name__,
        "trainable_parameter_names": sum(1 for _, value in model.named_parameters() if value.requires_grad),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--base-path", default="openvla/openvla-7b")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    payload = dataset_and_token_smoke(args.data_dir, args.base_path)
    if args.checkpoint is not None:
        payload["reload_forward"] = reload_forward_smoke(args.data_dir, args.base_path, args.checkpoint, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
