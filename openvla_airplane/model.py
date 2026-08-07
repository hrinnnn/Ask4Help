"""OpenVLA loading and prompt utilities shared by training and evaluation."""

from __future__ import annotations

from pathlib import Path
import json

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForVision2Seq, AutoProcessor

from prismatic.models.backbones.llm.prompting import PurePromptBuilder

from .dataset import AIRPLANE_INSTRUCTION
from .utils import move_pixel_values


def load_openvla(base_path: str, checkpoint: Path | None, device: int = 0):
    processor_path = checkpoint / "processor" if checkpoint is not None else base_path
    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
    base = AutoModelForVision2Seq.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if checkpoint is not None:
        adapter = checkpoint / "adapter"
        if not adapter.exists():
            raise FileNotFoundError(f"Missing LoRA adapter: {adapter}")
        model = PeftModel.from_pretrained(base, adapter)
    else:
        model = base
    model = model.to(device).eval()
    if checkpoint is not None and (checkpoint / "action_stats.json").exists():
        stats = json.loads((checkpoint / "action_stats.json").read_text())
        norm_stats = {"airplane": {"action": {"q01": stats["q01"], "q99": stats["q99"]}}}
        model.norm_stats = norm_stats
        model.get_base_model().norm_stats = norm_stats
    else:
        raise FileNotFoundError("A checkpoint action_stats.json is required for 8D action decoding")
    return model, processor


def build_prompt(tokenizer, instruction: str = AIRPLANE_INSTRUCTION) -> torch.Tensor:
    builder = PurePromptBuilder("openvla")
    builder.add_turn("human", f"What action should the robot take to {instruction}?")
    ids = tokenizer(builder.get_prompt(), truncation=True, return_tensors="pt").input_ids
    if not torch.all(ids[:, -1] == 29871):
        ids = torch.cat((ids, torch.tensor([[29871]], dtype=torch.long)), dim=1)
    return ids


def prepare_inference_inputs(model, processor, image: Image.Image, device: int = 0) -> dict:
    input_ids = build_prompt(processor.tokenizer).to(device)
    pixel_values = processor.image_processor.apply_transform(image)
    if isinstance(pixel_values, dict):
        pixel_values = {key: value.unsqueeze(0) for key, value in pixel_values.items()}
    else:
        pixel_values = pixel_values.unsqueeze(0)
    pixel_values = move_pixel_values(pixel_values, device)
    return {"input_ids": input_ids, "pixel_values": pixel_values}


@torch.inference_mode()
def predict_action(model, processor, image: Image.Image, device: int = 0) -> tuple:
    inputs = prepare_inference_inputs(model, processor, image, device)
    # Torch 2.2's cached single-token linear path raises SIGFPE on H20. Greedy
    # decoding without KV cache is numerically equivalent and remains stable.
    action = model.predict_action(**inputs, do_sample=False, use_cache=False)
    return action, inputs
