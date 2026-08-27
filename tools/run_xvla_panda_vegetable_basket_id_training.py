#!/usr/bin/env python3
"""Train a fresh Panda X-VLA ID policy with the active-block temporal mask."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from io import BytesIO
from pathlib import Path

import h5py
import numpy as np
import torch
from accelerate import Accelerator
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset


ACTION_HORIZON = 30
MODEL_ACTION_DIM = 20
ACTIVE_ACTION_DIM = 10
TASK = "put the vegetable into the yellow basket"


class PandaVegetableDataset(IterableDataset):
    """Yield every real observation anchor and a masked 30-step target."""

    def __init__(self, root: Path, *, training: bool, episodes: int = 128):
        self.root = root
        self.training = training
        self.paths = sorted((root / "data").glob("episode_*.h5"))[:episodes]
        if len(self.paths) != episodes:
            raise ValueError(f"expected {episodes} episodes, found {len(self.paths)}")

    @staticmethod
    def _image(value) -> Image.Image:
        value = np.asarray(value)
        if value.ndim == 3:
            return Image.fromarray(value.astype(np.uint8)).convert("RGB")
        return Image.open(BytesIO(bytes(value))).convert("RGB")

    def __iter__(self):
        paths = list(self.paths)
        if self.training:
            random.shuffle(paths)
        for path in paths:
            with h5py.File(path, "r") as h5:
                images = h5["images"]
                proprio = np.asarray(h5["proprio"], dtype=np.float32)
                actions = np.asarray(h5["abs_action_6d"], dtype=np.float32)
                for index in range(len(actions)):
                    valid = min(ACTION_HORIZON, len(actions) - index)
                    chunk = np.repeat(actions[-1][None], ACTION_HORIZON, axis=0)
                    chunk[:valid] = actions[index:index + valid]
                    target = np.zeros((ACTION_HORIZON, MODEL_ACTION_DIM), dtype=np.float32)
                    target[:, :ACTIVE_ACTION_DIM] = chunk
                    mask = np.zeros((ACTION_HORIZON, MODEL_ACTION_DIM), dtype=bool)
                    mask[:valid, :ACTIVE_ACTION_DIM] = True
                    yield {
                        "image": self._image(images[index]),
                        "proprio": np.concatenate([proprio[index], np.zeros(10, dtype=np.float32)]),
                        "action": target,
                        "action_valid_mask": mask,
                    }


def build_collate(processor, domain_id: int):
    def collate(rows):
        encoded = processor.encode_image([[row["image"]] for row in rows])
        language = processor.encode_language([TASK] * len(rows))
        return {
            **encoded,
            "input_ids": language["input_ids"],
            "domain_id": torch.full((len(rows),), domain_id, dtype=torch.long),
            "proprio": torch.from_numpy(np.stack([row["proprio"] for row in rows])),
            "action": torch.from_numpy(np.stack([row["action"] for row in rows])),
            "action_valid_mask": torch.from_numpy(np.stack([row["action_valid_mask"] for row in rows])),
        }
    return collate


def load_model(base: Path, xvla_root: Path):
    sys.path.insert(0, str(xvla_root))
    from models.modeling_xvla import XVLA
    from models.processing_xvla import XVLAProcessor

    model = XVLA.from_pretrained(str(base), torch_dtype=torch.bfloat16)
    if model.num_actions != ACTION_HORIZON:
        raise ValueError(f"expected num_actions={ACTION_HORIZON}, found {model.num_actions}")
    return model, XVLAProcessor.from_pretrained(str(base))


def build_optimizer(model, learning_rate: float, learning_coef: float):
    vlm = list(model.vlm.parameters())
    soft = list(model.transformer.soft_prompt_hub.parameters())
    action = list(model.transformer.action_decoder.parameters()) + list(model.transformer.action_encoder.parameters())
    excluded = {id(parameter) for parameter in vlm + soft + action}
    core = [parameter for parameter in model.parameters() if id(parameter) not in excluded]
    return AdamW(
        [
            {"name": "vlm", "params": vlm, "lr": 0.0},
            {"name": "transformer_core", "params": core, "lr": 0.0},
            {"name": "soft_prompts", "params": soft, "lr": learning_rate * learning_coef},
            {"name": "action_heads", "params": action, "lr": learning_rate},
        ],
        betas=(0.9, 0.95),
    )


def update_lrs(optimizer, step: int, args):
    def schedule(base):
        if step < args.freeze_steps:
            return 0.0
        progress = step - args.freeze_steps
        if progress < args.warmup_steps:
            return base * progress / max(1, args.warmup_steps)
        remain = max(1, args.steps - args.freeze_steps - args.warmup_steps)
        ratio = 0.5 * (1 + math.cos(math.pi * min(1.0, (progress - args.warmup_steps) / remain)))
        return base * (0.1 + 0.9 * ratio)

    values = {
        "vlm": args.learning_rate * args.learning_coef,
        "transformer_core": args.learning_rate,
        "soft_prompts": args.learning_rate * args.learning_coef,
        "action_heads": args.learning_rate,
    }
    for group in optimizer.param_groups:
        group["lr"] = schedule(values[group["name"]]) if group["name"] not in {"vlm", "transformer_core"} else 0.0


def masked_ee6d_loss(model, batch):
    enc = model.forward_vlm(batch["input_ids"], batch["image_input"], batch["image_mask"])
    target = batch["action"]
    batch_size = target.shape[0]
    t = (torch.rand(1, device=target.device) + torch.arange(batch_size, device=target.device) / batch_size) % (1 - 1e-5)
    noisy = torch.randn_like(target) * t.view(-1, 1, 1) + target * (1 - t).view(-1, 1, 1)
    proprio, noisy = model.action_space.preprocess(batch["proprio"], noisy)
    pred = model.transformer(
        domain_id=batch["domain_id"],
        action_with_noise=noisy,
        t=t,
        proprio=proprio,
        **enc,
    )
    mask = batch["action_valid_mask"].to(dtype=pred.dtype)[..., :ACTIVE_ACTION_DIM]

    def mean(value, weight):
        return (value * weight).sum() / weight.sum().clamp_min(1.0)

    position = mean((pred[..., :3] - target[..., :3]) ** 2, mask[..., :3]) * 500.0
    rotation = mean((pred[..., 3:9] - target[..., 3:9]) ** 2, mask[..., 3:9]) * 10.0
    gripper = mean(
        torch.nn.functional.binary_cross_entropy_with_logits(
            pred[..., 9], target[..., 9], reduction="none"
        ),
        mask[..., 9],
    )
    return position + rotation + gripper


def save_checkpoint(model, processor, accelerator, output: Path, step: int):
    path = output / f"ckpt-{step}"
    path.mkdir(parents=True, exist_ok=True)
    accelerator.unwrap_model(model).save_pretrained(path, safe_serialization=True)
    processor.save_pretrained(path)
    (path / "state.json").write_text(json.dumps({"global_step": step}, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--learning-coef", type=float, default=0.1)
    parser.add_argument("--freeze-steps", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=96300)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--domain-id", type=int, default=20)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    model, processor = load_model(args.base_model, args.xvla_root)
    dataset = PandaVegetableDataset(args.dataset, training=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0, collate_fn=build_collate(processor, args.domain_id))
    optimizer = build_optimizer(model, args.learning_rate, args.learning_coef)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    model.train()
    world_size = accelerator.num_processes
    effective_batch = args.batch_size * args.gradient_accumulation_steps * world_size
    config = {
        **vars(args),
        "base_model": str(args.base_model),
        "dataset": str(args.dataset),
        "world_size": world_size,
        "effective_global_batch_size": effective_batch,
        "active_action_dim": ACTIVE_ACTION_DIM,
        "model_action_dim": MODEL_ACTION_DIM,
        "action_horizon": ACTION_HORIZON,
        "domain_id": args.domain_id,
        "ood_included": False,
    }
    (args.output / "training_config.json").write_text(json.dumps(config, default=str, indent=2) + "\n", encoding="utf-8")
    (args.output / "ID_TRAINING_STARTED").write_text("started\n", encoding="utf-8")
    target_steps = 2 if args.smoke_only else args.steps
    iterator = iter(loader)
    optimizer_steps = 0
    started = time.time()
    while optimizer_steps < target_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = {key: value.to(accelerator.device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        update_lrs(optimizer, optimizer_steps, args)
        with accelerator.accumulate(model), accelerator.autocast():
            loss = masked_ee6d_loss(model, batch)
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            if accelerator.is_main_process:
                row = {
                    "step": optimizer_steps,
                    "loss": float(loss.detach().float()),
                    "valid_action_ratio": float(batch["action_valid_mask"].float().mean()),
                    "elapsed_sec": time.time() - started,
                }
                with (args.output / "train.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
                if optimizer_steps % (2 if args.smoke_only else args.save_interval) == 0 or optimizer_steps == target_steps:
                    save_checkpoint(model, processor, accelerator, args.output, optimizer_steps)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.smoke_only:
            reload_model, _ = load_model(args.output / "ckpt-2", args.xvla_root)
            reload_model.to(accelerator.device).eval()
            reload_batch = next(iter(loader))
            reload_batch = {key: value.to(accelerator.device) if isinstance(value, torch.Tensor) else value for key, value in reload_batch.items()}
            with torch.inference_mode(), accelerator.autocast():
                reload_loss = masked_ee6d_loss(reload_model, reload_batch)
            if not torch.isfinite(reload_loss):
                raise RuntimeError("non-finite reload loss")
            (args.output / "reload_forward_smoke.json").write_text(json.dumps({"finite_loss": float(reload_loss), "valid_action_ratio": float(reload_batch["action_valid_mask"].float().mean())}) + "\n", encoding="utf-8")
            (args.output / "RELOAD_SMOKE_COMPLETE").write_text("complete\n", encoding="utf-8")
        else:
            (args.output / "TRAINING_COMPLETE").write_text("complete\n", encoding="utf-8")
    accelerator.end_training()


if __name__ == "__main__":
    main()
