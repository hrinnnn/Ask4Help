#!/usr/bin/env python3
"""Train one Panda DAgger branch with 1:1 ID and expert-source sampling."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from io import BytesIO
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.run_xvla_panda_vegetable_basket_id_training import (  # noqa: E402
    ACTION_HORIZON,
    MODEL_ACTION_DIM,
    build_collate,
    build_optimizer,
    load_model,
    masked_ee6d_loss,
    save_checkpoint,
    update_lrs,
)


class MixedPandaDataset(IterableDataset):
    """Yield alternating ID and expert-suffix anchors forever."""

    def __init__(self, id_root: Path, expert_root: Path, *, seed: int):
        self.id_paths = sorted((id_root / "data").glob("episode_*.h5"))
        if not self.id_paths:
            raise ValueError(f"no ID episodes under {id_root / 'data'}")
        accepted_path = expert_root / "accepted_episodes.jsonl"
        if not accepted_path.is_file():
            raise FileNotFoundError(accepted_path)
        rows = [json.loads(line) for line in accepted_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.expert_items: list[tuple[Path, int]] = []
        for row in rows:
            path = Path(row["data_path"])
            with h5py.File(path, "r") as h5:
                length = len(h5["abs_action_6d"])
            start = int(row["expert_control_start"])
            if start < 0 or start >= length:
                raise ValueError(f"invalid expert suffix boundary in {path}: {start}/{length}")
            self.expert_items.extend((path, index) for index in range(start, length))
        if not self.expert_items:
            raise ValueError("expert collection has no nonempty suffix anchors")
        self.seed = seed

    @staticmethod
    def _image(value) -> Image.Image:
        value = np.asarray(value)
        if value.ndim == 3:
            return Image.fromarray(value.astype(np.uint8)).convert("RGB")
        return Image.open(BytesIO(bytes(value))).convert("RGB")

    @staticmethod
    def _row(path: Path, index: int) -> dict:
        with h5py.File(path, "r") as h5:
            images = h5["images"]
            proprio = np.asarray(h5["proprio"][index], dtype=np.float32).reshape(-1)
            actions = np.asarray(h5["abs_action_6d"], dtype=np.float32)
            if proprio.size < 10 or actions.ndim != 2 or actions.shape[1] < 10:
                raise ValueError(f"invalid X-VLA action contract in {path}")
            valid = min(ACTION_HORIZON, len(actions) - index)
            if valid < 1:
                raise ValueError(f"anchor outside action range: {path} {index}")
            chunk = np.repeat(actions[-1][None, :MODEL_ACTION_DIM], ACTION_HORIZON, axis=0)
            chunk[:valid] = actions[index:index + valid, :MODEL_ACTION_DIM]
            target = np.zeros((ACTION_HORIZON, MODEL_ACTION_DIM), dtype=np.float32)
            target[:, :10] = chunk[:, :10]
            mask = np.zeros((ACTION_HORIZON, MODEL_ACTION_DIM), dtype=bool)
            mask[:valid, :10] = True
            return {
                "image": MixedPandaDataset._image(images[index]),
                "proprio": np.concatenate([proprio[:10], np.zeros(10, dtype=np.float32)]),
                "action": target,
                "action_valid_mask": mask,
            }

    def __iter__(self):
        rng = random.Random(self.seed + 1009 * int(torch.utils.data.get_worker_info().id if torch.utils.data.get_worker_info() else 0))
        id_items = [(path, index) for path in self.id_paths for index in range(self._length(path))]
        expert_items = list(self.expert_items)
        while True:
            rng.shuffle(id_items)
            rng.shuffle(expert_items)
            for id_item, expert_item in zip(id_items, expert_items):
                yield self._row(*id_item)
                yield self._row(*expert_item)
            longer = id_items if len(id_items) > len(expert_items) else expert_items
            shorter = expert_items if len(id_items) > len(expert_items) else id_items
            for item in longer[len(shorter):]:
                yield self._row(*item)

    @staticmethod
    def _length(path: Path) -> int:
        with h5py.File(path, "r") as h5:
            return len(h5["abs_action_6d"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--id-dataset", type=Path, required=True)
    parser.add_argument("--expert-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--learning-coef", type=float, default=0.1)
    parser.add_argument("--freeze-steps", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--domain-id", type=int, default=20)
    parser.add_argument("--seed", type=int, default=96500)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--distributed-backend", choices=("nccl", "gloo"), default="nccl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[InitProcessGroupKwargs(backend=args.distributed_backend)],
    )
    if accelerator.is_main_process:
        args.output.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    model, processor = load_model(args.base_model, args.xvla_root)
    dataset = MixedPandaDataset(args.id_dataset, args.expert_dataset, seed=args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
        collate_fn=build_collate(processor, args.domain_id),
    )
    optimizer = build_optimizer(model, args.learning_rate, args.learning_coef)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    model.train()
    total_steps = 2 if args.smoke_only else args.steps
    config = {
        **vars(args),
        "base_model": str(args.base_model),
        "id_dataset": str(args.id_dataset),
        "expert_dataset": str(args.expert_dataset),
        "world_size": accelerator.num_processes,
        "effective_global_batch_size": args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes,
        "source_balance": "1:1 alternating ID and expert suffix anchors",
        "action_horizon": ACTION_HORIZON,
        "active_action_dim": 10,
        "temporal_mask": "real timesteps only; repeated final action is excluded",
    }
    (args.output / "training_config.json").write_text(json.dumps(config, default=str, indent=2) + "\n", encoding="utf-8")
    (args.output / "MIXED_TRAINING_STARTED").write_text("started\n", encoding="utf-8")
    iterator = iter(loader)
    started = time.time()
    optimizer_steps = 0
    while optimizer_steps < total_steps:
        batch = next(iterator)
        batch = {
            key: value.to(accelerator.device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
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
                interval = 2 if args.smoke_only else args.save_interval
                if optimizer_steps == total_steps or optimizer_steps % interval == 0:
                    save_checkpoint(model, processor, accelerator, args.output, optimizer_steps)
    accelerator.wait_for_everyone()
    reload_batch = next(iter(loader)) if args.smoke_only else None
    if args.smoke_only:
        accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.smoke_only:
            assert reload_batch is not None
            reload_model, _ = load_model(args.output / "ckpt-2", args.xvla_root)
            reload_model.to(accelerator.device).eval()
            reload_batch = {
                key: value.to(accelerator.device) if isinstance(value, torch.Tensor) else value
                for key, value in reload_batch.items()
            }
            with torch.inference_mode(), accelerator.autocast():
                reload_loss = masked_ee6d_loss(reload_model, reload_batch)
            if not torch.isfinite(reload_loss):
                raise RuntimeError("non-finite reload loss")
            (args.output / "RELOAD_SMOKE_COMPLETE").write_text("complete\n", encoding="utf-8")
        else:
            (args.output / "TRAINING_COMPLETE").write_text("complete\n", encoding="utf-8")
    accelerator.end_training()


if __name__ == "__main__":
    main()
