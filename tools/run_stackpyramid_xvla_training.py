#!/usr/bin/env python3
"""Train the official X-VLA model on StackPyramid HDF5 demonstrations.

The collector stores T observations and T-1 real actions. Every actionable
observation is kept as an anchor; the last anchors use repeated storage values
and an explicit temporal mask so padded values never enter the loss.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from tools.xvla_stackcube_temporal_mask import padded_action_chunk


# Keep the training instruction identical to the canonical StackPyramid task
# and evaluator. The red cube is placed next to the green cube.
TASK = "stack the red cube next to the green cube and place the blue cube on top"
REAL_ACTION_DIM = 8
MODEL_ACTION_DIM = 20
ACTION_HORIZON = 10


def _numeric_group_key(name: str) -> tuple[int, str]:
    try:
        return int(name.rsplit("_", 1)[1]), name
    except (IndexError, ValueError):
        return (10**9, name)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class Episode:
    split: str
    h5_path: Path
    group_name: str
    base_rgb: np.ndarray
    wrist_rgb: np.ndarray
    state: np.ndarray
    actions: np.ndarray


class StackPyramidH5Dataset(Dataset):
    """In-memory H5 reader with one item for every real action anchor."""

    def __init__(
        self,
        collection_root: Path,
        split: str,
        *,
        target_episodes: int | None = None,
        horizon: int = ACTION_HORIZON,
    ) -> None:
        self.collection_root = collection_root
        self.split = split
        self.horizon = horizon
        stage_root = collection_root / split
        if not stage_root.is_dir():
            raise FileNotFoundError(stage_root)
        h5_paths = sorted(stage_root.rglob("*.h5"))
        if not h5_paths:
            raise FileNotFoundError(f"no H5 trajectories under {stage_root}")

        summary_paths = sorted(stage_root.glob(f"oracle_summary_{split}.json"))
        summary = _read_json(summary_paths[0]) if summary_paths else None
        if summary is not None and int(summary["strict_successes"]) < (target_episodes or 1):
            raise ValueError(f"{split} oracle summary does not meet target: {summary}")

        episodes: list[Episode] = []
        for h5_path in h5_paths:
            with h5py.File(h5_path, "r") as handle:
                groups = sorted(
                    (name for name in handle.keys() if name.startswith("traj_")),
                    key=_numeric_group_key,
                )
                for group_name in groups:
                    group = handle[group_name]
                    base = np.asarray(group["obs/sensor_data/base_camera/rgb"], dtype=np.uint8)
                    wrist = np.asarray(group["obs/sensor_data/hand_camera/rgb"], dtype=np.uint8)
                    state = np.asarray(group["obs/state"], dtype=np.float32)
                    actions = np.asarray(group["actions"], dtype=np.float32)
                    if base.shape[0] != actions.shape[0] + 1:
                        raise ValueError(f"observation/action boundary mismatch: {h5_path}:{group_name}")
                    if wrist.shape[0] != base.shape[0] or state.shape[0] != base.shape[0]:
                        raise ValueError(f"multi-modal frame mismatch: {h5_path}:{group_name}")
                    if actions.ndim != 2 or actions.shape[1] != REAL_ACTION_DIM:
                        raise ValueError(f"expected [T,8] actions, got {actions.shape}")
                    if state.shape[1] < REAL_ACTION_DIM:
                        raise ValueError(f"state has too few dimensions: {state.shape}")
                    if not np.isfinite(state[:, :REAL_ACTION_DIM]).all() or not np.isfinite(actions).all():
                        raise ValueError(f"non-finite state/action in {h5_path}:{group_name}")
                    episodes.append(
                        Episode(
                            split=split,
                            h5_path=h5_path,
                            group_name=group_name,
                            base_rgb=base,
                            wrist_rgb=wrist,
                            state=state,
                            actions=actions,
                        )
                    )
                    if target_episodes is not None and len(episodes) >= target_episodes:
                        break
            if target_episodes is not None and len(episodes) >= target_episodes:
                break

        if target_episodes is not None and len(episodes) != target_episodes:
            raise ValueError(f"{split}: expected {target_episodes} episodes, found {len(episodes)}")
        if not episodes:
            raise ValueError(f"{split}: no episodes")
        self.episodes = episodes
        self.entries = [(episode_index, anchor) for episode_index, episode in enumerate(episodes) for anchor in range(len(episode.actions))]
        self.report = self._build_report()

    def _build_report(self) -> dict[str, Any]:
        distribution = {str(i): 0 for i in range(1, self.horizon + 1)}
        tail_anchors = 0
        for episode in self.episodes:
            for anchor in range(len(episode.actions)):
                valid = min(self.horizon, len(episode.actions) - anchor)
                distribution[str(valid)] += 1
                tail_anchors += int(valid < self.horizon)
        return {
            "split": self.split,
            "episodes": len(self.episodes),
            "total_anchors": len(self.entries),
            "tail_anchors": tail_anchors,
            "valid_target_count_distribution": distribution,
            "final_observation_valid_targets": 1,
            "action_horizon": self.horizon,
            "real_action_dim": REAL_ACTION_DIM,
            "model_action_dim": MODEL_ACTION_DIM,
        }

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode_index, anchor = self.entries[index]
        episode = self.episodes[episode_index]
        chunk, mask = padded_action_chunk(episode.actions, anchor, self.horizon)
        proprio = np.zeros(MODEL_ACTION_DIM, dtype=np.float32)
        proprio[:REAL_ACTION_DIM] = episode.state[anchor, :REAL_ACTION_DIM]
        return {
            "base": Image.fromarray(episode.base_rgb[anchor]),
            "wrist": Image.fromarray(episode.wrist_rgb[anchor]),
            "proprio": proprio,
            "actions": chunk.astype(np.float32),
            "action_valid_mask": mask,
        }


def collate_fn(batch: list[dict[str, Any]], processor: Any) -> dict[str, torch.Tensor]:
    images = [[row["base"], row["wrist"]] for row in batch]
    processed = processor.encode_image(images)
    language = processor.encode_language([TASK] * len(batch))
    return {
        **processed,
        **language,
        "domain_id": torch.zeros(len(batch), dtype=torch.long),
        "proprio": torch.from_numpy(np.stack([row["proprio"] for row in batch])),
        "action": torch.from_numpy(np.stack([row["actions"] for row in batch])),
        "action_valid_mask": torch.from_numpy(np.stack([row["action_valid_mask"] for row in batch])),
    }


def set_seed(seed: int, process_index: int = 0) -> None:
    value = seed + process_index
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _make_optimizer(model: Any, learning_rate: float, learning_coef: float, weight_decay: float) -> AdamW:
    vlm = list(model.vlm.parameters())
    soft = list(model.transformer.soft_prompt_hub.parameters())
    action = list(model.transformer.action_decoder.parameters()) + list(model.transformer.action_encoder.parameters())
    excluded = {id(param) for param in vlm + soft + action}
    core = [param for param in model.parameters() if id(param) not in excluded]
    return AdamW(
        [
            {"name": "vlm", "params": vlm, "lr": 0.0, "weight_decay": weight_decay},
            {"name": "transformer_core", "params": core, "lr": 0.0, "weight_decay": weight_decay},
            {"name": "soft_prompts", "params": soft, "lr": learning_rate * learning_coef, "weight_decay": weight_decay},
            {"name": "action_heads", "params": action, "lr": learning_rate, "weight_decay": weight_decay},
        ],
        betas=(0.9, 0.95),
    )


def _set_learning_rates(
    optimizer: AdamW,
    step: int,
    total_steps: int,
    learning_rate: float,
    learning_coef: float,
    freeze_steps: int,
    warmup_steps: int,
    freeze_vlm: bool,
) -> None:
    def schedule(base: float) -> float:
        if step < freeze_steps:
            return 0.0
        progress = step - freeze_steps
        if progress < warmup_steps:
            return base * progress / max(1, warmup_steps)
        remain = max(1, total_steps - freeze_steps - warmup_steps)
        ratio = 0.5 * (1.0 + math.cos(math.pi * min(1.0, (progress - warmup_steps) / remain)))
        return base * (0.1 + 0.9 * ratio)

    values = {
        "vlm": schedule(learning_rate * learning_coef),
        "transformer_core": schedule(learning_rate),
        "soft_prompts": schedule(learning_rate * learning_coef),
        "action_heads": schedule(learning_rate),
    }
    if freeze_vlm:
        values["vlm"] = 0.0
    if step < freeze_steps:
        values["soft_prompts"] = learning_rate * learning_coef
        values["action_heads"] = learning_rate
    for group in optimizer.param_groups:
        group["lr"] = values[group["name"]]


def _set_backbone_trainable(model: Any, trainable: bool, freeze_vlm: bool = False) -> None:
    for parameter in model.vlm.parameters():
        parameter.requires_grad = trainable and not freeze_vlm
    for parameter in model.transformer.parameters():
        parameter.requires_grad = True


def masked_flow_loss(model: Any, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    enc = model.forward_vlm(batch["input_ids"], batch["image_input"], batch["image_mask"])
    # Match inference: the flow latent must use the model-facing width before
    # noise is sampled.  Padding after noise would leave dummy dimensions
    # deterministic in training but random during iterative inference.
    proprio_m, action_target = model.action_space.preprocess(batch["proprio"], batch["action"])
    batch_size = action_target.shape[0]
    t = (torch.rand(1, device=action_target.device) + torch.arange(batch_size, device=action_target.device) / batch_size) % (1 - 1e-5)
    action_noisy = torch.randn_like(action_target) * t.view(-1, 1, 1) + action_target * (1 - t).view(-1, 1, 1)
    predicted = model.transformer(
        domain_id=batch["domain_id"],
        action_with_noise=action_noisy,
        t=t,
        proprio=proprio_m,
        **enc,
    )
    element_loss = (predicted[..., :REAL_ACTION_DIM] - action_target[..., :REAL_ACTION_DIM]).square()
    mask = batch["action_valid_mask"].to(device=element_loss.device, dtype=element_loss.dtype).unsqueeze(-1)
    denominator = mask.sum() * REAL_ACTION_DIM
    if float(denominator.item()) <= 0:
        raise ValueError("empty temporal loss mask")
    return (element_loss * mask).sum() / denominator * 100.0


def load_model(model_path: Path, xvla_root: Path, *, dtype: str) -> tuple[Any, Any]:
    sys.path.insert(0, str(xvla_root))
    from models.configuration_xvla import XVLAConfig
    from models.modeling_xvla import XVLA
    from models.processing_xvla import XVLAProcessor

    config = XVLAConfig.from_pretrained(str(model_path))
    config.action_mode = "auto"
    config.real_action_dim = REAL_ACTION_DIM
    config.max_action_dim = MODEL_ACTION_DIM
    config.num_actions = ACTION_HORIZON
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float32
    model = XVLA.from_pretrained(str(model_path), config=config, torch_dtype=torch_dtype)
    processor = XVLAProcessor.from_pretrained(str(model_path))
    return model, processor


def save_checkpoint(model: Any, processor: Any, accelerator: Any, output: Path, step: int) -> Path:
    checkpoint = output / f"ckpt-{step}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    accelerator.unwrap_model(model).save_pretrained(checkpoint, safe_serialization=True)
    processor.save_pretrained(checkpoint)
    (checkpoint / "state.json").write_text(json.dumps({"global_step": step}, indent=2) + "\n", encoding="utf-8")
    return checkpoint


def run_training(args: argparse.Namespace) -> None:
    from accelerate import Accelerator

    accelerator = Accelerator(mixed_precision="bf16" if args.dtype == "bf16" else "no")
    set_seed(args.seed, accelerator.process_index)
    dataset = StackPyramidH5Dataset(
        args.collection_root,
        args.split,
        target_episodes=args.target_episodes,
        horizon=ACTION_HORIZON,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    if accelerator.is_main_process:
        (args.output / "anchor_report.json").write_text(json.dumps(dataset.report, indent=2) + "\n", encoding="utf-8")
        (args.output / "training_config.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n", encoding="utf-8")
    model, processor = load_model(args.model, args.xvla_root, dtype=args.dtype)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda rows: collate_fn(rows, processor),
        drop_last=False,
    )
    optimizer = _make_optimizer(model, args.learning_rate, args.learning_coef, args.weight_decay)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    model.train()
    _set_backbone_trainable(model, False, freeze_vlm=args.freeze_vlm)
    iterator = iter(loader)
    started = time.time()
    total_steps = 2 if args.smoke_only else args.steps
    save_interval = 2 if args.smoke_only else args.save_interval
    for step in range(total_steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = {key: value.to(accelerator.device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        if step == args.freeze_steps:
            _set_backbone_trainable(model, True, freeze_vlm=args.freeze_vlm)
        _set_learning_rates(
            optimizer,
            step,
            total_steps,
            args.learning_rate,
            args.learning_coef,
            args.freeze_steps,
            args.warmup_steps,
            args.freeze_vlm,
        )
        with accelerator.autocast():
            loss = masked_flow_loss(model, batch)
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step = step + 1
        if accelerator.is_main_process and (global_step == 1 or global_step % args.log_interval == 0):
            log = {
                "step": global_step,
                "loss": float(loss.detach().float().item()),
                "lr_action": optimizer.param_groups[-1]["lr"],
                "elapsed_sec": time.time() - started,
            }
            with (args.output / "train.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(log) + "\n")
            print(json.dumps(log), flush=True)
        if accelerator.is_main_process and (global_step % save_interval == 0 or global_step == total_steps):
            save_checkpoint(model, processor, accelerator, args.output, global_step)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.smoke_only:
            reload_model, _reload_processor = load_model(
                args.output / "ckpt-2", args.xvla_root, dtype=args.dtype
            )
            reload_model.to(accelerator.device).eval()
            reload_batch = next(iter(loader))
            reload_batch = {
                key: value.to(accelerator.device) if isinstance(value, torch.Tensor) else value
                for key, value in reload_batch.items()
            }
            with torch.inference_mode(), accelerator.autocast():
                reload_loss = masked_flow_loss(reload_model, reload_batch)
            if not torch.isfinite(reload_loss):
                raise RuntimeError(f"non-finite reload smoke loss: {reload_loss}")
            (args.output / "reload_forward_smoke.json").write_text(
                json.dumps({"finite_loss": float(reload_loss.float()), "batch_size": args.batch_size}) + "\n",
                encoding="utf-8",
            )
            (args.output / "RELOAD_SMOKE_COMPLETE").write_text("complete\n", encoding="utf-8")
        else:
            (args.output / "TRAINING_COMPLETE").write_text("complete\n", encoding="utf-8")
    accelerator.end_training()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--split", default="id", choices=("id", "stage1_ood", "stage2_ood", "stage3_ood"))
    parser.add_argument("--target-episodes", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--learning-coef", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--freeze-steps", type=int, default=1000)
    parser.add_argument(
        "--freeze-vlm",
        action="store_true",
        help="Keep VLM parameters frozen while allowing transformer-core updates after freeze-steps.",
    )
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=5100)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--smoke-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
