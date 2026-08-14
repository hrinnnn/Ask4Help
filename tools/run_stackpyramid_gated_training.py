#!/usr/bin/env python3
"""Train a StackPyramid gated collection with a 1:1 ID/expert sampler."""

from __future__ import annotations

import argparse
import json
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
from torch.utils.data import Dataset, DataLoader, Sampler


REAL_ACTION_DIM = 8
MODEL_ACTION_DIM = 20
ACTION_HORIZON = 10
TASK = "stack the red cube on the green cube and place the blue cube on top"


@dataclass
class Episode:
    source: str
    base: np.ndarray
    wrist: np.ndarray
    state: np.ndarray
    actions: np.ndarray


def _groups(handle: h5py.File) -> list[str]:
    return sorted((name for name in handle if name.startswith("traj_")), key=lambda name: int(name.rsplit("_", 1)[1]))


class SourceH5Dataset(Dataset):
    """In-memory H5 dataset retaining every real action-bearing anchor."""

    def __init__(self, paths: list[tuple[str, Path]], horizon: int = ACTION_HORIZON):
        self.horizon = horizon
        self.episodes: list[Episode] = []
        self.entries: list[tuple[int, int]] = []
        self.source_entries: dict[str, list[int]] = {"id": [], "expert": []}
        for source, path in paths:
            with h5py.File(path, "r") as handle:
                for group_name in _groups(handle):
                    group = handle[group_name]
                    base = np.asarray(group["obs/sensor_data/base_camera/rgb"], dtype=np.uint8)
                    wrist = np.asarray(group["obs/sensor_data/hand_camera/rgb"], dtype=np.uint8)
                    state = np.asarray(group["obs/state"], dtype=np.float32)
                    actions = np.asarray(group["actions"], dtype=np.float32)
                    if base.shape[0] != actions.shape[0] + 1:
                        raise ValueError(f"observation/action boundary mismatch: {path}:{group_name}")
                    if wrist.shape[0] != base.shape[0] or state.shape[0] != base.shape[0]:
                        raise ValueError(f"frame/state mismatch: {path}:{group_name}")
                    if actions.ndim != 2 or actions.shape[1] != REAL_ACTION_DIM:
                        raise ValueError(f"expected [T,8] actions, got {actions.shape}")
                    episode_index = len(self.episodes)
                    self.episodes.append(Episode(source, base, wrist, state, actions))
                    for anchor in range(len(actions)):
                        index = len(self.entries)
                        self.entries.append((episode_index, anchor))
                        self.source_entries[source].append(index)
        if not self.entries or not self.source_entries["id"] or not self.source_entries["expert"]:
            raise ValueError("both ID and expert sources must contain anchors")
        self.report = self._report()

    def _report(self) -> dict[str, Any]:
        counts = {"id": {str(i): 0 for i in range(1, ACTION_HORIZON + 1)}, "expert": {str(i): 0 for i in range(1, ACTION_HORIZON + 1)}}
        for index, anchor in self.entries:
            source = self.episodes[index].source
            valid = min(self.horizon, len(self.episodes[index].actions) - anchor)
            counts[source][str(valid)] += 1
        return {
            "episodes": len(self.episodes),
            "anchors": {source: len(indices) for source, indices in self.source_entries.items()},
            "tail_anchors": {source: sum(counts[source][str(i)] for i in range(1, self.horizon)) for source in counts},
            "valid_target_count_distribution": counts,
            "final_observation_valid_targets": 1,
            "action_horizon": self.horizon,
            "real_action_dim": REAL_ACTION_DIM,
        }

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode_index, anchor = self.entries[index]
        episode = self.episodes[episode_index]
        real = episode.actions[anchor : anchor + self.horizon]
        valid = len(real)
        chunk = np.repeat(episode.actions[-1][None], self.horizon, axis=0)
        chunk[:valid] = real
        proprio = np.zeros(MODEL_ACTION_DIM, dtype=np.float32)
        proprio[:REAL_ACTION_DIM] = episode.state[anchor, :REAL_ACTION_DIM]
        return {
            "base": Image.fromarray(episode.base[anchor]),
            "wrist": Image.fromarray(episode.wrist[anchor]),
            "proprio": proprio,
            "actions": chunk,
            "action_valid_mask": np.arange(self.horizon) < valid,
            "source": episode.source,
        }


class BalancedBatchSampler(Sampler[list[int]]):
    def __init__(self, source_entries: dict[str, list[int]], batch_size: int, seed: int):
        if batch_size < 2 or batch_size % 2:
            raise ValueError("batch size must be an even number >= 2")
        self.source_entries = source_entries
        self.half = batch_size // 2
        self.seed = seed
        self.batches_per_epoch = max(1, max(len(source_entries["id"]), len(source_entries["expert"])) // self.half)

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self):
        rng = random.Random(self.seed)
        self.seed += 1
        for _ in range(self.batches_per_epoch):
            yield [
                *(rng.choices(self.source_entries["id"], k=self.half)),
                *(rng.choices(self.source_entries["expert"], k=self.half)),
            ]


def collate(rows: list[dict[str, Any]], processor: Any) -> dict[str, torch.Tensor]:
    images = [[row["base"], row["wrist"]] for row in rows]
    encoded = {**processor.encode_image(images), **processor.encode_language([TASK] * len(rows))}
    return {
        **encoded,
        "domain_id": torch.zeros(len(rows), dtype=torch.long),
        "proprio": torch.from_numpy(np.stack([row["proprio"] for row in rows])),
        "action": torch.from_numpy(np.stack([row["actions"] for row in rows])),
        "action_valid_mask": torch.from_numpy(np.stack([row["action_valid_mask"] for row in rows])),
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--id-h5", type=Path, required=True)
    parser.add_argument("--expert-h5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=8200)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root), str(args.xvla_root)]
    from run_stackpyramid_xvla_training import _make_optimizer, _set_backbone_trainable, _set_learning_rates, load_model, masked_flow_loss
    from models.processing_xvla import XVLAProcessor

    _set_seed(args.seed)
    dataset = SourceH5Dataset([("id", args.id_h5), ("expert", args.expert_h5)])
    sampler = BalancedBatchSampler(dataset.source_entries, args.batch_size, args.seed + 1)
    model, processor = load_model(args.model, args.xvla_root, dtype="bf16")
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0, collate_fn=lambda rows: collate(rows, processor))
    optimizer = _make_optimizer(model, args.learning_rate, 0.1, 0.0)
    device = torch.device("cuda")
    model.to(device)
    model.train()
    _set_backbone_trainable(model, False)
    (args.output / "training_config.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    (args.output / "anchor_report.json").write_text(json.dumps(dataset.report, indent=2) + "\n")
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
        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        _set_learning_rates(optimizer, step, total_steps, args.learning_rate, 0.1, 0, 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = masked_flow_loss(model, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step = step + 1
        if global_step == 1 or global_step % 20 == 0:
            row = {"step": global_step, "loss": float(loss.detach().float()), "elapsed_sec": time.time() - started}
            with (args.output / "train.jsonl").open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if global_step % save_interval == 0 or global_step == total_steps:
            checkpoint = args.output / f"ckpt-{global_step}"
            checkpoint.mkdir()
            model.save_pretrained(checkpoint, safe_serialization=True)
            processor.save_pretrained(checkpoint)
            (checkpoint / "state.json").write_text(json.dumps({"global_step": global_step}) + "\n")
    if args.smoke_only:
        reload_model, reload_processor = load_model(args.output / "ckpt-2", args.xvla_root, dtype="bf16")
        reload_model.to(device).eval()
        reload_batch = next(iter(loader))
        reload_batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in reload_batch.items()}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            reload_loss = masked_flow_loss(reload_model, reload_batch)
        if not torch.isfinite(reload_loss):
            raise RuntimeError(f"non-finite reload smoke loss: {reload_loss}")
        (args.output / "reload_forward_smoke.json").write_text(json.dumps({"finite_loss": float(reload_loss.float()), "batch_size": args.batch_size}) + "\n")
        (args.output / "RELOAD_SMOKE_COMPLETE").write_text("complete\n")
    else:
        (args.output / "TRAINING_COMPLETE").write_text("complete\n")


if __name__ == "__main__":
    main()
