#!/usr/bin/env python3
"""Prepare the X-VLA ID-training manifest for UncoverSpherePlace.

The current X-VLA handler is schema-generic for the two-camera Panda data
used here.  This entry point keeps the task-specific instruction and manifest
construction explicit without changing the model or the existing handler.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TASK = "uncover the sphere and place it in the bowl"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_manifest(dataset: Path, output: Path, *, expected_episodes: int = 128) -> dict:
    info = read_json(dataset / "meta" / "info.json")
    episodes = read_jsonl(dataset / "meta" / "episodes.jsonl")
    if len(episodes) != expected_episodes:
        raise ValueError(
            f"expected {expected_episodes} ID episodes, found {len(episodes)}"
        )

    datalist = []
    for expected_index, row in enumerate(episodes):
        episode_index = int(row["episode_index"])
        length = int(row["length"])
        if episode_index != expected_index:
            raise ValueError(f"episode indices are not contiguous at {expected_index}")
        if length < 10:
            raise ValueError(f"episode {episode_index} has only {length} actions")
        datalist.append({**row, "tasks": [TASK]})

    manifest = {
        "dataset_name": "panda_uncover_sphere_place_id_128",
        # Reuse the validated two-camera, 9D-state, 8D-action Panda handler.
        "robot_type": "panda_airplane",
        "root_path": str(dataset.resolve()),
        "chunks_size": int(info["chunks_size"]),
        "data_path": info["data_path"],
        "datalist": datalist,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def training_command(
    manifest: Path,
    output: Path,
    *,
    base_model: Path,
    python: Path,
    gpu_ids: str,
    steps: int,
    save_interval: int,
) -> list[str]:
    accelerate = python.with_name("accelerate")
    return [
        str(accelerate),
        "launch",
        "--num_processes", "2",
        "--multi_gpu",
        "--mixed_precision", "bf16",
        "--gpu_ids", gpu_ids,
        "train.py",
        "--models", str(base_model),
        "--output_dir", str(output),
        "--train_metas_path", str(manifest),
        "--batch_size", "8",
        "--num_actions", "10",
        "--num_views", "2",
        "--action_mode", "auto",
        "--real_action_dim", "8",
        "--max_action_dim", "20",
        "--distributed_backend", "gloo",
        "--gradient_checkpointing",
        "--gradient_accumulation_steps", "2",
        "--learning_rate", "1e-4",
        "--learning_coef", "0.1",
        "--weight_decay", "0",
        "--betas", "0.9", "0.95",
        "--max_grad_norm", "1.0",
        "--freeze_steps", "1000",
        "--warmup_steps", "2000",
        "--iters", str(steps),
        "--save_interval", str(save_interval),
        "--log_interval", "20",
        "--seed", "5400",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--gpu-ids", default="0,1")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--print-command", action="store_true")
    args = parser.parse_args()

    build_manifest(args.dataset, args.manifest)
    command = training_command(
        args.manifest,
        args.output,
        base_model=args.base_model,
        python=args.python,
        gpu_ids=args.gpu_ids,
        steps=args.steps,
        save_interval=args.save_interval,
    )
    if args.print_command:
        print(json.dumps(command))


if __name__ == "__main__":
    main()
