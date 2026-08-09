#!/usr/bin/env python3
"""Prepare and launch the first X-VLA StackCube ID training stage."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


TASK = "stack the red cube on the green cube"
ENV = Path("/data/zhaozhixuan/envs/xvla_official_5090")
XVLA = Path("/data/zhaozhixuan/X-VLA")
BASE_MODEL = Path(
    "/data/zhaozhixuan/Ask4Help-airplane-5090/results/xvla_airplane_v1/"
    "model_cache/X-VLA-Pt-local"
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_manifest(dataset: Path, output: Path) -> dict:
    info = read_json(dataset / "meta" / "info.json")
    episodes = read_jsonl(dataset / "meta" / "episodes.jsonl")
    if len(episodes) != 128:
        raise ValueError(f"expected 128 StackCube ID episodes, found {len(episodes)}")

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
        "dataset_name": "panda_stackcube_id_128",
        # The existing handler is schema-generic: two embedded images, 9D state, 8D actions.
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
    steps: int,
    save_interval: int,
) -> list[str]:
    return [
        str(ENV / "bin/accelerate"),
        "launch",
        "--num_processes",
        "2",
        "--multi_gpu",
        "--mixed_precision",
        "bf16",
        "--gpu_ids",
        "0,1",
        "train.py",
        "--models",
        str(BASE_MODEL),
        "--output_dir",
        str(output),
        "--train_metas_path",
        str(manifest),
        "--batch_size",
        "8",
        "--num_actions",
        "10",
        "--num_views",
        "2",
        "--action_mode",
        "auto",
        "--real_action_dim",
        "8",
        "--max_action_dim",
        "20",
        "--distributed_backend",
        "gloo",
        "--gradient_checkpointing",
        "--gradient_accumulation_steps",
        "2",
        "--learning_rate",
        "1e-4",
        "--learning_coef",
        "0.1",
        "--weight_decay",
        "0",
        "--betas",
        "0.9",
        "0.95",
        "--max_grad_norm",
        "1.0",
        "--freeze_steps",
        "1000",
        "--warmup_steps",
        "2000",
        "--iters",
        str(steps),
        "--save_interval",
        str(save_interval),
        "--log_interval",
        "20",
        "--seed",
        "5100",
    ]


def training_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0,1",
            "OMP_NUM_THREADS": "20",
            "MKL_NUM_THREADS": "20",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--print-command", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_manifest(args.dataset, args.manifest)
    command = training_command(
        args.manifest, args.output, steps=args.steps, save_interval=args.save_interval
    )
    if args.print_command:
        print(json.dumps(command))


if __name__ == "__main__":
    main()
