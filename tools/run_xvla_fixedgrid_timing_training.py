#!/usr/bin/env python3
"""Run one restartable fixed-timing X-VLA training job and reload smoke."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DEFAULT = ROOT / "configs/pipelines/xvla_fixedgrid_taskpolicy_knee_v1.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_training_meta(
    *,
    id_meta: Path,
    dataset: Path,
    output: Path,
    task: str,
    anchor: int,
    seed: int,
    expected_budget: int,
) -> Path:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    original = read_json(id_meta)
    original["dataset_name"] = f"panda_{task}_timing_id_replay_step_{anchor}_seed_{seed}"
    write_json(output / "00_id_replay.json", original)
    info = read_json(dataset / "meta/info.json")
    episodes = read_jsonl(dataset / "meta/episodes.jsonl")
    action_count = sum(int(row["length"]) for row in episodes)
    if action_count != expected_budget:
        raise RuntimeError(
            f"{task} step {anchor} selected dataset has {action_count} actions; "
            f"expected {expected_budget}"
        )
    write_json(
        output / "01_new_expert.json",
        {
            "dataset_name": f"panda_{task}_timing_expert_step_{anchor}_seed_{seed}",
            "robot_type": "panda_airplane",
            "root_path": str(dataset.resolve()),
            "chunks_size": int(info["chunks_size"]),
            "data_path": info["data_path"],
            "datalist": episodes,
        },
    )
    return output


def training_command(
    *,
    python: Path,
    repo: Path,
    start: Path,
    meta: Path,
    output: Path,
    train_script: Path,
    steps: int,
    save_interval: int,
    seed: int,
) -> list[str]:
    return [
        str(python),
        str(repo / "tools/xvla_train_with_panda_handler.py"),
        str(train_script.resolve()),
        "--models", str(start),
        "--output_dir", str(output),
        "--train_metas_path", str(meta),
        "--batch_size", "8",
        "--num_actions", "10",
        "--num_views", "2",
        "--action_mode", "auto",
        "--real_action_dim", "8",
        "--max_action_dim", "20",
        "--distributed_backend", "gloo",
        "--gradient_checkpointing",
        "--gradient_accumulation_steps", "4",
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
        "--seed", str(seed),
    ]


def child_env(gpu: int, repo: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "PYTHONPATH": f"{repo}:{repo / 'RLinf'}",
            "OMP_NUM_THREADS": "20",
            "MKL_NUM_THREADS": "20",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "ACCELERATE_MIXED_PRECISION": "bf16",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(29850 + gpu),
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
            "LOCAL_WORLD_SIZE": "1",
        }
    )
    return env


def reload_command(task: str, *, python: Path, repo: Path, checkpoint: Path, xvla_root: Path, output: Path) -> list[str]:
    if task == "stackcube":
        return [
            str(python), str(repo / "tools/evaluate_stackcube_xvla.py"),
            "--checkpoint", str(checkpoint), "--xvla-root", str(xvla_root),
            "--output-dir", str(output), "--episodes", "1", "--seed", "194000",
            "--split", "id", "--max-episode-steps", "10", "--flow-steps", "10",
        ]
    return [
        str(python), str(repo / "tools/evaluate_pick_single_ycb_airplane_xvla.py"),
        "--checkpoint", str(checkpoint), "--xvla-root", str(xvla_root),
        "--output-dir", str(output), "--split", "id", "--episodes", "1",
        "--seed", "195000", "--max-episode-steps", "10", "--flow-steps", "10",
    ]


def run(args: argparse.Namespace) -> None:
    manifest = read_json(args.manifest)
    task_spec = manifest["tasks"][args.task]
    if args.task == "stackcube":
        dataset_root = Path(args.dataset_root)
        id_meta = Path("/data/zhaozhixuan/Ask4Help-airplane-5090/results/xvla_stackcube_v1/manifests/panda_stackcube_id_128.json")
    else:
        dataset_root = Path(args.dataset_root)
        id_meta = Path("/data/zhaozhixuan/Ask4Help-airplane-5090/results/xvla_airplane_v1/manifests/panda_airplane_id.json")
    start = Path(task_spec["base_checkpoint"])
    python = Path(args.python)
    repo = Path(args.repo)
    xvla_root = Path(args.xvla_root)
    if not start.exists() or not id_meta.exists() or not dataset_root.exists():
        raise FileNotFoundError("training asset missing")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    meta = build_training_meta(
        id_meta=id_meta,
        dataset=dataset_root,
        output=args.output / "meta",
        task=args.task,
        anchor=args.anchor,
        seed=args.seed,
        expected_budget=args.expected_budget,
    )
    command = training_command(
        python=python,
        repo=repo,
        start=start,
        meta=meta,
        output=args.output / "train",
        train_script=xvla_root / "train.py",
        steps=args.steps,
        save_interval=args.save_interval,
        seed=args.seed,
    )
    write_json(
        args.output / "training_contract.json",
        {
            "task": args.task,
            "anchor": args.anchor,
            "seed": args.seed,
            "initial_checkpoint": str(start.resolve()),
            "dataset": str(dataset_root.resolve()),
            "expected_expert_budget": args.expected_budget,
            "source_balance": "one ID replay meta + one selected expert meta",
            "temporal_mask": True,
            "steps": args.steps,
            "command": command,
        },
    )
    log_path = args.output / "train.log"
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            ["taskset", "-c", args.cpu_set, *command],
            cwd=xvla_root,
            env=child_env(args.gpu, repo),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    checkpoint = args.output / "train" / f"ckpt-{args.steps}"
    model_file = checkpoint / "model.safetensors"
    if result.returncode != 0 or not model_file.is_file():
        raise RuntimeError(f"training failed rc={result.returncode}; see {log_path}")
    reload_output = args.output / "reload_forward"
    reload_log = args.output / "reload_forward.log"
    with reload_log.open("w", encoding="utf-8") as handle:
        reload_result = subprocess.run(
            ["taskset", "-c", args.cpu_set, *reload_command(args.task, python=python, repo=repo, checkpoint=checkpoint, xvla_root=xvla_root, output=reload_output)],
            cwd=repo,
            env=child_env(args.gpu, repo),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if reload_result.returncode not in (0, -6) or not (reload_output / "summary.json").is_file():
        raise RuntimeError(f"reload smoke failed rc={reload_result.returncode}; see {reload_log}")
    (args.output / "TRAINING_COMPLETE").write_text("complete\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST_DEFAULT)
    parser.add_argument("--task", choices=("stackcube", "airplane"), required=True)
    parser.add_argument("--anchor", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--expected-budget", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--cpu-set", default="0-19")
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--save-interval", type=int, default=500)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
