#!/usr/bin/env python3
"""Run restart-tolerant final evaluation for the OpenDrawer matched updates."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


METHODS = (
    "offline_oracle_sft_10000_from_id2000_retry2",
    "failure_recovery_sft_10000_from_id2000_retry1",
    "robot_gated_sft_10000_from_id2000_retry1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, default=10000)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--cpu-set", required=True)
    parser.add_argument("--poll-seconds", type=int, default=1800)
    return parser.parse_args()


def checkpoint_path(args: argparse.Namespace, method: str) -> Path:
    return (
        args.result_root.parent
        / method
        / "checkpoints"
        / f"global_step_{args.checkpoint_step}"
    )


def wait_for_checkpoint(args: argparse.Namespace, method: str) -> Path:
    checkpoint = checkpoint_path(args, method)
    while not (checkpoint / "actor/model_state_dict/full_weights.pt").is_file():
        print(f"waiting for {method}: {checkpoint}", flush=True)
        time.sleep(args.poll_seconds)
    return checkpoint


def valid_split(output: Path, episodes: int) -> bool:
    summary = output / "summary.json"
    videos = output / "videos"
    if not summary.is_file() or not videos.is_dir():
        return False
    try:
        payload = json.loads(summary.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return payload.get("episodes") == episodes and len(list(videos.glob("*.mp4"))) == episodes


def main() -> None:
    args = parse_args()
    args.result_root.mkdir(parents=True, exist_ok=True)
    state_path = args.result_root / "pipeline_state.json"
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "PYTHONPATH": f"{args.root}:{args.root / 'RLinf'}",
            "ASK4HELP_RLINF_ROOT": str(args.root / "RLinf"),
            "OMP_NUM_THREADS": "20",
            "MKL_NUM_THREADS": "20",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    methods_dir = args.result_root.parent
    for method in METHODS:
        checkpoint = wait_for_checkpoint(args, method)
        output = args.result_root / method
        if (output / "EVAL_COMPLETE").is_file() and all(
            valid_split(output / split, 100)
            for split in ("id", "handle_ood", "grasp_ood", "goal_ood")
        ):
            continue
        if output.exists() and any(output.iterdir()):
            raise RuntimeError(f"refusing to overwrite incomplete output: {output}")
        output.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps({"stage": method, "status": "running", "gpu": args.gpu}, indent=2) + "\n",
            encoding="utf-8",
        )
        command = [
            str(args.python),
            "-u",
            str(args.wrapper),
            "--python",
            str(args.python),
            "--evaluator",
            str(args.evaluator),
            "--checkpoint",
            str(checkpoint),
            "--pi05-base",
            str(args.root / "results/model_cache/pi05_base_pytorch_v1"),
            "--norm-stats",
            str(args.root / "results/id_policy_training_v1/norm_stats_open_drawer_id_raw_v1"),
            "--output-root",
            str(output),
            "--episodes",
            "100",
            "--execute-horizon",
            "5",
            "--max-episode-steps",
            "400",
            "--state-file",
            str(state_path),
        ]
        log_path = args.result_root / "logs" / f"{method}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            completed = subprocess.run(
                ["taskset", "-c", args.cpu_set, *command],
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0 or not all(
            valid_split(output / split, 100)
            for split in ("id", "handle_ood", "grasp_ood", "goal_ood")
        ):
            state_path.write_text(
                json.dumps(
                    {"stage": method, "status": "failed", "returncode": completed.returncode},
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            raise RuntimeError(f"final evaluation failed for {method}: {completed.returncode}")
        state_path.write_text(
            json.dumps({"stage": method, "status": "completed", "gpu": args.gpu}, indent=2) + "\n",
            encoding="utf-8",
        )

    comparison: dict[str, object] = {"checkpoint_step": args.checkpoint_step, "methods": {}}
    for method in METHODS:
        method_output = args.result_root / method
        rows: dict[str, object] = {}
        for split in ("id", "handle_ood", "grasp_ood", "goal_ood"):
            payload = json.loads((method_output / split / "summary.json").read_text(encoding="utf-8"))
            rows[split] = {
                key: payload.get(key)
                for key in (
                    "episodes",
                    "successes",
                    "success_rate",
                    "drawer_opened_rate",
                    "grasp_rate",
                    "lift_rate",
                    "in_target_rate",
                    "execute_horizon",
                    "max_episode_steps",
                    "seed_start",
                )
            }
        comparison["methods"][method] = rows
    (args.result_root / "comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )
    lines = ["# OpenDrawer matched-update final evaluation", "", f"Checkpoint step: {args.checkpoint_step}", "", "| Method | ID | Handle OOD | Grasp OOD | Goal OOD |", "|---|---:|---:|---:|---:|"]
    for method in METHODS:
        rows = comparison["methods"][method]
        rates = [f"{100.0 * float(rows[split]['success_rate']):.1f}%" for split in ("id", "handle_ood", "grasp_ood", "goal_ood")]
        lines.append(f"| {method} | " + " | ".join(rates) + " |")
    (args.result_root / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.result_root / "FINAL_EVAL_COMPLETE").write_text(
        "All matched-update methods have four verified 100-episode split evaluations.\n",
        encoding="utf-8",
    )
    state_path.write_text(
        json.dumps({"stage": "all_methods", "status": "completed"}, indent=2) + "\n",
        encoding="utf-8",
    )
    print("OPEN_DRAWER_MATCHED_FINAL_EVAL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
