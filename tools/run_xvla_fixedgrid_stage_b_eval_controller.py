#!/usr/bin/env python3
"""Evaluate all completed Stage-B timing policies on frozen ID/OOD seeds."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DEFAULT = ROOT / "configs/pipelines/xvla_fixedgrid_taskpolicy_knee_v1.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def check_gpu(gpu: int) -> None:
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    values = {int(i.strip()): int(m.strip()) for i, m in (line.split(",") for line in rows.splitlines())}
    if values.get(gpu, 2048) > 1024:
        raise RuntimeError(f"GPU {gpu} is not idle: {values.get(gpu)} MiB")
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    uuid = subprocess.check_output(
        ["nvidia-smi", "--id", str(gpu), "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    if any(line.startswith(uuid) for line in apps.splitlines() if line.strip()):
        raise RuntimeError(f"GPU {gpu} has a compute process")


def job_command(task: str, checkpoint: Path, output: Path, split: str, seed: int, python: str, repo: Path, xvla_root: Path) -> list[str]:
    if task == "stackcube":
        return [
            python, str(repo / "tools/evaluate_stackcube_xvla.py"),
            "--checkpoint", str(checkpoint), "--xvla-root", str(xvla_root),
            "--output-dir", str(output), "--episodes", "100", "--seed", str(seed),
            "--split", split, "--max-episode-steps", "150", "--flow-steps", "10",
        ]
    return [
        python, str(repo / "tools/evaluate_pick_single_ycb_airplane_xvla.py"),
        "--checkpoint", str(checkpoint), "--xvla-root", str(xvla_root),
        "--output-dir", str(output), "--episodes", "100", "--seed", str(seed),
        "--split", split, "--max-episode-steps", "150", "--flow-steps", "10",
    ]


def summary_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        summary = read_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    return int(summary.get("episodes", -1)) == 100


def run(args: argparse.Namespace) -> None:
    manifest = read_json(args.manifest)
    tasks = ["stackcube", "airplane"] if args.task == "both" else [args.task]
    state_path = args.run_root / "pipeline_state.json"
    state = read_json(state_path) if state_path.exists() else {
        "pipeline_id": manifest["pipeline_id"], "stage": "stage_b_evaluation",
        "started_at": now(), "completed_evals": [],
    }
    args.run_root.mkdir(parents=True, exist_ok=True)
    write_json(state_path, state)
    for task in tasks:
        anchors = [0, 10, 20, 30, 45] if task == "stackcube" else [0, 10, 20, 30]
        for anchor in anchors:
            for training_seed in (17001, 17002, 17003):
                checkpoint = args.training_root / task / f"step_{anchor}" / f"seed_{training_seed}" / "train/ckpt-2500"
                if not (checkpoint / "model.safetensors").is_file():
                    raise FileNotFoundError(checkpoint / "model.safetensors")
                for split, eval_seed in (("id", 153000 if task == "stackcube" else 163000), ("ood", 152000 if task == "stackcube" else 162000)):
                    job_id = f"{task}/step_{anchor}/seed_{training_seed}/{split}"
                    if job_id in state["completed_evals"]:
                        continue
                    output = args.run_root / task / f"step_{anchor}" / f"seed_{training_seed}" / split
                    if output.exists():
                        if summary_complete(output / "summary.json"):
                            state["completed_evals"].append(job_id)
                            write_json(state_path, state)
                            continue
                        raise FileExistsError(output)
                    check_gpu(args.gpu)
                    command = job_command(task, checkpoint, output, split, eval_seed, args.python, args.repo, args.xvla_root)
                    log = args.run_root / "logs" / f"{task}_step_{anchor}_seed_{training_seed}_{split}.log"
                    state.update({"current_eval": job_id, "command": command, "updated_at": now()})
                    write_json(state_path, state)
                    log.parent.mkdir(parents=True, exist_ok=True)
                    env = os.environ.copy()
                    env.update({"CUDA_VISIBLE_DEVICES": str(args.gpu), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
                    with log.open("w", encoding="utf-8") as handle:
                        result = subprocess.run(["taskset", "-c", args.cpu_set, *command], cwd=args.repo, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
                    if result.returncode not in (0, -6) or not summary_complete(output / "summary.json"):
                        state.update({"stage": "stage_b_evaluation_failed", "failed_eval": job_id, "returncode": result.returncode, "updated_at": now()})
                        write_json(state_path, state)
                        raise RuntimeError(f"evaluation failed: {job_id}; see {log}")
                    state["completed_evals"].append(job_id)
                    state.update({"stage": "stage_b_evaluation", "updated_at": now()})
                    write_json(state_path, state)
    state.update({"stage": "stage_b_evaluation_complete", "completed_at": now()})
    write_json(state_path, state)
    (args.run_root / "STAGE_B_EVAL_COMPLETE").write_text("complete\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST_DEFAULT)
    parser.add_argument("--task", choices=("stackcube", "airplane", "both"), default="both")
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--cpu-set", default="0-19")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
