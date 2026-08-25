#!/usr/bin/env python3
"""Restart-tolerant controller for the fixed-grid knee calibration stage."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def check_gpu(gpu: int, *, max_memory_mib: int = 1024) -> None:
    query = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = {}
    for line in query.splitlines():
        index, used = (int(part.strip()) for part in line.split(","))
        rows[index] = used
    if gpu not in rows:
        raise RuntimeError(f"GPU {gpu} is not visible")
    if rows[gpu] > max_memory_mib:
        raise RuntimeError(f"GPU {gpu} has {rows[gpu]} MiB in use; refusing launch")
    apps = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    if apps:
        uuid = subprocess.check_output(
            [
                "nvidia-smi",
                "--id", str(gpu),
                "--query-gpu=uuid",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        if any(line.startswith(uuid) for line in apps.splitlines()):
            raise RuntimeError(f"GPU {gpu} has a compute process; refusing launch")


def make_seed_manifest(path: Path, start: int, end: int) -> Path:
    if path.exists():
        payload = read_json(path)
        seeds = [int(value) for value in payload.get("seeds", [])]
        expected = list(range(start, end + 1))
        if seeds != expected:
            raise RuntimeError(f"existing seed manifest mismatch: {path}")
        return path
    write_json(path, {"format": "xvla_fixed_timing_calibration_seed_manifest_v1", "start": start, "end": end, "seeds": list(range(start, end + 1))})
    return path


def calibration_command(
    *,
    python: str,
    worktree: Path,
    checkpoint: Path,
    xvla_root: Path,
    output: Path,
    repo_id: Path,
    seed_manifest: Path,
    step: int,
    ood_split: str,
    seed_count: int,
) -> list[str]:
    return [
        python,
        str(worktree / "tools/collect_stackcube_xvla_dagger.py"),
        "--method", "fixed_timing",
        "--timing-step", str(step),
        "--checkpoint", str(checkpoint),
        "--xvla-root", str(xvla_root),
        "--output-dir", str(output),
        "--repo-id", str(repo_id),
        "--seed-manifest", str(seed_manifest),
        "--target", str(seed_count),
        "--consume-all-seeds",
        "--ood-split", ood_split,
        "--flow-steps", "10",
    ]


def completed_collection_evidence(output: Path, *, expected_episodes: int) -> bool:
    """Accept only a completed collection followed by the known teardown abort.

    ManiSkill/SAPIEN can abort during interpreter teardown after all collection
    artifacts have already been flushed.  The abort is acceptable only when an
    independent evidence check finds a complete summary and raw episode ledger
    with the expected denominator.  ``accepted_total`` may be lower because
    recoverability is a scientific calibration outcome, not an artifact-write
    failure; planner or mid-rollout crashes remain failures.
    """
    summary_path = output / "summary.json"
    episodes_path = output / "episodes.jsonl"
    if not summary_path.is_file() or not episodes_path.is_file():
        return False
    try:
        summary = read_json(summary_path)
        rows = [line for line in episodes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError):
        return False
    accepted = int(summary.get("accepted_total", -1))
    return (
        len(rows) == expected_episodes
        and int(summary.get("raw_total", -1)) == expected_episodes
        and 0 <= accepted <= expected_episodes
    )


def run_stackcube_calibration(args: argparse.Namespace) -> None:
    manifest = read_json(args.manifest)
    task = manifest["tasks"]["stackcube"]
    root = Path(args.run_root or manifest["run_root"])
    worktree = Path(args.worktree)
    checkpoint = Path(task["base_checkpoint"])
    xvla_root = Path(args.xvla_root)
    python = args.python
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if not (worktree / "tools/collect_stackcube_xvla_dagger.py").is_file():
        raise FileNotFoundError(worktree / "tools/collect_stackcube_xvla_dagger.py")
    check_gpu(args.gpu)
    root.mkdir(parents=True, exist_ok=True)
    seed_manifest = make_seed_manifest(
        root / f"manifests/calibration_stackcube_{args.seed_start}_{args.seed_end}.json",
        args.seed_start,
        args.seed_end,
    )
    seed_count = args.seed_end - args.seed_start + 1
    state_path = root / "pipeline_state.json"
    state = read_json(state_path) if state_path.exists() else {
        "pipeline_id": manifest["pipeline_id"],
        "task": "stackcube",
        "stage": "calibration",
        "started_at": _now(),
        "anchors": manifest["timing_anchors_env_steps"],
        "completed_steps": [],
        "seed_start": args.seed_start,
        "seed_end": args.seed_end,
    }
    write_json(state_path, state)
    for step in manifest["timing_anchors_env_steps"]:
        step = int(step)
        if step in state["completed_steps"]:
            continue
        output = root / "calibration" / f"step_{step}"
        repo_id = root / "datasets" / f"step_{step}"
        if output.exists() or repo_id.exists():
            raise FileExistsError(f"partial output exists; choose a retry root: step={step}")
        log = root / "logs" / f"calibration_step_{step}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        command = calibration_command(
            python=python,
            worktree=worktree,
            checkpoint=checkpoint,
            xvla_root=xvla_root,
            output=output,
            repo_id=repo_id,
            seed_manifest=seed_manifest,
            step=step,
            ood_split="ood",
            seed_count=seed_count,
        )
        state.update({"stage": f"calibration_step_{step}", "command": command, "updated_at": _now()})
        write_json(state_path, state)
        env = os.environ.copy()
        env.update({"CUDA_VISIBLE_DEVICES": str(args.gpu), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "20", "MKL_NUM_THREADS": "20"})
        with log.open("w", encoding="utf-8") as handle:
            result = subprocess.run(command, cwd=worktree, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
        teardown_abort = result.returncode == -6 and completed_collection_evidence(
            output, expected_episodes=seed_count
        )
        if result.returncode != 0 and not teardown_abort:
            state.update({"stage": f"calibration_step_{step}_failed", "returncode": result.returncode, "updated_at": _now()})
            write_json(state_path, state)
            raise RuntimeError(f"calibration step {step} failed; see {log}")
        if not (output / "summary.json").is_file() or not (output / "episodes.jsonl").is_file():
            raise RuntimeError(f"calibration step {step} missing summary/episodes evidence")
        if teardown_abort:
            state.setdefault("accepted_teardown_aborts", []).append({"step": step, "returncode": -6})
        state["completed_steps"].append(step)
        state.update({"stage": "calibration", "updated_at": _now()})
        write_json(state_path, state)
    state.update({"stage": "calibration_complete", "completed_at": _now()})
    write_json(state_path, state)
    (root / "CALIBRATION_COMPLETE").write_text("complete\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST_DEFAULT)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--seed-start", type=int, default=150000)
    parser.add_argument("--seed-end", type=int, default=150019)
    parser.add_argument("--stage", choices=("stackcube_calibration",), required=True)
    args = parser.parse_args()
    if args.seed_end < args.seed_start:
        raise ValueError("--seed-end must be >= --seed-start")
    if args.stage == "stackcube_calibration":
        run_stackcube_calibration(args)


if __name__ == "__main__":
    main()
