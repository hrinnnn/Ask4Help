#!/usr/bin/env python3
"""Restart-tolerant fixed-grid calibration controller for X-VLA Airplane."""

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


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def check_gpu(gpu: int, *, max_memory_mib: int = 1024) -> None:
    query = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    rows = {
        int(index.strip()): int(used.strip())
        for index, used in (line.split(",") for line in query.splitlines())
    }
    if gpu not in rows or rows[gpu] > max_memory_mib:
        raise RuntimeError(f"GPU {gpu} is not safely idle: {rows.get(gpu)} MiB")
    apps = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    uuid = subprocess.check_output(
        ["nvidia-smi", "--id", str(gpu), "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    if any(line.startswith(uuid) for line in apps.splitlines() if line.strip()):
        raise RuntimeError(f"GPU {gpu} has a compute process")


def seed_manifest(path: Path, start: int, end: int) -> Path:
    expected = list(range(start, end + 1))
    if path.exists():
        if [int(v) for v in read_json(path).get("seeds", [])] != expected:
            raise RuntimeError(f"seed manifest mismatch: {path}")
        return path
    write_json(path, {"format": "xvla_fixed_timing_airplane_seed_manifest_v1", "seeds": expected})
    return path


def collection_command(
    *,
    python: str,
    worktree: Path,
    checkpoint: Path,
    xvla_root: Path,
    calibration: Path,
    pca_asset: Path,
    output: Path,
    repo_id: Path,
    seed_manifest_path: Path,
    step: int,
    count: int,
) -> list[str]:
    return [
        python,
        str(worktree / "tools/collect_pick_single_ycb_airplane_xvla_dagger.py"),
        "--method", "fixed_timing",
        "--timing-step", str(step),
        "--checkpoint", str(checkpoint),
        "--xvla-root", str(xvla_root),
        "--calibration", str(calibration),
        "--pca-asset", str(pca_asset),
        "--output-dir", str(output),
        "--repo-id", str(repo_id),
        "--seed-manifest", str(seed_manifest_path),
        "--target", str(count),
        "--consume-all-seeds",
        "--only-split", "ood",
        "--flow-steps", "10",
    ]


def complete_evidence(output: Path, *, expected: int) -> bool:
    summary_path = output / "summary.json"
    episodes_path = output / "episodes.jsonl"
    if not summary_path.is_file() or not episodes_path.is_file():
        return False
    try:
        summary = read_json(summary_path)
        rows = [line for line in episodes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        accepted = int(summary.get("accepted_total", -1))
        raw = int(summary.get("raw_total", -1))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return len(rows) == expected and raw == expected and 0 <= accepted <= expected


def run(args: argparse.Namespace) -> None:
    manifest = read_json(args.manifest)
    task = manifest["tasks"]["airplane"]
    root = Path(args.run_root or manifest["run_root"]) / "airplane_calibration"
    worktree = Path(args.worktree)
    python = args.python
    checkpoint = Path(task["base_checkpoint"])
    calibration = Path(task["timing_calibration"])
    pca_asset = Path(task["timing_pca_asset"])
    xvla_root = Path(args.xvla_root)
    for path in (checkpoint, calibration, pca_asset):
        if not path.exists():
            raise FileNotFoundError(path)
    check_gpu(args.gpu)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = seed_manifest(
        root / f"manifests/calibration_airplane_{args.seed_start}_{args.seed_end}.json",
        args.seed_start,
        args.seed_end,
    )
    count = args.seed_end - args.seed_start + 1
    state_path = root / "pipeline_state.json"
    state = read_json(state_path) if state_path.exists() else {
        "pipeline_id": manifest["pipeline_id"],
        "task": "airplane",
        "stage": "calibration",
        "started_at": now(),
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
        command = collection_command(
            python=python,
            worktree=worktree,
            checkpoint=checkpoint,
            xvla_root=xvla_root,
            calibration=calibration,
            pca_asset=pca_asset,
            output=output,
            repo_id=repo_id,
            seed_manifest_path=manifest_path,
            step=step,
            count=count,
        )
        state.update({"stage": f"calibration_step_{step}", "command": command, "updated_at": now()})
        write_json(state_path, state)
        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "20",
            "MKL_NUM_THREADS": "20",
        })
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w", encoding="utf-8") as handle:
            result = subprocess.run(command, cwd=worktree, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
        teardown_abort = result.returncode == -6 and complete_evidence(output, expected=count)
        if result.returncode != 0 and not teardown_abort:
            state.update({"stage": f"calibration_step_{step}_failed", "returncode": result.returncode, "updated_at": now()})
            write_json(state_path, state)
            raise RuntimeError(f"airplane calibration step {step} failed; see {log}")
        if not complete_evidence(output, expected=count):
            raise RuntimeError(f"airplane calibration step {step} missing complete raw evidence")
        if teardown_abort:
            state.setdefault("accepted_teardown_aborts", []).append({"step": step, "returncode": -6})
        state["completed_steps"].append(step)
        state.update({"stage": "calibration", "updated_at": now()})
        write_json(state_path, state)
    state.update({"stage": "calibration_complete", "completed_at": now()})
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
    parser.add_argument("--seed-start", type=int, default=160000)
    parser.add_argument("--seed-end", type=int, default=160019)
    args = parser.parse_args()
    if args.seed_end < args.seed_start:
        raise ValueError("--seed-end must be >= --seed-start")
    run(args)


if __name__ == "__main__":
    main()
