#!/usr/bin/env python3
"""Durable Stage-C passive gate audit for the fixed-grid knee protocol.

Stage-C starts only after the matched-budget Stage-B utility marker exists.
It evaluates frozen validation-ID and held-out OOD streams, calibrates
thresholds on validation ID only, and writes gate-to-knee timing evidence.
Gated dataset collection/training remains a separate later stage and is not
silently treated as complete by this controller.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STAGE_B_ROOT_DEFAULT = Path(
    "/data/zhaozhixuan/Ask4Help-airplane-5090/results/"
    "xvla_fixedgrid_taskpolicy_knee_v1/stage_b_total_supervisor_v1"
)
EXPECTED_EPISODES = 50
METHODS = ["input_pca", "bridge_pca", "action_pca", "diffdagger", "failure_recovery"]


class WaitForResource(RuntimeError):
    """The controller should be relaunched after its sleep window."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def summary_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return int(read_json(path).get("episodes", -1)) == EXPECTED_EPISODES
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def asset_has_layer(path: Path, layer: str, python: str) -> bool:
    if not path.is_file():
        return False
    probe = subprocess.run(
        [
            python,
            "-c",
            (
                "import sys, torch; "
                "p=torch.load(sys.argv[1], map_location='cpu', weights_only=False); "
                "raise SystemExit(0 if sys.argv[2] in p.get('layers', {}) else 1)"
            ),
            str(path),
            layer,
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return probe.returncode == 0


def gpu_idle(gpu: int) -> bool:
    try:
        rows = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        values = {int(i.strip()): int(m.strip()) for i, m in (line.split(",") for line in rows.splitlines())}
        if values.get(gpu, 2048) > 1024:
            return False
        apps = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        uuid = subprocess.check_output(
            ["nvidia-smi", "--id", str(gpu), "--query-gpu=uuid", "--format=csv,noheader"],
            text=True,
        ).strip()
        return not any(line.startswith(uuid) for line in apps.splitlines() if line.strip())
    except (OSError, subprocess.CalledProcessError, ValueError):
        return False


def ensure_airplane_assets(
    *,
    args: argparse.Namespace,
    state: dict[str, Any],
    state_path: Path,
) -> Path:
    """Build a fresh ID-only asset when the legacy asset lacks input pooling."""

    if asset_has_layer(args.airplane_internal_assets, "vlm_input_pool", args.python):
        return args.airplane_internal_assets
    output_dir = args.run_root / "assets_airplane_input_pool"
    output_asset = output_dir / "multilayer_detector_assets.pt"
    marker = output_dir / "ASSETS_COMPLETE"
    if marker.is_file() and asset_has_layer(output_asset, "vlm_input_pool", args.python):
        return output_asset
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"partial Airplane asset build exists: {output_dir}")
    if not args.airplane_metadata.is_file():
        raise FileNotFoundError(args.airplane_metadata)
    if not gpu_idle(args.gpu):
        state.update({"stage": "stage_c_waiting_for_gpu", "waiting_step": "airplane_multilayer_asset_build", "updated_at": now()})
        write_json(state_path, state)
        time.sleep(900)
        raise WaitForResource("airplane_multilayer_asset_build")
    command = [
        args.python,
        str(args.repo / "tools/build_xvla_airplane_multilayer_assets.py"),
        "--checkpoint", str(args.airplane_checkpoint),
        "--xvla-root", str(args.xvla_root),
        "--metadata", str(args.airplane_metadata),
        "--output-dir", str(output_dir),
        "--batch-size", "8",
        "--probe-seed", "0",
        "--probe-steps", "5",
        "--device", "cuda",
    ]
    log = args.run_root / "logs" / "build_airplane_multilayer_assets.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    state.update({"stage": "stage_c_building_airplane_multilayer_assets", "command": command, "updated_at": now()})
    write_json(state_path, state)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "20", "MKL_NUM_THREADS": "20"}
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(["taskset", "-c", args.cpu_set, *command], cwd=args.repo, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0 or not asset_has_layer(output_asset, "vlm_input_pool", args.python):
        state.update({"stage": "stage_c_failed", "failed_step": "airplane_multilayer_asset_build", "returncode": result.returncode, "updated_at": now()})
        write_json(state_path, state)
        raise RuntimeError(f"Airplane multilayer asset build failed; see {log}")
    marker.write_text("ID-only multilayer detector asset includes vlm_input_pool\n", encoding="utf-8")
    state.update({"stage": "stage_c_running", "airplane_internal_assets": str(output_asset), "updated_at": now()})
    write_json(state_path, state)
    return output_asset


def run_logged(
    *,
    command: list[str],
    repo: Path,
    log_path: Path,
    output_summary: Path,
    state: dict[str, Any],
    state_path: Path,
    stage_name: str,
    gpu: int,
    cpu_set: str,
) -> None:
    if output_summary.is_file() and summary_complete(output_summary):
        state.setdefault("completed_steps", []).append(stage_name)
        write_json(state_path, state)
        return
    if output_summary.parent.exists() and any(output_summary.parent.iterdir()):
        raise FileExistsError(f"partial Stage-C output exists: {output_summary.parent}")
    if not gpu_idle(gpu):
        state.update({"stage": "stage_c_waiting_for_gpu", "waiting_step": stage_name, "updated_at": now()})
        write_json(state_path, state)
        time.sleep(900)
        raise WaitForResource(stage_name)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    state.update({"stage": stage_name, "command": command, "updated_at": now()})
    write_json(state_path, state)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(["taskset", "-c", cpu_set, *command], cwd=repo, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode not in (0, -6) or not summary_complete(output_summary):
        state.update({"stage": "stage_c_failed", "failed_step": stage_name, "returncode": result.returncode, "updated_at": now()})
        write_json(state_path, state)
        raise RuntimeError(f"Stage-C step failed: {stage_name}; see {log_path}")
    state.setdefault("completed_steps", []).append(stage_name)
    state.update({"stage": "stage_c_running", "updated_at": now()})
    write_json(state_path, state)


def run(args: argparse.Namespace) -> None:
    args.run_root.mkdir(parents=True, exist_ok=True)
    state_path = args.run_root / "pipeline_state.json"
    state = read_json(state_path) if state_path.exists() else {
        "pipeline_id": "xvla_fixedgrid_taskpolicy_knee_v1",
        "stage": "stage_c_waiting_for_stage_b",
        "started_at": now(),
        "completed_steps": [],
    }
    stage_b_marker = args.stage_b_root / "STAGE_B_UTILITY_COMPLETE"
    if not stage_b_marker.is_file():
        state.update({"stage": "stage_c_waiting_for_stage_b", "updated_at": now()})
        write_json(state_path, state)
        while not stage_b_marker.is_file():
            time.sleep(args.interval_seconds)

    # The historical Airplane detector asset predates input-pool capture.  Do
    # not substitute another layer: build the missing ID-only asset before
    # validation/OOD detector rollouts begin.
    airplane_internal_assets = ensure_airplane_assets(
        args=args, state=state, state_path=state_path
    )
    stack_checkpoint = args.stackcube_checkpoint
    airplane_checkpoint = args.airplane_checkpoint
    stack_internal = args.stackcube_internal_assets
    stack_external = args.stackcube_external_assets
    airplane_internal = airplane_internal_assets
    airplane_external = args.airplane_external_assets
    jobs = [
        {
            "task": "stackcube",
            "split": "id",
            "seed": 149000,
            "checkpoint": stack_checkpoint,
            "internal": stack_internal,
            "external": stack_external,
            "output": args.run_root / "stackcube" / "validation_id",
            "evaluator": args.repo / "tools/evaluate_xvla_stackcube_failure_detectors.py",
        },
        {
            "task": "airplane",
            "split": "id",
            "seed": 159000,
            "checkpoint": airplane_checkpoint,
            "internal": airplane_internal,
            "external": airplane_external,
            "output": args.run_root / "airplane" / "validation_id",
            "evaluator": args.repo / "tools/evaluate_xvla_airplane_failure_detectors.py",
        },
        {
            "task": "stackcube",
            "split": "ood",
            "seed": 151000,
            "checkpoint": stack_checkpoint,
            "internal": stack_internal,
            "external": stack_external,
            "output": args.run_root / "stackcube" / "heldout_ood",
            "evaluator": args.repo / "tools/evaluate_xvla_stackcube_failure_detectors.py",
        },
        {
            "task": "airplane",
            "split": "ood",
            "seed": 161000,
            "checkpoint": airplane_checkpoint,
            "internal": airplane_internal,
            "external": airplane_external,
            "output": args.run_root / "airplane" / "heldout_ood",
            "evaluator": args.repo / "tools/evaluate_xvla_airplane_failure_detectors.py",
        },
    ]
    for job in jobs:
        task = job["task"]
        command = [
            args.python,
            str(job["evaluator"]),
            "--checkpoint", str(job["checkpoint"]),
            "--xvla-root", str(args.xvla_root),
            "--multilayer-assets", str(job["internal"]),
            "--external-assets", str(job["external"]),
            "--output-dir", str(job["output"]),
            "--split", str(job["split"]),
            "--episodes", str(EXPECTED_EPISODES),
            "--seed", str(job["seed"]),
            "--execute-horizon", "5",
            "--max-episode-steps", "150",
            "--flow-steps", "10",
            "--probe-steps", "5",
            "--diff-timesteps", "16",
        ]
        run_logged(
            command=command,
            repo=args.repo,
            log_path=args.run_root / "logs" / f"{task}_{job['split']}.log",
            output_summary=job["output"] / "summary.json",
            state=state,
            state_path=state_path,
            stage_name=f"{task}_{job['split']}_passive",
            gpu=args.gpu,
            cpu_set=args.cpu_set,
        )
    calibration_steps = [
        ("stackcube", args.run_root / "stackcube" / "validation_id" / "summary.json"),
        ("airplane", args.run_root / "airplane" / "validation_id" / "summary.json"),
    ]
    for task, source in calibration_steps:
        output = args.run_root / task / "calibration_q95.json"
        if not output.exists():
            command = [
                args.python,
                str(args.repo / "tools/summarize_xvla_airplane_failure_detection.py"),
                "--calibrate", str(source),
                "--q", "0.95",
                "--output", str(output),
            ]
            output.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(command, cwd=args.repo, check=True)
        state.setdefault("completed_steps", []).append(f"{task}_threshold_calibration")
        write_json(state_path, state)

    audits = [
        ("stackcube", [10, 20], args.run_root / "stackcube" / "heldout_ood" / "summary.json"),
        ("airplane", [20], args.run_root / "airplane" / "heldout_ood" / "summary.json"),
    ]
    for task, knee_set, source in audits:
        output = args.run_root / task / "gate_to_knee.json"
        if output.exists():
            continue
        command = [
            args.python,
            str(args.repo / "tools/summarize_xvla_gate_to_knee.py"),
            "--summary", str(source),
            "--calibration", str(args.run_root / task / "calibration_q95.json"),
            "--task", task,
            "--knee-set", *[str(x) for x in knee_set],
            "--methods", *METHODS,
            "--knee-tolerance", "5",
            "--failure-recovery-step", "50",
            "--output", str(output),
        ]
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, cwd=args.repo, check=True)
        state.setdefault("completed_steps", []).append(f"{task}_gate_to_knee")
        write_json(state_path, state)

    state.update({"stage": "stage_c_gate_audit_complete_data_pending", "completed_at": now(), "updated_at": now()})
    write_json(state_path, state)
    (args.run_root / "STAGE_C_GATE_AUDIT_COMPLETE_DATA_PENDING").write_text("gate audit complete; gated data/training remain\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--stage-b-root", type=Path, default=STAGE_B_ROOT_DEFAULT)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--cpu-set", default="0-19")
    parser.add_argument("--interval-seconds", type=int, default=900)
    parser.add_argument("--stackcube-checkpoint", type=Path, required=True)
    parser.add_argument("--airplane-checkpoint", type=Path, required=True)
    parser.add_argument("--stackcube-internal-assets", type=Path, required=True)
    parser.add_argument("--stackcube-external-assets", type=Path, required=True)
    parser.add_argument("--airplane-internal-assets", type=Path, required=True)
    parser.add_argument("--airplane-external-assets", type=Path, required=True)
    parser.add_argument("--airplane-metadata", type=Path, required=True)
    args = parser.parse_args()
    while True:
        try:
            run(args)
            break
        except WaitForResource:
            continue


if __name__ == "__main__":
    main()
