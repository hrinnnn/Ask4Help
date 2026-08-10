#!/usr/bin/env python3
"""Persistent StackCube four-group collection-to-training pipeline."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path


METHODS = ("vlm_bridge_pca", "offline_oracle", "failure_recovery", "diffdagger")
ROOT = Path(os.environ.get("ASK4HELP_ROOT", Path(__file__).resolve().parents[1]))
RESULT = Path(os.environ["XVLA_STACKCUBE_FOUR_GROUP_RESULT"])
ENV = Path("/data/zhaozhixuan/envs/xvla_official_5090")
XVLA = Path("/data/zhaozhixuan/X-VLA")
CHECKPOINT = Path(
    "/data/zhaozhixuan/Ask4Help-airplane-5090/results/xvla_stackcube_v1/"
    "temporal_mask_v2/id_sft_from3500_to10000_official_2gpu_retry1/ckpt-7500"
)
BENCHMARK = Path(
    "/data/zhaozhixuan/Ask4Help-airplane-5090/results/xvla_stackcube_v1/"
    "temporal_mask_v2/failure_detection_ckpt7500_100id100ood_v1"
)
ASSETS = BENCHMARK / "assets_internal/multilayer_detector_assets.pt"
CALIBRATION = BENCHMARK / "calibration_q95.json"
CALIBRATION_ROLLOUTS = BENCHMARK / "calibration_id25/summary.json"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def child_env(gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "PYTHONPATH": f"{ROOT}:{ROOT / 'RLinf'}",
            "OMP_NUM_THREADS": "20",
            "MKL_NUM_THREADS": "20",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    return env


def diffdagger_gate_threshold(summary: dict, *, q: float, patience: int) -> tuple[float, list[float]]:
    """Calibrate the same consecutive-score statistic used by the online gate."""
    maxima = []
    for row in summary["rows"]:
        scores = [
            float(point["scores"]["diffdagger"])
            for point in row["timeline"]
            if point.get("scores", {}).get("diffdagger") is not None
        ]
        windows = [min(scores[index : index + patience]) for index in range(len(scores) - patience + 1)]
        if not windows:
            raise ValueError("calibration trajectory is shorter than gate patience")
        maxima.append(max(windows))
    rank = min(len(maxima), math.ceil((len(maxima) + 1) * q))
    return sorted(maxima)[rank - 1], maxima


def collector_command(
    method: str, output: Path, dataset: Path, target: int, diff_threshold: float
) -> list[str]:
    return [
        str(ENV / "bin/python"),
        str(ROOT / "tools/collect_stackcube_xvla_dagger.py"),
        "--method", method,
        "--checkpoint", str(CHECKPOINT),
        "--xvla-root", str(XVLA),
        "--internal-assets", str(ASSETS),
        "--calibration", str(CALIBRATION),
        "--output-dir", str(output),
        "--repo-id", str(dataset),
        "--target", str(target),
        "--id-seed", "70000",
        "--ood-seed", "80000",
        "--flow-steps", "10",
        "--diff-timesteps", "16",
        "--diff-patience", "2",
        "--diff-threshold", str(diff_threshold),
    ]


def launch_collections(stage: str, target: int, diff_threshold: float) -> None:
    processes = {}
    handles = []
    for gpu, method in enumerate(METHODS):
        output = RESULT / stage / "collections" / method
        dataset = RESULT / stage / "datasets" / method
        log_path = RESULT / "logs" / f"{stage}_{method}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("w", encoding="utf-8")
        handles.append(handle)
        processes[method] = subprocess.Popen(
            [
                "taskset", "-c", f"{gpu * 20}-{gpu * 20 + 19}",
                *collector_command(method, output, dataset, target, diff_threshold),
            ],
            cwd=ROOT,
            env=child_env(gpu),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        write_json(RESULT / "pids" / f"{stage}_{method}.json", {"pid": processes[method].pid})
        print(f"[stackcube-four-pipeline] stage={stage} method={method} gpu={gpu} pid={processes[method].pid}", flush=True)
    failures = {method: process.wait() for method, process in processes.items()}
    for handle in handles:
        handle.close()
    if any(code != 0 for code in failures.values()):
        raise RuntimeError(f"{stage} collections failed: {failures}")
    for method in METHODS:
        summary = json.loads(
            (RESULT / stage / "collections" / method / "summary.json").read_text(encoding="utf-8")
        )
        if int(summary["accepted_total"]) != target:
            raise RuntimeError(f"{method} admitted {summary['accepted_total']}/{target}")


def promote_formal_layout() -> None:
    for name in ("collections", "datasets"):
        source = RESULT / "formal" / name
        target = RESULT / name
        if target.exists():
            raise FileExistsError(target)
        target.symlink_to(source, target_is_directory=True)


def run_training() -> None:
    env = os.environ.copy()
    env.update(
        {
            "ASK4HELP_ROOT": str(ROOT),
            "XVLA_STACKCUBE_FOUR_GROUP_RESULT": str(RESULT),
            "XVLA_STACKCUBE_FOUR_GROUP_TRAIN_RUN": str(RESULT / "training_v1_5000"),
        }
    )
    log_path = RESULT / "logs/training_orchestrator.log"
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            [str(ENV / "bin/python"), str(ROOT / "tools/run_xvla_stackcube_four_group_training.py")],
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        write_json(RESULT / "pids/training_orchestrator.json", {"pid": process.pid})
        status = process.wait()
    if status != 0 or not (RESULT / "training_v1_5000/TRAINING_COMPLETE").exists():
        raise RuntimeError(f"training failed with status {status}")


def main() -> None:
    if RESULT.exists():
        raise FileExistsError(RESULT)
    RESULT.mkdir(parents=True)
    calibration_summary = json.loads(CALIBRATION_ROLLOUTS.read_text(encoding="utf-8"))
    diff_threshold, maxima = diffdagger_gate_threshold(
        calibration_summary, q=0.95, patience=2
    )
    write_json(
        RESULT / "diffdagger_gate_calibration_q95_patience2.json",
        {
            "source": str(CALIBRATION_ROLLOUTS),
            "successful_id_trajectories": len(maxima),
            "q": 0.95,
            "patience": 2,
            "threshold": diff_threshold,
            "trajectory_gate_maxima": maxima,
        },
    )
    write_json(RESULT / "pipeline_state.json", {"stage": "collector_smoke"})
    launch_collections("smoke", 1, diff_threshold)
    write_json(RESULT / "pipeline_state.json", {"stage": "formal_collection"})
    launch_collections("formal", 100, diff_threshold)
    promote_formal_layout()
    write_json(RESULT / "pipeline_state.json", {"stage": "training"})
    run_training()
    write_json(RESULT / "pipeline_state.json", {"stage": "complete"})
    (RESULT / "PIPELINE_COMPLETE").write_text("complete\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
