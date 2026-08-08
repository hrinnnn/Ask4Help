#!/usr/bin/env python3
"""Run centered-oracle validation, recollection, training, and OOD evaluation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


METHODS = ("vlm_pool_pca", "offline_oracle", "failure_recovery", "diffdagger")
ROOT = Path("/data/zhaozhixuan/Ask4Help-airplane-event-close-v2")
ENV = Path("/data/zhaozhixuan/envs/xvla_official_5090")
XVLA = Path("/data/zhaozhixuan/X-VLA")
BASE = Path("/data/zhaozhixuan/Ask4Help-airplane-5090/results/xvla_airplane_v1")
RESULT = BASE / "ood_dagger_id_ood_alternating_centered_oracle_v3"
CHECKPOINT = BASE / "id_sft_10000_official_2gpu/ckpt-2500"
CALIBRATION = BASE / "ood_dagger_v1/calibration/id_success20_q95.json"
PCA_ASSET = BASE / "ood_dagger_v1/assets/vlm_input_pool_pca.pt"
COLLECTOR = ROOT / "tools/collect_pick_single_ycb_airplane_xvla_dagger.py"
EVALUATOR = ROOT / "tools/evaluate_pick_single_ycb_airplane_xvla.py"
TRAINER = ROOT / "tools/run_xvla_airplane_event_close_training.py"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def centered_oracle_validation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [row for row in rows if row.get("accepted")]
    selected_attempts = []
    for row in accepted:
        selected = row["oracle"]["selected_candidate"]
        selected_attempts.append(next(item for item in row["oracle"]["attempts"] if item["candidate"] == selected))
    return {
        "raw_attempts": len(rows),
        "accepted": len(accepted),
        "all_centered": bool(selected_attempts)
        and all(item["candidate"].startswith("neck_center_x_minus_014") for item in selected_attempts),
        "max_approach_shift_mm": max(
            (1000.0 * float(item["object_xy_shift_before_close"]) for item in selected_attempts),
            default=float("inf"),
        ),
        "max_close_shift_mm": max(
            (1000.0 * float(item["object_xy_shift_during_close"]) for item in selected_attempts),
            default=float("inf"),
        ),
    }


def child_env(gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "OMP_NUM_THREADS": "20",
            "MKL_NUM_THREADS": "20",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    return env


def collector_command(method: str, output: Path, dataset: Path, *, validation: bool = False) -> list[str]:
    command = [
        str(ENV / "bin/python"),
        str(COLLECTOR),
        "--method",
        method,
        "--checkpoint",
        str(CHECKPOINT),
        "--xvla-root",
        str(XVLA),
        "--calibration",
        str(CALIBRATION),
        "--pca-asset",
        str(PCA_ASSET),
        "--output-dir",
        str(output),
        "--repo-id",
        str(dataset),
    ]
    if validation:
        command += [
            "--target", "20", "--offline-per-split", "10", "--max-attempts", "20",
            "--id-seed", "93000", "--ood-seed", "94000",
        ]
    else:
        command += [
            "--target", "100", "--offline-per-split", "50", "--max-attempts", "5000",
            "--id-seed", "70000", "--ood-seed", "80000",
        ]
    return command


def launch(command: list[str], *, gpu: int, log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(
        ["taskset", "-c", f"{gpu * 20}-{gpu * 20 + 19}", *command],
        cwd=ROOT,
        env=child_env(gpu),
        stdout=log,
        stderr=subprocess.STDOUT,
    )


def validate_collection(method: str) -> dict[str, Any]:
    collection = RESULT / "collections" / method
    dataset = RESULT / "datasets" / method
    summary = json.loads((collection / "summary.json").read_text(encoding="utf-8"))
    episodes = read_jsonl(collection / "training_episodes.jsonl")
    videos = list((collection / "raw_archive/videos").glob("*.mp4"))
    dataset_episodes = read_jsonl(dataset / "meta/episodes.jsonl")
    if summary["accepted_total"] != 100 or len(episodes) != 100 or len(dataset_episodes) != 100:
        raise RuntimeError(f"invalid {method} collection counts")
    if len(videos) != sum(summary["raw_by_split"].values()):
        raise RuntimeError(f"missing {method} raw videos")
    return {
        "accepted_by_split": summary["accepted_by_split"],
        "raw_by_split": summary["raw_by_split"],
        "videos": len(videos),
    }


def validate_collection_exit(method: str, returncode: int) -> dict[str, Any]:
    """Accept a known native teardown abort only after artifacts fully validate."""
    report = validate_collection(method)
    if returncode not in (0, -6):
        raise RuntimeError(f"collection failed for {method}: returncode={returncode}")
    report["returncode"] = returncode
    report["accepted_teardown_abort"] = returncode == -6
    return report


def run_validation() -> None:
    output = RESULT / "oracle_validation_20/collection"
    dataset = RESULT / "oracle_validation_20/dataset"
    process = launch(
        collector_command("offline_oracle", output, dataset, validation=True),
        gpu=0,
        log_path=RESULT / "logs/oracle_validation_20.log",
    )
    write_json(RESULT / "pids/oracle_validation_20.json", {"pid": process.pid})
    returncode = process.wait()
    rows = read_jsonl(output / "episodes.jsonl")
    report = centered_oracle_validation(rows)
    report["returncode"] = returncode
    write_json(RESULT / "oracle_validation_20/report.json", report)
    if (
        report["raw_attempts"] != 20
        or report["accepted"] < 18
        or not report["all_centered"]
        or report["max_approach_shift_mm"] > 20.0
        or report["max_close_shift_mm"] > 10.0
    ):
        raise RuntimeError(f"centered oracle validation failed: {report}")


def run_collections() -> None:
    processes = {}
    for gpu, method in enumerate(METHODS):
        output = RESULT / "collections" / method
        dataset = RESULT / "datasets" / method
        processes[method] = launch(
            collector_command(method, output, dataset),
            gpu=gpu,
            log_path=RESULT / f"logs/collection_{method}.log",
        )
        write_json(RESULT / f"pids/collection_{method}.json", {"pid": processes[method].pid})
    returncodes = {method: process.wait() for method, process in processes.items()}
    reports = {
        method: validate_collection_exit(method, returncodes[method])
        for method in METHODS
    }
    write_json(RESULT / "collection_report.json", reports)


def run_training() -> Path:
    run = RESULT / "training_v1_5000"
    env = os.environ.copy()
    env.update(
        {
            "XVLA_EVENT_CLOSE_RESULT_ROOT": str(RESULT),
            "XVLA_EVENT_CLOSE_TRAIN_RUN": str(run),
        }
    )
    log_path = RESULT / "logs/training_orchestrator.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [str(ENV / "bin/python"), str(TRAINER)],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        write_json(RESULT / "pids/training_orchestrator.json", {"pid": process.pid})
        returncode = process.wait()
    if returncode != 0 or not (run / "TRAINING_COMPLETE").exists():
        raise RuntimeError(f"training failed with returncode {returncode}")
    return run


def run_evaluation(training_run: Path) -> None:
    evaluation = RESULT / "eval_ood_ckpt5000_10_h150_v1"
    processes = {}
    for gpu, method in enumerate(METHODS):
        output = evaluation / method
        command = [
            str(ENV / "bin/python"), str(EVALUATOR),
            "--checkpoint", str(training_run / f"formal_5000/{method}/ckpt-5000"),
            "--xvla-root", str(XVLA), "--output-dir", str(output), "--split", "ood",
            "--episodes", "10", "--seed", "60000", "--execute-horizon", "5",
            "--max-episode-steps", "150", "--flow-steps", "10",
        ]
        processes[method] = launch(command, gpu=gpu, log_path=evaluation / f"logs/{method}.log")
        write_json(evaluation / f"pids/{method}.json", {"pid": processes[method].pid})
    returncodes = {method: process.wait() for method, process in processes.items()}
    report = {}
    for method in METHODS:
        summary_path = evaluation / method / "summary.json"
        if not summary_path.exists():
            raise RuntimeError(f"evaluation failed for {method}: {returncodes[method]}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary["episodes"] != 10 or len(list((evaluation / method / "videos").glob("*.mp4"))) != 10:
            raise RuntimeError(f"incomplete evaluation for {method}")
        report[method] = {
            "ever_grasped_successes": summary["ever_grasped_successes"],
            "strict_successes": summary["strict_successes"],
            "returncode": returncodes[method],
        }
    write_json(evaluation / "comparison.json", report)


def main() -> None:
    resume_stage = os.environ.get("XVLA_CENTERED_PIPELINE_RESUME_STAGE")
    if resume_stage == "post_collection":
        reports = {method: validate_collection(method) for method in METHODS}
        write_json(RESULT / "collection_report.json", reports)
        write_json(RESULT / "pipeline_state.json", {"stage": "training", "resumed": True})
        training_run = run_training()
        write_json(RESULT / "pipeline_state.json", {"stage": "evaluation", "resumed": True})
        run_evaluation(training_run)
        write_json(RESULT / "pipeline_state.json", {"stage": "complete", "resumed": True})
        (RESULT / "PIPELINE_COMPLETE").write_text("complete\n", encoding="utf-8")
        return
    if resume_stage is not None:
        raise ValueError(f"unsupported resume stage: {resume_stage}")
    if RESULT.exists():
        raise FileExistsError(RESULT)
    RESULT.mkdir(parents=True)
    write_json(RESULT / "pipeline_state.json", {"stage": "oracle_validation"})
    run_validation()
    write_json(RESULT / "pipeline_state.json", {"stage": "collection"})
    run_collections()
    write_json(RESULT / "pipeline_state.json", {"stage": "training"})
    training_run = run_training()
    write_json(RESULT / "pipeline_state.json", {"stage": "evaluation"})
    run_evaluation(training_run)
    write_json(RESULT / "pipeline_state.json", {"stage": "complete"})
    (RESULT / "PIPELINE_COMPLETE").write_text("complete\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
