#!/usr/bin/env python3
"""Restart-tolerant X-VLA StackCube Stage-2 OOD experiment pipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from tools.run_xvla_stackcube_four_group_pipeline import diffdagger_gate_threshold
from tools.run_xvla_stackcube_stage2_training import METHODS, select_idle_gpus


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


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
        }
    )
    return env


def launch_waves(
    jobs: list[tuple[str, list[str], Path, Path]],
    *,
    gpus: list[int],
    repo: Path,
    accept_completed_teardown: bool = False,
) -> None:
    for wave_start in range(0, len(jobs), len(gpus)):
        running = []
        for slot, ((name, command, log, completion), gpu) in enumerate(
            zip(jobs[wave_start : wave_start + len(gpus)], gpus)
        ):
            if completion.exists():
                continue
            if completion.parent.exists():
                raise RuntimeError(
                    f"partial output for {name} requires diagnosis and a new retry path: "
                    f"{completion.parent}"
                )
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("w", encoding="utf-8")
            process = subprocess.Popen(
                ["taskset", "-c", f"{slot * 20}-{slot * 20 + 19}", *command],
                cwd=repo,
                env=child_env(gpu, repo),
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            running.append((name, completion, process, handle, gpu))
        for name, completion, process, handle, gpu in running:
            status = process.wait()
            handle.close()
            complete = completion.exists()
            accepted_teardown = accept_completed_teardown and status == -6 and complete
            if (status != 0 and not accepted_teardown) or not complete:
                raise RuntimeError(
                    f"{name} failed on GPU {gpu}: status={status}, complete={complete}"
                )


def detector_phase(args: argparse.Namespace, gpus: list[int]) -> None:
    output = args.result / "failure_detection/eval_stage2_ood_100"
    command = [
        str(args.python),
        str(args.repo / "tools/evaluate_xvla_stackcube_failure_detectors.py"),
        "--checkpoint", str(args.checkpoint),
        "--xvla-root", str(args.xvla_root),
        "--multilayer-assets", str(args.internal_assets),
        "--external-assets", str(args.external_assets),
        "--output-dir", str(output),
        "--split", "stage2_ood",
        "--episodes", "100",
        "--seed", str(args.detector_seed),
        "--execute-horizon", "5",
        "--max-episode-steps", "150",
        "--flow-steps", "10",
        "--probe-steps", "5",
        "--probe-seed", "0",
        "--diff-timesteps", "16",
        "--diff-noise-samples", "1",
    ]
    launch_waves(
        [("stage2_detector", command, args.result / "logs/detector.log", output / "summary.json")],
        gpus=gpus,
        repo=args.repo,
        accept_completed_teardown=True,
    )
    metrics = args.result / "failure_detection/metrics"
    if not (metrics / "metrics.json").exists():
        if metrics.exists():
            raise RuntimeError(f"partial detector metrics: {metrics}")
        with (args.result / "logs/detector_metrics.log").open("w") as handle:
            subprocess.run(
                [
                    str(args.python),
                    str(args.repo / "tools/summarize_xvla_airplane_failure_detection.py"),
                    "--id-summary", str(args.id_detector_summary),
                    "--ood-summary", str(output / "summary.json"),
                    "--calibration", str(args.calibration),
                    "--output", str(metrics),
                ],
                cwd=args.repo,
                env=child_env(gpus[0], args.repo),
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )


def collection_phase(args: argparse.Namespace, gpus: list[int]) -> None:
    calibration_rollouts = json.loads(
        args.id_calibration_summary.read_text(encoding="utf-8")
    )
    diff_threshold, maxima = diffdagger_gate_threshold(
        calibration_rollouts, q=0.95, patience=2
    )
    write_json(
        args.result / "collection_calibration.json",
        {
            "source": str(args.id_calibration_summary),
            "diffdagger_threshold": diff_threshold,
            "q": 0.95,
            "patience": 2,
            "successful_id_trajectories": len(maxima),
            "internal_layer": "action_block_01",
            "internal_method": "action_block_01_pca",
        },
    )
    jobs = []
    for method in METHODS:
        output = args.result / "collections" / method
        dataset = args.result / "datasets" / method
        command = [
            str(args.python),
            str(args.repo / "tools/collect_stackcube_xvla_dagger.py"),
            "--method", method,
            "--checkpoint", str(args.checkpoint),
            "--xvla-root", str(args.xvla_root),
            "--internal-assets", str(args.internal_assets),
            "--calibration", str(args.calibration),
            "--output-dir", str(output),
            "--repo-id", str(dataset),
            "--target", "100",
            "--expert-action-budget", str(args.expert_action_budget),
            "--ood-split", "stage2_ood",
            "--id-seed", str(args.collection_id_seed),
            "--ood-seed", str(args.collection_ood_seed),
            "--flow-steps", "10",
            "--diff-timesteps", "16",
            "--diff-patience", "2",
            "--diff-threshold", str(diff_threshold),
            "--internal-layer", "action_block_01",
            "--probe-steps", "5",
            "--probe-seed", "0",
            "--failure-recovery-mode", "event",
        ]
        jobs.append(
            (
                f"collect_{method}",
                command,
                args.result / "logs" / f"collect_{method}.log",
                output / "summary.json",
            )
        )
    launch_waves(
        jobs,
        gpus=gpus,
        repo=args.repo,
        accept_completed_teardown=True,
    )
    for method in METHODS:
        summary = json.loads(
            (args.result / "collections" / method / "summary.json").read_text(
                encoding="utf-8"
            )
        )
        if int(summary["accepted_expert_actions"]) != args.expert_action_budget:
            raise RuntimeError(f"{method} did not meet the expert-action budget")


def training_phase(args: argparse.Namespace, gpus: list[int]) -> None:
    run = args.result / "training"
    complete = run / "TRAINING_COMPLETE"
    if complete.exists():
        return
    if run.exists():
        raise RuntimeError(f"partial training requires diagnosis: {run}")
    log = args.result / "logs/training_orchestrator.log"
    with log.open("w", encoding="utf-8") as handle:
        status = subprocess.run(
            [
                str(args.python),
                str(args.repo / "tools/run_xvla_stackcube_stage2_training.py"),
                "--repo", str(args.repo),
                "--xvla-root", str(args.xvla_root),
                "--python", str(args.python),
                "--start", str(args.checkpoint),
                "--id-meta", str(args.id_meta),
                "--datasets", str(args.result / "datasets"),
                "--run", str(run),
                "--expert-action-budget", str(args.expert_action_budget),
                "--steps", str(args.training_steps),
                "--save-interval", "500",
                "--gpus", *(str(gpu) for gpu in gpus),
            ],
            cwd=args.repo,
            env=child_env(gpus[0], args.repo),
            stdout=handle,
            stderr=subprocess.STDOUT,
        ).returncode
    if status != 0 or not complete.exists():
        raise RuntimeError(f"training orchestrator failed: {status}")


def evaluation_phase(args: argparse.Namespace, gpus: list[int]) -> None:
    jobs = []
    for index, method in enumerate(METHODS):
        output = args.result / "evaluations" / method / f"stage2_ood_{args.eval_episodes}"
        checkpoint = (
            args.result
            / "training"
            / f"formal_{args.training_steps}"
            / method
            / f"ckpt-{args.training_steps}"
        )
        command = [
            str(args.python),
            str(args.repo / "tools/evaluate_stackcube_xvla.py"),
            "--checkpoint", str(checkpoint),
            "--xvla-root", str(args.xvla_root),
            "--output-dir", str(output),
            "--episodes", str(args.eval_episodes),
            "--seed", str(args.eval_seed),
            "--split", "stage2_ood",
            "--execute-horizon", "5",
            "--max-episode-steps", "150",
            "--flow-steps", "10",
        ]
        jobs.append(
            (
                f"eval_{method}",
                command,
                args.result / "logs" / f"eval_{method}.log",
                output / "summary.json",
            )
        )
    launch_waves(
        jobs,
        gpus=gpus,
        repo=args.repo,
        accept_completed_teardown=True,
    )
    rows = []
    for method in METHODS:
        summary_path = (
            args.result
            / "evaluations"
            / method
            / f"stage2_ood_{args.eval_episodes}"
            / "summary.json"
        )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "method": method,
                "successes": int(summary["successes"]),
                "episodes": int(summary["episodes"]),
                "success_rate": float(summary["success_rate"]),
                "grasp_rate": float(summary["grasp_rate"]),
                "on_cube_rate": float(summary["on_cube_rate"]),
                "summary": str(summary_path),
            }
        )
    write_json(args.result / "final_summary.json", {"rows": rows})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--internal-assets", type=Path, required=True)
    parser.add_argument("--external-assets", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--id-calibration-summary", type=Path, required=True)
    parser.add_argument("--id-detector-summary", type=Path, required=True)
    parser.add_argument("--id-meta", type=Path, required=True)
    parser.add_argument("--expert-action-budget", type=int, default=2500)
    parser.add_argument("--training-steps", type=int, default=2500)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--detector-seed", type=int, default=96000)
    parser.add_argument("--collection-id-seed", type=int, default=97000)
    parser.add_argument("--collection-ood-seed", type=int, default=98000)
    parser.add_argument("--eval-seed", type=int, default=99000)
    parser.add_argument("--gpus", type=int, nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.result.mkdir(parents=True, exist_ok=True)
    gpus = args.gpus or select_idle_gpus(2)
    if len(gpus) != 2:
        raise ValueError("Stage-2 pipeline requires exactly two idle GPUs")
    write_json(
        args.result / "experiment_contract.json",
        {
            "task": "X-VLA StackCube Stage 2 OOD",
            "distribution_shift": "green target position only",
            "red_cube_distribution": "paired ID distribution",
            "affected_stage": "transport and placement",
            "base_checkpoint": str(args.checkpoint.resolve()),
            "internal_gate": "action_block_01 PCA residual",
            "methods": list(METHODS),
            "expert_action_budget_per_method": args.expert_action_budget,
            "training_steps": args.training_steps,
            "save_interval": 500,
            "evaluation_episodes": args.eval_episodes,
            "gpus": gpus,
        },
    )
    phases = (
        ("failure_detection", detector_phase),
        ("collection", collection_phase),
        ("training", training_phase),
        ("evaluation", evaluation_phase),
    )
    for name, phase in phases:
        write_json(args.result / "pipeline_state.json", {"stage": name})
        phase(args, gpus)
    write_json(args.result / "pipeline_state.json", {"stage": "complete"})
    (args.result / "PIPELINE_COMPLETE").write_text("complete\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
