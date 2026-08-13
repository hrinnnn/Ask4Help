#!/usr/bin/env python3
"""Restart-tolerant X-VLA StackCube target-OOD timing pipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from tools.run_xvla_stackcube_stage2_training import METHODS, select_idle_gpus


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def child_env(gpu: int, repo: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "PYTHONPATH": f"{repo}:{repo / 'RLinf'}",
        "OMP_NUM_THREADS": "20",
        "MKL_NUM_THREADS": "20",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    return env


def launch_waves(
    jobs: list[tuple[str, list[str], Path, Path]], *, gpus: list[int], repo: Path,
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
                raise RuntimeError(f"partial output for {name}: {completion.parent}")
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("w", encoding="utf-8")
            process = subprocess.Popen(
                ["taskset", "-c", f"{slot * 20}-{slot * 20 + 19}", *command],
                cwd=repo, env=child_env(gpu, repo), stdout=handle,
                stderr=subprocess.STDOUT,
            )
            running.append((name, completion, process, handle, gpu))
        for name, completion, process, handle, gpu in running:
            status = process.wait()
            handle.close()
            accepted = accept_completed_teardown and status == -6 and completion.exists()
            if (status != 0 and not accepted) or not completion.exists():
                raise RuntimeError(f"{name} failed on GPU {gpu}: status={status}")


def cohort_phase(args: argparse.Namespace, gpus: list[int]) -> None:
    screen = args.result / "cohort/policy_screen"
    command = [
        str(args.python), str(args.repo / "tools/evaluate_stackcube_xvla.py"),
        "--checkpoint", str(args.checkpoint), "--xvla-root", str(args.xvla_root),
        "--output-dir", str(screen), "--episodes", str(args.cohort_screen_episodes),
        "--seed", str(args.collection_seed), "--split", "stage2_ood",
        "--execute-horizon", "5", "--max-episode-steps", "150", "--flow-steps", "10",
    ]
    launch_waves(
        [("cohort_screen", command, args.result / "logs/cohort_screen.log", screen / "summary.json")],
        gpus=gpus, repo=args.repo, accept_completed_teardown=True,
    )
    manifest = args.result / "cohort/seed_manifest.json"
    if not manifest.exists():
        subprocess.run(
            [str(args.python), str(args.repo / "tools/build_stackcube_xvla_stage2_cohort.py"),
             "--policy-summary", str(screen / "summary.json"), "--output", str(manifest),
             "--count", str(args.cohort_size)],
            cwd=args.repo, check=True,
        )


def collection_phase(args: argparse.Namespace, gpus: list[int]) -> None:
    cohort = args.result / "cohort/seed_manifest.json"
    jobs = []
    for method in METHODS:
        output = args.result / "collection_pools" / method
        dataset = args.result / "dataset_pools" / method
        command = [
            str(args.python), str(args.repo / "tools/collect_stackcube_xvla_dagger.py"),
            "--method", method, "--checkpoint", str(args.checkpoint),
            "--xvla-root", str(args.xvla_root), "--output-dir", str(output),
            "--repo-id", str(dataset), "--target", str(args.cohort_size),
            "--seed-manifest", str(cohort), "--controlled-timing",
            "--ood-split", "stage2_ood", "--flow-steps", "10",
            "--failure-recovery-mode", "event",
        ]
        jobs.append((f"collect_{method}", command,
                     args.result / "logs" / f"collect_{method}.log",
                     output / "summary.json"))
    launch_waves(jobs, gpus=gpus, repo=args.repo, accept_completed_teardown=True)

    for method in METHODS:
        pool_summary = json.loads(
            (args.result / "collection_pools" / method / "summary.json").read_text(
                encoding="utf-8"
            )
        )
        if int(pool_summary["accepted_total"]) != args.cohort_size:
            raise RuntimeError(f"{method} did not complete the common seed cohort")
        if int(pool_summary["accepted_expert_actions"]) < args.pool_action_target:
            raise RuntimeError(f"{method} pool did not reach {args.pool_action_target} actions")
        output = args.result / "datasets" / method
        if not (output / "selection_manifest.json").exists():
            subprocess.run(
                [str(args.python), str(args.repo / "tools/select_stackcube_xvla_timing_budget.py"),
                 "--pool", str(args.result / "dataset_pools" / method),
                 "--output", str(output), "--budget", str(args.expert_action_budget)],
                cwd=args.repo, check=True,
            )
    quality = args.result / "intervention_quality/summary.json"
    if not quality.exists():
        subprocess.run(
            [str(args.python), str(args.repo / "tools/summarize_stackcube_xvla_timing_quality.py"),
             "--root", str(args.result), "--output", str(quality)],
            cwd=args.repo, check=True,
        )


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
            [str(args.python), str(args.repo / "tools/run_xvla_stackcube_stage2_training.py"),
             "--repo", str(args.repo), "--xvla-root", str(args.xvla_root),
             "--python", str(args.python), "--start", str(args.checkpoint),
             "--id-meta", str(args.id_meta), "--datasets", str(args.result / "datasets"),
             "--run", str(run), "--expert-action-budget", str(args.expert_action_budget),
             "--steps", str(args.training_steps), "--save-interval", "500",
             "--gpus", *(str(gpu) for gpu in gpus)],
            cwd=args.repo, env=child_env(gpus[0], args.repo), stdout=handle,
            stderr=subprocess.STDOUT,
        ).returncode
    if status != 0 or not complete.exists():
        raise RuntimeError(f"training orchestrator failed: {status}")


def evaluation_phase(args: argparse.Namespace, gpus: list[int]) -> None:
    rows = []
    for step in args.selection_steps:
        jobs = []
        for method in METHODS:
            output = args.result / "checkpoint_selection" / f"step_{step}" / method
            checkpoint = args.result / "training" / f"formal_{args.training_steps}" / method / f"ckpt-{step}"
            command = [
                str(args.python), str(args.repo / "tools/evaluate_stackcube_xvla.py"),
                "--checkpoint", str(checkpoint), "--xvla-root", str(args.xvla_root),
                "--output-dir", str(output), "--episodes", str(args.selection_episodes),
                "--seed", str(args.selection_seed), "--split", "stage2_ood",
                "--execute-horizon", "5", "--max-episode-steps", "150", "--flow-steps", "10",
            ]
            jobs.append((f"select_{method}_{step}", command,
                         args.result / "logs" / f"select_{method}_{step}.log",
                         output / "summary.json"))
        launch_waves(jobs, gpus=gpus, repo=args.repo, accept_completed_teardown=True)
        for method in METHODS:
            path = args.result / "checkpoint_selection" / f"step_{step}" / method / "summary.json"
            summary = json.loads(path.read_text(encoding="utf-8"))
            rows.append({"step": step, "method": method,
                         "success_rate": float(summary["success_rate"]), "summary": str(path)})
    write_json(args.result / "checkpoint_selection/summary.json", {"rows": rows})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--id-meta", type=Path, required=True)
    parser.add_argument("--expert-action-budget", type=int, default=2000)
    parser.add_argument("--pool-action-target", type=int, default=2200)
    parser.add_argument("--training-steps", type=int, default=10000)
    parser.add_argument("--cohort-screen-episodes", type=int, default=300)
    parser.add_argument("--cohort-size", type=int, default=200)
    parser.add_argument("--collection-seed", type=int, default=98000)
    parser.add_argument("--selection-steps", type=int, nargs="+", default=[2000, 4000, 6000, 8000, 10000])
    parser.add_argument("--selection-episodes", type=int, default=25)
    parser.add_argument("--selection-seed", type=int, default=99000)
    parser.add_argument("--gpus", type=int, nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.result.mkdir(parents=True, exist_ok=True)
    gpus = args.gpus or select_idle_gpus(2)
    if len(gpus) != 2:
        raise ValueError("timing pipeline requires exactly two idle GPUs")
    write_json(args.result / "experiment_contract.json", {
        "task": "X-VLA StackCube target-position OOD timing study",
        "distribution_shift": "green target position only; red cube paired with ID",
        "controlled_takeover_conditions": list(METHODS),
        "expert_action_budget_per_group": args.expert_action_budget,
        "pool_action_target_per_group": args.pool_action_target,
        "training_steps": args.training_steps,
        "save_interval": 500,
        "selection_steps": args.selection_steps,
        "gpus": gpus,
    })
    phases = (("cohort", cohort_phase), ("collection", collection_phase),
              ("training", training_phase), ("checkpoint_selection", evaluation_phase))
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
