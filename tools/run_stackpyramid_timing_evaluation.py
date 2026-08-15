#!/usr/bin/env python3
"""Run the held-out ID/OOD evaluation for all trained timing conditions."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import time
from pathlib import Path


STAGES = ("stage1_ood", "stage2_ood", "stage3_ood")
CONDITIONS = ("immediate", "pre_stage", "capability_boundary", "failure_recovery")
ID_SEED = 72000
OOD_SEEDS = {"stage1_ood": 73000, "stage2_ood": 74000, "stage3_ood": 75000}


def _seed_start(spec: object, default: int) -> int:
    if isinstance(spec, dict):
        return int(spec["start"])
    if isinstance(spec, list):
        return int(spec[0])
    return default


def write_state(path: Path, **updates: object) -> None:
    state = {"format": "stackpyramid_timing_evaluation_v1", **updates}
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    path.write_text(json.dumps(state, indent=2, default=str) + "\n", encoding="utf-8")


def run_condition(
    stage: str,
    condition: str,
    output: Path,
    gpu: str,
    cpu_set: str,
    args: argparse.Namespace,
    state_path: Path,
    id_seed: int,
    ood_seeds: dict[str, int],
) -> None:
    output.mkdir(parents=True, exist_ok=False)
    log = output / "evaluation.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONPATH"] = os.pathsep.join([str(args.repo_root), str(args.xvla_root), env.get("PYTHONPATH", "")])
    checkpoint = args.output_root / "training" / stage / condition / "ckpt-2000"
    if not checkpoint.is_dir():
        raise RuntimeError(f"missing checkpoint: {checkpoint}")
    script = args.repo_root / "tools" / "evaluate_stackpyramid_xvla.py"
    commands = (
        ("id", id_seed),
        (stage, ood_seeds[stage]),
    )
    with log.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps({"stage": stage, "condition": condition, "gpu": gpu, "checkpoint": str(checkpoint)}) + "\n")
        for split, seed in commands:
            split_output = output / split
            command = [
                str(args.python), str(script),
                "--checkpoint", str(checkpoint),
                "--xvla-root", str(args.xvla_root),
                "--output", str(split_output),
                "--split", split,
                "--episodes", "100",
                "--start-seed", str(seed),
                "--max-episode-steps", "250",
                "--execute-horizon", "5",
                "--flow-steps", "5",
                "--device", "cuda",
                "--sim-backend", "gpu",
                "--render-backend", "gpu",
            ]
            stream.write(json.dumps({"command": command, "split": split}) + "\n")
            stream.flush()
            process = subprocess.Popen(
                ["taskset", "-c", cpu_set, *command],
                cwd=args.repo_root,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
            write_state(state_path, phase="evaluation", running={"stage": stage, "condition": condition, "split": split, "pid": process.pid, "gpu": gpu, "output": str(split_output)})
            return_code = process.wait()
            if return_code not in (0, -6) or not (split_output / "EVAL_COMPLETE").is_file():
                raise RuntimeError(f"evaluation failed for {stage}/{condition}/{split}; rc={return_code}")
            summary = json.loads((split_output / "summary.json").read_text(encoding="utf-8"))
            if summary.get("episodes") != 100 or summary.get("video_count") != 100:
                raise RuntimeError(f"incomplete evaluation summary: {split_output}")
    (output / "CONDITION_EVAL_COMPLETE").write_text("complete\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--cpu-sets", default="0-7,8-15")
    parser.add_argument("--seed-manifest", type=Path)
    args = parser.parse_args()
    gpus = args.gpus.split(",")
    cpu_sets = args.cpu_sets.split(",")
    if len(gpus) != 2 or len(cpu_sets) != 2:
        raise ValueError("exactly two GPUs and two CPU sets are required")

    root = args.output_root
    id_seed = ID_SEED
    ood_seeds = dict(OOD_SEEDS)
    if args.seed_manifest is not None:
        manifest = json.loads(args.seed_manifest.read_text(encoding="utf-8"))
        final = manifest.get("final_evaluation", {})
        id_seed = _seed_start(final.get("id"), id_seed)
        ood_seeds = {
            stage: _seed_start(final.get(stage), ood_seeds[stage])
            for stage in STAGES
        }
    state_path = root / "evaluation_pipeline_state.json"
    training_marker = root / "TIMING_TRAINING_COMPLETE"
    while not training_marker.is_file():
        training_state = root / "training_pipeline_state.json"
        if training_state.is_file() and json.loads(training_state.read_text()).get("phase") == "failed":
            raise RuntimeError("training controller reported failure")
        write_state(state_path, phase="waiting_for_training", training_marker=str(training_marker))
        time.sleep(300)

    jobs = [(stage, condition) for stage in STAGES for condition in CONDITIONS]
    for wave_start in range(0, len(jobs), 2):
        wave = jobs[wave_start : wave_start + 2]
        write_state(state_path, phase="evaluation", wave=wave_start // 2 + 1, total_waves=(len(jobs) + 1) // 2, active=wave)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            for index, (stage, condition) in enumerate(wave):
                output = root / "evaluation" / stage / condition
                if (output / "CONDITION_EVAL_COMPLETE").is_file():
                    continue
                if output.exists():
                    raise RuntimeError(f"partial evaluation output exists: {output}")
                futures.append(executor.submit(run_condition, stage, condition, output, gpus[index], cpu_sets[index], args, state_path, id_seed, ood_seeds))
            for future in futures:
                future.result()

    (root / "TIMING_EVALUATION_COMPLETE").write_text("complete\n", encoding="utf-8")
    write_state(state_path, phase="complete", marker=str(root / "TIMING_EVALUATION_COMPLETE"), id_seed=id_seed, ood_seeds=ood_seeds, seed_manifest=str(args.seed_manifest.resolve()) if args.seed_manifest else None)


if __name__ == "__main__":
    main()
