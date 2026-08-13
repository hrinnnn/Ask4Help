#!/usr/bin/env python3
"""Train matched-budget X-VLA StackCube Stage-2 groups on two idle GPUs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


METHODS = ("immediate", "post_grasp", "post_lift", "failure_recovery")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def select_idle_gpus(count: int = 2, max_used_mib: int = 128) -> list[int]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    idle = []
    for line in output.splitlines():
        index, used, utilization = (int(part.strip()) for part in line.split(","))
        if used <= max_used_mib and utilization <= 5:
            idle.append(index)
    if len(idle) < count:
        raise RuntimeError(f"need {count} actually idle GPUs, found {idle}")
    return idle[:count]


def build_training_meta(
    method: str,
    *,
    id_meta: Path,
    dataset: Path,
    output: Path,
    expert_action_budget: int,
) -> Path:
    output.mkdir(parents=True, exist_ok=False)
    original = read_json(id_meta)
    original["dataset_name"] = f"panda_stackcube_stage2_id_replay_{method}"
    write_json(output / "00_id_replay.json", original)

    info = read_json(dataset / "meta/info.json")
    episodes = read_jsonl(dataset / "meta/episodes.jsonl")
    action_count = sum(int(row["length"]) for row in episodes)
    if not episodes or action_count != expert_action_budget:
        raise RuntimeError(
            f"{method} dataset has {action_count} actions, expected {expert_action_budget}"
        )
    write_json(
        output / "01_new_expert.json",
        {
            "dataset_name": f"panda_stackcube_stage2_new_expert_{method}",
            "robot_type": "panda_airplane",
            "root_path": str(dataset.resolve()),
            "chunks_size": int(info["chunks_size"]),
            "data_path": info["data_path"],
            "datalist": episodes,
        },
    )
    return output


def temporal_mask_report(episodes: list[dict], horizon: int = 10) -> dict:
    lengths = [int(row["length"]) for row in episodes]
    valid_counts = {str(value): 0 for value in range(1, horizon + 1)}
    for length in lengths:
        for anchor in range(length):
            valid_counts[str(min(horizon, length - anchor))] += 1
    return {
        "episodes": len(lengths),
        "total_anchors": sum(lengths),
        "tail_anchors": sum(min(horizon - 1, length) for length in lengths),
        "valid_target_count_distribution": valid_counts,
        "final_observation_valid_targets": 1,
    }


def training_command(
    python: Path,
    start: Path,
    meta: Path,
    output: Path,
    *,
    steps: int,
    save_interval: int,
    seed: int,
) -> list[str]:
    return [
        str(python),
        "train.py",
        "--models", str(start),
        "--output_dir", str(output),
        "--train_metas_path", str(meta),
        "--batch_size", "8",
        "--num_actions", "10",
        "--num_views", "2",
        "--action_mode", "auto",
        "--real_action_dim", "8",
        "--max_action_dim", "20",
        "--distributed_backend", "gloo",
        "--gradient_checkpointing",
        "--gradient_accumulation_steps", "4",
        "--learning_rate", "1e-4",
        "--learning_coef", "0.1",
        "--weight_decay", "0",
        "--betas", "0.9", "0.95",
        "--max_grad_norm", "1.0",
        "--freeze_steps", "1000",
        "--warmup_steps", "2000",
        "--iters", str(steps),
        "--save_interval", str(save_interval),
        "--log_interval", "20",
        "--seed", str(seed),
    ]


def child_env(
    gpu: int,
    *,
    root: Path,
    port: int,
    cpu_count: int,
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "PYTHONPATH": f"{root}:{root / 'RLinf'}",
            "OMP_NUM_THREADS": str(cpu_count),
            "MKL_NUM_THREADS": str(cpu_count),
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "ACCELERATE_MIXED_PRECISION": "bf16",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
            "LOCAL_WORLD_SIZE": "1",
        }
    )
    return env


def launch_phase(
    *,
    methods: tuple[str, ...],
    gpus: list[int],
    root: Path,
    xvla: Path,
    python: Path,
    start: Path,
    metas: dict[str, Path],
    run: Path,
    phase: str,
    steps: int,
    save_interval: int,
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    available_cpus = sorted(os.sched_getaffinity(0))
    if len(available_cpus) < len(gpus):
        raise RuntimeError(
            f"only {len(available_cpus)} CPUs are available for {len(gpus)} GPUs"
        )
    cpu_sets = [available_cpus[index::len(gpus)] for index in range(len(gpus))]
    for wave_start in range(0, len(methods), len(gpus)):
        wave = methods[wave_start : wave_start + len(gpus)]
        jobs = []
        for slot, (method, gpu) in enumerate(zip(wave, gpus)):
            output = run / phase / method
            if output.exists():
                raise FileExistsError(output)
            output.parent.mkdir(parents=True, exist_ok=True)
            log = run / "logs" / f"{phase}_{method}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("w", encoding="utf-8")
            command = training_command(
                python,
                start,
                metas[method],
                output,
                steps=steps,
                save_interval=save_interval,
                seed=7300,
            )
            process = subprocess.Popen(
                [
                    "taskset", "-c", ",".join(str(cpu) for cpu in cpu_sets[slot]),
                    *command,
                ],
                cwd=xvla,
                env=child_env(
                    gpu,
                    root=root,
                    port=29820 + wave_start + slot,
                    cpu_count=len(cpu_sets[slot]),
                ),
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            write_json(
                run / "pids" / f"{phase}_{method}.json",
                {"pid": process.pid, "gpu": gpu, "cpu_slot": slot},
            )
            jobs.append((method, process, handle))
            outputs[method] = output
        failures = {}
        for method, process, handle in jobs:
            failures[method] = process.wait()
            handle.close()
        if any(status != 0 for status in failures.values()):
            raise RuntimeError(f"{phase} wave failed: {failures}")
    return outputs


def reload_forward_smoke(
    *,
    methods: tuple[str, ...],
    gpus: list[int],
    root: Path,
    xvla: Path,
    python: Path,
    smoke_outputs: dict[str, Path],
    run: Path,
) -> None:
    for wave_start in range(0, len(methods), len(gpus)):
        jobs = []
        for slot, (method, gpu) in enumerate(
            zip(methods[wave_start : wave_start + len(gpus)], gpus)
        ):
            output = run / "reload_forward_smoke" / method
            log = run / "logs" / f"reload_forward_smoke_{method}.log"
            command = [
                str(python),
                str(root / "tools/evaluate_stackcube_xvla.py"),
                "--checkpoint", str(smoke_outputs[method] / "ckpt-2"),
                "--xvla-root", str(xvla),
                "--output-dir", str(output),
                "--episodes", "1",
                "--seed", str(94000 + wave_start + slot),
                "--split", "id",
                "--max-episode-steps", "10",
            ]
            handle = log.open("w", encoding="utf-8")
            process = subprocess.Popen(
                [
                    "taskset", "-c", ",".join(str(cpu) for cpu in cpu_sets[slot]),
                    *command,
                ],
                cwd=root,
                env=child_env(
                    gpu,
                    root=root,
                    port=29920 + wave_start + slot,
                    cpu_count=len(cpu_sets[slot]),
                ),
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            jobs.append((method, output, process, handle))
        for method, output, process, handle in jobs:
            status = process.wait()
            handle.close()
            if status != 0 or not (output / "summary.json").exists():
                raise RuntimeError(f"{method} reload/forward smoke failed: {status}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--start", type=Path, required=True)
    parser.add_argument("--id-meta", type=Path, required=True)
    parser.add_argument("--datasets", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--expert-action-budget", type=int, required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--gpus", type=int, nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run.exists():
        raise FileExistsError(args.run)
    args.run.mkdir(parents=True)
    gpus = args.gpus or select_idle_gpus(2)
    if len(gpus) != 2:
        raise ValueError("Stage-2 training requires exactly two idle GPUs")
    metas = {
        method: build_training_meta(
            method,
            id_meta=args.id_meta,
            dataset=args.datasets / method,
            output=args.run / "metas" / method,
            expert_action_budget=args.expert_action_budget,
        )
        for method in METHODS
    }
    mask_reports = {
        method: temporal_mask_report(
            read_jsonl(args.datasets / method / "meta/episodes.jsonl")
        )
        for method in METHODS
    }
    write_json(
        args.run / "training_contract.json",
        {
            "initial_checkpoint": str(args.start.resolve()),
            "methods": list(METHODS),
            "gpus": gpus,
            "source_sampling": "1:1 original ID replay and group new expert",
            "expert_action_budget_per_group": args.expert_action_budget,
            "per_device_batch": 8,
            "gradient_accumulation_steps": 4,
            "effective_global_batch": 32,
            "formal_steps": args.steps,
            "save_interval": args.save_interval,
            "tail_handling": "repeat final action for storage; action_valid_mask excludes padding",
            "temporal_mask_reports": mask_reports,
        },
    )
    smoke = launch_phase(
        methods=METHODS,
        gpus=gpus,
        root=args.repo,
        xvla=args.xvla_root,
        python=args.python,
        start=args.start,
        metas=metas,
        run=args.run,
        phase="smoke_2",
        steps=2,
        save_interval=2,
    )
    for method, output in smoke.items():
        if not (output / "ckpt-2/model.safetensors").exists():
            raise RuntimeError(f"missing {method} smoke checkpoint")
    reload_forward_smoke(
        methods=METHODS,
        gpus=gpus,
        root=args.repo,
        xvla=args.xvla_root,
        python=args.python,
        smoke_outputs=smoke,
        run=args.run,
    )
    formal = launch_phase(
        methods=METHODS,
        gpus=gpus,
        root=args.repo,
        xvla=args.xvla_root,
        python=args.python,
        start=args.start,
        metas=metas,
        run=args.run,
        phase=f"formal_{args.steps}",
        steps=args.steps,
        save_interval=args.save_interval,
    )
    for method, output in formal.items():
        for step in range(args.save_interval, args.steps + 1, args.save_interval):
            if not (output / f"ckpt-{step}/model.safetensors").exists():
                raise RuntimeError(f"missing {method} checkpoint {step}")
    (args.run / "TRAINING_COMPLETE").write_text("complete\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
