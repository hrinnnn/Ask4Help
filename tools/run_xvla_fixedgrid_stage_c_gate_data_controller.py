#!/usr/bin/env python3
"""Run Stage-C gate-selected collection, matched-budget training, and utility eval.

This controller is intentionally downstream of the passive Stage-C audit.  It
does not retune thresholds or choose a method from OOD outcomes.  Each gate
uses the frozen validation-ID q=.95 calibration, collects a frozen training
seed range, selects only complete expert episodes at the resolved task budget,
and then evaluates three independent 2500-step policies on frozen ID/OOD
seeds.
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
METHODS = ["input_pca", "bridge_pca", "action_pca", "diffdagger", "failure_recovery"]
TRAINING_SEEDS = [17001, 17002, 17003]
EXPECTED_COLLECTION_SEEDS = 400
EVAL_EPISODES = 100
EVAL_SEEDS = {
    "stackcube": {"id": 155000, "ood": 156000},
    "airplane": {"id": 165000, "ood": 166000},
}
TASKS = {
    "stackcube": {
        "budget": 520,
        "anchors": ["input_pca", "bridge_pca", "action_pca", "diffdagger", "failure_recovery"],
        "seed_start": 154000,
        "seed_end": 154399,
    },
    "airplane": {
        "budget": 2820,
        "anchors": ["input_pca", "bridge_pca", "action_pca", "diffdagger", "failure_recovery"],
        "seed_start": 164000,
        "seed_end": 164399,
    },
}
INTERNAL_LAYERS = {
    "input_pca": "vlm_input_pool",
    "bridge_pca": "vlm_action_bridge",
    "action_pca": "action_block_01",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def make_seed_manifest(path: Path, start: int, end: int) -> Path:
    expected = list(range(start, end + 1))
    if path.exists():
        payload = read_json(path)
        if [int(seed) for seed in payload.get("seeds", [])] != expected:
            raise RuntimeError(f"seed manifest mismatch: {path}")
        return path
    write_json(
        path,
        {
            "format": "xvla_fixedgrid_stage_c_gate_training_seed_manifest_v1",
            "start": start,
            "end": end,
            "seeds": expected,
        },
    )
    return path


def audit_seed_collisions(manifest: Path, new_ranges: list[tuple[int, int]]) -> None:
    payload = read_json(manifest)
    ranges: list[tuple[int, int]] = []
    for task in payload.get("tasks", {}).values():
        for key, value in task.get("seed_ranges", {}).items():
            if not isinstance(value, list) or len(value) != 2:
                continue
            ranges.append((int(value[0]), int(value[1])))
        for key in ("threshold_validation_id", "calibration_ood", "calibration_extension_ood", "gate_ood", "final_id", "final_ood"):
            value = task.get(key)
            if isinstance(value, list) and len(value) == 2:
                ranges.append((int(value[0]), int(value[1])))
    for start, end in new_ranges:
        if start > end:
            raise ValueError(f"invalid new seed range {start}--{end}")
        for old_start, old_end in ranges:
            if max(start, old_start) <= min(end, old_end):
                raise RuntimeError(
                    f"Stage-C gate training seed collision: {start}--{end} "
                    f"overlaps {old_start}--{old_end}"
                )
    if new_ranges[0][0] <= new_ranges[1][1] and new_ranges[1][0] <= new_ranges[0][1]:
        raise RuntimeError("StackCube and Airplane gate training seed ranges overlap")


def check_gpu(gpu: int) -> None:
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    values = {int(index.strip()): int(used.strip()) for index, used in (line.split(",") for line in rows.splitlines())}
    if values.get(gpu, 2048) > 1024:
        raise RuntimeError(f"GPU {gpu} is not idle: {values.get(gpu)} MiB")
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    uuid = subprocess.check_output(
        ["nvidia-smi", "--id", str(gpu), "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    if any(line.startswith(uuid) for line in apps.splitlines() if line.strip()):
        raise RuntimeError(f"GPU {gpu} has a compute process")


def wait_for_gpu(args: argparse.Namespace, state: dict[str, Any], state_path: Path, label: str) -> None:
    while True:
        try:
            check_gpu(args.gpu)
            return
        except (OSError, subprocess.CalledProcessError, RuntimeError, ValueError) as exc:
            state.update({"stage": "stage_c_gate_waiting_for_gpu", "waiting_for": label, "reason": str(exc), "updated_at": now()})
            write_json(state_path, state)
            time.sleep(args.interval_seconds)


def run_command(
    *,
    command: list[str],
    repo: Path,
    log: Path,
    args: argparse.Namespace,
    state: dict[str, Any],
    state_path: Path,
    label: str,
    allowed_returncodes: tuple[int, ...] = (0,),
) -> None:
    wait_for_gpu(args, state, state_path, label)
    log.parent.mkdir(parents=True, exist_ok=True)
    state.update({"stage": label, "command": command, "updated_at": now()})
    write_json(state_path, state)
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(args.gpu),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "20",
        "MKL_NUM_THREADS": "20",
    }
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            ["taskset", "-c", args.cpu_set, *command],
            cwd=repo,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode not in allowed_returncodes:
        state.update({"stage": "stage_c_gate_failed", "failed_step": label, "returncode": result.returncode, "updated_at": now()})
        write_json(state_path, state)
        raise RuntimeError(f"{label} failed; see {log}")


def collection_complete(output: Path, seed_count: int) -> bool:
    summary_path = output / "summary.json"
    episodes_path = output / "episodes.jsonl"
    training_path = output / "training_episodes.jsonl"
    if not summary_path.is_file() or not episodes_path.is_file() or not training_path.is_file():
        return False
    summary = read_json(summary_path)
    rows = [line for line in episodes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    training = [line for line in training_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return int(summary.get("raw_total", -1)) == seed_count and len(rows) == seed_count and len(training) == int(summary.get("accepted_total", -1))


def selected_complete(path: Path, budget: int) -> bool:
    info = path / "meta/info.json"
    manifest = path / "selection_manifest.json"
    return info.is_file() and manifest.is_file() and int(read_json(info).get("total_frames", -1)) == budget


def collection_command(
    *,
    task: str,
    method: str,
    args: argparse.Namespace,
    calibration: Path,
    seed_manifest: Path,
    output: Path,
    dataset: Path,
) -> list[str]:
    if task == "stackcube":
        command = [
            args.python,
            str(args.repo / "tools/collect_stackcube_xvla_dagger.py"),
            "--method", method,
            "--checkpoint", str(args.stackcube_checkpoint),
            "--xvla-root", str(args.xvla_root),
            "--internal-assets", str(args.stackcube_internal_assets),
            "--calibration", str(calibration),
            "--output-dir", str(output),
            "--repo-id", str(dataset),
            "--target", "100",
            "--pool-action-target", str(TASKS[task]["budget"]),
            "--seed-manifest", str(seed_manifest),
            "--consume-all-seeds",
            "--ood-split", "ood",
            "--flow-steps", "10",
            "--internal-layer", INTERNAL_LAYERS.get(method, "vlm_action_bridge"),
        ]
        return command
    command = [
        args.python,
        str(args.repo / "tools/collect_pick_single_ycb_airplane_xvla_dagger.py"),
        "--method", method,
        "--checkpoint", str(args.airplane_checkpoint),
        "--xvla-root", str(args.xvla_root),
        "--calibration", str(calibration),
        "--pca-asset", str(args.airplane_pca_asset),
        "--multilayer-assets", str(args.airplane_internal_assets),
        "--internal-layer", INTERNAL_LAYERS.get(method, "vlm_input_pool"),
        "--output-dir", str(output),
        "--repo-id", str(dataset),
        "--target", "100",
        "--pool-action-target", str(TASKS[task]["budget"]),
        "--seed-manifest", str(seed_manifest),
        "--consume-all-seeds",
        "--only-split", "ood",
        "--admission-endpoint", "ever_grasped",
        "--flow-steps", "10",
    ]
    return command


def eval_command(task: str, checkpoint: Path, output: Path, split: str, seed: int, args: argparse.Namespace) -> list[str]:
    evaluator = "evaluate_stackcube_xvla.py" if task == "stackcube" else "evaluate_pick_single_ycb_airplane_xvla.py"
    return [
        args.python,
        str(args.repo / "tools" / evaluator),
        "--checkpoint", str(checkpoint),
        "--xvla-root", str(args.xvla_root),
        "--output-dir", str(output),
        "--episodes", str(EVAL_EPISODES),
        "--seed", str(seed),
        "--split", split,
        "--max-episode-steps", "150",
        "--flow-steps", "10",
    ]


def run(args: argparse.Namespace) -> None:
    args.run_root.mkdir(parents=True, exist_ok=True)
    state_path = args.run_root / "pipeline_state.json"
    state = read_json(state_path) if state_path.exists() else {
        "pipeline_id": "xvla_fixedgrid_taskpolicy_knee_v1",
        "stage": "stage_c_gate_data_waiting_for_passive_audit",
        "started_at": now(),
        "completed_collections": [],
        "completed_selections": [],
        "completed_training": [],
        "completed_evaluations": [],
    }
    for key in ("completed_collections", "completed_selections", "completed_training", "completed_evaluations"):
        state.setdefault(key, [])
    write_json(state_path, state)
    passive_marker = args.passive_root / "STAGE_C_GATE_AUDIT_COMPLETE_DATA_PENDING"
    while not passive_marker.is_file():
        state.update({"stage": "stage_c_gate_data_waiting_for_passive_audit", "updated_at": now()})
        write_json(state_path, state)
        time.sleep(args.interval_seconds)

    pipeline_manifest = read_json(args.manifest)
    audit_seed_collisions(
        args.manifest,
        [(TASKS["stackcube"]["seed_start"], TASKS["stackcube"]["seed_end"]), (TASKS["airplane"]["seed_start"], TASKS["airplane"]["seed_end"])],
    )
    for task in TASKS:
        seed_manifest = make_seed_manifest(
            args.run_root / "manifests" / f"{task}_gate_training.json",
            TASKS[task]["seed_start"], TASKS[task]["seed_end"],
        )
        calibration = args.passive_root / task / "calibration_q95.json"
        if not calibration.is_file():
            raise FileNotFoundError(calibration)
        for method in METHODS:
            key = f"{task}/{method}"
            collection = args.run_root / "collections" / task / method
            dataset = args.run_root / "datasets" / task / method / "pool"
            if key in state["completed_collections"]:
                continue
            if collection.exists() or dataset.exists():
                if not collection_complete(collection, EXPECTED_COLLECTION_SEEDS):
                    raise FileExistsError(f"partial gate collection: {collection}")
            else:
                command = collection_command(
                    task=task, method=method, args=args, calibration=calibration,
                    seed_manifest=seed_manifest, output=collection, dataset=dataset,
                )
                run_command(
                    command=command, repo=args.repo,
                    log=args.run_root / "logs" / f"collect_{task}_{method}.log",
                    args=args, state=state, state_path=state_path,
                    label=f"stage_c_gate_collect_{task}_{method}",
                    allowed_returncodes=(0, -6),
                )
            if not collection_complete(collection, EXPECTED_COLLECTION_SEEDS):
                raise RuntimeError(f"incomplete gate collection evidence: {collection}")
            state["completed_collections"].append(key)
            state.update({"stage": "stage_c_gate_collecting", "updated_at": now()})
            write_json(state_path, state)

            selected = args.run_root / "datasets" / task / method / "selected"
            if key not in state["completed_selections"]:
                if selected.exists():
                    if not selected_complete(selected, int(TASKS[task]["budget"])):
                        raise FileExistsError(f"partial gate selection: {selected}")
                else:
                    command = [
                        args.python,
                        str(args.repo / "tools/select_xvla_fixedgrid_gate_budget.py"),
                        "--pool", str(dataset),
                        "--collection", str(collection),
                        "--allowed-seeds", str(seed_manifest),
                        "--output", str(selected),
                        "--budget", str(TASKS[task]["budget"]),
                        "--task", task,
                        "--method", method,
                    ]
                    run_command(
                        command=command, repo=args.repo,
                        log=args.run_root / "logs" / f"select_{task}_{method}.log",
                        args=args, state=state, state_path=state_path,
                        label=f"stage_c_gate_select_{task}_{method}",
                    )
                if not selected_complete(selected, int(TASKS[task]["budget"])):
                    raise RuntimeError(f"selected gate dataset failed budget audit: {selected}")
                state["completed_selections"].append(key)
                state.update({"stage": "stage_c_gate_selecting", "updated_at": now()})
                write_json(state_path, state)

    (args.run_root / "STAGE_C_GATE_DATA_COMPLETE").write_text("collection and exact-budget selection complete\n", encoding="utf-8")
    state.update({"stage": "stage_c_gate_data_complete", "updated_at": now()})
    write_json(state_path, state)

    for task in TASKS:
        for method in METHODS:
            for seed in TRAINING_SEEDS:
                key = f"{task}/{method}/{seed}"
                output = args.run_root / "training" / task / method / f"seed_{seed}"
                checkpoint = output / "train" / "ckpt-2500" / "model.safetensors"
                if key not in state["completed_training"]:
                    if output.exists():
                        if not (output / "TRAINING_COMPLETE").is_file():
                            raise FileExistsError(f"partial gate training output: {output}")
                    else:
                        command = [
                            args.python,
                            str(args.repo / "tools/run_xvla_fixedgrid_timing_training.py"),
                            "--task", task,
                            "--anchor", f"gate_{method}",
                            "--seed", str(seed),
                            "--dataset-root", str(args.run_root / "datasets" / task / method / "selected"),
                            "--expected-budget", str(TASKS[task]["budget"]),
                            "--output", str(output),
                            "--repo", str(args.repo),
                            "--xvla-root", str(args.xvla_root),
                            "--python", args.python,
                            "--gpu", str(args.gpu),
                            "--cpu-set", args.cpu_set,
                            "--steps", "2500",
                            "--save-interval", "500",
                        ]
                        run_command(
                            command=command, repo=args.repo,
                            log=args.run_root / "logs" / f"train_{task}_{method}_{seed}.log",
                            args=args, state=state, state_path=state_path,
                            label=f"stage_c_gate_train_{task}_{method}_{seed}",
                        )
                    if not checkpoint.is_file() or not (output / "TRAINING_COMPLETE").is_file():
                        raise RuntimeError(f"gate training missing checkpoint: {output}")
                    state["completed_training"].append(key)
                    state.update({"stage": "stage_c_gate_training", "updated_at": now()})
                    write_json(state_path, state)
    (args.run_root / "STAGE_C_GATE_TRAINING_COMPLETE").write_text("training complete\n", encoding="utf-8")

    for task in TASKS:
        for method in METHODS:
            for seed in TRAINING_SEEDS:
                checkpoint = args.run_root / "training" / task / method / f"seed_{seed}" / "train/ckpt-2500"
                for split in ("id", "ood"):
                    key = f"{task}/{method}/{seed}/{split}"
                    if key in state["completed_evaluations"]:
                        continue
                    output = args.run_root / "evaluation" / task / method / f"seed_{seed}" / split
                    summary = output / "summary.json"
                    if output.exists():
                        if not summary.is_file() or int(read_json(summary).get("episodes", -1)) != EVAL_EPISODES:
                            raise FileExistsError(f"partial gate evaluation output: {output}")
                    else:
                        command = eval_command(
                            task, checkpoint, output, split, EVAL_SEEDS[task][split], args
                        )
                        run_command(
                            command=command, repo=args.repo,
                            log=args.run_root / "logs" / f"eval_{task}_{method}_{seed}_{split}.log",
                            args=args, state=state, state_path=state_path,
                            label=f"stage_c_gate_eval_{task}_{method}_{seed}_{split}",
                        )
                    if not summary.is_file() or int(read_json(summary).get("episodes", -1)) != EVAL_EPISODES:
                        raise RuntimeError(f"gate evaluation missing denominator: {output}")
                    state["completed_evaluations"].append(key)
                    state.update({"stage": "stage_c_gate_evaluation", "updated_at": now()})
                    write_json(state_path, state)
    (args.run_root / "STAGE_C_GATE_EVALUATION_COMPLETE").write_text("evaluation complete\n", encoding="utf-8")

    for task in TASKS:
        for split in ("id", "ood"):
            output = args.run_root / "utility_summaries" / f"{task}_{split}.json"
            if output.is_file():
                continue
            command = [
                args.python,
                str(args.repo / "tools/summarize_xvla_stage_c_gate_utility.py"),
                "--evaluation-root", str(args.run_root / "evaluation"),
                "--dataset-root", str(args.run_root / "datasets"),
                "--task", task,
                "--methods", *METHODS,
                "--seeds", *[str(seed) for seed in TRAINING_SEEDS],
                "--budget", str(TASKS[task]["budget"]),
                "--split", split,
                "--output", str(output),
            ]
            # The summary is CPU-only; it does not need the experiment GPU.
            output.parent.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(command, cwd=args.repo, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                state.update({"stage": "stage_c_gate_failed", "failed_step": f"utility_{task}_{split}", "updated_at": now()})
                write_json(state_path, state)
                raise RuntimeError(result.stderr or result.stdout)
    (args.run_root / "STAGE_C_GATE_UTILITY_COMPLETE").write_text("utility summaries complete\n", encoding="utf-8")
    state.update({"stage": "stage_c_gate_utility_complete", "completed_at": now(), "updated_at": now()})
    write_json(state_path, state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--passive-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--cpu-set", default="0-19")
    parser.add_argument("--interval-seconds", type=int, default=900)
    parser.add_argument("--stackcube-checkpoint", type=Path, required=True)
    parser.add_argument("--stackcube-internal-assets", type=Path, required=True)
    parser.add_argument("--airplane-checkpoint", type=Path, required=True)
    parser.add_argument("--airplane-internal-assets", type=Path, required=True)
    parser.add_argument("--airplane-pca-asset", type=Path, required=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
