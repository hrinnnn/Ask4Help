#!/usr/bin/env python3
"""Continue object-variation pipeline after the formal ID base gate."""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = Path("/data/zhaozhixuan/Ask4Help-airplane-5090/results/object_variation_pick_single_ycb_v1")
PY = Path("/data/zhaozhixuan/Ask4Help-airplane-5090/RLinf/.venv/bin/python")
MODEL_BASE = Path("/data/zhaozhixuan/Ask4Help-open-drawer/results/model_cache/pi05_base_pytorch_v1")
OVERLAY = RUN / "runtime_overlay/numpy126"


def state(**updates: object) -> None:
    path = RUN / "pipeline_state.json"
    payload = json.loads(path.read_text()) if path.exists() else {}
    payload.update(updates)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def env_for(gpu: int | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{OVERLAY}:{ROOT}:{ROOT / 'RLinf'}"
    env["OMP_NUM_THREADS"] = "20"
    env["MKL_NUM_THREADS"] = "20"
    env["TOKENIZERS_PARALLELISM"] = "false"
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["ASK4HELP_RLINF_PLACEMENT"] = f"{gpu}-{gpu}"
    return env


def run(command: list[str], *, log: Path, gpu: int | None = None, check: bool = True) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            ["taskset", "-c", ",".join(str(cpu) for cpu in cpu_sets(1)[0]), *command],
            env=env_for(gpu),
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if check and result.returncode != 0:
        raise RuntimeError(f"stage failed rc={result.returncode}: {' '.join(command)}")
    return result.returncode


def wait_for(path: Path, *, timeout: int = 172800) -> None:
    start = time.time()
    while not path.exists():
        if time.time() - start > timeout:
            raise TimeoutError(f"timed out waiting for {path}")
        time.sleep(300)


def wait_for_id_gate() -> bool:
    start = time.time()
    validated = RUN / "ID_BASE_VALIDATED"
    decision = RUN / "NEEDS_USER_DECISION"
    while not validated.is_file():
        if decision.is_file():
            state(
                current_stage="id_base_not_accepted",
                next_stage="needs_user_decision_or_scientific_recovery",
                downstream_stage="blocked_by_id_gate",
                terminal_marker="NEEDS_USER_DECISION",
            )
            return False
        if time.time() - start > 172800:
            raise TimeoutError(f"timed out waiting for {validated}")
        time.sleep(300)
    return True


def selected_checkpoint() -> Path:
    payload = json.loads((RUN / "id_checkpoint_selection/selection.json").read_text())
    checkpoint = Path(payload["selected"]["checkpoint"])
    if not (checkpoint / "actor/model_state_dict/full_weights.pt").is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def run_parallel(commands: list[tuple], allowed_returncodes: set[int] | None = None) -> None:
    gpus = [item[2] for item in commands]
    if len(gpus) != len(set(gpus)):
        raise RuntimeError(f"parallel stage reuses a GPU: {gpus}")
    cpu_allocations = cpu_sets(len(commands))
    processes = []
    for index, item in enumerate(commands):
        command, log, gpu = item[:3]
        extra_env = item[3] if len(item) > 3 else {}
        log.parent.mkdir(parents=True, exist_ok=True)
        stream = log.open("w", encoding="utf-8")
        process_env = env_for(gpu)
        process_env.update(extra_env)
        processes.append(
            (
                subprocess.Popen(
                    ["taskset", "-c", ",".join(str(cpu) for cpu in cpu_allocations[index]), *command],
                    env=process_env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                ),
                stream,
                command,
            )
        )
    allowed = allowed_returncodes or {0}
    failures = []
    for process, stream, command in processes:
        rc = process.wait()
        stream.close()
        if rc not in allowed:
            failures.append((rc, command))
    if failures:
        raise RuntimeError(f"parallel stage failed: {failures}")


def run_parallel_waves(commands: list[tuple], wave_size: int = 4) -> None:
    """Run independent GPU jobs in bounded waves, never oversubscribing a GPU."""
    for start in range(0, len(commands), wave_size):
        run_parallel(commands[start : start + wave_size])


def cpu_sets(count: int, cores_per_job: int = 20) -> list[list[int]]:
    available = sorted(os.sched_getaffinity(0))
    reserved: set[int] = set()
    for token in os.environ.get("ASK4HELP_RESERVED_CPUSET", "").split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = (int(value) for value in token.split("-", 1))
            reserved.update(range(start, end + 1))
        else:
            reserved.add(int(token))
    available = [cpu for cpu in available if cpu not in reserved]
    if len(available) < count * cores_per_job:
        raise RuntimeError(
            f"need {count * cores_per_job} available CPUs for {count} jobs, found {len(available)}"
        )
    return [
        available[index * cores_per_job : (index + 1) * cores_per_job]
        for index in range(count)
    ]


def select_idle_gpus(count: int = 4, max_used_mib: int = 512) -> list[int]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    idle: list[int] = []
    for line in output.splitlines():
        index, used, utilization = (int(part.strip()) for part in line.split(","))
        if used <= max_used_mib and utilization <= 5:
            idle.append(index)
    if len(idle) < count:
        raise RuntimeError(f"need {count} actually idle GPUs, found {idle}")
    return idle[:count]


def checkpoint_for(output: Path, step: int) -> Path | None:
    candidates = list(
        output.glob(f"**/checkpoints/global_step_{step}/actor/model_state_dict/full_weights.pt")
    )
    return candidates[0].parent.parent.parent if candidates else None


def eval_evidence(output: Path, episodes: int) -> bool:
    summary = output / "summary.json"
    videos = output / "videos"
    if not summary.is_file():
        return False
    try:
        payload = json.loads(summary.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return int(payload.get("episodes", -1)) == episodes and len(list(videos.glob("*.mp4"))) == episodes


def main() -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    state(controller_downstream_pid=os.getpid(), downstream_stage="waiting_for_id_base")
    if not wait_for_id_gate():
        return
    checkpoint = selected_checkpoint()
    norm = RUN / "datasets/id_v1_retry1/norm_stats.json"
    dataset = RUN / "datasets/id_v1_retry1"
    methods = ("bridge_pca", "diffdagger", "failure_recovery", "offline_oracle")

    state(current_stage="detector_assets", next_stage="gate_calibration")
    assets = RUN / "detector_assets_v1"
    if not (assets / "ASSETS_COMPLETE").is_file():
        calibration_rc = run(
            [
                str(PY),
                str(ROOT / "tools/build_pick_single_ycb_object_variation_detector_assets.py"),
                "--checkpoint", str(checkpoint),
                "--pi05-base", str(MODEL_BASE),
                "--norm-stats", str(norm),
                "--dataset-root", str(dataset),
                "--output-dir", str(assets),
            ],
            log=RUN / "logs/detector_assets_v1.log",
            gpu=select_idle_gpus(1)[0],
        )

    state(current_stage="gate_calibration", next_stage="passive_detection")
    calibration = RUN / "gate_calibration_v1/calibration.json"
    if not calibration.is_file():
        calibration.parent.mkdir(parents=True, exist_ok=True)
        run(
            [
                str(PY),
                str(ROOT / "tools/calibrate_pick_single_ycb_object_variation_gates.py"),
                "--checkpoint", str(checkpoint),
                "--pi05-base", str(MODEL_BASE),
                "--norm-stats", str(norm),
                "--detector-assets", str(assets / "detector_assets.pt"),
                "--output", str(calibration),
            ],
            log=RUN / "logs/gate_calibration_v1.log",
            gpu=select_idle_gpus(1)[0],
            check=False,
        )
        if calibration_rc not in {0, -6} or not calibration.is_file():
            raise RuntimeError(f"gate calibration failed rc={calibration_rc} without calibration.json")
        if calibration_rc == -6:
            (calibration.parent / "SIMULATOR_EXIT_AFTER_ARTIFACTS").write_text(
                json.dumps({"returncode": calibration_rc, "accepted_because": "calibration.json is complete"}, indent=2) + "\n",
                encoding="utf-8",
            )

    state(current_stage="passive_detection", next_stage="four_method_collection")
    passive = RUN / "passive_detection_v1"
    if not (passive / "PASSIVE_COMPLETE").is_file():
        passive_gpus = select_idle_gpus(2)
        run_parallel(
            [
                (
                    [
                        str(PY),
                        str(ROOT / "tools/evaluate_pick_single_ycb_object_variation_detectors.py"),
                        "--checkpoint", str(checkpoint),
                        "--pi05-base", str(MODEL_BASE),
                        "--norm-stats", str(norm),
                        "--detector-assets", str(assets / "detector_assets.pt"),
                        "--calibration", str(calibration),
                        "--output-dir", str(passive / "id"),
                        "--split", "id", "--episodes", "100", "--seed", "14000",
                    ],
                    RUN / "logs/passive_id.log",
                    passive_gpus[0],
                ),
                (
                    [
                        str(PY),
                        str(ROOT / "tools/evaluate_pick_single_ycb_object_variation_detectors.py"),
                        "--checkpoint", str(checkpoint),
                        "--pi05-base", str(MODEL_BASE),
                        "--norm-stats", str(norm),
                        "--detector-assets", str(assets / "detector_assets.pt"),
                        "--calibration", str(calibration),
                        "--output-dir", str(passive / "ood"),
                        "--split", "ood", "--episodes", "100", "--seed", "15000",
                    ],
                    RUN / "logs/passive_ood.log",
                    passive_gpus[1],
                ),
            ],
            allowed_returncodes={0, -6},
        )
        if not eval_evidence(passive / "id", 100) or not eval_evidence(passive / "ood", 100):
            raise RuntimeError("passive detector evidence is incomplete after evaluator exit")
        (passive / "PASSIVE_COMPLETE").write_text("complete\n", encoding="utf-8")

    state(current_stage="four_method_collection", next_stage="four_dataset_audit")
    collections = RUN / "collections_v1"
    datasets = RUN / "datasets"
    path_manifest = RUN / "audit/collection_paths_v1.json"
    if path_manifest.is_file():
        collection_paths = json.loads(path_manifest.read_text(encoding="utf-8"))["methods"]
        changed_paths = False
        for method in methods:
            current = collection_paths[method]
            current_collection = Path(current["collection_dir"])
            current_dataset = Path(current["dataset_dir"])
            if (current_collection / "COLLECTION_COMPLETE").is_file() and current_dataset.is_dir():
                continue
            if current_collection.exists() or current_dataset.exists():
                retry = 1
                while True:
                    collection_dir = collections / f"{method}_retry{retry}"
                    dataset_dir = datasets / f"{method}_v1_retry{retry}"
                    if not collection_dir.exists() and not dataset_dir.exists():
                        break
                    retry += 1
                collection_paths[method] = {
                    "collection_dir": str(collection_dir),
                    "dataset_dir": str(dataset_dir),
                }
                changed_paths = True
        if changed_paths:
            path_manifest.write_text(
                json.dumps(
                    {
                        "format": "pick_single_ycb_object_variation_collection_paths_v1",
                        "methods": collection_paths,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
    else:
        collection_paths = {}
        for method in methods:
            base_collection = collections / method
            base_dataset = datasets / f"{method}_v1"
            if (base_collection / "COLLECTION_COMPLETE").is_file() and base_dataset.is_dir():
                collection_dir, dataset_dir = base_collection, base_dataset
            elif base_collection.exists() or base_dataset.exists():
                retry = 1
                while True:
                    collection_dir = collections / f"{method}_retry{retry}"
                    dataset_dir = datasets / f"{method}_v1_retry{retry}"
                    if not collection_dir.exists() and not dataset_dir.exists():
                        break
                    retry += 1
            else:
                collection_dir, dataset_dir = base_collection, base_dataset
            collection_paths[method] = {
                "collection_dir": str(collection_dir),
                "dataset_dir": str(dataset_dir),
            }
        path_manifest.parent.mkdir(parents=True, exist_ok=True)
        path_manifest.write_text(
            json.dumps(
                {
                    "format": "pick_single_ycb_object_variation_collection_paths_v1",
                    "methods": collection_paths,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    method_extras = {
        "bridge_pca": ["--detector-assets", str(assets / "detector_assets.pt"), "--bridge-threshold", str(json.loads(calibration.read_text())["pca_threshold"])],
        "diffdagger": ["--diff-calibration", str(calibration)],
        "failure_recovery": [],
        "offline_oracle": ["--only-split", "ood"],
    }
    missing_methods = [
        method for method in methods
        if not (Path(collection_paths[method]["collection_dir"]) / "COLLECTION_COMPLETE").is_file()
    ]
    # Collection is allowed to run in bounded waves when only two idle GPUs
    # remain because the main ID training keeps one GPU occupied.
    for start in range(0, len(missing_methods), 2):
        wave_methods = missing_methods[start : start + 2]
        wave_gpus = select_idle_gpus(len(wave_methods))
        wave_commands = []
        for method, gpu in zip(wave_methods, wave_gpus):
            out = Path(collection_paths[method]["collection_dir"])
            repo = Path(collection_paths[method]["dataset_dir"])
            common = [str(PY), str(ROOT / "tools/collect_pick_single_ycb_object_variation_gated.py"), "--method", method, "--output-dir", str(out), "--repo-id", str(repo), "--target-expert-trajectories", "100", "--max-attempts", "600", "--id-seed", "16000", "--ood-seed", "16001"]
            if method != "offline_oracle":
                common += ["--checkpoint", str(checkpoint), "--pi05-base", str(MODEL_BASE), "--norm-stats", str(norm)]
            wave_commands.append((common + method_extras[method], RUN / "logs" / f"collect_{method}.log", gpu))
        run_parallel(wave_commands, allowed_returncodes={0, -6})

    state(current_stage="four_dataset_audit", next_stage="matched_budget_selection")
    collection_audit = RUN / "audit/gated_collection_v1/report.json"
    if not (collection_audit.parent / "GATED_COLLECTION_AUDIT_PASSED").is_file():
        run(
            [
                str(PY),
                str(ROOT / "tools/audit_pick_single_ycb_object_variation_gated_collection.py"),
                "--collections-root", str(collections),
                "--datasets-root", str(datasets),
                "--paths-manifest", str(path_manifest),
                "--output", str(collection_audit),
                "--expected", "100",
            ],
            log=RUN / "logs/gated_collection_audit.log",
        )

    budget_root = datasets / "matched_expert_budget_v1"
    budget_marker = budget_root / "BUDGET_SELECTION_COMPLETE"
    if not budget_marker.is_file():
        if budget_root.exists():
            raise RuntimeError(f"partial matched-budget directory exists: {budget_root}")
        run(
            [
                str(PY),
                str(ROOT / "tools/prepare_pick_single_ycb_object_variation_matched_budget.py"),
                "--output-root", str(budget_root),
                "--expert-action-cap", "12000",
                *sum(([f"--{method.replace('_', '-')}", str(Path(collection_paths[method]["dataset_dir"]))] for method in methods), []),
            ],
            log=RUN / "logs/matched_budget_selection.log",
        )
    budget_manifest = json.loads((budget_root / "budget_manifest.json").read_text(encoding="utf-8"))

    state(current_stage="matched_training_smoke", next_stage="matched_training")
    training = RUN / "matched_training_v1"
    training_gpus = select_idle_gpus(4)
    training_gpu_map = dict(zip(methods, training_gpus))
    smoke_commands = []
    for method in methods:
        gpu = training_gpu_map[method]
        expert = budget_root / method
        out = training / f"{method}_smoke_2step"
        env_vars = env_for(gpu)
        env_vars.update({"EMBODIED_PATH": str(ROOT / "RLinf/examples/sft"), "OBJECT_VARIATION_ID_DATASET": str(dataset), "OBJECT_VARIATION_EXPERT_DATASET": str(expert), "OBJECT_VARIATION_NORM_STATS": str(norm), "OBJECT_VARIATION_MODEL_PATH": str(checkpoint), "OBJECT_VARIATION_RUN_ROOT": str(out), "OBJECT_VARIATION_EXPERIMENT_NAME": f"{method}_smoke_2step", "OBJECT_VARIATION_MAX_STEPS": "2", "OBJECT_VARIATION_SAVE_INTERVAL": "2", "OBJECT_VARIATION_TRAIN_SEED": "9206"})
        command = [str(PY), str(ROOT / "RLinf/examples/sft/train_vla_sft.py"), "--config-path", str(ROOT / "RLinf/examples/sft/config"), "--config-name", "pick_single_ycb_object_variation_matched_sft_openpi_pi05", "runner.max_steps=2", "runner.save_interval=2"]
        smoke_commands.append((command, RUN / "logs" / f"matched_{method}_smoke.log", gpu, env_vars))
    if not (training / "MATCHED_SMOKE_COMPLETE").is_file():
        run_parallel(smoke_commands)
        for method in methods:
            if checkpoint_for(training / f"{method}_smoke_2step", 2) is None:
                raise RuntimeError(f"missing 2-step smoke checkpoint: {method}")
        (training / "MATCHED_SMOKE_COMPLETE").parent.mkdir(parents=True, exist_ok=True)
        (training / "MATCHED_SMOKE_COMPLETE").write_text("complete\n", encoding="utf-8")

    reload_commands = []
    reload_root = training / "reload_forward_smoke"
    for method in methods:
        output = reload_root / method
        if eval_evidence(output, 1):
            continue
        checkpoint_path = checkpoint_for(training / f"{method}_smoke_2step", 2)
        if checkpoint_path is None:
            raise RuntimeError(f"missing smoke checkpoint for reload: {method}")
        reload_commands.append(
            (
                [
                    str(PY),
                    str(ROOT / "tools/evaluate_pick_single_ycb_object_variation_pi05.py"),
                    "--checkpoint", str(checkpoint_path),
                    "--pi05-base", str(MODEL_BASE),
                    "--norm-stats", str(norm),
                    "--output-dir", str(output),
                    "--split", "id", "--episodes", "1", "--seed", str(94000 + methods.index(method)),
                    "--execute-horizon", "5", "--max-episode-steps", "20",
                ],
                RUN / "logs" / f"matched_reload_{method}.log",
                training_gpu_map[method],
            )
        )
    if reload_commands:
        run_parallel(reload_commands, allowed_returncodes={0, -6, 120})
    if not all(eval_evidence(reload_root / method, 1) for method in methods):
        raise RuntimeError("matched smoke reload evidence is incomplete")
    (training / "MATCHED_SMOKE_RELOAD_PASSED").write_text("complete\n", encoding="utf-8")

    state(current_stage="matched_training", next_stage="final_evaluation")
    formal_commands = []
    for method in methods:
        gpu = training_gpu_map[method]
        expert = budget_root / method
        out = training / method
        command = [str(PY), str(ROOT / "RLinf/examples/sft/train_vla_sft.py"), "--config-path", str(ROOT / "RLinf/examples/sft/config"), "--config-name", "pick_single_ycb_object_variation_matched_sft_openpi_pi05", "runner.max_steps=5000", "runner.save_interval=500"]
        formal_env = env_for(gpu)
        formal_env.update({"EMBODIED_PATH": str(ROOT / "RLinf/examples/sft"), "OBJECT_VARIATION_ID_DATASET": str(dataset), "OBJECT_VARIATION_EXPERT_DATASET": str(expert), "OBJECT_VARIATION_NORM_STATS": str(norm), "OBJECT_VARIATION_MODEL_PATH": str(checkpoint), "OBJECT_VARIATION_RUN_ROOT": str(out), "OBJECT_VARIATION_EXPERIMENT_NAME": f"{method}_5000", "OBJECT_VARIATION_MAX_STEPS": "5000", "OBJECT_VARIATION_SAVE_INTERVAL": "500", "OBJECT_VARIATION_TRAIN_SEED": "9206"})
        formal_commands.append((command, RUN / "logs" / f"matched_{method}_5000.log", gpu, formal_env))
    if not (training / "FORMAL_TRAINING_COMPLETE").is_file():
        run_parallel(formal_commands)
        required_steps = tuple(range(500, 5001, 500))
        checkpoint_audit = {
            method: {str(step): str(checkpoint_for(training / method, step)) for step in required_steps}
            for method in methods
        }
        if any(value is None for row in checkpoint_audit.values() for value in row.values()):
            (training / "FORMAL_CHECKPOINT_AUDIT_FAILED").write_text(json.dumps(checkpoint_audit, indent=2) + "\n", encoding="utf-8")
            raise RuntimeError("one or more matched-training checkpoints are missing")
        (training / "formal_checkpoint_audit.json").write_text(json.dumps(checkpoint_audit, indent=2) + "\n", encoding="utf-8")
        (training / "FORMAL_TRAINING_COMPLETE").write_text("complete\n", encoding="utf-8")

    state(current_stage="final_evaluation", next_stage="result_registration")
    eval_root = RUN / "final_evaluation_v1"
    for split, seed in (("id", 17000), ("ood", 18000)):
        eval_commands = []
        for method in methods:
            gpu = training_gpu_map[method]
            checkpoint_path = checkpoint_for(training / method, 5000)
            if checkpoint_path is None:
                raise FileNotFoundError(f"missing matched checkpoint: {method}")
            output = eval_root / method / split
            if eval_evidence(output, 100):
                continue
            eval_commands.append(
                (
                    [
                        str(PY),
                        str(ROOT / "tools/evaluate_pick_single_ycb_object_variation_pi05.py"),
                        "--checkpoint", str(checkpoint_path),
                        "--pi05-base", str(MODEL_BASE),
                        "--norm-stats", str(norm),
                        "--output-dir", str(output),
                        "--split", split, "--episodes", "100", "--seed", str(seed),
                        "--execute-horizon", "5", "--max-episode-steps", "200",
                    ],
                    RUN / "logs" / f"final_{method}_{split}.log",
                    gpu,
                )
            )
        if eval_commands:
            run_parallel(eval_commands, allowed_returncodes={0, -6, 120})
        if not all(eval_evidence(eval_root / method / split, 100) for method in methods):
            raise RuntimeError(f"final {split} evaluation evidence is incomplete")

    comparison = {
        "format": "pick_single_ycb_object_variation_final_comparison_v2",
        "checkpoint": str(checkpoint),
        "matched_budget": budget_manifest,
        "methods": {},
    }
    for method in methods:
        comparison["methods"][method] = {}
        for split in ("id", "ood"):
            comparison["methods"][method][split] = json.loads(
                (eval_root / method / split / "summary.json").read_text(encoding="utf-8")
            )
    (RUN / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    markdown = [
        "# PickSingleYCB object-variation final comparison",
        "",
        f"Matched expert-action budget: {budget_manifest['common_expert_action_budget']}",
        "",
        "| Method | ID success | OOD success |",
        "|---|---:|---:|",
    ]
    for method in methods:
        markdown.append(
            f"| {method} | {comparison['methods'][method]['id']['successes']}/100 | {comparison['methods'][method]['ood']['successes']}/100 |"
        )
    (RUN / "comparison.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    (RUN / "PIPELINE_COMPLETE").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    state(current_stage="complete", next_stage=None, terminal_marker="PIPELINE_COMPLETE")


if __name__ == "__main__":
    main()
