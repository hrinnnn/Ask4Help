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
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["ASK4HELP_RLINF_PLACEMENT"] = f"{gpu}-{gpu}"
    return env


def run(command: list[str], *, log: Path, gpu: int | None = None, check: bool = True) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, env=env_for(gpu), stdout=stream, stderr=subprocess.STDOUT, check=False)
    if check and result.returncode != 0:
        raise RuntimeError(f"stage failed rc={result.returncode}: {' '.join(command)}")
    return result.returncode


def wait_for(path: Path, *, timeout: int = 172800) -> None:
    start = time.time()
    while not path.exists():
        if time.time() - start > timeout:
            raise TimeoutError(f"timed out waiting for {path}")
        time.sleep(300)


def selected_checkpoint() -> Path:
    payload = json.loads((RUN / "id_checkpoint_selection/selection.json").read_text())
    checkpoint = Path(payload["selected"]["checkpoint"])
    if not (checkpoint / "actor/model_state_dict/full_weights.pt").is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def run_parallel(commands: list[tuple]) -> None:
    processes = []
    for item in commands:
        command, log, gpu = item[:3]
        extra_env = item[3] if len(item) > 3 else {}
        log.parent.mkdir(parents=True, exist_ok=True)
        stream = log.open("w", encoding="utf-8")
        process_env = env_for(gpu)
        process_env.update(extra_env)
        processes.append((subprocess.Popen(command, env=process_env, stdout=stream, stderr=subprocess.STDOUT), stream, command))
    failures = []
    for process, stream, command in processes:
        rc = process.wait()
        stream.close()
        if rc != 0:
            failures.append((rc, command))
    if failures:
        raise RuntimeError(f"parallel stage failed: {failures}")


def main() -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    state(controller_downstream_pid=os.getpid(), downstream_stage="waiting_for_id_base")
    wait_for(RUN / "ID_BASE_VALIDATED")
    checkpoint = selected_checkpoint()
    norm = RUN / "datasets/id_v1_retry1/norm_stats.json"
    dataset = RUN / "datasets/id_v1_retry1"

    state(current_stage="detector_assets", next_stage="gate_calibration")
    assets = RUN / "detector_assets_v1"
    if not (assets / "ASSETS_COMPLETE").is_file():
        run([str(PY), str(ROOT / "tools/build_pick_single_ycb_object_variation_detector_assets.py"), "--checkpoint", str(checkpoint), "--pi05-base", str(MODEL_BASE), "--norm-stats", str(norm), "--dataset-root", str(dataset), "--output-dir", str(assets)], log=RUN / "logs/detector_assets_v1.log", gpu=1)

    state(current_stage="gate_calibration", next_stage="passive_detection")
    calibration = RUN / "gate_calibration_v1/calibration.json"
    if not calibration.is_file():
        calibration.parent.mkdir(parents=True, exist_ok=True)
        run([str(PY), str(ROOT / "tools/calibrate_pick_single_ycb_object_variation_gates.py"), "--checkpoint", str(checkpoint), "--pi05-base", str(MODEL_BASE), "--norm-stats", str(norm), "--detector-assets", str(assets / "detector_assets.pt"), "--output", str(calibration)], log=RUN / "logs/gate_calibration_v1.log", gpu=1)

    state(current_stage="passive_detection", next_stage="four_method_collection")
    passive = RUN / "passive_detection_v1"
    run_parallel([
        ([str(PY), str(ROOT / "tools/evaluate_pick_single_ycb_object_variation_detectors.py"), "--checkpoint", str(checkpoint), "--pi05-base", str(MODEL_BASE), "--norm-stats", str(norm), "--detector-assets", str(assets / "detector_assets.pt"), "--calibration", str(calibration), "--output-dir", str(passive / "id"), "--split", "id", "--episodes", "100", "--seed", "14000"], RUN / "logs/passive_id.log", 1),
        ([str(PY), str(ROOT / "tools/evaluate_pick_single_ycb_object_variation_detectors.py"), "--checkpoint", str(checkpoint), "--pi05-base", str(MODEL_BASE), "--norm-stats", str(norm), "--detector-assets", str(assets / "detector_assets.pt"), "--calibration", str(calibration), "--output-dir", str(passive / "ood"), "--split", "ood", "--episodes", "100", "--seed", "15000"], RUN / "logs/passive_ood.log", 3),
    ])

    state(current_stage="four_method_collection", next_stage="four_dataset_audit")
    collections = RUN / "collections_v1"
    datasets = RUN / "datasets"
    method_commands = [
        ("bridge_pca", 1, ["--detector-assets", str(assets / "detector_assets.pt"), "--bridge-threshold", str(json.loads(calibration.read_text())["pca_threshold"])]),
        ("diffdagger", 3, ["--diff-calibration", str(calibration)]),
        ("failure_recovery", 4, []),
        ("offline_oracle", 6, ["--only-split", "ood"]),
    ]
    commands = []
    for method, gpu, extras in method_commands:
        out = collections / method
        repo = datasets / f"{method}_v1"
        common = [str(PY), str(ROOT / "tools/collect_pick_single_ycb_object_variation_gated.py"), "--method", method, "--output-dir", str(out), "--repo-id", str(repo), "--target-expert-trajectories", "100", "--max-attempts", "600", "--id-seed", "16000", "--ood-seed", "16001"]
        if method != "offline_oracle":
            common += ["--checkpoint", str(checkpoint), "--pi05-base", str(MODEL_BASE), "--norm-stats", str(norm)]
        commands.append((common + extras, RUN / "logs" / f"collect_{method}.log", gpu))
    run_parallel(commands)

    for method, _gpu, _extras in method_commands:
        if not (collections / method / "COLLECTION_COMPLETE").is_file():
            raise RuntimeError(f"missing collection completion marker: {method}")

    state(current_stage="matched_training", next_stage="final_evaluation")
    training = RUN / "matched_training_v1"
    smoke_commands = []
    for method, gpu, _extras in method_commands:
        expert = datasets / f"{method}_v1"
        out = training / f"{method}_smoke_2step"
        env_vars = env_for(gpu)
        env_vars.update({"EMBODIED_PATH": str(ROOT / "RLinf/examples/sft"), "OBJECT_VARIATION_ID_DATASET": str(dataset), "OBJECT_VARIATION_EXPERT_DATASET": str(expert), "OBJECT_VARIATION_NORM_STATS": str(norm), "OBJECT_VARIATION_MODEL_PATH": str(checkpoint), "OBJECT_VARIATION_RUN_ROOT": str(out), "OBJECT_VARIATION_EXPERIMENT_NAME": f"{method}_smoke_2step", "OBJECT_VARIATION_MAX_STEPS": "2", "OBJECT_VARIATION_SAVE_INTERVAL": "2", "OBJECT_VARIATION_TRAIN_SEED": "9206"})
        command = [str(PY), str(ROOT / "RLinf/examples/sft/train_vla_sft.py"), "--config-path", str(ROOT / "RLinf/examples/sft/config"), "--config-name", "pick_single_ycb_object_variation_matched_sft_openpi_pi05", "runner.max_steps=2", "runner.save_interval=2"]
        smoke_commands.append((command, RUN / "logs" / f"matched_{method}_smoke.log", gpu, env_vars))
    run_parallel(smoke_commands)

    formal_commands = []
    for method, gpu, _extras in method_commands:
        expert = datasets / f"{method}_v1"
        out = training / method
        command = [str(PY), str(ROOT / "RLinf/examples/sft/train_vla_sft.py"), "--config-path", str(ROOT / "RLinf/examples/sft/config"), "--config-name", "pick_single_ycb_object_variation_matched_sft_openpi_pi05", "runner.max_steps=5000", "runner.save_interval=500"]
        formal_env = env_for(gpu)
        formal_env.update({"EMBODIED_PATH": str(ROOT / "RLinf/examples/sft"), "OBJECT_VARIATION_ID_DATASET": str(dataset), "OBJECT_VARIATION_EXPERT_DATASET": str(expert), "OBJECT_VARIATION_NORM_STATS": str(norm), "OBJECT_VARIATION_MODEL_PATH": str(checkpoint), "OBJECT_VARIATION_RUN_ROOT": str(out), "OBJECT_VARIATION_EXPERIMENT_NAME": f"{method}_5000", "OBJECT_VARIATION_MAX_STEPS": "5000", "OBJECT_VARIATION_SAVE_INTERVAL": "500", "OBJECT_VARIATION_TRAIN_SEED": "9206"})
        formal_commands.append((command, RUN / "logs" / f"matched_{method}_5000.log", gpu, formal_env))
    run_parallel(formal_commands)

    state(current_stage="final_evaluation", next_stage="result_registration")
    eval_commands = []
    for method, gpu, _extras in method_commands:
        checkpoint_path = next((p.parent.parent.parent for p in (training / method).glob("**/checkpoints/global_step_5000/actor/model_state_dict/full_weights.pt")), None)
        if checkpoint_path is None:
            raise FileNotFoundError(f"missing matched checkpoint: {method}")
        for split, seed in (("id", 17000), ("ood", 18000)):
            output = RUN / "final_evaluation_v1" / method / split
            command = [str(PY), str(ROOT / "tools/evaluate_pick_single_ycb_object_variation_pi05.py"), "--checkpoint", str(checkpoint_path), "--pi05-base", str(MODEL_BASE), "--norm-stats", str(norm), "--output-dir", str(output), "--split", split, "--episodes", "100", "--seed", str(seed), "--execute-horizon", "5", "--max-episode-steps", "200"]
            eval_commands.append((command, RUN / "logs" / f"final_{method}_{split}.log", gpu))
    run_parallel(eval_commands)
    comparison = {"format": "pick_single_ycb_object_variation_final_comparison_v1", "methods": {}}
    for method, _gpu, _extras in method_commands:
        comparison["methods"][method] = {}
        for split in ("id", "ood"):
            comparison["methods"][method][split] = json.loads((RUN / "final_evaluation_v1" / method / split / "summary.json").read_text())
    (RUN / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    (RUN / "PIPELINE_COMPLETE").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    state(current_stage="complete", next_stage=None, terminal_marker="PIPELINE_COMPLETE")


if __name__ == "__main__":
    main()
