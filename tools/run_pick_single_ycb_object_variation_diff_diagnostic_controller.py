#!/usr/bin/env python3
"""Run the explicitly diagnostic low-threshold Diff-DAgger continuation.

The canonical object-variation pipeline is intentionally left untouched. This
controller consumes the three already-audited canonical datasets plus a new
low-threshold Diff dataset, then runs an independently rooted matched-budget
training/evaluation comparison.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from tools.run_pick_single_ycb_object_variation_downstream_controller import (
    MODEL_BASE,
    OVERLAY,
    PY,
    ROOT,
    checkpoint_for,
    env_for,
    eval_evidence,
    run,
    run_parallel,
    select_idle_gpus,
    wait_for_idle_gpus,
)


BASE = Path("/data/zhaozhixuan/Ask4Help-airplane-5090/results/object_variation_pick_single_ycb_v1")
RUN = Path(
    os.environ.get(
        "OBJECT_VARIATION_DIAGNOSTIC_RUN_ROOT",
        str(BASE / "diagnostic_low_threshold_005_v1"),
    )
)
CHECKPOINT = BASE / (
    "id_training_v1/formal_10000_retry8/id_sft_10000_retry8/"
    "id_sft_10000_retry7_weights_only/checkpoints/global_step_4500"
)
NORM = BASE / "datasets/id_v1_retry1/norm_stats.json"
ID_DATASET = BASE / "datasets/id_v1_retry1"
COLLECTION_ROOT = BASE / "collections_diagnostic_v1"
DATASET_ROOT = BASE / "datasets"
CALIBRATION = BASE / "gate_calibration_v1/calibration.json"
METHODS = ("bridge_pca", "diffdagger", "failure_recovery", "offline_oracle")


def write_state(**updates: object) -> None:
    path = RUN / "pipeline_state.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    payload.update(updates)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def alive(pid_path: Path) -> bool:
    if not pid_path.is_file():
        return False
    try:
        pid = int(pid_path.read_text().strip())
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def diff_manifest_payload(dataset: Path, threshold: float) -> dict[str, object]:
    return {
        "format": "pick_single_ycb_object_variation_diffdagger_low_threshold_diagnostic_v1",
        "method": "diffdagger",
        "checkpoint": str(CHECKPOINT),
        "canonical_calibration": str(CALIBRATION),
        "canonical_threshold": 0.6456781893968583,
        "override_threshold": threshold,
        "q": 0.95,
        "patience": 2,
        "id_seed_start": 16000,
        "ood_seed_start": 16001,
        "alternating": True,
        "target_accepted": 100,
        "max_attempts": 600,
        "execute_horizon": 5,
        "dataset_dir": str(dataset),
        "canonical_collection_preserved": True,
    }


def write_diff_manifest(output: Path, dataset: Path, threshold: float) -> None:
    output.mkdir(parents=True, exist_ok=True)
    manifest = diff_manifest_payload(dataset, threshold)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def launch_diff_collection(output: Path, dataset: Path, threshold: float, suffix: str) -> None:
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to reuse non-empty diagnostic output: {output}")
    # The collector refuses an already-existing output directory. Keep the
    # request manifest beside it, then attach the same manifest after the
    # collector has created the directory itself.
    request_manifest = output.parent / f"{output.name}.request.json"
    request_manifest.write_text(
        json.dumps(diff_manifest_payload(dataset, threshold), indent=2) + "\n",
        encoding="utf-8",
    )
    gpu = select_idle_gpus(1)[0]
    cpus = list(range(0, 20))
    log = BASE / "logs" / f"diffdagger_low_threshold_{suffix}.log"
    pid_path = output.with_suffix(".pid")
    env = env_for(gpu)
    env.update({"RLINF_RAY_ADDRESS": "local", "RAY_TMPDIR": f"/sdd/object_variation_diff_{suffix}"})
    command = [
        str(PY),
        str(ROOT / "tools/collect_pick_single_ycb_object_variation_gated.py"),
        "--method", "diffdagger",
        "--checkpoint", str(CHECKPOINT),
        "--pi05-base", str(MODEL_BASE),
        "--norm-stats", str(NORM),
        "--diff-calibration", str(CALIBRATION),
        "--diff-threshold", str(threshold),
        "--output-dir", str(output),
        "--repo-id", str(dataset),
        "--target-expert-trajectories", "100",
        "--max-attempts", "600",
        "--id-seed", "16000",
        "--ood-seed", "16001",
        "--execute-horizon", "5",
    ]
    log.parent.mkdir(parents=True, exist_ok=True)
    stream = log.open("w", encoding="utf-8")
    process = subprocess.Popen(
        ["taskset", "-c", ",".join(str(cpu) for cpu in cpus), *command],
        env=env,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    pid_path.write_text(str(process.pid) + "\n", encoding="utf-8")
    stream.close()
    for _ in range(100):
        if output.is_dir():
            (output / "manifest.json").write_text(request_manifest.read_text(encoding="utf-8"), encoding="utf-8")
            break
        time.sleep(0.1)


def wait_for_diff() -> tuple[Path, Path, float]:
    candidates = (
        ("005_retry1", COLLECTION_ROOT / "diffdagger_low_threshold_005_retry1", DATASET_ROOT / "diffdagger_low_threshold_005_retry1_v1", 0.05),
        ("005", COLLECTION_ROOT / "diffdagger_low_threshold_005", DATASET_ROOT / "diffdagger_low_threshold_005_v1", 0.05),
        ("002", COLLECTION_ROOT / "diffdagger_low_threshold_002", DATASET_ROOT / "diffdagger_low_threshold_002_v1", 0.02),
        ("001", COLLECTION_ROOT / "diffdagger_low_threshold_001", DATASET_ROOT / "diffdagger_low_threshold_001_v1", 0.01),
    )
    for suffix, output, dataset, threshold in candidates:
        pid_path = output.with_suffix(".pid")
        if not output.exists() and suffix != "005":
            launch_diff_collection(output, dataset, threshold, suffix)
        while True:
            if (output / "COLLECTION_COMPLETE").is_file():
                return output, dataset, threshold
            if (output / "COLLECTION_FAILED").is_file():
                break
            if alive(pid_path):
                time.sleep(300)
                continue
            if not output.exists() or not any(output.iterdir()):
                if suffix == "005":
                    launch_diff_collection(output, dataset, threshold, suffix)
                    continue
            break
    raise RuntimeError("all diagnostic Diff thresholds failed to collect 100 accepted trajectories")


def training_env(gpu: int, method: str, expert: Path, output: Path, steps: int) -> dict[str, str]:
    env = env_for(gpu)
    ray_roots = {
        "bridge_pca": "/tmp/ovbp",
        "failure_recovery": "/tmp/ovfr",
        "offline_oracle": "/tmp/ovbc",
        "diffdagger": "/tmp/ovdiff",
    }
    env.update(
        {
            "RLINF_RAY_ADDRESS": "local",
            "RAY_TMPDIR": ray_roots[method],
            "TMPDIR": ray_roots[method],
            "EMBODIED_PATH": str(ROOT / "RLinf/examples/sft"),
            "OBJECT_VARIATION_ID_DATASET": str(ID_DATASET),
            "OBJECT_VARIATION_EXPERT_DATASET": str(expert),
            "OBJECT_VARIATION_NORM_STATS": str(NORM),
            "OBJECT_VARIATION_MODEL_PATH": str(CHECKPOINT),
            "OBJECT_VARIATION_RUN_ROOT": str(output),
            "OBJECT_VARIATION_EXPERIMENT_NAME": f"{method}_{steps}",
            "OBJECT_VARIATION_MAX_STEPS": str(steps),
            "OBJECT_VARIATION_SAVE_INTERVAL": str(steps if steps == 2 else 500),
            "OBJECT_VARIATION_TRAIN_SEED": "9206",
        }
    )
    env.pop("RAY_ADDRESS", None)
    return env


def run_method_waves(methods: tuple[str, ...], builder, *, allowed_returncodes: set[int] | None = None, max_parallel: int = 3) -> None:
    """Run method branches in bounded waves when fewer than four GPUs are free."""

    for start in range(0, len(methods), max_parallel):
        wave_methods = methods[start : start + max_parallel]
        wave_gpus = wait_for_idle_gpus(len(wave_methods))
        run_parallel(
            [builder(method, gpu) for method, gpu in zip(wave_methods, wave_gpus)],
            allowed_returncodes=allowed_returncodes,
        )


def main() -> None:
    if (RUN / "PIPELINE_COMPLETE").is_file():
        raise RuntimeError(f"diagnostic pipeline already complete: {RUN}")
    RUN.mkdir(parents=True, exist_ok=True)
    write_state(
        format="pick_single_ycb_object_variation_low_threshold_diagnostic_controller_v1",
        authorized=True,
        canonical_pipeline=str(BASE),
        current_stage="waiting_for_low_threshold_diff_collection",
        next_stage="collection_audit",
        checkpoint=str(CHECKPOINT),
        canonical_diff_preserved=True,
        controller_pid=os.getpid(),
    )

    diff_collection, diff_dataset, diff_threshold = wait_for_diff()
    paths = {
        "bridge_pca": {
            "collection_dir": str(BASE / "collections_v1/bridge_pca_retry1"),
            "dataset_dir": str(BASE / "datasets/bridge_pca_v1_retry1"),
        },
        "diffdagger": {"collection_dir": str(diff_collection), "dataset_dir": str(diff_dataset)},
        "failure_recovery": {
            "collection_dir": str(BASE / "collections_v1/failure_recovery"),
            "dataset_dir": str(BASE / "datasets/failure_recovery_v1"),
        },
        "offline_oracle": {
            "collection_dir": str(BASE / "collections_v1/offline_oracle"),
            "dataset_dir": str(BASE / "datasets/offline_oracle_v1"),
        },
    }
    audit_root = RUN / "collection_audit"
    audit_root.mkdir(parents=True, exist_ok=True)
    paths_manifest = audit_root / "collection_paths.json"
    paths_manifest.write_text(json.dumps({"format": "diagnostic_collection_paths_v1", "methods": paths}, indent=2) + "\n", encoding="utf-8")
    write_state(current_stage="collection_audit", next_stage="matched_budget")
    run(
        [
            str(PY), str(ROOT / "tools/audit_pick_single_ycb_object_variation_gated_collection.py"),
            "--collections-root", str(COLLECTION_ROOT), "--datasets-root", str(DATASET_ROOT),
            "--paths-manifest", str(paths_manifest), "--output", str(audit_root / "report.json"), "--expected", "100",
        ],
        log=RUN / "collection_audit.log",
    )

    budget_root = RUN / "matched_expert_budget"
    if budget_root.exists() and not (budget_root / "BUDGET_SELECTION_COMPLETE").is_file():
        budget_root = RUN / "matched_expert_budget_retry1"
    write_state(current_stage="matched_budget", next_stage="matched_training_smoke")
    if not (budget_root / "BUDGET_SELECTION_COMPLETE").is_file():
        run(
            [
                str(PY), str(ROOT / "tools/prepare_pick_single_ycb_object_variation_matched_budget.py"),
                "--output-root", str(budget_root), "--expert-action-cap", "12000",
                "--bridge-pca", paths["bridge_pca"]["dataset_dir"],
                "--diffdagger", paths["diffdagger"]["dataset_dir"],
                "--failure-recovery", paths["failure_recovery"]["dataset_dir"],
                "--offline-oracle", paths["offline_oracle"]["dataset_dir"],
            ],
            log=RUN / "matched_budget.log",
        )
    budget_manifest = json.loads((budget_root / "budget_manifest.json").read_text(encoding="utf-8"))

    training = RUN / "matched_training"

    def smoke_builder(method: str, gpu: int):
        output = training / f"{method}_smoke_2step"
        command = [str(PY), str(ROOT / "RLinf/examples/sft/train_vla_sft.py"), "--config-path", str(ROOT / "RLinf/examples/sft/config"), "--config-name", "pick_single_ycb_object_variation_matched_sft_openpi_pi05", "runner.max_steps=2", "runner.save_interval=2"]
        return command, RUN / "logs" / f"matched_{method}_smoke.log", gpu, training_env(gpu, method, budget_root / method, output, 2)

    write_state(current_stage="matched_training_smoke", next_stage="matched_training")
    run_method_waves(METHODS, smoke_builder, max_parallel=1)
    for method in METHODS:
        if checkpoint_for(training / f"{method}_smoke_2step", 2) is None:
            raise RuntimeError(f"missing diagnostic 2-step checkpoint: {method}")
    (training / "MATCHED_SMOKE_COMPLETE").write_text("complete\n", encoding="utf-8")

    reload_root = training / "reload_forward_smoke"

    def reload_builder(method: str, gpu: int):
        output = reload_root / method
        command = [
            str(PY), str(ROOT / "tools/evaluate_pick_single_ycb_object_variation_pi05.py"),
            "--checkpoint", str(checkpoint_for(training / f"{method}_smoke_2step", 2)),
            "--pi05-base", str(MODEL_BASE), "--norm-stats", str(NORM), "--output-dir", str(output),
            "--split", "id", "--episodes", "1", "--seed", str(94000 + METHODS.index(method)),
            "--execute-horizon", "5", "--max-episode-steps", "20",
        ]
        return command, RUN / "logs" / f"reload_{method}.log", gpu

    run_method_waves(METHODS, reload_builder, allowed_returncodes={0, -6, 120}, max_parallel=1)
    if not all(eval_evidence(reload_root / method, 1) for method in METHODS):
        raise RuntimeError("diagnostic reload evidence is incomplete")
    (training / "MATCHED_SMOKE_RELOAD_PASSED").write_text("complete\n", encoding="utf-8")

    def formal_builder(method: str, gpu: int):
        output = training / method
        command = [str(PY), str(ROOT / "RLinf/examples/sft/train_vla_sft.py"), "--config-path", str(ROOT / "RLinf/examples/sft/config"), "--config-name", "pick_single_ycb_object_variation_matched_sft_openpi_pi05", "runner.max_steps=5000", "runner.save_interval=500"]
        return command, RUN / "logs" / f"matched_{method}_5000.log", gpu, training_env(gpu, method, budget_root / method, output, 5000)

    write_state(current_stage="matched_training", next_stage="final_evaluation")
    run_method_waves(METHODS, formal_builder)
    required = tuple(range(500, 5001, 500))
    checkpoint_audit = {method: {str(step): str(checkpoint_for(training / method, step)) for step in required} for method in METHODS}
    if any(value == "None" for row in checkpoint_audit.values() for value in row.values()):
        raise RuntimeError("diagnostic matched training checkpoint audit failed")
    (training / "formal_checkpoint_audit.json").write_text(json.dumps(checkpoint_audit, indent=2) + "\n", encoding="utf-8")
    (training / "FORMAL_TRAINING_COMPLETE").write_text("complete\n", encoding="utf-8")

    eval_root = RUN / "final_evaluation"
    write_state(current_stage="final_evaluation", next_stage="result_registration")
    for split, seed in (("id", 17000), ("ood", 18000)):
        def eval_builder(method: str, gpu: int):
            output = eval_root / method / split
            command = [
                str(PY), str(ROOT / "tools/evaluate_pick_single_ycb_object_variation_pi05.py"),
                "--checkpoint", str(checkpoint_for(training / method, 5000)), "--pi05-base", str(MODEL_BASE),
                "--norm-stats", str(NORM), "--output-dir", str(output), "--split", split,
                "--episodes", "100", "--seed", str(seed), "--execute-horizon", "5", "--max-episode-steps", "200",
            ]
            return command, RUN / "logs" / f"final_{method}_{split}.log", gpu

        run_method_waves(METHODS, eval_builder, allowed_returncodes={0, -6, 120})
        if not all(eval_evidence(eval_root / method / split, 100) for method in METHODS):
            raise RuntimeError(f"diagnostic final {split} evidence is incomplete")

    comparison = {
        "format": "pick_single_ycb_object_variation_low_threshold_diagnostic_comparison_v1",
        "diagnostic": True,
        "canonical_pipeline": str(BASE),
        "checkpoint": str(CHECKPOINT),
        "diff_threshold_override": diff_threshold,
        "canonical_diff_threshold": 0.6456781893968583,
        "matched_budget": budget_manifest,
        "methods": {},
    }
    for method in METHODS:
        comparison["methods"][method] = {}
        for split in ("id", "ood"):
            comparison["methods"][method][split] = json.loads((eval_root / method / split / "summary.json").read_text(encoding="utf-8"))
    (RUN / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    lines = ["# PickSingleYCB object-variation low-threshold diagnostic", "", f"Diff threshold override: {diff_threshold}", f"Matched expert-action budget: {budget_manifest['common_expert_action_budget']}", "", "| Method | ID success | OOD success |", "|---|---:|---:|"]
    for method in METHODS:
        lines.append(f"| {method} | {comparison['methods'][method]['id']['successes']}/100 | {comparison['methods'][method]['ood']['successes']}/100 |")
    (RUN / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_state(current_stage="complete", next_stage=None, terminal_marker="PIPELINE_COMPLETE")
    (RUN / "PIPELINE_COMPLETE").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
