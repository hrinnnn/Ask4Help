#!/usr/bin/env python3
"""Durable controller for the complete Panda object-variation experiment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


RESULT_ROOT = Path(
    "/data/zhaozhixuan/Ask4Help-airplane-5090/results/"
    "xvla_panda_put_vegetable_basket_object_ood_v1"
)
WORK_ROOT = Path("/data/zhaozhixuan/xvla_panda_vegetable_work")
PYTHON = Path("/data/zhaozhixuan/envs/xvla_official_5090/bin/python")
XVLA_ROOT = Path("/data/zhaozhixuan/X-VLA")
BASE_MODEL = Path(
    "/data/zhaozhixuan/Ask4Help-airplane-5090/results/"
    "xvla_airplane_v1/model_cache/X-VLA-Pt-local"
)
RLINF_ROOT = WORK_ROOT / "RLinf"
TASK_MODULE = WORK_ROOT / "tools/panda_vegetable_basket_variants.py"
DOMAIN_ID = 20
STATE_PATH = RESULT_ROOT / "pipeline_state.json"
LOG_ROOT = RESULT_ROOT / "logs/full_pipeline"
ORACLE_ROOT = RESULT_ROOT / "preflight/oracle_gate_v11"
ORACLE_PROFILE = {
    "lift_height": 0.35,
    "release_max_steps": 60,
    "max_episode_steps": 150,
    "closing_axis_mode": "object_local_y",
    "retry_attempts": 3,
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def write_state(stage: str, status: str, **extra) -> None:
    payload = {
        "pipeline_id": "xvla_panda_put_vegetable_basket_object_ood_v1",
        "owner_thread": "019ffbc4-f3a9-78f3-8684-e0b4cba3552a",
        "stage": stage,
        "status": status,
        "updated_at": now(),
        **extra,
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[panda-pipeline] {stage} {status} {extra}", flush=True)


def fresh_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 100):
        candidate = path.with_name(f"{path.name}_retry{index}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"no fresh path for {path}")


def process_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def base_env(visible_devices: str = "") -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "CUDA_VISIBLE_DEVICES": visible_devices,
            "OMP_NUM_THREADS": "20",
            "MKL_NUM_THREADS": "20",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    return env


def run_logged(stage: str, command: list[str], *, visible_devices: str, log_path: Path) -> None:
    """Run one stage under the durable controller and retain its command log."""

    if log_path.exists():
        log_path = fresh_path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_state(stage, "running", command=command, log=str(log_path), gpu=visible_devices)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=str(WORK_ROOT),
            env=base_env(visible_devices),
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        write_state(stage, "running", pid=process.pid, command=command, log=str(log_path), gpu=visible_devices)
        # Keep the controller passive while the child owns the long stage.
        # The short sleep avoids a tight polling loop before reaping it.
        time.sleep(60)
        return_code = process.wait()
    if return_code != 0:
        write_state(stage, "engineering_failure", pid=process.pid, return_code=return_code, log=str(log_path))
        raise RuntimeError(f"{stage} exited with {return_code}")
    write_state(stage, "process_complete", pid=process.pid, log=str(log_path))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def valid_oracle_split(path: Path) -> tuple[bool, dict | None]:
    summary_path = path / "summary.json"
    if not summary_path.is_file():
        return False, None
    try:
        summary = read_json(summary_path)
    except (OSError, json.JSONDecodeError):
        return False, None
    rows = summary.get("rows")
    if summary.get("episodes") != 20 or not isinstance(rows, list) or len(rows) != 20:
        return False, summary
    if len(list((path / "videos").glob("*.mp4"))) != 20:
        return False, summary
    if len(list((path / "metadata").glob("*.json"))) != 20:
        return False, summary
    if len(list((path / "data").glob("*.h5"))) != 20:
        return False, summary
    return True, summary


def audit_oracle(path: Path) -> dict:
    import h5py

    valid, summary = valid_oracle_split(path)
    if not valid or summary is None:
        raise RuntimeError(f"incomplete Oracle split: {path}")
    for row in summary["rows"]:
        with h5py.File(row["data_path"], "r") as h5:
            if h5["proprio"].ndim != 2 or h5["proprio"].shape[1] != 10:
                raise RuntimeError(f"invalid proprio shape in {row['data_path']}")
            if h5["abs_action_6d"].ndim != 2 or h5["abs_action_6d"].shape[1] != 10:
                raise RuntimeError(f"invalid action shape in {row['data_path']}")
            if h5["images"].shape[0] != h5["proprio"].shape[0] + 1:
                raise RuntimeError(f"image/action boundary mismatch in {row['data_path']}")
    if int(summary["successes"]) < 19:
        raise RuntimeError(f"Oracle scientific gate failed: {path} {summary['successes']}/20")
    return summary


def ensure_oracle_gate() -> tuple[Path, Path]:
    while True:
        id_ok, id_summary = valid_oracle_split(ORACLE_ROOT / "id")
        ood_ok, ood_summary = valid_oracle_split(ORACLE_ROOT / "ood")
        if id_ok and ood_ok:
            id_summary = audit_oracle(ORACLE_ROOT / "id")
            ood_summary = audit_oracle(ORACLE_ROOT / "ood")
            (ORACLE_ROOT / "ORACLE_GATE_PASSED").write_text("passed\n", encoding="utf-8")
            write_state(
                "oracle_gate_v11",
                "complete",
                marker=str(ORACLE_ROOT / "ORACLE_GATE_PASSED"),
                id_successes=id_summary["successes"],
                ood_successes=ood_summary["successes"],
            )
            return ORACLE_ROOT / "id", ORACLE_ROOT / "ood"
        pids = []
        for split in ("id", "ood"):
            pid_path = ORACLE_ROOT / f"{split}.pid"
            if pid_path.is_file():
                try:
                    pids.append(int(pid_path.read_text().strip()))
                except ValueError:
                    pass
        if any(process_alive(pid) for pid in pids):
            write_state(
                "oracle_gate_v11",
                "running",
                id_episodes=(id_summary or {}).get("episodes", 0),
                ood_episodes=(ood_summary or {}).get("episodes", 0),
            )
            time.sleep(300)
            continue
        raise RuntimeError("oracle_gate_v11 ended without complete evidence")


def planner_command(split: str, episodes: int, seed_start: int, output: Path) -> list[str]:
    return [
        str(PYTHON),
        str(WORK_ROOT / "tools/collect_xvla_panda_vegetable_basket_planner_oracle.py"),
        "--rlinf-root", str(RLINF_ROOT),
        "--task-module", str(TASK_MODULE),
        "--split", split,
        "--episodes", str(episodes),
        "--seed-start", str(seed_start),
        "--lift-height", str(ORACLE_PROFILE["lift_height"]),
        "--release-max-steps", str(ORACLE_PROFILE["release_max_steps"]),
        "--max-episode-steps", str(ORACLE_PROFILE["max_episode_steps"]),
        "--closing-axis-mode", str(ORACLE_PROFILE["closing_axis_mode"]),
        "--retry-attempts", str(ORACLE_PROFILE["retry_attempts"]),
        "--output", str(output),
    ]


def materialize_id_dataset() -> tuple[Path, Path, Path]:
    raw = RESULT_ROOT / "id_demo_raw_v2"
    if not (raw / "summary.json").is_file():
        raw = fresh_path(raw)
        run_logged("id_demo_raw_v2", planner_command("id", 160, 96300, raw), visible_devices="6", log_path=LOG_ROOT / "id_demo_raw_v2.log")
    summary = read_json(raw / "summary.json")
    if int(summary.get("successes", 0)) < 128:
        raise RuntimeError(f"ID Oracle demonstrations insufficient: {summary.get('successes')}/160")
    dataset = RESULT_ROOT / "dataset/id_demos_128_v2"
    if not (dataset / "DATASET_MATERIALIZED").is_file():
        dataset = fresh_path(dataset)
        run_logged(
            "id_dataset_materialize_v2",
            [
                str(PYTHON), str(WORK_ROOT / "tools/build_xvla_panda_vegetable_basket_id_dataset.py"),
                "--raw-root", str(raw), "--output", str(dataset), "--target-episodes", "128",
            ],
            visible_devices="",
            log_path=LOG_ROOT / "id_dataset_materialize_v2.log",
        )
    audit = RESULT_ROOT / "dataset/id_audit_v2"
    if not (audit / "DATASET_AUDIT_PASSED").is_file():
        audit = fresh_path(audit)
        run_logged(
            "id_dataset_audit_v2",
            [
                str(PYTHON), str(WORK_ROOT / "tools/audit_xvla_panda_vegetable_basket_dataset.py"),
                "--dataset", str(dataset), "--output", str(audit), "--expected-episodes", "128",
            ],
            visible_devices="",
            log_path=LOG_ROOT / "id_dataset_audit_v2.log",
        )
    if not (audit / "DATASET_AUDIT_PASSED").is_file():
        raise RuntimeError("ID dataset audit marker missing")
    write_state("id_dataset_audit_v2", "complete", raw=str(raw), dataset=str(dataset), audit=str(audit))
    return raw, dataset, audit


def train_command(base_model: Path, dataset: Path, output: Path, steps: int, *, port: int, smoke: bool = False) -> list[str]:
    command = [
        "taskset", "-c", "120-159", str(PYTHON), "-m", "accelerate.commands.launch",
        "--num_processes", "2", "--multi_gpu", "--main_process_port", str(port),
        "--mixed_precision", "bf16", "--gpu_ids", "0,1",
        str(WORK_ROOT / "tools/run_xvla_panda_vegetable_basket_id_training.py"),
        "--xvla-root", str(XVLA_ROOT), "--base-model", str(base_model),
        "--dataset", str(dataset), "--output", str(output), "--steps", str(steps),
        "--save-interval", "500", "--batch-size", "8", "--gradient-accumulation-steps", "8",
        "--learning-rate", "1e-4", "--learning-coef", "0.1", "--freeze-steps", "1000",
        "--warmup-steps", "2000", "--domain-id", str(DOMAIN_ID), "--distributed-backend", "gloo", "--seed", "96300",
    ]
    if smoke:
        command.append("--smoke-only")
    return command


def train_id_policy(dataset: Path) -> Path:
    smoke = RESULT_ROOT / "training/id_smoke_v2"
    if not (smoke / "RELOAD_SMOKE_COMPLETE").is_file():
        smoke = fresh_path(smoke)
        run_logged("id_train_smoke_v2", train_command(BASE_MODEL, dataset, smoke, 2, port=29561, smoke=True), visible_devices="5,6", log_path=LOG_ROOT / "id_train_smoke_v2.log")
    training = RESULT_ROOT / "training/id_sft_10000_v2"
    if not (training / "TRAINING_COMPLETE").is_file():
        training = fresh_path(training)
        run_logged("id_sft_10000_v2", train_command(BASE_MODEL, dataset, training, 10000, port=29562), visible_devices="5,6", log_path=LOG_ROOT / "id_sft_10000_v2.log")
    missing = [str(training / f"ckpt-{step}" / "model.safetensors") for step in range(500, 10001, 500) if not (training / f"ckpt-{step}" / "model.safetensors").is_file()]
    if missing:
        raise RuntimeError(f"ID training marker/checkpoint mismatch: {missing[:3]}")
    write_state("id_sft_10000_v2", "complete", training=str(training))
    return training


def evaluator_command(checkpoint: Path, split: str, episodes: int, seed_start: int, output: Path) -> list[str]:
    return [
        str(PYTHON), str(WORK_ROOT / "tools/evaluate_xvla_panda_vegetable_basket.py"),
        "--xvla-root", str(XVLA_ROOT), "--checkpoint", str(checkpoint),
        "--rlinf-root", str(RLINF_ROOT), "--task-module", str(TASK_MODULE),
        "--split", split, "--episodes", str(episodes), "--seed-start", str(seed_start),
        "--output", str(output), "--domain-id", str(DOMAIN_ID), "--flow-steps", "10",
        "--execute-horizon", "5", "--max-episode-steps", "150",
    ]


def complete_eval(output: Path, episodes: int) -> bool:
    summary_path = output / "summary.json"
    if not summary_path.is_file():
        return False
    summary = read_json(summary_path)
    return (
        summary.get("episodes") == episodes
        and summary.get("videos") == episodes
        and summary.get("actions") == episodes
        and (output / "EVAL_COMPLETE").is_file()
    )


def select_and_gate(training: Path) -> tuple[Path, Path]:
    root = RESULT_ROOT / "id_checkpoint_selection_v2"
    root.mkdir(parents=True, exist_ok=True)
    selection_path = root / "selection.json"
    if selection_path.is_file() and (root / "CHECKPOINT_SELECTION_COMPLETE").is_file():
        payload = read_json(selection_path)
        return Path(payload["checkpoint"]), Path(payload["formal_gate"])
    records = []
    selected: Path | None = None
    for step in range(500, 10001, 500):
        checkpoint = training / f"ckpt-{step}"
        if not checkpoint.is_dir():
            raise RuntimeError(f"missing checkpoint {checkpoint}")
        output = fresh_path(root / f"probe_{step}")
        run_logged(
            f"id_probe_{step}",
            evaluator_command(checkpoint, "id", 20, 97000 + step, output),
            visible_devices="6",
            log_path=LOG_ROOT / f"id_probe_{step}.log",
        )
        if not complete_eval(output, 20):
            raise RuntimeError(f"incomplete ID probe {output}")
        summary = read_json(output / "summary.json")
        record = {"step": step, "checkpoint": str(checkpoint), "successes": summary["strict_successes"], "summary": str(output / "summary.json")}
        records.append(record)
        if selected is None and int(record["successes"]) >= 17:
            selected = checkpoint
            break
    if selected is None:
        marker = RESULT_ROOT / "ID_BASE_NOT_ACCEPTED"
        marker.write_text(json.dumps({"records": records, "required": "17/20 probe and 80/100 formal ID gate"}, indent=2) + "\n", encoding="utf-8")
        write_state("id_checkpoint_selection_v2", "scientific_stop", marker=str(marker), records=records)
        raise SystemExit("ID_BASE_NOT_ACCEPTED")
    formal = fresh_path(root / f"formal_id_{selected.name}")
    run_logged(
        "id_formal_gate_100_v2",
        evaluator_command(selected, "id", 100, 98000, formal),
        visible_devices="6",
        log_path=LOG_ROOT / "id_formal_gate_100_v2.log",
    )
    if not complete_eval(formal, 100):
        raise RuntimeError("formal ID gate evidence incomplete")
    formal_summary = read_json(formal / "summary.json")
    if int(formal_summary["strict_successes"]) < 80:
        marker = RESULT_ROOT / "ID_BASE_NOT_ACCEPTED"
        marker.write_text(json.dumps({"formal_summary": str(formal / "summary.json"), "strict_successes": formal_summary["strict_successes"], "required": 80}, indent=2) + "\n", encoding="utf-8")
        write_state("id_formal_gate_100_v2", "scientific_stop", marker=str(marker))
        raise SystemExit("ID_BASE_NOT_ACCEPTED")
    (formal / "ID_BASE_VALIDATED").write_text("validated\n", encoding="utf-8")
    payload = {"checkpoint": str(selected), "formal_gate": str(formal), "records": records, "criterion": "earliest checkpoint with 17/20 probe and 80/100 formal ID gate"}
    selection_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (root / "CHECKPOINT_SELECTION_COMPLETE").write_text("complete\n", encoding="utf-8")
    write_state("id_formal_gate_100_v2", "complete", checkpoint=str(selected), formal_gate=str(formal))
    return selected, formal


def build_passive_assets(checkpoint: Path, dataset: Path) -> tuple[Path, Path]:
    internal = RESULT_ROOT / "passive/detector_assets_internal_v1"
    if not (internal / "DETECTOR_ASSETS_COMPLETE").is_file():
        internal = fresh_path(internal)
        run_logged(
            "passive_internal_assets",
            [
                str(PYTHON), str(WORK_ROOT / "tools/build_xvla_panda_vegetable_basket_detector_assets.py"),
                "--checkpoint", str(checkpoint), "--xvla-root", str(XVLA_ROOT), "--dataset", str(dataset),
                "--output-dir", str(internal), "--domain-id", str(DOMAIN_ID), "--batch-size", "8",
                "--probe-seed", "0", "--probe-steps", "5", "--pca-dim", "512", "--ridge", "1e-6",
            ],
            visible_devices="6",
            log_path=LOG_ROOT / "passive_internal_assets.log",
        )
    external = RESULT_ROOT / "passive/detector_assets_external_v1"
    if not (external / "EXTERNAL_ASSETS_COMPLETE").is_file():
        external = fresh_path(external)
        run_logged(
            "passive_external_assets",
            [
                str(PYTHON), str(WORK_ROOT / "tools/build_xvla_panda_vegetable_basket_external_assets.py"),
                "--dataset", str(dataset), "--output-dir", str(external), "--batch-size", "64", "--crsail-k", "5",
            ],
            visible_devices="5",
            log_path=LOG_ROOT / "passive_external_assets.log",
        )
    return internal, external


def passive_eval_command(checkpoint: Path, internal: Path, external: Path, split: str, episodes: int, seed: int, output: Path) -> list[str]:
    return [
        str(PYTHON), str(WORK_ROOT / "tools/evaluate_xvla_panda_vegetable_basket_failure_detectors.py"),
        "--checkpoint", str(checkpoint), "--xvla-root", str(XVLA_ROOT), "--rlinf-root", str(RLINF_ROOT),
        "--task-module", str(TASK_MODULE), "--multilayer-assets", str(internal), "--external-assets", str(external),
        "--output-dir", str(output), "--split", split, "--episodes", str(episodes), "--seed-start", str(seed),
        "--domain-id", str(DOMAIN_ID), "--flow-steps", "10", "--execute-horizon", "5", "--max-episode-steps", "150",
        "--probe-steps", "5", "--diff-timesteps", "16", "--diff-noise-samples", "1",
    ]


def run_passive(checkpoint: Path, dataset: Path) -> tuple[Path, Path, Path]:
    internal, external = build_passive_assets(checkpoint, dataset)
    calibration = RESULT_ROOT / "passive/calibration_id_25_v1"
    if not (calibration / "EVALUATION_COMPLETE").is_file():
        calibration = fresh_path(calibration)
        run_logged("passive_calibration_id_25", passive_eval_command(checkpoint, internal, external, "id", 25, 100000, calibration), visible_devices="6", log_path=LOG_ROOT / "passive_calibration_id_25.log")
    thresholds = RESULT_ROOT / "passive/calibration_thresholds_v1"
    if not (thresholds / "CALIBRATION_COMPLETE").is_file():
        thresholds = fresh_path(thresholds)
        run_logged(
            "passive_threshold_calibration",
            [str(PYTHON), str(WORK_ROOT / "tools/calibrate_xvla_panda_vegetable_basket_detectors.py"), "--rollouts", str(calibration), "--output-dir", str(thresholds), "--quantile", "0.95", "--minimum-successes", "20"],
            visible_devices="",
            log_path=LOG_ROOT / "passive_threshold_calibration.log",
        )
    test_id = RESULT_ROOT / "passive/eval_id_100_v1"
    if not complete_eval(test_id, 100):
        test_id = fresh_path(test_id)
        run_logged("passive_eval_id_100", passive_eval_command(checkpoint, internal, external, "id", 100, 50000, test_id), visible_devices="6", log_path=LOG_ROOT / "passive_eval_id_100.log")
    test_ood = RESULT_ROOT / "passive/eval_ood_100_v1"
    if not complete_eval(test_ood, 100):
        test_ood = fresh_path(test_ood)
        run_logged("passive_eval_ood_100", passive_eval_command(checkpoint, internal, external, "ood", 100, 60000, test_ood), visible_devices="6", log_path=LOG_ROOT / "passive_eval_ood_100.log")
    metrics = RESULT_ROOT / "passive/metrics_v1"
    if not (metrics / "METRICS_COMPLETE").is_file():
        metrics = fresh_path(metrics)
        run_logged(
            "passive_metrics",
            [str(PYTHON), str(WORK_ROOT / "tools/compute_xvla_panda_vegetable_basket_failure_metrics.py"), "--id-rollouts", str(test_id), "--ood-rollouts", str(test_ood), "--thresholds", str(thresholds), "--output-dir", str(metrics)],
            visible_devices="",
            log_path=LOG_ROOT / "passive_metrics.log",
        )
    if not (metrics / "METRICS_COMPLETE").is_file():
        raise RuntimeError("passive metrics marker missing")
    write_state("passive_metrics", "complete", internal=str(internal), external=str(external), thresholds=str(thresholds), metrics=str(metrics))
    return internal, external, thresholds


def collection_command(method: str, checkpoint: Path | None, internal: Path | None, threshold: float | None, output: Path) -> list[str]:
    command = [
        str(PYTHON), str(WORK_ROOT / "tools/collect_xvla_panda_vegetable_basket_dagger.py"),
        "--method", method, "--rlinf-root", str(RLINF_ROOT), "--task-module", str(TASK_MODULE),
        "--output-dir", str(output), "--target-successes", "100", "--offline-id-target", "50", "--offline-ood-target", "50",
        "--id-seed-start", "101000", "--ood-seed-start", "102000", "--max-attempts", "400", "--max-policy-steps", "50",
        "--max-episode-steps", "150", "--flow-steps", "10", "--probe-steps", "5", "--diff-timesteps", "16", "--diff-noise-samples", "1",
        "--lift-height", "0.35", "--release-max-steps", "60",
    ]
    if checkpoint is not None:
        command += ["--checkpoint", str(checkpoint), "--xvla-root", str(XVLA_ROOT)]
    if internal is not None:
        command += ["--multilayer-assets", str(internal)]
    if threshold is not None:
        command += ["--gate-threshold", str(threshold)]
    if method == "internal_pca":
        command += ["--gate-score-name", "vlm_action_bridge_pca", "--gate-patience", "1"]
    elif method == "diffdagger":
        command += ["--gate-patience", "2"]
    return command


def run_collections(checkpoint: Path, internal: Path, thresholds: Path) -> dict[str, Path]:
    threshold_payload = read_json(thresholds / "thresholds.json")
    methods = {
        "internal_pca": float(threshold_payload["methods"]["vlm_action_bridge_pca"]["threshold"]),
        "diffdagger": float(threshold_payload["methods"]["diffdagger"]["threshold"]),
        "failure_recovery": None,
        "offline_bc": None,
    }
    outputs: dict[str, Path] = {}
    for index, (method, threshold) in enumerate(methods.items()):
        output = RESULT_ROOT / f"collections/{method}_v1"
        complete = (output / "COLLECTION_COMPLETE").is_file()
        if not complete:
            output = fresh_path(output)
            run_logged(
                f"collection_{method}",
                collection_command(method, None if method == "offline_bc" else checkpoint, internal if method == "internal_pca" else None, threshold, output),
                visible_devices=str(5 + (index % 3)) if method != "offline_bc" else "",
                log_path=LOG_ROOT / f"collection_{method}.log",
            )
        summary = read_json(output / "summary.json")
        accepted = int(summary.get("accepted", 0))
        if method == "offline_bc":
            valid = summary.get("accepted_by_split") == {"id": 50, "ood": 50}
        else:
            valid = accepted >= 100
        if not valid or not (output / "accepted_episodes.jsonl").is_file():
            raise RuntimeError(f"collection target failed for {method}: {summary.get('accepted_by_split')} {accepted}")
        outputs[method] = output
    write_state("collections", "complete", outputs={key: str(value) for key, value in outputs.items()})
    return outputs


def mixed_train_command(checkpoint: Path, dataset: Path, expert: Path, output: Path, steps: int, port: int, smoke: bool) -> list[str]:
    return [
        "taskset", "-c", "120-159", str(PYTHON), "-m", "accelerate.commands.launch", "--num_processes", "2", "--multi_gpu",
        "--main_process_port", str(port), "--mixed_precision", "bf16", "--gpu_ids", "0,1",
        str(WORK_ROOT / "tools/run_xvla_panda_vegetable_basket_mixed_training.py"),
        "--xvla-root", str(XVLA_ROOT), "--base-model", str(checkpoint), "--id-dataset", str(dataset),
        "--expert-dataset", str(expert), "--output", str(output), "--steps", str(steps), "--save-interval", "500",
        "--batch-size", "8", "--gradient-accumulation-steps", "8", "--learning-rate", "1e-4", "--learning-coef", "0.1",
        "--freeze-steps", "1000", "--warmup-steps", "2000", "--domain-id", str(DOMAIN_ID), "--distributed-backend", "gloo", "--seed", "96500",
    ] + (["--smoke-only"] if smoke else [])


def train_branches(checkpoint: Path, dataset: Path, collections: dict[str, Path]) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    for index, method in enumerate(("internal_pca", "diffdagger", "failure_recovery", "offline_bc")):
        expert = collections[method]
        smoke = RESULT_ROOT / f"training/branch_{method}_smoke_v1"
        if not (smoke / "RELOAD_SMOKE_COMPLETE").is_file():
            smoke = fresh_path(smoke)
            run_logged(f"train_{method}_smoke", mixed_train_command(checkpoint, dataset, expert, smoke, 2, 29600 + index, True), visible_devices="5,6", log_path=LOG_ROOT / f"train_{method}_smoke.log")
        output = RESULT_ROOT / f"training/branch_{method}_5000_v1"
        if not (output / "TRAINING_COMPLETE").is_file():
            output = fresh_path(output)
            run_logged(f"train_{method}_5000", mixed_train_command(checkpoint, dataset, expert, output, 5000, 29610 + index, False), visible_devices="5,6", log_path=LOG_ROOT / f"train_{method}_5000.log")
        missing = [str(output / f"ckpt-{step}" / "model.safetensors") for step in range(500, 5001, 500) if not (output / f"ckpt-{step}" / "model.safetensors").is_file()]
        if missing:
            raise RuntimeError(f"missing branch checkpoints for {method}: {missing[:2]}")
        outputs[method] = output
    write_state("branch_training", "complete", outputs={key: str(value) for key, value in outputs.items()})
    return outputs


def final_evaluation(branches: dict[str, Path]) -> None:
    rows = []
    for index, (method, training) in enumerate(branches.items()):
        checkpoint = training / "ckpt-5000"
        method_root = RESULT_ROOT / f"final_evaluation/{method}_v1"
        id_output = method_root / "id"
        if not complete_eval(id_output, 100):
            id_output = fresh_path(id_output)
            run_logged(f"final_{method}_id", evaluator_command(checkpoint, "id", 100, 110000, id_output), visible_devices=str(5 + (index % 3)), log_path=LOG_ROOT / f"final_{method}_id.log")
        ood_output = method_root / "ood"
        if not complete_eval(ood_output, 100):
            ood_output = fresh_path(ood_output)
            run_logged(f"final_{method}_ood", evaluator_command(checkpoint, "ood", 100, 120000, ood_output), visible_devices=str(5 + (index % 3)), log_path=LOG_ROOT / f"final_{method}_ood.log")
        id_summary = read_json(id_output / "summary.json")
        ood_summary = read_json(ood_output / "summary.json")
        rows.append({
            "method": method,
            "checkpoint": str(checkpoint),
            "id": {"episodes": id_summary["episodes"], "strict_successes": id_summary["strict_successes"], "ever_grasped": id_summary["ever_grasped_successes"], "summary": str(id_output / "summary.json")},
            "ood": {"episodes": ood_summary["episodes"], "strict_successes": ood_summary["strict_successes"], "ever_grasped": ood_summary["ever_grasped_successes"], "summary": str(ood_output / "summary.json")},
            "videos": {"id": id_summary["videos"], "ood": ood_summary["videos"]},
        })
    output = RESULT_ROOT / "final_evaluation/comparison_v1"
    output.mkdir(parents=True, exist_ok=True)
    payload = {"format": "xvla_panda_vegetable_basket_final_comparison_v1", "rows": rows, "success_definition": "released object inside basket region, above target plane, static and not grasped"}
    (output / "comparison.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = ["# Panda Eggplant Object-OOD Final Evaluation", "", "| Method | ID strict | OOD strict | ID ever grasped | OOD ever grasped |", "|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['method']} | {row['id']['strict_successes']}/100 | {row['ood']['strict_successes']}/100 | {row['id']['ever_grasped']}/100 | {row['ood']['ever_grasped']}/100 |")
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "FINAL_EVALUATION_COMPLETE").write_text("complete\n", encoding="utf-8")
    write_state("final_evaluation", "complete", comparison=str(output / "comparison.json"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-stage", default="oracle_gate_v11")
    return parser.parse_args()


def main() -> None:
    parse_args()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    write_state("controller_start", "running", controller_pid=os.getpid(), next_stage="oracle_gate_v11")
    ensure_oracle_gate()
    _raw, dataset, _audit = materialize_id_dataset()
    training = train_id_policy(dataset)
    checkpoint, _formal = select_and_gate(training)
    internal, _external, thresholds = run_passive(checkpoint, dataset)
    collections = run_collections(checkpoint, internal, thresholds)
    branches = train_branches(checkpoint, dataset, collections)
    final_evaluation(branches)
    (RESULT_ROOT / "PIPELINE_COMPLETE").write_text("complete\n", encoding="utf-8")
    write_state("PIPELINE_COMPLETE", "complete", marker=str(RESULT_ROOT / "PIPELINE_COMPLETE"))
    print("[panda-pipeline] PIPELINE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
