#!/usr/bin/env python3
"""Restart-tolerant controller for the Panda X-VLA vegetable-basket pipeline."""

from __future__ import annotations

import json
import os
import subprocess
import sys
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
TASK_MODULE = WORK_ROOT / "tools/panda_vegetable_basket_variants.py"
RLINF_ROOT = WORK_ROOT / "RLinf"
DOMAIN_ID = 20
LOG_ROOT = RESULT_ROOT / "logs"
STATE_PATH = RESULT_ROOT / "pipeline_state.json"


def now() -> float:
    return time.time()


def write_state(stage: str, status: str, **extra) -> None:
    payload = {
        "pipeline_id": "xvla_panda_put_vegetable_basket_object_ood_v1",
        "stage": stage,
        "status": status,
        "updated_at": now(),
        **extra,
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def fresh_path(base: Path) -> Path:
    if not base.exists():
        return base
    for index in range(1, 100):
        candidate = base.with_name(f"{base.name}_retry{index}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"no fresh retry path available for {base}")


def process_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def valid_oracle_split(path: Path) -> tuple[bool, dict | None]:
    summary_path = path / "summary.json"
    if not summary_path.exists():
        return False, None
    summary = load_summary(summary_path)
    episodes = int(summary.get("episodes", -1))
    videos = len(list((path / "videos").glob("*.mp4")))
    metadata = len(list((path / "metadata").glob("*.json")))
    return episodes == 20 and videos == 20 and metadata == 20, summary


def wait_for_oracle_v6() -> tuple[Path, Path]:
    root = RESULT_ROOT / "oracle_gate_v6"
    id_path, ood_path = root / "id", root / "ood"
    write_state(
        "oracle_gate_v6",
        "waiting",
        id_output=str(id_path),
        ood_output=str(ood_path),
        id_pid_file=str(root / "id.pid"),
        ood_pid_file=str(root / "ood.pid"),
    )
    while True:
        id_complete, id_summary = valid_oracle_split(id_path)
        ood_complete, ood_summary = valid_oracle_split(ood_path)
        id_pid = int((root / "id.pid").read_text()) if (root / "id.pid").exists() else None
        ood_pid = int((root / "ood.pid").read_text()) if (root / "ood.pid").exists() else None
        if id_complete and ood_complete:
            write_state("oracle_gate_v6", "auditing", id_summary=id_summary, ood_summary=ood_summary)
            if int(id_summary["successes"]) < 19 or int(ood_summary["successes"]) < 19:
                (RESULT_ROOT / "ORACLE_NOT_ACCEPTED").write_text(
                    json.dumps(
                        {
                            "id_successes": id_summary["successes"],
                            "ood_successes": ood_summary["successes"],
                            "required": "at least 19/20 strict successes per split",
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                write_state("oracle_gate_v6", "scientific_stop", marker="ORACLE_NOT_ACCEPTED")
                raise SystemExit("ORACLE_NOT_ACCEPTED")
            (root / "ORACLE_GATE_PASSED").write_text("passed\n", encoding="utf-8")
            write_state("oracle_gate_v6", "complete", marker=str(root / "ORACLE_GATE_PASSED"))
            return id_path, ood_path
        write_state(
            "oracle_gate_v6",
            "running" if process_alive(id_pid) or process_alive(ood_pid) else "recovery_needed",
            id_pid=id_pid,
            ood_pid=ood_pid,
            id_episodes=(id_summary or {}).get("episodes", 0),
            ood_episodes=(ood_summary or {}).get("episodes", 0),
        )
        if not process_alive(id_pid) and not id_complete:
            raise RuntimeError("oracle_gate_v6 ID process exited without complete evidence")
        if not process_alive(ood_pid) and not ood_complete:
            raise RuntimeError("oracle_gate_v6 OOD process exited without complete evidence")
        time.sleep(300)


def run_logged(stage: str, command: list[str], env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_state(stage, "running", command=command, log=str(log_path))
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=str(WORK_ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        write_state(stage, "running", pid=process.pid, command=command, log=str(log_path))
        return_code = process.wait()
    if return_code != 0:
        write_state(stage, "engineering_failure", pid=process.pid, return_code=return_code, log=str(log_path))
        raise RuntimeError(f"{stage} failed with exit code {return_code}")
    write_state(stage, "process_complete", pid=process.pid, log=str(log_path))


def base_env(*, visible_devices: str = "") -> dict[str, str]:
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
        }
    )
    return env


def collect_id_dataset() -> Path:
    raw_root = fresh_path(RESULT_ROOT / "id_demo_raw_v1")
    command = [
        str(PYTHON), str(WORK_ROOT / "tools/collect_xvla_panda_vegetable_basket_planner_oracle.py"),
        "--rlinf-root", str(RLINF_ROOT), "--task-module", str(TASK_MODULE),
        "--split", "id", "--episodes", "160", "--seed-start", "96300",
        "--lift-height", "0.35", "--release-max-steps", "30", "--max-episode-steps", "120",
        "--closing-axis-mode", "object_local_y", "--output", str(raw_root),
    ]
    run_logged("id_demo_raw_v1", command, base_env(visible_devices="6"), LOG_ROOT / "id_demo_raw_v1.log")
    summary = load_summary(raw_root / "summary.json")
    if int(summary.get("successes", 0)) < 128:
        write_state("id_demo_raw_v1", "scientific_stop", successes=summary.get("successes"), required=128)
        raise SystemExit("ID_DEMOS_INSUFFICIENT")
    dataset_root = fresh_path(RESULT_ROOT / "dataset/id_demos_128_v1")
    command = [
        str(PYTHON), str(WORK_ROOT / "tools/build_xvla_panda_vegetable_basket_id_dataset.py"),
        "--raw-root", str(raw_root), "--output", str(dataset_root), "--target-episodes", "128",
    ]
    run_logged("id_dataset_materialize_v1", command, base_env(), LOG_ROOT / "id_dataset_materialize_v1.log")
    return dataset_root


def audit_id_dataset(dataset_root: Path) -> Path:
    audit_root = fresh_path(RESULT_ROOT / "dataset/id_audit_v1")
    command = [
        str(PYTHON), str(WORK_ROOT / "tools/audit_xvla_panda_vegetable_basket_dataset.py"),
        "--dataset", str(dataset_root), "--output", str(audit_root), "--expected-episodes", "128",
    ]
    run_logged("id_dataset_audit_v1", command, base_env(), LOG_ROOT / "id_dataset_audit_v1.log")
    if not (audit_root / "DATASET_AUDIT_PASSED").exists():
        raise RuntimeError("ID dataset audit did not produce DATASET_AUDIT_PASSED")
    return audit_root


def train_command(output: Path, dataset_root: Path, steps: int, smoke: bool = False) -> list[str]:
    return [
        "taskset", "-c", "120-159", str(PYTHON), "-m", "accelerate.commands.launch",
        "--num_processes", "2", "--multi_gpu", "--mixed_precision", "bf16", "--gpu_ids", "0,1",
        str(WORK_ROOT / "tools/run_xvla_panda_vegetable_basket_id_training.py"),
        "--xvla-root", str(XVLA_ROOT), "--base-model", str(BASE_MODEL),
        "--dataset", str(dataset_root), "--output", str(output), "--steps", str(steps),
        "--save-interval", "500", "--batch-size", "32", "--gradient-accumulation-steps", "2",
        "--learning-rate", "1e-4", "--learning-coef", "0.1", "--freeze-steps", "1000",
        "--warmup-steps", "2000", "--domain-id", str(DOMAIN_ID), "--seed", "96300",
    ] + (["--smoke-only"] if smoke else [])


def train_id_policy(dataset_root: Path) -> Path:
    smoke_root = fresh_path(RESULT_ROOT / "training/id_smoke_v1")
    run_logged(
        "id_train_smoke_v1",
        train_command(smoke_root, dataset_root, 2, smoke=True),
        base_env(visible_devices="5,6"),
        LOG_ROOT / "id_train_smoke_v1.log",
    )
    if not (smoke_root / "RELOAD_SMOKE_COMPLETE").exists():
        raise RuntimeError("ID train smoke missing RELOAD_SMOKE_COMPLETE")
    training_root = fresh_path(RESULT_ROOT / "training/id_sft_10000_v1")
    run_logged(
        "id_sft_10000_v1",
        train_command(training_root, dataset_root, 10000),
        base_env(visible_devices="5,6"),
        LOG_ROOT / "id_sft_10000_v1.log",
    )
    if not (training_root / "TRAINING_COMPLETE").exists():
        raise RuntimeError("ID training missing TRAINING_COMPLETE")
    return training_root


def evaluate_checkpoint(checkpoint: Path, output: Path, seed_start: int, episodes: int) -> None:
    command = [
        str(PYTHON), str(WORK_ROOT / "tools/evaluate_xvla_panda_vegetable_basket.py"),
        "--xvla-root", str(XVLA_ROOT), "--checkpoint", str(checkpoint),
        "--rlinf-root", str(RLINF_ROOT), "--task-module", str(TASK_MODULE),
        "--split", "id", "--episodes", str(episodes), "--seed-start", str(seed_start),
        "--output", str(output), "--domain-id", str(DOMAIN_ID), "--flow-steps", "10",
        "--execute-horizon", "5", "--max-episode-steps", "120",
    ]
    run_logged(
        f"id_checkpoint_probe_{checkpoint.name}",
        ["taskset", "-c", "160-179"] + command,
        base_env(visible_devices="7"),
        LOG_ROOT / f"id_checkpoint_probe_{checkpoint.name}.log",
    )


def select_and_gate(training_root: Path) -> tuple[Path, Path]:
    selection_root = RESULT_ROOT / "id_checkpoint_selection"
    selection_root.mkdir(parents=True, exist_ok=True)
    selected = None
    probes = []
    for step in range(500, 10001, 500):
        checkpoint = training_root / f"ckpt-{step}"
        if not checkpoint.exists():
            raise RuntimeError(f"missing checkpoint {checkpoint}")
        output = fresh_path(selection_root / f"step_{step}")
        evaluate_checkpoint(checkpoint, output, 97000 + step, 20)
        summary = load_summary(output / "summary.json")
        row = {"step": step, "successes": summary["strict_successes"], "summary": str(output / "summary.json")}
        probes.append(row)
        if selected is None and int(summary["strict_successes"]) >= 17:
            selected = checkpoint
            break
    if selected is None:
        selected = training_root / "ckpt-10000"
        (RESULT_ROOT / "ID_BASE_NOT_ACCEPTED").write_text(
            json.dumps({"probes": probes, "required": "at least 17/20 at selection and 80/100 formal gate"}, indent=2) + "\n",
            encoding="utf-8",
        )
        write_state("id_checkpoint_selection", "scientific_stop", selected=str(selected), probes=probes, marker="ID_BASE_NOT_ACCEPTED")
        raise SystemExit("ID_BASE_NOT_ACCEPTED")
    (selection_root / "selected_checkpoint.json").write_text(
        json.dumps({"checkpoint": str(selected), "probes": probes}, indent=2) + "\n", encoding="utf-8"
    )
    formal_root = fresh_path(RESULT_ROOT / "id_formal_gate_100")
    evaluate_checkpoint(selected, formal_root, 98000, 100)
    summary = load_summary(formal_root / "summary.json")
    if int(summary["strict_successes"]) < 80:
        (RESULT_ROOT / "ID_BASE_NOT_ACCEPTED").write_text(
            json.dumps({"formal_summary": str(formal_root / "summary.json"), "strict_successes": summary["strict_successes"], "required": 80}, indent=2) + "\n",
            encoding="utf-8",
        )
        write_state("id_formal_gate_100", "scientific_stop", summary=summary, marker="ID_BASE_NOT_ACCEPTED")
        raise SystemExit("ID_BASE_NOT_ACCEPTED")
    (formal_root / "ID_BASE_VALIDATED").write_text("validated\n", encoding="utf-8")
    write_state("id_formal_gate_100", "complete", checkpoint=str(selected), summary=summary)
    return selected, formal_root


def main() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    write_state("controller_start", "running", controller_pid=os.getpid())
    wait_for_oracle_v6()
    dataset_root = collect_id_dataset()
    audit_root = audit_id_dataset(dataset_root)
    training_root = train_id_policy(dataset_root)
    selected, formal_gate = select_and_gate(training_root)
    write_state(
        "id_base_complete",
        "ready_for_passive_detection",
        dataset=str(dataset_root),
        audit=str(audit_root),
        training=str(training_root),
        selected_checkpoint=str(selected),
        formal_gate=str(formal_gate),
        next_stage="passive_detection_and_four_method_dagger",
    )
    # The next stage is intentionally an explicit handoff marker. The full
    # passive/DAgger controller is added before this controller is authorized
    # to advance, so an incomplete implementation cannot masquerade as a run.
    raise SystemExit("ID_BASE_READY_FOR_PASSIVE_STAGE")


if __name__ == "__main__":
    main()

