#!/usr/bin/env python3
"""Advance ID training to checkpoint selection and the formal ID gate."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = Path("/data/zhaozhixuan/Ask4Help-airplane-5090/results/object_variation_pick_single_ycb_v1")
PY = Path("/data/zhaozhixuan/Ask4Help-airplane-5090/RLinf/.venv/bin/python")
MODEL = Path("/data/zhaozhixuan/Ask4Help-open-drawer/results/model_cache/pi05_base_pytorch_v1")


def write_state(**updates: object) -> None:
    path = RUN / "pipeline_state.json"
    state = json.loads(path.read_text()) if path.exists() else {}
    state.update(updates)
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def checkpoint_for(step: int) -> Path | None:
    candidates = list((RUN / "id_training_v1" / "formal_10000_retry1").glob(f"**/checkpoints/global_step_{step}/actor/model_state_dict/full_weights.pt"))
    return candidates[0].parent.parent.parent if candidates else None


def evidence_complete(output: Path, expected: int) -> bool:
    summary = output / "summary.json"
    videos = output / "videos"
    return summary.is_file() and len(list(videos.glob("*.mp4"))) == expected


def stable_checkpoint(step: int) -> Path | None:
    checkpoint = checkpoint_for(step)
    if checkpoint is None:
        return None
    weights = checkpoint / "actor/model_state_dict/full_weights.pt"
    if not weights.is_file() or weights.stat().st_size < 1024 * 1024:
        return None
    first_size = weights.stat().st_size
    time.sleep(2)
    if not weights.is_file() or weights.stat().st_size != first_size:
        return None
    return checkpoint


def training_pid(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        return int(path.read_text().strip())
    except ValueError:
        return None


def process_alive(pid: int | None) -> bool:
    return pid is not None and Path(f"/proc/{pid}").exists()


def stop_training(pid: int | None) -> None:
    if not process_alive(pid):
        return
    os.kill(pid, signal.SIGTERM)
    deadline = time.time() + 300
    while process_alive(pid) and time.time() < deadline:
        time.sleep(10)
    if process_alive(pid):
        raise RuntimeError(f"training PID {pid} did not stop after SIGTERM")


def probe_output(root: Path, step: int) -> Path:
    base = root / f"step_{step}"
    if not base.exists() or evidence_complete(base, 20):
        return base
    retry = 1
    while True:
        candidate = root / f"step_{step}_retry{retry}"
        if not candidate.exists():
            return candidate
        if evidence_complete(candidate, 20):
            return candidate
        retry += 1


def run_eval(checkpoint: Path, output: Path, episodes: int, seed: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if evidence_complete(output, episodes):
        return
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1"
    env["OMP_NUM_THREADS"] = "20"
    env["MKL_NUM_THREADS"] = "20"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONPATH"] = f"{RUN / 'runtime_overlay' / 'numpy126'}:{ROOT / 'RLinf'}:{ROOT}"
    command = [
        str(PY),
        str(ROOT / "tools/evaluate_pick_single_ycb_object_variation_pi05.py"),
        "--checkpoint",
        str(checkpoint),
        "--pi05-base",
        str(MODEL),
        "--norm-stats",
        str(RUN / "datasets/id_v1_retry1/norm_stats.json"),
        "--output-dir",
        str(output),
        "--split",
        "id",
        "--episodes",
        str(episodes),
        "--seed",
        str(seed),
        "--execute-horizon",
        "5",
        "--max-episode-steps",
        "200",
    ]
    log = output.parent / f"{output.name}.log"
    with log.open("w", encoding="utf-8") as stream:
        subprocess.run(
            ["taskset", "-c", "0-19", *command],
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if not evidence_complete(output, episodes):
        raise RuntimeError(f"incomplete evaluation evidence: {output}")


def main() -> None:
    training_pid_path = RUN / "id_training_v1/formal_10000_retry1/train.pid"
    probe_root = RUN / "id_checkpoint_probe_20id"
    results_path = probe_root / "probe_results.json"
    probe_root.mkdir(parents=True, exist_ok=True)
    existing_results = json.loads(results_path.read_text()) if results_path.is_file() else {}
    early_marker = RUN / "ID_EARLY_STOP_SELECTED"
    write_state(
        current_stage="id_sft_formal_training_retry1",
        next_stage="id_checkpoint_probe_and_early_stop",
        post_training_controller_pid=os.getpid(),
    )

    early_selected: dict | None = None
    while True:
        pid = training_pid(training_pid_path)
        for step in range(500, 10001, 500):
            checkpoint = stable_checkpoint(step)
            if checkpoint is None or str(step) in existing_results:
                continue
            write_state(current_stage="id_checkpoint_probe", next_stage="id_checkpoint_probe_and_early_stop")
            output = probe_output(probe_root, step)
            run_eval(checkpoint, output, 20, 20000 + step)
            summary = json.loads((output / "summary.json").read_text())
            row = {
                "step": step,
                "checkpoint": str(checkpoint),
                "output": str(output),
                "strict_successes": int(summary["successes"]),
                "episodes": int(summary["episodes"]),
                "video_count": int(summary["videos"]),
                "gate": "passed_gt_80_percent" if int(summary["successes"]) > 16 else "not_passed",
            }
            existing_results[str(step)] = row
            results_path.write_text(json.dumps(existing_results, indent=2) + "\n", encoding="utf-8")
            if int(summary["successes"]) > 16:
                early_selected = row
                early_marker.write_text(json.dumps({"selected": row, "rule": ">16/20 strict ID successes"}, indent=2) + "\n", encoding="utf-8")
                write_state(current_stage="id_early_stop_requested", next_stage="id_formal_checkpoint_validation")
                stop_training(pid)
                break
        if early_selected is not None or not process_alive(pid):
            break
        time.sleep(60)

    all_steps = list(range(500, 10001, 500))
    stop_step = early_selected["step"] if early_selected else 10000
    required_steps = list(range(500, stop_step + 1, 500))
    all_checkpoints = {step: stable_checkpoint(step) for step in required_steps}
    if any(value is None for value in all_checkpoints.values()):
        payload = {str(k): str(v) for k, v in all_checkpoints.items()}
        (RUN / "ID_TRAINING_CHECKPOINT_AUDIT_FAILED").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        write_state(current_stage="id_checkpoint_audit_failed", next_stage="needs_user_decision_or_engineering_retry", terminal_marker="ID_TRAINING_CHECKPOINT_AUDIT_FAILED")
        raise SystemExit(2)
    (RUN / "id_training_v1/formal_10000_retry1/checkpoint_audit_all_500.json").write_text(
        json.dumps({str(k): str(v) for k, v in all_checkpoints.items()}, indent=2) + "\n", encoding="utf-8"
    )

    selection = list(existing_results.values())
    if not selection:
        raise RuntimeError("no checkpoint probe evidence was produced")
    selected = early_selected or sorted(selection, key=lambda row: (-row["strict_successes"], row["step"]))[0]
    (RUN / "id_checkpoint_selection").mkdir(parents=True, exist_ok=True)
    (RUN / "id_checkpoint_selection/selection.json").write_text(
        json.dumps({"candidates": selection, "selected": selected, "early_stop": early_selected is not None}, indent=2) + "\n", encoding="utf-8"
    )

    write_state(current_stage="id_formal_checkpoint_validation", next_stage="passive_failure_detection")
    formal = RUN / "id_formal_gate_100"
    run_eval(Path(selected["checkpoint"]), formal, 100, 50000)
    summary = json.loads((formal / "summary.json").read_text())
    payload = {
        "selected": selected,
        "formal_summary": summary,
        "formal_id_rule": "first checkpoint with >80 percent on 20 ID probe; 100-ID evaluation is retained as independent validation evidence",
    }
    if summary["successes"] < 80 and early_selected is None:
        (RUN / "ID_BASE_NOT_ACCEPTED").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        (RUN / "NEEDS_USER_DECISION").write_text(json.dumps({"reason": "no 20-ID checkpoint probe passed and fallback 100-ID validation was below 80/100", "evidence": payload}, indent=2) + "\n", encoding="utf-8")
        write_state(current_stage="id_base_not_accepted", next_stage="needs_user_decision_or_scientific_recovery", terminal_marker="NEEDS_USER_DECISION")
        return
    (RUN / "ID_BASE_VALIDATED").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_state(current_stage="id_base_validated", next_stage="passive_failure_detection", terminal_marker="ID_BASE_VALIDATED")


if __name__ == "__main__":
    main()
