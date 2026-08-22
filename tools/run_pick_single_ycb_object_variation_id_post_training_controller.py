#!/usr/bin/env python3
"""Advance ID training to checkpoint selection and the formal ID gate."""

from __future__ import annotations

import json
import os
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
    write_state(current_stage="id_sft_formal_training_retry1", next_stage="id_checkpoint_selection_and_formal_gate", post_training_controller_pid=os.getpid())
    while True:
        pid = int(training_pid_path.read_text()) if training_pid_path.exists() else None
        if pid is None or not (Path(f"/proc/{pid}").exists()):
            break
        time.sleep(300)

    all_steps = list(range(500, 10001, 500))
    all_checkpoints = {step: checkpoint_for(step) for step in all_steps}
    if any(value is None for value in all_checkpoints.values()):
        (RUN / "ID_TRAINING_CHECKPOINT_AUDIT_FAILED").write_text(json.dumps({str(k): str(v) for k, v in all_checkpoints.items()}, indent=2) + "\n", encoding="utf-8")
        write_state(current_stage="id_checkpoint_audit_failed", next_stage="needs_user_decision_or_engineering_retry", terminal_marker="ID_TRAINING_CHECKPOINT_AUDIT_FAILED")
        raise SystemExit(2)
    (RUN / "id_training_v1/formal_10000_retry1/checkpoint_audit_all_500.json").write_text(
        json.dumps({str(k): str(v) for k, v in all_checkpoints.items()}, indent=2) + "\n",
        encoding="utf-8",
    )

    steps = [2000, 4000, 6000, 8000, 10000]
    checkpoints = {step: all_checkpoints[step] for step in steps}
    selection: list[dict] = []
    write_state(current_stage="id_checkpoint_selection", next_stage="formal_id_gate")
    for index, step in enumerate(steps):
        output = RUN / "id_checkpoint_selection" / f"step_{step}"
        run_eval(checkpoints[step], output, 20, 20000 + index * 20)
        summary = json.loads((output / "summary.json").read_text())
        selection.append({"step": step, "checkpoint": str(checkpoints[step]), "strict_successes": summary["successes"], "episodes": summary["episodes"], "video_count": summary["videos"]})
    selected = sorted(selection, key=lambda row: (-row["strict_successes"], row["step"]))[0]
    (RUN / "id_checkpoint_selection/selection.json").write_text(json.dumps({"candidates": selection, "selected": selected}, indent=2) + "\n", encoding="utf-8")

    write_state(current_stage="formal_id_gate", next_stage="passive_failure_detection")
    formal = RUN / "id_formal_gate_100"
    run_eval(Path(selected["checkpoint"]), formal, 100, 50000)
    summary = json.loads((formal / "summary.json").read_text())
    if summary["successes"] < 80:
        payload = {"selected": selected, "formal_summary": summary}
        (RUN / "ID_BASE_NOT_ACCEPTED").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        (RUN / "NEEDS_USER_DECISION").write_text(
            json.dumps(
                {
                    "reason": "selected ID policy did not meet the predeclared 80/100 strict-success gate",
                    "evidence": payload,
                    "allowed_next_decisions": [
                        "inspect the failure evidence and approve a new scientifically declared ID recovery",
                        "stop the object-variation pipeline without downstream claims",
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        write_state(current_stage="id_base_not_accepted", next_stage="needs_user_decision_or_scientific_recovery", terminal_marker="NEEDS_USER_DECISION")
        return
    (RUN / "ID_BASE_VALIDATED").write_text(json.dumps({"selected": selected, "formal_summary": summary}, indent=2) + "\n", encoding="utf-8")
    write_state(current_stage="id_base_validated", next_stage="passive_failure_detection", terminal_marker="ID_BASE_VALIDATED")


if __name__ == "__main__":
    main()
