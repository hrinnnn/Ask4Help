#!/usr/bin/env python3
"""Run paired ID/radial-OOD Oracle smoke or gate with full evidence."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "RLinf")]

from rlinf.envs.maniskill.stack_cube_privileged_oracle import (  # noqa: E402
    StackCubePrivilegedChunkOracle,
)
from tools.collect_stackcube_xvla_dagger import (  # noqa: E402
    MAIN_CAMERA_CANDIDATES,
    WRIST_CAMERA_CANDIDATES,
    _build_frames,
    _extract_record,
    _select_camera,
)
from tools.evaluate_stackcube_xvla import bool_scalar  # noqa: E402
from tools.stackcube_stage1_radial_ood import (  # noqa: E402
    STACK_CUBE_RADIAL_OOD_SPLIT,
    STACK_CUBE_TASK,
    register_stackcube_stage1_radial_variants,
    radial_env_id,
    stage1_radial_reset_metadata,
    stage1_radial_state_record,
)
from toolkits.lerobot.collect_maniskill_plug_lerobot_joint import (  # noqa: E402
    write_episode_video_durably,
)


CHUNK_SIZE = 5
MAX_EPISODE_STEPS = 100


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def make_env(split: str) -> Any:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    return gym.make(
        radial_env_id(split),
        robot_uids="panda_wristcam",
        num_envs=1,
        obs_mode="rgb",
        control_mode="pd_joint_delta_pos",
        reward_mode="sparse",
        render_mode="rgb_array",
        sim_backend="physx_cpu",
        sim_config={"sim_freq": 100, "control_freq": 10},
        sensor_configs={"width": 384, "height": 384},
        max_episode_steps=MAX_EPISODE_STEPS,
    )


def save_video(
    records: list[Any],
    actions: list[np.ndarray],
    output: Path,
    index: int,
    seed: int,
) -> str:
    main_camera = _select_camera(
        records[0].obs, "", ("base_camera",) + MAIN_CAMERA_CANDIDATES, "main"
    )
    wrist_camera = _select_camera(
        records[0].obs, "", ("hand_camera",) + WRIST_CAMERA_CANDIDATES, "wrist"
    )
    frames = _build_frames(
        records=records,
        actions=actions,
        task=STACK_CUBE_TASK,
        main_camera=main_camera,
        wrist_camera=wrist_camera,
    )
    path = write_episode_video_durably(
        frames,
        video_dir=output / "videos",
        episode_index=index,
        seed=seed,
        fps=10,
    )
    return str(path)


def run_episode(env: Any, split: str, seed: int, output: Path, index: int) -> dict[str, Any]:
    raw_obs, _ = env.reset(seed=seed)
    records = [_extract_record(raw_obs)]
    actions: list[np.ndarray] = []
    state_timeline: list[dict[str, Any]] = [stage1_radial_state_record(env)]
    oracle_phases: list[str] = []
    ever_grasped = ever_lifted = ever_placed = success = False
    terminated = truncated = False
    oracle = StackCubePrivilegedChunkOracle(chunk_size=CHUNK_SIZE)
    failure_reason: str | None = None
    planning_failures = 0

    reset_metadata = stage1_radial_reset_metadata(env, split=split, paired_seed=seed)
    while len(actions) < MAX_EPISODE_STEPS and not success:
        plan = oracle.plan(env)
        oracle_phases.append(str(plan.phase))
        planning_failures += int(not plan.planning_succeeded)
        for local_step in range(CHUNK_SIZE):
            action = plan.action_at(raw_obs["agent"]["qpos"], local_step)
            raw_obs, _, terminated, truncated, info = env.step(
                torch.as_tensor(action, device=env.unwrapped.device).unsqueeze(0)
            )
            actions.append(np.asarray(action, dtype=np.float32))
            records.append(_extract_record(raw_obs))
            current = stage1_radial_state_record(env, info)
            state_timeline.append(current)
            predicates = current["predicates"]
            ever_grasped |= bool(predicates["red_grasped"])
            ever_lifted |= bool(predicates["red_lifted"])
            ever_placed |= bool(predicates["red_placed"])
            success = bool_scalar(info.get("success", False))
            if success or bool_scalar(terminated) or bool_scalar(truncated):
                break
        if planning_failures and not actions:
            break

    if not success:
        failure_reason = (
            "oracle_planning_failure"
            if planning_failures
            else "terminated_or_timeout"
            if bool_scalar(terminated) or bool_scalar(truncated)
            else "horizon_exhausted"
        )
    action_path = output / "actions" / f"episode_{index:03d}_seed_{seed}.npy"
    action_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(action_path, np.asarray(actions, dtype=np.float32))
    video_path = save_video(records, actions, output, index, seed)
    return {
        "episode_index": index,
        "seed": seed,
        "split": split,
        "success": bool(success),
        "ever_grasped": bool(ever_grasped),
        "ever_lifted": bool(ever_lifted),
        "ever_placed": bool(ever_placed),
        "steps": len(actions),
        "oracle_phases": oracle_phases,
        "oracle_phase_counts": dict(Counter(oracle_phases)),
        "planning_failures": planning_failures,
        "failure_reason": failure_reason,
        "metadata": reset_metadata,
        "state_timeline": state_timeline,
        "actions": str(action_path),
        "video": video_path,
    }


def summarize(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    return {
        "split": split,
        "episodes": len(rows),
        "successes": sum(int(row["success"]) for row in rows),
        "success_rate": float(np.mean([row["success"] for row in rows])) if rows else 0.0,
        "ever_grasped": sum(int(row["ever_grasped"]) for row in rows),
        "ever_lifted": sum(int(row["ever_lifted"]) for row in rows),
        "ever_placed": sum(int(row["ever_placed"]) for row in rows),
        "video_count": sum(int(bool(row.get("video"))) for row in rows),
        "action_count": sum(int(bool(row.get("actions"))) for row in rows),
        "failure_reasons": dict(
            Counter(row["failure_reason"] for row in rows if row["failure_reason"])
        ),
        "phase_counts": dict(Counter(phase for row in rows for phase in row["oracle_phases"])),
        "state_timeline_complete": all(bool(row.get("state_timeline")) for row in rows),
        "metadata_complete": all(bool(row.get("metadata")) for row in rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "gate"), default="smoke")
    parser.add_argument("--seed-start", type=int, default=971100)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--min-success-rate", type=float, default=0.90)
    parser.add_argument("--local-output", type=Path, required=True)
    parser.add_argument("--persistent-output", type=Path, required=True)
    args = parser.parse_args()
    if args.local_output.exists() or args.persistent_output.exists():
        raise FileExistsError("radial Oracle output roots must be new")
    args.local_output.mkdir(parents=True)
    write_json(
        args.local_output / "protocol.json",
        {
            "task": "StackCube Stage1 radial-distance OOD Oracle qualification",
            "mode": args.mode,
            "paired_seed_start": args.seed_start,
            "episodes_per_split": args.episodes,
            "minimum_success_rate": args.min_success_rate,
            "splits": ["id", STACK_CUBE_RADIAL_OOD_SPLIT],
            "red_ood_rule": "red_id + 0.04 * normalize(red_id - green)",
            "max_episode_steps": MAX_EPISODE_STEPS,
            "chunk_size": CHUNK_SIZE,
            "env_lifecycle": "one gym.make per split, repeated reset for all episodes, close after split",
            "full_denominator_required": True,
        },
    )
    summaries: dict[str, Any] = {}
    register_stackcube_stage1_radial_variants()
    for split in ("id", STACK_CUBE_RADIAL_OOD_SPLIT):
        split_output = args.local_output / split
        rows = []
        env = make_env(split)
        try:
            for index in range(args.episodes):
                seed = args.seed_start + index
                row = run_episode(env, split, seed, split_output, index)
                rows.append(row)
                append_jsonl(split_output / "episodes.jsonl", row)
                print(
                    f"[radial-oracle] split={split} {index + 1}/{args.episodes} "
                    f"seed={seed} success={int(row['success'])} steps={row['steps']}",
                    flush=True,
                )
        finally:
            env.close()
        summaries[split] = summarize(rows, split)
        write_json(split_output / "summary.json", summaries[split])

    passed = all(
        summary["episodes"] == args.episodes
        and summary["success_rate"] >= args.min_success_rate
        and summary["video_count"] == args.episodes
        and summary["action_count"] == args.episodes
        and summary["state_timeline_complete"]
        and summary["metadata_complete"]
        for summary in summaries.values()
    )
    report = {
        "task": "StackCube Stage1 radial-distance OOD Oracle qualification",
        "mode": args.mode,
        "seed_start": args.seed_start,
        "episodes_per_split": args.episodes,
        "minimum_success_rate": args.min_success_rate,
        "red_ood_rule": "red_id + 0.04 * normalize(red_id - green)",
        "summaries": summaries,
        "qualification_passed": passed,
    }
    write_json(args.local_output / "summary.json", report)
    marker_prefix = "ORACLE_SMOKE" if args.mode == "smoke" else "ORACLE_GATE"
    marker = f"{marker_prefix}_{'PASSED' if passed else 'FAILED'}"
    (args.local_output / marker).write_text(
        "Radial StackCube Oracle qualification decision.\n", encoding="utf-8"
    )
    if args.persistent_output.exists():
        raise FileExistsError("refusing to overwrite persistent output")
    args.persistent_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(args.local_output, args.persistent_output)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
