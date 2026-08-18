#!/usr/bin/env python3
"""Read-only audit of the current v4 StackPyramid Oracle execution trace."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import MethodType
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch


def _scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    return array.reshape(-1)[0].item() if array.size == 1 else value


def _bool(value: Any) -> bool:
    return bool(_scalar(value))


def _pose_snapshot(recorder: Any) -> dict[str, Any]:
    base = recorder.unwrapped
    cubes = (base.cubeA, base.cubeB, base.cubeC)
    positions = [
        cube.pose.p.detach().cpu().numpy().reshape(-1, 3)[0].astype(float).tolist()
        for cube in cubes
    ]
    target_xy_distance = float(np.linalg.norm(np.asarray(positions[0][:2]) - np.asarray(positions[1][:2])))
    target_xy_tolerance = float(
        np.linalg.norm(2 * base.cube_half_size[:2].detach().cpu().numpy()) + 0.005
    )
    return {
        "action_step": len(recorder.actions),
        "red_pose": positions[0],
        "green_pose": positions[1],
        "blue_pose": positions[2],
        "red_green_xy_distance": target_xy_distance,
        "red_green_xy_tolerance": target_xy_tolerance,
        "grasped": [
            _bool(base.agent.is_grasping(cube)) for cube in cubes
        ],
        "gripper_closed": bool(recorder.gripper_closed),
    }


def _wrap_trace(recorder: Any, oracle: Any) -> dict[str, Any]:
    trace: dict[str, Any] = {"planner_calls": [], "pose_history": []}
    original_step = recorder.step

    def audited_step(self: Any, action: Any):
        result = original_step(action)
        trace["pose_history"].append(_pose_snapshot(self))
        return result

    recorder.step = MethodType(audited_step, recorder)
    trace["pose_history"].append(_pose_snapshot(recorder))
    planner = oracle.planner
    for name in ("open_gripper", "close_gripper", "move_to_pose_with_screw"):
        original = getattr(planner, name)

        def wrapped(*args: Any, _name=name, _original=original, **kwargs: Any):
            before = _pose_snapshot(recorder)
            start = len(recorder.actions)
            result = _original(*args, **kwargs)
            after = _pose_snapshot(recorder)
            trace["planner_calls"].append(
                {
                    "call_index": len(trace["planner_calls"]),
                    "method": _name,
                    "action_step_start": start,
                    "action_step_end": len(recorder.actions),
                    "dry_run": bool(kwargs.get("dry_run", False)),
                    "before": before,
                    "after": after,
                }
            )
            return result

        setattr(planner, name, wrapped)
    return trace


def _save_video(frames: list[np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=10, codec="libx264", macro_block_size=None) as writer:
        for frame in frames:
            writer.append_data(frame)


def audit_episode(env: Any, *, seed: int, output: Path, index: int, oracle_cls: Any) -> dict[str, Any]:
    from tools.collect_stackpyramid_xvla_dagger import StepRecorder

    recorder = StepRecorder(env)
    recorder.reset(seed=seed)
    oracle = oracle_cls(recorder)
    trace = _wrap_trace(recorder, oracle)
    error = None
    try:
        oracle.run()
    except Exception as exc:
        error = repr(exc)
    evaluation = recorder.unwrapped.evaluate()
    success = _bool(evaluation["success"] if isinstance(evaluation, dict) else evaluation)
    action_array = np.asarray(recorder.actions, dtype=np.float32)
    action_path = output / "actions" / f"episode_{index:03d}_seed_{seed}.npy"
    action_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(action_path, action_array)
    pose_path = output / "state" / f"episode_{index:03d}_seed_{seed}.json"
    pose_path.parent.mkdir(parents=True, exist_ok=True)
    pose_path.write_text(json.dumps(trace["pose_history"], indent=2) + "\n", encoding="utf-8")
    trace_path = output / "planner_calls" / f"episode_{index:03d}_seed_{seed}.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(json.dumps(trace["planner_calls"], indent=2) + "\n", encoding="utf-8")
    video_path = output / "videos" / f"episode_{index:03d}_seed_{seed}.mp4"
    _save_video(recorder.frames, video_path)
    closed = action_array[:, -1] < 0.0 if action_array.size else np.asarray([], dtype=bool)
    transitions = [
        {
            "action_step": int(index),
            "from_closed": bool(closed[index - 1]),
            "to_closed": bool(closed[index]),
        }
        for index in range(1, len(closed))
        if bool(closed[index]) != bool(closed[index - 1])
    ]
    return {
        "seed": seed,
        "strict_success": bool(success),
        "oracle_error": error,
        "action_steps": int(len(action_array)),
        "event_first_steps": dict(recorder.event_first_steps),
        "event_history": recorder.event_history,
        "gripper_transitions": transitions,
        "planner_calls": trace["planner_calls"],
        "actions": str(action_path),
        "state": str(pose_path),
        "video": str(video_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=84400)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--geometry", default="v4")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    sys.path[:0] = [str(args.repo_root), str(args.xvla_root)]
    import os

    os.environ["STACKPYRAMID_OOD_GEOMETRY"] = args.geometry
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    from tools.collect_stackpyramid_xvla_dagger import (
        StackPyramidOracle,
        StepRecorder,
        _install_rrt_fallback,
    )
    from tools.stackpyramid_task import register_stackpyramid_splits, stackpyramid_env_id

    _install_rrt_fallback()
    register_stackpyramid_splits()
    env = gym.make(
        stackpyramid_env_id("id"),
        obs_mode="rgb+state",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        sim_backend="cpu",
        render_backend="cpu",
    )
    rows = []
    try:
        for index in range(args.episodes):
            rows.append(
                audit_episode(
                    env,
                    seed=args.seed_start + index,
                    output=args.output,
                    index=index,
                    oracle_cls=StackPyramidOracle,
                )
            )
            print(json.dumps(rows[-1], indent=2), flush=True)
    finally:
        env.close()
    summary = {
        "format": "stackpyramid_current_oracle_readonly_audit_v1",
        "geometry": args.geometry,
        "seed_start": args.seed_start,
        "episodes": len(rows),
        "rows": rows,
        "strict_successes": sum(int(row["strict_success"]) for row in rows),
        "oracle_errors": sum(int(row["oracle_error"] is not None) for row in rows),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "AUDIT_COMPLETE").write_text(
        "Current v4 Oracle read-only trace audit complete.\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
