#!/usr/bin/env python3
"""Collect raw StackPyramid motion-planning attempts on local storage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np

from mani_skill.examples.motionplanning.panda.solutions import solveStackPyramid
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.utils.wrappers.record import RecordEpisode

from tools.stackpyramid_task import (
    reset_metadata,
    register_stackpyramid_splits,
    stackpyramid_env_id,
)


def _install_rrt_fallback() -> None:
    original = PandaArmMotionPlanningSolver.move_to_pose_with_screw
    if getattr(original, "_stackpyramid_rrt_fallback", False):
        return

    def move_with_fallback(self, pose, dry_run=False, refine_steps=0):
        result = original(self, pose, dry_run=dry_run, refine_steps=refine_steps)
        if result != -1:
            return result
        return self.move_to_pose_with_RRTConnect(
            pose, dry_run=dry_run, refine_steps=refine_steps
        )

    move_with_fallback._stackpyramid_rrt_fallback = True
    PandaArmMotionPlanningSolver.move_to_pose_with_screw = move_with_fallback


def _value(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {str(key): _value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _scalar(value):
    value = _value(value)
    if isinstance(value, list) and len(value) == 1:
        return _scalar(value[0])
    return value


def _evaluation_details(env):
    base = env.unwrapped
    positions = [cube.pose.p.detach().cpu().numpy()[0] for cube in (base.cubeA, base.cubeB, base.cubeC)]
    threshold = float(np.linalg.norm(2 * base.cube_half_size[:2].detach().cpu().numpy()) + 0.005)
    xy_ab = float(np.linalg.norm((positions[0] - positions[1])[:2])) <= threshold
    xy_cb = float(np.linalg.norm((positions[2] - positions[1])[:2])) <= threshold
    xy_ca = float(np.linalg.norm((positions[2] - positions[0])[:2])) <= threshold
    z_cb = abs(float(positions[2][2] - positions[1][2])) > 0.02
    z_ca = abs(float(positions[2][2] - positions[0][2])) > 0.02
    static = [bool(cube.is_static(lin_thresh=1e-2, ang_thresh=0.5).detach().cpu().numpy()[0]) for cube in (base.cubeA, base.cubeB, base.cubeC)]
    grasped = [bool(base.agent.is_grasping(cube).detach().cpu().numpy()[0]) for cube in (base.cubeA, base.cubeB, base.cubeC)]
    return {
        "positions": [position.tolist() for position in positions],
        "xy_threshold": threshold,
        "xy_ab": xy_ab,
        "xy_cb": xy_cb,
        "xy_ca": xy_ca,
        "z_cb": z_cb,
        "z_ca": z_ca,
        "static": static,
        "grasped": grasped,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("id", "stage1_ood", "stage2_ood", "stage3_ood"), required=True)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--start-seed", type=int, default=13000)
    parser.add_argument("--sim-backend", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--render-backend", choices=("gpu", "cpu"), default="gpu")
    args = parser.parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")

    register_stackpyramid_splits()
    _install_rrt_fallback()
    output_dir = args.output / stackpyramid_env_id(args.split) / "motionplanning"
    output_dir.mkdir(parents=True, exist_ok=True)
    env = gym.make(
        stackpyramid_env_id(args.split),
        obs_mode="rgb+state",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
    )
    env = RecordEpisode(
        env,
        output_dir=str(output_dir),
        trajectory_name=f"oracle_{args.split}",
        save_video=True,
        source_type="motionplanning",
        source_desc="official ManiSkill Panda StackPyramid solver on controlled split",
        video_fps=30,
        record_reward=False,
        save_on_reset=False,
    )

    attempts = []
    try:
        for offset in range(args.episodes):
            seed = args.start_seed + offset
            env.reset(seed=seed)
            initial_metadata = reset_metadata(env, split=args.split)
            planner_error = None
            result = None
            try:
                result = solveStackPyramid(env, seed=seed, debug=False, vis=False)
            except Exception as exc:  # preserve the raw attempt and diagnose later
                planner_error = repr(exc)

            success = False
            elapsed_steps = None
            evaluation = None
            if result != -1 and result is not None:
                final = result[-1]
                success = bool(_scalar(final["success"]))
                elapsed_steps = int(_scalar(final["elapsed_steps"]))
                evaluation = {
                    "success": _value(env.unwrapped.evaluate()),
                    "details": _evaluation_details(env),
                }
            attempts.append(
                {
                    "seed": seed,
                    "split": args.split,
                    "strict_success": success,
                    "elapsed_steps": elapsed_steps,
                    "planner_error": planner_error,
                    "final_evaluation": evaluation,
                    "initial_metadata": initial_metadata,
                }
            )
            env.flush_trajectory(save=True)
            env.flush_video(save=True)
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "strict_success": success,
                        "elapsed_steps": elapsed_steps,
                        "planner_error": planner_error,
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )
    finally:
        env.close()

    summary = {
        "format": "stackpyramid_oracle_raw_attempts_v1",
        "env_id": stackpyramid_env_id(args.split),
        "split": args.split,
        "episodes": len(attempts),
        "strict_successes": sum(item["strict_success"] for item in attempts),
        "success_rate": sum(item["strict_success"] for item in attempts) / len(attempts),
        "attempts": attempts,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"oracle_summary_{args.split}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
