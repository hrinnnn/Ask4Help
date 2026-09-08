#!/usr/bin/env python3
"""Isolated Panda pose planner for scenes that contain custom articulations."""

from __future__ import annotations

import json
import os
import site
import sys

external_site_packages = os.environ.get("PANDA_PLANNER_SITE_PACKAGES")
if external_site_packages:
    site.addsitedir(external_site_packages)

import gymnasium as gym
import numpy as np
import sapien

import mani_skill.envs  # noqa: F401
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)


def _path_cost(positions: np.ndarray, qpos: np.ndarray) -> float:
    """Prefer short joint motion while retaining the planner's valid path."""
    values = np.asarray(positions, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 7:
        return float("inf")
    start = np.asarray(qpos, dtype=np.float64).reshape(-1)[:7]
    deltas = np.diff(np.vstack([start, values[:, :7]]), axis=0)
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0])
    return float(np.abs(deltas * weights).sum() + 0.01 * len(values))


def _plan_target(solver, target: np.ndarray, qpos: np.ndarray, timestep: float, mode: str):
    screw = solver.planner.plan_screw(
        target,
        qpos,
        time_step=timestep,
        use_point_cloud=False,
    )
    if mode != "shortest_joint_path":
        if screw.get("status") == "Success":
            return screw, "screw", _path_cost(screw["position"], qpos)
        fallback = solver.planner.plan_qpos_to_pose(
            target,
            qpos,
            time_step=timestep,
            wrt_world=True,
        )
        return fallback, "qpos", _path_cost(fallback["position"], qpos) if fallback.get("status") == "Success" else float("inf")

    candidates = []
    if screw.get("status") == "Success":
        candidates.append((screw, "screw"))
    qpos_path = solver.planner.plan_qpos_to_pose(
        target,
        qpos,
        time_step=timestep,
        wrt_world=True,
    )
    if qpos_path.get("status") == "Success":
        candidates.append((qpos_path, "qpos"))
    if not candidates:
        return screw, "none", float("inf")
    result, strategy = min(candidates, key=lambda item: _path_cost(item[0]["position"], qpos))
    return result, strategy, _path_cost(result["position"], qpos)


def main() -> None:
    env = gym.make(
        "PickCube-v1",
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode=None,
        render_backend=os.environ.get("PANDA_PLANNER_RENDER_BACKEND", "cpu"),
        sim_backend="cpu",
    )
    env.reset(seed=0)
    base = env.unwrapped
    solver = PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=False,
        base_pose=base.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
    )
    print("READY", flush=True)
    try:
        for line in sys.stdin:
            request = json.loads(line)
            if request.get("command") == "close":
                break
            target_world = sapien.Pose(request["target_p"], request["target_q"])
            planning_pose = solver._transform_pose_for_planning(target_world)
            target = np.concatenate([planning_pose.p, planning_pose.q])
            qpos = np.asarray(request["qpos"], dtype=np.float64)
            timestep = float(request["time_step"])
            mode = request.get("planner_mode", "screw_then_qpos")
            result, strategy, path_cost = _plan_target(solver, target, qpos, timestep, mode)
            response = {"status": result.get("status", "Unknown")}
            if result.get("status") == "Success":
                response["positions"] = np.asarray(result["position"]).tolist()
                response["strategy"] = strategy
                response["path_cost"] = path_cost
            print("RESULT " + json.dumps(response, separators=(",", ":")), flush=True)
    finally:
        solver.close()
        env.close()


if __name__ == "__main__":
    main()
