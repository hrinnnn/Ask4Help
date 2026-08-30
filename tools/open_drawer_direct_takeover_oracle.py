#!/usr/bin/env python3
"""Current-state OpenDrawer takeover oracle.

The legacy oracle always inserted a handle-retreat motion before attempting a
grasp.  That is unnecessary once the drawer is already open and can create the
large lift/rotation seen in timing videos.  This module keeps only the
task-required branch: open the drawer if needed, then move directly from the
current TCP state to a shortest-path object grasp, lift, transport and release.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from rlinf.envs.maniskill.open_drawer_retrieve_place_spec import DRAWER_OPEN_THRESHOLD
from toolkits.lerobot.validate_open_drawer_retrieve_place_oracle import (
    _close_until_object_grasped,
    _hold_gripper,
    _move_to_best_pose,
    _move_to_pose,
    _scalar,
    _top_down_grasp,
    _vector,
)


def _drawer_is_open(base: Any) -> bool:
    return bool(_vector(base.drawer.get_qpos(), 1)[0] <= -DRAWER_OPEN_THRESHOLD)


def _object_grasp_candidates(base: Any, center: np.ndarray, closing: np.ndarray) -> list[Any]:
    """Return two sign choices so the planner can choose the shortest rotation."""

    axis = np.asarray(closing, dtype=np.float64).reshape(-1)[:3].copy()
    axis[2] = 0.0
    norm = float(np.linalg.norm(axis))
    if norm < 1e-8:
        axis = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        axis /= norm
    return [
        _top_down_grasp(base, center, axis),
        _top_down_grasp(base, center, -axis),
    ]


def _open_drawer_from_current_state(env: Any, planner: Any, stages: dict[str, Any]) -> bool:
    """Perform only the necessary handle interaction when the drawer is closed."""

    import sapien

    base = env.unwrapped
    handle_center = _vector(base.handle_world_position, 3)
    handle_grasp = _top_down_grasp(base, handle_center, np.asarray([0.0, 1.0, 0.0]))
    reached, steps = _move_to_pose(
        env,
        planner,
        handle_grasp * sapien.Pose([0.0, 0.0, -0.060]),
        gripper=1.0,
        position_tolerance=0.018,
    )
    stages["direct_handle_pregrasp_completed"] = reached
    stages["direct_handle_pregrasp_steps"] = steps
    reached, steps = _move_to_pose(
        env, planner, handle_grasp, gripper=1.0, position_tolerance=0.018
    )
    stages["direct_handle_reach_completed"] = stages["direct_handle_pregrasp_completed"] and reached
    stages["direct_handle_reach_steps"] = steps
    if not stages["direct_handle_reach_completed"]:
        stages["direct_pull_completed"] = False
        stages["direct_pull_steps"] = 0
        return False
    _hold_gripper(env, gripper=-1.0, steps=8)
    tcp = base.agent.tcp.pose.sp
    pull_target = sapien.Pose(tcp.p + np.asarray([-0.37, 0.0, 0.0]), tcp.q)
    moved, steps = _move_to_pose(
        env, planner, pull_target, gripper=-1.0, position_tolerance=0.018
    )
    stages["direct_pull_completed"] = moved
    stages["direct_pull_steps"] = steps
    return _drawer_is_open(base)


def continue_episode(env: Any, planner: Any, *, seed: int | None = None) -> dict[str, Any]:
    """Continue from the live takeover state without resetting or retreating."""

    import sapien

    base = env.unwrapped
    stages: dict[str, Any] = {
        "seed": None if seed is None else int(seed),
        "split": base.rlinf_split,
        "takeover_from_current_state": True,
        "oracle_mode": "direct_grasp_from_current_state",
        "planner_mode": getattr(planner, "planner_mode", os.environ.get("PANDA_PLANNER_MODE", "unknown")),
        "unnecessary_handle_retreat_removed": True,
    }
    drawer_opened_before = _drawer_is_open(base)
    stages["drawer_opened_before_takeover"] = drawer_opened_before
    if not drawer_opened_before:
        drawer_opened = _open_drawer_from_current_state(env, planner, stages)
    else:
        drawer_opened = True
        stages["direct_handle_pregrasp_completed"] = True
        stages["direct_handle_pregrasp_steps"] = 0
        stages["direct_handle_reach_completed"] = True
        stages["direct_handle_reach_steps"] = 0
        stages["direct_pull_completed"] = True
        stages["direct_pull_steps"] = 0
    stages["drawer_opened_after_takeover"] = bool(drawer_opened and _drawer_is_open(base))
    if not stages["drawer_opened_after_takeover"]:
        stages["success"] = False
        stages["failure_phase"] = "drawer_open"
        return stages

    grasped_before = _scalar(base.agent.is_grasping(base.obj))
    stages["object_grasped_before_takeover"] = grasped_before
    if grasped_before:
        grasped = True
        stages["direct_object_pregrasp_completed"] = True
        stages["direct_object_pregrasp_steps"] = 0
        stages["direct_object_reach_completed"] = True
        stages["direct_object_reach_steps"] = 0
        stages["direct_object_close_steps"] = 0
    else:
        # Do not lift or retreat before grasping.  Plan directly from the
        # current TCP state and let shortest-path selection choose the least
        # rotationally expensive sign of the object closing axis.
        _hold_gripper(env, gripper=1.0, steps=2)
        object_matrix = base.obj.pose.to_transformation_matrix()[0].detach().cpu().numpy()
        object_center = object_matrix[:3, 3]
        object_closing = object_matrix[:3, 1]
        grasps = _object_grasp_candidates(base, object_center, object_closing)
        reached, steps = _move_to_best_pose(
            env,
            planner,
            [grasp * sapien.Pose([0.0, 0.0, -0.100]) for grasp in grasps],
            gripper=1.0,
            position_tolerance=0.022,
        )
        stages["direct_object_pregrasp_completed"] = reached
        stages["direct_object_pregrasp_steps"] = steps
        reached, steps = _move_to_best_pose(
            env, planner, grasps, gripper=1.0, position_tolerance=0.018
        )
        stages["direct_object_reach_completed"] = stages["direct_object_pregrasp_completed"] and reached
        stages["direct_object_reach_steps"] = steps
        if stages["direct_object_reach_completed"]:
            grasped, close_steps = _close_until_object_grasped(env, base, max_steps=18, stable_steps=3)
        else:
            grasped, close_steps = False, 0
        stages["direct_object_close_steps"] = close_steps
    stages["object_grasped_after_takeover"] = bool(grasped)
    if not grasped:
        stages["success"] = False
        stages["failure_phase"] = "direct_grasp"
        return stages

    initial_object_z = float(_vector(base.obj.pose.p, 3)[2])
    already_lifted = initial_object_z >= 0.12
    if already_lifted:
        lifted = True
        stages["direct_lift_completed"] = True
        stages["direct_lift_steps"] = 0
    else:
        object_in_tcp = base.agent.tcp.pose.sp.inv() * base.obj.pose.sp
        lifted_object = sapien.Pose(
            base.obj.pose.sp.p + np.asarray([0.0, 0.0, 0.12]), base.obj.pose.sp.q
        )
        moved, steps = _move_to_pose(
            env, planner, lifted_object * object_in_tcp.inv(), gripper=-1.0,
            position_tolerance=0.020,
        )
        lifted = bool(moved and _scalar(base.agent.is_grasping(base.obj)))
        stages["direct_lift_completed"] = moved
        stages["direct_lift_steps"] = steps
    stages["object_lifted_after_takeover"] = bool(lifted)
    if not lifted:
        stages["success"] = False
        stages["failure_phase"] = "direct_lift"
        return stages

    object_in_tcp = base.agent.tcp.pose.sp.inv() * base.obj.pose.sp
    target_xy = _vector(base.target_tray.pose.p, 3)[:2]
    current_object = base.obj.pose.sp
    above_target = sapien.Pose([target_xy[0], target_xy[1], 0.15], current_object.q)
    moved, steps = _move_to_pose(
        env, planner, above_target * object_in_tcp.inv(), gripper=-1.0,
        position_tolerance=0.030,
    )
    stages["direct_transport_completed"] = moved
    stages["direct_transport_steps"] = steps
    place_target = sapien.Pose([target_xy[0], target_xy[1], 0.043], base.obj.pose.sp.q)
    moved, steps = _move_to_pose(
        env, planner, place_target * object_in_tcp.inv(), gripper=-1.0,
        position_tolerance=0.030,
    )
    stages["direct_place_completed"] = stages["direct_transport_completed"] and moved
    stages["direct_place_steps"] = steps
    _hold_gripper(env, gripper=1.0, steps=4)
    evaluation = base.evaluate()
    for name in (
        "success", "ever_drawer_opened", "ever_grasped", "ever_lifted",
        "object_in_target", "object_released", "is_robot_static",
    ):
        stages[name] = _scalar(evaluation[name])
    stages["final_object_position"] = _vector(base.obj.pose.p, 3).tolist()
    stages["target_position"] = _vector(base.target_tray.pose.p, 3).tolist()
    stages["failure_phase"] = None if stages["success"] else "direct_place"
    return stages
