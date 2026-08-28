"""Panda-to-X-VLA EE6D action conversion for the vegetable-basket task."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import sapien
from scipy.spatial.transform import Rotation


MODEL_ACTION_DIM = 20
ACTIVE_ACTION_DIM = 10
ACTION_HORIZON = 30


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def tcp_pose_world(env: Any) -> np.ndarray:
    """Return Panda TCP pose as ``xyz + quaternion(wxyz)`` in world frame."""

    link = getattr(env.unwrapped.agent, "tcp", None)
    if link is None:
        link = next(
            link for link in env.unwrapped.agent.robot.get_links()
            if link.name in {"panda_hand_tcp", "ee_gripper_link"}
        )
    return _array(link.pose.raw_pose).reshape(-1, 7)[0]


def world_pose_to_base(env: Any, world_pose: np.ndarray) -> np.ndarray:
    """Express a world-frame pose in the Panda robot-root frame."""

    root = env.unwrapped.agent.robot.pose
    root_p = _array(root.p).reshape(-1, 3)[0]
    root_q = _array(root.q).reshape(-1, 4)[0]
    root_rot = Rotation.from_quat(root_q[[1, 2, 3, 0]])
    world_q = _array(world_pose[3:]).reshape(4)
    world_rot = Rotation.from_quat(world_q[[1, 2, 3, 0]])
    base_p = root_rot.inv().apply(_array(world_pose[:3]) - root_p)
    base_rot = root_rot.inv() * world_rot
    base_q = base_rot.as_quat()[[3, 0, 1, 2]]
    return np.concatenate([base_p, base_q]).astype(np.float32)


def controller_target_pose_base(env: Any) -> np.ndarray | None:
    """Return the active Panda arm controller target in root-frame pose."""

    controller = getattr(env.unwrapped.agent, "controller", None)
    arm = getattr(controller, "controllers", {}).get("arm") if controller else None
    target = getattr(arm, "_target_pose", None)
    if target is None:
        return None
    return _array(target.raw_pose).reshape(-1, 7)[0]


def rotation_to_6d(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Encode a wxyz quaternion using X-VLA's first-two-column 6D format."""

    q = _array(quaternion_wxyz).reshape(4)
    matrix = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    return matrix[:, :2].reshape(6).astype(np.float32)


def rotation_from_6d(value: np.ndarray) -> Rotation:
    """Decode X-VLA's continuous 6D rotation representation."""

    value = _array(value).reshape(6)
    first = value[:3]
    second = value[3:]
    first = first / max(float(np.linalg.norm(first)), 1e-8)
    second = second - first * float(np.dot(first, second))
    second = second / max(float(np.linalg.norm(second)), 1e-8)
    third = np.cross(first, second)
    return Rotation.from_matrix(np.column_stack([first, second, third]))


def encode_base_ee6d(base_pose: np.ndarray, gripper_01: float) -> np.ndarray:
    """Encode one base-frame Panda pose in the active 10D X-VLA block."""

    result = np.zeros(MODEL_ACTION_DIM, dtype=np.float32)
    result[:ACTIVE_ACTION_DIM] = np.concatenate(
        [
            _array(base_pose[:3]),
            rotation_to_6d(base_pose[3:]),
            [float(np.clip(gripper_01, 0.0, 1.0))],
        ]
    )
    return result


def model_target_to_panda_action(
    env: Any,
    model_action: np.ndarray,
    *,
    position_limit: float = 0.1,
    rotation_limit: float = 0.1,
) -> np.ndarray:
    """Map an absolute base-frame X-VLA target to Panda's 7D env action."""

    action = _array(model_action).reshape(-1)
    if action.size < ACTIVE_ACTION_DIM:
        raise ValueError(f"model action must have at least 10 values, got {action.size}")
    current_world = tcp_pose_world(env)
    current_base = controller_target_pose_base(env)
    if current_base is None:
        current_base = world_pose_to_base(env, current_world)
    target_base_xyz = action[:3]
    target_base_rot = rotation_from_6d(action[3:9])
    delta_xyz = np.clip(target_base_xyz - current_base[:3], -position_limit, position_limit)
    current_base_rot = Rotation.from_quat(current_base[3:][[1, 2, 3, 0]])
    delta_rot = (current_base_rot.inv() * target_base_rot).as_rotvec()
    delta_rot = np.clip(delta_rot, -rotation_limit, rotation_limit)
    grip = float(np.clip(action[9] * 2.0 - 1.0, -1.0, 1.0))
    return np.concatenate([delta_xyz, delta_rot, [grip]]).astype(np.float32)


def target_world_to_panda_action(
    env: Any,
    target_world_xyz: np.ndarray,
    gripper_env: float,
    *,
    position_limit: float = 0.05,
) -> np.ndarray | dict[str, np.ndarray]:
    """Build a Panda command for a world-frame waypoint."""

    current = tcp_pose_world(env)
    return target_world_pose_to_panda_action(
        env,
        target_world_xyz,
        current[3:],
        gripper_env,
        position_limit=position_limit,
    )


def target_world_pose_to_panda_action(
    env: Any,
    target_world_xyz: np.ndarray,
    target_world_quaternion_wxyz: np.ndarray,
    gripper_env: float,
    *,
    position_limit: float = 0.05,
    rotation_limit: float = 0.1,
) -> np.ndarray | dict[str, np.ndarray]:
    """Build a Panda command for a world-frame pose waypoint."""

    current = tcp_pose_world(env)
    previous_target = controller_target_pose_base(env)
    if previous_target is None:
        previous_target = world_pose_to_base(env, current)
    target_world = np.concatenate(
        [
            _array(target_world_xyz).reshape(3),
            _array(target_world_quaternion_wxyz).reshape(4),
        ]
    )
    desired_base = world_pose_to_base(env, target_world)
    desired_base_xyz = desired_base[:3]
    delta = desired_base_xyz - previous_target[:3]
    delta = np.clip(delta, -position_limit, position_limit)
    previous_rot = Rotation.from_quat(previous_target[3:][[1, 2, 3, 0]])
    desired_rot = Rotation.from_quat(desired_base[3:][[1, 2, 3, 0]])
    delta_rot = (previous_rot.inv() * desired_rot).as_rotvec()
    delta_rot = np.clip(delta_rot, -rotation_limit, rotation_limit)
    arm = np.concatenate([delta, delta_rot]).astype(np.float32)
    if isinstance(env.action_space, gym.spaces.Dict):
        return {"arm": arm, "gripper": np.asarray([gripper_env], dtype=np.float32)}
    return np.concatenate([arm, [gripper_env]]).astype(np.float32)


def action_space_gripper_bounds(env: Any) -> tuple[float, float]:
    """Return the environment's open and closed gripper action values."""

    space = env.action_space
    if isinstance(space, gym.spaces.Dict):
        low, high = space["gripper"].low[0], space["gripper"].high[0]
    else:
        low, high = space.low[-1], space.high[-1]
    return float(low), float(high)


def model_target_to_panda_joint_action(
    env: Any,
    model_action: np.ndarray,
    *,
    pinocchio: Any | None = None,
    tcp_link_name: str = "panda_hand_tcp",
) -> np.ndarray:
    """Map an absolute base-frame EE6D target to Panda joint-position control.

    Oracle trajectories are generated with ManiSkill's Panda joint-position
    planner. Using the same IK target here keeps policy rollout dynamics
    aligned with the recorded EE6D targets instead of mixing them with a
    delta-pose controller.
    """

    value = _array(model_action).reshape(-1)
    if value.size < ACTIVE_ACTION_DIM:
        raise ValueError(f"model action must have at least 10 values, got {value.size}")
    if pinocchio is None:
        pinocchio = env.unwrapped.agent.robot.create_pinocchio_model()
    link_names = [link.name for link in env.unwrapped.agent.robot.get_links()]
    if tcp_link_name not in link_names:
        raise ValueError(f"cannot resolve {tcp_link_name}; available={link_names}")
    target_rotation = rotation_from_6d(value[3:9])
    target_pose = sapien.Pose(
        p=value[:3], q=target_rotation.as_quat()[[3, 0, 1, 2]]
    )
    current = _array(env.unwrapped.agent.robot.get_qpos()).reshape(-1).astype(np.float64)
    try:
        result, success, _error = pinocchio.compute_inverse_kinematics(
            link_names.index(tcp_link_name),
            target_pose,
            initial_qpos=current,
            active_qmask=np.asarray([1] * 7 + [0, 0], dtype=np.int32),
            max_iterations=100,
            dt=0.1,
            damp=1e-6,
        )
    except RuntimeError:
        # A policy can emit a temporarily unreachable target. Keep the raw
        # rollout alive and record a no-motion command for that decision;
        # the episode outcome remains governed by the task predicate.
        result, success = current[:7], False
    arm = np.asarray(result if success else current[:7], dtype=np.float32).reshape(-1)[:7]
    arm = np.clip(arm, env.action_space.low[:7], env.action_space.high[:7])
    gripper = float(np.clip(-0.01 + 0.05 * value[9], -0.01, 0.04))
    return np.concatenate([arm, [gripper]]).astype(np.float32)
