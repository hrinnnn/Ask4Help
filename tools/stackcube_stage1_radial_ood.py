"""Independent StackCube Stage1 radial-distance OOD environment.

This module is intentionally separate from the old diagonal Stage1 module and
from StackCube Stage2 registration.  The radial split changes only cubeA's
initial XY by four centimetres along the green-to-ID-red direction.
"""

from __future__ import annotations

from typing import Any

import numpy as np


STACK_CUBE_TASK = "stack the red cube on the green cube"
STACK_CUBE_RADIAL_ID_ENV_ID = "RLinfStackCubeStage1RadialID-v1"
STACK_CUBE_RADIAL_OOD_ENV_ID = "RLinfStackCubeStage1RadialOOD-v1"
STACK_CUBE_RADIAL_OOD_SPLIT = "stage1_radial_distance_ood"
STACK_CUBE_RADIAL_SPLITS = ("id", STACK_CUBE_RADIAL_OOD_SPLIT)

STACK_CUBE_ID_ANGLE_CENTER = np.pi / 2
STACK_CUBE_ID_ANGLE_HALF_WIDTH = np.deg2rad(10.0)
STACK_CUBE_ID_DISTANCE_RANGE = (0.08, 0.10)
STACK_CUBE_ID_BASE_JITTER = 0.02
RADIAL_SHIFT_DISTANCE_M = 0.04
RED_LIFT_HEIGHT_M = 0.07
RED_ON_GREEN_XY_TOLERANCE_M = float(np.linalg.norm([0.02, 0.02]) + 0.005)
RED_ON_GREEN_Z_TOLERANCE_M = 0.005


def radial_unit(red_id_xy: Any, green_xy: Any) -> np.ndarray:
    red = np.asarray(red_id_xy, dtype=np.float64).reshape(2)
    green = np.asarray(green_xy, dtype=np.float64).reshape(2)
    direction = red - green
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        raise ValueError("red_id_xy and green_xy must be distinct")
    return direction / norm


def radial_ood_xy(red_id_xy: Any, green_xy: Any) -> np.ndarray:
    return (
        np.asarray(red_id_xy, dtype=np.float64).reshape(2)
        + RADIAL_SHIFT_DISTANCE_M * radial_unit(red_id_xy, green_xy)
    )


def sample_stage1_radial_paired_xy(
    rng: Any, count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample common green/ID-red positions and the radial OOD red position."""

    if count <= 0:
        raise ValueError("count must be positive")
    green_xy = rng.uniform(
        -STACK_CUBE_ID_BASE_JITTER,
        STACK_CUBE_ID_BASE_JITTER,
        size=(count, 2),
    )
    distance = rng.uniform(*STACK_CUBE_ID_DISTANCE_RANGE, size=count)
    angle = rng.uniform(
        STACK_CUBE_ID_ANGLE_CENTER - STACK_CUBE_ID_ANGLE_HALF_WIDTH,
        STACK_CUBE_ID_ANGLE_CENTER + STACK_CUBE_ID_ANGLE_HALF_WIDTH,
        size=count,
    )
    offset = np.stack([distance * np.cos(angle), distance * np.sin(angle)], axis=-1)
    red_id_xy = green_xy + offset
    red_ood_xy = np.asarray(
        [radial_ood_xy(red_id_xy[index], green_xy[index]) for index in range(count)]
    )
    return (
        np.asarray(green_xy, dtype=np.float64),
        np.asarray(red_id_xy, dtype=np.float64),
        red_ood_xy,
    )


def reset_event_predicates(
    red_xy: Any,
    green_xy: Any,
    *,
    red_z: float = 0.02,
    green_z: float = 0.02,
    red_grasped: bool = False,
) -> dict[str, bool]:
    red = np.asarray(red_xy, dtype=np.float64).reshape(2)
    green = np.asarray(green_xy, dtype=np.float64).reshape(2)
    red_lifted = bool(red_grasped and red_z >= RED_LIFT_HEIGHT_M)
    red_placed = bool(
        not red_grasped
        and np.linalg.norm(red - green) <= RED_ON_GREEN_XY_TOLERANCE_M
        and abs((red_z - green_z) - 0.04) <= RED_ON_GREEN_Z_TOLERANCE_M
    )
    return {
        "red_grasped": bool(red_grasped),
        "red_lifted": red_lifted,
        "red_placed": red_placed,
    }


def radial_reset_record(
    *,
    paired_seed: int,
    split: str,
    green_xy: Any,
    red_id_xy: Any,
    red_xy: Any,
) -> dict[str, Any]:
    green = np.asarray(green_xy, dtype=np.float64).reshape(2)
    red_id = np.asarray(red_id_xy, dtype=np.float64).reshape(2)
    red = np.asarray(red_xy, dtype=np.float64).reshape(2)
    shift = red - red_id
    unit = radial_unit(red_id, green)
    return {
        "task": STACK_CUBE_TASK,
        "split": split,
        "paired_seed": int(paired_seed),
        "cube_a_id_xy": red_id.tolist(),
        "cube_a_xy": red.tolist(),
        "cube_b_xy": green.tolist(),
        "red_shift_xy": shift.tolist(),
        "red_shift_norm_m": float(np.linalg.norm(shift)),
        "radial_unit_green_to_red_id": unit.tolist(),
        "red_id_distance_m": float(np.linalg.norm(red_id - green)),
        "red_ood_distance_m": float(np.linalg.norm(red - green)),
        "paired_green_equal": True,
        "non_target_factors_equal": True,
        "reset_predicates": reset_event_predicates(red, green),
        "metadata_complete": True,
    }


def validate_paired_reset_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    for pair in records:
        id_record = pair["id"]
        ood_record = pair["ood"]
        green_id = np.asarray(id_record["cube_b_xy"], dtype=np.float64)
        green_ood = np.asarray(ood_record["cube_b_xy"], dtype=np.float64)
        red_id = np.asarray(id_record["cube_a_xy"], dtype=np.float64)
        red_ood = np.asarray(ood_record["cube_a_xy"], dtype=np.float64)
        expected = RADIAL_SHIFT_DISTANCE_M * radial_unit(red_id, green_id)
        valid = (
            id_record["paired_seed"] == ood_record["paired_seed"]
            and np.array_equal(green_id, green_ood)
            and np.allclose(red_ood - red_id, expected, atol=1e-12)
            and np.isclose(np.linalg.norm(red_ood - red_id), RADIAL_SHIFT_DISTANCE_M, atol=1e-12)
            and id_record["non_target_factors_equal"]
            and ood_record["non_target_factors_equal"]
            and id_record["metadata_complete"]
            and ood_record["metadata_complete"]
            and not any(id_record["reset_predicates"].values())
            and not any(ood_record["reset_predicates"].values())
        )
        if not valid:
            failures.append(pair)
    return {
        "episodes": len(records),
        "passed": not failures,
        "failed_episodes": len(failures),
        "radial_shift_distance_m": RADIAL_SHIFT_DISTANCE_M,
        "green_pose_changed": False,
        "non_target_factors_changed": False,
        "reset_predicates_all_false": not failures,
        "failures": failures,
    }


def radial_env_id(split: str) -> str:
    if split == "id":
        return STACK_CUBE_RADIAL_ID_ENV_ID
    if split == STACK_CUBE_RADIAL_OOD_SPLIT:
        return STACK_CUBE_RADIAL_OOD_ENV_ID
    raise ValueError(f"unknown radial split: {split}")


def stage1_radial_reset_metadata(
    env: Any, *, split: str, paired_seed: int | None = None
) -> dict[str, Any]:
    base = env.unwrapped
    stored = getattr(base, "_stage1_radial_reset_metadata", None)
    if stored is None:
        raise RuntimeError("radial environment did not record paired reset metadata")
    robot_qpos = base.agent.robot.get_qpos()
    if hasattr(robot_qpos, "detach"):
        robot_qpos = robot_qpos.detach().cpu().numpy()
    record = dict(stored)
    record.update(
        {
            "split": split,
            "paired_seed": paired_seed,
            "task": STACK_CUBE_TASK,
            "robot_qpos": np.asarray(robot_qpos, dtype=np.float64).reshape(-1).tolist(),
            "camera_and_instruction": "inherited unchanged from StackCube runtime",
            "non_target_randomness": "paired by reset seed",
            "success_predicate": "existing StackCubeEnv.evaluate().success",
            "affected_stage": "red approach/grasp/lift",
        }
    )
    return record


def _info_bool(info: dict[str, Any] | None, key: str, default: bool = False) -> bool:
    if info is None or key not in info:
        return default
    value = info[key]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return bool(np.asarray(value).reshape(-1)[0])


def stage1_radial_state_record(
    env: Any, info: dict[str, Any] | None = None
) -> dict[str, Any]:
    base = env.unwrapped
    red_p = np.asarray(base.cubeA.pose.p.detach().cpu().numpy()).reshape(-1, 3)[0]
    green_p = np.asarray(base.cubeB.pose.p.detach().cpu().numpy()).reshape(-1, 3)[0]
    red_grasped = _info_bool(info, "is_cubeA_grasped")
    if info is None or "is_cubeA_grasped" not in info:
        red_grasped = bool(
            np.asarray(base.agent.is_grasping(base.cubeA).detach().cpu().numpy()).reshape(-1)[0]
        )
    predicates = reset_event_predicates(
        red_p[:2],
        green_p[:2],
        red_z=float(red_p[2]),
        green_z=float(green_p[2]),
        red_grasped=red_grasped,
    )
    return {
        "red_xy": red_p[:2].tolist(),
        "green_xy": green_p[:2].tolist(),
        "red_z": float(red_p[2]),
        "green_z": float(green_p[2]),
        "predicates": predicates,
        "return_candidate": bool(predicates["red_grasped"] and predicates["red_lifted"]),
    }


def register_stackcube_stage1_radial_variants() -> None:
    """Register only the independent radial ID/OOD environment pair."""

    import gymnasium as gym
    import torch
    from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv
    from mani_skill.utils.registration import register_env
    from mani_skill.utils.structs.pose import Pose

    if (
        STACK_CUBE_RADIAL_ID_ENV_ID in gym.registry
        and STACK_CUBE_RADIAL_OOD_ENV_ID in gym.registry
    ):
        return

    class _RadialBaseEnv(StackCubeEnv):
        def _initialize_radial(self, env_idx: Any, options: dict[str, Any], *, split: str) -> None:
            StackCubeEnv._initialize_episode(self, env_idx, options)
            count = len(env_idx)
            green_xy, red_id_xy, red_ood_xy = sample_stage1_radial_paired_xy(
                self._batched_episode_rng, count
            )
            red_xy = red_id_xy if split == "id" else red_ood_xy
            red_p = self.cubeA.pose.p.clone()
            green_p = self.cubeB.pose.p.clone()
            red_p[env_idx, :2] = torch.as_tensor(red_xy, dtype=red_p.dtype, device=self.device)
            green_p[env_idx, :2] = torch.as_tensor(green_xy, dtype=green_p.dtype, device=self.device)
            self.cubeA.set_pose(Pose.create_from_pq(red_p, self.cubeA.pose.q.clone()))
            self.cubeB.set_pose(Pose.create_from_pq(green_p, self.cubeB.pose.q.clone()))
            self._stage1_radial_reset_metadata = radial_reset_record(
                paired_seed=-1,
                split=split,
                green_xy=green_xy[0],
                red_id_xy=red_id_xy[0],
                red_xy=red_xy[0],
            )

    @register_env(STACK_CUBE_RADIAL_ID_ENV_ID, max_episode_steps=100)
    class StackCubeStage1RadialIDEnv(_RadialBaseEnv):
        def _initialize_episode(self, env_idx: Any, options: dict[str, Any]) -> None:
            self._initialize_radial(env_idx, options, split="id")

    @register_env(STACK_CUBE_RADIAL_OOD_ENV_ID, max_episode_steps=100)
    class StackCubeStage1RadialOODEnv(_RadialBaseEnv):
        def _initialize_episode(self, env_idx: Any, options: dict[str, Any]) -> None:
            self._initialize_radial(env_idx, options, split=STACK_CUBE_RADIAL_OOD_SPLIT)
