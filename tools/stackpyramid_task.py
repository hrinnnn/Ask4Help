"""Controlled StackPyramid distributions for stage-localized OOD experiments."""

from __future__ import annotations

import os
from typing import Any

import numpy as np


STACKPYRAMID_ID_ENV_ID = "Ask4HelpStackPyramidID-v1"
STACKPYRAMID_STAGE1_ENV_ID = "Ask4HelpStackPyramidStage1OOD-v1"
STACKPYRAMID_STAGE2_ENV_ID = "Ask4HelpStackPyramidStage2OOD-v1"
STACKPYRAMID_STAGE3_ENV_ID = "Ask4HelpStackPyramidStage3OOD-v1"
STACKPYRAMID_SPLITS = ("id", "stage1_ood", "stage2_ood", "stage3_ood")
STACKPYRAMID_TASK = "stack the red cube next to the green cube and place the blue cube on top"
STACKPYRAMID_RESET_INVARIANTS = ("red_grasped", "red_lifted", "red_placed", "blue_lifted")

# Positions are deliberately narrow and separated. OOD offsets remain on the table.
_ID_CENTERS = np.asarray(
    [[-0.02, -0.04], [0.02, -0.04], [0.00, 0.04]], dtype=np.float64
)
_OOD_SHIFTS_V1 = {
    "stage1_ood": np.asarray([0.08, 0.10], dtype=np.float64),
    "stage2_ood": np.asarray([0.12, 0.10], dtype=np.float64),
    "stage3_ood": np.asarray([0.12, -0.16], dtype=np.float64),
}
# V2 was a diagnostic geometry. Its stage-3 offset placed the blue cube too
# close to the red/green pair and failed the Oracle feasibility gate. V3 keeps
# the same affected object and stage semantics while preserving physical
# clearance for the blue-cube grasp.
_OOD_SHIFTS_V2 = {
    "stage1_ood": np.asarray([0.045, 0.045], dtype=np.float64),
    "stage2_ood": np.asarray([0.060, 0.050], dtype=np.float64),
    "stage3_ood": np.asarray([0.060, -0.080], dtype=np.float64),
}
_OOD_SHIFTS_V3 = {
    "stage1_ood": np.asarray([0.045, 0.045], dtype=np.float64),
    "stage2_ood": np.asarray([0.060, 0.050], dtype=np.float64),
    "stage3_ood": np.asarray([0.100, -0.120], dtype=np.float64),
}

# V1-V3 are diagnostic. V4 separates the base cubes so that every intended
# stage predicate is false at reset and remains physically valid after the
# designated single-object shift, including the full reset-jitter envelope.
_ID_CENTERS_V4 = np.asarray(
    [[-0.080, -0.080], [0.080, -0.080], [0.000, 0.100]], dtype=np.float64
)
_OOD_SHIFTS_V4 = {
    "stage1_ood": np.asarray([0.120, 0.100], dtype=np.float64),
    "stage2_ood": np.asarray([0.120, 0.100], dtype=np.float64),
    # Keep the blue target inside the known-solvable tabletop corridor while
    # preserving a distinct blue-only distribution shift.
    "stage3_ood": np.asarray([0.100, -0.120], dtype=np.float64),
}
_REGISTERED = False


def stackpyramid_geometry_version() -> str:
    return os.environ.get("STACKPYRAMID_OOD_GEOMETRY", "v1")


def stackpyramid_ood_shifts() -> dict[str, np.ndarray]:
    version = stackpyramid_geometry_version()
    if version == "v1":
        return _OOD_SHIFTS_V1
    if version == "v2":
        return _OOD_SHIFTS_V2
    if version == "v3":
        return _OOD_SHIFTS_V3
    if version == "v4":
        return _OOD_SHIFTS_V4
    raise ValueError(f"unknown STACKPYRAMID_OOD_GEOMETRY={version!r}")


def stackpyramid_id_centers() -> np.ndarray:
    """Return the frozen ID centers for the selected benchmark version."""
    if stackpyramid_geometry_version() == "v4":
        return _ID_CENTERS_V4
    return _ID_CENTERS


def sample_stackpyramid_xy(rng: Any, count: int, *, split: str) -> np.ndarray:
    if count <= 0:
        raise ValueError("count must be positive")
    if split not in STACKPYRAMID_SPLITS:
        raise ValueError(f"unknown StackPyramid split: {split}")
    # ManiSkill's batched RNG supplies the leading environment dimension.
    jitter = np.asarray(rng.uniform(-0.008, 0.008, size=(3, 2)))
    if jitter.ndim == 2:
        jitter = jitter[None, ...]
    if jitter.shape[0] != count:
        raise ValueError(f"unexpected StackPyramid jitter shape {jitter.shape} for {count} environments")
    centers = np.broadcast_to(stackpyramid_id_centers(), (count, 3, 2)).copy()
    if split != "id":
        target = {"stage1_ood": 0, "stage2_ood": 1, "stage3_ood": 2}[split]
        centers[:, target] += stackpyramid_ood_shifts()[split]
    return centers + jitter


def stackpyramid_env_id(split: str) -> str:
    env_ids = {
        "id": STACKPYRAMID_ID_ENV_ID,
        "stage1_ood": STACKPYRAMID_STAGE1_ENV_ID,
        "stage2_ood": STACKPYRAMID_STAGE2_ENV_ID,
        "stage3_ood": STACKPYRAMID_STAGE3_ENV_ID,
    }
    if stackpyramid_geometry_version() == "v4":
        return env_ids[split].replace("-v1", "-v4")
    return env_ids[split]


def _scalar_bool(value: Any) -> bool:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return bool(np.asarray(value).reshape(-1)[0])


def stackpyramid_reset_invariants(env: Any) -> dict[str, bool]:
    """Check that reset starts before every controlled task stage."""
    base = env.unwrapped
    red = np.asarray(base.cubeA.pose.p.detach().cpu().numpy()).reshape(-1, 3)[0]
    green = np.asarray(base.cubeB.pose.p.detach().cpu().numpy()).reshape(-1, 3)[0]
    blue = np.asarray(base.cubeC.pose.p.detach().cpu().numpy()).reshape(-1, 3)[0]
    threshold = float(np.linalg.norm(2 * base.cube_half_size[:2].detach().cpu().numpy()) + 0.005)
    resting_z = float(base.cube_half_size[2].detach().cpu().numpy())
    red_grasped = _scalar_bool(base.agent.is_grasping(base.cubeA))
    return {
        "red_grasped": red_grasped,
        "red_lifted": float(red[2]) > resting_z + 0.015,
        "red_placed": (
            float(np.linalg.norm((red - green)[:2])) <= threshold
            and not red_grasped
            and float(red[2]) <= resting_z + 0.03
        ),
        "blue_lifted": float(blue[2]) > resting_z + 0.015,
    }


def reset_metadata(env: Any, *, split: str) -> dict[str, Any]:
    base = env.unwrapped

    def array(value: Any) -> list[float]:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float64).reshape(-1).tolist()

    cubes = (base.cubeA, base.cubeB, base.cubeC)
    positions = [array(cube.pose.p) for cube in cubes]
    reset_invariants = stackpyramid_reset_invariants(env)
    if stackpyramid_geometry_version() == "v4" and any(reset_invariants.values()):
        raise RuntimeError(f"StackPyramid v4 reset invariant failed: {reset_invariants}")
    return {
        "split": split,
        "task": STACKPYRAMID_TASK,
        "objects": {"red": "cubeA", "green": "cubeB", "blue": "cubeC"},
        "cube_poses": {
            color: {"p": positions[index], "q": array(cubes[index].pose.q)}
            for index, color in enumerate(("red", "green", "blue"))
        },
        "robot_qpos": array(base.agent.robot.get_qpos()),
        "affected_object": {"id": None, "stage1_ood": "red", "stage2_ood": "green", "stage3_ood": "blue"}[split],
        "ood_geometry": stackpyramid_geometry_version(),
        "reset_invariants": reset_invariants,
        "reset_invariant_pass": not any(reset_invariants.values()),
        "max_episode_steps": 250,
    }


def _set_xy(env: Any, env_idx: Any, xy: np.ndarray) -> None:
    import torch
    from mani_skill.utils.structs.pose import Pose

    base = env.unwrapped
    cubes = (base.cubeA, base.cubeB, base.cubeC)
    position = cubes[0].pose.p
    if isinstance(env_idx, torch.Tensor):
        pose_index = env_idx.to(device=position.device)
    else:
        pose_index = torch.as_tensor(env_idx, device=position.device)
    selected = position[pose_index]
    selected_count = 1 if selected.ndim == 1 else int(selected.shape[0])
    xy = np.asarray(xy)
    if xy.ndim == 2:
        xy = xy[None, ...]
    if xy.shape[0] != selected_count:
        if xy.shape[1] == selected_count and xy.shape[0] == 3:
            xy = np.transpose(xy, (1, 0, 2))
        else:
            raise ValueError(
                f"unexpected StackPyramid xy shape {xy.shape} for {selected_count} environments"
            )
    for cube_index, cube in enumerate(cubes):
        position = cube.pose.p.clone()
        target = position[pose_index, :2]
        values = torch.as_tensor(
            xy[:, cube_index], dtype=position.dtype, device=base.device
        )
        if target.shape != values.shape:
            raise ValueError(
                f"StackPyramid pose assignment mismatch: env_idx={env_idx!r}, "
                f"position={tuple(position.shape)}, target={tuple(target.shape)}, "
                f"xy={tuple(xy.shape)}, values={tuple(values.shape)}"
            )
        position[pose_index, :2] = values
        quaternion = torch.zeros_like(cube.pose.q)
        quaternion[..., 0] = 1.0
        cube.set_pose(Pose.create_from_pq(position, quaternion))


def register_stackpyramid_splits() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    from mani_skill.envs.tasks.tabletop.stack_pyramid import StackPyramidEnv
    from mani_skill.utils.registration import register_env

    if not all(stackpyramid_env_id(split) in gym.registry for split in STACKPYRAMID_SPLITS):
        for split in STACKPYRAMID_SPLITS:
            env_id = stackpyramid_env_id(split)

            def make_initialize(split_name: str):
                def initialize(env, env_idx, options):
                    StackPyramidEnv._initialize_episode(env, env_idx, options)
                    xy = sample_stackpyramid_xy(
                        env._batched_episode_rng, len(env_idx), split=split_name
                    )
                    _set_xy(env, env_idx, xy)

                return initialize

            initialize_fn = make_initialize(split)

            if env_id in gym.registry:
                continue

            @register_env(env_id, max_episode_steps=250)
            class ControlledStackPyramidEnv(StackPyramidEnv):
                def _initialize_episode(
                    self, env_idx, options: dict, _initialize=initialize_fn
                ) -> None:
                    _initialize(self, env_idx, options)

    _REGISTERED = True
