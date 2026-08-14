"""Controlled StackPyramid distributions for stage-localized OOD experiments."""

from __future__ import annotations

from typing import Any

import numpy as np


STACKPYRAMID_ID_ENV_ID = "Ask4HelpStackPyramidID-v1"
STACKPYRAMID_STAGE1_ENV_ID = "Ask4HelpStackPyramidStage1OOD-v1"
STACKPYRAMID_STAGE2_ENV_ID = "Ask4HelpStackPyramidStage2OOD-v1"
STACKPYRAMID_STAGE3_ENV_ID = "Ask4HelpStackPyramidStage3OOD-v1"
STACKPYRAMID_SPLITS = ("id", "stage1_ood", "stage2_ood", "stage3_ood")
STACKPYRAMID_TASK = "stack the red cube next to the green cube and place the blue cube on top"

# Positions are deliberately narrow and separated. OOD offsets remain on the table.
_ID_CENTERS = np.asarray(
    [[-0.04, -0.04], [0.04, -0.04], [0.00, 0.04]], dtype=np.float64
)
_OOD_SHIFTS = {
    "stage1_ood": np.asarray([0.08, 0.10], dtype=np.float64),
    "stage2_ood": np.asarray([0.12, 0.10], dtype=np.float64),
    "stage3_ood": np.asarray([0.12, -0.16], dtype=np.float64),
}
_REGISTERED = False


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
    centers = np.broadcast_to(_ID_CENTERS, (count, 3, 2)).copy()
    if split != "id":
        target = {"stage1_ood": 0, "stage2_ood": 1, "stage3_ood": 2}[split]
        centers[:, target] += _OOD_SHIFTS[split]
    return centers + jitter


def stackpyramid_env_id(split: str) -> str:
    return {
        "id": STACKPYRAMID_ID_ENV_ID,
        "stage1_ood": STACKPYRAMID_STAGE1_ENV_ID,
        "stage2_ood": STACKPYRAMID_STAGE2_ENV_ID,
        "stage3_ood": STACKPYRAMID_STAGE3_ENV_ID,
    }[split]


def reset_metadata(env: Any, *, split: str) -> dict[str, Any]:
    base = env.unwrapped

    def array(value: Any) -> list[float]:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float64).reshape(-1).tolist()

    cubes = (base.cubeA, base.cubeB, base.cubeC)
    positions = [array(cube.pose.p) for cube in cubes]
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
        "max_episode_steps": 250,
    }


def _set_xy(env: Any, env_idx: Any, xy: np.ndarray) -> None:
    import torch
    from mani_skill.utils.structs.pose import Pose

    base = env.unwrapped
    cubes = (base.cubeA, base.cubeB, base.cubeC)
    selected = cubes[0].pose.p[env_idx]
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
        target = position[env_idx, :2]
        values = torch.as_tensor(
            xy[:, cube_index], dtype=position.dtype, device=base.device
        )
        if target.shape != values.shape:
            raise ValueError(
                f"StackPyramid pose assignment mismatch: env_idx={env_idx!r}, "
                f"position={tuple(position.shape)}, target={tuple(target.shape)}, "
                f"xy={tuple(xy.shape)}, values={tuple(values.shape)}"
            )
        position[env_idx, :2] = values
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
