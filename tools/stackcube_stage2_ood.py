"""Stage-2 StackCube OOD with an ID-matched red cube and shifted green target."""

from __future__ import annotations

from typing import Any

import numpy as np

from rlinf.envs.maniskill.stack_cube_variants import (
    STACK_CUBE_ID_ENV_ID,
    STACK_CUBE_OOD_ENV_ID,
    register_controlled_stack_cube_variants,
    reset_metadata,
    sample_stack_cube_xy,
)


STACK_CUBE_STAGE2_OOD_ENV_ID = "RLinfStackCubeStage2OOD-v1"
STACK_CUBE_STAGE2_OOD_SPLIT = "stage2_ood"
STACK_CUBE_SPLITS = ("id", "ood", STACK_CUBE_STAGE2_OOD_SPLIT)

_REGISTERED = False


def sample_stack_cube_stage2_xy(
    rng: Any, count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample shifted green targets while preserving the paired ID red poses.

    Returns the Stage-2 green positions, red positions, and counterfactual ID
    green positions. The target is reflected across the red cube, preserving
    the ID object-target distance while moving only the placement target.
    """

    id_green_xy, red_xy = sample_stack_cube_xy(rng, count, split="id")
    stage2_green_xy = 2.0 * red_xy - id_green_xy
    return stage2_green_xy, red_xy, id_green_xy


def stack_cube_env_id(split: str) -> str:
    if split == "id":
        return STACK_CUBE_ID_ENV_ID
    if split == "ood":
        return STACK_CUBE_OOD_ENV_ID
    if split == STACK_CUBE_STAGE2_OOD_SPLIT:
        return STACK_CUBE_STAGE2_OOD_ENV_ID
    raise ValueError(f"unknown StackCube split: {split}")


def stack_cube_reset_metadata(env: Any, *, split: str) -> dict[str, Any]:
    if split != STACK_CUBE_STAGE2_OOD_SPLIT:
        return reset_metadata(env, split=split)
    metadata = reset_metadata(env, split="ood")
    red_xy = np.asarray(metadata["cube_a_pose"]["p"][:2], dtype=np.float64)
    stage2_green_xy = np.asarray(
        metadata["cube_b_pose"]["p"][:2], dtype=np.float64
    )
    paired_id_green_xy = 2.0 * red_xy - stage2_green_xy
    metadata.update(
        {
            "split": STACK_CUBE_STAGE2_OOD_SPLIT,
            "distribution_factor": "green_target_position_only",
            "affected_stage": "transport_and_placement",
            "paired_id_cube_b_xy": paired_id_green_xy.tolist(),
            "target_shift_distance": float(
                np.linalg.norm(stage2_green_xy - paired_id_green_xy)
            ),
        }
    )
    return metadata


def register_stack_cube_splits() -> None:
    """Register the historical splits and the Stage-2 target-only split."""

    global _REGISTERED
    register_controlled_stack_cube_variants()
    if _REGISTERED:
        return

    import gymnasium as gym

    if STACK_CUBE_STAGE2_OOD_ENV_ID in gym.registry:
        _REGISTERED = True
        return

    import torch
    from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv
    from mani_skill.utils.registration import register_env
    from mani_skill.utils.structs.pose import Pose

    @register_env(STACK_CUBE_STAGE2_OOD_ENV_ID, max_episode_steps=100)
    class ControlledStackCubeStage2OODEnv(StackCubeEnv):
        def _initialize_episode(self, env_idx, options: dict) -> None:
            StackCubeEnv._initialize_episode(self, env_idx, options)
            green_xy, red_xy, _ = sample_stack_cube_stage2_xy(
                self._batched_episode_rng, len(env_idx)
            )
            red_p = self.cubeA.pose.p.clone()
            green_p = self.cubeB.pose.p.clone()
            red_p[env_idx, :2] = torch.as_tensor(
                red_xy, dtype=red_p.dtype, device=self.device
            )
            green_p[env_idx, :2] = torch.as_tensor(
                green_xy, dtype=green_p.dtype, device=self.device
            )
            self.cubeA.set_pose(Pose.create_from_pq(red_p, self.cubeA.pose.q.clone()))
            self.cubeB.set_pose(
                Pose.create_from_pq(green_p, self.cubeB.pose.q.clone())
            )

    _REGISTERED = True
