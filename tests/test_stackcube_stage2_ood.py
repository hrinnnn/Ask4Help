import numpy as np

from tools.stackcube_stage2_ood import sample_stack_cube_stage2_xy
from rlinf.envs.maniskill.stack_cube_variants import (
    STACK_CUBE_ID_DISTANCE_RANGE,
    sample_stack_cube_xy,
)


def test_stage2_preserves_red_pose_for_paired_id_seed() -> None:
    stage2_green, stage2_red, paired_id_green = sample_stack_cube_stage2_xy(
        np.random.default_rng(41), 256
    )
    id_green, id_red = sample_stack_cube_xy(
        np.random.default_rng(41), 256, split="id"
    )
    np.testing.assert_allclose(stage2_red, id_red)
    np.testing.assert_allclose(paired_id_green, id_green)
    np.testing.assert_allclose(stage2_green, 2.0 * id_red - id_green)


def test_stage2_changes_only_target_and_keeps_pair_distance() -> None:
    stage2_green, red, id_green = sample_stack_cube_stage2_xy(
        np.random.default_rng(7), 512
    )
    id_distance = np.linalg.norm(red - id_green, axis=-1)
    stage2_distance = np.linalg.norm(red - stage2_green, axis=-1)
    np.testing.assert_allclose(stage2_distance, id_distance)
    assert stage2_distance.min() >= STACK_CUBE_ID_DISTANCE_RANGE[0]
    assert stage2_distance.max() <= STACK_CUBE_ID_DISTANCE_RANGE[1]
    assert np.allclose((red - id_green) + (red - stage2_green), 0.0)
