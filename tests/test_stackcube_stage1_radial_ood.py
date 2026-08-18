from pathlib import Path

import numpy as np

from tools.stackcube_stage1_radial_ood import (
    RADIAL_SHIFT_DISTANCE_M,
    radial_ood_xy,
    radial_unit,
    sample_stage1_radial_paired_xy,
    validate_paired_reset_records,
)


def test_radial_shift_has_fixed_norm_and_direction() -> None:
    green, red_id, red_ood = sample_stage1_radial_paired_xy(
        np.random.default_rng(971000), 20
    )
    for green_xy, red_id_xy, red_ood_xy in zip(green, red_id, red_ood):
        shift = red_ood_xy - red_id_xy
        expected = RADIAL_SHIFT_DISTANCE_M * radial_unit(red_id_xy, green_xy)
        assert np.allclose(shift, expected)
        assert np.isclose(np.linalg.norm(shift), RADIAL_SHIFT_DISTANCE_M)


def test_paired_reset_validation_rejects_non_radial_change() -> None:
    rng = np.random.default_rng(971001)
    green, red_id, red_ood = sample_stage1_radial_paired_xy(rng, 1)
    pair = {
        "id": {
            "paired_seed": 1,
            "cube_a_xy": red_id[0].tolist(),
            "cube_b_xy": green[0].tolist(),
            "non_target_factors_equal": True,
            "metadata_complete": True,
            "reset_predicates": {"red_grasped": False, "red_lifted": False, "red_placed": False},
        },
        "ood": {
            "paired_seed": 1,
            "cube_a_xy": (red_ood[0] + np.asarray([0.001, 0.0])).tolist(),
            "cube_b_xy": green[0].tolist(),
            "non_target_factors_equal": True,
            "metadata_complete": True,
            "reset_predicates": {"red_grasped": False, "red_lifted": False, "red_placed": False},
        },
    }
    assert not validate_paired_reset_records([pair])["passed"]


def test_independent_radial_assets_exist() -> None:
    assert Path("configs/stackcube_stage1_radial_two_way_task_spec.json").exists()
    assert Path("configs/stackcube_stage1_radial_two_way_seed_manifest.json").exists()
