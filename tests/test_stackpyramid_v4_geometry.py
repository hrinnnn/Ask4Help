from __future__ import annotations

import numpy as np

from tools.stackpyramid_task import stackpyramid_id_centers, stackpyramid_ood_shifts


def test_v4_reset_and_shift_geometry_stays_outside_next_to_tolerance(monkeypatch) -> None:
    monkeypatch.setenv("STACKPYRAMID_OOD_GEOMETRY", "v4")
    centers = stackpyramid_id_centers()
    shifts = stackpyramid_ood_shifts()
    jitter_bound = 0.008
    next_to_threshold = 0.0616

    for split, target in (("id", None), ("stage1_ood", 0), ("stage2_ood", 1), ("stage3_ood", 2)):
        split_centers = centers.copy()
        if target is not None:
            split_centers[target] += shifts[split]
        for first in range(3):
            for second in range(first + 1, 3):
                delta = np.abs(split_centers[first] - split_centers[second])
                worst_case_distance = np.linalg.norm(np.maximum(delta - 2 * jitter_bound, 0.0))
                assert worst_case_distance > next_to_threshold


def test_v4_shifts_are_single_object_and_red_green_are_separated(monkeypatch) -> None:
    monkeypatch.setenv("STACKPYRAMID_OOD_GEOMETRY", "v4")
    centers = stackpyramid_id_centers()
    assert np.isclose(np.linalg.norm(centers[0] - centers[1]), 0.16)
    assert set(stackpyramid_ood_shifts()) == {"stage1_ood", "stage2_ood", "stage3_ood"}
