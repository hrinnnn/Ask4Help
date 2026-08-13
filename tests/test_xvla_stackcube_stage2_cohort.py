from tools.build_stackcube_xvla_stage2_cohort import select_cohort


def test_cohort_requires_grasp_lift_drop_and_failure() -> None:
    rows = [
        {"seed": 1, "success": False, "grasped_once": True, "lifted_once": True,
         "stable_lift_boundary_once": True, "dropped_after_lift_two_boundaries": True},
        {"seed": 2, "success": True, "grasped_once": True, "lifted_once": True,
         "stable_lift_boundary_once": True, "dropped_after_lift_two_boundaries": True},
        {"seed": 3, "success": False, "grasped_once": True, "lifted_once": False,
         "stable_lift_boundary_once": False, "dropped_after_lift_two_boundaries": True},
    ]
    assert [row["seed"] for row in select_cohort(rows, 5)] == [1]
