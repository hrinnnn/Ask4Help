from tools.collect_stackcube_xvla_dagger import exact_budget_subset
from tools.run_xvla_stackcube_stage2_pipeline import METHODS, choose_common_checkpoint


def test_stage2_pipeline_compares_the_four_formal_groups() -> None:
    assert METHODS == (
        "immediate",
        "post_grasp",
        "post_lift",
        "failure_recovery",
    )


def test_common_checkpoint_uses_earliest_clear_non_saturated_separation() -> None:
    rows = []
    for step, rates in ((2000, [0.20, 0.24, 0.25, 0.21]),
                        (4000, [0.20, 0.38, 0.45, 0.22]),
                        (6000, [0.25, 0.50, 0.52, 0.30])):
        rows.extend(
            {"step": step, "method": method, "success_rate": rate}
            for method, rate in zip(METHODS, rates)
        )
    selected = choose_common_checkpoint(rows, steps=[2000, 4000, 6000])
    assert selected["step"] == 4000


def test_common_checkpoint_falls_back_to_final_prespecified_step() -> None:
    rows = [
        {"step": step, "method": method, "success_rate": 0.2 + index * 0.01}
        for step in (2000, 4000)
        for index, method in enumerate(METHODS)
    ]
    selected = choose_common_checkpoint(rows, steps=[2000, 4000])
    assert selected["step"] == 4000
    assert selected["reason"] == "maximum_prespecified_step_no_earlier_common_stop"


def test_2002_is_exactly_reachable_by_the_observed_full_suffix_lengths() -> None:
    assert exact_budget_subset([11] * 182, 2002) is not None
    assert exact_budget_subset([26] * 77, 2002) is not None
    assert exact_budget_subset([11] * 200, 2000) is None


def test_clean_cohort_requires_a_stage_localized_failure() -> None:
    rows = [
        {
            "success": False,
            "grasped_once": True,
            "lifted_once": True,
            "stable_lift_boundary_once": True,
            "dropped_after_lift_two_boundaries": True,
        },
        {
            "success": False,
            "grasped_once": True,
            "lifted_once": False,
            "stable_lift_boundary_once": False,
            "dropped_after_lift_two_boundaries": False,
        },
    ]
    candidates = [
        row
        for row in rows
        if (
            not row["success"]
            and row["grasped_once"]
            and row["lifted_once"]
            and row["stable_lift_boundary_once"]
            and row["dropped_after_lift_two_boundaries"]
        )
    ]
    assert len(candidates) == 1


def test_cpu_partition_is_disjoint_and_balanced() -> None:
    available = list(range(8))
    partitions = [available[index::2] for index in range(2)]
    assert set(partitions[0]).isdisjoint(partitions[1])
    assert sorted(partitions[0] + partitions[1]) == available
    assert len(partitions[0]) == len(partitions[1])
