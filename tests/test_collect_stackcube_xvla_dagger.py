from tools.collect_stackcube_xvla_dagger import (
    FailureRecoveryState,
    admitted_suffix,
    alternating_split,
    collection_complete,
    consecutive_gate,
    exact_budget_subset,
)
from tools.run_xvla_stackcube_four_group_pipeline import diffdagger_gate_threshold
from tools.run_xvla_stackcube_four_group_training import select_idle_gpus


def test_raw_attempts_alternate_id_ood() -> None:
    assert [alternating_split(index) for index in range(6)] == [
        "id", "ood", "id", "ood", "id", "ood"
    ]


def test_raw_attempts_can_alternate_stage2_ood() -> None:
    assert [alternating_split(index, "stage2_ood") for index in range(4)] == [
        "id", "stage2_ood", "id", "stage2_ood"
    ]


def test_action_budget_takes_precedence_over_episode_target() -> None:
    assert not collection_complete(
        100, 2499, target_episodes=100, expert_action_budget=2500
    )
    assert collection_complete(
        91, 2500, target_episodes=100, expert_action_budget=2500
    )


def test_failure_recovery_event_latches_after_drop() -> None:
    state = FailureRecoveryState()
    state.update(
        env_step=10,
        currently_grasped=True,
        on_cube=False,
        success=False,
        cube_z=0.08,
    )
    assert state.reason is None
    state.update(
        env_step=15,
        currently_grasped=False,
        on_cube=False,
        success=False,
        cube_z=0.03,
    )
    assert state.reason == "dropped_after_grasp"


def test_controlled_timing_requires_lift_and_two_drop_boundaries() -> None:
    state = FailureRecoveryState()
    state.update(env_step=10, currently_grasped=True, on_cube=False, success=False, cube_z=0.08)
    assert state.timing_trigger("post_grasp") == "post_grasp"
    assert state.timing_trigger("post_lift") == "post_lift"
    state.update(env_step=15, currently_grasped=False, on_cube=False, success=False, cube_z=0.03)
    assert state.timing_trigger("failure_recovery") is None
    assert state.timing_trigger("failure_recovery") == "dropped_after_lift_two_boundaries"


def test_exact_budget_subset_never_slices_an_episode() -> None:
    assert exact_budget_subset([17, 23, 31, 40], 80) == [0, 1, 3]
    assert exact_budget_subset([17, 23], 39) is None


def test_only_successful_nonempty_expert_suffix_is_admitted() -> None:
    assert admitted_suffix(True, 50, 73) == (50, 73)
    assert admitted_suffix(False, 50, 73) is None
    assert admitted_suffix(True, None, 73) is None
    assert admitted_suffix(True, 73, 73) is None


def test_consecutive_gate_resets_below_threshold() -> None:
    count, alarm = consecutive_gate(2.0, 1.0, 0, 2)
    assert (count, alarm) == (1, False)
    count, alarm = consecutive_gate(0.5, 1.0, count, 2)
    assert (count, alarm) == (0, False)
    count, alarm = consecutive_gate(2.0, 1.0, count, 2)
    count, alarm = consecutive_gate(2.0, 1.0, count, 2)
    assert (count, alarm) == (2, True)


def test_diffdagger_calibration_matches_patience_statistic() -> None:
    summary = {
        "rows": [
            {"timeline": [{"scores": {"diffdagger": value}} for value in scores]}
            for scores in ([1.0, 4.0, 3.0], [2.0, 8.0, 5.0], [3.0, 7.0, 6.0])
        ]
    }
    threshold, maxima = diffdagger_gate_threshold(summary, q=0.5, patience=2)
    assert maxima == [3.0, 5.0, 6.0]
    assert threshold == 5.0


def test_idle_gpu_selection_uses_reported_memory(monkeypatch) -> None:
    monkeypatch.setattr(
        "tools.run_xvla_stackcube_four_group_training.subprocess.check_output",
        lambda *args, **kwargs: "0, 30000\n1, 18\n2, 19\n3, 500\n4, 18\n",
    )
    assert select_idle_gpus(4) == [1, 2, 3, 4]
