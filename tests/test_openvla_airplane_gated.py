from openvla_airplane.gated import (
    admitted_expert_suffix,
    alternating_split,
    calibrate_gate,
    update_patience_gate,
)


def test_raw_attempts_strictly_alternate_id_and_ood():
    assert [alternating_split(index) for index in range(6)] == ["id", "ood", "id", "ood", "id", "ood"]


def test_patience_gate_requires_consecutive_exceedances():
    count, alarm = update_patience_gate(2.0, 1.0, 0, 2)
    assert (count, alarm) == (1, False)
    count, alarm = update_patience_gate(0.5, 1.0, count, 2)
    assert (count, alarm) == (0, False)
    count, _ = update_patience_gate(2.0, 1.0, count, 2)
    count, alarm = update_patience_gate(3.0, 1.0, count, 2)
    assert (count, alarm) == (2, True)


def test_calibration_matches_complete_temporal_gate():
    result = calibrate_gate([[1.0, 3.0, 2.0], [4.0, 2.0, 5.0]], quantile=1.0, patience=2)
    assert result["episode_scores"] == [2.0, 2.0]
    assert result["threshold"] == 2.0


def test_only_successful_nonempty_expert_suffix_is_admitted():
    assert admitted_expert_suffix(True, 7, 10) == (7, 10)
    assert admitted_expert_suffix(False, 7, 10) is None
    assert admitted_expert_suffix(True, None, 10) is None
    assert admitted_expert_suffix(True, 10, 10) is None
