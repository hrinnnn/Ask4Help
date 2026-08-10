from tools.collect_stackcube_xvla_dagger import (
    admitted_suffix,
    alternating_split,
    consecutive_gate,
)


def test_raw_attempts_alternate_id_ood() -> None:
    assert [alternating_split(index) for index in range(6)] == [
        "id", "ood", "id", "ood", "id", "ood"
    ]


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
