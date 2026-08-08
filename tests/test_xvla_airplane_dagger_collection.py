from tools.collect_pick_single_ycb_airplane_xvla_dagger import alternating_split, consecutive_gate


def test_consecutive_gate_resets_and_triggers_at_patience() -> None:
    count, alarm = consecutive_gate(2.0, 1.0, 0, 2)
    assert (count, alarm) == (1, False)
    count, alarm = consecutive_gate(0.5, 1.0, count, 2)
    assert (count, alarm) == (0, False)
    count, alarm = consecutive_gate(2.0, 1.0, count, 2)
    count, alarm = consecutive_gate(3.0, 1.0, count, 2)
    assert (count, alarm) == (2, True)


def test_raw_attempt_splits_are_strictly_alternating() -> None:
    assert [alternating_split(index) for index in range(6)] == [
        "id", "ood", "id", "ood", "id", "ood"
    ]
