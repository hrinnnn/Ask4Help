from tools.collect_stackcube_xvla_dagger import (
    admitted_suffix,
    alternating_split,
    consecutive_gate,
)
from tools.run_xvla_stackcube_four_group_pipeline import diffdagger_gate_threshold
from tools.run_xvla_stackcube_four_group_training import select_idle_gpus


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
