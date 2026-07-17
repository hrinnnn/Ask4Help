from __future__ import annotations

import json

import pytest

from tools.plot_dopamine_reward_curve import (
    aggregate_by_position,
    load_metric_records,
    metric_series,
    moving_average,
    summarize,
)


def test_load_and_summarize_metric_records(tmp_path):
    path = tmp_path / "metrics.jsonl"
    records = [
        {"env_idx": 0, "phi_next": 0.2, "shaping_reward": 0.2, "valid": True},
        {"env_idx": 1, "phi_next": 0.4, "shaping_reward": -0.1, "valid": True},
        {"env_idx": 0, "shaping_reward": 0.0, "valid": False, "skipped": True},
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    loaded = load_metric_records(path)
    summary = summarize(loaded)

    assert loaded == records
    assert summary["records"] == 3
    assert summary["queried_records"] == 2
    assert summary["valid_records"] == 2
    assert summary["environment_count"] == 2
    assert summary["phi_next"]["mean"] == pytest.approx(0.3)
    assert summary["shaping_reward"]["min"] == pytest.approx(-0.1)


def test_series_smoothing_and_aggregation_ignore_skipped_rows():
    records = [
        {"env_idx": 0, "phi_next": 0.0},
        {"env_idx": 0, "phi_next": 1.0},
        {"env_idx": 1, "phi_next": 0.5},
        {"env_idx": 1, "phi_next": 0.9, "skipped": True},
    ]

    series = metric_series(records, "phi_next")
    means, deviations = aggregate_by_position(series, window=1)

    assert series == {0: [0.0, 1.0], 1: [0.5]}
    assert moving_average([0.0, 1.0, 1.0], 2) == [0.0, 0.5, 1.0]
    assert means == pytest.approx([0.25, 1.0])
    assert deviations == pytest.approx([0.25, 0.0])


def test_invalid_json_reports_line_number(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("{}\nnot-json\n")

    with pytest.raises(ValueError, match="line 2"):
        load_metric_records(path)
