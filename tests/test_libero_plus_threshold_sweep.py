from __future__ import annotations

import importlib.util
from pathlib import Path


PATH = Path(__file__).resolve().parents[1] / "tools" / "libero_plus_failure" / "sweep_thresholds.py"
SPEC = importlib.util.spec_from_file_location("libero_plus_threshold_sweep", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _record(identifier: str, success: bool, scores: list[float]) -> dict[str, object]:
    return {"episode_id": identifier, "success": success, "scores": {"bridge_llmd": scores}}


def test_sweep_finds_the_best_threshold_and_respects_fpr_budget() -> None:
    records = [
        _record("success-low", True, [0.1]),
        _record("success-high", True, [0.8]),
        _record("failure-low", False, [0.2]),
        _record("failure-high", False, [0.9]),
    ]
    result = MODULE.scan_method(records, method="bridge_llmd", calibrated_threshold=1.0, fractions=(0.0, 0.5, 1.0))
    assert result["fraction_sweep"][1]["threshold"] == 0.5
    assert result["best_balanced_accuracy"]["threshold"] == 0.9
    assert result["best_balanced_accuracy"]["balanced_accuracy"] == 0.75
    assert result["best_recall_at_fpr"]["0.05"]["threshold"] == 0.9


def test_sweep_only_scans_the_requested_nonnegative_interval() -> None:
    records = [_record("success", True, [0.4]), _record("failure", False, [1.4])]
    result = MODULE.scan_method(records, method="bridge_llmd", calibrated_threshold=1.0, fractions=(0.0, 1.0))
    assert result["exact_candidate_count"] == 2
    assert result["best_balanced_accuracy"]["threshold"] == 1.0
