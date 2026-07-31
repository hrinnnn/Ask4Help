from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PATH = Path(__file__).parents[1] / "tools" / "libero_plus_failure" / "render_leaderboard_table.py"
SPEC = importlib.util.spec_from_file_location("render_libero_plus_leaderboard_table", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _method(value: float) -> dict:
    return {"episodes": 2, "roc_auc": value, "aucpr": value, "aucpdt": value, "balanced_accuracy": value,
            "weighted_accuracy": value, "tpr": value, "tnr": value, "fpr": 1 - value, "precision": value,
            "recall": value, "f1": value, "mean_normalized_detection_time": 1 - value, "twa": value}


def test_table_renderer_writes_percentage_markdown_and_long_csv(tmp_path: Path) -> None:
    summary = {
        "all": {"bridge_llmd": _method(0.8)},
        "groups": {"category=Camera Viewpoints": {"bridge_llmd": _method(0.7), "runtime_ms": {}}},
        "runtime_ms": {"policy_mean_ms": 10.0, "feature_probe_mean_ms": 2.0, "total_mean_ms": 12.0,
                       "feature_probe_overhead_percent": 20.0},
    }
    output = tmp_path / "table"
    MODULE.write_tables(summary, output)
    markdown = (output / "leaderboard.md").read_text(encoding="utf-8")
    assert "Bridge LLMD" in markdown
    assert "80.0%" in markdown
    assert "Camera Viewpoints" in markdown
    assert (output / "leaderboard.csv").read_text(encoding="utf-8").count("bridge_llmd") == 2
    assert json.loads((output / "runtime.json").read_text(encoding="utf-8"))["total_mean_ms"] == 12.0
