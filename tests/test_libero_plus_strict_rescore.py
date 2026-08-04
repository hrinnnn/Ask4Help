from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "libero_plus_strict_rescore",
    ROOT / "tools" / "libero_plus_failure" / "rescore_strict_metrics.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_rank_fusion_aligns_shorter_acc_trace_and_preserves_or() -> None:
    records = [
        {
            "episode_id": "a",
            "success": True,
            "scores": {"final_llmd": [1.0, 2.0], "acc": [10.0]},
        },
        {
            "episode_id": "b",
            "success": False,
            "scores": {"final_llmd": [3.0, 1.5], "acc": [5.0]},
        },
    ]

    MODULE.attach_strict_rank_fusion(records)

    first = records[0]["scores"]["vla_fail_rank_or_strict"]
    second = records[1]["scores"]["vla_fail_rank_or_strict"]
    assert len(first) == len(records[0]["scores"]["final_llmd"])
    assert first[0] < second[0]
    # At the second decision point, record a is anomalous by ACC even though
    # its LLMD is smaller; logical OR must retain that anomaly.
    assert first[1] > second[1]


def test_empirical_cdf_uses_mid_ranks_for_ties() -> None:
    ranks = MODULE.empirical_cdf([1.0, 1.0, 3.0, 5.0])
    assert ranks[1.0] == 0.375
    assert ranks[3.0] == 0.75
    assert ranks[5.0] == 1.0
