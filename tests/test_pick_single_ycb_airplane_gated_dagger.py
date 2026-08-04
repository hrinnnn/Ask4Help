from __future__ import annotations

import importlib.util
from pathlib import Path


PATH = Path(__file__).resolve().parents[1] / "tools" / "collect_pick_single_ycb_airplane_gated_dagger.py"
SPEC = importlib.util.spec_from_file_location("airplane_gated", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_raw_attempts_strictly_alternate_id_then_ood():
    assert [MODULE.alternating_split(index) for index in range(6)] == ["id", "ood"] * 3


def test_bridge_pca_uses_strict_threshold_exceedance():
    assert not MODULE.should_query_bridge_pca(0.5, 0.5)
    assert MODULE.should_query_bridge_pca(0.50001, 0.5)


def test_all_real_terminal_expert_actions_are_admitted():
    assert MODULE.admitted_expert_suffix(success=True, expert_start=50, action_count=53) == (50, 53)
    assert MODULE.admitted_expert_suffix(success=True, expert_start=50, action_count=51) == (50, 51)


def test_failed_or_empty_suffix_is_not_admitted():
    assert MODULE.admitted_expert_suffix(success=False, expert_start=50, action_count=60) is None
    assert MODULE.admitted_expert_suffix(success=True, expert_start=None, action_count=60) is None
    assert MODULE.admitted_expert_suffix(success=True, expert_start=60, action_count=60) is None


def test_zero_action_raw_attempt_remains_a_rejected_training_example():
    assert MODULE.admitted_expert_suffix(success=False, expert_start=0, action_count=0) is None
