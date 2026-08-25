from __future__ import annotations

import ast
from pathlib import Path


def _collector_tree():
    path = Path(__file__).resolve().parents[1] / "tools" / "collect_stackcube_xvla_dagger.py"
    return ast.parse(path.read_text(encoding="utf-8"))


def test_fixed_timing_condition_is_registered() -> None:
    tree = _collector_tree()
    timing_assign = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "TIMING_CONDITIONS" for target in node.targets)
    )
    values = ast.literal_eval(timing_assign.value)
    assert "fixed_timing" in values


def test_fixed_timing_step_contract_is_bounded() -> None:
    tree = _collector_tree()
    horizon_assign = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "TASK_HORIZON" for target in node.targets)
    )
    horizon = ast.literal_eval(horizon_assign.value)
    assert horizon == 150
    assert 0 <= 100 < horizon
