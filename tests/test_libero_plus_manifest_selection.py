from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
PATH = ROOT / "tools" / "libero_plus_failure" / "build_official_manifest.py"
SPEC = importlib.util.spec_from_file_location("libero_plus_manifest_selection", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _rows() -> list[dict]:
    return [
        {"plus_task_id": task * 100 + offset, "clean_task_index": task, "difficulty_level": offset % 5 + 1}
        for task in range(2)
        for offset in range(8)
    ]


def test_task_balanced_hash_subset_is_reproducible_and_balanced() -> None:
    first, first_info = MODULE.task_balanced_hash_subset(_rows(), count=10, seed=7)
    second, second_info = MODULE.task_balanced_hash_subset(_rows(), count=10, seed=7)
    assert first == second
    assert first_info == second_info
    assert first_info["per_clean_task"] == {"0": 5, "1": 5}
    assert len({row["plus_task_id"] for row in first}) == 10
