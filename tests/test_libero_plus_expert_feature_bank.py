from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))
MODULE_PATH = ROOT / "tools" / "libero_plus_failure" / "build_expert_feature_bank.py"
SPEC = importlib.util.spec_from_file_location("libero_plus_expert_feature_bank", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_official_expert_selection_is_balanced_and_never_uses_action_tail(tmp_path: Path) -> None:
    meta = tmp_path / "meta"
    meta.mkdir()
    tasks = [{"task_index": index, "task": "task-%d" % index} for index in range(10)]
    episodes = [
        {"episode_index": task * 20 + demo, "tasks": ["task-%d" % task], "length": 25 + demo}
        for task in range(10)
        for demo in range(12)
    ]
    (meta / "tasks.jsonl").write_text("\n".join(json.dumps(row) for row in tasks) + "\n", encoding="utf-8")
    (meta / "episodes.jsonl").write_text("\n".join(json.dumps(row) for row in episodes) + "\n", encoding="utf-8")
    selected = MODULE.select_records(meta, seed=11)
    assert len(selected) == 1000
    assert {row["task_id"] for row in selected} == {str(index) for index in range(10)}
    for row in selected:
        assert int(row["anchor_id"]) + 10 <= int(row["episode_length"])
        assert row["success_source"] == "official_expert_demo"
