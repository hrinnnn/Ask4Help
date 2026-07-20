from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "select_pi05_checkpoint.py"
    spec = importlib.util.spec_from_file_location("select_pi05_checkpoint", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_success_rate_reads_online_runner_episode_summary(tmp_path):
    (tmp_path / "episodes.json").write_text(
        json.dumps([{"success": True}, {"success": False}, {"success": True}]),
        encoding="utf-8",
    )
    assert _module().success_rate(tmp_path) == 2 / 3


def test_common_selector_requires_both_members_to_be_weak_but_viable():
    path = Path(__file__).parents[1] / "tools" / "select_common_pi05_checkpoint.py"
    spec = importlib.util.spec_from_file_location("select_common_pi05_checkpoint", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    selected = module.select_common_step(
        {50: 0.2, 100: 0.3, 150: 0.6},
        {50: 0.3, 100: 0.4, 150: 0.4},
    )
    assert selected["selected_step"] == 100


def test_common_selector_falls_back_to_last_shared_step_when_neither_is_viable():
    path = Path(__file__).parents[1] / "tools" / "select_common_pi05_checkpoint.py"
    spec = importlib.util.spec_from_file_location("select_common_pi05_checkpoint", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.select_common_step({50: 0.0, 100: 0.1}, {50: 0.2, 100: 0.0})["selected_step"] == 100
