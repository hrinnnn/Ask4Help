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
