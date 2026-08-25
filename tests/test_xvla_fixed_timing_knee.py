from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "summarize_xvla_fixed_timing_knee.py"
    spec = importlib.util.spec_from_file_location("fixed_timing_knee", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_knee_prefers_tradeoff_bend() -> None:
    module = _load_module()
    # Early: expensive but aligned; late: cheap but deviated; middle is the bend.
    assert module._knee_index(
        np.asarray([1.0, 0.8, 0.45, 0.2]),
        np.asarray([0.05, 0.08, 0.15, 0.9]),
    ) == 2


def test_aligned_distance_is_zero_for_identical_suffix() -> None:
    module = _load_module()
    seq = np.arange(18, dtype=np.float32).reshape(6, 3)
    assert module.aligned_distance(seq, seq, np.ones(3, dtype=np.float32)) == 0.0


def test_anchor_loader_supports_airplane_ever_grasped_endpoint(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path / "calibration"
    anchor = root / "step_0"
    anchor.mkdir(parents=True)
    state_path = anchor / "state.npy"
    np.save(state_path, np.zeros((3, 2), dtype=np.float32))
    (anchor / "episodes.jsonl").write_text(
        '{"seed": 7, "ever_grasped": true, "expert_start_step": 0, "expert_action_steps": 2, "steps": 2, "task_states": "state.npy"}\n',
        encoding="utf-8",
    )
    loaded = module._load_anchor(root, 0, endpoint="ever_grasped")
    assert list(loaded) == [7]
