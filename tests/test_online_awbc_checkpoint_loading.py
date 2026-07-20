from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "maniskill_pi05_vfd_online_awbc.py"
    spec = importlib.util.spec_from_file_location("online_awbc_runner", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_sft_full_weights_from_global_step_directory(tmp_path):
    checkpoint = tmp_path / "global_step_50" / "actor" / "model_state_dict"
    checkpoint.mkdir(parents=True)
    weights = checkpoint / "full_weights.pt"
    weights.write_bytes(b"weights")
    assert _module().resolve_sft_full_weights(tmp_path / "global_step_50") == weights


def test_resolve_sft_full_weights_returns_none_for_openpi_model_directory(tmp_path):
    assert _module().resolve_sft_full_weights(tmp_path) is None


def test_parse_args_supports_pure_policy_evaluation(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "maniskill_pi05_vfd_online_awbc.py",
            "--mode",
            "online",
            "--output-dir",
            str(tmp_path),
            "--no-compute-vfd",
        ],
    )
    assert _module().parse_args().compute_vfd is False
