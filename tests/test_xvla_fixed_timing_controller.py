from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "run_xvla_fixedgrid_knee_controller.py"
    spec = importlib.util.spec_from_file_location("fixed_timing_controller", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_calibration_command_contains_fixed_timing_step() -> None:
    module = _load_module()
    command = module.calibration_command(
        python="python",
        worktree=Path("/worktree"),
        checkpoint=Path("/ckpt"),
        xvla_root=Path("/xvla"),
        output=Path("/out"),
        repo_id=Path("/dataset"),
        seed_manifest=Path("/seeds.json"),
        step=45,
        ood_split="ood",
        seed_count=2,
    )
    assert "fixed_timing" in command
    assert "--timing-step" in command
    assert command[command.index("--timing-step") + 1] == "45"
