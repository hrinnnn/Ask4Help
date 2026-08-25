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


def test_completed_collection_evidence_requires_full_denominator(tmp_path: Path) -> None:
    module = _load_module()
    output = tmp_path / "collection"
    output.mkdir()
    (output / "episodes.jsonl").write_text('{"seed": 1}\n{"seed": 2}\n', encoding="utf-8")
    (output / "summary.json").write_text(
        '{"raw_total": 2, "accepted_total": 2}\n', encoding="utf-8"
    )
    assert module.completed_collection_evidence(output, expected_episodes=2)
    assert not module.completed_collection_evidence(output, expected_episodes=3)
