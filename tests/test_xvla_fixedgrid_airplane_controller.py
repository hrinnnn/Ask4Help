from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "run_xvla_fixedgrid_airplane_controller.py"
    spec = importlib.util.spec_from_file_location("fixed_airplane_controller", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_collection_command_freezes_ood_fixed_step(tmp_path: Path) -> None:
    module = _load_module()
    command = module.collection_command(
        python="python",
        worktree=tmp_path / "worktree",
        checkpoint=tmp_path / "ckpt",
        xvla_root=tmp_path / "xvla",
        calibration=tmp_path / "calibration.json",
        pca_asset=tmp_path / "asset.pt",
        output=tmp_path / "out",
        repo_id=tmp_path / "dataset",
        seed_manifest_path=tmp_path / "seeds.json",
        step=45,
        count=20,
    )
    assert "fixed_timing" in command
    assert command[command.index("--timing-step") + 1] == "45"
    assert command[command.index("--only-split") + 1] == "ood"
    assert "--consume-all-seeds" in command


def test_complete_evidence_allows_recoverability_failures(tmp_path: Path) -> None:
    module = _load_module()
    output = tmp_path / "collection"
    output.mkdir()
    (output / "episodes.jsonl").write_text("{}\n{}\n", encoding="utf-8")
    (output / "summary.json").write_text(
        '{"raw_total": 2, "accepted_total": 1}\n', encoding="utf-8"
    )
    assert module.complete_evidence(output, expected=2)
