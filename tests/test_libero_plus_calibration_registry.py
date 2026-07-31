from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "tools" / "libero_plus_failure" / "register_calibration.py"
SPEC = importlib.util.spec_from_file_location("libero_plus_calibration_registry", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_registry_is_immutable_and_binds_thresholds_to_assets(tmp_path: Path) -> None:
    assets = tmp_path / "assets.pt"
    assets.write_bytes(b"reference-assets")
    asset_sha = hashlib.sha256(assets.read_bytes()).hexdigest()
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(json.dumps({"reference_assets_sha256": asset_sha, "thresholds": {"llmd": {"threshold": 3.0}}}))

    destination = MODULE.register_calibration(
        thresholds_path=thresholds,
        reference_assets_path=assets,
        registry_root=tmp_path / "registry",
        calibration_id="libero10-q95-v1",
        policy_checkpoint="pi05_libero",
        source_commit="abc123",
        protocol={"delta": 0.05},
    )
    manifest = json.loads((destination / "registry_manifest.json").read_text())
    assert manifest["reference_assets_sha256"] == asset_sha
    assert (destination / "thresholds.json").read_text() == thresholds.read_text()
    with pytest.raises(FileExistsError):
        MODULE.register_calibration(
            thresholds_path=thresholds, reference_assets_path=assets, registry_root=tmp_path / "registry",
            calibration_id="libero10-q95-v1", policy_checkpoint="pi05_libero", source_commit="abc123", protocol={},
        )


def test_registry_rejects_asset_mismatch(tmp_path: Path) -> None:
    assets = tmp_path / "assets.pt"
    assets.write_bytes(b"reference-assets")
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(json.dumps({"reference_assets_sha256": "wrong", "thresholds": {"llmd": {"threshold": 3.0}}}))
    with pytest.raises(ValueError, match="SHA mismatch"):
        MODULE.register_calibration(
            thresholds_path=thresholds, reference_assets_path=assets, registry_root=tmp_path / "registry",
            calibration_id="libero10-q95-v1", policy_checkpoint="pi05_libero", source_commit="abc123", protocol={},
        )
