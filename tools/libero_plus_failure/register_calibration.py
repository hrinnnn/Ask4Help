#!/usr/bin/env python3
"""Register an immutable, SHA-bound LIBERO failure-detector calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Mapping


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def register_calibration(
    *, thresholds_path: Path, reference_assets_path: Path, registry_root: Path, calibration_id: str,
    policy_checkpoint: str, source_commit: str, protocol: Mapping[str, Any],
) -> Path:
    """Create a non-overwritable calibration bundle after compatibility checks."""

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", calibration_id):
        raise ValueError("calibration_id may contain only letters, digits, dot, underscore, and hyphen")
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    if not thresholds.get("thresholds"):
        raise ValueError("threshold file contains no detector thresholds")
    asset_sha = sha256(reference_assets_path)
    if thresholds.get("reference_assets_sha256") != asset_sha:
        raise ValueError("threshold/reference asset SHA mismatch")
    destination = registry_root / calibration_id
    if destination.exists():
        raise FileExistsError("refusing to overwrite existing calibration " + str(destination))

    destination.mkdir(parents=True, exist_ok=False)
    copied_thresholds = destination / "thresholds.json"
    shutil.copy2(thresholds_path, copied_thresholds)
    manifest = {
        "format": "libero_plus_failure_calibration_registry_v1",
        "calibration_id": calibration_id,
        "thresholds_sha256": sha256(copied_thresholds),
        "reference_assets_path": str(reference_assets_path),
        "reference_assets_sha256": asset_sha,
        "policy_checkpoint": policy_checkpoint,
        "source_commit": source_commit,
        "successful_policy_rollouts": thresholds.get("successful_policy_rollouts"),
        "calibration_source": thresholds.get("calibration_source"),
        "protocol": dict(protocol),
    }
    (destination / "registry_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--reference-assets", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--calibration-id", required=True)
    parser.add_argument("--policy-checkpoint", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--protocol-json", type=Path)
    parser.add_argument("--protocol", default="{}", help="inline JSON protocol metadata")
    args = parser.parse_args()
    protocol = (
        json.loads(args.protocol_json.read_text(encoding="utf-8"))
        if args.protocol_json is not None
        else json.loads(args.protocol)
    )
    destination = register_calibration(
        thresholds_path=args.thresholds,
        reference_assets_path=args.reference_assets,
        registry_root=args.registry_root,
        calibration_id=args.calibration_id,
        policy_checkpoint=args.policy_checkpoint,
        source_commit=args.source_commit,
        protocol=protocol,
    )
    print(destination)


if __name__ == "__main__":
    main()
