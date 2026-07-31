#!/usr/bin/env python3
"""Resume-safe download and verification of the official LIBERO-Plus assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path


ASSET_URL = "https://hf-mirror.com/datasets/Sylvest/LIBERO-plus/resolve/main/assets.zip?download=true"
ASSET_SHA256 = "96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero-root", type=Path, required=True, help="official clone's libero/libero path")
    parser.add_argument("--archive-dir", type=Path, required=True)
    args = parser.parse_args()
    target = args.libero_root / "assets"
    archive = args.archive_dir / "assets.zip"
    if target.is_dir() and any(target.iterdir()):
        raise FileExistsError("assets directory already exists; refuse to mix versions: " + str(target))
    args.archive_dir.mkdir(parents=True, exist_ok=True)
    if not archive.is_file() or sha256(archive) != ASSET_SHA256:
        subprocess.run(["curl", "--fail", "--location", "--continue-at", "-", "--output", str(archive), ASSET_URL], check=True)
    actual_sha = sha256(archive)
    if actual_sha != ASSET_SHA256:
        raise RuntimeError("official assets.zip SHA256 mismatch: " + actual_sha)
    with zipfile.ZipFile(archive) as bundle:
        roots = {Path(name).parts[0] for name in bundle.namelist() if name and not name.endswith("/")}
        if roots != {"assets"}:
            raise ValueError("unexpected official assets.zip top level: " + repr(sorted(roots)))
        bundle.extractall(args.libero_root)
    if not target.is_dir() or not any(target.iterdir()):
        raise RuntimeError("asset extraction created no assets directory")
    manifest = {
        "format": "libero_plus_official_assets_v1",
        "source": "Sylvest/LIBERO-plus/assets.zip",
        "url": ASSET_URL,
        "archive": str(archive),
        "archive_sha256": actual_sha,
        "target": str(target),
        "top_level_entries": sorted(path.name for path in target.iterdir()),
    }
    (args.archive_dir / "assets_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
