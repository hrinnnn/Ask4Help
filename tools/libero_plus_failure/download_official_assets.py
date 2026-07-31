#!/usr/bin/env python3
"""Resume-safe download and verification of the official LIBERO-Plus assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
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


def asset_member_prefix(names: list[str]) -> str:
    """Find the archive prefix immediately preceding the official assets dir.

    The upstream `assets.zip` currently retains its historic build-machine
    prefix (`inspire/.../LIBERO-plus-0/assets/`) instead of a portable
    top-level `assets/` directory.  Exactly one assets subtree is accepted.
    """

    prefixes = set()
    for name in names:
        parts = Path(name).parts
        for index, part in enumerate(parts):
            if part == "assets":
                prefixes.add("/".join(parts[: index + 1]) + "/")
                break
    if len(prefixes) != 1:
        raise ValueError("cannot uniquely locate official assets directory: " + repr(sorted(prefixes)))
    return next(iter(prefixes))


def extract_assets(bundle: zipfile.ZipFile, target: Path) -> str:
    """Extract only the official assets subtree into a clean target directory."""

    prefix = asset_member_prefix(bundle.namelist())
    resolved_target = target.resolve()
    for member in bundle.infolist():
        if not member.filename.startswith(prefix):
            continue
        relative = member.filename[len(prefix) :]
        if not relative:
            continue
        destination = target / relative
        resolved_destination = destination.resolve()
        if resolved_destination.parent != resolved_target and resolved_target not in resolved_destination.parents:
            raise ValueError("unsafe path in official assets.zip: " + relative)
        if member.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        with bundle.open(member) as source, destination.open("wb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
    return prefix


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
        archive_prefix = extract_assets(bundle, target)
    if not target.is_dir() or not any(target.iterdir()):
        raise RuntimeError("asset extraction created no assets directory")
    manifest = {
        "format": "libero_plus_official_assets_v1",
        "source": "Sylvest/LIBERO-plus/assets.zip",
        "url": ASSET_URL,
        "archive": str(archive),
        "archive_sha256": actual_sha,
        "target": str(target),
        "archive_assets_prefix": archive_prefix,
        "top_level_entries": sorted(path.name for path in target.iterdir()),
    }
    (args.archive_dir / "assets_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
