#!/usr/bin/env python3
"""Build the auditable clean/Plus task pairing used by the main table."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from libero_plus_failure_protocol import build_libero_plus_manifest  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classification", type=Path, required=True)
    parser.add_argument("--clean-tasks", type=Path, required=True, help="JSON list from the unmodified LIBERO-10 install")
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite " + str(args.output))
    classifications = json.loads(args.classification.read_text(encoding="utf-8"))
    clean = json.loads(args.clean_tasks.read_text(encoding="utf-8"))
    if args.suite not in classifications:
        raise KeyError("classification has no suite " + args.suite)
    rows = build_libero_plus_manifest(classifications=classifications[args.suite], clean_tasks=clean)
    payload = {
        "format": "libero_plus_failure_task_manifest_v1",
        "suite": args.suite,
        "categories": ["Camera Viewpoints", "Robot Initial States", "Objects Layout"],
        "classification": str(args.classification),
        "classification_sha256": sha256(args.classification),
        "clean_tasks": str(args.clean_tasks),
        "clean_tasks_sha256": sha256(args.clean_tasks),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
