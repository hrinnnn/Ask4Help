#!/usr/bin/env python3
"""Fit immutable no-training detector assets from the cached expert bank."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "tools" / "libero_plus_failure_assets.py"
SPEC = importlib.util.spec_from_file_location("libero_plus_failure_assets", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--knn-k", type=int, default=10)
    args = parser.parse_args()
    cache = torch.load(args.feature_cache, map_location="cpu", weights_only=False)
    manifest = MODULE.save_assets(feature_cache=cache, output_dir=args.output_dir, knn_k=args.knn_k)
    print(manifest["reference_assets_path"])


if __name__ == "__main__":
    main()
