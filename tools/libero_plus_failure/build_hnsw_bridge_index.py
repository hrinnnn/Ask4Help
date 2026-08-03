#!/usr/bin/env python3
"""Build a versioned low-latency HNSW index from frozen bridge kNN assets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hnsw_knn import build_and_save  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, default=128)
    args = parser.parse_args()
    print(
        json.dumps(
            build_and_save(
                assets_path=args.assets,
                output_dir=args.output_dir,
                m=args.m,
                ef_construction=args.ef_construction,
                ef_search=args.ef_search,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
