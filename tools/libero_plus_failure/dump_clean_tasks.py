#!/usr/bin/env python3
"""Export canonical clean LIBERO-10 task names before loading LIBERO-Plus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from libero.libero import benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite " + str(args.output))
    suite = benchmark.get_benchmark_dict()[args.suite]()
    rows = [{"task_index": index, "name": str(suite.get_task(index).name)} for index in range(suite.n_tasks)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
