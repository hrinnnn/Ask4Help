#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlinf.data.awbc import AWBCProgressManifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Concatenate AWBC manifests in the same order as ConcatDataset."
    )
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    offset = 0
    combined = []
    for input_path in args.input:
        path = Path(input_path).expanduser()
        manifest = AWBCProgressManifest.load(path)
        if set(record.dataset_index for record in manifest) != set(range(len(manifest))):
            raise ValueError(f"manifest indices must be contiguous from zero: {path}")
        with path.open(encoding="utf-8") as file:
            rows = [json.loads(line) for line in file if line.strip()]
        for row in rows:
            row["dataset_index"] = int(row["dataset_index"]) + offset
            combined.append(row)
        offset += len(manifest)

    with output.open("w", encoding="utf-8") as file:
        for row in sorted(combined, key=lambda item: item["dataset_index"]):
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "rows": len(combined)}, indent=2))


if __name__ == "__main__":
    main()
