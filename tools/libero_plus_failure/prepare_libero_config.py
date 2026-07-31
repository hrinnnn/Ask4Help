#!/usr/bin/env python3
"""Create a non-interactive, isolated LIBERO or LIBERO-Plus config directory."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero-root", type=Path, required=True, help=".../libero/libero directory")
    parser.add_argument("--config-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.libero_root.resolve()
    required = {"bddl_files": root / "bddl_files", "init_files": root / "init_files", "assets": root / "assets"}
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("LIBERO root is incomplete: " + ", ".join(missing))
    config = args.config_dir / "config.yaml"
    if config.exists():
        existing = config.read_text(encoding="utf-8")
        expected = "benchmark_root: " + str(root)
        if expected not in existing:
            raise FileExistsError("refusing to replace a config for another LIBERO root: " + str(config))
        print(config)
        return
    args.config_dir.mkdir(parents=True, exist_ok=False)
    config.write_text(
        "\n".join(
            [
                "benchmark_root: " + str(root),
                "bddl_files: " + str(required["bddl_files"]),
                "init_states: " + str(required["init_files"]),
                "datasets: " + str(root.parent / "datasets"),
                "assets: " + str(required["assets"]),
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(config)


if __name__ == "__main__":
    main()
