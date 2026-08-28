#!/usr/bin/env python3
"""Evaluate one fixed-timing updated OpenDrawer policy on ID and Grasp-OOD."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def valid(output: Path, episodes: int) -> bool:
    summary = output / "summary.json"
    if not summary.is_file():
        return False
    try:
        payload = json.loads(summary.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if payload.get("episodes") != episodes or len(payload.get("rows", [])) != episodes:
        return False
    if len(list((output / "videos").glob("*.mp4"))) != episodes:
        return False
    for row in payload["rows"]:
        for key in ("actions", "states", "timeline", "reset_metadata", "video"):
            path = Path(str(row.get(key, "")))
            if not path.is_file():
                return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pi05-base", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--id-seed", type=int, required=True)
    parser.add_argument("--ood-seed", type=int, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu-set", default="0-19")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty evaluation root: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "PYTHONPATH": os.pathsep.join([str(args.root), str(args.root / "RLinf"), env.get("PYTHONPATH", "")]),
            "ASK4HELP_RLINF_ROOT": str(args.root / "RLinf"),
            "OMP_NUM_THREADS": "20",
            "MKL_NUM_THREADS": "20",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    for split, seed in (("id", args.id_seed), ("grasp_ood", args.ood_seed)):
        output = args.output_root / split
        output.mkdir(parents=True, exist_ok=True)
        command = [
            str(args.python), "-u", str(args.evaluator),
            "--checkpoint", str(args.checkpoint),
            "--pi05-base", str(args.pi05_base),
            "--norm-stats", str(args.norm_stats),
            "--output-dir", str(output),
            "--episodes", str(args.episodes),
            "--seed", str(seed),
            "--split", split,
            "--execute-horizon", "5",
            "--max-episode-steps", "400",
        ]
        log = args.output_root / f"{split}.log"
        with log.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps({"command": command, "gpu": args.gpu, "cpu_set": args.cpu_set}) + "\n")
            stream.flush()
            result = subprocess.run(["taskset", "-c", args.cpu_set, *command], env=env, stdout=stream, stderr=subprocess.STDOUT, check=False)
        if not valid(output, args.episodes):
            (args.output_root / "EVAL_FAILED").write_text(f"split={split} returncode={result.returncode}\n", encoding="utf-8")
            raise SystemExit(f"incomplete evaluation for {split}: returncode={result.returncode}")
        (output / "EVAL_COMPLETE").write_text("summary and per-episode evidence verified\n", encoding="utf-8")
    (args.output_root / "EVAL_COMPLETE").write_text("ID and Grasp-OOD summaries and evidence verified\n", encoding="utf-8")
    print("OPEN_DRAWER_TIMING_EVAL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
