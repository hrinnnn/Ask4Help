#!/usr/bin/env python3
"""Audit StackPyramid oracle coverage and immutable-base policy gaps."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import imageio
import numpy as np


SPLITS = ("id", "stage1_ood", "stage2_ood", "stage3_ood")


EXPECTED_PREDICATES = {
    "stage1_ood": {"prefix": "red_grasped", "target": "red_lifted"},
    "stage2_ood": {"prefix": "red_lifted", "target": "red_placed"},
    "stage3_ood": {"prefix": "red_placed", "target": "blue_lifted"},
}


def _seed_values(spec: object, episodes: int) -> list[int]:
    """Expand a declared contiguous seed range or an explicit seed list."""
    if isinstance(spec, dict):
        start = int(spec["start"])
        count = int(spec["count"])
        if count != episodes:
            raise ValueError(f"seed manifest count {count} != requested episodes {episodes}")
        return [start + index for index in range(count)]
    values = [int(value) for value in spec]
    if len(values) != episodes:
        raise ValueError(f"seed manifest list has {len(values)} values, expected {episodes}")
    return values


def _load_seed_manifest(args: argparse.Namespace) -> dict | None:
    if args.seed_manifest is None:
        return None
    manifest = json.loads(args.seed_manifest.read_text(encoding="utf-8"))
    expected_format = f"stackpyramid_timing_protocol_seed_manifest_{args.geometry}"
    if manifest.get("format") != expected_format:
        raise ValueError(f"seed manifest format does not match geometry: {manifest.get('format')}")
    if manifest.get("geometry") != args.geometry or not manifest.get("declared_before_execution"):
        raise ValueError("seed manifest must declare the selected geometry before execution")
    if not manifest.get("paired_reset", {}).get("enabled"):
        raise ValueError("seed manifest must enable paired reset")
    if manifest.get("stage_predicate") != EXPECTED_PREDICATES:
        raise ValueError("seed manifest stage predicates do not match the frozen contract")
    for group in ("oracle", "base_policy"):
        for split in SPLITS:
            _seed_values(manifest.get(group, {}).get(split), args.episodes)
    return manifest


def run_oracle_split(args: argparse.Namespace, split: str, output: Path) -> dict:
    from tools.collect_stackpyramid_xvla_dagger import (
        StackPyramidOracle,
        StepRecorder,
        _summary,
        _install_rrt_fallback,
        frame_array,
    )
    from tools.stackpyramid_task import register_stackpyramid_splits, stackpyramid_env_id

    register_stackpyramid_splits()
    _install_rrt_fallback()
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    output.mkdir(parents=True)
    videos = output / "videos"
    videos.mkdir()
    env = StepRecorder(gym.make(
        stackpyramid_env_id(split),
        obs_mode="rgb+state",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        sim_backend=args.oracle_sim_backend,
        render_backend=args.oracle_render_backend,
    ))
    rows = []
    try:
        if args.seed_manifest_data is not None:
            seeds = _seed_values(args.seed_manifest_data["oracle"][split], args.episodes)
        else:
            seeds = [args.seed + SPLITS.index(split) * 2000 + index for index in range(args.episodes)]
        for index, seed in enumerate(seeds):
            env.reset(seed=seed)
            error = None
            try:
                StackPyramidOracle(env).run()
            except Exception as exc:  # retain the failed reset for audit
                error = repr(exc)
            final = _summary(env)
            video = videos / f"{split}_{seed}.mp4"
            with imageio.get_writer(video, fps=10, codec="libx264", macro_block_size=None) as writer:
                for frame in env.frames:
                    rendered = frame_array(frame)
                    if rendered is not None:
                        writer.append_data(rendered)
            rows.append({
                "episode_index": index,
                "seed": seed,
                "split": split,
                "strict_success": bool(final["success"]),
                "ever_grasped": any(final["grasped"]),
                "steps": len(env.actions),
                "oracle_error": error,
                "video": str(video),
            })
            print(json.dumps(rows[-1]), flush=True)
    finally:
        env.close()
    summary = {
        "kind": "oracle",
        "split": split,
        "episodes": len(rows),
        "strict_success": sum(int(row["strict_success"]) for row in rows),
        "ever_grasped": sum(int(row["ever_grasped"]) for row in rows),
        "video_count": len(list(videos.glob("*.mp4"))),
        "rows": rows,
    }
    (output / "episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def run_policy_split(args: argparse.Namespace, split: str, output: Path) -> dict:
    command = [
        str(args.python),
        str(Path(__file__).with_name("evaluate_stackpyramid_xvla.py")),
        "--checkpoint", str(args.checkpoint),
        "--xvla-root", str(args.xvla_root),
        "--output", str(output),
        "--split", split,
        "--episodes", str(args.episodes),
        "--start-seed", str(
            _seed_values(args.seed_manifest_data["base_policy"][split], args.episodes)[0]
            if args.seed_manifest_data is not None
            else args.seed + 10000 + SPLITS.index(split) * 2000
        ),
        "--max-episode-steps", str(args.max_episode_steps),
        "--execute-horizon", "5",
        "--flow-steps", str(args.flow_steps),
        "--sim-backend", args.policy_sim_backend,
        "--render-backend", args.policy_render_backend,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).resolve().parents[1]), str(args.xvla_root)])
    output.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(command, env=env, check=False)
    summary_path = output / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"base-policy audit failed for {split}: rc={result.returncode}")
    summary = json.loads(summary_path.read_text())
    summary["return_code"] = result.returncode
    if summary.get("episodes") != args.episodes or summary.get("video_count") != args.episodes:
        raise RuntimeError(f"incomplete base-policy audit for {split}: {summary}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=70000)
    parser.add_argument("--max-episode-steps", type=int, default=250)
    parser.add_argument("--flow-steps", type=int, default=5)
    parser.add_argument("--oracle-sim-backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--oracle-render-backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--policy-sim-backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--policy-render-backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--geometry", choices=("v1", "v2", "v3"), default="v2")
    parser.add_argument("--seed-manifest", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    os.environ["STACKPYRAMID_OOD_GEOMETRY"] = args.geometry
    args.seed_manifest_data = _load_seed_manifest(args)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(args.xvla_root.resolve()))
    oracle = {}
    policy = {}
    for split in SPLITS:
        oracle[split] = run_oracle_split(args, split, args.output / "oracle" / split)
    for split in SPLITS:
        policy[split] = run_policy_split(args, split, args.output / "policy" / split)
    oracle_pass = all(summary["strict_success"] / args.episodes >= 0.90 for summary in oracle.values())
    base_policy_pass = (
        policy["id"]["strict_success"] / args.episodes >= 0.80
        and all(policy[split]["strict_success"] / args.episodes <= 0.50 for split in SPLITS[1:])
    )
    audit = {
        "format": "stackpyramid_protocol_audit_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "geometry": args.geometry,
        "seed_manifest": str(args.seed_manifest.resolve()) if args.seed_manifest else None,
        "benchmark_version": (args.seed_manifest_data or {}).get("benchmark_version"),
        "episodes_per_split": args.episodes,
        "oracle": oracle,
        "base_policy": policy,
        "gates": {"oracle_pass": oracle_pass, "base_policy_pass": base_policy_pass},
        "rules": {
            "oracle_strict_success_minimum": 0.90,
            "base_policy_id_strict_success_minimum": 0.80,
            "base_policy_ood_strict_success_maximum": 0.50,
        },
    }
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if oracle_pass and base_policy_pass:
        (args.output / "PROTOCOL_AUDIT_COMPLETE").write_text("complete\n", encoding="utf-8")
    else:
        (args.output / "PROTOCOL_AUDIT_FAILED").write_text(json.dumps(audit["gates"]) + "\n", encoding="utf-8")
        raise RuntimeError(f"protocol audit gates failed: {audit['gates']}")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
