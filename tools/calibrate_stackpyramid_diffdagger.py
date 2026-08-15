#!/usr/bin/env python3
"""Calibrate Diff-DAgger's fixed q=.95 threshold on successful ID rollouts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--successful-rollouts", type=int, default=25)
    parser.add_argument("--max-attempts", type=int, default=60)
    parser.add_argument("--start-seed", type=int, default=46000)
    parser.add_argument("--flow-steps", type=int, default=5)
    parser.add_argument("--diff-timesteps", type=int, default=16)
    parser.add_argument("--max-episode-steps", type=int, default=150)
    parser.add_argument("--sim-backend", choices=("gpu", "cpu"), default="cpu")
    parser.add_argument("--render-backend", choices=("gpu", "cpu"), default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.successful_rollouts < 2:
        raise ValueError("at least two successful rollouts are required")

    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root), str(args.xvla_root)]
    from models.modeling_xvla import XVLA
    from models.processing_xvla import XVLAProcessor
    from tools.collect_stackpyramid_xvla_dagger import _diff_score, _predict, _summary
    from tools.stackpyramid_task import register_stackpyramid_splits, stackpyramid_env_id

    register_stackpyramid_splits()
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    device = torch.device("cuda")
    model = XVLA.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16).to(device).eval()
    processor = XVLAProcessor.from_pretrained(args.checkpoint)
    env = gym.make(
        stackpyramid_env_id("id"),
        obs_mode="rgb+state",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
    )
    rows: list[dict[str, object]] = []
    maxima: list[float] = []
    attempt = 0
    try:
        while attempt < args.max_attempts and len(maxima) < args.successful_rollouts:
            seed = args.start_seed + attempt
            raw_obs, _ = env.reset(seed=seed)
            executed = 0
            success = False
            scores: list[float] = []
            while executed < args.max_episode_steps and not success:
                generated, _bridge, inputs, encoding = _predict(
                    model, processor, raw_obs, device, seed + executed, args.flow_steps
                )
                scores.append(_diff_score(model, inputs, encoding, generated, args.diff_timesteps))
                for action in np.asarray(generated[:10], dtype=np.float32)[:5]:
                    raw_obs, _, terminated, truncated, _ = env.step(action)
                    executed += 1
                    success = bool(_summary(env)["success"])
                    if bool(terminated) or bool(truncated) or success:
                        break
            maximum = max(scores) if scores else float("nan")
            row = {
                "attempt": attempt,
                "seed": seed,
                "success": bool(success),
                "steps": executed,
                "score_count": len(scores),
                "max_score": maximum,
            }
            rows.append(row)
            if success and np.isfinite(maximum):
                maxima.append(float(maximum))
            print(json.dumps(row), flush=True)
            attempt += 1
    finally:
        env.close()

    if len(maxima) < args.successful_rollouts:
        raise RuntimeError(
            f"Diff calibration found {len(maxima)} successful ID rollouts; "
            f"required {args.successful_rollouts}"
        )
    rank = min(len(maxima), math.ceil((len(maxima) + 1) * 0.95))
    threshold = sorted(maxima)[rank - 1]
    result = {
        "format": "stackpyramid_diffdagger_calibration_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "q": 0.95,
        "successful_rollouts": len(maxima),
        "attempts": len(rows),
        "diff_timesteps": args.diff_timesteps,
        "flow_steps": args.flow_steps,
        "threshold": float(threshold),
        "trajectory_maxima": maxima,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (args.output.parent / "DIFF_CALIBRATION_COMPLETE").write_text("complete\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
