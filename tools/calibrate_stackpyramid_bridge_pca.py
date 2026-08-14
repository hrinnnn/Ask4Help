#!/usr/bin/env python3
"""Calibrate Bridge-PCA's fixed q=.95 alarm threshold on independent ID rollouts."""

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
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--start-seed", type=int, default=45000)
    parser.add_argument("--flow-steps", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root), str(args.xvla_root)]
    from models.modeling_xvla import XVLA
    from models.processing_xvla import XVLAProcessor
    from rlinf.algorithms.vla_fail import PCAResidualStatistics, pca_residual_score
    from tools.collect_stackpyramid_xvla_dagger import _predict, _summary, register_stackpyramid_splits
    from tools.stackpyramid_task import stackpyramid_env_id

    register_stackpyramid_splits()
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    device = torch.device("cuda")
    model = XVLA.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16).to(device).eval()
    processor = XVLAProcessor.from_pretrained(args.checkpoint)
    payload = torch.load(args.asset, map_location="cpu", weights_only=False)
    stats = PCAResidualStatistics.from_state_dict(payload["statistics"])
    env = gym.make(stackpyramid_env_id("id"), obs_mode="rgb+state", control_mode="pd_joint_pos", render_mode="rgb_array", sim_backend="gpu", render_backend="gpu")
    maxima = []
    rows = []
    try:
        for episode in range(args.episodes):
            raw_obs, _ = env.reset(seed=args.start_seed + episode)
            values = []
            executed = 0
            success = False
            while executed < 150 and not success:
                generated, bridge, _inputs, _encoding = _predict(model, processor, raw_obs, device, args.start_seed + episode + executed, args.flow_steps)
                values.append(float(pca_residual_score(bridge.unsqueeze(1), stats)[0].item()))
                for action in np.asarray(generated[:10], dtype=np.float32)[:5]:
                    raw_obs, _, terminated, truncated, _ = env.step(action)
                    executed += 1
                    success = _summary(env)["success"]
                    if bool(terminated) or bool(truncated) or success:
                        break
            maximum = max(values) if values else float("nan")
            maxima.append(maximum)
            rows.append({"seed": args.start_seed + episode, "steps": executed, "success": bool(success), "max_score": maximum, "score_count": len(values)})
            print(json.dumps(rows[-1]), flush=True)
    finally:
        env.close()
    finite = [value for value in maxima if np.isfinite(value)]
    if len(finite) < max(2, args.episodes - 1):
        raise RuntimeError(f"too many non-finite calibration trajectories: {len(finite)}/{args.episodes}")
    rank = min(len(finite), math.ceil((len(finite) + 1) * 0.95))
    threshold = sorted(finite)[rank - 1]
    result = {
        "format": "stackpyramid_bridge_pca_calibration_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "asset": str(args.asset.resolve()),
        "q": 0.95,
        "episodes": len(rows),
        "successful_id_rollouts": sum(int(row["success"]) for row in rows),
        "threshold": float(threshold),
        "trajectory_maxima": maxima,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
