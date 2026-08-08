#!/usr/bin/env python3
"""Calibrate PCA and DiffDAgger gates on successful ID policy rollouts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "RLinf")]

from rlinf.algorithms.vla_fail import (  # noqa: E402
    PCAResidualStatistics,
    pca_residual_score,
)
from rlinf.envs.maniskill.pick_single_ycb_airplane_variants import (  # noqa: E402
    PICK_SINGLE_YCB_AIRPLANE_TASK,
    register_controlled_pick_single_ycb_airplane_variants,
)
from toolkits.lerobot.collect_maniskill_pick_single_ycb_airplane_lerobot import (  # noqa: E402
    _build_env,
)
from toolkits.lerobot.collect_maniskill_peg_lerobot_joint import _bool_scalar  # noqa: E402
from tools.pick_single_ycb_airplane_eval_common import clip_action_chunk  # noqa: E402
from tools.xvla_airplane_runtime import XVLAAirplanePolicy  # noqa: E402


def _higher_quantile(values: list[float], quantile: float) -> float:
    array = np.sort(np.asarray(values, dtype=np.float64))
    index = min(int(np.ceil(quantile * len(array))) - 1, len(array) - 1)
    return float(array[max(0, index)])


def _patience_episode_score(scores: list[float], patience: int) -> float:
    return max(
        min(scores[index : index + patience])
        for index in range(len(scores) - patience + 1)
    )


def _grasped(env: Any) -> bool:
    return _bool_scalar(env.unwrapped.agent.is_grasping(env.unwrapped.obj))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--pca-asset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--successful-episodes", type=int, default=20)
    parser.add_argument("--max-attempts", type=int, default=50)
    parser.add_argument("--seed", type=int, default=51000)
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--max-episode-steps", type=int, default=150)
    parser.add_argument("--execute-horizon", type=int, default=5)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--diff-timesteps", type=int, default=16)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    asset = torch.load(args.pca_asset, map_location="cpu", weights_only=False)
    statistics = PCAResidualStatistics.from_state_dict(asset["statistics"])
    policy = XVLAAirplanePolicy(args.checkpoint, args.xvla_root)

    import gymnasium as gym  # noqa: F401,E402
    import mani_skill.envs  # noqa: F401,E402

    register_controlled_pick_single_ycb_airplane_variants()
    env_args = argparse.Namespace(
        split="id",
        image_size=384,
        control_freq=10,
        max_episode_steps=args.max_episode_steps,
        sim_backend="physx_cpu",
    )
    env = _build_env(env_args, control_mode="pd_joint_delta_pos")
    low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    accepted: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    try:
        for attempt_index in range(args.max_attempts):
            if len(accepted) >= args.successful_episodes:
                break
            seed = args.seed + attempt_index
            raw_obs, _ = env.reset(seed=seed)
            pca_scores: list[float] = []
            diff_scores: list[float] = []
            steps = 0
            ever_grasped = False
            while steps < args.max_episode_steps:
                actions, feature, inputs, encoding = policy.predict(
                    raw_obs,
                    PICK_SINGLE_YCB_AIRPLANE_TASK,
                    seed=seed * 1000 + steps,
                    steps=args.flow_steps,
                )
                pca_scores.append(
                    float(pca_residual_score(feature.unsqueeze(1), statistics)[0])
                )
                diff_scores.append(
                    policy.diffdagger_score(
                        inputs,
                        encoding,
                        actions,
                        num_timesteps=args.diff_timesteps,
                        num_noise_samples=1,
                    )
                )
                chunk = clip_action_chunk(actions, low, high, args.execute_horizon)
                stop = False
                for action in chunk:
                    raw_obs, _, terminated, truncated, _ = env.step(
                        torch.as_tensor(action, device=env.unwrapped.device).unsqueeze(0)
                    )
                    steps += 1
                    ever_grasped |= _grasped(env)
                    if _bool_scalar(terminated) or _bool_scalar(truncated):
                        stop = True
                        break
                if stop:
                    break
            row = {
                "attempt_index": attempt_index,
                "seed": seed,
                "ever_grasped": ever_grasped,
                "steps": steps,
                "pca_scores": pca_scores,
                "diff_scores": diff_scores,
            }
            attempts.append(row)
            if ever_grasped and len(diff_scores) >= args.patience:
                accepted.append(row)
            print(
                f"[xvla-calibration] attempt={attempt_index + 1} "
                f"accepted={len(accepted)}/{args.successful_episodes}",
                flush=True,
            )
    finally:
        env.close()

    if len(accepted) != args.successful_episodes:
        raise RuntimeError(f"only collected {len(accepted)} successful ID trajectories")
    pca_episode_scores = [max(row["pca_scores"]) for row in accepted]
    diff_episode_scores = [
        _patience_episode_score(row["diff_scores"], args.patience) for row in accepted
    ]
    payload = {
        "format": "xvla_airplane_ood_gate_calibration_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "pca_asset": str(args.pca_asset.resolve()),
        "success_label": "ever_grasped",
        "successful_episodes": len(accepted),
        "quantile": args.quantile,
        "diff_patience": args.patience,
        "pca_threshold": _higher_quantile(pca_episode_scores, args.quantile),
        "diff_threshold": _higher_quantile(diff_episode_scores, args.quantile),
        "pca_episode_scores": pca_episode_scores,
        "diff_episode_scores": diff_episode_scores,
        "attempts": attempts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "attempts"}, indent=2))


if __name__ == "__main__":
    main()
