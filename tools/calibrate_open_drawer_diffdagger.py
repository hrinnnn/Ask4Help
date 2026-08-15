#!/usr/bin/env python3
"""Calibrate OpenDrawer DiffDAgger flow uncertainty on ID policy rollouts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RLINF_ROOT = Path(os.environ.get("ASK4HELP_RLINF_ROOT", ROOT / "RLinf"))
sys.path[:0] = [str(ROOT), str(RLINF_ROOT)]

from rlinf.envs.maniskill.open_drawer_retrieve_place_spec import ENV_IDS, reset_metadata  # noqa: E402
from tools.collect_open_drawer_dagger import _build_split_env, _model_obs  # noqa: E402
from tools.maniskill_pi05_vfd_online_awbc import _load_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pi05-base", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=91000)
    parser.add_argument("--seed-list", type=Path)
    parser.add_argument("--execute-horizon", type=int, default=5)
    parser.add_argument("--max-policy-steps", type=int, default=240)
    parser.add_argument("--timesteps", type=int, default=16)
    parser.add_argument("--noise-samples", type=int, default=1)
    parser.add_argument("--sim-backend", choices=("physx_cpu", "gpu"), default="physx_cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.seed_list is not None:
        episode_seeds = [
            int(line.strip())
            for line in args.seed_list.read_text().splitlines()
            if line.strip()
        ]
        if not episode_seeds:
            raise ValueError(f"seed list is empty: {args.seed_list}")
    else:
        episode_seeds = [args.seed_start + index for index in range(args.num_episodes)]
    model = _load_model(args.checkpoint, args.norm_stats, args.pi05_base)
    scores: list[float] = []
    episodes: list[dict[str, object]] = []
    low = None
    high = None
    try:
        for episode, seed in enumerate(episode_seeds):
            env = _build_split_env("id", args, control_mode="pd_joint_delta_pos")
            try:
                raw_obs, _info = env.reset(seed=seed)
                reset = reset_metadata(env, split="id")
                if low is None:
                    low = np.asarray(env.action_space.low).reshape(-1)
                    high = np.asarray(env.action_space.high).reshape(-1)
                episode_scores: list[float] = []
                actions = 0
                ever_grasped = ever_lifted = ever_success = False
                terminated = truncated = False
                while actions < args.max_policy_steps:
                    env_obs = _model_obs(raw_obs)
                    with torch.inference_mode():
                        predicted, prediction_info = model.predict_action_batch(
                            env_obs=env_obs, mode="eval", compute_values=False
                        )
                        uncertainty = model.compute_diffdagger_uncertainty(
                            env_obs,
                            prediction_info["forward_inputs"]["model_action"],
                            num_timesteps=args.timesteps,
                            num_noise_samples=args.noise_samples,
                        )
                    score = float(uncertainty[0].item())
                    scores.append(score)
                    episode_scores.append(score)
                    chunk = np.clip(
                        predicted.detach().float().cpu().numpy()[0][:args.execute_horizon],
                        low,
                        high,
                    ).astype(np.float32)
                    for action in chunk:
                        raw_obs, _reward, terminated, truncated, info = env.step(
                            torch.as_tensor(action, device=env.unwrapped.device).unsqueeze(0)
                        )
                        actions += 1
                        ever_grasped |= bool(info.get("ever_grasped", False))
                        ever_lifted |= bool(info.get("ever_lifted", False))
                        ever_success |= bool(info.get("success", False))
                        if terminated or truncated or actions >= args.max_policy_steps:
                            break
                    if terminated or truncated:
                        break
                episodes.append({
                    "episode": episode,
                    "seed": seed,
                    "actions": actions,
                    "scores": episode_scores,
                    "max_score": max(episode_scores) if episode_scores else None,
                    "ever_grasped": ever_grasped,
                    "ever_lifted": ever_lifted,
                    "success": ever_success,
                    "reset": reset,
                })
                print(
                    f"[open-drawer-diff-calibration] episode={episode + 1}/{len(episode_seeds)} "
                    f"actions={actions} scores={len(episode_scores)} success={int(ever_success)}",
                    flush=True,
                )
            finally:
                env.close()
    finally:
        del model
        torch.cuda.empty_cache()
    payload = {
        "format": "open_drawer_diffdagger_calibration_v1",
        "checkpoint": str(args.checkpoint),
        "norm_stats": str(args.norm_stats),
        "split": "id",
        "num_episodes": len(episode_seeds),
        "seed_start": args.seed_start,
        "seeds": episode_seeds,
        "timesteps": args.timesteps,
        "noise_samples": args.noise_samples,
        "scores": scores,
        "episodes": episodes,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
