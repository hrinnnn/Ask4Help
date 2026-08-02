#!/usr/bin/env python3
"""Calibrate the StackCube Flow-SDE DiffDAgger gate on successful ID rollouts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "RLinf")]

from rlinf.algorithms.diffdagger import EmpiricalCDF  # noqa: E402
from tools.collect_stackcube_gated_dagger import (  # noqa: E402
    EXECUTE_HORIZON,
    _diffdagger_score,
    _policy_prediction,
)
from tools.maniskill_pi05_vfd_online_awbc import _bool, _build_env, _load_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pi05-base", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--successful-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--max-attempts", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--num-timesteps", type=int, default=16)
    parser.add_argument("--num-noise-samples", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.successful_episodes <= 0 or args.max_attempts < args.successful_episodes:
        raise ValueError("invalid successful-episodes/max-attempts")
    model = _load_model(args.checkpoint, args.norm_stats, args.pi05_base)
    env = _build_env(100, task="stack", split="id")
    scores: list[float] = []
    episodes: list[dict[str, object]] = []
    try:
        for attempt in range(args.max_attempts):
            if len(episodes) >= args.successful_episodes:
                break
            seed = args.seed + attempt
            raw_obs, info = env.reset(seed=seed)
            episode_scores: list[float] = []
            success = terminated = truncated = False
            step = 0
            while step < 100 and not (success or terminated or truncated):
                candidate, prediction = _policy_prediction(
                    model, raw_obs, info, seed=seed, step=step
                )
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    torch.manual_seed(seed * 10_000 + step)
                    torch.cuda.manual_seed_all(seed * 10_000 + step)
                    score = _diffdagger_score(
                        model, raw_obs, info, prediction,
                        num_timesteps=args.num_timesteps,
                        num_noise_samples=args.num_noise_samples,
                    )
                episode_scores.append(score)
                for action in candidate:
                    raw_obs, _reward, terminated, truncated, info = env.step(
                        torch.as_tensor(action, device=env.unwrapped.device).unsqueeze(0)
                    )
                    step += 1
                    success = _bool(info.get("success", False))
                    if success or _bool(terminated) or _bool(truncated):
                        break
            print(
                f"[calibration] attempt={attempt + 1} seed={seed} success={int(success)} "
                f"chunks={len(episode_scores)} accepted={len(episodes) + int(success)}/{args.successful_episodes}",
                flush=True,
            )
            if success and episode_scores:
                scores.extend(episode_scores)
                episodes.append({"seed": seed, "chunks": len(episode_scores), "scores": episode_scores})
    finally:
        env.close()
        del model
        torch.cuda.empty_cache()

    if len(episodes) != args.successful_episodes:
        raise RuntimeError(f"only {len(episodes)}/{args.successful_episodes} successful ID episodes")
    cdf = EmpiricalCDF(scores)
    payload = {
        "method": "flow_sde_diffdagger",
        "alpha": args.alpha,
        "threshold": cdf.quantile(args.alpha),
        "num_timesteps": args.num_timesteps,
        "num_noise_samples": args.num_noise_samples,
        "successful_id_episodes": len(episodes),
        "scores": scores,
        "episodes": episodes,
        "checkpoint": str(args.checkpoint),
        "norm_stats": str(args.norm_stats),
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[calibration] threshold={payload['threshold']:.8f} scores={len(scores)}", flush=True)


if __name__ == "__main__":
    main()
