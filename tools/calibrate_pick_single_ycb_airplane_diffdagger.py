#!/usr/bin/env python3
"""Calibrate DiffDAgger on strict-success airplane ID policy rollouts only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RLINF_ROOT = ROOT / "RLinf"
sys.path[:0] = [str(ROOT), str(RLINF_ROOT)]

from rlinf.envs.maniskill.pick_single_ycb_airplane_variants import reset_metadata  # noqa: E402
from toolkits.lerobot.collect_maniskill_peg_lerobot_joint import _extract_record  # noqa: E402
from tools.collect_pick_single_ycb_airplane_gated_dagger import (  # noqa: E402
    EXECUTE_HORIZON,
    POLICY_HORIZON,
    _bool,
    _diff_score,
    _env_args,
    _policy_prediction,
    _save_raw_attempt,
)
from toolkits.lerobot.collect_maniskill_pick_single_ycb_airplane_lerobot import _build_env  # noqa: E402
from tools.maniskill_pi05_vfd_online_awbc import _load_model  # noqa: E402
from tools.pick_single_ycb_airplane_eval_common import clip_action_chunk  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pi05-base", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--successful-rollouts", type=int, default=20)
    parser.add_argument("--seed", type=int, default=76000)
    parser.add_argument("--max-attempts", type=int, default=1000)
    parser.add_argument("--num-timesteps", type=int, default=16)
    parser.add_argument("--num-noise-samples", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--control-freq", type=int, default=10)
    parser.add_argument("--sim-backend", choices=("physx_cpu", "gpu"), default="physx_cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    model = _load_model(args.checkpoint, args.norm_stats, args.pi05_base)
    env = _build_env(_env_args("id", args), control_mode="pd_joint_delta_pos")
    rows: list[dict] = []
    successful_scores: list[float] = []
    successful = 0
    try:
        for attempt_index in range(args.max_attempts):
            if successful >= args.successful_rollouts:
                break
            seed = args.seed + attempt_index
            raw_obs, _info = env.reset(seed=seed)
            metadata = reset_metadata(env, split="id")
            records = [_extract_record(raw_obs)]
            actions: list[np.ndarray] = []
            scores: list[float] = []
            success = False
            while len(actions) < POLICY_HORIZON and not success:
                predicted, model_actions = _policy_prediction(model, raw_obs, seed=seed, step=len(actions))
                scores.append(_diff_score(
                    model,
                    raw_obs,
                    model_actions,
                    num_timesteps=args.num_timesteps,
                    num_noise_samples=args.num_noise_samples,
                ))
                low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
                high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
                chunk = clip_action_chunk(predicted, low, high, int(model.config.action_horizon))
                for action in chunk[:EXECUTE_HORIZON]:
                    raw_obs, _reward, terminated, truncated, info = env.step(
                        torch.as_tensor(action, device=env.unwrapped.device).unsqueeze(0)
                    )
                    actions.append(np.asarray(action, dtype=np.float32))
                    records.append(_extract_record(raw_obs))
                    success = _bool(info.get("success", False))
                    if success or _bool(terminated) or _bool(truncated):
                        break
            video = _save_raw_attempt(
                output_dir=args.output_dir,
                episode_index=attempt_index,
                seed=seed,
                records=records,
                actions=actions,
                sources=["policy"] * len(actions),
                control_freq=args.control_freq,
            )
            if success:
                successful += 1
                successful_scores.extend(scores)
            row = {
                "attempt_index": attempt_index,
                "seed": seed,
                "success": success,
                "steps": len(actions),
                "scores": scores,
                "admitted_to_calibration": success,
                "video": video,
                **metadata,
            }
            rows.append(row)
            (args.output_dir / "episodes.jsonl").write_text(
                "".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8"
            )
            print(
                f"[airplane-diff-calibration] attempt={attempt_index + 1} "
                f"successful={successful}/{args.successful_rollouts} scores={len(successful_scores)}",
                flush=True,
            )
    finally:
        env.close()
        del model
        torch.cuda.empty_cache()

    if successful != args.successful_rollouts or not successful_scores:
        raise RuntimeError(
            f"collected {successful}/{args.successful_rollouts} strict-success ID rollouts"
        )
    payload = {
        "format": "pick_airplane_diffdagger_calibration_v1",
        "successful_rollouts": successful,
        "raw_attempts": len(rows),
        "num_timesteps": args.num_timesteps,
        "num_noise_samples": args.num_noise_samples,
        "scores": successful_scores,
        "q95": float(np.quantile(np.asarray(successful_scores), 0.95)),
    }
    (args.output_dir / "calibration.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
