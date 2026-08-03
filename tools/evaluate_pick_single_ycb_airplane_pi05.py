#!/usr/bin/env python3
"""Evaluate an SFT pi0.5 checkpoint on the controlled PickSingleYCB airplane.

The evaluator keeps the policy interface identical to ID SFT: two RGB views,
Panda qpos proprioception, and 8-D ``pd_joint_delta_pos`` actions.  It stores
every rollout's reset metadata, executed actions, and a durable dual-view
video so detector experiments can reuse the exact policy traces later.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RLINF_ROOT = ROOT / "RLinf"
sys.path[:0] = [str(ROOT), str(RLINF_ROOT)]

from rlinf.envs.maniskill.pick_single_ycb_airplane_variants import (  # noqa: E402
    PICK_SINGLE_YCB_AIRPLANE_TASK,
    register_controlled_pick_single_ycb_airplane_variants,
    reset_metadata,
)
from toolkits.lerobot.collect_maniskill_pick_single_ycb_airplane_lerobot import (  # noqa: E402
    _build_env,
    controlled_env_id,
    write_episode_video_durably,
)
from toolkits.lerobot.collect_maniskill_peg_lerobot_joint import (  # noqa: E402
    MAIN_CAMERA_CANDIDATES,
    WRIST_CAMERA_CANDIDATES,
    _bool_scalar,
    _build_frames,
    _extract_record,
    _select_camera,
)
from tools.maniskill_pi05_vfd_online_awbc import _load_model  # noqa: E402
from tools.pick_single_ycb_airplane_eval_common import clip_action_chunk  # noqa: E402


def model_observation(raw_obs: dict[str, Any]) -> dict[str, Any]:
    """Construct the exact two-view, proprioceptive OpenPI policy input."""

    sensors = raw_obs["sensor_data"]
    states = raw_obs["agent"]["qpos"]
    return {
        "main_images": sensors["base_camera"]["rgb"],
        "wrist_images": sensors["hand_camera"]["rgb"],
        "extra_view_images": None,
        "states": states,
        "task_descriptions": [PICK_SINGLE_YCB_AIRPLANE_TASK],
        "task_ids": torch.zeros(1, dtype=torch.long, device=states.device),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pi05-base", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("id", "ood"), default="id")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=30000)
    parser.add_argument("--execute-horizon", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--control-freq", type=int, default=10)
    parser.add_argument("--sim-backend", choices=("physx_cpu", "gpu"), default="physx_cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0 or args.execute_horizon <= 0:
        raise ValueError("episodes and execute-horizon must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    action_dir = args.output_dir / "actions"
    action_dir.mkdir()

    import gymnasium as gym  # noqa: F401
    import mani_skill.envs  # noqa: F401

    register_controlled_pick_single_ycb_airplane_variants()
    model = _load_model(args.checkpoint, args.norm_stats, args.pi05_base)
    env_args = argparse.Namespace(
        split=args.split,
        image_size=args.image_size,
        control_freq=args.control_freq,
        max_episode_steps=args.max_episode_steps,
        sim_backend=args.sim_backend,
    )
    env = _build_env(env_args, control_mode="pd_joint_delta_pos")
    low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    rows: list[dict[str, Any]] = []
    try:
        for episode_index in range(args.episodes):
            seed = args.seed + episode_index
            raw_obs, _info = env.reset(seed=seed)
            metadata = reset_metadata(env, split=args.split)
            records, executed = [_extract_record(raw_obs)], []
            success = False
            while len(executed) < args.max_episode_steps and not success:
                with torch.inference_mode():
                    predicted, _ = model.predict_action_batch(
                        env_obs=model_observation(raw_obs), mode="eval", compute_values=False
                    )
                chunk = clip_action_chunk(
                    predicted.detach().float().cpu().numpy(), low, high, args.execute_horizon
                )
                for action in chunk:
                    raw_obs, _reward, terminated, truncated, info = env.step(
                        torch.as_tensor(action, device=env.unwrapped.device).unsqueeze(0)
                    )
                    executed.append(action.copy())
                    records.append(_extract_record(raw_obs))
                    success = _bool_scalar(info.get("success"))
                    if success or _bool_scalar(terminated) or _bool_scalar(truncated):
                        break

            main_camera = _select_camera(records[0].obs, "", ("base_camera",) + MAIN_CAMERA_CANDIDATES, "main")
            wrist_camera = _select_camera(records[0].obs, "", ("hand_camera",) + WRIST_CAMERA_CANDIDATES, "wrist")
            frames = _build_frames(
                records=records,
                actions=executed,
                task=PICK_SINGLE_YCB_AIRPLANE_TASK,
                main_camera=main_camera,
                wrist_camera=wrist_camera,
            )
            video_path = write_episode_video_durably(
                frames, video_dir=args.output_dir / "videos", episode_index=episode_index, seed=seed, fps=args.control_freq
            )
            actions_path = action_dir / f"episode_{episode_index:06d}_seed_{seed:06d}.npy"
            np.save(actions_path, np.asarray(executed, dtype=np.float32))
            row = {
                "episode_index": episode_index,
                "seed": seed,
                "success": bool(success),
                "steps": len(executed),
                "video": str(video_path),
                "actions": str(actions_path),
                **metadata,
            }
            rows.append(row)
            print(
                f"[rollout] split={args.split} episode={episode_index + 1}/{args.episodes} "
                f"seed={seed} success={int(success)} cumulative={sum(int(row['success']) for row in rows)}/{len(rows)}",
                flush=True,
            )
    finally:
        env.close()

    summary = {
        "checkpoint": str(args.checkpoint),
        "pi05_base": str(args.pi05_base),
        "norm_stats": str(args.norm_stats),
        "split": args.split,
        "episodes": len(rows),
        "successes": sum(int(row["success"]) for row in rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "execute_horizon": args.execute_horizon,
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
