#!/usr/bin/env python3
"""Evaluate an X-VLA checkpoint on UncoverSpherePlace split variants."""

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

from rlinf.envs.maniskill.uncover_sphere_place import (  # noqa: E402
    UNCOVER_ENV_IDS,
    register_uncover_sphere_place_variants,
)
from toolkits.lerobot.collect_maniskill_peg_lerobot_joint import (  # noqa: E402
    MAIN_CAMERA_CANDIDATES,
    WRIST_CAMERA_CANDIDATES,
    _build_frames,
    _extract_record,
    _select_camera,
)
from toolkits.lerobot.collect_maniskill_plug_lerobot_joint import (  # noqa: E402
    write_episode_video_durably,
)
from tools.xvla_airplane_runtime import XVLAAirplanePolicy  # noqa: E402


TASK = "uncover the sphere and place it in the bowl"
SPLITS = tuple(UNCOVER_ENV_IDS)


def bool_scalar(value: Any) -> bool:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return bool(np.asarray(value, dtype=bool).reshape(-1).any())


def clip_action_chunk(
    predicted: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    execute_horizon: int,
) -> np.ndarray:
    array = np.asarray(predicted, dtype=np.float32)
    if array.ndim == 3:
        if array.shape[0] != 1:
            raise ValueError(f"expected one rollout environment, got {array.shape}")
        array = array[0]
    if array.ndim != 2 or array.shape[1] < low.size:
        raise ValueError(f"invalid X-VLA action shape: {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("X-VLA produced non-finite actions")
    return np.clip(array[:execute_horizon, : low.size], low, high).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--split", choices=SPLITS, default="id")
    parser.add_argument("--execute-horizon", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=2500)
    parser.add_argument("--flow-steps", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "actions").mkdir()

    policy = XVLAAirplanePolicy(args.checkpoint, args.xvla_root)
    register_uncover_sphere_place_variants()
    env = gym.make(
        UNCOVER_ENV_IDS[args.split],
        robot_uids="panda_wristcam",
        num_envs=1,
        obs_mode="rgb",
        control_mode="pd_joint_delta_pos",
        reward_mode="sparse",
        render_mode="rgb_array",
        sim_backend="physx_cpu",
        sim_config={"sim_freq": 100, "control_freq": 10},
        sensor_configs={"width": 224, "height": 224},
        max_episode_steps=args.max_episode_steps,
    )
    low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    rows: list[dict[str, Any]] = []
    try:
        for episode_index in range(args.episodes):
            seed = args.seed + episode_index
            raw_obs, _ = env.reset(seed=seed)
            metadata = env.unwrapped.reset_metadata()
            records, actions = [_extract_record(raw_obs)], []
            success = mug_parked = sphere_grasped = sphere_in_bowl = False
            sphere_released = sphere_static = False
            decision = 0
            while len(actions) < args.max_episode_steps and not success:
                predicted, _, _, _ = policy.predict(
                    raw_obs,
                    TASK,
                    seed=seed * 1000 + decision,
                    steps=args.flow_steps,
                )
                decision += 1
                chunk = clip_action_chunk(predicted, low, high, args.execute_horizon)
                for action in chunk:
                    raw_obs, _, terminated, truncated, info = env.step(
                        torch.as_tensor(action, device=env.unwrapped.device).unsqueeze(0)
                    )
                    actions.append(action.copy())
                    records.append(_extract_record(raw_obs))
                    phase = env.unwrapped.evaluate()
                    mug_parked |= bool_scalar(phase["ever_mug_parked"])
                    sphere_grasped |= bool_scalar(phase["ever_sphere_grasped"])
                    sphere_in_bowl |= bool_scalar(phase["sphere_in_bowl"])
                    sphere_released = bool_scalar(phase["sphere_released"])
                    sphere_static = bool_scalar(phase["sphere_static"])
                    success = bool_scalar(phase["success"])
                    if success or bool_scalar(terminated) or bool_scalar(truncated):
                        break

            main_camera = _select_camera(
                records[0].obs, "", ("base_camera",) + MAIN_CAMERA_CANDIDATES, "main"
            )
            wrist_camera = _select_camera(
                records[0].obs, "", ("hand_camera",) + WRIST_CAMERA_CANDIDATES, "wrist"
            )
            frames = _build_frames(
                records=records,
                actions=actions,
                task=TASK,
                main_camera=main_camera,
                wrist_camera=wrist_camera,
            )
            video = write_episode_video_durably(
                frames,
                video_dir=args.output_dir / "videos",
                episode_index=episode_index,
                seed=seed,
                fps=10,
            )
            actions_path = args.output_dir / "actions" / f"episode_{episode_index:06d}.npy"
            np.save(actions_path, np.asarray(actions, dtype=np.float32))
            row = {
                "episode_index": episode_index,
                "seed": seed,
                "split": args.split,
                "success": bool(success),
                "ever_mug_parked": bool(mug_parked),
                "ever_sphere_grasped": bool(sphere_grasped),
                "sphere_in_bowl": bool(sphere_in_bowl),
                "sphere_released": bool(sphere_released),
                "sphere_static": bool(sphere_static),
                "steps": len(actions),
                "decisions": decision,
                "video": str(video),
                "actions": str(actions_path),
                **metadata,
            }
            rows.append(row)
            print(
                f"[xvla-uncover] {episode_index + 1}/{args.episodes} seed={seed} "
                f"success={int(success)} cumulative="
                f"{sum(int(x['success']) for x in rows)}/{len(rows)}",
                flush=True,
            )
    finally:
        env.close()

    summary = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "episodes": len(rows),
        "successes": sum(int(row["success"]) for row in rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "mug_parked_rate": float(np.mean([row["ever_mug_parked"] for row in rows])),
        "sphere_grasp_rate": float(np.mean([row["ever_sphere_grasped"] for row in rows])),
        "sphere_in_bowl_rate": float(np.mean([row["sphere_in_bowl"] for row in rows])),
        "execute_horizon": args.execute_horizon,
        "max_episode_steps": args.max_episode_steps,
        "flow_steps": args.flow_steps,
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
