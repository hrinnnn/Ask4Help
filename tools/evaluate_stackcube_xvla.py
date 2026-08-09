#!/usr/bin/env python3
"""Evaluate an X-VLA checkpoint on the controlled StackCube task."""

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

from rlinf.envs.maniskill.stack_cube_variants import (  # noqa: E402
    STACK_CUBE_ID_ENV_ID,
    STACK_CUBE_OOD_ENV_ID,
    STACK_CUBE_TASK,
    register_controlled_stack_cube_variants,
    reset_metadata,
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
    return np.clip(array[:execute_horizon, : low.size], low, high).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--split", choices=("id", "ood"), default="id")
    parser.add_argument("--execute-horizon", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=100)
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
    register_controlled_stack_cube_variants()
    env = gym.make(
        STACK_CUBE_ID_ENV_ID if args.split == "id" else STACK_CUBE_OOD_ENV_ID,
        robot_uids="panda_wristcam",
        num_envs=1,
        obs_mode="rgb",
        control_mode="pd_joint_delta_pos",
        reward_mode="sparse",
        render_mode="rgb_array",
        sim_backend="physx_cpu",
        sim_config={"sim_freq": 100, "control_freq": 10},
        sensor_configs={"width": 384, "height": 384},
        max_episode_steps=args.max_episode_steps,
    )
    low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    rows: list[dict[str, Any]] = []
    try:
        for episode_index in range(args.episodes):
            seed = args.seed + episode_index
            raw_obs, _ = env.reset(seed=seed)
            metadata = reset_metadata(env, split=args.split)
            records, actions = [_extract_record(raw_obs)], []
            success = grasped = on_cube = static = False
            decision = 0
            while len(actions) < args.max_episode_steps and not success:
                predicted, _, _, _ = policy.predict(
                    raw_obs,
                    STACK_CUBE_TASK,
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
                    grasped |= bool_scalar(info.get("is_cubeA_grasped", False))
                    on_cube |= bool_scalar(info.get("is_cubeA_on_cubeB", False))
                    static |= bool_scalar(info.get("is_cubeA_static", False))
                    success = bool_scalar(info.get("success", False))
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
                task=STACK_CUBE_TASK,
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
                "success": success,
                "grasped_once": grasped,
                "on_cube_once": on_cube,
                "static_once": static,
                "steps": len(actions),
                "video": str(video),
                "actions": str(actions_path),
                **metadata,
            }
            rows.append(row)
            print(
                f"[xvla-stackcube] {episode_index + 1}/{args.episodes} seed={seed} "
                f"success={int(success)} cumulative={sum(int(x['success']) for x in rows)}/{len(rows)}",
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
        "grasp_rate": float(np.mean([row["grasped_once"] for row in rows])),
        "on_cube_rate": float(np.mean([row["on_cube_once"] for row in rows])),
        "execute_horizon": args.execute_horizon,
        "max_episode_steps": args.max_episode_steps,
        "flow_steps": args.flow_steps,
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
