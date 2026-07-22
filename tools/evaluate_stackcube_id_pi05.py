#!/usr/bin/env python3
"""Evaluate an RLinf pi0.5 checkpoint on held-out controlled StackCube ID seeds."""

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
    STACK_CUBE_TASK,
    register_controlled_stack_cube_variant,
    reset_metadata,
)
from toolkits.lerobot.collect_maniskill_peg_lerobot_joint import (  # noqa: E402
    _build_frames,
    _extract_record,
    _select_camera,
    MAIN_CAMERA_CANDIDATES,
    WRIST_CAMERA_CANDIDATES,
)
from toolkits.lerobot.collect_maniskill_plug_lerobot_joint import (  # noqa: E402
    write_episode_video_durably,
)
from tools.maniskill_pi05_vfd_online_awbc import _load_model  # noqa: E402


def _bool(value: Any) -> bool:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return bool(np.asarray(value, dtype=bool).reshape(-1).any())


def _model_obs(raw_obs: dict[str, Any]) -> dict[str, Any]:
    sensor_data = raw_obs["sensor_data"]
    return {
        "main_images": sensor_data["base_camera"]["rgb"],
        "wrist_images": sensor_data["hand_camera"]["rgb"],
        "extra_view_images": None,
        "states": raw_obs["agent"]["qpos"],
        "task_descriptions": [STACK_CUBE_TASK],
        "task_ids": torch.zeros(1, dtype=torch.long, device=raw_obs["agent"]["qpos"].device),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pi05-base", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--execute-horizon", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = _load_model(args.checkpoint, args.norm_stats, args.pi05_base)
    register_controlled_stack_cube_variant()
    env = gym.make(
        STACK_CUBE_ID_ENV_ID,
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
    low = np.asarray(env.action_space.low).reshape(-1)
    high = np.asarray(env.action_space.high).reshape(-1)
    rows = []
    try:
        for episode in range(args.episodes):
            seed = args.seed + episode
            raw_obs, info = env.reset(seed=seed)
            metadata = reset_metadata(env)
            records = [_extract_record(raw_obs)]
            actions = []
            success = False
            grasped = on_cube = static = False
            while len(actions) < args.max_episode_steps and not success:
                with torch.inference_mode():
                    predicted, _ = model.predict_action_batch(
                        env_obs=_model_obs(raw_obs), mode="eval", compute_values=False
                    )
                chunk = predicted.detach().float().cpu().numpy()[0]
                chunk = np.clip(chunk[: args.execute_horizon], low, high).astype(np.float32)
                for action in chunk:
                    raw_obs, _reward, terminated, truncated, info = env.step(
                        torch.as_tensor(action, device=env.unwrapped.device).unsqueeze(0)
                    )
                    actions.append(action)
                    records.append(_extract_record(raw_obs))
                    grasped |= _bool(info.get("is_cubeA_grasped", False))
                    on_cube |= _bool(info.get("is_cubeA_on_cubeB", False))
                    static |= _bool(info.get("is_cubeA_static", False))
                    success = _bool(info.get("success", False))
                    if success or _bool(terminated) or _bool(truncated):
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
            write_episode_video_durably(
                frames,
                video_dir=args.output_dir / "videos",
                episode_index=episode,
                seed=seed,
                fps=10,
            )
            row = {
                "episode_index": episode,
                "seed": seed,
                "success": success,
                "grasped_once": grasped,
                "on_cube_once": on_cube,
                "static_once": static,
                "steps": len(actions),
                **metadata,
            }
            rows.append(row)
            print(
                f"[rollout] episode={episode + 1}/{args.episodes} seed={seed} "
                f"success={int(success)} cumulative={sum(int(r['success']) for r in rows)}/{len(rows)}",
                flush=True,
            )
    finally:
        env.close()
    summary = {
        "episodes": len(rows),
        "successes": sum(int(row["success"]) for row in rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "grasp_rate": float(np.mean([row["grasped_once"] for row in rows])),
        "on_cube_rate": float(np.mean([row["on_cube_once"] for row in rows])),
        "execute_horizon": args.execute_horizon,
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()

