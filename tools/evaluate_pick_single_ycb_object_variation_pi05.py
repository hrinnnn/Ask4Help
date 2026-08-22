#!/usr/bin/env python3
"""Policy-only evaluator for the PickSingleYCB object-variation task."""

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

from rlinf.envs.maniskill.pick_single_ycb_object_variation import (  # noqa: E402
    PICK_SINGLE_YCB_OBJECT_ID_ENV_ID,
    PICK_SINGLE_YCB_OBJECT_OOD_ENV_ID,
    PICK_SINGLE_YCB_OBJECT_TASK,
    register_controlled_pick_single_ycb_object_variants,
    reset_metadata,
)
from toolkits.lerobot.collect_maniskill_peg_lerobot_joint import (  # noqa: E402
    _build_frames,
    _extract_record,
    _select_camera,
)
from tools.maniskill_pi05_vfd_online_awbc import _load_model  # noqa: E402
from tools.pick_single_ycb_airplane_eval_common import clip_action_chunk  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pi05-base", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("id", "ood"), default="id")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=19000)
    parser.add_argument("--execute-horizon", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=20)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "actions").mkdir()
    (args.output_dir / "videos").mkdir()

    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    import imageio.v2 as imageio

    register_controlled_pick_single_ycb_object_variants()
    env_id = PICK_SINGLE_YCB_OBJECT_ID_ENV_ID if args.split == "id" else PICK_SINGLE_YCB_OBJECT_OOD_ENV_ID
    model = _load_model(args.checkpoint, args.norm_stats, args.pi05_base)
    env = gym.make(
        env_id,
        num_envs=1,
        robot_uids="panda_wristcam",
        obs_mode="rgb",
        control_mode="pd_joint_delta_pos",
        reward_mode="sparse",
        sim_backend="physx_cpu",
        sim_config={"sim_freq": 100, "control_freq": 10},
        render_mode="rgb_array",
        max_episode_steps=args.max_episode_steps,
    )
    rows: list[dict[str, Any]] = []
    try:
        for episode_index in range(args.episodes):
            seed = args.seed + episode_index
            raw_obs, _ = env.reset(seed=seed)
            metadata = reset_metadata(env, split=args.split)
            records, executed = [_extract_record(raw_obs)], []
            success = False
            while len(executed) < args.max_episode_steps and not success:
                policy_obs = {
                    "main_images": raw_obs["sensor_data"]["base_camera"]["rgb"],
                    "wrist_images": raw_obs["sensor_data"]["hand_camera"]["rgb"],
                    "extra_view_images": None,
                    "states": raw_obs["agent"]["qpos"],
                    "task_descriptions": [PICK_SINGLE_YCB_OBJECT_TASK],
                    "task_ids": torch.zeros(1, dtype=torch.long, device=raw_obs["agent"]["qpos"].device),
                }
                with torch.inference_mode():
                    predicted, _ = model.predict_action_batch(env_obs=policy_obs, mode="eval", compute_values=False)
                chunk = clip_action_chunk(
                    predicted.detach().float().cpu().numpy(),
                    np.asarray(env.action_space.low, dtype=np.float32).reshape(-1),
                    np.asarray(env.action_space.high, dtype=np.float32).reshape(-1),
                    args.execute_horizon,
                )
                for action in chunk:
                    raw_obs, _reward, terminated, truncated, info = env.step(
                        torch.as_tensor(action, device=env.unwrapped.device).reshape(1, -1)
                    )
                    executed.append(np.asarray(action, dtype=np.float32))
                    records.append(_extract_record(raw_obs))
                    success = bool(np.asarray(info.get("success", False)).reshape(-1)[0])
                    if success or bool(np.asarray(terminated).reshape(-1)[0]) or bool(np.asarray(truncated).reshape(-1)[0]):
                        break
            frames = _build_frames(
                records=records,
                actions=executed,
                task=PICK_SINGLE_YCB_OBJECT_TASK,
                main_camera=_select_camera(records[0].obs, "base_camera", ("base_camera",), "main"),
                wrist_camera=_select_camera(records[0].obs, "hand_camera", ("hand_camera",), "wrist"),
            )
            video = args.output_dir / "videos" / f"episode_{episode_index:06d}_seed_{seed:06d}.mp4"
            writer = imageio.get_writer(video, format="FFMPEG", fps=10, codec="libx264", pixelformat="yuv420p")
            try:
                for frame in frames:
                    main = frame["image"]
                    wrist = frame["wrist_image"]
                    height = max(main.shape[0], wrist.shape[0])
                    if main.shape[0] != height:
                        main = np.pad(main, ((0, height - main.shape[0]), (0, 0), (0, 0)))
                    if wrist.shape[0] != height:
                        wrist = np.pad(wrist, ((0, height - wrist.shape[0]), (0, 0), (0, 0)))
                    writer.append_data(np.concatenate([main, np.full((height, 4, 3), 32, dtype=np.uint8), wrist], axis=1))
            finally:
                writer.close()
            action_path = args.output_dir / "actions" / f"episode_{episode_index:06d}_seed_{seed:06d}.npy"
            np.save(action_path, np.asarray(executed, dtype=np.float32))
            rows.append({"episode_index": episode_index, "seed": seed, "success": success, "steps": len(executed), "video": str(video), "actions": str(action_path), **metadata})
    finally:
        env.close()
    summary = {
        "format": "pick_single_ycb_object_variation_pi05_eval_v1",
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "episodes": len(rows),
        "successes": sum(int(row["success"]) for row in rows),
        "videos": len(list((args.output_dir / "videos").glob("*.mp4"))),
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "EVAL_COMPLETE").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

