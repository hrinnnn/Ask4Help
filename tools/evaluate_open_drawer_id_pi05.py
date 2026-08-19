#!/usr/bin/env python3
"""Evaluate a frozen pi0.5 OpenDrawerRetrievePlace policy on fixed splits."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RLINF_ROOT = Path(os.environ.get("ASK4HELP_RLINF_ROOT", ROOT / "RLinf"))
sys.path[:0] = [str(ROOT), str(RLINF_ROOT)]

from rlinf.envs.maniskill.open_drawer_retrieve_place_spec import (  # noqa: E402
    ENV_IDS,
    TASK_INSTRUCTION,
    reset_metadata,
    validate_reset_metadata,
)
from toolkits.lerobot.collect_maniskill_peg_lerobot_joint import (  # noqa: E402
    MAIN_CAMERA_CANDIDATES,
    WRIST_CAMERA_CANDIDATES,
    _camera_image,
    _build_frames,
    _extract_record,
    _select_camera,
)
from toolkits.lerobot.collect_maniskill_plug_lerobot_joint import (  # noqa: E402
    write_episode_video_durably,
)
from tools.maniskill_pi05_vfd_online_awbc import _load_model  # noqa: E402


SPLITS = tuple(ENV_IDS)


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
        "task_descriptions": [TASK_INSTRUCTION],
        "task_ids": torch.zeros(
            1,
            dtype=torch.long,
            device=raw_obs["agent"]["qpos"].device,
        ),
    }


def _split_env_id(split: str) -> str:
    try:
        return ENV_IDS[split]
    except KeyError as exc:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}") from exc


def _camera_provenance(records: list[Any], main_camera: str, wrist_camera: str) -> dict[str, Any]:
    """Record the selected camera keys and actual RGB shapes from the reset observation."""

    main_image = _camera_image(records[0].obs, main_camera)
    wrist_image = _camera_image(records[0].obs, wrist_camera)
    if main_image is None or wrist_image is None:
        raise RuntimeError("reset observation is missing a selected base or wrist RGB image")
    return {
        "main": main_camera,
        "wrist": wrist_camera,
        "main_shape": list(main_image.shape),
        "wrist_shape": list(wrist_image.shape),
        "requested_size": [384, 384],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pi05-base", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", choices=SPLITS, default="id")
    parser.add_argument("--execute-horizon", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=400)
    return parser.parse_args()


def main() -> None:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    import rlinf.envs.maniskill.open_drawer_retrieve_place  # noqa: F401

    args = parse_args()
    if args.episodes <= 0 or args.execute_horizon <= 0 or args.max_episode_steps <= 0:
        raise ValueError("episodes, execute-horizon and max-episode-steps must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "videos").mkdir(exist_ok=True)
    (args.output_dir / "episodes").mkdir(exist_ok=True)
    (args.output_dir / "provenance.json").write_text(
        json.dumps(
            {
                "task": "OpenDrawerRetrievePlace",
                "checkpoint": str(args.checkpoint),
                "pi05_base": str(args.pi05_base),
                "norm_stats": str(args.norm_stats),
                "split": args.split,
                "seed_start": args.seed,
                "episodes": args.episodes,
                "execute_horizon": args.execute_horizon,
                "max_episode_steps": args.max_episode_steps,
                "success_predicate": "strict OpenDrawer task success",
                "evidence_format": "episodes/<episode>/actions.npy, states.npy, timeline.json",
                "reset_metadata_format": "env_id, instruction, split, control_mode, camera, physical ranges, lifecycle",
                "environment_policy": "one fresh environment instance per episode",
            },
            indent=2,
        )
        + "\n"
    )
    model = _load_model(args.checkpoint, args.norm_stats, args.pi05_base)
    rows: list[dict[str, Any]] = []
    for episode in range(args.episodes):
        env = gym.make(
            _split_env_id(args.split),
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
        try:
            low = np.asarray(env.action_space.low).reshape(-1)
            high = np.asarray(env.action_space.high).reshape(-1)
            seed = args.seed + episode
            raw_obs, info = env.reset(seed=seed)
            records = [_extract_record(raw_obs)]
            main_camera = _select_camera(
                records[0].obs, "", ("base_camera",) + MAIN_CAMERA_CANDIDATES, "main"
            )
            wrist_camera = _select_camera(
                records[0].obs, "", ("hand_camera",) + WRIST_CAMERA_CANDIDATES, "wrist"
            )
            reset_provenance = reset_metadata(env, split=args.split)
            reset_provenance["camera"] = _camera_provenance(records, main_camera, wrist_camera)
            reset_errors = validate_reset_metadata(reset_provenance, split=args.split)
            if reset_errors:
                raise RuntimeError(f"reset provenance failed for episode {episode}: {reset_errors}")
            actions: list[np.ndarray] = []
            event_timeline: list[dict[str, Any]] = [{"reset": True}]
            success = False
            ever_drawer_opened = False
            ever_grasped = False
            ever_lifted = False
            ever_in_target = False
            ever_released = False
            ever_static = False
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
                    event_timeline.append(
                        {
                            key: bool(_bool(value))
                            for key, value in info.items()
                            if key
                            in {
                                "ever_drawer_opened",
                                "ever_grasped",
                                "ever_lifted",
                                "object_in_target",
                                "object_released",
                                "is_robot_static",
                                "success",
                            }
                        }
                    )
                    ever_drawer_opened |= _bool(info.get("ever_drawer_opened", False))
                    ever_grasped |= _bool(info.get("ever_grasped", False))
                    ever_lifted |= _bool(info.get("ever_lifted", False))
                    ever_in_target |= _bool(info.get("object_in_target", False))
                    ever_released |= _bool(info.get("object_released", False))
                    ever_static |= _bool(info.get("is_robot_static", False))
                    success = _bool(info.get("success", False))
                    if success or _bool(terminated) or _bool(truncated):
                        break
            action_array = np.asarray(actions, dtype=np.float32)
            state_array = np.asarray([record.state for record in records], dtype=np.float32)
            if action_array.ndim != 2 or action_array.shape[1] != 8:
                raise RuntimeError(f"actions must be [T,8], got {action_array.shape}")
            if state_array.ndim != 2 or state_array.shape != (len(actions) + 1, 9):
                raise RuntimeError(f"states must be [T+1,9], got {state_array.shape}")
            episode_dir = args.output_dir / "episodes" / f"episode_{episode:06d}"
            episode_dir.mkdir(parents=True, exist_ok=True)
            np.save(episode_dir / "actions.npy", action_array)
            np.save(episode_dir / "states.npy", state_array)
            (episode_dir / "reset_metadata.json").write_text(
                json.dumps(reset_provenance, indent=2) + "\n"
            )
            timeline = [
                {
                    "step": index,
                    "state": np.asarray(record.state, dtype=np.float32).tolist(),
                    "events": event_timeline[index],
                }
                for index, record in enumerate(records)
            ]
            frames = _build_frames(
                records=records,
                actions=actions,
                task=TASK_INSTRUCTION,
                main_camera=main_camera,
                wrist_camera=wrist_camera,
            )
            video_path = write_episode_video_durably(
                frames,
                video_dir=args.output_dir / "videos",
                episode_index=episode,
                seed=seed,
                fps=10,
            )
            row = {
                "episode_index": episode,
                "seed": seed,
                "split": args.split,
                "success": bool(success),
                "ever_drawer_opened": bool(ever_drawer_opened),
                "ever_grasped": bool(ever_grasped),
                "ever_lifted": bool(ever_lifted),
                "ever_in_target": bool(ever_in_target),
                "ever_released": bool(ever_released),
                "ever_static": bool(ever_static),
                "steps": len(actions),
                "video": str(video_path),
                "actions": str(episode_dir / "actions.npy"),
                "states": str(episode_dir / "states.npy"),
                "timeline": str(episode_dir / "timeline.json"),
                "reset_metadata": str(episode_dir / "reset_metadata.json"),
                "env_id": reset_provenance["env_id"],
                "instruction": reset_provenance["instruction"],
                "control_mode": reset_provenance["control_mode"],
                "camera": reset_provenance["camera"],
                "lifecycle_at_reset": reset_provenance["lifecycle"],
            }
            (episode_dir / "timeline.json").write_text(
                json.dumps(
                    {
                        "episode_index": episode,
                        "seed": seed,
                        "steps": len(actions),
                        "tail_observation_retained": len(records) == len(actions) + 1,
                        "reset_metadata": reset_provenance,
                        "timeline": timeline,
                    },
                    indent=2,
                )
                + "\n"
            )
            rows.append(row)
            with (args.output_dir / "episodes.jsonl").open("a") as episodes_file:
                episodes_file.write(json.dumps(row, sort_keys=True) + "\n")
            print(
                f"[rollout] split={args.split} episode={episode + 1}/{args.episodes} "
                f"seed={seed} success={int(success)} "
                f"cumulative={sum(int(r['success']) for r in rows)}/{len(rows)}",
                flush=True,
            )
        finally:
            env.close()
    summary = {
        "task": "OpenDrawerRetrievePlace",
        "split": args.split,
        "episodes": len(rows),
        "successes": sum(int(row["success"]) for row in rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "drawer_opened_rate": float(np.mean([row["ever_drawer_opened"] for row in rows])),
        "grasp_rate": float(np.mean([row["ever_grasped"] for row in rows])),
        "lift_rate": float(np.mean([row["ever_lifted"] for row in rows])),
        "in_target_rate": float(np.mean([row["ever_in_target"] for row in rows])),
        "execute_horizon": args.execute_horizon,
        "max_episode_steps": args.max_episode_steps,
        "seed_start": args.seed,
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
