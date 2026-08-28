#!/usr/bin/env python3
"""Collect policy-only OpenDrawer timelines for D-path timing calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from collect_open_drawer_fixed_timing import _bool, _build_env, _jsonable, _model_obs, _phase, _task_state
from rlinf.envs.maniskill.open_drawer_retrieve_place_spec import TASK_INSTRUCTION, reset_metadata, validate_reset_metadata
from toolkits.lerobot.collect_maniskill_peg_lerobot_joint import MAIN_CAMERA_CANDIDATES, WRIST_CAMERA_CANDIDATES, _build_frames, _camera_image, _extract_record, _select_camera
from toolkits.lerobot.collect_maniskill_pick_single_ycb_airplane_lerobot import write_episode_video_durably
from tools.maniskill_pi05_vfd_online_awbc import _load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pi05-base", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=78700)
    parser.add_argument("--max-episode-steps", type=int, default=400)
    parser.add_argument("--execute-horizon", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "episodes").mkdir()
    (args.output_root / "videos").mkdir()
    env = _build_env("grasp_ood", max_episode_steps=args.max_episode_steps, sim_backend="physx_cpu", image_size=384)
    model = _load_model(args.checkpoint, args.norm_stats, args.pi05_base)
    rows: list[dict[str, Any]] = []
    try:
        for episode in range(args.episodes):
            seed = args.seed + episode
            raw_obs, _info = env.reset(seed=seed)
            records = [_extract_record(raw_obs)]
            task_states = [_task_state(env)]
            main_camera = _select_camera(records[0].obs, "", ("base_camera",) + MAIN_CAMERA_CANDIDATES, "main")
            wrist_camera = _select_camera(records[0].obs, "", ("hand_camera",) + WRIST_CAMERA_CANDIDATES, "wrist")
            metadata = reset_metadata(env, split="grasp_ood")
            metadata["camera"] = {
                "main": main_camera,
                "wrist": wrist_camera,
                "main_shape": list(_camera_image(records[0].obs, main_camera).shape),
                "wrist_shape": list(_camera_image(records[0].obs, wrist_camera).shape),
                "requested_size": [384, 384],
            }
            errors = validate_reset_metadata(metadata, split="grasp_ood")
            if errors:
                raise RuntimeError(f"reset metadata failed for seed {seed}: {errors}")
            actions: list[np.ndarray] = []
            terminated = truncated = False
            while len(actions) < args.max_episode_steps and not (terminated or truncated):
                with torch.inference_mode():
                    predicted, _ = model.predict_action_batch(env_obs=_model_obs(raw_obs, TASK_INSTRUCTION), mode="eval", compute_values=False)
                chunk = predicted.detach().float().cpu().numpy()[0][: args.execute_horizon]
                low = np.asarray(env.action_space.low).reshape(-1)
                high = np.asarray(env.action_space.high).reshape(-1)
                for action in np.clip(chunk, low, high).astype(np.float32):
                    raw_obs, _reward, terminated, truncated, info = env.step(torch.as_tensor(action, device=env.unwrapped.device).unsqueeze(0))
                    actions.append(action)
                    records.append(_extract_record(raw_obs))
                    task_states.append(_task_state(env))
                    if terminated or truncated or _bool(info.get("success", False)):
                        break
            action_array = np.asarray(actions, dtype=np.float32)
            state_array = np.asarray([record.state for record in records], dtype=np.float32)
            episode_dir = args.output_root / "episodes" / f"episode_{episode:06d}"
            episode_dir.mkdir()
            np.save(episode_dir / "actions.npy", action_array)
            np.save(episode_dir / "states.npy", state_array)
            (episode_dir / "reset_metadata.json").write_text(json.dumps(_jsonable(metadata), indent=2) + "\n", encoding="utf-8")
            (episode_dir / "task_state_timeline.json").write_text(json.dumps({"format": "open_drawer_policy_only_task_state_timeline_v1", "seed": seed, "rows": [{"step": i, "phase": _phase(s), **s} for i, s in enumerate(task_states)]}, indent=2) + "\n", encoding="utf-8")
            frames = _build_frames(records=records, actions=list(action_array), task=TASK_INSTRUCTION, main_camera=main_camera, wrist_camera=wrist_camera)
            video = write_episode_video_durably(frames, video_dir=args.output_root / "videos", episode_index=episode, seed=seed, fps=10)
            (episode_dir / "timeline.json").write_text(json.dumps({"seed": seed, "steps": len(actions), "tail_observation_retained": len(records) == len(actions) + 1, "timeline": [{"step": i, "events": {"phase": _phase(s), "ever_drawer_opened": s["ever_drawer_opened"], "ever_grasped": s["ever_grasped"], "ever_lifted": s["ever_lifted"], "success": s["success"]}} for i, s in enumerate(task_states)]}, indent=2) + "\n", encoding="utf-8")
            rows.append({"episode_index": episode, "seed": seed, "split": "grasp_ood", "success": bool(task_states[-1]["success"]), "steps": len(actions), "video": str(video), "actions": str(episode_dir / "actions.npy"), "states": str(episode_dir / "states.npy"), "timeline": str(episode_dir / "timeline.json"), "task_state_timeline": str(episode_dir / "task_state_timeline.json"), "reset_metadata": str(episode_dir / "reset_metadata.json")})
            print(json.dumps({"episode": episode, "steps": len(actions), "success": bool(task_states[-1]["success"])}, flush=True))
    finally:
        env.close()
        del model
        torch.cuda.empty_cache()
    summary = {"format": "open_drawer_policy_only_timeline_v1", "task": "OpenDrawerRetrievePlace", "split": "grasp_ood", "episodes": len(rows), "successes": sum(int(r["success"]) for r in rows), "success_rate": float(np.mean([r["success"] for r in rows])), "rows": rows, "checkpoint": str(args.checkpoint), "max_episode_steps": args.max_episode_steps, "execute_horizon": args.execute_horizon}
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output_root / "POLICY_ONLY_COMPLETE").write_text("policy-only task-state timelines complete\n", encoding="utf-8")


if __name__ == "__main__":
    main()
