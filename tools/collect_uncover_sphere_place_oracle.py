#!/usr/bin/env python3
"""Collect replayable UncoverSpherePlace oracle demonstrations.

The collector records the official current-state Panda oracle directly in
``pd_joint_delta_pos`` mode. Every accepted episode keeps the complete
observation/action sequence, reset metadata, stage predicates, and a
side-by-side base/wrist video.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _load_modules(root: Path):
    package_root = root.parents[1]
    for name, path in (
        ("rlinf", package_root),
        ("rlinf.envs", package_root / "envs"),
        ("rlinf.envs.maniskill", root),
    ):
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module

    def load(name: str, filename: str):
        spec = importlib.util.spec_from_file_location(name, root / filename)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load {filename}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    env_module = load("rlinf.envs.maniskill.uncover_sphere_place", "uncover_sphere_place.py")
    load("rlinf.envs.maniskill.peg_privileged_oracle", "peg_privileged_oracle.py")
    oracle_module = load(
        "rlinf.envs.maniskill.uncover_sphere_place_privileged_oracle",
        "uncover_sphere_place_privileged_oracle.py",
    )
    return env_module, oracle_module


def _flag(value: Any) -> bool:
    return bool(value.reshape(-1)[0]) if hasattr(value, "reshape") else bool(value)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _run_episode(env: Any, oracle_module: Any, seed: int, max_chunks: int):
    from toolkits.lerobot.collect_maniskill_peg_lerobot_joint import _extract_record

    obs, _ = env.reset(seed=seed)
    records = [_extract_record(obs)]
    actions: list[np.ndarray] = []
    phases: list[str] = []
    oracle = oracle_module.UncoverSpherePlacePrivilegedChunkOracle(chunk_size=10)
    last_info: dict[str, Any] = {}
    terminated = truncated = False

    for _chunk in range(max_chunks):
        plan = oracle.plan(env)
        phases.append(plan.phase)
        if not plan.planning_succeeded:
            return None, {"seed": seed, "reason": "planner_failed", "phase": plan.phase}
        for step_index in range(10):
            qpos = env.unwrapped.agent.robot.get_qpos()
            action = plan.action_at(qpos, step_index)
            obs, _reward, terminated, truncated, last_info = env.step(
                torch.as_tensor(action, device=env.unwrapped.device).unsqueeze(0)
            )
            action_np = _to_numpy(action).reshape(-1).astype(np.float32)
            if not np.isfinite(action_np).all():
                return None, {"seed": seed, "reason": "nonfinite_action", "phase": plan.phase}
            actions.append(action_np)
            records.append(_extract_record(obs))
            info = env.unwrapped.evaluate()
            if _flag(info["success"]):
                break
            if _flag(terminated) or _flag(truncated):
                break
        if _flag(last_info.get("success", False)) or _flag(env.unwrapped.evaluate()["success"]):
            break
        if _flag(terminated) or _flag(truncated):
            break

    info = env.unwrapped.evaluate()
    row = {
        "seed": int(seed),
        "success": _flag(info["success"]),
        "steps": len(actions),
        "phases": phases,
        "final_predicates": {
            key: _flag(value)
            for key, value in info.items()
            if key in {
                "ever_mug_parked",
                "ever_sphere_grasped",
                "sphere_in_bowl",
                "sphere_released",
                "sphere_static",
                "success",
            }
        },
    }
    if not row["success"] or len(records) != len(actions) + 1:
        row["reason"] = "unsuccessful_or_misaligned"
        return None, row
    return (records, actions, row, dict(env.unwrapped.reset_metadata())), row


def _write_video(frames: list[dict[str, Any]], video_dir: Path, episode_index: int, seed: int, fps: int) -> Path:
    from toolkits.lerobot.collect_maniskill_plug_lerobot_joint import write_episode_video_durably

    return write_episode_video_durably(
        frames, video_dir=video_dir, episode_index=episode_index, seed=seed, fps=fps
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rlinf-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-id", required=True, type=Path)
    parser.add_argument("--split", choices=("id", "handle_ood", "goal_ood"), default="id")
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=64)
    parser.add_argument("--max-chunks", type=int, default=500)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()
    if args.output_dir.exists() or args.repo_id.exists():
        raise FileExistsError("refusing to overwrite an existing collection")

    sys.path.insert(0, str(args.rlinf_root.parent))
    sys.path.insert(0, str(args.rlinf_root))
    env_module, oracle_module = _load_modules(args.rlinf_root)
    env_module.register_uncover_sphere_place_variants()
    import gymnasium as gym

    env = gym.make(
        env_module.UNCOVER_ENV_IDS[args.split],
        obs_mode="rgb",
        control_mode="pd_joint_delta_pos",
        reward_mode="sparse",
        render_mode="rgb_array",
        sim_backend="physx_cpu",
        sensor_configs={"width": args.image_size, "height": args.image_size},
        max_episode_steps=5000,
    )
    args.output_dir.mkdir(parents=True)
    args.repo_id.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.joinpath("raw_attempts.jsonl").write_text("", encoding="utf-8")
    accepted_rows: list[dict[str, Any]] = []
    attempts = 0
    dataset = None
    main_camera = wrist_camera = ""
    try:
        while len(accepted_rows) < args.num_episodes and attempts < args.max_attempts:
            seed = args.seed + attempts
            attempts += 1
            result, attempt_row = _run_episode(env, oracle_module, seed, args.max_chunks)
            with args.output_dir.joinpath("raw_attempts.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(attempt_row) + "\n")
            if result is None:
                print(f"[uncover-collect] rejected seed={seed} reason={attempt_row.get('reason')}", flush=True)
                continue
            records, actions, row, metadata = result
            from toolkits.lerobot.collect_maniskill_peg_lerobot_joint import (
                _build_frames,
                _create_dataset,
                _select_camera,
                validate_visual_motion,
            )

            if not main_camera:
                main_camera = _select_camera(records[0].obs, "", ("base_camera",), "main")
                wrist_camera = _select_camera(records[0].obs, "", ("hand_camera",), "wrist")
            frames = _build_frames(
                records=records,
                actions=actions,
                task="uncover the sphere and place it in the bowl",
                main_camera=main_camera,
                wrist_camera=wrist_camera,
            )
            visual = validate_visual_motion(frames, min_peak_mean_abs_delta=1.0)
            if dataset is None:
                dataset = _create_dataset(
                    repo_id=str(args.repo_id),
                    image_shape=tuple(frames[0]["image"].shape),
                    wrist_image_shape=tuple(frames[0]["wrist_image"].shape),
                    fps=args.fps,
                    image_writer_threads=4,
                    image_writer_processes=0,
                )
            for frame in frames:
                dataset.add_frame(frame)
            dataset.save_episode()
            episode_index = len(accepted_rows)
            video = _write_video(
                frames, args.output_dir / "videos", episode_index, seed, args.fps
            )
            accepted_rows.append(
                {
                    **row,
                    **metadata,
                    "episode_index": episode_index,
                    "actions": len(actions),
                    "frames": len(records),
                    "visual_motion": visual,
                    "video_path": str(video),
                }
            )
            args.output_dir.joinpath("episodes.jsonl").write_text(
                "".join(json.dumps(item) + "\n" for item in accepted_rows), encoding="utf-8"
            )
            print(
                f"[uncover-collect] accepted={len(accepted_rows)}/{args.num_episodes} "
                f"seed={seed} steps={len(actions)}",
                flush=True,
            )
    finally:
        if dataset is not None and getattr(dataset, "image_writer", None) is not None:
            dataset.image_writer.wait_until_done()
        env.close()
    if len(accepted_rows) != args.num_episodes:
        raise RuntimeError(
            f"accepted {len(accepted_rows)}/{args.num_episodes} episodes after {attempts} attempts"
        )
    (args.output_dir / "collection_metadata.json").write_text(
        json.dumps(
            {
                "task": "UncoverSpherePlace",
                "split": args.split,
                "episodes": len(accepted_rows),
                "attempts": attempts,
                "dataset": str(args.repo_id),
                "camera_keys": {"main": main_camera, "wrist": wrist_camera},
                "action_shape": [8],
                "observation_state_shape": [9],
                "action_alignment": "frames == actions + 1; every frame carries the preceding action",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[uncover-collect] complete output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
