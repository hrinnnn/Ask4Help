#!/usr/bin/env python3
"""Replay saved X-VLA actions and write real-robot-compatible gripper pose logs.

This diagnostic deliberately does not load a VLA checkpoint or any detector.  It
replays the recorded action arrays from a passive OOD rollout, checks the reset
metadata and first RGB frame against the source artifact, and stores only the
end-effector pose, gripper width, and simulator phase labels used for audit.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RLINF_ROOT = ROOT / "RLinf"
sys.path[:0] = [str(ROOT), str(RLINF_ROOT)]

from toolkits.lerobot.collect_maniskill_peg_lerobot_joint import (  # noqa: E402
    MAIN_CAMERA_CANDIDATES,
    WRIST_CAMERA_CANDIDATES,
    _camera_image,
    _extract_record,
    _make_video_frame,
    _select_camera,
)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _bool_scalar(value: Any) -> bool:
    return bool(_to_numpy(value).reshape(-1)[0])


def _pose_snapshot(env: Any, *, task: str, info: dict[str, Any] | None = None) -> dict[str, Any]:
    base = env.unwrapped
    pose = base.agent.tcp.pose
    position = _to_numpy(pose.p).reshape(-1)[:3].astype(np.float32)
    quaternion_wxyz = _to_numpy(pose.q).reshape(-1)[:4].astype(np.float32)
    qpos = _to_numpy(base.agent.robot.get_qpos()).reshape(-1).astype(np.float32)
    if qpos.size < 9:
        raise RuntimeError(f"expected Panda qpos with >=9 values, got {qpos.shape}")
    width = float(qpos[-2:].sum())
    output: dict[str, Any] = {
        "position": position.tolist(),
        "quaternion_wxyz": quaternion_wxyz.tolist(),
        "gripper_width": width,
    }
    if task == "stackcube":
        output["grasped"] = bool(base.agent.is_grasping(base.cubeA))
        output["on_cube"] = bool(
            _bool_scalar((info or {}).get("is_cubeA_on_cubeB", False))
        )
        output["success"] = bool(_bool_scalar((info or {}).get("success", False)))
    else:
        output["grasped"] = bool(base.agent.is_grasping(base.obj))
        output["strict_success"] = bool(
            _bool_scalar((info or {}).get("success", False))
        )
    return output


def _build_env(task: str, split: str, *, image_size: int, max_episode_steps: int, control_freq: int, sim_backend: str):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    if task == "stackcube":
        from tools.stackcube_stage2_ood import register_stack_cube_splits, stack_cube_env_id

        register_stack_cube_splits()
        return gym.make(
            stack_cube_env_id(split),
            robot_uids="panda_wristcam",
            num_envs=1,
            obs_mode="rgb",
            control_mode="pd_joint_delta_pos",
            reward_mode="sparse",
            render_mode="rgb_array",
            sim_backend=sim_backend,
            sim_config={"sim_freq": 100, "control_freq": control_freq},
            sensor_configs={"width": image_size, "height": image_size},
            max_episode_steps=max_episode_steps,
        )

    from toolkits.lerobot.collect_maniskill_pick_single_ycb_airplane_lerobot import (
        _build_env as build_airplane_env,
    )

    args = SimpleNamespace(
        split=split,
        image_size=image_size,
        control_freq=control_freq,
        max_episode_steps=max_episode_steps,
        sim_backend=sim_backend,
    )
    return build_airplane_env(args, control_mode="pd_joint_delta_pos")


def _reset_metadata(env: Any, task: str, split: str) -> dict[str, Any]:
    if task == "stackcube":
        from tools.stackcube_stage2_ood import stack_cube_reset_metadata

        return stack_cube_reset_metadata(env, split=split)
    from rlinf.envs.maniskill.pick_single_ycb_airplane_variants import reset_metadata

    return reset_metadata(env, split=split)


def _metadata_max_error(expected: dict[str, Any], actual: dict[str, Any]) -> float:
    errors: list[float] = []

    def visit(left: Any, right: Any) -> None:
        if isinstance(left, dict) and isinstance(right, dict):
            for key, value in left.items():
                if key in right:
                    visit(value, right[key])
            return
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            if len(left) != len(right):
                errors.append(float("inf"))
                return
            for l_item, r_item in zip(left, right):
                visit(l_item, r_item)
            return
        if isinstance(left, (int, float, np.number)) and isinstance(right, (int, float, np.number)):
            errors.append(abs(float(left) - float(right)))

    visit(expected, actual)
    return max(errors, default=0.0)


def _source_first_frame(path: Path) -> np.ndarray:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "-",
    ]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE)
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(result.stdout)).convert("RGB"))


def _replay_frame(obs: dict[str, Any]) -> np.ndarray:
    record = _extract_record(obs)
    main = _select_camera(
        record.obs,
        "",
        ("base_camera",) + MAIN_CAMERA_CANDIDATES,
        "main",
    )
    wrist = _select_camera(
        record.obs,
        "",
        ("hand_camera",) + WRIST_CAMERA_CANDIDATES,
        "wrist",
    )
    return _make_video_frame(
        {"image": _camera_image(record.obs, main), "wrist_image": _camera_image(record.obs, wrist)}
    )


def _row_paths(row: dict[str, Any]) -> tuple[Path, Path]:
    action_path = Path(row["actions"])
    video_path = Path(row["video"])
    if not action_path.is_file():
        raise FileNotFoundError(action_path)
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    return action_path, video_path


def replay(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    rows = list(summary.get("rows", []))
    if not rows:
        raise ValueError(f"summary has no rows: {args.summary}")
    if args.limit is not None:
        rows = rows[: args.limit]
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    (args.output / "pose").mkdir(parents=True)

    env = _build_env(
        args.task,
        args.split,
        image_size=args.image_size,
        max_episode_steps=args.max_episode_steps,
        control_freq=args.control_freq,
        sim_backend=args.sim_backend,
    )
    output_rows: list[dict[str, Any]] = []
    try:
        for row in rows:
            action_path, video_path = _row_paths(row)
            actions = np.load(action_path).astype(np.float32)
            print(
                f"REPLAY_EPISODE_START task={args.task} "
                f"episode={int(row['episode_index'])} seed={int(row['seed'])}",
                flush=True,
            )
            raw_obs, _ = env.reset(seed=int(row["seed"]))
            actual_metadata = _reset_metadata(env, args.task, args.split)
            expected_metadata = {
                key: row[key]
                for key in actual_metadata
                if key in row
            }
            metadata_error = _metadata_max_error(expected_metadata, actual_metadata)
            replay_first = _replay_frame(raw_obs)
            source_first = _source_first_frame(video_path)
            source_shape = tuple(int(value) for value in source_first.shape)
            replay_shape = tuple(int(value) for value in replay_first.shape)
            if replay_first.shape != source_first.shape:
                from PIL import Image

                source_first = np.asarray(
                    Image.fromarray(source_first).resize(
                        (replay_first.shape[1], replay_first.shape[0]),
                        Image.Resampling.BILINEAR,
                    )
                )
            frame_delta = np.abs(
                replay_first.astype(np.int16) - source_first.astype(np.int16)
            )
            poses = [_pose_snapshot(env, task=args.task)]
            terminated = truncated = False
            for action in actions:
                if terminated or truncated:
                    break
                action_tensor = torch.as_tensor(
                    action,
                    dtype=torch.float32,
                    device=env.unwrapped.device,
                ).reshape(1, -1)
                raw_obs, _, terminated, truncated, info = env.step(action_tensor)
                poses.append(_pose_snapshot(env, task=args.task, info=info))

            seed = int(row["seed"])
            pose_array = np.asarray(
                [
                    [
                        *pose["position"],
                        *pose["quaternion_wxyz"],
                        pose["gripper_width"],
                        float(pose.get("grasped", False)),
                        float(pose.get("on_cube", pose.get("strict_success", False))),
                    ]
                    for pose in poses
                ],
                dtype=np.float32,
            )
            np.savez_compressed(
                args.output / "pose" / f"episode_{int(row['episode_index']):06d}_seed_{seed:06d}.npz",
                pose=pose_array,
                position=pose_array[:, :3],
                quaternion_wxyz=pose_array[:, 3:7],
                gripper_width=pose_array[:, 7],
                phase_flag_1=pose_array[:, 8],
                phase_flag_2=pose_array[:, 9],
            )
            source_steps = int(row.get("steps", len(actions)))
            replay_steps = len(poses) - 1
            output_rows.append(
                {
                    "episode_index": int(row["episode_index"]),
                    "seed": seed,
                    "source_steps": source_steps,
                    "action_array_steps": int(len(actions)),
                    "replay_steps": replay_steps,
                    "step_count_match": bool(source_steps == len(actions) == replay_steps),
                    "reset_metadata_max_abs_error": metadata_error,
                    "reset_metadata_match": bool(metadata_error <= args.metadata_tolerance),
                    "initial_rgb_mae": float(frame_delta.mean()),
                    "initial_rgb_max_abs_error": int(frame_delta.max()),
                    "initial_rgb_match": bool(frame_delta.mean() <= args.rgb_mae_tolerance),
                    "source_rgb_shape": list(source_shape),
                    "replay_rgb_shape": list(replay_shape),
                    "source_resized_for_comparison": bool(source_shape != replay_shape),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "pose_file": str(
                        args.output
                        / "pose"
                        / f"episode_{int(row['episode_index']):06d}_seed_{seed:06d}.npz"
                    ),
                }
            )
            print(
                f"REPLAY_EPISODE_COMPLETE task={args.task} "
                f"episode={int(row['episode_index'])} seed={seed} "
                f"steps={replay_steps} rgb_mae={float(frame_delta.mean()):.4f}",
                flush=True,
            )
    finally:
        env.close()

    payload = {
        "format": "xvla_pose_replay_audit_v1",
        "task": args.task,
        "split": args.split,
        "summary": str(args.summary),
        "episodes_requested": len(rows),
        "episodes_replayed": len(output_rows),
        "metadata_tolerance": args.metadata_tolerance,
        "rgb_mae_tolerance": args.rgb_mae_tolerance,
        "all_step_counts_match": all(row["step_count_match"] for row in output_rows),
        "all_reset_metadata_match": all(row["reset_metadata_match"] for row in output_rows),
        "all_initial_rgb_match": all(row["initial_rgb_match"] for row in output_rows),
        "rows": output_rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "replay_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("stackcube", "airplane"), required=True)
    parser.add_argument("--split", choices=("id", "ood"), required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--max-episode-steps", type=int, default=150)
    parser.add_argument("--control-freq", type=int, default=10)
    parser.add_argument("--sim-backend", choices=("physx_cpu", "gpu"), default="physx_cpu")
    parser.add_argument("--metadata-tolerance", type=float, default=1e-5)
    parser.add_argument("--rgb-mae-tolerance", type=float, default=5.0)
    args = parser.parse_args()
    payload = replay(args)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
