#!/usr/bin/env python3
"""Collect and audit Panda Oracle trajectories for the basket task."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import gymnasium as gym
import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from panda_vegetable_basket_adapter import (
    action_space_gripper_bounds,
    encode_base_ee6d,
    target_world_pose_to_panda_action,
    tcp_pose_world,
    world_pose_to_base,
)


TASK = "put the vegetable into the yellow basket"
ENV_IDS = {
    "id": "XVLAPandaPutVegetableInBasketID-v1",
    "ood": "XVLAPandaPutVegetableInBasketOOD-v1",
}


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _pose(actor):
    return _array(actor.pose.raw_pose).reshape(-1, 7)[0].astype(np.float32)


def _rgb(obs):
    value = _array(obs["sensor_data"]["3rd_view_camera"]["rgb"])
    return value[0].astype(np.uint8) if value.ndim == 4 else value.astype(np.uint8)


def _load_task_module(path: Path) -> None:
    spec = importlib.util.spec_from_file_location("panda_vegetable_basket_variants", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load task module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def _is_true(value) -> bool:
    return bool(_array(value).reshape(-1)[0])


def _record_action(
    env,
    desired_xyz,
    desired_quaternion,
    grip,
    frames,
    proprio,
    actions,
    source_states,
    target_states,
):
    current = tcp_pose_world(env)
    target_world = np.concatenate(
        [np.asarray(desired_xyz, dtype=np.float32), np.asarray(desired_quaternion, dtype=np.float32)]
    )
    current_base = world_pose_to_base(env, current)
    target_base = world_pose_to_base(env, target_world)
    frames.append(_rgb(env._last_obs))
    proprio.append(encode_base_ee6d(current_base, float(grip > 0.0))[:10])
    actions.append(encode_base_ee6d(target_base, float(grip > 0.0))[:10])
    base = env.unwrapped
    source_states.append(_pose(base.objs[base.source_obj_name]))
    target_states.append(_pose(base.objs[base.target_obj_name]))
    return target_world_pose_to_panda_action(env, desired_xyz, desired_quaternion, grip)


def _save_attempt(output, episode_index, frames, proprio, actions, source_states, target_states, metadata):
    data_dir = output / "data"
    video_dir = output / "videos"
    meta_dir = output / "metadata"
    data_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    stem = f"episode_{episode_index:06d}"
    h5_path = data_dir / f"{stem}.h5"
    video_path = video_dir / f"{stem}.mp4"
    meta_path = meta_dir / f"{stem}.json"
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("proprio", data=np.asarray(proprio, dtype=np.float32))
        h5.create_dataset("abs_action_6d", data=np.asarray(actions, dtype=np.float32))
        h5.create_dataset("object_pose", data=np.asarray(source_states, dtype=np.float32))
        h5.create_dataset("target_pose", data=np.asarray(target_states, dtype=np.float32))
        encoded = []
        for frame in frames:
            ok, data = cv2.imencode(
                ".jpg",
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 90],
            )
            if not ok:
                raise RuntimeError("failed to encode RGB frame")
            encoded.append(np.frombuffer(data.tobytes(), dtype=np.uint8))
        image_dtype = h5py.vlen_dtype(np.dtype("uint8"))
        images = h5.create_dataset("images", shape=(len(encoded),), dtype=image_dtype)
        for index, item in enumerate(encoded):
            images[index] = item
        h5.attrs["seed"] = int(metadata["seed"])
        h5.attrs["success"] = bool(metadata["success"])
        h5.attrs["split"] = metadata["split"]
        h5.attrs["robot"] = metadata["robot"]
    if not frames:
        raise RuntimeError("attempt has no RGB frames")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open {video_path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    metadata = dict(metadata)
    metadata.update(
        {
            "data_path": str(h5_path),
            "video_path": str(video_path),
            "metadata_path": str(meta_path),
            "num_frames": len(frames),
            "num_actions": len(actions),
        }
    )
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return h5_path, video_path, meta_path


def run_episode(env, split, seed, episode_index, output, close_offset, orientation_yaw_deg):
    obs, _ = env.reset(seed=int(seed))
    env._last_obs = obs
    base = env.unwrapped
    source = base.source_obj_name
    target = base.target_obj_name
    start_source = _pose(base.objs[source])
    start_target = _pose(base.objs[target])
    initial_tcp = tcp_pose_world(env)
    object_rotation = Rotation.from_quat(start_source[3:][[1, 2, 3, 0]])
    closing_axis = object_rotation.apply(np.array([0.0, 1.0, 0.0]))
    closing_axis[2] = 0.0
    closing_axis /= max(float(np.linalg.norm(closing_axis)), 1e-8)
    grasp_pose = base.agent.build_grasp_pose(
        np.array([0.0, 0.0, -1.0]), closing_axis, start_source[:3]
    )
    grasp_center = np.asarray(grasp_pose.p, dtype=np.float32)
    grasp_quaternion = np.asarray(grasp_pose.q, dtype=np.float32)
    if orientation_yaw_deg:
        grasp_rotation = Rotation.from_quat(grasp_quaternion[[1, 2, 3, 0]])
        grasp_rotation = grasp_rotation * Rotation.from_euler(
            "z", orientation_yaw_deg, degrees=True
        )
        grasp_quaternion = grasp_rotation.as_quat()[[3, 0, 1, 2]].astype(np.float32)
    source_center = grasp_center
    target_center = start_target[:3]
    opened, closed = action_space_gripper_bounds(env)[1], action_space_gripper_bounds(env)[0]
    frames, proprio, actions, source_states, target_states = [], [], [], [], []
    timeline = []
    grasp_steps = 0
    max_source_z = float(start_source[2])
    phases = [
        ("hover", source_center + np.array([0, 0, 0.16], dtype=np.float32), opened, 18),
        ("descend", source_center + np.array([0, 0, close_offset], dtype=np.float32), opened, 8),
        ("close", source_center + np.array([0, 0, close_offset], dtype=np.float32), closed, 8),
        ("lift", source_center + np.array([0, 0, 0.18], dtype=np.float32), closed, 12),
        ("transport", target_center + np.array([0, 0, 0.18], dtype=np.float32), closed, 14),
        ("lower", target_center + np.array([0, 0, 0.035], dtype=np.float32), closed, 8),
        ("release", target_center + np.array([0, 0, 0.035], dtype=np.float32), opened, 8),
    ]
    step_index = 0
    for phase, desired, grip, count in phases:
        phase_start = step_index
        for _ in range(count):
            command = _record_action(
                env,
                desired,
                grasp_quaternion,
                grip,
                frames,
                proprio,
                actions,
                source_states,
                target_states,
            )
            env._last_obs, _, terminated, truncated, info = env.step(command)
            step_index += 1
            source_pose = _pose(base.objs[source])
            max_source_z = max(max_source_z, float(source_pose[2]))
            grasped = _is_true(base.agent.is_grasping(base.objs[source]))
            grasp_steps += int(grasped)
            if phase not in {item["phase"] for item in timeline}:
                timeline.append({"phase": phase, "first_step": step_index})
            if _is_true(terminated) or _is_true(truncated):
                break
        if step_index > phase_start and timeline[-1]["phase"] == phase:
            timeline[-1]["last_step"] = step_index
        if _is_true(terminated) or _is_true(truncated):
            break
    final_info = base.evaluate()
    success = _is_true(final_info["success"])
    metadata = {
        "episode_index": episode_index,
        "split": split,
        "seed": int(seed),
        "robot": type(base.agent).__name__,
        "source_object": source,
        "target_object": target,
        "instruction": TASK,
        "configured_source_pose": _array(base.xyz_configs[0, 0]).tolist(),
        "configured_target_pose": _array(base.xyz_configs[0, 1]).tolist(),
        "start_source_pose": start_source.tolist(),
        "start_target_pose": start_target.tolist(),
        "final_source_pose": _pose(base.objs[source]).tolist(),
        "final_target_pose": _pose(base.objs[target]).tolist(),
        "success": success,
        "final_eval": {
            key: bool(_array(value).reshape(-1)[0])
            if _array(value).dtype == bool
            else float(_array(value).reshape(-1)[0])
            for key, value in final_info.items()
        },
        "max_source_z": max_source_z,
        "grasp_steps": grasp_steps,
        "close_offset": float(close_offset),
        "orientation_yaw_deg": float(orientation_yaw_deg),
        "grasp_center": grasp_center.tolist(),
        "grasp_closing_axis": closing_axis.tolist(),
        "grasp_pose_quaternion": grasp_quaternion.tolist(),
        "timeline": timeline,
        "expert_control_start": 0,
        "expert_control_end": len(actions),
        "action_contract": "base-frame absolute EE6D targets encoded in active 10D block",
    }
    return _save_attempt(output, episode_index, frames, proprio, actions, source_states, target_states, metadata), metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rlinf-root", type=Path, required=True)
    parser.add_argument("--task-module", type=Path, required=True)
    parser.add_argument("--split", choices=("id", "ood"), required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--close-offset", type=float, default=0.0)
    parser.add_argument("--orientation-yaw-deg", type=float, default=0.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.rlinf_root))
    _load_task_module(args.task_module)
    env = gym.make(
        ENV_IDS[args.split],
        obs_mode="rgb+segmentation",
        render_mode="rgb_array",
        sim_backend="physx_cpu",
    )
    rows = []
    try:
        for episode_index in range(args.episodes):
            _, metadata = run_episode(
                env,
                args.split,
                args.seed_start + episode_index,
                episode_index,
                args.output,
                args.close_offset,
                args.orientation_yaw_deg,
            )
            rows.append(metadata)
            print(json.dumps(metadata), flush=True)
    finally:
        env.close()
    summary = {
        "format": "xvla_panda_vegetable_basket_oracle_collection_v1",
        "split": args.split,
        "episodes": len(rows),
        "successes": sum(int(row["success"]) for row in rows),
        "videos": len(list((args.output / "videos").glob("*.mp4"))),
        "rows": rows,
        "strict_success_definition": "released object inside basket region, above target plane, static and not grasped",
    }
    (args.output / "episodes.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output / f"{args.split.upper()}_ORACLE_COMPLETE").write_text("complete\n", encoding="utf-8")
    if summary["successes"] != summary["episodes"]:
        raise SystemExit("ORACLE_COLLECTION_HAS_FAILURES")


if __name__ == "__main__":
    main()
