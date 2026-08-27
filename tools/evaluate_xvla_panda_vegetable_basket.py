#!/usr/bin/env python3
"""Evaluate a Panda X-VLA checkpoint with durable per-episode evidence."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import torch
from PIL import Image

from panda_vegetable_basket_adapter import (
    encode_base_ee6d,
    model_target_to_panda_action,
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


def _bool(value) -> bool:
    return bool(_array(value).reshape(-1)[0])


def _pose(actor) -> np.ndarray:
    return _array(actor.pose.raw_pose).reshape(-1, 7)[0].astype(np.float32)


def _rgb(obs) -> np.ndarray:
    value = _array(obs["sensor_data"]["3rd_view_camera"]["rgb"])
    return value[0].astype(np.uint8) if value.ndim == 4 else value.astype(np.uint8)


def _load_task_module(path: Path) -> None:
    spec = importlib.util.spec_from_file_location("panda_vegetable_basket_variants", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load task module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def _current_gripper_01(env) -> float:
    qpos = _array(env.unwrapped.agent.robot.get_qpos()).reshape(-1)
    width = float(np.mean(qpos[-2:]))
    return float(np.clip((width + 0.01) / 0.05, 0.0, 1.0))


def _save_video(path: Path, frames: list[np.ndarray]) -> None:
    if not frames:
        raise RuntimeError("cannot save empty video")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video {path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def _eval_one(model, processor, env, split: str, seed: int, episode_index: int, output: Path, domain_id: int, flow_steps: int, execute_horizon: int, max_steps: int) -> dict:
    obs, _ = env.reset(seed=int(seed))
    base = env.unwrapped
    source = base.objs[base.source_obj_name]
    target = base.objs[base.target_obj_name]
    frames = [_rgb(obs)]
    actions = []
    states = []
    timeline = []
    ever_grasped = False
    ever_on_target = False
    terminated = False
    for low_step in range(max_steps):
        tcp = tcp_pose_world(env)
        proprio = encode_base_ee6d(
            world_pose_to_base(env, tcp), _current_gripper_01(env)
        )
        image = Image.fromarray(_rgb(obs))
        encoded = processor.encode_image([[image]])
        language = processor.encode_language([TASK])
        tensors = {
            "input_ids": language["input_ids"].to("cuda"),
            "image_input": encoded["image_input"].to("cuda"),
            "image_mask": encoded["image_mask"].to("cuda"),
            "domain_id": torch.tensor([domain_id], device="cuda", dtype=torch.long),
            "proprio": torch.from_numpy(proprio[None]).to("cuda"),
        }
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            predicted = model.generate_actions(
                tensors["input_ids"],
                tensors["image_input"],
                tensors["image_mask"],
                tensors["domain_id"],
                tensors["proprio"],
                steps=flow_steps,
            )[0].float().cpu().numpy()
        chunk_rows = []
        for action_index, predicted_action in enumerate(predicted[:execute_horizon]):
            command = model_target_to_panda_action(env, predicted_action[:10])
            obs, _reward, terminated_value, truncated_value, info = env.step(command)
            command_array = _array(command).reshape(-1).astype(np.float32)
            actions.append(command_array)
            frames.append(_rgb(obs))
            source_pose = _pose(base.objs[base.source_obj_name])
            target_pose = _pose(base.objs[base.target_obj_name])
            current_grasped = _bool(base.agent.is_grasping(source))
            current_eval = base.evaluate()
            current_on_target = _bool(current_eval["src_on_target"])
            ever_grasped = ever_grasped or current_grasped
            ever_on_target = ever_on_target or current_on_target
            states.append(
                np.concatenate(
                    [source_pose, target_pose, tcp_pose_world(env), [current_grasped, current_on_target]]
                ).astype(np.float32)
            )
            chunk_rows.append(
                {
                    "low_step": low_step,
                    "chunk_index": action_index,
                    "source": "policy",
                    "predicted_action": np.asarray(predicted_action[:10], dtype=np.float32).tolist(),
                    "command": command_array.tolist(),
                    "grasped": current_grasped,
                    "on_target": current_on_target,
                }
            )
            if _bool(terminated_value) or _bool(truncated_value):
                terminated = True
                break
        timeline.extend(chunk_rows)
        if terminated or _bool(base.evaluate()["success"]):
            break
    final_eval = base.evaluate()
    success = _bool(final_eval["success"])
    stem = f"episode_{episode_index:06d}"
    video_path = output / "videos" / f"{stem}.mp4"
    action_path = output / "actions" / f"{stem}.npy"
    state_path = output / "states" / f"{stem}.npy"
    timeline_path = output / "timelines" / f"{stem}.json"
    reset_path = output / "reset_metadata" / f"{stem}.json"
    _save_video(video_path, frames)
    np.save(action_path, np.asarray(actions, dtype=np.float32))
    np.save(state_path, np.asarray(states, dtype=np.float32))
    reset_metadata = {
        "seed": int(seed),
        "split": split,
        "robot": type(base.agent).__name__,
        "source_object": base.source_obj_name,
        "target_object": base.target_obj_name,
        "source_model_scale": float(base.episode_model_scales[base.source_obj_name]),
        "target_model_scale": float(base.episode_model_scales[base.target_obj_name]),
        "configured_source_pose": _array(base.xyz_configs[0, 0]).tolist(),
        "configured_target_pose": _array(base.xyz_configs[0, 1]).tolist(),
        "start_source_pose": _pose(source).tolist(),
        "start_target_pose": _pose(target).tolist(),
    }
    reset_path.write_text(json.dumps(reset_metadata, indent=2) + "\n", encoding="utf-8")
    timeline_path.write_text(json.dumps(timeline, indent=2) + "\n", encoding="utf-8")
    return {
        "episode_index": episode_index,
        "seed": int(seed),
        "split": split,
        "success": success,
        "strict_success": success,
        "ever_grasped": ever_grasped,
        "ever_on_target": ever_on_target,
        "num_actions": len(actions),
        "num_frames": len(frames),
        "video_path": str(video_path),
        "action_path": str(action_path),
        "state_path": str(state_path),
        "timeline_path": str(timeline_path),
        "reset_metadata_path": str(reset_path),
        "final_eval": {
            key: bool(_array(value).reshape(-1)[0])
            if _array(value).dtype == bool
            else float(_array(value).reshape(-1)[0])
            for key, value in final_eval.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rlinf-root", type=Path, required=True)
    parser.add_argument("--task-module", type=Path, required=True)
    parser.add_argument("--split", choices=("id", "ood"), required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--domain-id", type=int, default=20)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--execute-horizon", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=120)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    for name in ("videos", "actions", "states", "timelines", "reset_metadata"):
        (args.output / name).mkdir()
    sys.path.insert(0, str(args.rlinf_root))
    sys.path.insert(0, str(args.xvla_root))
    _load_task_module(args.task_module)
    from models.modeling_xvla import XVLA
    from models.processing_xvla import XVLAProcessor

    model = XVLA.from_pretrained(str(args.checkpoint), torch_dtype=torch.bfloat16).cuda().eval()
    processor = XVLAProcessor.from_pretrained(str(args.checkpoint))
    rows = []
    for index in range(args.episodes):
        env = gym.make(
            ENV_IDS[args.split],
            obs_mode="rgb+segmentation",
            render_mode="rgb_array",
            sim_backend="physx_cpu",
            control_mode="pd_ee_body_target_delta_pose_real",
            max_episode_steps=args.max_episode_steps,
        )
        try:
            row = _eval_one(
                model,
                processor,
                env,
                args.split,
                args.seed_start + index,
                index,
                args.output,
                args.domain_id,
                args.flow_steps,
                args.execute_horizon,
                args.max_episode_steps,
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
        finally:
            env.close()
    summary = {
        "format": "xvla_panda_vegetable_basket_policy_eval_v1",
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "episodes": len(rows),
        "successes": sum(int(row["strict_success"]) for row in rows),
        "strict_successes": sum(int(row["strict_success"]) for row in rows),
        "ever_grasped_successes": sum(int(row["ever_grasped"]) for row in rows),
        "ever_on_target_successes": sum(int(row["ever_on_target"]) for row in rows),
        "videos": len(list((args.output / "videos").glob("*.mp4"))),
        "actions": len(list((args.output / "actions").glob("*.npy"))),
        "states": len(list((args.output / "states").glob("*.npy"))),
        "timelines": len(list((args.output / "timelines").glob("*.json"))),
        "rows": rows,
        "protocol": {
            "domain_id": args.domain_id,
            "flow_steps": args.flow_steps,
            "execute_horizon": args.execute_horizon,
            "max_episode_steps": args.max_episode_steps,
            "failure_definition": "not strict_success",
        },
    }
    (args.output / "episodes.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output / "EVAL_COMPLETE").write_text("complete\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

