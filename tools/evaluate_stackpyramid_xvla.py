#!/usr/bin/env python3
"""Evaluate a trained X-VLA policy on controlled StackPyramid splits."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import imageio
import numpy as np
import torch
from PIL import Image


REAL_ACTION_DIM = 8
MODEL_ACTION_DIM = 20
ACTION_HORIZON = 10
TASK = "stack the red cube next to the green cube and place the blue cube on top"


def scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.size == 1:
        return array.reshape(-1)[0].item()
    return value


def bool_scalar(value: Any) -> bool:
    return bool(scalar(value))


def image_array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3 or array.shape[-1] < 3:
        raise ValueError(f"expected HWC RGB image, got {array.shape}")
    return np.ascontiguousarray(array[..., :3].astype(np.uint8))


def frame_array(value: Any) -> np.ndarray | None:
    if isinstance(value, dict):
        for key in ("base_camera", "hand_camera", "rgb"):
            if key in value:
                frame = frame_array(value[key])
                if frame is not None:
                    return frame
        return None
    try:
        return image_array(value)
    except ValueError:
        return None


def make_policy(checkpoint: Path, xvla_root: Path, device: torch.device) -> Any:
    sys.path.insert(0, str(xvla_root))
    from models.modeling_xvla import XVLA
    from models.processing_xvla import XVLAProcessor

    model = XVLA.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device).eval()
    processor = XVLAProcessor.from_pretrained(checkpoint)
    return model, processor


def prepare_inputs(model: Any, processor: Any, raw_obs: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    sensors = raw_obs["sensor_data"]
    images = [[
        Image.fromarray(image_array(sensors["base_camera"]["rgb"])),
        Image.fromarray(image_array(sensors["hand_camera"]["rgb"])),
    ]]
    inputs = {
        **processor.encode_image(images),
        **processor.encode_language([TASK]),
    }
    # StackPyramid exposes the Panda state as a flat 64-D observation rather
    # than the nested agent/qpos structure used by PickSingleYCB.
    qpos = raw_obs["state"]
    if isinstance(qpos, torch.Tensor):
        qpos = qpos.detach().cpu().numpy()
    qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
    proprio = np.zeros((1, MODEL_ACTION_DIM), dtype=np.float32)
    proprio[0, : min(REAL_ACTION_DIM, qpos.size)] = qpos[:REAL_ACTION_DIM]
    inputs.update({
        "domain_id": torch.zeros(1, dtype=torch.long),
        "proprio": torch.from_numpy(proprio),
    })
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }


@torch.inference_mode()
def predict(model: Any, inputs: dict[str, torch.Tensor], device: torch.device, seed: int, steps: int) -> np.ndarray:
    torch.manual_seed(seed)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        encoding = model.forward_vlm(inputs["input_ids"], inputs["image_input"], inputs["image_mask"])
        prior = torch.randn(
            1, ACTION_HORIZON, model.action_space.dim_action,
            device=device, dtype=inputs["proprio"].dtype,
        )
        action = torch.zeros_like(prior)
        for index in range(max(1, steps), 0, -1):
            time = torch.full((1,), index / float(max(1, steps)), device=device, dtype=prior.dtype)
            noisy = prior * time[:, None, None] + action * (1 - time[:, None, None])
            proprio, noisy = model.action_space.preprocess(inputs["proprio"], noisy)
            action = model.transformer(
                domain_id=inputs["domain_id"],
                action_with_noise=noisy,
                proprio=proprio,
                t=time,
                **encoding,
            )
        action = model.action_space.postprocess(action)
    return action.float().cpu().numpy()[0]


def details(env: Any) -> dict[str, Any]:
    base = env.unwrapped
    cubes = (base.cubeA, base.cubeB, base.cubeC)
    positions = [cube.pose.p.detach().cpu().numpy().reshape(-1, 3)[0] for cube in cubes]
    threshold = float(np.linalg.norm(2 * base.cube_half_size[:2].detach().cpu().numpy()) + 0.005)
    xy_ab = float(np.linalg.norm((positions[0] - positions[1])[:2])) <= threshold
    xy_cb = float(np.linalg.norm((positions[2] - positions[1])[:2])) <= threshold
    xy_ca = float(np.linalg.norm((positions[2] - positions[0])[:2])) <= threshold
    z_cb = abs(float(positions[2][2] - positions[1][2])) > 0.02
    z_ca = abs(float(positions[2][2] - positions[0][2])) > 0.02
    grasped = [bool_scalar(base.agent.is_grasping(cube)) for cube in cubes]
    evaluation = base.evaluate()
    return {
        "positions": [position.tolist() for position in positions],
        "xy_ab": xy_ab,
        "xy_cb": xy_cb,
        "xy_ca": xy_ca,
        "z_cb": z_cb,
        "z_ca": z_ca,
        "grasped": grasped,
        "success": bool_scalar(evaluation["success"] if isinstance(evaluation, dict) else evaluation),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("id", "stage1_ood", "stage2_ood", "stage3_ood"), required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--start-seed", type=int, required=True)
    parser.add_argument("--max-episode-steps", type=int, default=250)
    parser.add_argument("--execute-horizon", type=int, default=5)
    parser.add_argument("--flow-steps", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sim-backend", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--render-backend", choices=("gpu", "cpu"), default="gpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    videos = args.output / "videos"
    videos.mkdir()
    (args.output / "config.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    from tools.stackpyramid_task import register_stackpyramid_splits, stackpyramid_env_id

    register_stackpyramid_splits()
    device = torch.device(args.device)
    model, processor = make_policy(args.checkpoint, args.xvla_root, device)
    env = gym.make(
        stackpyramid_env_id(args.split),
        obs_mode="rgb+state",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
    )
    rows: list[dict[str, Any]] = []
    try:
        for episode_index in range(args.episodes):
            seed = args.start_seed + episode_index
            raw_obs, _ = env.reset(seed=seed)
            frames: list[np.ndarray] = []
            first = frame_array(env.render())
            if first is not None:
                frames.append(first)
            executed = 0
            ever_grasped = False
            ever_base = False
            ever_pyramid = False
            final_info: dict[str, Any] = {}
            while executed < args.max_episode_steps and not ever_pyramid:
                inputs = prepare_inputs(model, processor, raw_obs, device)
                chunk = predict(model, inputs, device, seed + executed, args.flow_steps)
                chunk = np.clip(
                    chunk[:ACTION_HORIZON],
                    np.asarray(env.action_space.low, dtype=np.float32),
                    np.asarray(env.action_space.high, dtype=np.float32),
                )
                for action in chunk[: args.execute_horizon]:
                    raw_obs, _, terminated, truncated, info = env.step(action.astype(np.float32))
                    executed += 1
                    final_info = info if isinstance(info, dict) else {}
                    current = details(env)
                    ever_grasped |= any(current["grasped"])
                    ever_base |= bool(current["xy_ab"] and (current["z_cb"] or current["z_ca"]))
                    ever_pyramid |= bool(current["success"])
                    frame = frame_array(env.render())
                    if frame is not None:
                        frames.append(frame)
                    if bool_scalar(terminated) or bool_scalar(truncated) or executed >= args.max_episode_steps:
                        break
                if bool_scalar(terminated) or bool_scalar(truncated):
                    break
            final = details(env)
            video_path = videos / f"{args.split}_{seed}.mp4"
            if frames:
                with imageio.get_writer(video_path, fps=10, codec="libx264", macro_block_size=None) as writer:
                    for frame in frames:
                        writer.append_data(frame)
            row = {
                "episode_index": episode_index,
                "seed": seed,
                "split": args.split,
                "steps": executed,
                "ever_grasped": bool(ever_grasped),
                "ever_base_completed": bool(ever_base),
                "strict_success": bool(ever_pyramid),
                "final": final,
                "video": str(video_path),
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=True), flush=True)
    finally:
        env.close()

    summary = {
        "format": "stackpyramid_xvla_policy_eval_v1",
        "split": args.split,
        "episodes": len(rows),
        "ever_grasped": sum(int(row["ever_grasped"]) for row in rows),
        "ever_base_completed": sum(int(row["ever_base_completed"]) for row in rows),
        "strict_success": sum(int(row["strict_success"]) for row in rows),
        "video_count": len(list(videos.glob("*.mp4"))),
        "rows": rows,
    }
    (args.output / "episodes.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8"
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n")
    if summary["episodes"] != args.episodes or summary["video_count"] != args.episodes:
        raise RuntimeError(f"incomplete evaluation artifacts: {summary}")
    (args.output / "EVAL_COMPLETE").write_text("complete\n")


if __name__ == "__main__":
    main()
