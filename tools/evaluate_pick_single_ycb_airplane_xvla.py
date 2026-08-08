#!/usr/bin/env python3
"""Evaluate an X-VLA checkpoint on the controlled airplane task."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

ROOT = Path(__file__).resolve().parents[1]
RLINF_ROOT = ROOT / "RLinf"
sys.path[:0] = [str(ROOT), str(RLINF_ROOT)]

from rlinf.envs.maniskill.pick_single_ycb_airplane_variants import (  # noqa: E402
    PICK_SINGLE_YCB_AIRPLANE_TASK,
    register_controlled_pick_single_ycb_airplane_variants,
    reset_metadata,
)
from toolkits.lerobot.collect_maniskill_pick_single_ycb_airplane_lerobot import (  # noqa: E402
    _build_env,
    write_episode_video_durably,
)
from toolkits.lerobot.collect_maniskill_peg_lerobot_joint import (  # noqa: E402
    MAIN_CAMERA_CANDIDATES,
    WRIST_CAMERA_CANDIDATES,
    _bool_scalar,
    _build_frames,
    _extract_record,
    _select_camera,
)
from tools.pick_single_ycb_airplane_eval_common import clip_action_chunk  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("id", "ood"), default="id")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=50000)
    parser.add_argument("--execute-horizon", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=150)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--control-freq", type=int, default=10)
    parser.add_argument("--sim-backend", choices=("physx_cpu", "gpu"), default="physx_cpu")
    return parser.parse_args()


def _rgb_image(value: torch.Tensor) -> Image.Image:
    array = value.detach().cpu().numpy()
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected one HWC RGB image, got {array.shape}")
    # The training handler decodes JPEG/PNG bytes through OpenCV without a
    # BGR-to-RGB conversion. Mirror that exact channel order at evaluation.
    return Image.fromarray(np.ascontiguousarray(array[..., ::-1].astype(np.uint8)))


def _grasped(env: Any) -> bool:
    return _bool_scalar(env.unwrapped.agent.is_grasping(env.unwrapped.obj))


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "actions").mkdir()

    sys.path.insert(0, str(args.xvla_root.resolve()))
    from models.modeling_xvla import XVLA  # noqa: E402
    from models.processing_xvla import XVLAProcessor  # noqa: E402

    import gymnasium as gym  # noqa: F401,E402
    import mani_skill.envs  # noqa: F401,E402

    register_controlled_pick_single_ycb_airplane_variants()
    device = torch.device("cuda:0")
    model = XVLA.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16).to(device).eval()
    processor = XVLAProcessor.from_pretrained(args.checkpoint)
    image_transform = transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    language = processor.encode_language([PICK_SINGLE_YCB_AIRPLANE_TASK])["input_ids"].to(device)

    env_args = argparse.Namespace(
        split=args.split,
        image_size=args.image_size,
        control_freq=args.control_freq,
        max_episode_steps=args.max_episode_steps,
        sim_backend=args.sim_backend,
    )
    env = _build_env(env_args, control_mode="pd_joint_delta_pos")
    low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    rows: list[dict[str, Any]] = []
    try:
        for episode_index in range(args.episodes):
            seed = args.seed + episode_index
            torch.manual_seed(seed)
            raw_obs, _ = env.reset(seed=seed)
            metadata = reset_metadata(env, split=args.split)
            records, executed = [_extract_record(raw_obs)], []
            ever_grasped = strict_success = False
            while len(executed) < args.max_episode_steps and not strict_success:
                sensors = raw_obs["sensor_data"]
                images = torch.stack(
                    [
                        image_transform(_rgb_image(sensors["base_camera"]["rgb"])),
                        image_transform(_rgb_image(sensors["hand_camera"]["rgb"])),
                    ]
                ).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
                proprio = torch.zeros((1, 20), dtype=torch.bfloat16, device=device)
                qpos = raw_obs["agent"]["qpos"].reshape(-1).to(device=device, dtype=torch.bfloat16)
                proprio[0, : min(9, qpos.numel())] = qpos[:9]
                inputs = {
                    "input_ids": language,
                    "image_input": images,
                    "image_mask": torch.ones((1, 2), dtype=torch.bool, device=device),
                    "domain_id": torch.zeros(1, dtype=torch.long, device=device),
                    "proprio": proprio,
                }
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    predicted = model.generate_actions(**inputs, steps=args.flow_steps)
                chunk = clip_action_chunk(predicted.float().cpu().numpy(), low, high, args.execute_horizon)
                for action in chunk:
                    raw_obs, _, terminated, truncated, info = env.step(
                        torch.as_tensor(action, device=env.unwrapped.device).unsqueeze(0)
                    )
                    executed.append(action.copy())
                    records.append(_extract_record(raw_obs))
                    ever_grasped |= _grasped(env)
                    strict_success |= _bool_scalar(info.get("success"))
                    if strict_success or _bool_scalar(terminated) or _bool_scalar(truncated):
                        break

            main_camera = _select_camera(records[0].obs, "", ("base_camera",) + MAIN_CAMERA_CANDIDATES, "main")
            wrist_camera = _select_camera(records[0].obs, "", ("hand_camera",) + WRIST_CAMERA_CANDIDATES, "wrist")
            frames = _build_frames(
                records=records,
                actions=executed,
                task=PICK_SINGLE_YCB_AIRPLANE_TASK,
                main_camera=main_camera,
                wrist_camera=wrist_camera,
            )
            video_path = write_episode_video_durably(
                frames,
                video_dir=args.output_dir / "videos",
                episode_index=episode_index,
                seed=seed,
                fps=args.control_freq,
            )
            actions_path = args.output_dir / "actions" / f"episode_{episode_index:06d}_seed_{seed:06d}.npy"
            np.save(actions_path, np.asarray(executed, dtype=np.float32))
            row = {
                "episode_index": episode_index,
                "seed": seed,
                "ever_grasped": ever_grasped,
                "strict_success": strict_success,
                "steps": len(executed),
                "video": str(video_path),
                "actions": str(actions_path),
                **metadata,
            }
            rows.append(row)
            print(
                f"[rollout] {episode_index + 1}/{args.episodes} seed={seed} "
                f"grasp={int(ever_grasped)} strict={int(strict_success)}",
                flush=True,
            )
    finally:
        env.close()

    summary = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "episodes": len(rows),
        "ever_grasped_successes": sum(int(row["ever_grasped"]) for row in rows),
        "ever_grasped_rate": float(np.mean([row["ever_grasped"] for row in rows])),
        "strict_successes": sum(int(row["strict_success"]) for row in rows),
        "strict_success_rate": float(np.mean([row["strict_success"] for row in rows])),
        "execute_horizon": args.execute_horizon,
        "max_episode_steps": args.max_episode_steps,
        "flow_steps": args.flow_steps,
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
