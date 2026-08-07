"""Pure-policy OpenVLA evaluation on the controlled airplane splits."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

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

from .model import load_openvla, prepare_inference_inputs
from .dataset import AIRPLANE_INSTRUCTION
from .runtime import DetectorBank


def _base_image(raw_obs: dict) -> Image.Image:
    value = raw_obs["sensor_data"]["base_camera"]["rgb"]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.ndim == 4:
        value = value[0]
    return Image.fromarray(value.astype(np.uint8)).convert("RGB")


def _grasped(env) -> bool:
    return _bool_scalar(env.unwrapped.agent.is_grasping(env.unwrapped.obj))


def _run_episode(env, model, processor, detector_bank, split: str, seed: int, episode_index: int, output: Path, device: int, max_steps: int):
    raw_obs, _ = env.reset(seed=seed)
    metadata = reset_metadata(env, split=split)
    low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    records, actions, timeline = [_extract_record(raw_obs)], [], []
    ever_grasped = False
    strict_success = False
    for step in range(max_steps):
        image = _base_image(raw_obs)
        start = time.perf_counter()
        inputs = prepare_inference_inputs(model, processor, image, device)
        with torch.inference_mode():
            action, _ = model.predict_action(**inputs, unnorm_key="airplane", do_sample=False)
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start) * 1000.0
        scores = {} if detector_bank is None else detector_bank.score(model, inputs)
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), low, high)
        if action.size != 8:
            raise ValueError(f"OpenVLA returned {action.size} actions; expected 8D pd_joint_delta_pos")
        timeline.append({"step": step, "policy_latency_ms": latency_ms, "action": action.tolist(), "scores": scores})
        raw_obs, _reward, terminated, truncated, info = env.step(
            torch.as_tensor(action, device=env.unwrapped.device).unsqueeze(0)
        )
        actions.append(action)
        records.append(_extract_record(raw_obs))
        ever_grasped = ever_grasped or _grasped(env)
        strict_success = _bool_scalar(info.get("success"))
        if strict_success or _bool_scalar(terminated) or _bool_scalar(truncated):
            break
    main_camera = _select_camera(records[0].obs, "", ("base_camera",) + MAIN_CAMERA_CANDIDATES, "main")
    wrist_camera = _select_camera(records[0].obs, "", ("hand_camera",) + WRIST_CAMERA_CANDIDATES, "wrist")
    frames = _build_frames(records, actions, PICK_SINGLE_YCB_AIRPLANE_TASK, main_camera, wrist_camera)
    video = write_episode_video_durably(frames, output / "videos", episode_index, seed, fps=10)
    return {
        "episode_index": episode_index,
        "seed": seed,
        "split": split,
        "steps": len(actions),
        "ever_grasped": bool(ever_grasped),
        "strict_success": bool(strict_success),
        "video": str(video),
        "timeline": timeline,
        **metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-path", default="openvla/openvla-7b")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("id", "ood"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-episode-steps", type=int, default=250)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--detector-assets", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    register_controlled_pick_single_ycb_airplane_variants()
    model, processor = load_openvla(args.base_path, args.checkpoint, args.device)
    detector_bank = None if args.detector_assets is None else DetectorBank(args.detector_assets, args.device)
    env_args = argparse.Namespace(split=args.split, image_size=384, control_freq=10, max_episode_steps=args.max_episode_steps, sim_backend="physx_cpu")
    env = _build_env(env_args, control_mode="pd_joint_delta_pos")
    rows = []
    try:
        for index in range(args.episodes):
            row = _run_episode(env, model, processor, detector_bank, args.split, args.seed + index, index, args.output, args.device, args.max_episode_steps)
            rows.append(row)
            print(f"[rollout] {args.split} {index + 1}/{args.episodes} seed={row['seed']} ever_grasped={int(row['ever_grasped'])}", flush=True)
    finally:
        env.close()
    summary = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "episodes": len(rows),
        "ever_grasped": int(sum(row["ever_grasped"] for row in rows)),
        "strict_success": int(sum(row["strict_success"] for row in rows)),
        "ever_grasped_rate": float(np.mean([row["ever_grasped"] for row in rows])),
        "max_episode_steps": args.max_episode_steps,
        "failure_definition": "not ever_grasped",
        "rows": rows,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
