#!/usr/bin/env python3
"""Small, auditable ManiSkill StackPyramid environment smoke test."""

from __future__ import annotations

import json
import os
from pathlib import Path

import gymnasium as gym
import imageio.v3 as iio
import mani_skill.envs  # noqa: F401  Registers ManiSkill environments.
import numpy as np


def _shape_summary(value):
    if isinstance(value, dict):
        return {str(key): _shape_summary(item) for key, item in value.items()}
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None:
        return {"shape": list(shape), "dtype": str(dtype)}
    return {"type": type(value).__name__}


def _frame_array(frame):
    if isinstance(frame, dict):
        for key in ("base_camera", "hand_camera", "rgb"):
            if key in frame:
                return _frame_array(frame[key])
        return None
    if hasattr(frame, "detach"):
        frame = frame.detach().cpu()
    array = np.asarray(frame)
    if array.ndim == 4:
        array = array[0]
    if array.ndim == 3 and array.shape[-1] in (3, 4):
        return array[..., :3].astype(np.uint8)
    return None


def main() -> None:
    output = Path(os.environ["STACKPYRAMID_SMOKE_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    env = gym.make(
        "StackPyramid-v1",
        num_envs=1,
        obs_mode="rgb+state",
        control_mode="pd_ee_delta_pose",
        render_mode="rgb_array",
        sim_backend="gpu",
    )
    observations, reset_info = env.reset(seed=20260814)
    frames = []
    initial_frame = _frame_array(env.render())
    if initial_frame is not None:
        frames.append(initial_frame)

    action = np.zeros(env.action_space.shape, dtype=np.float32)
    step_records = []
    for step in range(3):
        observations, reward, terminated, truncated, info = env.step(action)
        frame = _frame_array(env.render())
        if frame is not None:
            frames.append(frame)
        step_records.append(
            {
                "step": step + 1,
                "reward": np.asarray(reward).tolist(),
                "terminated": np.asarray(terminated).tolist(),
                "truncated": np.asarray(truncated).tolist(),
                "info_keys": sorted(str(key) for key in info),
            }
        )

    evaluation = env.unwrapped.evaluate()
    summary = {
        "env_id": "StackPyramid-v1",
        "seed": 20260814,
        "obs_mode": "rgb+state",
        "control_mode": "pd_ee_delta_pose",
        "sim_backend": "gpu",
        "action_space": {
            "shape": list(env.action_space.shape),
            "dtype": str(env.action_space.dtype),
            "low": np.asarray(env.action_space.low).tolist(),
            "high": np.asarray(env.action_space.high).tolist(),
        },
        "observation_structure": _shape_summary(observations),
        "reset_info_keys": sorted(str(key) for key in reset_info),
        "steps": step_records,
        "evaluation_structure": _shape_summary(evaluation),
        "render_frames": len(frames),
    }
    (output / "stackpyramid_smoke.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    if frames:
        iio.imwrite(output / "stackpyramid_reset_step_smoke.mp4", np.stack(frames), fps=10)
    env.close()


if __name__ == "__main__":
    main()
