#!/usr/bin/env python3
"""Zero-shot evaluation of RLinf's official ManiSkill pi0.5 checkpoint.

The released checkpoint was trained on PutOnPlateInScene25Main with a custom
Panda controller, one c19 camera, and physical-unit 7-D end-effector deltas.
This evaluator preserves that policy interface while replacing only the
ManiSkill task with PegInsertionSide-v1 or PlugCharger-v1.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _bootstrap_rlinf() -> Path:
    root = Path(__file__).resolve().parents[1] / "RLinf"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.environ.setdefault("RLINF_ROOT", str(root))
    return root


RLINF_ROOT = _bootstrap_rlinf()


TASKS = {
    "peg": ("PegInsertionSide-v1", "insert the peg into the hole"),
    "plug": ("PlugCharger-v1", "plug the charger into the receptacle"),
}


def clip_action_chunk(actions: Any, low: np.ndarray, high: np.ndarray, chunk_size: int) -> np.ndarray:
    """Validate, truncate, and clip a checkpoint-native 7-D action chunk."""

    array = np.asarray(actions, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.shape[1] != 7:
        raise ValueError(f"Expected checkpoint-native actions [H,7], got {array.shape}")
    if array.shape[0] < chunk_size:
        array = np.concatenate(
            [array, np.repeat(array[-1:], chunk_size - array.shape[0], axis=0)], axis=0
        )
    return np.clip(array[:chunk_size], low, high).astype(np.float32)


def _compose_model_config(checkpoint: Path):
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict

    config_dir = RLINF_ROOT / "examples" / "embodiment" / "config"
    os.environ["EMBODIED_PATH"] = str(config_dir.parent)
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="maniskill_ppo_openpi_pi05")
    model_cfg = copy.deepcopy(cfg.actor.model)
    with open_dict(model_cfg):
        model_cfg.model_path = str(checkpoint)
        model_cfg.add_value_head = False
        model_cfg.load_to_device = True
        model_cfg.num_action_chunks = 5
        model_cfg.action_dim = 7
        model_cfg.num_steps = 4
        model_cfg.openpi.config_name = "pi05_maniskill"
        model_cfg.openpi.action_horizon = 5
        model_cfg.openpi.action_chunk = 5
        model_cfg.openpi.action_env_dim = 7
        model_cfg.openpi.num_steps = 4
        model_cfg.openpi.add_value_head = False
        model_cfg.openpi.noise_method = "flow_sde"
        model_cfg.openpi.train_expert_only = False
    return model_cfg


def _load_model(checkpoint: Path):
    from rlinf.models import get_model

    model = get_model(_compose_model_config(checkpoint))
    if model is None:
        raise RuntimeError(f"Could not load official ManiSkill pi0.5 checkpoint: {checkpoint}")
    return model.to("cuda").eval().requires_grad_(False)


def _build_env(task: str, max_episode_steps: int, image_size: int):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    from rlinf.envs.maniskill.tasks.panda_table_agent import (  # noqa: F401
        PandaBridgeDatasetFlatTable,
    )

    env_id, _ = TASKS[task]
    return gym.make(
        env_id,
        robot_uids="panda_bridgedataset_flat_table",
        num_envs=1,
        obs_mode="rgb",
        control_mode="pd_ee_body_target_delta_pose_real",
        sim_backend="physx_cpu",
        reward_mode="sparse",
        render_mode="rgb_array",
        sim_config={"sim_freq": 500, "control_freq": 5},
        sensor_configs={"width": image_size, "height": image_size},
        max_episode_steps=max_episode_steps,
    )


def _checkpoint_native_state(env: Any) -> torch.Tensor:
    from transforms3d.euler import mat2euler

    base = env.unwrapped
    transform = base.agent.ee_pose_at_robot_base.to_transformation_matrix().detach().cpu().numpy()
    position = transform[:, :3, 3]
    euler = np.stack([mat2euler(value[:3, :3], "sxyz") for value in transform], axis=0)
    gripper = (base.agent.robot.get_qpos().detach().cpu().numpy()[:, -1:] * 2.0)
    state = np.concatenate([position, euler, gripper], axis=1).astype(np.float32)
    return torch.as_tensor(state, device=base.device)


def _model_observation(env: Any, observation: dict[str, Any], prompt: str) -> dict[str, Any]:
    image = observation["sensor_data"]["c19_front_view"]["rgb"].to(torch.uint8)
    return {
        "main_images": image,
        "states": _checkpoint_native_state(env),
        "task_descriptions": [prompt],
    }


def _bool(value: Any) -> bool:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return bool(np.asarray(value, dtype=bool).reshape(-1).any())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--image-size", type=int, default=384)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0 or args.chunk_size <= 0:
        raise ValueError("episodes and chunk-size must be positive")
    if not (args.checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(args.checkpoint / "model.safetensors")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = _load_model(args.checkpoint)
    env = _build_env(args.task, args.max_episode_steps, args.image_size)
    _, prompt = TASKS[args.task]
    low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    rows: list[dict[str, Any]] = []
    try:
        for episode_index in range(args.episodes):
            seed = args.seed + episode_index
            observation, info = env.reset(seed=seed)
            success = _bool(info.get("success", False))
            steps = chunks = 0
            while steps < args.max_episode_steps and not success:
                env_obs = _model_observation(env, observation, prompt)
                with torch.inference_mode():
                    predicted, _ = model.predict_action_batch(
                        env_obs=env_obs, mode="eval", compute_values=False
                    )
                actions = clip_action_chunk(
                    predicted.detach().float().cpu().numpy(), low, high, args.chunk_size
                )
                chunks += 1
                for action in actions:
                    env_action = torch.as_tensor(action, device=env.unwrapped.device).unsqueeze(0)
                    observation, _reward, terminated, truncated, info = env.step(env_action)
                    steps += 1
                    success = _bool(info.get("success", False))
                    if success or _bool(terminated) or _bool(truncated):
                        break
            rows.append(
                {
                    "episode_index": episode_index,
                    "seed": seed,
                    "success": success,
                    "steps": steps,
                    "chunks": chunks,
                }
            )
            successes = sum(int(row["success"]) for row in rows)
            print(
                f"[eval] task={args.task} episode={episode_index + 1}/{args.episodes} "
                f"seed={seed} success={int(success)} cumulative={successes}/{len(rows)} "
                f"success_rate={successes / len(rows):.3f}",
                flush=True,
            )
    finally:
        env.close()
    successes = sum(int(row["success"]) for row in rows)
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_source": "RLinf/RLinf-Pi05-ManiSkill-25Main-SFT",
        "checkpoint_training_task": "PutOnPlateInScene25Main-v3",
        "evaluation_task": TASKS[args.task][0],
        "policy_interface": {
            "camera": "c19_front_view",
            "action": "7-D physical-unit end-effector delta pose + gripper",
            "control_mode": "pd_ee_body_target_delta_pose_real",
            "action_chunk": args.chunk_size,
        },
        "episodes": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows),
        "rollouts": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
