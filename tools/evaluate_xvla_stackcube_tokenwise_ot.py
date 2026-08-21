#!/usr/bin/env python3
"""Passive StackCube rollout smoke for the token-wise PCA/OT detector."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.stackcube_stage2_ood import (  # noqa: E402
    STACK_CUBE_SPLITS,
    register_stack_cube_splits,
    stack_cube_env_id,
    stack_cube_reset_metadata,
)
from tools.xvla_tokenwise_ot import (  # noqa: E402
    asset_from_state_dict,
    select_monotonic_multiview_phase,
)
from tools.xvla_airplane_runtime import XVLAAirplanePolicy  # noqa: E402


def bool_scalar(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu().reshape(-1)[0].item())
    return bool(np.asarray(value).reshape(-1)[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--base-asset", type=Path, required=True)
    parser.add_argument("--wrist-asset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=50000)
    parser.add_argument("--max-episode-steps", type=int, default=50)
    parser.add_argument("--execute-horizon", type=int, default=5)
    parser.add_argument("--flow-steps", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sim-backend", choices=("physx_cpu", "gpu"), default="physx_cpu")
    return parser.parse_args()


def load_assets(path: Path, device: torch.device) -> list[Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "xvla_stackcube_tokenwise_pca_ot_asset_v1":
        raise ValueError(f"unexpected asset format: {path}")
    return [asset_from_state_dict(state, device=device) for state in payload["assets"]]


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.episodes < 1:
        raise ValueError("episodes must be positive")
    device = torch.device(args.device)
    policy = XVLAAirplanePolicy(args.checkpoint, args.xvla_root, device=args.device)
    base_assets = load_assets(args.base_asset, device)
    wrist_assets = load_assets(args.wrist_asset, device)

    import gymnasium as gym  # noqa: F401
    import mani_skill.envs  # noqa: F401

    register_stack_cube_splits()
    rows: list[dict[str, Any]] = []
    try:
        for split in ("id", "ood"):
            env = gym.make(
                stack_cube_env_id(split),
                robot_uids="panda_wristcam",
                num_envs=1,
                obs_mode="rgb",
                control_mode="pd_joint_delta_pos",
                reward_mode="sparse",
                render_mode="rgb_array",
                sim_backend=args.sim_backend,
                sim_config={"sim_freq": 100, "control_freq": 10},
                sensor_configs={"width": 224, "height": 224},
                max_episode_steps=args.max_episode_steps,
            )
            try:
                low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
                high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
                for episode_index in range(args.episodes):
                    seed = args.seed + episode_index + (0 if split == "id" else 1000)
                    torch.manual_seed(seed)
                    raw_obs, _ = env.reset(seed=seed)
                    metadata = stack_cube_reset_metadata(env, split=split)
                    previous_phase: int | None = None
                    executed = 0
                    timeline: list[dict[str, Any]] = []
                    success = False
                    while executed < args.max_episode_steps and not success:
                        inputs = policy.prepare(raw_obs, "stack the red cube on the green cube")
                        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                            encoding = policy.model.forward_vlm(
                                inputs["input_ids"], inputs["image_input"], inputs["image_mask"]
                            )
                            base_tokens = policy.model.vlm._encode_image(inputs["image_input"][:, 0])
                            wrist_tokens = policy.model.vlm._encode_image(inputs["image_input"][:, 1])
                            valid_base = torch.ones(base_tokens.shape[:2], dtype=torch.bool, device=device)
                            valid_wrist = torch.ones(wrist_tokens.shape[:2], dtype=torch.bool, device=device)
                            phase, scores = select_monotonic_multiview_phase(
                                [base_tokens, wrist_tokens],
                                [valid_base, valid_wrist],
                                [base_assets, wrist_assets],
                                previous_phase=previous_phase,
                                topk=min(4, base_tokens.shape[1]),
                            )
                            generated = policy._generate_from_encoding(
                                inputs, encoding, steps=args.flow_steps
                            )
                        previous_phase = phase
                        chunk = generated.float().cpu().numpy()[0]
                        chunk = np.clip(chunk, low, high)
                        timeline.append(
                            {
                                "env_step": executed,
                                "phase": phase,
                                "ot_cost": float(scores["ot_cost"][0].item()),
                                "aligned_topk_cost": float(scores["aligned_topk_cost"][0].item()),
                                "pca_topk_z": float(scores["pca_topk_z"][0].item()),
                            }
                        )
                        for action in chunk[: args.execute_horizon]:
                            raw_obs, _reward, terminated, truncated, info = env.step(
                                torch.as_tensor(action, device=env.unwrapped.device).unsqueeze(0)
                            )
                            executed += 1
                            success = bool_scalar(info.get("success", False))
                            if success or bool_scalar(terminated) or bool_scalar(truncated):
                                break
                    rows.append(
                        {
                            "split": split,
                            "episode_index": episode_index,
                            "seed": seed,
                            "success": success,
                            "steps": executed,
                            "timeline": timeline,
                            **metadata,
                        }
                    )
                    print(
                        f"[tokenwise-ot] split={split} episode={episode_index + 1}/{args.episodes} "
                        f"steps={executed} success={int(success)}",
                        flush=True,
                    )
            finally:
                env.close()
    finally:
        del policy
        torch.cuda.empty_cache()

    summary = {
        "format": "xvla_stackcube_tokenwise_ot_passive_smoke_v1",
        "checkpoint": str(args.checkpoint),
        "episodes_per_split": args.episodes,
        "execute_horizon": args.execute_horizon,
        "max_episode_steps": args.max_episode_steps,
        "flow_steps": args.flow_steps,
        "splits": ["id", "ood"],
        "rows": rows,
    }
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output / "SMOKE_COMPLETE").write_text("complete\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
