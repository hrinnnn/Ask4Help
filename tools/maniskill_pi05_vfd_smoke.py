#!/usr/bin/env python3
"""Run a two-checkpoint pi0.5 VFD smoke on one ManiSkill observation."""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--member-0", type=Path, required=True)
    parser.add_argument("--member-1", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-action-samples", type=int, default=1)
    parser.add_argument(
        "--velocity-eval-times",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 0.9],
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def collect_observation(seed: int) -> dict:
    import gymnasium as gym
    import mani_skill  # noqa: F401

    from rlinf.envs.maniskill.peg_insertion_side_variants import (
        PANDA_WIDE_WRISTCAM_UID,
        PEG_INSERTION_SIDE_WIDE_OBSERVER_WIDE_WRIST_ENV_ID,
        default_peg_instruction,
        register_rlinf_peg_insertion_side_variants,
        wrap_rlt_openpi_joint_obs,
    )

    register_rlinf_peg_insertion_side_variants()
    env = gym.make(
        PEG_INSERTION_SIDE_WIDE_OBSERVER_WIDE_WRIST_ENV_ID,
        robot_uids=PANDA_WIDE_WRISTCAM_UID,
        num_envs=1,
        obs_mode="rgb",
        control_mode="pd_joint_delta_pos",
        sim_backend="gpu",
        reward_mode="sparse",
        sim_config={"sim_freq": 100, "control_freq": 10},
        sensor_configs={"width": 384, "height": 384},
    )
    try:
        raw_obs, infos = env.reset(seed=seed)
        observation = wrap_rlt_openpi_joint_obs(
            raw_obs,
            infos=infos,
            task_descriptions=default_peg_instruction(num_envs=1),
            num_envs=1,
            device=raw_obs["agent"]["qpos"].device,
            is_peg_insertion_side=True,
        )
        return {
            key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in observation.items()
        }
    finally:
        env.close()


def compose_model_config(model_path: Path, norm_stats_path: Path):
    from hydra import compose, initialize_config_dir

    rlinf_root = Path(os.environ["RLINF_ROOT"]).resolve()
    config_dir = rlinf_root / "examples" / "embodiment" / "config"
    os.environ["EMBODIED_PATH"] = str(config_dir.parent)
    os.environ["PI05_MODEL_PATH"] = str(model_path)
    os.environ["NORM_STATS_PATH"] = str(norm_stats_path)
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="maniskill_awbc_collect_openpi_pi05")
    model_cfg = copy.deepcopy(cfg.rollout.model)
    model_cfg.model_path = str(model_path)
    model_cfg.openpi_data.norm_stats_path = str(norm_stats_path)
    return model_cfg


def load_model(model_path: Path, norm_stats_path: Path):
    from rlinf.models import get_model

    model = get_model(compose_model_config(model_path, norm_stats_path))
    if model is None:
        raise RuntimeError("RLinf did not build the requested OpenPI model.")
    model = model.to("cuda").eval()
    model.requires_grad_(False)
    return model


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    args = parse_args()
    for path in (args.member_0, args.member_1, args.norm_stats):
        if not path.exists():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the pi0.5 VFD smoke.")

    started_at = time.time()
    observation = collect_observation(args.seed)
    synchronize()

    load_started = time.perf_counter()
    member_0 = load_model(args.member_0, args.norm_stats)
    member_1 = load_model(args.member_1, args.norm_stats)
    synchronize()
    load_seconds = time.perf_counter() - load_started
    allocated_after_load = torch.cuda.memory_allocated()

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    vfd_started = time.perf_counter()
    action_candidates, scores = member_0.compute_vfd_uncertainty(
        observation,
        member_1,
        num_action_samples=args.num_action_samples,
        velocity_eval_times=args.velocity_eval_times,
        generator=generator,
    )
    synchronize()
    vfd_seconds = time.perf_counter() - vfd_started

    result = {
        "status": "passed",
        "started_at_unix": started_at,
        "member_0": str(args.member_0.resolve()),
        "member_1": str(args.member_1.resolve()),
        "norm_stats": str(args.norm_stats.resolve()),
        "seed": args.seed,
        "num_action_samples": args.num_action_samples,
        "velocity_eval_times": args.velocity_eval_times,
        "action_candidates_shape": list(action_candidates.shape),
        "vfd_scores": scores.detach().float().cpu().tolist(),
        "finite": bool(torch.isfinite(scores).all().item()),
        "load_seconds": load_seconds,
        "vfd_seconds": vfd_seconds,
        "cuda_memory_allocated_gib_after_load": allocated_after_load / 2**30,
        "cuda_memory_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(),
    }
    if not result["finite"]:
        raise RuntimeError(f"VFD returned a non-finite score: {scores}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
