#!/usr/bin/env python3
"""Calibrate object-variation PCA and Diff gates from successful ID policy runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RLINF_ROOT = ROOT / "RLinf"
sys.path[:0] = [str(ROOT), str(RLINF_ROOT)]

from rlinf.algorithms.vla_fail import PCAResidualStatistics, pca_residual_score  # noqa: E402
from rlinf.envs.maniskill.pick_single_ycb_object_variation import (  # noqa: E402
    PICK_SINGLE_YCB_OBJECT_ID_ENV_ID,
    PICK_SINGLE_YCB_OBJECT_TASK,
    register_controlled_pick_single_ycb_object_variants,
)
from tools.evaluate_pick_single_ycb_object_variation_pi05 import _load_model  # noqa: E402
from tools.pick_single_ycb_airplane_eval_common import clip_action_chunk  # noqa: E402


def model_observation(raw_obs):
    states = raw_obs["agent"]["qpos"]
    return {
        "main_images": raw_obs["sensor_data"]["base_camera"]["rgb"],
        "wrist_images": raw_obs["sensor_data"]["hand_camera"]["rgb"],
        "extra_view_images": None,
        "states": states,
        "task_descriptions": [PICK_SINGLE_YCB_OBJECT_TASK],
        "task_ids": torch.zeros(1, dtype=torch.long, device=states.device),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pi05-base", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--detector-assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=13000)
    parser.add_argument("--target-successes", type=int, default=20)
    parser.add_argument("--max-attempts", type=int, default=50)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    register_controlled_pick_single_ycb_object_variants()
    model = _load_model(args.checkpoint, args.norm_stats, args.pi05_base)
    payload = torch.load(args.detector_assets, map_location="cpu", weights_only=False)
    pca_stats = PCAResidualStatistics.from_state_dict(payload["statistics"]["bridge_pca_residual"])
    prior = payload["fixed_prior"].to("cuda")
    env = gym.make(
        PICK_SINGLE_YCB_OBJECT_ID_ENV_ID,
        num_envs=1,
        robot_uids="panda_wristcam",
        obs_mode="rgb",
        control_mode="pd_joint_delta_pos",
        reward_mode="sparse",
        sim_backend="physx_cpu",
        sim_config={"sim_freq": 100, "control_freq": 10},
        render_mode="rgb_array",
        max_episode_steps=200,
    )
    successes: list[dict] = []
    attempts: list[dict] = []
    try:
        for attempt in range(args.max_attempts):
            seed = args.seed_start + attempt
            raw_obs, _ = env.reset(seed=seed)
            pca_trace: list[float] = []
            diff_trace: list[float] = []
            success = False
            steps = 0
            while steps < 200 and not success:
                with torch.inference_mode():
                    features = model.extract_multilayer_llmd_features(
                        model_observation(raw_obs), prior, action_expert_fractions=(0.5,), capture_vlm=True, include_action_expert_final=True
                    )
                    pca_trace.append(float(pca_residual_score(features["vlm_bridge_final_mean"], pca_stats)[0].item()))
                    predicted, result = model.predict_action_batch(env_obs=model_observation(raw_obs), mode="train")
                    model_actions = result["forward_inputs"]["model_action"]
                    diff_trace.append(float(model.compute_diffdagger_uncertainty(model_observation(raw_obs), model_actions, num_timesteps=16, num_noise_samples=1).reshape(-1)[0].detach().cpu()))
                chunk = clip_action_chunk(predicted.detach().float().cpu().numpy(), np.asarray(env.action_space.low, dtype=np.float32).reshape(-1), np.asarray(env.action_space.high, dtype=np.float32).reshape(-1), 5)
                for action in chunk:
                    raw_obs, _reward, terminated, truncated, info = env.step(torch.as_tensor(action, device=env.unwrapped.device).reshape(1, -1))
                    steps += 1
                    success = bool(np.asarray(info.get("success", False)).reshape(-1)[0])
                    if success or bool(np.asarray(terminated).reshape(-1)[0]) or bool(np.asarray(truncated).reshape(-1)[0]):
                        break
            row = {"seed": seed, "success": success, "steps": steps, "pca_trace": pca_trace, "diff_trace": diff_trace}
            attempts.append(row)
            if success:
                successes.append(row)
                print(f"calibration_success={len(successes)}/{args.target_successes} seed={seed}", flush=True)
            if len(successes) >= args.target_successes:
                break
    finally:
        env.close()
    if len(successes) < args.target_successes:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"format": "pick_single_ycb_object_variation_gate_calibration_v1", "attempts": attempts, "successes": len(successes), "passed": False}, indent=2) + "\n", encoding="utf-8")
        raise SystemExit(2)
    pca_maxima = [max(row["pca_trace"]) for row in successes if row["pca_trace"]]
    diff_windows = [max(min(trace[i : i + 2]) for i in range(len(trace) - 1)) for row in successes for trace in [row["diff_trace"]] if len(trace) >= 2]
    output = {
        "format": "pick_single_ycb_object_variation_gate_calibration_v1",
        "checkpoint": str(args.checkpoint),
        "norm_stats": str(args.norm_stats),
        "detector_assets": str(args.detector_assets),
        "q": 0.95,
        "patience": 2,
        "successful_id_rollouts": len(successes),
        "attempts": len(attempts),
        "pca_threshold": float(np.quantile(pca_maxima, 0.95)),
        "diff_threshold": float(np.quantile(diff_windows, 0.95)),
        "pca_maxima": pca_maxima,
        "diff_patience_scores": diff_windows,
        "scores": [value for row in successes for value in row["diff_trace"]],
        "successes": successes,
        "passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    (args.output.parent / "CALIBRATION_COMPLETE").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

