#!/usr/bin/env python3
"""Passive X-VLA UncoverSpherePlace rollout with internal and action signals."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RLINF_ROOT = ROOT / "RLinf"
sys.path[:0] = [str(ROOT), str(RLINF_ROOT)]

from rlinf.envs.maniskill.uncover_sphere_place import (  # noqa: E402
    UNCOVER_ENV_IDS,
    register_uncover_sphere_place_variants,
)
from rlinf.algorithms.vla_fail import velocity_normalized_acc  # noqa: E402
from toolkits.lerobot.collect_maniskill_peg_lerobot_joint import (  # noqa: E402
    MAIN_CAMERA_CANDIDATES,
    WRIST_CAMERA_CANDIDATES,
    _build_frames,
    _extract_record,
    _select_camera,
)
from toolkits.lerobot.collect_maniskill_plug_lerobot_joint import (  # noqa: E402
    write_episode_video_durably,
)
from tools.evaluate_stackcube_vla_fail import PandaEndEffectorProjector  # noqa: E402
from tools.libero_plus_failure.rollout_records import single_sample_overlap  # noqa: E402
from tools.evaluate_uncover_sphere_place_xvla import (  # noqa: E402
    TASK,
    bool_scalar,
    clip_action_chunk,
)
from tools.xvla_airplane_failure_detection import (  # noqa: E402
    XVLAMultilayerProbe,
    XVLAMultilayerScorer,
)
from tools.xvla_airplane_runtime import XVLAAirplanePolicy  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--multilayer-assets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=tuple(UNCOVER_ENV_IDS), required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--execute-horizon", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=2500)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--probe-steps", type=int, default=5)
    parser.add_argument("--probe-seed", type=int, default=0)
    parser.add_argument("--diff-timesteps", type=int, default=16)
    parser.add_argument("--diff-noise-samples", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "actions").mkdir()
    (args.output_dir / "features").mkdir()

    device = torch.device(args.device)
    policy = XVLAAirplanePolicy(args.checkpoint, args.xvla_root, device=args.device)
    probe = XVLAMultilayerProbe(
        policy.model, probe_seed=args.probe_seed, probe_steps=args.probe_steps
    )
    scorer = XVLAMultilayerScorer(args.multilayer_assets, device=args.device, knn_k=10)
    register_uncover_sphere_place_variants()
    env = gym.make(
        UNCOVER_ENV_IDS[args.split],
        robot_uids="panda_wristcam",
        num_envs=1,
        obs_mode="rgb",
        control_mode="pd_joint_delta_pos",
        reward_mode="sparse",
        render_mode="rgb_array",
        sim_backend="physx_cpu",
        sim_config={"sim_freq": 100, "control_freq": 10},
        sensor_configs={"width": 224, "height": 224},
        max_episode_steps=args.max_episode_steps,
    )
    projector = PandaEndEffectorProjector(env, requested_link=None)
    low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    rows: list[dict[str, Any]] = []
    try:
        for episode_index in range(args.episodes):
            seed = args.seed + episode_index
            torch.manual_seed(seed)
            raw_obs, _ = env.reset(seed=seed)
            metadata = env.unwrapped.reset_metadata()
            records = [_extract_record(raw_obs)]
            executed: list[np.ndarray] = []
            timeline: list[dict[str, Any]] = []
            feature_rows: list[np.ndarray] = []
            previous_points = None
            previous_acc_ema = None
            success = False
            ever_mug_parked = ever_sphere_grasped = False
            ordered_names: list[str] = []
            decision = 0
            while len(executed) < args.max_episode_steps and not success:
                inputs = policy.prepare(raw_obs, TASK)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    features, encoding = probe.extract(inputs)
                    torch.manual_seed(seed * 1000 + decision)
                    generated = policy._generate_from_encoding(
                        inputs, encoding, steps=args.flow_steps
                    )
                    internal_scores = scorer.score(features)
                    cuda_index = device.index
                    if cuda_index is None:
                        cuda_index = torch.cuda.current_device()
                    with torch.random.fork_rng(devices=[cuda_index]):
                        diff_score = policy.diffdagger_score(
                            inputs,
                            encoding,
                            generated.float().cpu().numpy(),
                            num_timesteps=args.diff_timesteps,
                            num_noise_samples=args.diff_noise_samples,
                        )
                full_chunk = clip_action_chunk(
                    generated.float().cpu().numpy(),
                    low,
                    high,
                    policy.model.num_actions,
                )
                qpos = raw_obs["agent"]["qpos"].detach().cpu().numpy()[0]
                points = projector.project(qpos, full_chunk)
                acc_raw = acc_ema = stac = None
                if previous_points is not None:
                    acc_raw, acc_ema = velocity_normalized_acc(
                        previous_points,
                        points,
                        execute_horizon=args.execute_horizon,
                        min_velocity=1e-3,
                        previous_ema=previous_acc_ema,
                        ema_alpha=0.9,
                    )
                    stac = single_sample_overlap(
                        previous_points.numpy(),
                        points.numpy(),
                        execute_horizon=args.execute_horizon,
                    )
                scores = {
                    **internal_scores,
                    "diffdagger": float(diff_score),
                    "acc": acc_ema,
                    "stac_single": stac,
                }
                if not ordered_names:
                    ordered_names = sorted(features)
                feature_rows.append(
                    np.stack(
                        [features[name][0].detach().cpu().numpy() for name in ordered_names]
                    ).astype(np.float16)
                )
                timeline.append(
                    {
                        "decision_index": decision,
                        "env_step": len(executed),
                        "scores": scores,
                        "acc_raw": acc_raw,
                        "action_chunk": full_chunk.tolist(),
                    }
                )
                previous_points = points
                previous_acc_ema = acc_ema
                decision += 1
                for action in full_chunk[: args.execute_horizon]:
                    raw_obs, _, terminated, truncated, _info = env.step(
                        torch.as_tensor(action, device=env.unwrapped.device).unsqueeze(0)
                    )
                    executed.append(action.copy())
                    records.append(_extract_record(raw_obs))
                    phase = env.unwrapped.evaluate()
                    ever_mug_parked |= bool_scalar(phase["ever_mug_parked"])
                    ever_sphere_grasped |= bool_scalar(phase["ever_sphere_grasped"])
                    success = bool_scalar(phase["success"])
                    if success or bool_scalar(terminated) or bool_scalar(truncated):
                        break

            main_camera = _select_camera(
                records[0].obs, "", ("base_camera",) + MAIN_CAMERA_CANDIDATES, "main"
            )
            wrist_camera = _select_camera(
                records[0].obs, "", ("hand_camera",) + WRIST_CAMERA_CANDIDATES, "wrist"
            )
            frames = _build_frames(
                records=records,
                actions=executed,
                task=TASK,
                main_camera=main_camera,
                wrist_camera=wrist_camera,
            )
            video = write_episode_video_durably(
                frames,
                video_dir=args.output_dir / "videos",
                episode_index=episode_index,
                seed=seed,
                fps=10,
            )
            action_path = args.output_dir / "actions" / f"episode_{episode_index:06d}.npy"
            feature_path = args.output_dir / "features" / f"episode_{episode_index:06d}.npz"
            np.save(action_path, np.asarray(executed, dtype=np.float32))
            np.savez_compressed(
                feature_path,
                layer_names=np.asarray(ordered_names),
                features=np.asarray(feature_rows, dtype=np.float16),
            )
            row = {
                "episode_index": episode_index,
                "seed": seed,
                "split": args.split,
                "success": bool(success),
                "ever_mug_parked": bool(ever_mug_parked),
                "ever_sphere_grasped": bool(ever_sphere_grasped),
                "steps": len(executed),
                "decisions": decision,
                "timeline": timeline,
                "video": str(video),
                "actions": str(action_path),
                "features": str(feature_path),
                **metadata,
            }
            rows.append(row)
            print(
                f"[xvla-uncover-detectors] {args.split} "
                f"episode={episode_index + 1}/{args.episodes} "
                f"success={int(success)}", flush=True
            )
    finally:
        probe.close()
        env.close()

    summary = {
        "format": "xvla_uncover_failure_rollouts_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "split": args.split,
        "episodes": len(rows),
        "successes": sum(int(row["success"]) for row in rows),
        "ever_mug_parked": sum(int(row["ever_mug_parked"]) for row in rows),
        "ever_sphere_grasped": sum(int(row["ever_sphere_grasped"]) for row in rows),
        "execute_horizon": args.execute_horizon,
        "max_episode_steps": args.max_episode_steps,
        "flow_steps": args.flow_steps,
        "probe_steps": args.probe_steps,
        "probe_seed": args.probe_seed,
        "diff_timesteps": args.diff_timesteps,
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
