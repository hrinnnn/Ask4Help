#!/usr/bin/env python3
"""Collect threshold-free airplane rollouts for every failure baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RLINF_ROOT = ROOT / "RLinf"
sys.path[:0] = [str(ROOT), str(RLINF_ROOT)]

from rlinf.algorithms.vla_fail import (  # noqa: E402
    KNNStatistics,
    LLMDStatistics,
    PCAResidualStatistics,
    knn_score,
    llmd_score,
    pca_residual_score,
    velocity_normalized_acc,
)
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
    _build_frames,
    _extract_record,
    _select_camera,
)
from tools.evaluate_pick_single_ycb_airplane_pi05 import model_observation  # noqa: E402
from tools.evaluate_stackcube_vla_fail import PandaEndEffectorProjector  # noqa: E402
from tools.libero_plus_failure.rollout_records import single_sample_overlap  # noqa: E402
from tools.maniskill_pi05_vfd_online_awbc import _load_model  # noqa: E402
from tools.pick_single_ycb_airplane_eval_common import clip_action_chunk  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def bool_scalar(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu().reshape(-1)[0].item())
    return bool(np.asarray(value).reshape(-1)[0])


def save_npz_durably(path: Path, **arrays: np.ndarray) -> None:
    """Build seek-heavy NPZ locally, then stream the finished file to OSSFS."""

    with tempfile.TemporaryDirectory(prefix="airplane-detector-") as directory:
        local = Path(directory) / path.name
        np.savez_compressed(local, **arrays)
        with np.load(local) as payload:
            if set(payload.files) != set(arrays):
                raise RuntimeError("local feature archive failed its key audit")
        shutil.copyfile(local, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pi05-base", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--detector-assets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("id", "ood"), required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--execute-horizon", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--control-freq", type=int, default=10)
    parser.add_argument("--sim-backend", choices=("physx_cpu", "gpu"), default="physx_cpu")
    parser.add_argument("--min-velocity", type=float, default=1e-3)
    parser.add_argument("--ema-alpha", type=float, default=0.9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    actions_dir = args.output_dir / "actions"
    features_dir = args.output_dir / "features"
    actions_dir.mkdir()
    features_dir.mkdir()

    asset = torch.load(args.detector_assets, map_location="cpu", weights_only=False)
    if asset.get("format") != "pick_single_ycb_airplane_detector_assets_v1":
        raise ValueError("wrong detector asset format")
    stats = asset["statistics"]
    bridge_llmd = LLMDStatistics.from_state_dict(stats["bridge_llmd"])
    bridge_knn = KNNStatistics.from_state_dict(stats["bridge_deep_knn"])
    bridge_pca = PCAResidualStatistics.from_state_dict(stats["bridge_pca_residual"])
    final_llmd = LLMDStatistics.from_state_dict(stats["final_llmd"])
    prior = asset["fixed_prior"].to("cuda")

    import gymnasium as gym  # noqa: F401
    import mani_skill.envs  # noqa: F401

    register_controlled_pick_single_ycb_airplane_variants()
    model = _load_model(args.checkpoint, args.norm_stats, args.pi05_base)
    env_args = argparse.Namespace(
        split=args.split,
        image_size=args.image_size,
        control_freq=args.control_freq,
        max_episode_steps=args.max_episode_steps,
        sim_backend=args.sim_backend,
    )
    env = _build_env(env_args, control_mode="pd_joint_delta_pos")
    projector = PandaEndEffectorProjector(env, requested_link=None)
    low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    rows: list[dict[str, Any]] = []
    try:
        for episode_index in range(args.episodes):
            seed = args.seed + episode_index
            raw_obs, info = env.reset(seed=seed)
            metadata = reset_metadata(env, split=args.split)
            records = [_extract_record(raw_obs)]
            executed: list[np.ndarray] = []
            timeline: list[dict[str, Any]] = []
            bridge_features: list[np.ndarray] = []
            final_features: list[np.ndarray] = []
            previous_points = None
            previous_acc_ema = None
            ever_grasped = False
            relaxed_distance = False
            relaxed_static = False
            success = False
            while len(executed) < args.max_episode_steps and not success:
                with torch.inference_mode():
                    features = model.extract_multilayer_llmd_features(
                        model_observation(raw_obs), prior, include_action_expert_final=True
                    )
                    bridge = features["vlm_bridge_final_mean"]
                    final = features["action_expert_final"]
                    predicted, _ = model.predict_action_batch(
                        env_obs=model_observation(raw_obs), mode="eval", compute_values=False
                    )
                full_chunk = clip_action_chunk(
                    predicted.detach().float().cpu().numpy(), low, high, int(model.config.action_horizon)
                )
                qpos = raw_obs["agent"]["qpos"].detach().cpu().numpy()[0]
                points = projector.project(qpos, full_chunk)
                acc_raw = acc_ema = stac = None
                if previous_points is not None:
                    acc_raw, acc_ema = velocity_normalized_acc(
                        previous_points,
                        points,
                        execute_horizon=args.execute_horizon,
                        min_velocity=args.min_velocity,
                        previous_ema=previous_acc_ema,
                        ema_alpha=args.ema_alpha,
                    )
                    stac = single_sample_overlap(
                        previous_points.numpy(), points.numpy(), execute_horizon=args.execute_horizon
                    )
                scores = {
                    "bridge_deep_knn": float(knn_score(bridge, bridge_knn)[0].item()),
                    "bridge_llmd": float(llmd_score(bridge, bridge_llmd)[0].item()),
                    "bridge_pca_residual": float(pca_residual_score(bridge, bridge_pca)[0].item()),
                    "final_llmd": float(llmd_score(final, final_llmd)[0].item()),
                    "acc": None if acc_ema is None else float(acc_ema),
                    "stac_single": None if stac is None else float(stac),
                }
                timeline.append(
                    {
                        "decision_index": len(timeline),
                        "env_step": len(executed),
                        "scores": scores,
                        "acc_raw": None if acc_raw is None else float(acc_raw),
                        "action_chunk": full_chunk.tolist(),
                    }
                )
                bridge_features.append(bridge.detach().cpu().float().numpy())
                final_features.append(final.detach().cpu().float().numpy())
                previous_points = points
                previous_acc_ema = acc_ema
                for action in full_chunk[: args.execute_horizon]:
                    raw_obs, _reward, terminated, truncated, info = env.step(
                        torch.as_tensor(action, device=env.unwrapped.device).unsqueeze(0)
                    )
                    executed.append(action.copy())
                    records.append(_extract_record(raw_obs))
                    evaluation = env.unwrapped.evaluate()
                    distance = float(torch.linalg.norm(evaluation["obj_to_goal_pos"], dim=-1).reshape(-1)[0])
                    static = bool_scalar(evaluation["is_robot_static"])
                    ever_grasped |= bool_scalar(evaluation["is_grasped"])
                    relaxed_distance |= distance <= 0.10
                    relaxed_static |= distance <= 0.10 and static
                    success = bool_scalar(info.get("success", False))
                    if success or bool_scalar(terminated) or bool_scalar(truncated):
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
            actions_path = actions_dir / f"episode_{episode_index:06d}_seed_{seed:06d}.npy"
            feature_path = features_dir / f"episode_{episode_index:06d}_seed_{seed:06d}.npz"
            np.save(actions_path, np.asarray(executed, dtype=np.float32))
            save_npz_durably(
                feature_path,
                bridge=np.asarray(bridge_features, dtype=np.float32),
                action_expert_final=np.asarray(final_features, dtype=np.float32),
            )
            row = {
                "episode_index": episode_index,
                "seed": seed,
                "success": success,
                "relaxed_10cm_static": relaxed_static,
                "relaxed_10cm_distance": relaxed_distance,
                "ever_grasped": ever_grasped,
                "steps": len(executed),
                "execute_horizon": args.execute_horizon,
                "timeline": timeline,
                "video": str(video_path),
                "actions": str(actions_path),
                "features": str(feature_path),
                **metadata,
            }
            rows.append(row)
            print(
                f"[airplane-detectors] split={args.split} episode={episode_index + 1}/{args.episodes} "
                f"seed={seed} success={int(success)}",
                flush=True,
            )
    finally:
        env.close()
        del model
        torch.cuda.empty_cache()

    summary = {
        "format": "pick_single_ycb_airplane_detector_rollouts_v1",
        "checkpoint": str(args.checkpoint),
        "norm_stats": str(args.norm_stats),
        "detector_assets": str(args.detector_assets),
        "detector_assets_sha256": sha256(args.detector_assets),
        "split": args.split,
        "episodes": len(rows),
        "successes": sum(int(row["success"]) for row in rows),
        "baselines": [
            "bridge_deep_knn",
            "bridge_llmd",
            "bridge_pca_residual",
            "final_llmd",
            "acc",
            "stac_single",
            "final_llmd_or_acc (derived during threshold sweep)",
        ],
        "thresholds": None,
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
