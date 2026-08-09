#!/usr/bin/env python3
"""Passive X-VLA StackCube rollout with all internal and external detectors."""

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

from rlinf.algorithms.vla_fail import velocity_normalized_acc  # noqa: E402
from rlinf.envs.maniskill.stack_cube_variants import (  # noqa: E402
    STACK_CUBE_ID_ENV_ID,
    STACK_CUBE_OOD_ENV_ID,
    STACK_CUBE_TASK,
    register_controlled_stack_cube_variants,
    reset_metadata,
)
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
from tools.build_pick_single_ycb_airplane_external_detector_assets import (  # noqa: E402
    _chw_float_image,
)
from tools.evaluate_stackcube_vla_fail import PandaEndEffectorProjector  # noqa: E402
from tools.libero_plus_failure.rollout_records import single_sample_overlap  # noqa: E402
from tools.evaluate_stackcube_xvla import clip_action_chunk  # noqa: E402
from tools.pick_single_ycb_airplane_external_detectors import (  # noqa: E402
    CRSAILBank,
    OfficialFIDeLMemory,
    crsail_score,
    official_fidel_euclidean_score,
)
from tools.xvla_airplane_failure_detection import (  # noqa: E402
    XVLAMultilayerProbe,
    XVLAMultilayerScorer,
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
    parser.add_argument("--multilayer-assets", type=Path, required=True)
    parser.add_argument("--external-assets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("id", "ood"), required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--execute-horizon", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=150)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--probe-steps", type=int, default=5)
    parser.add_argument("--probe-seed", type=int, default=0)
    parser.add_argument("--diff-timesteps", type=int, default=16)
    parser.add_argument("--diff-noise-samples", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--control-freq", type=int, default=10)
    parser.add_argument("--sim-backend", choices=("physx_cpu", "gpu"), default="physx_cpu")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def move_bank(payload: dict[str, Any], device: torch.device) -> CRSAILBank:
    bank = CRSAILBank.from_state_dict(payload)
    return CRSAILBank(
        values=bank.values.to(device),
        k=bank.k,
        center=None if bank.center is None else bank.center.to(device),
        scale=None if bank.scale is None else bank.scale.to(device),
    )


def main() -> None:
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

    external = torch.load(args.external_assets, map_location="cpu", weights_only=False)
    if external.get("format") != "pick_single_ycb_airplane_external_detector_assets_v1":
        raise ValueError("unexpected external detector asset format")
    fidel = OfficialFIDeLMemory.from_state_dict(external["fidel"]["memory"])
    fidel = OfficialFIDeLMemory(mean=fidel.mean.to(device))
    crsail_state = move_bank(external["crsail"]["observable_state"], device)
    crsail_vision = move_bank(external["crsail"]["vision_resnet18"], device)
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.DEFAULT
    external_encoder = resnet18(weights=weights)
    external_encoder.fc = torch.nn.Identity()
    external_encoder.eval().to(device)
    external_preprocess = weights.transforms()

    import gymnasium as gym  # noqa: F401
    import mani_skill.envs  # noqa: F401

    register_controlled_stack_cube_variants()
    env = gym.make(
        STACK_CUBE_ID_ENV_ID if args.split == "id" else STACK_CUBE_OOD_ENV_ID,
        robot_uids="panda_wristcam",
        num_envs=1,
        obs_mode="rgb",
        control_mode="pd_joint_delta_pos",
        reward_mode="sparse",
        render_mode="rgb_array",
        sim_backend=args.sim_backend,
        sim_config={"sim_freq": 100, "control_freq": args.control_freq},
        sensor_configs={"width": args.image_size, "height": args.image_size},
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
            metadata = reset_metadata(env, split=args.split)
            records = [_extract_record(raw_obs)]
            executed: list[np.ndarray] = []
            timeline: list[dict[str, Any]] = []
            feature_rows: list[np.ndarray] = []
            previous_points = None
            previous_acc_ema = None
            grasped_once = False
            on_cube_once = False
            success = False
            while len(executed) < args.max_episode_steps and not success:
                inputs = policy.prepare(raw_obs, STACK_CUBE_TASK)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    features, encoding = probe.extract(inputs)
                    generated = policy._generate_from_encoding(
                        inputs, encoding, steps=args.flow_steps
                    )
                    internal_scores = scorer.score(features)
                    # DiffDAgger samples extra noise. Preserve both CPU and CUDA
                    # RNG so enabling this passive detector cannot alter future
                    # policy action chunks in the same rollout.
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
                    base_image = _chw_float_image(
                        raw_obs["sensor_data"]["base_camera"]["rgb"]
                    )
                    external_feature = external_encoder(
                        external_preprocess(base_image.unsqueeze(0)).to(device)
                    )
                    fidel_score, fidel_time = official_fidel_euclidean_score(
                        external_feature.unsqueeze(1), fidel
                    )
                    qpos = raw_obs["agent"]["qpos"].reshape(1, -1).to(device, torch.float32)
                    state_score = crsail_score(qpos, crsail_state)
                    vision_score = crsail_score(
                        external_feature, crsail_vision, cosine=True
                    )
                full_chunk = clip_action_chunk(
                    generated.float().cpu().numpy(), low, high, policy.model.num_actions
                )
                qpos_numpy = raw_obs["agent"]["qpos"].detach().cpu().numpy()[0]
                points = projector.project(qpos_numpy, full_chunk)
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
                    "acc": None if acc_ema is None else float(acc_ema),
                    "stac_single": None if stac is None else float(stac),
                    "diffdagger": float(diff_score),
                    "fidel_official": float(fidel_score[0].item()),
                    "crsail_observable_state_k5": float(state_score[0].item()),
                    "crsail_vision_k5": float(vision_score[0].item()),
                }
                ordered_names = sorted(features)
                feature_rows.append(
                    np.stack(
                        [features[name][0].detach().cpu().numpy() for name in ordered_names]
                    ).astype(np.float16)
                )
                timeline.append(
                    {
                        "decision_index": len(timeline),
                        "env_step": len(executed),
                        "scores": scores,
                        "acc_raw": None if acc_raw is None else float(acc_raw),
                        "fidel_matched_time": int(fidel_time[0].item()),
                        "action_chunk": full_chunk.tolist(),
                    }
                )
                previous_points = points
                previous_acc_ema = acc_ema
                for action in full_chunk[: args.execute_horizon]:
                    raw_obs, _, terminated, truncated, info = env.step(
                        torch.as_tensor(action, device=env.unwrapped.device).unsqueeze(0)
                    )
                    executed.append(action.copy())
                    records.append(_extract_record(raw_obs))
                    grasped_once |= bool_scalar(info.get("is_cubeA_grasped", False))
                    on_cube_once |= bool_scalar(info.get("is_cubeA_on_cubeB", False))
                    success |= bool_scalar(info.get("success", False))
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
                task=STACK_CUBE_TASK,
                main_camera=main_camera,
                wrist_camera=wrist_camera,
            )
            video = write_episode_video_durably(
                frames,
                video_dir=args.output_dir / "videos",
                episode_index=episode_index,
                seed=seed,
                fps=args.control_freq,
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
                "success": success,
                "grasped_once": grasped_once,
                "on_cube_once": on_cube_once,
                "steps": len(executed),
                "execute_horizon": args.execute_horizon,
                "timeline": timeline,
                "video": str(video),
                "actions": str(action_path),
                "features": str(feature_path),
                **metadata,
            }
            rows.append(row)
            print(
                f"[xvla-stackcube-failure] split={args.split} "
                f"episode={episode_index + 1}/{args.episodes} seed={seed} "
                f"grasp={int(grasped_once)} on_cube={int(on_cube_once)} "
                f"success={int(success)}",
                flush=True,
            )
    finally:
        probe.close()
        env.close()

    summary = {
        "format": "xvla_stackcube_failure_rollouts_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "split": args.split,
        "episodes": len(rows),
        "successes": sum(int(row["success"]) for row in rows),
        "grasped_once": sum(int(row["grasped_once"]) for row in rows),
        "on_cube_once": sum(int(row["on_cube_once"]) for row in rows),
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
