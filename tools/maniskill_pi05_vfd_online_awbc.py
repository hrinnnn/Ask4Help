#!/usr/bin/env python3
"""Collect PegInsertionSide online AWBC trajectories with a fixed VFD gate.

The runner deliberately uses one environment.  A ManiSkill motion-planning
oracle can then plan from the exact current simulator state after a VFD query,
which is the required semantics for a one-chunk expert intervention.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _bootstrap_rlinf() -> Path:
    root = Path(__file__).resolve().parents[1] / "RLinf"
    if str(root) not in os.sys.path:
        os.sys.path.insert(0, str(root))
    os.environ.setdefault("RLINF_ROOT", str(root))
    return root


RLINF_ROOT = _bootstrap_rlinf()

from rlinf.algorithms.online_awbc import (  # noqa: E402
    FixedThresholdChunkController,
    FixedVFDThreshold,
    uniformly_spaced_chunk_indices,
)
from rlinf.data.maniskill_peg_progress import peg_privileged_phi  # noqa: E402
from rlinf.data.online_awbc import (  # noqa: E402
    OnlineAWBCChunk,
    OnlineAWBCFrame,
    build_online_awbc_manifest,
)
from rlinf.envs.maniskill.peg_insertion_side_variants import (  # noqa: E402
    PANDA_WIDE_WRISTCAM_UID,
    PEG_INSERTION_SIDE_WIDE_OBSERVER_WIDE_WRIST_ENV_ID,
    default_peg_instruction,
    init_peg_insertion_event_state,
    maybe_augment_peg_insertion_info,
    register_rlinf_peg_insertion_side_variants,
    wrap_rlt_openpi_joint_obs,
)
from rlinf.envs.maniskill.peg_privileged_oracle import (  # noqa: E402
    PegPrivilegedChunkOracle,
)
from toolkits.lerobot.collect_maniskill_peg_lerobot_joint import (  # noqa: E402
    _build_frames,
    _create_dataset,
    _extract_record,
    _resolve_output_path,
    _select_camera,
    _write_episode_video,
    MAIN_CAMERA_CANDIDATES,
    WRIST_CAMERA_CANDIDATES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("oracle", "calibrate", "online"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-id", default="")
    parser.add_argument("--member-0", type=Path)
    parser.add_argument("--member-1", type=Path)
    parser.add_argument("--norm-stats", type=Path)
    parser.add_argument("--threshold-path", type=Path)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--successes", type=int, default=20)
    parser.add_argument("--max-attempts", type=int, default=100)
    parser.add_argument("--samples-per-episode", type=int, default=5)
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-action-samples", type=int, default=5)
    parser.add_argument(
        "--sim-backend",
        default="physx_cpu",
        choices=("physx_cpu", "gpu"),
        help="Use CPU physics for compatibility with ManiSkill's official planner; rendering remains on CUDA.",
    )
    parser.add_argument("--save-videos", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _compose_model_config(model_path: Path, norm_stats_path: Path):
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict

    config_dir = RLINF_ROOT / "examples" / "embodiment" / "config"
    os.environ["EMBODIED_PATH"] = str(config_dir.parent)
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="maniskill_awbc_collect_openpi_pi05")
    model_cfg = copy.deepcopy(cfg.rollout.model)
    with open_dict(model_cfg):
        model_cfg.model_path = str(model_path)
        model_cfg.openpi_data.norm_stats_path = str(norm_stats_path)
        model_cfg.is_lora = False
        model_cfg.load_to_device = True
    return model_cfg


def _load_model(model_path: Path, norm_stats_path: Path):
    from rlinf.models import get_model

    model = get_model(_compose_model_config(model_path, norm_stats_path))
    if model is None:
        raise RuntimeError(f"Could not load pi0.5 model from {model_path}")
    return model.to("cuda").eval().requires_grad_(False)


def _build_env(max_episode_steps: int, *, sim_backend: str = "physx_cpu"):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    register_rlinf_peg_insertion_side_variants()
    return gym.make(
        PEG_INSERTION_SIDE_WIDE_OBSERVER_WIDE_WRIST_ENV_ID,
        robot_uids=PANDA_WIDE_WRISTCAM_UID,
        num_envs=1,
        obs_mode="rgb",
        control_mode="pd_joint_delta_pos",
        sim_backend=sim_backend,
        reward_mode="sparse",
        sim_config={"sim_freq": 100, "control_freq": 10},
        sensor_configs={"width": 384, "height": 384},
        max_episode_steps=max_episode_steps,
        render_mode="rgb_array",
    )


def _wrap_obs(raw_obs: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
    copied = copy.deepcopy(raw_obs)
    return wrap_rlt_openpi_joint_obs(
        copied,
        infos=info,
        task_descriptions=default_peg_instruction(num_envs=1),
        num_envs=1,
        device=raw_obs["agent"]["qpos"].device,
        is_peg_insertion_side=True,
    )


def _action_chunk(actions: torch.Tensor, chunk_size: int) -> np.ndarray:
    array = actions.detach().float().cpu().numpy()
    if array.ndim == 3:
        array = array[0]
    if array.ndim == 1:
        array = array.reshape(-1, 8)
    if array.ndim != 2 or array.shape[1] != 8:
        raise ValueError(f"Expected pi0.5 action chunk [H,8], got {array.shape}")
    if array.shape[0] < chunk_size:
        array = np.concatenate([array, np.repeat(array[-1:], chunk_size - array.shape[0], 0)])
    return array[:chunk_size].astype(np.float32)


def _bool(value: Any) -> bool:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return bool(np.asarray(value, dtype=bool).reshape(-1).any())


def _augment_info(env: Any, info: dict[str, Any], event_state: dict[str, torch.Tensor]) -> dict[str, Any]:
    augmented = maybe_augment_peg_insertion_info(
        env=env.unwrapped,
        infos=dict(info),
        event_state=event_state,
        device=env.unwrapped.agent.robot.get_qpos().device,
        is_peg_insertion_side=True,
    )
    env.unwrapped._online_partial_insert = _bool(augmented.get("partial_insert_once"))
    return augmented


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _run_episode(
    *,
    env: Any,
    seed: int,
    mode: str,
    member_0: Any | None,
    member_1: Any | None,
    controller: FixedThresholdChunkController | None,
    chunk_size: int,
    num_action_samples: int,
    episode_index: int,
    dataset_offset: int,
) -> tuple[list[dict[str, Any]], list[OnlineAWBCFrame], list[OnlineAWBCChunk], list[float], bool]:
    raw_obs, info = env.reset(seed=seed)
    event_state = init_peg_insertion_event_state(num_envs=1, device=raw_obs["agent"]["qpos"].device)
    info = _augment_info(env, info, event_state)
    records = [_extract_record(raw_obs)]
    actions: list[np.ndarray] = []
    frames: list[OnlineAWBCFrame] = []
    chunks: list[OnlineAWBCChunk] = []
    vfd_scores: list[float] = []
    oracle = PegPrivilegedChunkOracle(chunk_size=chunk_size)
    main_camera = _select_camera(records[0].obs, "", MAIN_CAMERA_CANDIDATES, "main")
    wrist_camera = _select_camera(records[0].obs, "", WRIST_CAMERA_CANDIDATES, "wrist")
    terminated = truncated = False
    while not (terminated or truncated):
        phi = peg_privileged_phi(info)
        start_frame = len(actions)
        oracle_plan = None
        if mode == "oracle":
            source = "expert"
            vfd = 0.0
            oracle_plan = oracle.plan(env)
            action_sequence = oracle_plan.actions
        else:
            assert member_0 is not None and member_1 is not None
            env_obs = _wrap_obs(raw_obs, info)
            with torch.no_grad():
                policy_actions, _ = member_0.predict_action_batch(
                    env_obs=env_obs, mode="eval", compute_values=False
                )
                generator = torch.Generator(device="cuda").manual_seed(seed + start_frame)
                _candidates, score = member_0.compute_vfd_uncertainty(
                    env_obs,
                    member_1,
                    num_action_samples=num_action_samples,
                    generator=generator,
                )
            vfd = float(score.reshape(-1)[0].detach().cpu())
            vfd_scores.append(vfd)
            source = "expert" if controller is not None and controller.decide([vfd]).expert_mask.item() else "policy"
            if source == "expert":
                oracle_plan = oracle.plan(env)
                action_sequence = oracle_plan.actions
            else:
                action_sequence = _action_chunk(policy_actions, chunk_size)

        for step_index, action in enumerate(action_sequence):
            if oracle_plan is not None:
                action = oracle_plan.action_at(raw_obs["agent"]["qpos"], step_index)
            frames.append(
                OnlineAWBCFrame(
                    dataset_index=dataset_offset + len(actions),
                    episode_index=episode_index,
                    frame_index=len(actions),
                    source=source,
                    phi=phi,
                )
            )
            env_action = torch.as_tensor(action, device=raw_obs["agent"]["qpos"].device).unsqueeze(0)
            raw_obs, _reward, terminated, truncated, info = env.step(env_action)
            info = _augment_info(env, info, event_state)
            actions.append(np.asarray(action, dtype=np.float32))
            records.append(_extract_record(raw_obs))
            if _bool(info.get("success", False)) or _bool(terminated) or _bool(truncated):
                break

        next_frame = len(actions)
        success = _bool(info.get("success", False)) or peg_privileged_phi(info) == 1.0
        chunks.append(
            OnlineAWBCChunk(
                dataset_index=dataset_offset + start_frame,
                episode_index=episode_index,
                frame_index=start_frame,
                next_frame_index=next_frame,
                source=source,
                phi=phi,
                phi_next=peg_privileged_phi(info),
                vfd_score=vfd,
                threshold=float(controller.threshold.threshold) if controller else float("nan"),
                success=success,
            )
        )
        if success:
            terminated = True

    return (
        _build_frames(records=records, actions=actions, task="insert the peg in the hole", main_camera=main_camera, wrist_camera=wrist_camera),
        frames,
        chunks,
        vfd_scores,
        bool(_bool(info.get("success", False)) or peg_privileged_phi(info) == 1.0),
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode != "oracle":
        for path in (args.member_0, args.member_1, args.norm_stats):
            if path is None or not path.exists():
                raise FileNotFoundError(path)
        if not torch.cuda.is_available():
            raise RuntimeError("pi0.5 VFD online collection requires CUDA")
        member_0 = _load_model(args.member_0, args.norm_stats)
        member_1 = _load_model(args.member_1, args.norm_stats)
    else:
        member_0 = member_1 = None

    threshold = None
    if args.mode == "online":
        if args.threshold_path is None or not args.threshold_path.is_file():
            raise FileNotFoundError("online mode requires an existing --threshold-path")
        threshold_data = json.loads(args.threshold_path.read_text())
        threshold = FixedVFDThreshold(**threshold_data)
    controller = None if threshold is None else FixedThresholdChunkController(threshold)
    target = args.successes if args.mode == "calibrate" else args.episodes
    env = _build_env(args.max_episode_steps, sim_backend=args.sim_backend)
    dataset = None
    all_frames: list[OnlineAWBCFrame] = []
    all_chunks: list[OnlineAWBCChunk] = []
    calibration_scores: list[float] = []
    episode_rows: list[dict[str, Any]] = []
    saved = attempts = 0
    try:
        while saved < target and attempts < args.max_attempts:
            seed = args.seed + attempts
            attempts += 1
            frames, metadata, chunks, scores, success = _run_episode(
                env=env,
                seed=seed,
                mode=args.mode,
                member_0=member_0,
                member_1=member_1,
                controller=controller,
                chunk_size=args.chunk_size,
                num_action_samples=args.num_action_samples,
                episode_index=saved,
                dataset_offset=len(all_frames),
            )
            episode_rows.append({"seed": seed, "success": success, "chunks": len(chunks)})
            if args.mode == "calibrate":
                if success:
                    calibration_scores.extend(scores[index] for index in uniformly_spaced_chunk_indices(len(scores), samples_per_episode=args.samples_per_episode))
                    saved += 1
                continue
            if args.mode == "oracle" and not success:
                continue
            if not frames:
                continue
            if dataset is None:
                if not args.repo_id:
                    raise ValueError("--repo-id is required for oracle and online modes")
                dataset = _create_dataset(
                    repo_id=args.repo_id,
                    image_shape=tuple(frames[0]["image"].shape),
                    wrist_image_shape=tuple(frames[0]["wrist_image"].shape),
                    fps=10,
                    image_writer_threads=4,
                    image_writer_processes=4,
                )
            for frame in frames:
                dataset.add_frame(frame)
            dataset.save_episode()
            all_frames.extend(metadata)
            all_chunks.extend(chunks)
            if args.save_videos:
                _write_episode_video(
                    frames,
                    video_dir=args.output_dir / "videos",
                    episode_index=saved,
                    seed=seed,
                    fps=10,
                )
            saved += 1
    finally:
        if dataset is not None and getattr(dataset, "image_writer", None) is not None:
            dataset.image_writer.wait_until_done()
        env.close()

    if saved < target:
        raise RuntimeError(f"completed {saved}/{target} targets after {attempts} attempts")
    (args.output_dir / "episodes.json").write_text(json.dumps(episode_rows, indent=2) + "\n")
    if args.mode == "calibrate":
        calibrated = FixedVFDThreshold.calibrate(calibration_scores, quantile=args.quantile)
        (args.output_dir / "fixed_vfd_threshold.json").write_text(
            json.dumps(calibrated.__dict__, indent=2) + "\n"
        )
        _write_jsonl(args.output_dir / "calibration_scores.jsonl", [{"score": score} for score in calibration_scores])
        return

    manifest = build_online_awbc_manifest(all_frames, all_chunks)
    _write_jsonl(args.output_dir / "online_chunks.jsonl", [chunk.diagnostic_dict() for chunk in all_chunks])
    _write_jsonl(args.output_dir / "progress.jsonl", manifest)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "mode": args.mode,
                "episodes": saved,
                "dataset": str(_resolve_output_path(args.repo_id)),
                "progress_manifest": str(args.output_dir / "progress.jsonl"),
                "chunks": len(all_chunks),
                "expert_chunks": sum(chunk.source == "expert" for chunk in all_chunks),
                "policy_chunks": sum(chunk.source == "policy" for chunk in all_chunks),
                "timestamp": time.time(),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
