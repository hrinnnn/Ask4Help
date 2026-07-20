#!/usr/bin/env python3
"""Collect ManiSkill Peg/Plug online AWBC trajectories with a fixed VFD gate.

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
from rlinf.data.maniskill_plug_progress import PlugProgressState  # noqa: E402
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
from rlinf.envs.maniskill.plug_charger_variants import (  # noqa: E402
    PLUG_CHARGER_ID_ENV_ID,
    PLUG_CHARGER_OOD_ENV_ID,
    PLUG_CHARGER_TASK,
    default_plug_instruction,
    register_controlled_plug_charger_variants,
    reset_metadata as plug_reset_metadata,
    wrap_plug_charger_openpi_joint_obs,
)
from rlinf.envs.maniskill.plug_privileged_oracle import (  # noqa: E402
    PlugChargerPrivilegedChunkOracle,
)
from toolkits.lerobot.collect_maniskill_peg_lerobot_joint import (  # noqa: E402
    _build_frames,
    _create_dataset,
    _extract_record,
    _resolve_output_path,
    _select_camera,
    MAIN_CAMERA_CANDIDATES,
    WRIST_CAMERA_CANDIDATES,
)
from toolkits.lerobot.collect_maniskill_plug_lerobot_joint import (  # noqa: E402
    write_episode_video_durably,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("oracle", "calibrate", "online"), required=True)
    parser.add_argument("--task", choices=("peg", "plug"), default="peg")
    parser.add_argument("--split", choices=("id", "ood"), default="id")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-id", default="")
    parser.add_argument("--member-0", type=Path)
    parser.add_argument("--member-1", type=Path)
    parser.add_argument(
        "--pi05-base",
        type=Path,
        help="Base OpenPI PyTorch directory used to instantiate SFT full_weights.pt checkpoints.",
    )
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
        "--compute-vfd",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute the paired-model VFD score at each chunk. Disable for pure policy evaluation.",
    )
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


def resolve_sft_full_weights(path: Path) -> Path | None:
    """Resolve the portable full state dict emitted by RLinf SFT checkpoints."""
    if path.is_file():
        return path
    candidates = (
        path / "actor" / "model_state_dict" / "full_weights.pt",
        path / "model_state_dict" / "full_weights.pt",
        path / "full_weights.pt",
    )
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _load_model(model_path: Path, norm_stats_path: Path, pi05_base: Path | None):
    from rlinf.models import get_model

    fine_tuned_weights = resolve_sft_full_weights(model_path)
    initial_path = pi05_base if fine_tuned_weights is not None else model_path
    if initial_path is None or not initial_path.is_dir():
        raise FileNotFoundError(
            "A --pi05-base directory is required when loading an RLinf SFT full_weights.pt checkpoint"
        )
    model = get_model(_compose_model_config(initial_path, norm_stats_path))
    if model is None:
        raise RuntimeError(f"Could not load pi0.5 model from {initial_path}")
    if fine_tuned_weights is not None:
        state_dict = torch.load(fine_tuned_weights, map_location="cpu", mmap=True, weights_only=True)
        incompatible = model.load_state_dict(state_dict, strict=False)
        allowed_unexpected = {"paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"}
        unexpected = set(incompatible.unexpected_keys) - allowed_unexpected
        if incompatible.missing_keys or unexpected:
            raise RuntimeError(
                "SFT checkpoint is incompatible with the selected pi0.5 base: "
                f"missing={incompatible.missing_keys}, unexpected={sorted(unexpected)}"
            )
    return model.to("cuda").eval().requires_grad_(False)


def _build_env(
    max_episode_steps: int,
    *,
    task: str,
    split: str,
    sim_backend: str = "physx_cpu",
):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    if task == "peg":
        register_rlinf_peg_insertion_side_variants()
        env_id = PEG_INSERTION_SIDE_WIDE_OBSERVER_WIDE_WRIST_ENV_ID
        robot_uids = PANDA_WIDE_WRISTCAM_UID
    else:
        register_controlled_plug_charger_variants()
        env_id = PLUG_CHARGER_ID_ENV_ID if split == "id" else PLUG_CHARGER_OOD_ENV_ID
        robot_uids = "panda_wristcam"
    return gym.make(
        env_id,
        robot_uids=robot_uids,
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


def _wrap_obs(raw_obs: dict[str, Any], info: dict[str, Any], *, task: str) -> dict[str, Any]:
    if task == "plug":
        return wrap_plug_charger_openpi_joint_obs(
            copy.deepcopy(raw_obs),
            task_descriptions=default_plug_instruction(num_envs=1),
        )
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


def _augment_peg_info(env: Any, info: dict[str, Any], event_state: dict[str, torch.Tensor]) -> dict[str, Any]:
    augmented = maybe_augment_peg_insertion_info(
        env=env.unwrapped,
        infos=dict(info),
        event_state=event_state,
        device=env.unwrapped.agent.robot.get_qpos().device,
        is_peg_insertion_side=True,
    )
    env.unwrapped._online_partial_insert = _bool(augmented.get("partial_insert_once"))
    return augmented


def _task_label(task: str) -> str:
    return "insert the peg in the hole" if task == "peg" else PLUG_CHARGER_TASK


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
    compute_vfd: bool,
    episode_index: int,
    dataset_offset: int,
    task: str,
    split: str,
) -> tuple[
    list[dict[str, Any]],
    list[OnlineAWBCFrame],
    list[OnlineAWBCChunk],
    list[float],
    bool,
    dict[str, Any],
]:
    raw_obs, info = env.reset(seed=seed)
    scenario_metadata = plug_reset_metadata(env) if task == "plug" else {"task": "peg"}
    event_state = (
        init_peg_insertion_event_state(num_envs=1, device=raw_obs["agent"]["qpos"].device)
        if task == "peg"
        else PlugProgressState()
    )
    if task == "peg":
        info = _augment_peg_info(env, info, event_state)
    phi_of = (lambda: peg_privileged_phi(info)) if task == "peg" else (lambda: event_state.update(env, info))
    records = [_extract_record(raw_obs)]
    actions: list[np.ndarray] = []
    frames: list[OnlineAWBCFrame] = []
    chunks: list[OnlineAWBCChunk] = []
    vfd_scores: list[float] = []
    oracle = (
        PegPrivilegedChunkOracle(chunk_size=chunk_size)
        if task == "peg"
        else PlugChargerPrivilegedChunkOracle(chunk_size=chunk_size)
    )
    main_camera = _select_camera(records[0].obs, "", MAIN_CAMERA_CANDIDATES, "main")
    wrist_camera = _select_camera(records[0].obs, "", WRIST_CAMERA_CANDIDATES, "wrist")
    terminated = truncated = False
    while not (terminated or truncated):
        phi = phi_of()
        start_frame = len(actions)
        oracle_plan = None
        if mode == "oracle":
            source = "expert"
            vfd = 0.0
            oracle_plan = oracle.plan(env)
            action_sequence = oracle_plan.actions
        else:
            assert member_0 is not None
            env_obs = _wrap_obs(raw_obs, info, task=task)
            with torch.no_grad():
                policy_actions, _ = member_0.predict_action_batch(
                    env_obs=env_obs, mode="eval", compute_values=False
                )
                if compute_vfd:
                    assert member_1 is not None
                    generator = torch.Generator(device="cuda").manual_seed(seed + start_frame)
                    _candidates, score = member_0.compute_vfd_uncertainty(
                        env_obs,
                        member_1,
                        num_action_samples=num_action_samples,
                        generator=generator,
                    )
            if compute_vfd:
                vfd = float(score.reshape(-1)[0].detach().cpu())
                vfd_scores.append(vfd)
                source = "expert" if controller is not None and controller.decide([vfd]).expert_mask.item() else "policy"
            else:
                vfd = float("nan")
                source = "policy"
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
            if task == "peg":
                info = _augment_peg_info(env, info, event_state)
            actions.append(np.asarray(action, dtype=np.float32))
            records.append(_extract_record(raw_obs))
            if _bool(info.get("success", False)) or _bool(terminated) or _bool(truncated):
                break

        next_frame = len(actions)
        phi_next = phi_of()
        success = _bool(info.get("success", False)) or phi_next == 1.0
        chunks.append(
            OnlineAWBCChunk(
                dataset_index=dataset_offset + start_frame,
                episode_index=episode_index,
                frame_index=start_frame,
                next_frame_index=next_frame,
                source=source,
                phi=phi,
                phi_next=phi_next,
                vfd_score=vfd,
                threshold=float(controller.threshold.threshold) if controller else float("nan"),
                success=success,
            )
        )
        if success:
            terminated = True

    return (
        _build_frames(records=records, actions=actions, task=_task_label(task), main_camera=main_camera, wrist_camera=wrist_camera),
        frames,
        chunks,
        vfd_scores,
        bool(_bool(info.get("success", False)) or phi_of() == 1.0),
        scenario_metadata,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode != "oracle":
        required_paths = (args.member_0, args.norm_stats)
        if args.compute_vfd:
            required_paths += (args.member_1,)
        for path in required_paths:
            if path is None or not path.exists():
                raise FileNotFoundError(path)
        if not torch.cuda.is_available():
            raise RuntimeError("pi0.5 VFD online collection requires CUDA")
        member_0 = _load_model(args.member_0, args.norm_stats, args.pi05_base)
        member_1 = _load_model(args.member_1, args.norm_stats, args.pi05_base) if args.compute_vfd else None
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
    env = _build_env(args.max_episode_steps, task=args.task, split=args.split, sim_backend=args.sim_backend)
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
            frames, metadata, chunks, scores, success, scenario_metadata = _run_episode(
                env=env,
                seed=seed,
                mode=args.mode,
                member_0=member_0,
                member_1=member_1,
                controller=controller,
                chunk_size=args.chunk_size,
                num_action_samples=args.num_action_samples,
                compute_vfd=args.compute_vfd,
                episode_index=saved,
                dataset_offset=len(all_frames),
                task=args.task,
                split=args.split,
            )
            row = {"seed": seed, "success": success, "chunks": len(chunks), "task": args.task, "split": args.split}
            if args.task == "plug":
                row.update(scenario_metadata)
            episode_rows.append(row)
            completed_successes = sum(bool(episode["success"]) for episode in episode_rows)
            print(
                f"[rollout] attempt={attempts} seed={seed} success={int(success)} "
                f"chunks={len(chunks)} cumulative_success={completed_successes}/{len(episode_rows)} "
                f"success_rate={completed_successes / len(episode_rows):.3f}",
                flush=True,
            )
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
                write_episode_video_durably(
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
    success_count = sum(bool(episode["success"]) for episode in episode_rows)
    success_rate = success_count / len(episode_rows) if episode_rows else 0.0
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
                "successes": success_count,
                "success_rate": success_rate,
                "compute_vfd": args.compute_vfd,
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
