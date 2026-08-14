#!/usr/bin/env python3
"""Collect OpenDrawer Offline, Failure-Recovery, or Robot-Gated expert data."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RLINF_ROOT = Path(os.environ.get("ASK4HELP_RLINF_ROOT", ROOT / "RLinf"))
sys.path[:0] = [str(ROOT), str(RLINF_ROOT)]

from rlinf.algorithms.vla_fail import (  # noqa: E402
    KNNStatistics,
    LLMDStatistics,
    PCAResidualStatistics,
    fixed_gaussian_prior,
    knn_score,
    llmd_score,
    pca_residual_score,
)
from rlinf.envs.maniskill.open_drawer_retrieve_place_spec import (  # noqa: E402
    ENV_IDS,
    TASK_INSTRUCTION,
    reset_metadata,
)
from toolkits.lerobot.collect_open_drawer_retrieve_place_lerobot import (  # noqa: E402
    _build_env,
    _create_dataset,
    _extract_record,
    _joint_delta_arm_bounds,
    _replay,
    _run_reference,
)
from toolkits.lerobot.collect_maniskill_peg_lerobot_joint import (  # noqa: E402
    MAIN_CAMERA_CANDIDATES,
    WRIST_CAMERA_CANDIDATES,
    _build_frames,
    _select_camera,
)
from toolkits.lerobot.collect_maniskill_pick_single_ycb_airplane_lerobot import (  # noqa: E402
    write_episode_video_durably,
)
from toolkits.lerobot.validate_open_drawer_retrieve_place_oracle import (  # noqa: E402
    PandaPosePlannerClient,
    continue_episode,
)
from tools.evaluate_open_drawer_id_pi05 import _bool, _model_obs  # noqa: E402
from tools.maniskill_pi05_vfd_online_awbc import _load_model  # noqa: E402


OOD_SPLITS = ("handle_ood", "grasp_ood", "goal_ood")


def _stats_from_spec(spec: dict[str, Any]) -> Any:
    kind = spec["kind"]
    if kind == "llmd":
        return LLMDStatistics.from_state_dict(spec["statistics"])
    if kind == "knn":
        return KNNStatistics.from_state_dict(spec["statistics"])
    if kind == "pca_residual":
        return PCAResidualStatistics.from_state_dict(spec["statistics"])
    raise ValueError(f"unknown detector kind: {kind}")


def _load_gate(path: Path) -> tuple[dict[str, tuple[str, str, Any]], torch.Tensor]:
    asset_path = path / "detector_assets.pt" if path.is_dir() else path
    payload = torch.load(asset_path, map_location="cpu", weights_only=False)
    if payload.get("format") != "open_drawer_internal_detector_assets_v1":
        raise ValueError(f"unsupported detector assets: {asset_path}")
    detectors = {
        name: (spec["layer"], spec["kind"], _stats_from_spec(spec))
        for name, spec in payload["detectors"].items()
    }
    return detectors, payload["fixed_prior"]


def _score(feature: torch.Tensor, kind: str, stats: Any) -> float:
    if kind == "llmd":
        return float(llmd_score(feature, stats).max().item())
    if kind == "knn":
        return float(knn_score(feature, stats).max().item())
    return float(pca_residual_score(feature, stats).max().item())


def _build_split_env(split: str, args: argparse.Namespace, *, control_mode: str):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    import rlinf.envs.maniskill.open_drawer_retrieve_place  # noqa: F401

    return gym.make(
        ENV_IDS[split], robot_uids="panda_wristcam", num_envs=1, obs_mode="rgb",
        control_mode=control_mode, reward_mode="sparse", render_mode="rgb_array",
        sim_backend=args.sim_backend, sim_config={"sim_freq": 100, "control_freq": 10},
        sensor_configs={"width": 384, "height": 384}, max_episode_steps=400,
    )


class _RecordingProxy:
    def __init__(self, env: Any, records: list[Any], actions: list[np.ndarray]):
        self._env = env
        self._records = records
        self._actions = actions

    @property
    def unwrapped(self):
        return self._env.unwrapped

    def step(self, action, *args, **kwargs):
        observation, reward, terminated, truncated, info = self._env.step(action, *args, **kwargs)
        self._actions.append(np.asarray(action, dtype=np.float32).reshape(-1))
        self._records.append(_extract_record(observation))
        return observation, reward, terminated, truncated, info

    def __getattr__(self, name: str):
        return getattr(self._env, name)


def _oracle_full(solver_env: Any, replay_env: Any, seed: int, planner: PandaPosePlannerClient, lower: np.ndarray, upper: np.ndarray):
    reference = _run_reference(solver_env, seed, planner)
    if reference is None:
        return None
    _reference_records, solver_actions, metadata = reference
    replay = _replay(replay_env, seed, solver_actions, lower, upper)
    if replay is None:
        return None
    records, actions = replay
    return records, actions, metadata


def _policy_until_trigger(*, env: Any, model: Any, prior: torch.Tensor, detectors: dict[str, tuple[str, str, Any]],
                          thresholds: dict[str, float], seed: int, method: str, execute_horizon: int,
                          max_policy_steps: int):
    raw_obs, _info = env.reset(seed=seed)
    initial_metadata = reset_metadata(env, split=env.unwrapped.rlinf_split)
    prefix_records = [_extract_record(raw_obs)]
    prefix_actions: list[np.ndarray] = []
    timeline: list[dict[str, Any]] = []
    ever_opened = ever_grasped = ever_lifted = False
    trigger: dict[str, Any] | None = None
    alarm_streak = 0
    low = np.asarray(env.action_space.low).reshape(-1)
    high = np.asarray(env.action_space.high).reshape(-1)
    while len(prefix_actions) < max_policy_steps:
        env_obs = _model_obs(raw_obs)
        with torch.inference_mode():
            features = model.extract_multilayer_llmd_features(
                env_obs, prior, include_action_expert_final=True
            )
            predicted, _ = model.predict_action_batch(env_obs=env_obs, mode="eval", compute_values=False)
        scores = {name: _score(features[layer], kind, stats) for name, (layer, kind, stats) in detectors.items()}
        alarms = {name: score >= thresholds[name] for name, score in scores.items()}
        alarmed = any(alarms.values())
        alarm_streak = alarm_streak + 1 if alarmed else 0
        timeline.append({"decision_index": len(timeline), "env_step": len(prefix_actions), "scores": scores, "alarms": alarms})
        explicit_failure = (
            (ever_grasped and not _bool(env.unwrapped.agent.is_grasping(env.unwrapped.obj)))
            or (len(prefix_actions) >= 80 and not ever_lifted)
        )
        if (method == "robot_gated" and alarm_streak >= 2) or (method == "failure_recovery" and explicit_failure):
            trigger = {
                "method": method,
                "env_step": len(prefix_actions),
                "decision_index": len(timeline) - 1,
                "alarm_streak": alarm_streak,
                "explicit_failure": bool(explicit_failure),
                "scores": scores,
                "alarms": alarms,
            }
            break
        chunk = np.clip(predicted.detach().float().cpu().numpy()[0][:execute_horizon], low, high).astype(np.float32)
        for action in chunk:
            raw_obs, _reward, terminated, truncated, info = env.step(torch.as_tensor(action, device=env.unwrapped.device).unsqueeze(0))
            prefix_actions.append(action)
            prefix_records.append(_extract_record(raw_obs))
            ever_opened |= _bool(info.get("ever_drawer_opened", False))
            ever_grasped |= _bool(info.get("ever_grasped", False))
            ever_lifted |= _bool(info.get("ever_lifted", False))
            if _bool(info.get("success", False)) or _bool(terminated) or _bool(truncated):
                return prefix_records, prefix_actions, timeline, None, initial_metadata
            if len(prefix_actions) >= max_policy_steps:
                break
    return prefix_records, prefix_actions, timeline, trigger, initial_metadata


def _write_video(records: list[Any], actions: list[np.ndarray], out: Path, index: int, seed: int) -> str:
    main = _select_camera(records[0].obs, "", ("base_camera",) + MAIN_CAMERA_CANDIDATES, "main")
    wrist = _select_camera(records[0].obs, "", ("hand_camera",) + WRIST_CAMERA_CANDIDATES, "wrist")
    frames = _build_frames(records=records, actions=actions, task=TASK_INSTRUCTION, main_camera=main, wrist_camera=wrist)
    path = write_episode_video_durably(frames, video_dir=out, episode_index=index, seed=seed, fps=10)
    return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("offline_oracle", "failure_recovery", "robot_gated"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pi05-base", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--detector-assets", type=Path)
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target-per-source", type=int, default=50)
    parser.add_argument("--max-attempts", type=int, default=400)
    parser.add_argument("--id-seed-start", type=int, default=81000)
    parser.add_argument("--ood-seed-start", type=int, default=82000)
    parser.add_argument("--execute-horizon", type=int, default=5)
    parser.add_argument("--max-policy-steps", type=int, default=240)
    parser.add_argument("--sim-backend", choices=("physx_cpu", "gpu"), default="physx_cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    if args.method != "offline_oracle" and (args.detector_assets is None or args.thresholds is None):
        raise ValueError("gated and failure-recovery collection require detector assets and thresholds")
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "raw_videos").mkdir()
    (args.output_root / "pids").mkdir(exist_ok=True)
    manifest_path = args.output_root / "raw_attempts.jsonl"
    accepted_path = args.output_root / "accepted_experts.jsonl"
    dataset = None
    planner = PandaPosePlannerClient()
    model = None
    detectors: dict[str, tuple[str, str, Any]] = {}
    thresholds: dict[str, float] = {}
    prior = None
    if args.method != "offline_oracle":
        detectors, asset_prior = _load_gate(args.detector_assets)
        payload = json.loads(args.thresholds.read_text())
        thresholds = {name: float(spec["threshold"]) for name, spec in payload["detectors"].items()}
        if set(thresholds) != set(detectors):
            raise ValueError("threshold names do not match detector assets")
        model = _load_model(args.checkpoint, args.norm_stats, args.pi05_base)
        prior = asset_prior.to("cuda")

    counts = {"id": 0, "ood": 0}
    accepted = 0
    attempts = 0
    try:
        while attempts < args.max_attempts and min(counts.values()) < args.target_per_source:
            source = "id" if attempts % 2 == 0 else "ood"
            split = "id" if source == "id" else OOD_SPLITS[(attempts // 2) % len(OOD_SPLITS)]
            seed = (args.id_seed_start if source == "id" else args.ood_seed_start) + (attempts // 2)
            attempts += 1
            raw_row: dict[str, Any] = {"attempt_index": attempts - 1, "source": source, "split": split, "seed": seed, "method": args.method}
            full_records: list[Any] | None = None
            full_actions: list[np.ndarray] | None = None
            expert_records: list[Any] | None = None
            expert_actions: list[np.ndarray] | None = None
            trigger = None
            metadata: dict[str, Any] = {}
            success = False
            if args.method == "offline_oracle":
                solver_env = _build_split_env(split, args, control_mode="pd_joint_pos")
                replay_env = _build_split_env(split, args, control_mode="pd_joint_delta_pos")
                try:
                    lower, upper = _joint_delta_arm_bounds(replay_env)
                    result = _oracle_full(solver_env, replay_env, seed, planner, lower, upper)
                    if result is not None:
                        full_records, full_actions, metadata = result
                        metadata["split"] = split
                        success = True
                        expert_records, expert_actions = full_records, full_actions
                finally:
                    solver_env.close()
                    replay_env.close()
            else:
                policy_env = _build_split_env(split, args, control_mode="pd_joint_delta_pos")
                result = _policy_until_trigger(env=policy_env, model=model, prior=prior, detectors=detectors,
                                               thresholds=thresholds, seed=seed, method=args.method,
                                               execute_horizon=args.execute_horizon, max_policy_steps=args.max_policy_steps)
                prefix_records, prefix_actions, timeline, trigger, initial_metadata = result
                metadata["reset"] = initial_metadata
                raw_row["policy_timeline"] = timeline
                if trigger is not None:
                    expert_records = [prefix_records[-1]]
                    expert_actions = []
                    proxy = _RecordingProxy(policy_env, expert_records, expert_actions)
                    expert_result = continue_episode(proxy, planner, seed=seed)
                    success = bool(expert_result.get("success", False))
                    full_records = prefix_records + expert_records[1:]
                    full_actions = prefix_actions + expert_actions
                    metadata["expert_result"] = expert_result
                else:
                    full_records, full_actions = prefix_records, prefix_actions
                policy_env.close()
            if full_records is not None and full_actions is not None and full_actions:
                raw_row["success"] = bool(success)
                raw_row["num_actions"] = len(full_actions)
                raw_row["video"] = _write_video(full_records, full_actions, args.output_root / "raw_videos", attempts - 1, seed)
            else:
                raw_row["success"] = False
                raw_row["num_actions"] = 0
                raw_row["video"] = None
            raw_row["trigger"] = trigger
            raw_row["reset_metadata"] = metadata
            with manifest_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(raw_row, sort_keys=True) + "\n")
            if not success or not expert_records or not expert_actions:
                print(f"[open-drawer-collect] attempt={attempts} source={source} split={split} success=0 trigger={bool(trigger)}", flush=True)
                continue
            if counts[source] >= args.target_per_source:
                print(
                    f"[open-drawer-collect] attempt={attempts} source={source} "
                    f"success=1 quota_full=1; keeping raw attempt only",
                    flush=True,
                )
                continue
            if dataset is None:
                preview = _build_frames(records=expert_records, actions=expert_actions, task=TASK_INSTRUCTION,
                                        main_camera=_select_camera(expert_records[0].obs, "", ("base_camera",) + MAIN_CAMERA_CANDIDATES, "main"),
                                        wrist_camera=_select_camera(expert_records[0].obs, "", ("hand_camera",) + WRIST_CAMERA_CANDIDATES, "wrist"))
                dataset = _create_dataset(repo_id=str(args.output_root / "lerobot_dataset"), image_shape=tuple(preview[0]["image"].shape),
                                          wrist_image_shape=tuple(preview[0]["wrist_image"].shape), fps=10, image_writer_threads=4, image_writer_processes=4)
            main = _select_camera(expert_records[0].obs, "", ("base_camera",) + MAIN_CAMERA_CANDIDATES, "main")
            wrist = _select_camera(expert_records[0].obs, "", ("hand_camera",) + WRIST_CAMERA_CANDIDATES, "wrist")
            for frame in _build_frames(records=expert_records, actions=expert_actions, task=TASK_INSTRUCTION, main_camera=main, wrist_camera=wrist):
                dataset.add_frame(frame)
            dataset.save_episode()
            counts[source] += 1
            accepted += 1
            accepted_row = {"episode_index": accepted - 1, "attempt_index": attempts - 1, "source": source, "split": split,
                            "seed": seed, "expert_actions": len(expert_actions), "trigger": trigger,
                            "raw_video": raw_row["video"], "metadata": metadata}
            with accepted_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(accepted_row, sort_keys=True) + "\n")
            print(f"[open-drawer-collect] accepted={accepted} id={counts['id']} ood={counts['ood']} method={args.method}", flush=True)
    finally:
        if dataset is not None and getattr(dataset, "image_writer", None) is not None:
            dataset.image_writer.wait_until_done()
        planner.close()
        if model is not None:
            del model
            torch.cuda.empty_cache()
    summary = {
        "format": "open_drawer_dagger_collection_v1", "method": args.method,
        "target_per_source": args.target_per_source, "accepted": accepted,
        "accepted_id": counts["id"], "accepted_ood": counts["ood"], "attempts": attempts,
        "dataset": str(args.output_root / "lerobot_dataset") if dataset is not None else None,
        "raw_attempts": str(manifest_path), "accepted_manifest": str(accepted_path),
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if min(counts.values()) < args.target_per_source:
        raise SystemExit(f"collection incomplete: id={counts['id']} ood={counts['ood']} target={args.target_per_source}")
    (args.output_root / "COLLECTION_COMPLETE").write_text("balanced accepted expert data and raw attempts verified\n")


if __name__ == "__main__":
    main()
