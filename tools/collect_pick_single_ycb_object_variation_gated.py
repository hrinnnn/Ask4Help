#!/usr/bin/env python3
"""Collect the four object-variation expert-data branches."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RLINF_ROOT = ROOT / "RLinf"
sys.path[:0] = [str(ROOT), str(RLINF_ROOT)]

from rlinf.algorithms.diffdagger import DiffDAggerQueryGate  # noqa: E402
from rlinf.algorithms.vla_fail import PCAResidualStatistics, pca_residual_score  # noqa: E402
from rlinf.envs.maniskill.pick_single_ycb_object_variation import (  # noqa: E402
    PICK_SINGLE_YCB_OBJECT_ID_ENV_ID,
    PICK_SINGLE_YCB_OBJECT_OOD_ENV_ID,
    PICK_SINGLE_YCB_OBJECT_TASK,
    register_controlled_pick_single_ycb_object_variants,
    reset_metadata,
)
from toolkits.lerobot.collect_maniskill_peg_lerobot_joint import (  # noqa: E402
    _build_frames,
    _convert_solver_action_to_joint_delta,
    _create_dataset,
    _extract_record,
    _joint_delta_arm_bounds,
    _select_camera,
    _to_numpy,
)
from toolkits.lerobot.diagnose_pick_single_ycb_object_variation_oracle import (  # noqa: E402
    run_oracle,
)
from tools.evaluate_pick_single_ycb_object_variation_pi05 import (  # noqa: E402
    _load_model,
)
from tools.pick_single_ycb_airplane_eval_common import clip_action_chunk  # noqa: E402

LOG = logging.getLogger("collect_pick_single_ycb_object_variation_gated")
EXECUTE_HORIZON = 5
POLICY_PREFIX_LIMIT = 50
TASK_HORIZON = 200


def model_observation(raw_obs: dict[str, Any]) -> dict[str, Any]:
    states = raw_obs["agent"]["qpos"]
    return {
        "main_images": raw_obs["sensor_data"]["base_camera"]["rgb"],
        "wrist_images": raw_obs["sensor_data"]["hand_camera"]["rgb"],
        "extra_view_images": None,
        "states": states,
        "task_descriptions": [PICK_SINGLE_YCB_OBJECT_TASK],
        "task_ids": torch.zeros(1, dtype=torch.long, device=states.device),
    }


def _env(split: str, control_mode: str):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    register_controlled_pick_single_ycb_object_variants()
    env_id = PICK_SINGLE_YCB_OBJECT_ID_ENV_ID if split == "id" else PICK_SINGLE_YCB_OBJECT_OOD_ENV_ID
    return gym.make(
        env_id,
        num_envs=1,
        robot_uids="panda_wristcam",
        obs_mode="rgb",
        control_mode=control_mode,
        reward_mode="sparse",
        sim_backend="physx_cpu",
        sim_config={"sim_freq": 100, "control_freq": 10},
        render_mode="rgb_array",
        max_episode_steps=TASK_HORIZON,
    )


def _success(value: Any) -> bool:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return bool(np.asarray(value).reshape(-1)[0])


def _save_video(frames: list[dict[str, Any]], destination: Path) -> None:
    from toolkits.lerobot.collect_maniskill_pick_single_ycb_object_variation_lerobot import _write_video

    _write_video(frames, destination, fps=10)


def _expert_takeover(env, *, seed: int, records: list[Any], actions: list[np.ndarray], lower: np.ndarray, upper: np.ndarray):
    original_step = env.step
    expert_actions: list[np.ndarray] = []
    expert_records: list[Any] = []

    def step_hook(solver_action, *args, **kwargs):
        current_qpos = env.unwrapped.agent.robot.get_qpos()[0].detach().cpu().numpy()
        converted = _convert_solver_action_to_joint_delta(current_qpos, solver_action, lower, upper)
        result = original_step(torch.as_tensor(converted, device=env.unwrapped.device).reshape(1, -1), *args, **kwargs)
        expert_actions.append(converted.astype(np.float32))
        expert_records.append(_extract_record(result[0]))
        return result

    env.step = step_hook  # type: ignore[method-assign]
    try:
        oracle = run_oracle(env, seed=seed, reset_before=False, force_planner_pd_joint_pos=True)
    finally:
        env.step = original_step  # type: ignore[method-assign]
    if not oracle["accepted"]:
        return None, oracle
    records.extend(expert_records)
    actions.extend(expert_actions)
    return oracle, oracle


def _pca_score(model, raw_obs, prior, stats) -> float:
    with torch.inference_mode():
        features = model.extract_multilayer_llmd_features(
            model_observation(raw_obs),
            prior,
            action_expert_fractions=(0.5,),
            capture_vlm=True,
            include_action_expert_final=True,
        )
    return float(pca_residual_score(features["vlm_bridge_final_mean"], stats)[0].item())


def _attempt(args, split, seed, model, pca_prior, pca_stats, diff_gate, lower, upper):
    env = _env(split, "pd_joint_delta_pos")
    raw_obs, _ = env.reset(seed=seed)
    metadata = reset_metadata(env, split=split)
    records = [_extract_record(raw_obs)]
    actions: list[np.ndarray] = []
    sources: list[str] = []
    timeline: list[dict[str, Any]] = []
    expert_start: int | None = None
    gate_reason = None
    success = False
    try:
        while len(actions) < TASK_HORIZON and not success:
            if args.method == "offline_oracle" and expert_start is None:
                expert_start = len(actions)
                gate_reason = "offline_oracle_from_reset"
            elif args.method == "failure_recovery" and expert_start is None and len(actions) >= POLICY_PREFIX_LIMIT:
                expert_start = len(actions)
                gate_reason = "no_strict_success_after_autonomous_prefix"
            elif args.method in ("bridge_pca", "diffdagger") and expert_start is None and len(actions) < POLICY_PREFIX_LIMIT:
                score = None
                threshold = None
                if args.method == "bridge_pca":
                    score = _pca_score(model, raw_obs, pca_prior, pca_stats)
                    threshold = args.bridge_threshold
                    if score > threshold:
                        expert_start = len(actions)
                        gate_reason = "bridge_pca_threshold"
                else:
                    with torch.inference_mode():
                        predicted, result = model.predict_action_batch(env_obs=model_observation(raw_obs), mode="train")
                    model_actions = result["forward_inputs"]["model_action"]
                    score = float(model.compute_diffdagger_uncertainty(model_observation(raw_obs), model_actions, num_timesteps=16, num_noise_samples=1).reshape(-1)[0].detach().cpu())
                    decision = diff_gate.decide(torch.tensor([score], device="cuda"))
                    threshold = float(decision.threshold)
                    if bool(decision.query_mask.item()):
                        expert_start = len(actions)
                        gate_reason = "diffdagger_threshold"
                timeline.append({"step": len(actions), "score": score, "threshold": threshold, "controller": "policy"})

            if expert_start is not None:
                oracle, oracle_detail = _expert_takeover(env, seed=seed, records=records, actions=actions, lower=lower, upper=upper)
                success = bool(oracle_detail["accepted"])
                sources.extend(["expert"] * (len(actions) - len(sources)))
                break

            with torch.inference_mode():
                predicted, _ = model.predict_action_batch(env_obs=model_observation(raw_obs), mode="eval")
            chunk = clip_action_chunk(
                predicted.detach().float().cpu().numpy(),
                np.asarray(env.action_space.low, dtype=np.float32).reshape(-1),
                np.asarray(env.action_space.high, dtype=np.float32).reshape(-1),
                args.execute_horizon,
            )
            for action in chunk:
                raw_obs, _reward, terminated, truncated, info = env.step(torch.as_tensor(action, device=env.unwrapped.device).reshape(1, -1))
                actions.append(np.asarray(action, dtype=np.float32))
                sources.append("policy")
                records.append(_extract_record(raw_obs))
                success = _success(info.get("success", False))
                if success or _success(terminated) or _success(truncated) or len(actions) >= TASK_HORIZON:
                    break
        metadata.update({"method": args.method, "seed": seed, "split": split, "success": success, "expert_start_step": expert_start, "expert_action_steps": 0 if expert_start is None else len(actions) - expert_start, "gate_reason": gate_reason, "timeline": timeline, "sources": sources})
        return records, actions, expert_start, metadata
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("bridge_pca", "diffdagger", "failure_recovery", "offline_oracle"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--pi05-base", type=Path)
    parser.add_argument("--norm-stats", type=Path)
    parser.add_argument("--detector-assets", type=Path)
    parser.add_argument("--diff-calibration", type=Path)
    parser.add_argument("--bridge-threshold", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-id", type=Path, required=True)
    parser.add_argument("--target-expert-trajectories", type=int, default=100)
    parser.add_argument("--max-attempts", type=int, default=500)
    parser.add_argument("--id-seed", type=int, default=16000)
    parser.add_argument("--ood-seed", type=int, default=16001)
    parser.add_argument("--only-split", choices=("id", "ood"))
    args = parser.parse_args()
    if args.output_dir.exists() or args.repo_id.exists():
        raise FileExistsError("refusing to overwrite gated collection output")
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "raw_archive/videos").mkdir(parents=True)
    (args.output_dir / "raw_archive/actions").mkdir(parents=True)
    if args.method != "offline_oracle":
        if not args.checkpoint or not args.pi05_base or not args.norm_stats:
            raise ValueError("gated methods require checkpoint, pi05-base, and norm stats")
        model = _load_model(args.checkpoint, args.norm_stats, args.pi05_base)
    else:
        model = None
    pca_prior = pca_stats = None
    if args.method == "bridge_pca":
        payload = torch.load(args.detector_assets, map_location="cpu", weights_only=False)
        pca_prior = payload["fixed_prior"].to("cuda")
        pca_stats = PCAResidualStatistics.from_state_dict(payload["statistics"]["bridge_pca_residual"])
    diff_gate = None
    if args.method == "diffdagger":
        scores = json.loads(args.diff_calibration.read_text())["scores"]
        diff_gate = DiffDAggerQueryGate(scores, alpha=0.95, patience=2)
    probe_env = _env("id", "pd_joint_delta_pos")
    lower, upper = _joint_delta_arm_bounds(probe_env)
    probe_env.close()
    accepted = 0
    rows: list[dict[str, Any]] = []
    dataset = None
    for attempt in range(args.max_attempts):
        if accepted >= args.target_expert_trajectories:
            break
        split = args.only_split or ("id" if attempt % 2 == 0 else "ood")
        seed = (args.id_seed if split == "id" else args.ood_seed) + (attempt // 2 if not args.only_split else attempt)
        result = _attempt(args, split, seed, model, pca_prior, pca_stats, diff_gate, lower, upper)
        records, actions, expert_start, metadata = result
        if not actions:
            continue
        row = {"attempt": attempt, **metadata, "accepted": False}
        raw_frames = _build_frames(records=records, actions=actions, task=PICK_SINGLE_YCB_OBJECT_TASK, main_camera="base_camera", wrist_camera="hand_camera")
        _save_video(raw_frames, args.output_dir / "raw_archive/videos" / f"episode_{attempt:06d}_seed_{seed:06d}.mp4")
        np.save(args.output_dir / "raw_archive/actions" / f"episode_{attempt:06d}_seed_{seed:06d}.npy", np.asarray(actions, dtype=np.float32))
        if metadata["success"] and expert_start is not None and len(actions) > expert_start:
            suffix_records = records[expert_start:]
            suffix_actions = actions[expert_start:]
            suffix_frames = _build_frames(records=suffix_records, actions=suffix_actions, task=PICK_SINGLE_YCB_OBJECT_TASK, main_camera="base_camera", wrist_camera="hand_camera")
            if dataset is None:
                dataset = _create_dataset(repo_id=str(args.repo_id), image_shape=tuple(suffix_frames[0]["image"].shape), wrist_image_shape=tuple(suffix_frames[0]["wrist_image"].shape), fps=10, image_writer_threads=4, image_writer_processes=2)
            for frame in suffix_frames:
                dataset.add_frame(frame)
            dataset.save_episode()
            _save_video(suffix_frames, args.output_dir / f"accepted_suffix_videos/episode_{accepted:06d}_seed_{seed:06d}.mp4")
            row["accepted"] = True
            accepted += 1
        rows.append(row)
        if accepted % 10 == 0:
            print(f"accepted={accepted}/{args.target_expert_trajectories} attempts={attempt + 1}", flush=True)
    if dataset is not None and getattr(dataset, "image_writer", None) is not None:
        dataset.image_writer.wait_until_done()
    summary = {"format": "pick_single_ycb_object_variation_gated_collection_v1", "method": args.method, "accepted": accepted, "target": args.target_expert_trajectories, "raw_attempts": len(rows), "video_count": len(list((args.output_dir / "raw_archive/videos").glob("*.mp4"))), "object_variation_only": True}
    (args.output_dir / "attempts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    marker = "COLLECTION_COMPLETE" if accepted == args.target_expert_trajectories else "COLLECTION_FAILED"
    (args.output_dir / marker).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if accepted != args.target_expert_trajectories:
        raise SystemExit(2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
