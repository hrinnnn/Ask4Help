#!/usr/bin/env python3
"""Collect OOD expert data for four X-VLA airplane DAgger/BC groups."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "RLinf")]

from rlinf.algorithms.vla_fail import PCAResidualStatistics, pca_residual_score  # noqa: E402
from rlinf.envs.maniskill.pick_single_ycb_airplane_variants import (  # noqa: E402
    PICK_SINGLE_YCB_AIRPLANE_TASK,
    register_controlled_pick_single_ycb_airplane_variants,
    reset_metadata,
)
from toolkits.lerobot.collect_maniskill_peg_lerobot_joint import (  # noqa: E402
    MAIN_CAMERA_CANDIDATES,
    WRIST_CAMERA_CANDIDATES,
    _build_frames,
    _create_dataset,
    _extract_record,
    _joint_delta_arm_bounds,
    _select_camera,
)
from toolkits.lerobot.collect_maniskill_pick_single_ycb_airplane_lerobot import _build_env  # noqa: E402
from tools.collect_pick_single_ycb_airplane_gated_dagger import (  # noqa: E402
    ORACLE_CLOSE_MAX_STEPS,
    ORACLE_STABLE_GRASP_STEPS,
    _plan_and_execute_expert,
    _save_raw_attempt,
    admitted_expert_suffix,
)
from tools.pick_single_ycb_airplane_eval_common import clip_action_chunk  # noqa: E402
from tools.xvla_airplane_runtime import XVLAAirplanePolicy  # noqa: E402

EXECUTE_HORIZON = 5
POLICY_EPISODE_HORIZON = 150
FAILURE_RECOVERY_STEP = 50
TASK_HORIZON = 250
FIXED_TIMING_METHOD = "fixed_timing"


def consecutive_gate(score: float, threshold: float, count: int, patience: int) -> tuple[int, bool]:
    count = count + 1 if score > threshold else 0
    return count, count >= patience


def _bool(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu().reshape(-1)[0].item())
    return bool(np.asarray(value).reshape(-1)[0])


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def task_state(env: Any) -> np.ndarray:
    """Compact geometry/proprioception state used for timing deviation."""
    base = env.unwrapped
    object_p = base.obj.pose.p.reshape(-1, 3)[0].detach().cpu().numpy()
    goal_p = base.goal_site.pose.p.reshape(-1, 3)[0].detach().cpu().numpy()
    tcp_p = base.agent.tcp.pose.p.reshape(-1, 3)[0].detach().cpu().numpy()
    qpos = base.agent.robot.get_qpos().reshape(-1).detach().cpu().numpy()
    grasped = _bool(base.agent.is_grasping(base.obj))
    strict_success = _bool(base.evaluate().get("success", False))
    return np.concatenate(
        [
            object_p,
            goal_p,
            tcp_p,
            tcp_p - object_p,
            goal_p - object_p,
            qpos[:9],
            np.asarray([grasped, strict_success], dtype=np.float32),
        ]
    ).astype(np.float32)


def alternating_split(attempt_index: int) -> str:
    return "id" if attempt_index % 2 == 0 else "ood"


def _env_args(args: argparse.Namespace, split: str) -> argparse.Namespace:
    return argparse.Namespace(
        split=split,
        image_size=args.image_size,
        control_freq=args.control_freq,
        max_episode_steps=TASK_HORIZON,
        sim_backend=args.sim_backend,
    )


def _run_attempt(
    *,
    method: str,
    split: str,
    seed: int,
    policy: XVLAAirplanePolicy | None,
    policy_env: Any,
    solver_env: Any,
    lower: np.ndarray,
    upper: np.ndarray,
    pca_statistics: PCAResidualStatistics,
    pca_threshold: float,
    diff_threshold: float,
    diff_patience: int,
    diff_timesteps: int,
    flow_steps: int,
    timing_step: int | None = None,
) -> tuple[list[Any], list[np.ndarray], int | None, dict[str, Any]]:
    raw_obs, _ = policy_env.reset(seed=seed)
    records = [_extract_record(raw_obs)]
    actions: list[np.ndarray] = []
    sources: list[str] = []
    timeline: list[dict[str, Any]] = []
    task_states = [task_state(policy_env)]
    expert_start = 0 if method == "offline_oracle" else None
    gate_count = 0

    while len(actions) < POLICY_EPISODE_HORIZON and expert_start is None:
        step = len(actions)
        if method == FIXED_TIMING_METHOD:
            if timing_step is None or timing_step < 0:
                raise ValueError("fixed_timing requires a non-negative --timing-step")
            if step >= timing_step:
                expert_start = step
                timeline.append(
                    {
                        "env_step": step,
                        "controller": "expert",
                        "score": None,
                        "threshold": None,
                        "alarm": True,
                        "timing_reason": f"fixed_step_{timing_step}",
                    }
                )
                break
        if method == "failure_recovery" and step >= FAILURE_RECOVERY_STEP:
            expert_start = step
            timeline.append({"env_step": step, "controller": "expert", "alarm": True})
            break
        assert policy is not None
        predicted, feature, inputs, encoding = policy.predict(
            raw_obs,
            PICK_SINGLE_YCB_AIRPLANE_TASK,
            seed=seed * 1000 + step,
            steps=flow_steps,
        )
        score = threshold = None
        alarm = False
        if method == "vlm_pool_pca":
            score = float(pca_residual_score(feature.unsqueeze(1), pca_statistics)[0])
            threshold = pca_threshold
            gate_count, alarm = consecutive_gate(score, threshold, gate_count, 1)
        elif method == "diffdagger":
            score = policy.diffdagger_score(
                inputs,
                encoding,
                predicted,
                num_timesteps=diff_timesteps,
                num_noise_samples=1,
            )
            threshold = diff_threshold
            gate_count, alarm = consecutive_gate(score, threshold, gate_count, diff_patience)
        if alarm:
            expert_start = step
            timeline.append({"env_step": step, "controller": "expert", "score": score, "threshold": threshold, "alarm": True})
            break
        timeline.append({"env_step": step, "controller": "policy", "score": score, "threshold": threshold, "alarm": False})
        low = np.asarray(policy_env.action_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(policy_env.action_space.high, dtype=np.float32).reshape(-1)
        for action in clip_action_chunk(predicted, low, high, EXECUTE_HORIZON):
            raw_obs, _, terminated, truncated, _ = policy_env.step(
                torch.as_tensor(action, device=policy_env.unwrapped.device).unsqueeze(0)
            )
            actions.append(np.asarray(action, dtype=np.float32))
            sources.append("policy")
            records.append(_extract_record(raw_obs))
            task_states.append(task_state(policy_env))
            if _bool(terminated) or _bool(truncated):
                break
        if _bool(terminated) or _bool(truncated):
            break

    oracle_result = None
    strict_success = False
    if expert_start is not None:
        expert_records, expert_actions, oracle_result = _plan_and_execute_expert(
            policy_env,
            solver_env,
            seed=seed,
            raw_obs=raw_obs,
            lower=lower,
            upper=upper,
            state_callback=task_state,
        )
        records.extend(expert_records)
        actions.extend(expert_actions)
        sources.extend(["expert"] * len(expert_actions))
        expert_ever_grasped = bool(oracle_result.pop("ever_grasped", False))
        expert_states = oracle_result.pop(
            "state_timeline", np.empty((0, 0), dtype=np.float32)
        )
        if np.asarray(expert_states).size:
            task_states.extend(np.asarray(expert_states, dtype=np.float32))
        strict_success = bool(oracle_result["accepted"])
    else:
        expert_ever_grasped = False

    return records, actions, expert_start, {
        "seed": seed,
        "split": split,
        "method": method,
        "strict_success": strict_success,
        "ever_grasped": bool(
            expert_ever_grasped or np.max(np.asarray(task_states)[:, -2]) > 0.5
        ),
        "steps": len(actions),
        "expert_start_step": expert_start,
        "expert_action_steps": 0 if expert_start is None else len(actions) - expert_start,
        "timeline": timeline,
        "fixed_timing_step": timing_step,
        "task_state_dim": int(task_states[0].shape[0]),
        "task_states": np.stack(task_states),
        "oracle": oracle_result,
        "sources": sources,
        **reset_metadata(policy_env, split=split),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=("vlm_pool_pca", "offline_oracle", "failure_recovery", "diffdagger", FIXED_TIMING_METHOD),
        required=True,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--pca-asset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-id", type=Path, required=True)
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--seed-manifest", type=Path)
    parser.add_argument("--consume-all-seeds", action="store_true")
    parser.add_argument("--only-split", choices=("id", "ood"))
    parser.add_argument("--id-seed", type=int, default=70000)
    parser.add_argument("--ood-seed", type=int, default=80000)
    parser.add_argument("--offline-per-split", type=int, default=50)
    parser.add_argument("--max-attempts", type=int, default=5000)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--control-freq", type=int, default=10)
    parser.add_argument("--sim-backend", choices=("physx_cpu", "gpu"), default="physx_cpu")
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--diff-timesteps", type=int, default=16)
    parser.add_argument("--diff-patience", type=int, default=2)
    parser.add_argument("--timing-step", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.method == FIXED_TIMING_METHOD and args.timing_step is None:
        raise ValueError("fixed_timing requires --timing-step")
    if args.output_dir.exists() or args.repo_id.exists():
        raise FileExistsError("output and dataset paths must be new")
    args.output_dir.mkdir(parents=True)
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    asset = torch.load(args.pca_asset, map_location="cpu", weights_only=False)
    pca_statistics = PCAResidualStatistics.from_state_dict(asset["statistics"])
    policy = None if args.method == "offline_oracle" else XVLAAirplanePolicy(args.checkpoint, args.xvla_root)
    _write_json(args.output_dir / "collection_provenance.json", {
        "format": "xvla_airplane_ood_dagger_collection_v1",
        "method": args.method,
        "checkpoint": str(args.checkpoint.resolve()),
        "pca_asset": str(args.pca_asset.resolve()),
        "calibration": str(args.calibration.resolve()),
        "split_schedule": "offline_exact_50_50" if args.method == "offline_oracle" else "strict_raw_id_ood_alternation",
        "only_split": args.only_split,
        "seed_manifest": str(args.seed_manifest.resolve()) if args.seed_manifest else None,
        "consume_all_seeds": args.consume_all_seeds,
        "policy_episode_horizon": POLICY_EPISODE_HORIZON,
        "fixed_timing_step": args.timing_step,
        "failure_recovery_step": FAILURE_RECOVERY_STEP,
        "oracle_close": {
            "strategy": "stop_after_stable_grasp",
            "max_steps": ORACLE_CLOSE_MAX_STEPS,
            "stable_steps": ORACLE_STABLE_GRASP_STEPS,
        },
        "admission": "ever_grasped_and_nonempty_full_expert_suffix"
        if args.method == FIXED_TIMING_METHOD
        else "strict_oracle_success_and_nonempty_full_expert_suffix",
        "tail_handling": "all_real_actions_saved_temporal_mask_at_training",
    })

    register_controlled_pick_single_ycb_airplane_variants()
    policy_envs = {
        split: _build_env(_env_args(args, split), control_mode="pd_joint_delta_pos")
        for split in ("id", "ood")
    }
    solver_envs = {
        split: _build_env(_env_args(args, split), control_mode="pd_joint_pos")
        for split in ("id", "ood")
    }
    bounds = {split: _joint_delta_arm_bounds(policy_envs[split]) for split in ("id", "ood")}
    dataset = None
    accepted = 0
    accepted_by_split = {"id": 0, "ood": 0}
    raw_by_split = {"id": 0, "ood": 0}
    next_seed = {"id": args.id_seed, "ood": args.ood_seed}
    frozen_seeds = None
    if args.seed_manifest is not None:
        payload = json.loads(args.seed_manifest.read_text(encoding="utf-8"))
        frozen_seeds = [int(seed) for seed in payload["seeds"]]
        if not frozen_seeds:
            raise ValueError("seed manifest contains no seeds")
    try:
        for attempt_index in range(args.max_attempts):
            if not args.consume_all_seeds and accepted >= args.target:
                break
            if frozen_seeds is not None and attempt_index >= len(frozen_seeds):
                break
            split = args.only_split or alternating_split(attempt_index)
            if args.method == "offline_oracle" and accepted_by_split[split] >= args.offline_per_split:
                split = "ood" if split == "id" else "id"
            if args.method == "offline_oracle" and accepted_by_split[split] >= args.offline_per_split:
                break
            if frozen_seeds is not None:
                seed = frozen_seeds[attempt_index]
            else:
                seed = next_seed[split]
                next_seed[split] += 1
            raw_by_split[split] += 1
            lower, upper = bounds[split]
            records, actions, expert_start, row = _run_attempt(
                method=args.method, split=split, seed=seed, policy=policy,
                policy_env=policy_envs[split], solver_env=solver_envs[split],
                lower=lower, upper=upper,
                pca_statistics=pca_statistics,
                pca_threshold=float(calibration["pca_threshold"]),
                diff_threshold=float(calibration["diff_threshold"]),
                diff_patience=args.diff_patience, diff_timesteps=args.diff_timesteps,
                flow_steps=args.flow_steps, timing_step=args.timing_step,
            )
            sources = row.pop("sources")
            task_states = row.pop("task_states")
            row["attempt_index"] = attempt_index
            state_path = (
                args.output_dir
                / "raw_archive"
                / "task_states"
                / f"episode_{attempt_index:06d}_seed_{seed:06d}.npy"
            )
            state_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(state_path, np.asarray(task_states, dtype=np.float32))
            row["task_states"] = str(state_path)
            row["video"] = _save_raw_attempt(
                output_dir=args.output_dir, episode_index=attempt_index, seed=seed,
                records=records, actions=actions, sources=sources, control_freq=args.control_freq,
            )
            admission_success = (
                row["ever_grasped"]
                if args.method == FIXED_TIMING_METHOD
                else row["strict_success"]
            )
            admitted = admitted_expert_suffix(
                success=admission_success,
                expert_start=expert_start,
                action_count=len(actions),
            )
            row["accepted"] = admitted is not None
            _append_jsonl(args.output_dir / "episodes.jsonl", row)
            if admitted is not None:
                begin, end = admitted
                frames = _build_frames(
                    records=records[begin : end + 1], actions=actions[begin:end],
                    task=PICK_SINGLE_YCB_AIRPLANE_TASK,
                    main_camera=_select_camera(records[0].obs, "", ("base_camera",) + MAIN_CAMERA_CANDIDATES, "main"),
                    wrist_camera=_select_camera(records[0].obs, "", ("hand_camera",) + WRIST_CAMERA_CANDIDATES, "wrist"),
                )
                if dataset is None:
                    dataset = _create_dataset(
                        repo_id=str(args.repo_id), image_shape=tuple(frames[0]["image"].shape),
                        wrist_image_shape=tuple(frames[0]["wrist_image"].shape), fps=args.control_freq,
                        image_writer_threads=4, image_writer_processes=0,
                    )
                for frame in frames:
                    dataset.add_frame(frame)
                dataset.save_episode()
                _append_jsonl(args.output_dir / "training_episodes.jsonl", {
                    "dataset_episode_index": accepted, "raw_attempt_index": attempt_index,
                    "seed": seed, "split": split, "expert_start_step": begin,
                    "expert_action_steps": end - begin,
                })
                accepted += 1
                accepted_by_split[split] += 1
            print(
                f"[xvla-collector] method={args.method} attempt={attempt_index + 1} "
                f"accepted={accepted}/{args.target} accepted_by_split={accepted_by_split}",
                flush=True,
            )
    finally:
        if dataset is not None and getattr(dataset, "image_writer", None) is not None:
            dataset.image_writer.wait_until_done()
        for env in (*policy_envs.values(), *solver_envs.values()):
            env.close()
    if not args.consume_all_seeds and accepted != args.target:
        raise RuntimeError(f"collected {accepted}/{args.target} accepted trajectories")
    _write_json(args.output_dir / "summary.json", {
        "method": args.method, "accepted_total": accepted,
        "accepted_by_split": accepted_by_split, "raw_by_split": raw_by_split,
        "raw_total": sum(raw_by_split.values()),
        "admission_endpoint": "ever_grasped" if args.method == FIXED_TIMING_METHOD else "strict_success",
        "dataset": str(args.repo_id), "raw_archive": str(args.output_dir / "raw_archive"),
    })


if __name__ == "__main__":
    main()
