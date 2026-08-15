#!/usr/bin/env python3
"""Collect fixed-timing StackPyramid expert suffixes on one OOD split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import imageio
import numpy as np
import torch


def _bool(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return bool(np.asarray(value).reshape(-1)[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("stage1_ood", "stage2_ood", "stage3_ood"), required=True)
    parser.add_argument("--condition", choices=("immediate", "pre_stage", "capability_boundary", "failure_recovery"), required=True)
    parser.add_argument("--target", type=int, default=20)
    parser.add_argument("--start-seed", type=int, default=63000)
    parser.add_argument("--max-attempts", type=int, default=40)
    parser.add_argument("--pre-offset", type=int, default=25)
    parser.add_argument("--boundary-offset", type=int, default=5)
    parser.add_argument("--recovery-delay", type=int, default=25)
    parser.add_argument("--flow-steps", type=int, default=5)
    parser.add_argument("--sim-backend", choices=("gpu", "cpu"), default="cpu")
    parser.add_argument("--render-backend", choices=("gpu", "cpu"), default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root), str(args.xvla_root)]
    from tools.collect_stackpyramid_xvla_dagger import (
        ACTION_HORIZON,
        EXECUTE_HORIZON,
        MAX_EPISODE_STEPS,
        REAL_ACTION_DIM,
        StackPyramidOracle,
        StepRecorder,
        _copy_record,
        _predict,
        _summary,
    )
    from tools.stackpyramid_task import register_stackpyramid_splits, stackpyramid_env_id

    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    from models.modeling_xvla import XVLA
    from models.processing_xvla import XVLAProcessor

    audit = json.loads((args.audit / "boundary_audit.json").read_text(encoding="utf-8"))
    key = {"stage1_ood": "stage1", "stage2_ood": "stage2", "stage3_ood": "stage3"}[args.split]
    boundary = audit["summaries"][args.split]["median_boundary_steps"][key]
    if boundary is None:
        raise RuntimeError(f"missing audited boundary for {args.split}:{key}")
    start_step = {
        "immediate": 0,
        "pre_stage": max(0, int(boundary) - args.pre_offset),
        "capability_boundary": max(0, int(boundary) - args.boundary_offset),
        "failure_recovery": min(MAX_EPISODE_STEPS - 1, int(boundary) + args.recovery_delay),
    }[args.condition]

    register_stackpyramid_splits()
    device = torch.device("cuda")
    model = XVLA.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16).to(device).eval()
    processor = XVLAProcessor.from_pretrained(args.checkpoint)
    env = StepRecorder(
        gym.make(
            stackpyramid_env_id(args.split),
            obs_mode="rgb+state",
            control_mode="pd_joint_pos",
            render_mode="rgb_array",
            sim_backend=args.sim_backend,
            render_backend=args.render_backend,
        )
    )
    accepted: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    try:
        for attempt in range(args.max_attempts):
            if len(accepted) >= args.target:
                break
            seed = args.start_seed + attempt
            raw_obs, _ = env.reset(seed=seed)
            expert_start: int | None = None
            terminated = truncated = False
            ever_grasped = False
            ever_base = False
            success = False
            while len(env.actions) < min(start_step, MAX_EPISODE_STEPS) and not (terminated or truncated):
                generated, _, _, _ = _predict(
                    model, processor, raw_obs, device, seed + len(env.actions), args.flow_steps
                )
                env.current_source = "policy"
                for action in np.clip(generated[:ACTION_HORIZON], low, high)[:EXECUTE_HORIZON]:
                    raw_obs, _, terminated, truncated, _ = env.step(action)
                    current = _summary(env)
                    ever_grasped |= any(current["grasped"])
                    ever_base |= current["xy_ab"]
                    success |= current["success"]
                    if terminated or truncated or len(env.actions) >= start_step:
                        break
            if not terminated and not truncated and len(env.actions) >= start_step:
                expert_start = len(env.actions)
                env.current_source = "expert"
                try:
                    StackPyramidOracle(env).run()
                except Exception as exc:
                    oracle_error = repr(exc)
                else:
                    oracle_error = None
            else:
                oracle_error = "policy_terminated_before_intervention"
            final = _summary(env)
            ever_grasped |= any(final["grasped"])
            ever_base |= final["xy_ab"]
            success |= final["success"]
            row = {
                "attempt": attempt,
                "seed": seed,
                "split": args.split,
                "condition": args.condition,
                "audited_boundary_step": int(boundary),
                "scheduled_intervention_step": int(start_step),
                "expert_start_step": expert_start,
                "expert_action_steps": 0 if expert_start is None else max(0, len(env.actions) - expert_start),
                "steps": len(env.actions),
                "success": bool(success),
                "ever_grasped": bool(ever_grasped),
                "ever_base_completed": bool(ever_base),
                "oracle_error": oracle_error,
            }
            video_path = args.output / "raw_videos" / f"attempt_{attempt:06d}_seed_{seed:06d}.mp4"
            video_path.parent.mkdir(parents=True, exist_ok=True)
            with imageio.get_writer(video_path, fps=10, codec="libx264", macro_block_size=None) as writer:
                for frame in env.frames:
                    writer.append_data(frame)
            row["video"] = str(video_path)
            rows.append(row)
            with (args.output / "episodes.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            if success and expert_start is not None and len(env.actions) > expert_start:
                accepted.append(
                    {
                        "seed": seed,
                        "split": args.split,
                        "condition": args.condition,
                        "records": [_copy_record(record) for record in env.records[expert_start:]],
                        "actions": [action.copy() for action in env.actions[expert_start:]],
                    }
                )
            print(json.dumps({"attempt": attempt, "accepted": len(accepted), "condition": args.condition, "success": bool(success)}), flush=True)
    finally:
        env.env.close()

    if len(accepted) < args.target:
        raise RuntimeError(f"timing collection incomplete: {len(accepted)}/{args.target}")
    import h5py

    with h5py.File(args.output / "accepted_suffixes.h5", "w") as handle:
        for index, item in enumerate(accepted):
            group = handle.create_group(f"traj_{index:06d}")
            records = item["records"]
            obs = group.create_group("obs")
            sensor = obs.create_group("sensor_data")
            sensor.create_group("base_camera").create_dataset("rgb", data=np.stack([record["base"] for record in records]))
            sensor.create_group("hand_camera").create_dataset("rgb", data=np.stack([record["wrist"] for record in records]))
            obs.create_dataset("state", data=np.stack([record["state"] for record in records]))
            group.create_dataset("actions", data=np.asarray(item["actions"], dtype=np.float32))
    (args.output / "training_episodes.jsonl").write_text(
        "".join(json.dumps({"index": i, "seed": item["seed"], "split": item["split"], "condition": item["condition"], "expert_action_steps": len(item["actions"])}) + "\n" for i, item in enumerate(accepted)),
        encoding="utf-8",
    )
    summary = {
        "format": "stackpyramid_timing_collection_v1",
        "split": args.split,
        "condition": args.condition,
        "target_accepted": args.target,
        "accepted_total": len(accepted),
        "raw_attempts": len(rows),
        "raw_successes": sum(int(row["success"]) for row in rows),
        "expert_action_steps": sum(len(item["actions"]) for item in accepted),
        "audited_boundary_step": int(boundary),
        "scheduled_intervention_step": int(start_step),
        "dataset": str((args.output / "accepted_suffixes.h5").resolve()),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output / "COLLECTION_COMPLETE").write_text("complete\n", encoding="utf-8")


if __name__ == "__main__":
    main()
