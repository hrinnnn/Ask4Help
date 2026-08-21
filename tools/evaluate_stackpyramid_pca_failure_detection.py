#!/usr/bin/env python3
"""Passive trajectory-level failure detection with an ID-only PCA asset."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import imageio
import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("id", "stage2_ood", "stage3_ood"), required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--start-seed", type=int, required=True)
    parser.add_argument("--max-episode-steps", type=int, default=600)
    parser.add_argument("--execute-horizon", type=int, default=5)
    parser.add_argument("--flow-steps", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sim-backend", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--render-backend", choices=("gpu", "cpu"), default="gpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    (args.output / "config.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")

    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root), str(args.xvla_root)]
    os.environ["STACKPYRAMID_OOD_GEOMETRY"] = "v4"
    from tools.stackpyramid_pca_compat import PCAResidualStatistics, pca_residual_score
    from tools.collect_stackpyramid_xvla_dagger import _predict
    from tools.evaluate_stackpyramid_xvla import (
        bool_scalar,
        details,
        frame_array,
        json_state,
        make_policy,
        stage_events,
    )
    from tools.stackpyramid_task import (
        register_stackpyramid_splits,
        reset_metadata,
        stackpyramid_env_id,
        stackpyramid_geometry_version,
    )

    register_stackpyramid_splits()
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    device = torch.device(args.device)
    model, processor = make_policy(args.checkpoint, args.xvla_root, device)
    payload = torch.load(args.asset, map_location="cpu", weights_only=False)
    stats = PCAResidualStatistics.from_state_dict(payload["statistics"])

    def make_env() -> Any:
        return gym.make(
            stackpyramid_env_id(args.split),
            obs_mode="rgb+state",
            control_mode="pd_joint_pos",
            render_mode="rgb_array",
            sim_backend=args.sim_backend,
            render_backend=args.render_backend,
            max_episode_steps=args.max_episode_steps,
        )

    env = make_env()
    rows: list[dict[str, Any]] = []
    try:
        for episode_index in range(args.episodes):
            if episode_index > 0:
                env.close()
                env = make_env()
            seed = args.start_seed + episode_index
            raw_obs, _ = env.reset(seed=seed)
            metadata = reset_metadata(env, split=args.split)
            invariants = metadata.get("reset_invariants", {})
            if metadata.get("ood_geometry") != "v4" or any(invariants.values()):
                raise RuntimeError(f"invalid v4 reset for {args.split}: {metadata}")
            initial_red = env.unwrapped.cubeA.pose.p.detach().cpu().numpy().reshape(-1, 3)[0]
            initial_blue = env.unwrapped.cubeC.pose.p.detach().cpu().numpy().reshape(-1, 3)[0]
            initial_z = {"red": float(initial_red[2]), "blue": float(initial_blue[2])}
            frames: list[np.ndarray] = []
            first = frame_array(env.render())
            if first is not None:
                frames.append(first)
            executed = 0
            ever_grasped = False
            ever_base = False
            ever_success = False
            events = {name: False for name in ("red_grasped", "red_lifted", "red_placed", "blue_grasped", "blue_lifted")}
            event_steps: dict[str, int | None] = {name: None for name in events}
            scores: list[float] = []
            score_steps: list[int] = []
            formal_actions: list[np.ndarray] = []
            formal_states: list[dict[str, Any]] = []
            if args.formal_evidence if hasattr(args, "formal_evidence") else True:
                formal_states.append({"step": 0, "state": json_state(raw_obs["state"]), "details": details(env), "stage_events": dict(events)})
            while executed < args.max_episode_steps and not ever_success:
                generated, bridge, inputs, _encoding = _predict(
                    model, processor, raw_obs, device, seed + executed, args.flow_steps
                )
                score = float(pca_residual_score(bridge.unsqueeze(1), stats)[0].item())
                scores.append(score)
                score_steps.append(executed)
                chunk = np.clip(
                    generated[:10],
                    np.asarray(env.action_space.low, dtype=np.float32),
                    np.asarray(env.action_space.high, dtype=np.float32),
                )
                for action in chunk[: args.execute_horizon]:
                    raw_obs, _, terminated, truncated, _info = env.step(action.astype(np.float32))
                    executed += 1
                    formal_actions.append(np.asarray(action, dtype=np.float32).copy())
                    current = details(env)
                    reached = stage_events(env, initial_z, events)
                    for name, value in reached.items():
                        if value and not events[name]:
                            events[name] = True
                            event_steps[name] = executed
                    formal_states.append({"step": executed, "state": json_state(raw_obs["state"]), "details": current, "stage_events": dict(events)})
                    ever_grasped |= any(current["grasped"])
                    ever_base |= bool(current["xy_ab"] and (current["z_cb"] or current["z_ca"]))
                    ever_success |= bool(current["success"])
                    frame = frame_array(env.render())
                    if frame is not None:
                        frames.append(frame)
                    if bool_scalar(terminated) or bool_scalar(truncated) or executed >= args.max_episode_steps:
                        break
                if bool_scalar(terminated) or bool_scalar(truncated):
                    break
            final = details(env)
            max_score = max(scores) if scores else float("nan")
            max_index = int(np.argmax(scores)) if scores else -1
            video_path = args.output / "videos" / f"{args.split}_{seed}.mp4"
            video_path.parent.mkdir(parents=True, exist_ok=True)
            with imageio.get_writer(video_path, fps=10, codec="libx264", macro_block_size=None) as writer:
                for frame in frames:
                    writer.append_data(frame)
            actions_path = args.output / "actions" / f"{args.split}_{seed}.npy"
            states_path = args.output / "states" / f"{args.split}_{seed}.json"
            actions_path.parent.mkdir(parents=True, exist_ok=True)
            states_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(actions_path, np.asarray(formal_actions, dtype=np.float32))
            states_path.write_text(json.dumps(formal_states) + "\n")
            rows.append({
                "episode_index": episode_index,
                "seed": seed,
                "split": args.split,
                "steps": executed,
                "timeout": executed >= args.max_episode_steps and not ever_success,
                "ever_grasped": bool(ever_grasped),
                "ever_base_completed": bool(ever_base),
                "strict_success": bool(ever_success),
                "failure_label": not bool(ever_success),
                "stage_events": events,
                "stage_event_steps": event_steps,
                "max_pca_score": max_score,
                "max_pca_score_step": score_steps[max_index] if max_index >= 0 else None,
                "first_alarm_step": next((step for step, value in zip(score_steps, scores) if value > args.threshold), None),
                "score_timeline": [{"step": step, "score": value} for step, value in zip(score_steps, scores)],
                "reset_metadata": metadata,
                "video": str(video_path),
                "actions": str(actions_path),
                "state_timeline": str(states_path),
            })
            print(json.dumps(rows[-1], ensure_ascii=True), flush=True)
    finally:
        env.close()

    summary = {
        "format": "stackpyramid_pca_passive_failure_detection_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "asset": str(args.asset.resolve()),
        "split": args.split,
        "episodes": len(rows),
        "threshold": args.threshold,
        "max_episode_steps": args.max_episode_steps,
        "geometry": stackpyramid_geometry_version(),
        "env_id": stackpyramid_env_id(args.split),
        "strict_success": sum(int(row["strict_success"]) for row in rows),
        "ever_grasped": sum(int(row["ever_grasped"]) for row in rows),
        "ever_base_completed": sum(int(row["ever_base_completed"]) for row in rows),
        "failure_episodes": sum(int(row["failure_label"]) for row in rows),
        "video_count": len(list((args.output / "videos").glob("*.mp4"))),
        "action_array_count": len(list((args.output / "actions").glob("*.npy"))),
        "state_timeline_count": len(list((args.output / "states").glob("*.json"))),
        "rows": rows,
        "passive_only": True,
        "expert_involved": False,
        "training_involved": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "EVAL_COMPLETE").write_text("complete; passive PCA failure detection only\n")


if __name__ == "__main__":
    main()
