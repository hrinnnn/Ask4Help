#!/usr/bin/env python3
"""Run a fixed-denominator object-variation Oracle gate with videos."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio

ROOT = Path(__file__).resolve().parents[1]
RLINF_ROOT = ROOT / "RLinf"
if str(RLINF_ROOT) not in sys.path:
    sys.path.insert(0, str(RLINF_ROOT))

from rlinf.envs.maniskill.pick_single_ycb_object_variation import (
    PICK_SINGLE_YCB_OBJECT_ID_ENV_ID,
    PICK_SINGLE_YCB_OBJECT_OOD_ENV_ID,
    register_controlled_pick_single_ycb_object_variants,
)
from toolkits.lerobot.diagnose_pick_single_ycb_object_variation_oracle import (
    run_oracle,
)


def frame_to_numpy(frame):
    if hasattr(frame, "detach"):
        frame = frame.detach().cpu().numpy()
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("id", "ood"), required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--min-success", type=int, default=19)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.count < 1 or not 0 <= args.min_success <= args.count:
        raise ValueError("invalid count/min-success")

    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    register_controlled_pick_single_ycb_object_variants()
    env_id = PICK_SINGLE_YCB_OBJECT_ID_ENV_ID if args.split == "id" else PICK_SINGLE_YCB_OBJECT_OOD_ENV_ID
    env = gym.make(
        env_id,
        num_envs=1,
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        reward_mode="sparse",
        sim_backend="physx_cpu",
        sim_config={"sim_freq": 100, "control_freq": 10},
        render_mode="rgb_array",
        max_episode_steps=200,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    video_dir = args.output_dir / "videos"
    video_dir.mkdir()
    rows: list[dict] = []
    try:
        for index in range(args.count):
            seed = args.seed_start + index
            frames: list = []
            original_reset, original_step = env.reset, env.step

            def reset_hook(*reset_args, **reset_kwargs):
                frames.clear()
                result = original_reset(*reset_args, **reset_kwargs)
                frames.append(frame_to_numpy(env.render()))
                return result

            def step_hook(action, *step_args, **step_kwargs):
                result = original_step(action, *step_args, **step_kwargs)
                frames.append(frame_to_numpy(env.render()))
                return result

            env.reset = reset_hook  # type: ignore[method-assign]
            env.step = step_hook  # type: ignore[method-assign]
            try:
                row = run_oracle(env, seed=seed)
            finally:
                env.reset, env.step = original_reset, original_step  # type: ignore[method-assign]
            video_path = video_dir / f"episode_{index:06d}_seed_{seed:06d}.mp4"
            imageio.mimsave(video_path, frames, fps=10, codec="libx264")
            row = {"episode_index": index, **row, "video": str(video_path)}
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        env.close()

    success = sum(bool(row["success"]) for row in rows)
    summary = {
        "format": "pick_single_ycb_object_variation_oracle_gate_v1",
        "split": args.split,
        "seed_start": args.seed_start,
        "episodes": len(rows),
        "video_count": len(list(video_dir.glob("*.mp4"))),
        "strict_success": success,
        "min_success": args.min_success,
        "passed": success >= args.min_success and len(rows) == args.count,
        "rows": "episodes.jsonl",
    }
    (args.output_dir / "episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    marker = "ORACLE_GATE_PASSED" if summary["passed"] else "ORACLE_GATE_FAILED"
    (args.output_dir / marker).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if not summary["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
