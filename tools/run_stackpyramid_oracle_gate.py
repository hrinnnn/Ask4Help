#!/usr/bin/env python3
"""Audit StackPyramid Oracle success on one controlled split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import imageio


REQUIRED_EVENT_ORDER = ("red_grasped", "red_lifted", "red_placed", "blue_grasped", "blue_lifted")


def _event_order(recorder: object) -> dict[str, object]:
    first_steps = dict(getattr(recorder, "event_first_steps", {}))
    observed = [name for name in REQUIRED_EVENT_ORDER if name in first_steps]
    order_pass = observed == list(REQUIRED_EVENT_ORDER) and all(
        first_steps[previous] < first_steps[current]
        for previous, current in zip(REQUIRED_EVENT_ORDER, REQUIRED_EVENT_ORDER[1:])
    )
    return {
        "required_events": list(REQUIRED_EVENT_ORDER),
        "first_event_steps": first_steps,
        "observed_order": observed,
        "event_order_pass": order_pass,
    }


def _bool(value: object) -> bool:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return bool(np.asarray(value).reshape(-1)[0])


def _seed_values(spec: object) -> list[int]:
    if isinstance(spec, dict):
        start = int(spec["start"])
        count = int(spec["count"])
        return list(range(start, start + count))
    return [int(seed) for seed in spec]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("id", "stage1_ood", "stage2_ood", "stage3_ood"), required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--start-seed", type=int, required=True)
    parser.add_argument("--seed-manifest", type=Path)
    parser.add_argument("--sim-backend", choices=("gpu", "cpu"), default="cpu")
    parser.add_argument("--render-backend", choices=("gpu", "cpu"), default="cpu")
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--fresh-env-per-episode", action="store_true")
    parser.add_argument("--formal-evidence", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    sys.path[:0] = [str(args.repo_root), str(args.xvla_root)]

    manifest = None
    if args.seed_manifest is not None:
        manifest = json.loads(args.seed_manifest.read_text(encoding="utf-8"))
        seeds = _seed_values(manifest["oracle"][args.split])
        if len(seeds) != args.episodes:
            raise ValueError(
                f"seed manifest has {len(seeds)} seeds for {args.split}, "
                f"expected {args.episodes}"
            )
    else:
        seeds = [args.start_seed + index for index in range(args.episodes)]

    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    from tools.collect_stackpyramid_xvla_dagger import (
        StackPyramidOracle,
        StepRecorder,
        _install_rrt_fallback,
    )
    from tools.stackpyramid_task import register_stackpyramid_splits, reset_metadata, stackpyramid_env_id

    _install_rrt_fallback()
    register_stackpyramid_splits()
    def make_recorder():
        return StepRecorder(gym.make(
            stackpyramid_env_id(args.split),
            obs_mode="rgb+state",
            control_mode="pd_joint_pos",
            render_mode="rgb_array",
            sim_backend=args.sim_backend,
            render_backend=args.render_backend,
            max_episode_steps=args.max_episode_steps,
        ))

    env = make_recorder()
    rows: list[dict[str, object]] = []
    try:
        for index in range(args.episodes):
            if args.fresh_env_per_episode and index > 0:
                env.env.close()
                env = make_recorder()
            seed = int(seeds[index])
            env.reset(seed=seed)
            metadata = reset_metadata(env, split=args.split)
            error = None
            try:
                StackPyramidOracle(env).run()
            except Exception as exc:  # retain failed audit attempts
                error = repr(exc)
            evaluation = env.unwrapped.evaluate()
            success = _bool(evaluation["success"] if isinstance(evaluation, dict) else evaluation)
            rows.append(
                {
                    "episode_index": index,
                    "seed": seed,
                    "split": args.split,
                    "actions": len(env.actions),
                    "strict_success": success,
                    "oracle_error": error,
                    "reset_invariants": dict(env.reset_invariants),
                    "reset_invariant_pass": not any(env.reset_invariants.values()),
                    "event_order": _event_order(env),
                    "reset_metadata": metadata,
                }
            )
            if args.formal_evidence:
                video_path = args.output / "videos" / f"{args.split}_{seed}.mp4"
                actions_path = args.output / "actions" / f"{args.split}_{seed}.npy"
                states_path = args.output / "states" / f"{args.split}_{seed}.json"
                video_path.parent.mkdir(parents=True, exist_ok=True)
                actions_path.parent.mkdir(parents=True, exist_ok=True)
                states_path.parent.mkdir(parents=True, exist_ok=True)
                with imageio.get_writer(video_path, fps=30, codec="libx264", macro_block_size=None) as writer:
                    for frame in env.frames:
                        writer.append_data(frame)
                np.save(actions_path, np.asarray(env.actions, dtype=np.float32))
                timeline = [
                    {"step": step, "state": record["state"].tolist(), "events": env.event_history[min(step, len(env.event_history) - 1)]}
                    for step, record in enumerate(env.records)
                ]
                states_path.write_text(json.dumps(timeline) + "\n", encoding="utf-8")
                rows[-1].update({"video": str(video_path), "actions_path": str(actions_path), "state_timeline": str(states_path)})
            print(json.dumps(rows[-1]), flush=True)
    finally:
        env.env.close()

    summary = {
        "format": "stackpyramid_oracle_gate_v2",
        "split": args.split,
        "episodes": len(rows),
        "strict_successes": sum(int(row["strict_success"]) for row in rows),
        "success_rate": sum(int(row["strict_success"]) for row in rows) / len(rows),
        "seed_manifest": str(args.seed_manifest.resolve()) if args.seed_manifest else None,
        "sim_backend": args.sim_backend,
        "render_backend": args.render_backend,
        "reset_invariant_failures": sum(not row["reset_invariant_pass"] for row in rows),
        "event_order_failures": sum(not row["event_order"]["event_order_pass"] for row in rows),
        "max_episode_steps": args.max_episode_steps,
        "fresh_env_per_episode": args.fresh_env_per_episode,
        "formal_evidence": args.formal_evidence,
        "video_count": len(list((args.output / "videos").glob("*.mp4"))) if args.formal_evidence else None,
        "action_array_count": len(list((args.output / "actions").glob("*.npy"))) if args.formal_evidence else None,
        "state_timeline_count": len(list((args.output / "states").glob("*.json"))) if args.formal_evidence else None,
        "rows": rows,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    if (
        summary["episodes"] != args.episodes
        or summary["success_rate"] < 0.90
        or summary["reset_invariant_failures"]
        or summary["event_order_failures"]
        or (args.formal_evidence and (summary["video_count"] != args.episodes or summary["action_array_count"] != args.episodes or summary["state_timeline_count"] != args.episodes))
    ):
        raise RuntimeError(f"Oracle gate failed: {summary}")
    (args.output / "ORACLE_GATE_COMPLETE").write_text("complete\n", encoding="utf-8")


if __name__ == "__main__":
    main()
