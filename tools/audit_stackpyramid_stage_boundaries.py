#!/usr/bin/env python3
"""Audit stage boundary steps for the controlled StackPyramid task."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _bool(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return bool(np.asarray(value).reshape(-1)[0])


def _seed_values(spec: object) -> list[int]:
    if isinstance(spec, dict):
        start = int(spec["start"])
        count = int(spec["count"])
        return list(range(start, start + count))
    return [int(seed) for seed in spec]


class BoundaryRecorder:
    def __init__(self, env: Any, split: str):
        from tools.collect_stackpyramid_xvla_dagger import StepRecorder

        self.recorder = StepRecorder(env)
        self.split = split
        self.initial_z: dict[str, float] = {}
        self.boundaries = {"stage1": None, "stage2": None, "stage3": None}

    @property
    def unwrapped(self) -> Any:
        return self.recorder.unwrapped

    @property
    def actions(self) -> list[np.ndarray]:
        return self.recorder.actions

    def reset(self, **kwargs: Any):
        raw_obs, info = self.recorder.reset(**kwargs)
        base = self.unwrapped
        self.initial_z = {
            "red": float(base.cubeA.pose.p.detach().cpu().numpy().reshape(-1, 3)[0, 2]),
            "blue": float(base.cubeC.pose.p.detach().cpu().numpy().reshape(-1, 3)[0, 2]),
        }
        self.boundaries = {"stage1": None, "stage2": None, "stage3": None}
        return raw_obs, info

    def _update_boundaries(self) -> None:
        base = self.unwrapped
        red = base.cubeA.pose.p.detach().cpu().numpy().reshape(-1, 3)[0]
        blue = base.cubeC.pose.p.detach().cpu().numpy().reshape(-1, 3)[0]
        green = base.cubeB.pose.p.detach().cpu().numpy().reshape(-1, 3)[0]
        red_grasped = _bool(base.agent.is_grasping(base.cubeA))
        blue_grasped = _bool(base.agent.is_grasping(base.cubeC))
        red_lifted = red[2] > self.initial_z["red"] + 0.015
        blue_lifted = blue[2] > self.initial_z["blue"] + 0.015
        xy_threshold = float(
            np.linalg.norm(2 * base.cube_half_size[:2].detach().cpu().numpy()) + 0.005
        )
        red_placed = (
            float(np.linalg.norm((red - green)[:2])) <= xy_threshold
            and not red_grasped
            and red[2] <= self.initial_z["red"] + 0.03
        )
        step = len(self.actions)
        # The stage boundary is a physical completion event.  Some valid
        # Panda grasps do not satisfy the simulator's instantaneous contact
        # predicate on the same step, so requiring both contact and height
        # would miss valid OOD stage-1 executions.
        if self.boundaries["stage1"] is None and red_lifted:
            self.boundaries["stage1"] = step
        if self.boundaries["stage2"] is None and red_placed:
            self.boundaries["stage2"] = step
        if self.boundaries["stage3"] is None and blue_lifted:
            self.boundaries["stage3"] = step

    def step(self, action: Any):
        result = self.recorder.step(action)
        self._update_boundaries()
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self.recorder, name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, default=Path("/data/zhaozhixuan/xvla_stackcube_data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--start-seed", type=int, default=62000)
    parser.add_argument("--seed-manifest", type=Path)
    parser.add_argument("--sim-backend", choices=("gpu", "cpu"), default="cpu")
    parser.add_argument("--render-backend", choices=("gpu", "cpu"), default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    root = Path(__file__).resolve().parents[1]
    task_root = args.task_root if args.task_root.is_dir() else root
    sys.path[:0] = [str(task_root), str(root), str(args.xvla_root)]
    if args.seed_manifest is not None:
        manifest = json.loads(args.seed_manifest.read_text(encoding="utf-8"))
        seeds = manifest["oracle"]
    else:
        seeds = None
    from tools.collect_stackpyramid_xvla_dagger import StackPyramidOracle
    from tools.stackpyramid_task import register_stackpyramid_splits, stackpyramid_env_id

    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    register_stackpyramid_splits()
    rows: list[dict[str, Any]] = []
    for split in ("stage1_ood", "stage2_ood", "stage3_ood"):
        env = BoundaryRecorder(
            gym.make(
                stackpyramid_env_id(split),
                obs_mode="rgb+state",
                control_mode="pd_joint_pos",
                render_mode="rgb_array",
                sim_backend=args.sim_backend,
                render_backend=args.render_backend,
            ),
            split,
        )
        try:
            for episode in range(args.episodes):
                if seeds is None:
                    seed = args.start_seed + (episode * 10) + (0 if split == "stage1_ood" else 1 if split == "stage2_ood" else 2)
                else:
                    split_seeds = _seed_values(seeds["id" if split == "id" else split])
                    if episode >= len(split_seeds):
                        raise ValueError(f"seed manifest is too short for {split}: {len(split_seeds)}")
                    seed = int(split_seeds[episode])
                env.reset(seed=seed)
                oracle_error = None
                try:
                    StackPyramidOracle(env).run()
                except Exception as exc:  # keep audit evidence for failed oracle runs
                    oracle_error = repr(exc)
                final = env.unwrapped.evaluate()
                rows.append(
                    {
                        "split": split,
                        "seed": seed,
                        "episodes_index": episode,
                        "boundaries": dict(env.boundaries),
                        "actions": len(env.actions),
                        "success": _bool(final["success"] if isinstance(final, dict) else final),
                        "oracle_error": oracle_error,
                    }
                )
        finally:
            env.recorder.env.close()

    # The stock StackPyramid execution has two physical capability boundaries:
    # constructing the red-green base and starting the blue-cube transfer.
    # Stage-1 and Stage-2 OOD alter the base-construction behavior, while
    # Stage-3 OOD alters the blue-cube behavior.  Keep all raw predicates, but
    # expose the boundary used by the timing sweep explicitly.
    timing_boundary_source = {
        "stage1_ood": "stage2",
        "stage2_ood": "stage2",
        "stage3_ood": "stage3",
    }
    summaries: dict[str, Any] = {}
    for split in ("stage1_ood", "stage2_ood", "stage3_ood"):
        subset = [row for row in rows if row["split"] == split]
        values: dict[str, int | None] = {}
        for stage in ("stage1", "stage2", "stage3"):
            valid = [row["boundaries"][stage] for row in subset if row["boundaries"][stage] is not None]
            values[stage] = int(round(float(np.median(valid)))) if valid else None
        summaries[split] = {
            "episodes": len(subset),
            "successful_oracle_episodes": sum(int(row["success"]) for row in subset),
            "median_boundary_steps": values,
            "timing_boundary": {
                "source": timing_boundary_source[split],
                "step": values[timing_boundary_source[split]],
            },
            "rows": subset,
        }
    args.output.mkdir(parents=True)
    if args.seed_manifest is not None:
        (args.output / "seed_manifest.json").write_text(
            args.seed_manifest.read_text(encoding="utf-8"), encoding="utf-8"
        )
    (args.output / "boundary_audit.json").write_text(
        json.dumps({"format": "stackpyramid_stage_boundary_audit_v1", "summaries": summaries}, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output / "AUDIT_COMPLETE").write_text("complete\n", encoding="utf-8")


if __name__ == "__main__":
    main()
