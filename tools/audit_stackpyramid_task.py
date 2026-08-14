#!/usr/bin/env python3
"""Audit paired reset semantics for the controlled StackPyramid variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import mani_skill.envs  # noqa: F401

from tools.stackpyramid_task import (
    STACKPYRAMID_SPLITS,
    reset_metadata,
    register_stackpyramid_splits,
    stackpyramid_env_id,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--sim-backend", choices=("gpu", "cpu"), default="cpu")
    parser.add_argument("--render-backend", choices=("gpu", "cpu"), default="cpu")
    args = parser.parse_args()
    register_stackpyramid_splits()
    records = []
    for seed in range(12000, 12000 + args.seeds):
        per_seed = {}
        for split in STACKPYRAMID_SPLITS:
            env = gym.make(
                stackpyramid_env_id(split),
                num_envs=1,
                obs_mode="state",
                control_mode="pd_joint_pos",
                sim_backend=args.sim_backend,
                render_backend=args.render_backend,
            )
            _, info = env.reset(seed=seed)
            metadata = reset_metadata(env, split=split)
            evaluation = env.unwrapped.evaluate()
            per_seed[split] = {
                "metadata": metadata,
                "initial_success": bool(evaluation["success"].detach().cpu().reshape(-1)[0].item()),
                "reset_info_keys": sorted(str(key) for key in info),
            }
            env.close()
        id_positions = np.asarray(
            [per_seed["id"]["metadata"]["cube_poses"][color]["p"][:2] for color in ("red", "green", "blue")]
        )
        for split in STACKPYRAMID_SPLITS[1:]:
            split_positions = np.asarray(
                [per_seed[split]["metadata"]["cube_poses"][color]["p"][:2] for color in ("red", "green", "blue")]
            )
            affected = {"stage1_ood": 0, "stage2_ood": 1, "stage3_ood": 2}[split]
            non_target_delta = np.delete(split_positions - id_positions, affected, axis=0)
            if not np.allclose(non_target_delta, 0.0, atol=1e-7):
                raise AssertionError(f"paired reset changed non-target objects for {split}, seed {seed}")
            if np.linalg.norm(split_positions[affected] - id_positions[affected]) <= 0.05:
                raise AssertionError(f"OOD shift too small for {split}, seed {seed}")
        records.append({"seed": seed, "splits": per_seed})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "format": "stackpyramid_paired_reset_audit_v1",
                "env_ids": {split: stackpyramid_env_id(split) for split in STACKPYRAMID_SPLITS},
                "seeds": [record["seed"] for record in records],
                "records": records,
                "passed": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
