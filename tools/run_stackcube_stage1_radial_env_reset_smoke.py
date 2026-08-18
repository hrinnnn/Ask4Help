#!/usr/bin/env python3
"""Real ManiSkill registration/reset smoke for radial StackCube variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from stackcube_stage1_radial_ood import (
    RADIAL_SHIFT_DISTANCE_M,
    STACK_CUBE_RADIAL_OOD_SPLIT,
    radial_unit,
    radial_env_id,
    register_stackcube_stage1_radial_variants,
    stage1_radial_reset_metadata,
    stage1_radial_state_record,
)


def make_env(split: str) -> Any:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    return gym.make(
        radial_env_id(split),
        robot_uids="panda_wristcam",
        num_envs=1,
        obs_mode="rgb",
        control_mode="pd_joint_delta_pos",
        reward_mode="sparse",
        render_mode="rgb_array",
        sim_backend="physx_cpu",
        sim_config={"sim_freq": 100, "control_freq": 10},
        sensor_configs={"width": 384, "height": 384},
        max_episode_steps=100,
    )


def run(seed_start: int, episodes: int) -> dict[str, Any]:
    register_stackcube_stage1_radial_variants()
    envs = {split: make_env(split) for split in ("id", STACK_CUBE_RADIAL_OOD_SPLIT)}
    rows: list[dict[str, Any]] = []
    try:
        for offset in range(episodes):
            seed = seed_start + offset
            observations = {}
            for split, env in envs.items():
                env.reset(seed=seed)
                metadata = stage1_radial_reset_metadata(env, split=split, paired_seed=seed)
                state = stage1_radial_state_record(env)
                observations[split] = {"metadata": metadata, "state": state}
            id_meta = observations["id"]["metadata"]
            ood_meta = observations[STACK_CUBE_RADIAL_OOD_SPLIT]["metadata"]
            green_id = np.asarray(id_meta["cube_b_xy"], dtype=np.float64)
            green_ood = np.asarray(ood_meta["cube_b_xy"], dtype=np.float64)
            red_id = np.asarray(id_meta["cube_a_xy"], dtype=np.float64)
            red_ood = np.asarray(ood_meta["cube_a_xy"], dtype=np.float64)
            expected_shift = RADIAL_SHIFT_DISTANCE_M * radial_unit(red_id, green_id)
            id_qpos = np.asarray(id_meta["robot_qpos"], dtype=np.float64)
            ood_qpos = np.asarray(ood_meta["robot_qpos"], dtype=np.float64)
            id_predicates = observations["id"]["state"]["predicates"]
            ood_predicates = observations[STACK_CUBE_RADIAL_OOD_SPLIT]["state"]["predicates"]
            rows.append(
                {
                    "seed": seed,
                    "green_pose_equal": bool(np.array_equal(green_id, green_ood)),
                    "robot_qpos_equal": bool(np.array_equal(id_qpos, ood_qpos)),
                    "red_shift_matches_rule": bool(np.allclose(red_ood - red_id, expected_shift, atol=1e-12)),
                    "red_shift_norm_m": float(np.linalg.norm(red_ood - red_id)),
                    "metadata_complete": all(
                        key in id_meta and key in ood_meta
                        for key in ("cube_a_xy", "cube_a_id_xy", "cube_b_xy", "red_shift_xy", "robot_qpos")
                    ),
                    "reset_predicates_all_false": not any(id_predicates.values()) and not any(ood_predicates.values()),
                }
            )
    finally:
        for env in envs.values():
            env.close()

    report = {
        "task": "StackCube Stage1 radial-distance OOD real-runtime reset smoke",
        "seed_start": seed_start,
        "episodes": episodes,
        "env_ids": {split: radial_env_id(split) for split in envs},
        "red_ood_rule": "red_id + 0.04 * normalize(red_id - green)",
        "green_pose_strictly_equal": all(row["green_pose_equal"] for row in rows),
        "robot_qpos_strictly_equal": all(row["robot_qpos_equal"] for row in rows),
        "red_shift_rule_strict": all(row["red_shift_matches_rule"] for row in rows),
        "red_shift_norm_strict": all(
            np.isclose(row["red_shift_norm_m"], RADIAL_SHIFT_DISTANCE_M, atol=1e-12)
            for row in rows
        ),
        "metadata_complete": all(row["metadata_complete"] for row in rows),
        "reset_predicates_all_false": all(row["reset_predicates_all_false"] for row in rows),
        "rows": rows,
    }
    report["passed"] = all(
        report[key]
        for key in (
            "green_pose_strictly_equal",
            "robot_qpos_strictly_equal",
            "red_shift_rule_strict",
            "red_shift_norm_strict",
            "metadata_complete",
            "reset_predicates_all_false",
        )
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=971000)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.seed_start, args.episodes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
