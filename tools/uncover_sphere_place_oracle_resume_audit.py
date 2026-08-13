#!/usr/bin/env python3
"""Verify oracle continuation from physical intermediate simulator states."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path


def load_modules(root: Path):
    for name, path in (("rlinf", None), ("rlinf.envs", None), ("rlinf.envs.maniskill", root)):
        module = types.ModuleType(name)
        module.__path__ = [str(path)] if path else []
        sys.modules[name] = module

    def load(name: str, filename: str):
        spec = importlib.util.spec_from_file_location(name, root / filename)
        if spec is None or spec.loader is None:
            raise RuntimeError(filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    env_module = load("rlinf.envs.maniskill.uncover_sphere_place", "uncover_sphere_place.py")
    load("rlinf.envs.maniskill.peg_privileged_oracle", "peg_privileged_oracle.py")
    oracle_module = load(
        "rlinf.envs.maniskill.uncover_sphere_place_privileged_oracle",
        "uncover_sphere_place_privileged_oracle.py",
    )
    return env_module, oracle_module


def flag(value) -> bool:
    return bool(value.reshape(-1)[0]) if hasattr(value, "reshape") else bool(value)


def step_plan(env, oracle, max_chunks: int, stop):
    for chunk in range(max_chunks):
        plan = oracle.plan(env)
        for index in range(10):
            qpos = env.unwrapped.agent.robot.get_qpos()
            _, _, terminated, truncated, _ = env.step(plan.action_at(qpos, index))
            if flag(terminated) or flag(truncated):
                return chunk + 1, False
        info = env.unwrapped.evaluate()
        if stop(info, oracle):
            return chunk + 1, True
    return max_chunks, False


def run(split: str, seed: int, env_module, oracle_module) -> dict:
    import gymnasium as gym

    env = gym.make(
        env_module.UNCOVER_ENV_IDS[split],
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        render_mode=None,
    )
    env.reset(seed=seed)
    first = oracle_module.UncoverSpherePlacePrivilegedChunkOracle(chunk_size=10)
    first_chunks, reached_sphere = step_plan(
        env,
        first,
        250,
        lambda _info, oracle: oracle._phase == "sphere_reach",
    )
    if not reached_sphere:
        env.close()
        return {"split": split, "seed": seed, "stage": "cover_parked", "success": False}

    resumed_after_cover = oracle_module.UncoverSpherePlacePrivilegedChunkOracle(chunk_size=10)
    resumed_after_cover.resume_from_current_state("sphere_reach")
    cover_chunks, cover_success = step_plan(
        env,
        resumed_after_cover,
        250,
        lambda info, _oracle: flag(info["success"]),
    )

    env.reset(seed=seed)
    first = oracle_module.UncoverSpherePlacePrivilegedChunkOracle(chunk_size=10)
    first_chunks_grasp, reached_grasp = step_plan(
        env,
        first,
        250,
        lambda info, _oracle: flag(info["sphere_grasped"]),
    )
    if not reached_grasp:
        env.close()
        return {
            "split": split,
            "seed": seed,
            "stage": "sphere_grasped",
            "cover_parked_continuation": cover_success,
            "success": False,
        }

    resumed_after_grasp = oracle_module.UncoverSpherePlacePrivilegedChunkOracle(chunk_size=10)
    resumed_after_grasp.resume_from_current_state("sphere_lift")
    grasp_chunks, grasp_success = step_plan(
        env,
        resumed_after_grasp,
        250,
        lambda info, _oracle: flag(info["success"]),
    )
    env.close()
    return {
        "split": split,
        "seed": seed,
        "cover_parked_first_chunks": first_chunks,
        "cover_parked_continuation_chunks": cover_chunks,
        "cover_parked_continuation": cover_success,
        "sphere_grasped_first_chunks": first_chunks_grasp,
        "sphere_grasped_continuation_chunks": grasp_chunks,
        "sphere_grasped_continuation": grasp_success,
        "success": cover_success and grasp_success,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rlinf-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-seeds", type=int, default=3)
    args = parser.parse_args()
    env_module, oracle_module = load_modules(args.rlinf_root)
    env_module.register_uncover_sphere_place_variants()
    rows = [
        run(split, seed, env_module, oracle_module)
        for split in ("id", "handle_ood", "goal_ood")
        for seed in range(args.num_seeds)
    ]
    payload = {
        "task": "UncoverSpherePlace",
        "num_seeds": args.num_seeds,
        "continuation_successes": sum(row["success"] for row in rows),
        "total": len(rows),
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
