#!/usr/bin/env python3
"""Run the auditable oracle gate for the UncoverSpherePlace task."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path


def _load_modules(root: Path):
    for name, path in (("rlinf", None), ("rlinf.envs", None), ("rlinf.envs.maniskill", root)):
        module = types.ModuleType(name)
        module.__path__ = [str(path)] if path else []
        sys.modules[name] = module

    def load(name: str, filename: str):
        spec = importlib.util.spec_from_file_location(name, root / filename)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load {filename}")
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


def _bool(value) -> bool:
    return bool(value.reshape(-1)[0]) if hasattr(value, "reshape") else bool(value)


def run_split(
    env_module,
    oracle_module,
    split: str,
    seeds: list[int],
    max_chunks: int,
    env_max_steps: int | None,
) -> list[dict]:
    import gymnasium as gym

    env_id = env_module.UNCOVER_ENV_IDS[split]
    results = []
    for seed in seeds:
        env = gym.make(
            env_id,
            obs_mode="state",
            control_mode="pd_joint_delta_pos",
            render_mode=None,
        )
        if env_max_steps is not None and hasattr(env, "_max_episode_steps"):
            env._max_episode_steps = env_max_steps
        env.reset(seed=seed)
        oracle = oracle_module.UncoverSpherePlacePrivilegedChunkOracle(chunk_size=10)
        last_phase = None
        terminated = False
        chunks = 0
        for chunks in range(1, max_chunks + 1):
            plan = oracle.plan(env)
            if plan.phase != last_phase:
                last_phase = plan.phase
            for step_index in range(10):
                qpos = env.unwrapped.agent.robot.get_qpos()
                _, _, terminated, truncated, _ = env.step(plan.action_at(qpos, step_index))
                if _bool(terminated) or _bool(truncated):
                    terminated = True
                    break
            if terminated:
                break
            info = env.unwrapped.evaluate()
            if _bool(info["success"]):
                break

        info = env.unwrapped.evaluate()
        results.append(
            {
                "split": split,
                "seed": seed,
                "chunks": chunks,
                "oracle_phase": oracle._phase,
                "success": _bool(info["success"]),
                "mug_parked": _bool(info["mug_parked"]),
                "ever_mug_parked": _bool(info["ever_mug_parked"]),
                "sphere_grasped": _bool(info["sphere_grasped"]),
                "ever_sphere_grasped": _bool(info["ever_sphere_grasped"]),
                "sphere_in_bowl": _bool(info["sphere_in_bowl"]),
                "sphere_released": _bool(info["sphere_released"]),
                "sphere_static": _bool(info["sphere_static"]),
            }
        )
        env.close()
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rlinf-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--max-chunks", type=int, default=250)
    parser.add_argument("--env-max-steps", type=int, default=None)
    args = parser.parse_args()

    env_module, oracle_module = _load_modules(args.rlinf_root)
    env_module.register_uncover_sphere_place_variants()
    seeds = list(range(args.num_seeds))
    rows = []
    for split in ("id", "handle_ood", "goal_ood"):
        rows.extend(
            run_split(
                env_module,
                oracle_module,
                split,
                seeds,
                args.max_chunks,
                args.env_max_steps,
            )
        )
    summary = {
        split: sum(row["success"] for row in rows if row["split"] == split)
        for split in ("id", "handle_ood", "goal_ood")
    }
    payload = {
        "task": "UncoverSpherePlace",
        "num_seeds": args.num_seeds,
        "max_chunks": args.max_chunks,
        "summary_successes": summary,
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
