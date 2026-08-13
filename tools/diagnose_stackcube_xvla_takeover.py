#!/usr/bin/env python3
"""Diagnose one controlled X-VLA StackCube takeover at chunk resolution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "RLinf")]

from rlinf.envs.maniskill.stack_cube_privileged_oracle import StackCubePrivilegedChunkOracle
from rlinf.envs.maniskill.stack_cube_variants import STACK_CUBE_TASK
from tools.collect_stackcube_xvla_dagger import FailureRecoveryState, _make_env
from tools.evaluate_stackcube_xvla import bool_scalar, clip_action_chunk
from tools.stackcube_stage2_ood import register_stack_cube_splits
from tools.xvla_airplane_runtime import XVLAAirplanePolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--condition", choices=("post_grasp", "post_lift", "failure_recovery"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    policy = XVLAAirplanePolicy(args.checkpoint, args.xvla_root)
    register_stack_cube_splits()
    env = _make_env("stage2_ood")
    obs, _ = env.reset(seed=args.seed)
    low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    state = FailureRecoveryState()
    oracle = StackCubePrivilegedChunkOracle(chunk_size=5)
    active = False
    rows = []
    success = False
    try:
        for decision in range(30):
            if not active and state.timing_trigger(args.condition):
                active = True
                initialized = oracle.initialize_from_state(env)
            else:
                initialized = None
            if active:
                plan = oracle.plan(env)
                candidate = plan.actions
                phase = plan.phase
                planned = plan.planning_succeeded
            else:
                predicted, _, _, _ = policy.predict(obs, STACK_CUBE_TASK, seed=args.seed * 1000 + decision, steps=10)
                candidate = clip_action_chunk(predicted, low, high, 5)
                plan = None
                phase = "policy"
                planned = True
            before_grasp = bool_scalar(env.unwrapped.agent.is_grasping(env.unwrapped.cubeA))
            before_z = float(env.unwrapped.cubeA.pose.p.reshape(-1, 3)[0, 2].item())
            for local_step, action in enumerate(candidate):
                if plan is not None:
                    action = plan.action_at(obs["agent"]["qpos"], local_step)
                obs, _, terminated, truncated, info = env.step(
                    torch.as_tensor(action, device=env.unwrapped.device).unsqueeze(0)
                )
                current_grasped = bool_scalar(info.get("is_cubeA_grasped", False))
                current_on_cube = bool_scalar(info.get("is_cubeA_on_cubeB", False))
                success = bool_scalar(info.get("success", False))
                cube_z = float(env.unwrapped.cubeA.pose.p.reshape(-1, 3)[0, 2].item())
                state.update(env_step=decision * 5 + local_step + 1,
                             currently_grasped=current_grasped, on_cube=current_on_cube,
                             success=success, cube_z=cube_z)
                if success or bool_scalar(terminated) or bool_scalar(truncated):
                    break
            rows.append({
                "decision": decision, "controller": "expert" if active else "policy",
                "initialized_phase": initialized, "phase": phase,
                "planning_succeeded": planned, "before_grasped": before_grasp,
                "before_cube_z": before_z, "after_grasped": state.currently_grasped,
                "after_cube_z": state.cube_z, "success": success,
            })
            if success:
                break
    finally:
        env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"seed": args.seed, "condition": args.condition,
                                      "success": success, "rows": rows}, indent=2) + "\n")
    print(json.dumps({"success": success, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
