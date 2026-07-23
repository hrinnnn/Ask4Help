#!/usr/bin/env python3
"""Reload a StackCube pi0.5 checkpoint and run one real forward pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tools.maniskill_pi05_vfd_online_awbc import _build_env, _load_model, _wrap_obs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pi05-base", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=30000)
    args = parser.parse_args()

    model = _load_model(args.checkpoint, args.norm_stats, args.pi05_base)
    env = _build_env(100, task="stack", split="ood", sim_backend="physx_cpu")
    try:
        raw_obs, info = env.reset(seed=args.seed)
        with torch.inference_mode():
            actions, _ = model.predict_action_batch(
                env_obs=_wrap_obs(raw_obs, info, task="stack"),
                mode="eval",
                compute_values=False,
            )
        finite = bool(torch.isfinite(actions).all().item())
        summary = {
            "checkpoint": str(args.checkpoint),
            "shape": list(actions.shape),
            "finite": finite,
            "action_abs_max": float(actions.abs().max().item()),
        }
    finally:
        env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not finite or actions.ndim != 3 or actions.shape[-1] != 8:
        raise RuntimeError("reloaded checkpoint produced an invalid action tensor")


if __name__ == "__main__":
    main()
