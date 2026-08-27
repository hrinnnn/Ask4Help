#!/usr/bin/env python3
"""Run a small Panda vegetable-basket task and visibility smoke."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _load_task_module(path: Path) -> None:
    spec = importlib.util.spec_from_file_location("panda_vegetable_basket_variants", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load task module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def _rgb(obs):
    value = _array(obs["sensor_data"]["3rd_view_camera"]["rgb"])
    return value[0].astype(np.uint8) if value.ndim == 4 else value.astype(np.uint8)


def _pose(actor):
    return _array(actor.pose.raw_pose).reshape(-1, 7)[0].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rlinf-root", type=Path, required=True)
    parser.add_argument("--task-module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=96100)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.rlinf_root))
    _load_task_module(args.task_module)

    rows = []
    for split, env_id in (
        ("id", "XVLAPandaPutVegetableInBasketID-v1"),
        ("ood", "XVLAPandaPutVegetableInBasketOOD-v1"),
    ):
        env = gym.make(
            env_id,
            obs_mode="rgb+segmentation",
            render_mode="rgb_array",
            sim_backend="physx_cpu",
        )
        try:
            obs, _ = env.reset(seed=args.seed + (0 if split == "id" else 100))
            base = env.unwrapped
            source = base.source_obj_name
            target = base.target_obj_name
            frame = _rgb(obs)
            image_path = args.output / f"{split}_reset.png"
            cv2.imwrite(str(image_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            initial_eval = base.evaluate()
            rows.append(
                {
                    "split": split,
                    "env_id": env_id,
                    "action_space": str(env.action_space),
                    "action_shape": list(getattr(env.action_space, "shape", ())),
                    "robot": type(base.agent).__name__,
                    "source_object": source,
                    "target_object": target,
                    "source_pose": _pose(base.objs[source]).tolist(),
                    "target_pose": _pose(base.objs[target]).tolist(),
                    "rgb_shape": list(frame.shape),
                    "rgb_path": str(image_path),
                    "initial_success": bool(_array(initial_eval["success"]).reshape(-1)[0]),
                }
            )
        finally:
            env.close()

    report = {
        "format": "xvla_panda_vegetable_basket_smoke_v1",
        "seed": args.seed,
        "rows": rows,
        "same_source_xy": bool(np.allclose(rows[0]["source_pose"][:2], rows[1]["source_pose"][:2])),
        "same_target_pose": bool(np.allclose(rows[0]["target_pose"], rows[1]["target_pose"])),
        "panda_action_dim_7": all(row["action_shape"] == [7] for row in rows),
        "rgb_visible_frames": all(Path(row["rgb_path"]).stat().st_size > 0 for row in rows),
    }
    (args.output / "smoke.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not (report["panda_action_dim_7"] and report["rgb_visible_frames"]):
        raise SystemExit("PANDA_TASK_SMOKE_FAILED")
    (args.output / "SMOKE_COMPLETE").write_text("complete\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

