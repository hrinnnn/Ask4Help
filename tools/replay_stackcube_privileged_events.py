#!/usr/bin/env python3
"""Replay one persisted StackCube expert demo to recover privileged milestones."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
RLINF_ROOT = ROOT / "RLinf"
sys.path[:0] = [str(ROOT), str(RLINF_ROOT)]

from rlinf.data.maniskill_stack_cube_progress import StackCubeProgressState  # noqa: E402
from tools.maniskill_pi05_vfd_online_awbc import _build_env  # noqa: E402


def _bool(value) -> bool:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return bool(torch.as_tensor(value).any().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(str(args.dataset))
    positions = [index for index, episode in enumerate(dataset.hf_dataset["episode_index"]) if int(episode) == args.episode_index]
    if not positions:
        raise ValueError(f"dataset has no episode {args.episode_index}")
    env = _build_env(100, task="stack", split="id")
    rows = []
    try:
        _obs, info = env.reset(seed=args.seed)
        progress = StackCubeProgressState()
        rows.append({"episode_index": args.episode_index, "frame_index": 0, "phi": progress.update(env, info), "valid": True})
        for frame_index, position in enumerate(positions, start=1):
            item = dataset[position]
            action = torch.as_tensor(item.get("action", item.get("actions")), dtype=torch.float32)
            _obs, _reward, terminated, truncated, info = env.step(action.reshape(1, -1).to(env.unwrapped.device))
            rows.append({"episode_index": args.episode_index, "frame_index": frame_index, "phi": progress.update(env, info), "valid": True})
            if _bool(terminated) or _bool(truncated):
                if frame_index != len(positions):
                    raise RuntimeError("replay terminated before consuming the stored expert action sequence")
                break
    finally:
        env.close()
    if rows[-1]["phi"] < 1.0:
        raise RuntimeError("stored expert replay did not reproduce privileged StackCube success")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "terminal_phi": rows[-1]["phi"], "episode_index": args.episode_index}))


if __name__ == "__main__":
    main()
