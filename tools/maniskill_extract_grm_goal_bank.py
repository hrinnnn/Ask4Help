#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _to_image(value) -> Image.Image:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim == 3 and array.shape[0] in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if np.issubdtype(array.dtype, np.floating):
        array = array * 255.0 if array.max() <= 1.0 else array
    array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array[..., :3]).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a ManiSkill GRM goal image from a successful LeRobot demo."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--main-key", default="image")
    parser.add_argument("--wrist-key", default="wrist_image")
    args = parser.parse_args()

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(args.dataset)
    episode_positions = [
        position
        for position, episode_index in enumerate(dataset.hf_dataset["episode_index"])
        if int(episode_index) == args.episode_index
    ]
    if not episode_positions:
        raise ValueError(f"dataset has no episode_index={args.episode_index}")

    final_position = episode_positions[-1]
    item = dataset[final_position]
    task_dir = Path(args.output_dir).expanduser() / f"task_{args.task_id:03d}"
    task_dir.mkdir(parents=True, exist_ok=True)
    _to_image(item[args.main_key]).save(task_dir / "goal_main.png")
    wrist = item.get(args.wrist_key, item[args.main_key])
    _to_image(wrist).save(task_dir / "goal_wrist.png")

    task_description = str(item.get("task", "insert the peg in the hole"))
    metadata = {
        "task_id": args.task_id,
        "task_description": task_description,
        "source_demo": str(Path(args.dataset).expanduser()),
        "source_episode_index": args.episode_index,
        "source_dataset_index": final_position,
        "views": {"main": "goal_main.png", "wrist": "goal_wrist.png"},
    }
    (task_dir / "meta.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(task_dir)


if __name__ == "__main__":
    main()
