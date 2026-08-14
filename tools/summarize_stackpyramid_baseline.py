#!/usr/bin/env python3
"""Create a compact, auditable summary for the StackPyramid baseline splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SPLITS = ("id_retry1", "stage1_ood", "stage2_ood", "stage3_ood")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for split in SPLITS:
        summary_path = args.root / split / "summary.json"
        complete = args.root / split / "EVAL_COMPLETE"
        if not summary_path.exists() or not complete.exists():
            raise FileNotFoundError(f"incomplete split: {split}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        expected = int(summary["episodes"])
        video_count = len(list((args.root / split / "videos").glob("*.mp4")))
        if expected != 100 or video_count != expected:
            raise ValueError(f"artifact count mismatch for {split}: {summary}")
        rows.append(
            {
                "split": "id" if split == "id_retry1" else split,
                "episodes": expected,
                "ever_grasped": int(summary["ever_grasped"]),
                "ever_base_completed": int(summary["ever_base_completed"]),
                "strict_success": int(summary["strict_success"]),
                "video_count": video_count,
                "path": str((args.root / split).resolve()),
            }
        )
    payload = {
        "format": "stackpyramid_xvla_baseline_comparison_v1",
        "checkpoint": "ckpt-10000",
        "policy_evaluation": "pure_policy",
        "max_episode_steps": 250,
        "execute_horizon": 5,
        "flow_steps": 5,
        "rows": rows,
    }
    (args.root / "comparison.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# StackPyramid X-VLA baseline",
        "",
        "固定 checkpoint `ckpt-10000`，纯 policy，`execute_horizon=5`，",
        "`max_episode_steps=250`，每个 split 100 条。主结果同时报告抓取、",
        "底座完成和严格金字塔成功。",
        "",
        "| Split | Ever grasped | Base completed | Strict success | Videos |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['split']} | {row['ever_grasped']}/100 | "
            f"{row['ever_base_completed']}/100 | {row['strict_success']}/100 | "
            f"{row['video_count']} |"
        )
    lines += [
        "",
        f"结果根：`{args.root.resolve()}`",
        "",
        "说明：Stage-1 和 Stage-3 的失败发生在不同任务阶段，不能合并成一个笼统的 OOD 标签。",
    ]
    (args.root / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
