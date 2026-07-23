#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from rlinf.data.awbc_annotation import (
    DatasetFrame,
    ProgressEstimate,
    build_awbc_manifest_rows,
    select_episode_anchors,
)
from rlinf.models.embodiment.reward.dopamine_grm_reward_model import (
    DopamineGRMRewardModel,
)


class _CapturingDopamineGRM(DopamineGRMRewardModel):
    def __init__(self, cfg):
        self.captured_records: list[dict] = []
        super().__init__(cfg)

    def _write_metric_record(self, record):
        self.captured_records.append(record)


def _parse_successful_episodes(value: str | None) -> set[int]:
    if not value:
        return set()
    path = Path(value).expanduser()
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("successful_episodes", [])
        return {int(item) for item in data}
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def _load_cache(path: Path) -> dict[int, ProgressEstimate]:
    if not path.is_file():
        return {}
    estimates = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            data = json.loads(line)
            estimate = ProgressEstimate(
                dataset_index=int(data["dataset_index"]),
                episode_index=int(data["episode_index"]),
                frame_index=int(data["frame_index"]),
                phi=None if data.get("phi") is None else float(data["phi"]),
                valid=bool(data["valid"]),
                confidence=float(data.get("confidence", 1.0)),
                mode_phis=dict(data.get("mode_phis", {})),
            )
            if estimate.dataset_index in estimates:
                raise ValueError(
                    f"cache has duplicate dataset_index={estimate.dataset_index}"
                )
            estimates[estimate.dataset_index] = estimate
    return estimates


def _append_cache(path: Path, estimate: ProgressEstimate) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(
                {
                    "dataset_index": estimate.dataset_index,
                    "episode_index": estimate.episode_index,
                    "frame_index": estimate.frame_index,
                    "phi": estimate.phi,
                    "valid": estimate.valid,
                    "confidence": estimate.confidence,
                    "mode_phis": estimate.mode_phis,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def _reset_estimator(estimator: _CapturingDopamineGRM) -> None:
    estimator.prev_phi.zero_()
    estimator.has_prev_phi.zero_()
    estimator.reward_call_counts.zero_()
    estimator.previous_grm_obs = [None] * estimator.num_envs


def _load_source_manifest(path: str | None) -> dict[int, str]:
    if not path:
        return {}
    source_by_index: dict[int, str] = {}
    with Path(path).expanduser().open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            index = int(row["dataset_index"])
            source = str(row["source"])
            if source not in {"policy", "expert"}:
                raise ValueError(f"invalid source for dataset_index={index}: {source}")
            if index in source_by_index:
                raise ValueError(f"duplicate source dataset_index={index}")
            source_by_index[index] = source
    return source_by_index


def _observation(item, start_item, *, done: bool, success: bool) -> dict:
    main = item["image"].unsqueeze(0)
    wrist = item.get("wrist_image", item["image"]).unsqueeze(0)
    start_main = start_item["image"].unsqueeze(0)
    start_wrist = start_item.get("wrist_image", start_item["image"]).unsqueeze(0)
    return {
        "main_images": main,
        "wrist_images": wrist,
        "reference_start_main_images": start_main,
        "reference_start_wrist_images": start_wrist,
        "task_descriptions": [str(item.get("task", "insert the peg in the hole"))],
        "task_ids": torch.tensor([int(item.get("task_index", 0))]),
        "dones": torch.tensor([done]),
        "env_infos": {
            "episode": {"success_once": torch.tensor([success])}
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate a ManiSkill LeRobot dataset for Robo-Dopamine AWBC."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--goal-bank-dir", required=True)
    parser.add_argument("--grm-endpoint", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-path")
    parser.add_argument("--source", choices=("expert", "policy"), required=True)
    parser.add_argument(
        "--source-manifest",
        help="Optional JSONL with per-dataset-index policy/expert source labels.",
    )
    parser.add_argument("--stride-steps", type=int, default=5)
    parser.add_argument(
        "--lookahead-steps",
        type=int,
        help="Progress horizon for AWBC targets; defaults to stride-steps.",
    )
    parser.add_argument("--successful-episodes")
    parser.add_argument("--assume-all-success", action="store_true")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    args = parser.parse_args()

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(args.dataset)
    episode_column = [int(value) for value in dataset.hf_dataset["episode_index"]]
    frame_column = [int(value) for value in dataset.hf_dataset["frame_index"]]
    frames = [
        DatasetFrame(position, episode_index, frame_column[position])
        for position, episode_index in enumerate(episode_column)
    ]
    episodes = sorted(set(episode_column))
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]
        frames = [frame for frame in frames if frame.episode_index in episodes]

    successful_episodes = _parse_successful_episodes(args.successful_episodes)
    if args.assume_all_success:
        successful_episodes = set(episodes)
    if args.source == "policy" and not args.assume_all_success and not args.successful_episodes:
        raise ValueError(
            "policy annotation requires --successful-episodes or --assume-all-success"
        )

    config = OmegaConf.create(
        {
            "grm_endpoint": args.grm_endpoint,
            "model_name": args.model_name,
            "goal_bank_dir": args.goal_bank_dir,
            "modes": ["incremental", "forward", "backward"],
            "num_envs": 1,
            "grm_interval_chunks": 1,
            "consistency_check": True,
            "request_timeout": args.request_timeout,
        }
    )
    estimator = _CapturingDopamineGRM(config)
    output_path = Path(args.output).expanduser()
    cache_path = Path(
        args.cache_path or f"{output_path}.progress_cache.jsonl"
    ).expanduser()
    estimates = _load_cache(cache_path)

    for episode_index in episodes:
        episode_frames = [
            frame for frame in frames if frame.episode_index == episode_index
        ]
        anchors = select_episode_anchors(episode_frames, args.stride_steps)
        start_frame = anchors[0]
        start_item = dataset[start_frame.dataset_index]
        _reset_estimator(estimator)

        if start_frame.dataset_index not in estimates:
            start_estimate = ProgressEstimate(
                dataset_index=start_frame.dataset_index,
                episode_index=episode_index,
                frame_index=start_frame.frame_index,
                phi=0.0,
                valid=True,
                confidence=1.0,
            )
            estimates[start_frame.dataset_index] = start_estimate
            _append_cache(cache_path, start_estimate)

        for anchor_position, anchor in enumerate(anchors[1:], start=1):
            if anchor.dataset_index in estimates:
                cached = estimates[anchor.dataset_index]
                if cached.valid and cached.phi is not None:
                    cached_item = dataset[anchor.dataset_index]
                    estimator.prev_phi[0] = cached.phi
                    estimator.has_prev_phi[0] = True
                    estimator.previous_grm_obs[0] = {
                        "main_images": cached_item["image"],
                        "wrist_images": cached_item.get(
                            "wrist_image", cached_item["image"]
                        ),
                    }
                continue

            item = dataset[anchor.dataset_index]
            terminal = anchor_position == len(anchors) - 1
            success = terminal and episode_index in successful_episodes
            before_records = len(estimator.captured_records)
            estimator.compute_reward(
                _observation(item, start_item, done=terminal, success=success)
            )
            if len(estimator.captured_records) != before_records + 1:
                raise RuntimeError("GRM annotation did not emit exactly one metric record")
            metric = estimator.captured_records[-1]
            valid = bool(metric.get("valid", False))
            phi = metric.get("phi_next") if valid else None
            mode_phis = {
                mode: values.get("phi")
                for mode, values in metric.get("mode_metrics", {}).items()
            }
            estimate = ProgressEstimate(
                dataset_index=anchor.dataset_index,
                episode_index=episode_index,
                frame_index=anchor.frame_index,
                phi=None if phi is None else float(phi),
                valid=valid and phi is not None,
                confidence=float(metric.get("consistency_confidence") or 1.0)
                if valid
                else 0.0,
                mode_phis=mode_phis,
            )
            estimates[anchor.dataset_index] = estimate
            _append_cache(cache_path, estimate)

    rows = build_awbc_manifest_rows(
        frames,
        list(estimates.values()),
        stride_steps=args.stride_steps,
        source=args.source,
        successful_episodes=successful_episodes,
        lookahead_steps=args.lookahead_steps,
    )
    source_by_index = _load_source_manifest(args.source_manifest)
    if source_by_index:
        expected = {int(row["dataset_index"]) for row in rows}
        if set(source_by_index) != expected:
            missing = sorted(expected - set(source_by_index))
            extra = sorted(set(source_by_index) - expected)
            raise ValueError(
                f"source manifest index mismatch: missing={missing[:8]} extra={extra[:8]}"
            )
        for row in rows:
            row["source"] = source_by_index[int(row["dataset_index"])]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    valid_rows = sum(bool(row["valid"]) for row in rows)
    summary = {
        "dataset": args.dataset,
        "output": str(output_path),
        "cache": str(cache_path),
        "source": args.source,
        "stride_steps": args.stride_steps,
        "lookahead_steps": args.lookahead_steps or args.stride_steps,
        "episodes": len(episodes),
        "rows": len(rows),
        "valid_rows": valid_rows,
        "valid_rate": valid_rows / max(1, len(rows)),
    }
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
