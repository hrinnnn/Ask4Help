#!/usr/bin/env python3
"""Collect four fair X-VLA StackCube DAgger/BC datasets."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "RLinf")]

from rlinf.envs.maniskill.stack_cube_privileged_oracle import (  # noqa: E402
    StackCubePrivilegedChunkOracle,
)
from rlinf.envs.maniskill.stack_cube_variants import (  # noqa: E402
    STACK_CUBE_ID_ENV_ID,
    STACK_CUBE_OOD_ENV_ID,
    STACK_CUBE_TASK,
    register_controlled_stack_cube_variants,
    reset_metadata,
)
from toolkits.lerobot.collect_maniskill_peg_lerobot_joint import (  # noqa: E402
    MAIN_CAMERA_CANDIDATES,
    WRIST_CAMERA_CANDIDATES,
    _build_frames,
    _create_dataset,
    _extract_record,
    _select_camera,
)
from toolkits.lerobot.collect_maniskill_plug_lerobot_joint import (  # noqa: E402
    write_episode_video_durably,
)
from tools.evaluate_stackcube_xvla import bool_scalar, clip_action_chunk  # noqa: E402
from tools.xvla_airplane_runtime import XVLAAirplanePolicy  # noqa: E402


METHODS = ("vlm_bridge_pca", "offline_oracle", "failure_recovery", "diffdagger")
EXECUTE_HORIZON = 5
TASK_HORIZON = 150
FAILURE_RECOVERY_STEP = 50


@dataclass(frozen=True)
class BridgePCA:
    mean: torch.Tensor
    eigenvectors: torch.Tensor
    pca_dim: int

    @classmethod
    def load(cls, path: Path, device: torch.device) -> "BridgePCA":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        source = payload["layers"]["vlm_action_bridge"]
        return cls(
            mean=source["mean"].to(device=device, dtype=torch.float32),
            eigenvectors=source["eigenvectors"].to(device=device, dtype=torch.float32),
            pca_dim=int(source["pca_dim"]),
        )

    @torch.inference_mode()
    def score(self, feature: torch.Tensor) -> float:
        centered = feature.to(self.mean.device, torch.float32) - self.mean
        coordinates = centered @ self.eigenvectors
        residual_dims = self.eigenvectors.shape[1] - self.pca_dim
        return float(coordinates[:, :residual_dims].norm(dim=-1)[0].item())


def alternating_split(attempt_index: int) -> str:
    if attempt_index < 0:
        raise ValueError("attempt_index must be non-negative")
    return "id" if attempt_index % 2 == 0 else "ood"


def consecutive_gate(score: float, threshold: float, count: int, patience: int) -> tuple[int, bool]:
    count = count + 1 if score > threshold else 0
    return count, count >= patience


def admitted_suffix(success: bool, expert_start: int | None, action_count: int) -> tuple[int, int] | None:
    """Retain every real expert action; temporal masking handles short tails."""
    if not success or expert_start is None or expert_start >= action_count:
        return None
    return expert_start, action_count


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _make_env(split: str):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    return gym.make(
        STACK_CUBE_ID_ENV_ID if split == "id" else STACK_CUBE_OOD_ENV_ID,
        robot_uids="panda_wristcam",
        num_envs=1,
        obs_mode="rgb",
        control_mode="pd_joint_delta_pos",
        reward_mode="sparse",
        render_mode="rgb_array",
        sim_backend="physx_cpu",
        sim_config={"sim_freq": 100, "control_freq": 10},
        sensor_configs={"width": 384, "height": 384},
        max_episode_steps=TASK_HORIZON,
    )


def _run_attempt(
    *,
    method: str,
    split: str,
    seed: int,
    env: Any,
    policy: XVLAAirplanePolicy | None,
    bridge_pca: BridgePCA,
    pca_threshold: float,
    diff_threshold: float,
    diff_patience: int,
    diff_timesteps: int,
    flow_steps: int,
) -> tuple[list[Any], list[np.ndarray], list[str], int | None, dict[str, Any]]:
    raw_obs, _ = env.reset(seed=seed)
    records = [_extract_record(raw_obs)]
    actions: list[np.ndarray] = []
    sources: list[str] = []
    timeline: list[dict[str, Any]] = []
    expert_start = 0 if method == "offline_oracle" else None
    gate_count = 0
    oracle = StackCubePrivilegedChunkOracle(chunk_size=EXECUTE_HORIZON)
    success = grasped = on_cube = False
    terminated = truncated = False

    while len(actions) < TASK_HORIZON and not (success or bool_scalar(terminated) or bool_scalar(truncated)):
        step = len(actions)
        score = threshold = None
        alarm = expert_start is not None
        if expert_start is None and method == "failure_recovery" and step >= FAILURE_RECOVERY_STEP:
            expert_start, alarm = step, True
        elif expert_start is None:
            assert policy is not None
            predicted, feature, inputs, encoding = policy.predict(
                raw_obs, STACK_CUBE_TASK, seed=seed * 1000 + step, steps=flow_steps
            )
            if method == "vlm_bridge_pca":
                score, threshold = bridge_pca.score(feature), pca_threshold
                gate_count, alarm = consecutive_gate(score, threshold, gate_count, 1)
            elif method == "diffdagger":
                score = policy.diffdagger_score(
                    inputs,
                    encoding,
                    predicted,
                    num_timesteps=diff_timesteps,
                    num_noise_samples=1,
                )
                threshold = diff_threshold
                gate_count, alarm = consecutive_gate(
                    score, threshold, gate_count, diff_patience
                )
            if alarm:
                expert_start = step

        if expert_start is not None:
            plan = oracle.plan(env)
            candidate = plan.actions
            source = "expert"
        else:
            plan = None
            source = "policy"
            low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
            high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
            candidate = clip_action_chunk(predicted, low, high, EXECUTE_HORIZON)

        timeline.append(
            {
                "decision_index": len(timeline),
                "env_step": step,
                "controller": source,
                "score": score,
                "threshold": threshold,
                "alarm": bool(alarm),
            }
        )
        for local_step, action in enumerate(candidate):
            if plan is not None:
                action = plan.action_at(raw_obs["agent"]["qpos"], local_step)
            raw_obs, _, terminated, truncated, info = env.step(
                torch.as_tensor(action, device=env.unwrapped.device).unsqueeze(0)
            )
            actions.append(np.asarray(action, dtype=np.float32))
            sources.append(source)
            records.append(_extract_record(raw_obs))
            grasped |= bool_scalar(info.get("is_cubeA_grasped", False))
            on_cube |= bool_scalar(info.get("is_cubeA_on_cubeB", False))
            success |= bool_scalar(info.get("success", False))
            if success or bool_scalar(terminated) or bool_scalar(truncated):
                break

    return records, actions, sources, expert_start, {
        "seed": seed,
        "split": split,
        "method": method,
        "success": success,
        "grasped_once": grasped,
        "on_cube_once": on_cube,
        "steps": len(actions),
        "expert_start_step": expert_start,
        "expert_action_steps": 0 if expert_start is None else len(actions) - expert_start,
        "timeline": timeline,
        **reset_metadata(env, split=split),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--internal-assets", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-id", type=Path, required=True)
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--offline-per-split", type=int, default=50)
    parser.add_argument("--id-seed", type=int, default=70000)
    parser.add_argument("--ood-seed", type=int, default=80000)
    parser.add_argument("--max-attempts", type=int, default=5000)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--diff-timesteps", type=int, default=16)
    parser.add_argument("--diff-patience", type=int, default=2)
    parser.add_argument("--diff-threshold", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() or args.repo_id.exists():
        raise FileExistsError("output and dataset paths must be new")
    args.output_dir.mkdir(parents=True)
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    methods = calibration["methods"]
    pca_threshold = float(methods["vlm_action_bridge_pca"]["threshold"])
    diff_threshold = (
        float(args.diff_threshold)
        if args.diff_threshold is not None
        else float(methods["diffdagger"]["threshold"])
    )
    device = torch.device("cuda")
    bridge_pca = BridgePCA.load(args.internal_assets, device)
    policy = None if args.method == "offline_oracle" else XVLAAirplanePolicy(
        args.checkpoint, args.xvla_root
    )
    _write_json(
        args.output_dir / "collection_provenance.json",
        {
            "format": "xvla_stackcube_four_group_collection_v1",
            "method": args.method,
            "checkpoint": str(args.checkpoint.resolve()),
            "internal_assets": str(args.internal_assets.resolve()),
            "calibration": str(args.calibration.resolve()),
            "bridge_pca_threshold": pca_threshold,
            "diffdagger_threshold": diff_threshold,
            "raw_split_schedule": (
                "offline_exact_50_50"
                if args.method == "offline_oracle"
                else "strict_id_ood_alternation"
            ),
            "task_horizon": TASK_HORIZON,
            "failure_recovery_step": FAILURE_RECOVERY_STEP,
            "admission": "task_success_and_nonempty_full_expert_suffix",
            "tail_handling": "all_real_actions_saved; temporal action_valid_mask during training",
        },
    )
    register_controlled_stack_cube_variants()
    envs = {split: _make_env(split) for split in ("id", "ood")}
    dataset = None
    accepted = 0
    accepted_by_split = {"id": 0, "ood": 0}
    raw_by_split = {"id": 0, "ood": 0}
    next_seed = {"id": args.id_seed, "ood": args.ood_seed}
    try:
        for attempt_index in range(args.max_attempts):
            if accepted >= args.target:
                break
            split = alternating_split(attempt_index)
            if args.method == "offline_oracle" and accepted_by_split[split] >= args.offline_per_split:
                split = "ood" if split == "id" else "id"
            if args.method == "offline_oracle" and accepted_by_split[split] >= args.offline_per_split:
                break
            seed = next_seed[split]
            next_seed[split] += 1
            raw_by_split[split] += 1
            records, actions, sources, expert_start, row = _run_attempt(
                method=args.method,
                split=split,
                seed=seed,
                env=envs[split],
                policy=policy,
                bridge_pca=bridge_pca,
                pca_threshold=pca_threshold,
                diff_threshold=diff_threshold,
                diff_patience=args.diff_patience,
                diff_timesteps=args.diff_timesteps,
                flow_steps=args.flow_steps,
            )
            row["attempt_index"] = attempt_index
            frames = _build_frames(
                records=records,
                actions=actions,
                task=STACK_CUBE_TASK,
                main_camera=_select_camera(
                    records[0].obs, "", ("base_camera",) + MAIN_CAMERA_CANDIDATES, "main"
                ),
                wrist_camera=_select_camera(
                    records[0].obs, "", ("hand_camera",) + WRIST_CAMERA_CANDIDATES, "wrist"
                ),
            )
            row["video"] = str(
                write_episode_video_durably(
                    frames,
                    video_dir=args.output_dir / "raw_archive/videos",
                    episode_index=attempt_index,
                    seed=seed,
                    fps=10,
                )
            )
            action_stem = args.output_dir / "raw_archive/actions" / f"episode_{attempt_index:06d}_seed_{seed:06d}"
            action_stem.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(action_stem) + ".npy", np.asarray(actions, dtype=np.float32))
            Path(str(action_stem) + ".sources.json").write_text(
                json.dumps(sources) + "\n", encoding="utf-8"
            )
            admitted = admitted_suffix(row["success"], expert_start, len(actions))
            row["accepted"] = admitted is not None
            _append_jsonl(args.output_dir / "episodes.jsonl", row)
            if admitted is not None:
                begin, end = admitted
                label_frames = _build_frames(
                    records=records[begin : end + 1],
                    actions=actions[begin:end],
                    task=STACK_CUBE_TASK,
                    main_camera=_select_camera(
                        records[0].obs, "", ("base_camera",) + MAIN_CAMERA_CANDIDATES, "main"
                    ),
                    wrist_camera=_select_camera(
                        records[0].obs, "", ("hand_camera",) + WRIST_CAMERA_CANDIDATES, "wrist"
                    ),
                )
                if dataset is None:
                    dataset = _create_dataset(
                        repo_id=str(args.repo_id),
                        image_shape=tuple(label_frames[0]["image"].shape),
                        wrist_image_shape=tuple(label_frames[0]["wrist_image"].shape),
                        fps=10,
                        image_writer_threads=4,
                        image_writer_processes=0,
                    )
                for frame in label_frames:
                    dataset.add_frame(frame)
                dataset.save_episode()
                _append_jsonl(
                    args.output_dir / "training_episodes.jsonl",
                    {
                        "dataset_episode_index": accepted,
                        "raw_attempt_index": attempt_index,
                        "seed": seed,
                        "split": split,
                        "expert_start_step": begin,
                        "expert_action_steps": end - begin,
                    },
                )
                accepted += 1
                accepted_by_split[split] += 1
            print(
                f"[xvla-stackcube-collect] method={args.method} raw={attempt_index + 1} "
                f"split={split} accepted={accepted}/{args.target} by_split={accepted_by_split}",
                flush=True,
            )
    finally:
        if dataset is not None and getattr(dataset, "image_writer", None) is not None:
            dataset.image_writer.wait_until_done()
        for env in envs.values():
            env.close()

    if accepted != args.target:
        raise RuntimeError(f"collected only {accepted}/{args.target} accepted trajectories")
    _write_json(
        args.output_dir / "summary.json",
        {
            "method": args.method,
            "accepted_total": accepted,
            "accepted_by_split": accepted_by_split,
            "raw_by_split": raw_by_split,
            "raw_total": sum(raw_by_split.values()),
            "dataset": str(args.repo_id.resolve()),
        },
    )


if __name__ == "__main__":
    main()
