#!/usr/bin/env python3
"""Collect internal-feature-gated expert suffixes for UncoverSpherePlace.

The policy runs until the selected internal score crosses an ID-only
calibration threshold. The privileged oracle then resumes from the live
simulator state and completes the episode. Raw policy rollouts and accepted
expert suffixes are kept separately so that data selection and intervention
timing remain auditable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RLINF_ROOT = ROOT / "RLinf"
sys.path[:0] = [str(ROOT), str(RLINF_ROOT)]

from rlinf.envs.maniskill.uncover_sphere_place import (  # noqa: E402
    UNCOVER_ENV_IDS,
    register_uncover_sphere_place_variants,
)
from rlinf.envs.maniskill.uncover_sphere_place_privileged_oracle import (  # noqa: E402
    UncoverSpherePlacePrivilegedChunkOracle,
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
from tools.evaluate_uncover_sphere_place_xvla import (  # noqa: E402
    TASK,
    bool_scalar,
    clip_action_chunk,
)
from tools.xvla_airplane_failure_detection import (  # noqa: E402
    XVLAMultilayerProbe,
    XVLAMultilayerScorer,
)
from tools.xvla_airplane_runtime import XVLAAirplanePolicy  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--multilayer-assets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--detector", default="vlm_action_bridge_pca")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--target-id", type=int, default=64)
    parser.add_argument("--target-ood", type=int, default=64)
    parser.add_argument("--seed", type=int, default=70000)
    parser.add_argument("--id-seed", type=int, default=70000)
    parser.add_argument("--ood-seed", type=int, default=71000)
    parser.add_argument("--split", choices=("id", "handle_ood", "goal_ood", "mixed"), default="mixed")
    parser.add_argument("--execute-horizon", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=2500)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--probe-steps", type=int, default=5)
    parser.add_argument("--probe-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _build_env(gym: Any, split: str, args: argparse.Namespace) -> Any:
    return gym.make(
        UNCOVER_ENV_IDS[split],
        robot_uids="panda_wristcam",
        num_envs=1,
        obs_mode="rgb",
        control_mode="pd_joint_delta_pos",
        reward_mode="sparse",
        render_mode="rgb_array",
        sim_backend="physx_cpu",
        sim_config={"sim_freq": 100, "control_freq": 10},
        sensor_configs={"width": 224, "height": 224},
        max_episode_steps=args.max_episode_steps,
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "detach"):
        return _jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _source_split(attempt: int, args: argparse.Namespace) -> tuple[str, str, int]:
    if args.split != "mixed":
        source = "id" if args.split == "id" else "ood"
        seed = (args.id_seed if source == "id" else args.ood_seed) + attempt
        return source, args.split, seed
    if attempt % 2 == 0:
        index = attempt // 2
        return "id", "id", args.id_seed + index
    index = attempt // 2
    ood_split = "handle_ood" if index % 2 == 0 else "goal_ood"
    return "ood", ood_split, args.ood_seed + index


def _resume_phase(env: Any) -> str:
    state = env.unwrapped.evaluate()
    if not bool_scalar(state["ever_mug_parked"]):
        return "cover_reach"
    if not bool_scalar(state["ever_sphere_grasped"]):
        return "sphere_reach"
    return "sphere_lift"


def _step(
    env: Any,
    action: np.ndarray,
    raw_obs: dict[str, Any],
    records: list[Any],
    actions: list[np.ndarray],
) -> tuple[dict[str, Any], bool, bool]:
    raw_obs, _, terminated, truncated, _info = env.step(
        torch.as_tensor(action, device=env.unwrapped.device).unsqueeze(0)
    )
    actions.append(np.asarray(action, dtype=np.float32).copy())
    records.append(_extract_record(raw_obs))
    state = env.unwrapped.evaluate()
    success = bool_scalar(state["success"])
    done = success or bool_scalar(terminated) or bool_scalar(truncated)
    return raw_obs, success, done


def _summary_state(env: Any) -> dict[str, bool]:
    state = env.unwrapped.evaluate()
    names = (
        "success",
        "ever_mug_parked",
        "ever_sphere_grasped",
        "sphere_in_bowl",
        "sphere_released",
        "sphere_static",
    )
    return {name: bool_scalar(state[name]) for name in names}


def _video(
    records: list[Any],
    actions: list[np.ndarray],
    output: Path,
    index: int,
    seed: int,
) -> str:
    if not actions or len(records) != len(actions) + 1:
        raise ValueError(
            f"video alignment requires records=actions+1, got {len(records)} and {len(actions)}"
        )
    main = _select_camera(records[0].obs, "", ("base_camera",) + MAIN_CAMERA_CANDIDATES, "main")
    wrist = _select_camera(records[0].obs, "", ("hand_camera",) + WRIST_CAMERA_CANDIDATES, "wrist")
    frames = _build_frames(
        records=records,
        actions=actions,
        task=TASK,
        main_camera=main,
        wrist_camera=wrist,
    )
    return str(write_episode_video_durably(frames, video_dir=output, episode_index=index, seed=seed, fps=10))


def _mixed_done(counts: dict[str, int], args: argparse.Namespace) -> bool:
    if args.split == "id":
        return counts["id"] >= args.target_id
    if args.split in {"handle_ood", "goal_ood"}:
        return counts["ood"] >= args.target_ood
    return counts["id"] >= args.target_id and counts["ood"] >= args.target_ood


def main() -> None:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    for name in ("raw_videos", "expert_videos", "actions"):
        (args.output_dir / name).mkdir()

    if not np.isfinite(args.threshold):
        raise ValueError("threshold must be finite")
    policy = XVLAAirplanePolicy(args.checkpoint, args.xvla_root, device=args.device)
    probe = XVLAMultilayerProbe(policy.model, probe_seed=args.probe_seed, probe_steps=args.probe_steps)
    scorer = XVLAMultilayerScorer(args.multilayer_assets, device=args.device, knn_k=10)
    if args.detector not in scorer.layers and args.detector not in {
        f"{name}_pca" for name in scorer.layers
    }:
        available = sorted(
            name
            for name in scorer.layers
            for name in (f"{name}_pca", f"{name}_llmd", f"{name}_knn")
        )
        raise ValueError(f"unknown detector {args.detector}; available={available}")

    register_uncover_sphere_place_variants()
    env = None
    dataset = None
    low = high = None
    raw_path = args.output_dir / "raw_attempts.jsonl"
    accepted_path = args.output_dir / "accepted_experts.jsonl"
    counts = {"id": 0, "ood": 0}
    rows: list[dict[str, Any]] = []
    accepted = 0
    attempts = 0
    try:
        while attempts < args.episodes and not _mixed_done(counts, args):
            source, split, seed = _source_split(attempts, args)
            attempts += 1
            env = _build_env(gym, split, args)
            raw_obs, _ = env.reset(seed=seed)
            metadata = _jsonable(env.unwrapped.reset_metadata())
            records = [_extract_record(raw_obs)]
            full_actions: list[np.ndarray] = []
            expert_records: list[Any] | None = None
            expert_actions: list[np.ndarray] = []
            timeline: list[dict[str, Any]] = []
            success = False
            trigger: dict[str, Any] | None = None
            policy_decisions = 0
            expert_active = False
            # Keep the oracle's planning horizon aligned with the validated
            # action-horizon-10 dataset; the deployment loop may still execute
            # only the first five low-level actions before replanning.
            oracle = UncoverSpherePlacePrivilegedChunkOracle(chunk_size=10)
            low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
            high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)

            try:
                while len(full_actions) < args.max_episode_steps and not success:
                    if not expert_active:
                        inputs = policy.prepare(raw_obs, TASK)
                        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                            features, encoding = probe.extract(inputs)
                            torch.manual_seed(seed * 1000 + policy_decisions)
                            generated = policy._generate_from_encoding(
                                inputs, encoding, steps=args.flow_steps
                            )
                        scores = scorer.score(features)
                        selected_score = float(scores[args.detector])
                        point = {
                            "decision_index": policy_decisions,
                            "env_step": len(full_actions),
                            "scores": scores,
                            "selected_detector": args.detector,
                            "selected_score": selected_score,
                            "threshold": float(args.threshold),
                            "expert_active": False,
                        }
                        policy_decisions += 1
                        if selected_score > args.threshold:
                            trigger = {
                                "decision_index": point["decision_index"],
                                "env_step": point["env_step"],
                                "detector": args.detector,
                                "score": selected_score,
                                "threshold": float(args.threshold),
                                "resume_phase": _resume_phase(env),
                            }
                            point["expert_active"] = True
                            timeline.append(point)
                            expert_active = True
                            oracle.resume_from_current_state(trigger["resume_phase"])
                            expert_records = [records[-1]]
                            continue
                        timeline.append(point)
                        chunk = clip_action_chunk(
                            generated.float().cpu().numpy(), low, high, args.execute_horizon
                        )
                        for action in chunk:
                            raw_obs, success, done = _step(
                                env, action, raw_obs, records, full_actions
                            )
                            if done:
                                break
                        continue

                    plan = oracle.plan(env)
                    for step_index in range(args.execute_horizon):
                        qpos = env.unwrapped.agent.robot.get_qpos()
                        action = plan.action_at(qpos, step_index).astype(np.float32)
                        raw_obs, success, done = _step(
                            env, action, raw_obs, records, full_actions
                        )
                        expert_actions.append(action.copy())
                        assert expert_records is not None
                        expert_records.append(records[-1])
                        if done:
                            break
            finally:
                state = _summary_state(env)
                full_video = None
                suffix_video = None
                if full_actions:
                    full_video = _video(
                        records, full_actions, args.output_dir / "raw_videos", attempts - 1, seed
                    )
                if expert_records and expert_actions:
                    suffix_video = _video(
                        expert_records,
                        expert_actions,
                        args.output_dir / "expert_videos",
                        accepted,
                        seed,
                    )
                raw_row = {
                    "attempt_index": attempts - 1,
                    "source": source,
                    "split": split,
                    "seed": seed,
                    "success": bool(state["success"]),
                    "policy_decisions": policy_decisions,
                    "full_actions": len(full_actions),
                    "expert_actions": len(expert_actions),
                    "trigger": trigger,
                    "reset_metadata": metadata,
                    "timeline": timeline,
                    "state": state,
                    "raw_video": full_video,
                    "expert_video": suffix_video,
                }
                with raw_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(raw_row, sort_keys=True) + "\n")
                rows.append(raw_row)

                if state["success"] and expert_actions and trigger is not None:
                    if dataset is None:
                        preview = _build_frames(
                            records=expert_records,
                            actions=expert_actions,
                            task=TASK,
                            main_camera=_select_camera(
                                expert_records[0].obs,
                                "",
                                ("base_camera",) + MAIN_CAMERA_CANDIDATES,
                                "main",
                            ),
                            wrist_camera=_select_camera(
                                expert_records[0].obs,
                                "",
                                ("hand_camera",) + WRIST_CAMERA_CANDIDATES,
                                "wrist",
                            ),
                        )
                        dataset = _create_dataset(
                            repo_id=str(args.output_dir / "lerobot_dataset"),
                            image_shape=tuple(preview[0]["image"].shape),
                            wrist_image_shape=tuple(preview[0]["wrist_image"].shape),
                            fps=10,
                            image_writer_threads=4,
                            image_writer_processes=4,
                        )
                    for frame in _build_frames(
                        records=expert_records,
                        actions=expert_actions,
                        task=TASK,
                        main_camera=_select_camera(
                            expert_records[0].obs,
                            "",
                            ("base_camera",) + MAIN_CAMERA_CANDIDATES,
                            "main",
                        ),
                        wrist_camera=_select_camera(
                            expert_records[0].obs,
                            "",
                            ("hand_camera",) + WRIST_CAMERA_CANDIDATES,
                            "wrist",
                        ),
                    ):
                        dataset.add_frame(frame)
                    dataset.save_episode()
                    accepted += 1
                    counts[source] += 1
                    accepted_row = {
                        "episode_index": accepted - 1,
                        "attempt_index": attempts - 1,
                        "source": source,
                        "split": split,
                        "seed": seed,
                        "expert_actions": len(expert_actions),
                        "trigger": trigger,
                        "raw_video": full_video,
                        "expert_video": suffix_video,
                    }
                    with accepted_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(accepted_row, sort_keys=True) + "\n")
                    np.save(
                        args.output_dir / "actions" / f"episode_{accepted - 1:06d}.npy",
                        np.asarray(expert_actions, dtype=np.float32),
                    )
                    print(
                        f"[uncover-gated] accepted={accepted} id={counts['id']} "
                        f"ood={counts['ood']} split={split} suffix={len(expert_actions)}",
                        flush=True,
                    )
                else:
                    print(
                        f"[uncover-gated] attempt={attempts} source={source} split={split} "
                        f"success={int(state['success'])} trigger={int(trigger is not None)}",
                        flush=True,
                    )
                env.close()
                env = None
    finally:
        if env is not None:
            env.close()
        if dataset is not None and getattr(dataset, "image_writer", None) is not None:
            dataset.image_writer.wait_until_done()
        probe.close()
        del policy
        torch.cuda.empty_cache()

    summary = {
        "format": "uncover_sphere_place_internal_gated_collection_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "detector": args.detector,
        "threshold": float(args.threshold),
        "split": args.split,
        "target_id": args.target_id,
        "target_ood": args.target_ood,
        "accepted": accepted,
        "accepted_id": counts["id"],
        "accepted_ood": counts["ood"],
        "attempts": attempts,
        "dataset": str(args.output_dir / "lerobot_dataset") if dataset is not None else None,
        "raw_attempts": str(raw_path),
        "accepted_manifest": str(accepted_path),
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not _mixed_done(counts, args):
        raise SystemExit(
            f"collection incomplete: id={counts['id']} target={args.target_id}; "
            f"ood={counts['ood']} target={args.target_ood}"
        )
    (args.output_dir / "COLLECTION_COMPLETE").write_text(
        "internal-feature-gated successful expert suffixes collected\n"
    )


if __name__ == "__main__":
    main()
