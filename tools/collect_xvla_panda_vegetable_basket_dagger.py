#!/usr/bin/env python3
"""Collect Panda DAgger branches with real expert continuation suffixes."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import gymnasium as gym
import h5py
import numpy as np
import sapien
import torch
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.collect_xvla_panda_vegetable_basket_planner_oracle import (  # noqa: E402
    _array,
    _bool,
    _move,
    _pose,
    RecordingEnv,
)
from tools.evaluate_xvla_panda_vegetable_basket_failure_detectors import (  # noqa: E402
    ENV_IDS,
    PandaXVLAPolicy,
    pose,
    rgb,
)
from tools.panda_vegetable_basket_adapter import encode_base_ee6d, rotation_from_6d  # noqa: E402
from tools.xvla_airplane_failure_detection import XVLAMultilayerProbe, XVLAMultilayerScorer  # noqa: E402


TASK = "put the vegetable into the yellow basket"
MODEL_ACTION_DIM = 20
ACTIVE_ACTION_DIM = 10
EXECUTE_HORIZON = 5


def load_task_module(path: Path) -> None:
    spec = importlib.util.spec_from_file_location("panda_vegetable_basket_variants", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import task module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def target_to_joint_action(env, model_action: np.ndarray, pinocchio, tcp_index: int) -> np.ndarray:
    """Map one base-frame X-VLA target to the joint-position Panda action."""

    value = np.asarray(model_action, dtype=np.float32).reshape(-1)
    if value.size < ACTIVE_ACTION_DIM:
        raise ValueError(f"model action needs 10 active values, got {value.shape}")
    target_rotation = rotation_from_6d(value[3:9])
    target_pose = sapien.Pose(p=value[:3], q=target_rotation.as_quat()[[3, 0, 1, 2]])
    current = np.asarray(env.unwrapped.agent.robot.get_qpos(), dtype=np.float64).reshape(-1)
    result, success, _error = pinocchio.compute_inverse_kinematics(
        tcp_index,
        target_pose,
        initial_qpos=current[:7],
        active_qmask=np.ones(7, dtype=np.int32),
        max_iterations=100,
        dt=0.1,
        damp=1e-6,
    )
    arm = np.asarray(result if success else current[:7], dtype=np.float32).reshape(-1)[:7]
    arm = np.clip(arm, env.action_space.low[:7], env.action_space.high[:7])
    gripper = float(np.clip(-0.01 + 0.05 * value[9], -0.01, 0.04))
    return np.concatenate([arm, [gripper]]).astype(np.float32)


def _task_metadata(base, split: str, seed: int) -> dict:
    source = base.objs[base.source_obj_name]
    target = base.objs[base.target_obj_name]
    return {
        "seed": int(seed),
        "split": split,
        "robot": type(base.agent).__name__,
        "source_object": base.source_obj_name,
        "target_object": base.target_obj_name,
        "source_model_scale": float(base.episode_model_scales[base.source_obj_name]),
        "target_model_scale": float(base.episode_model_scales[base.target_obj_name]),
        "instruction": TASK,
        "configured_source_pose": _array(base.xyz_configs[0, 0]).tolist(),
        "configured_target_pose": _array(base.xyz_configs[0, 1]).tolist(),
        "start_source_pose": _pose(source).tolist(),
        "start_target_pose": _pose(target).tolist(),
    }


def expert_continuation(recorder: RecordingEnv, *, lift_height: float, release_max_steps: int) -> dict:
    """Run the frozen Panda planner from the current policy state."""

    base = recorder.unwrapped
    source = base.objs[base.source_obj_name]
    target = base.objs[base.target_obj_name]
    source_start = _pose(source)
    source_sp = source.pose.sp
    target_sp = target.pose.sp
    object_matrix = np.asarray(source_sp.to_transformation_matrix(), dtype=np.float64)
    closing = object_matrix[:3, :3] @ np.array([0.0, 1.0, 0.0], dtype=np.float64)
    closing[2] = 0.0
    closing /= max(float(np.linalg.norm(closing)), 1e-8)
    grasp_pose = base.agent.build_grasp_pose(np.array([0.0, 0.0, -1.0]), closing, source_sp.p)
    from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver

    planner = PandaArmMotionPlanningSolver(
        recorder,
        debug=False,
        vis=False,
        base_pose=base.agent.robot.pose.sp,
        visualize_target_grasp_pose=False,
        print_env_info=False,
    )
    stages: dict[str, bool | str] = {}
    currently_grasped = _bool(base.agent.is_grasping(source))
    stable_grasp = currently_grasped
    try:
        if not currently_grasped:
            recorder.current_phase = "expert_open"
            planner.open_gripper(t=4)
            recorder.current_phase = "expert_pregrasp"
            stages["pregrasp_reached"] = _move(planner, grasp_pose * sapien.Pose([0, 0, -0.06]))
            recorder.current_phase = "expert_grasp"
            stages["grasp_reached"] = bool(stages["pregrasp_reached"] and _move(planner, grasp_pose))
            recorder.current_phase = "expert_close"
            planner.close_gripper(t=8)
            stable_grasp = any(_bool(base.agent.is_grasping(source)) for _ in range(4))
        else:
            stages["pregrasp_reached"] = True
            stages["grasp_reached"] = True
        stages["stable_grasp"] = stable_grasp
        if stable_grasp:
            object_in_tcp = base.agent.tcp_pose.sp.inv() * source.pose.sp
            lift_object = sapien.Pose(
                source.pose.sp.p + np.array([0.0, 0.0, lift_height]), source.pose.sp.q
            )
            recorder.current_phase = "expert_lift"
            stages["lift_command_completed"] = _move(planner, lift_object * object_in_tcp.inv())
            stages["lifted"] = bool(
                stages["lift_command_completed"]
                and _bool(base.agent.is_grasping(source))
                and float(source.pose.p[0, 2].cpu()) - float(source_start[2]) >= 0.05
            )
            if stages["lifted"]:
                planner.close_gripper(t=6)
                object_in_tcp = base.agent.tcp_pose.sp.inv() * source.pose.sp
                release_object = sapien.Pose(
                    target_sp.p + np.array([0.0, 0.0, 0.08]), source.pose.sp.q
                )
                current_object = source.pose.sp
                midpoint = sapien.Pose(
                    [
                        (current_object.p[0] + release_object.p[0]) / 2.0,
                        (current_object.p[1] + release_object.p[1]) / 2.0,
                        max(current_object.p[2], release_object.p[2]) + 0.08,
                    ],
                    current_object.q,
                )
                recorder.current_phase = "expert_transport"
                midpoint_ok = _move(planner, midpoint * object_in_tcp.inv())
                stages["transport_midpoint_completed"] = midpoint_ok
                stages["transport_midpoint_grasped"] = _bool(base.agent.is_grasping(source))
                if midpoint_ok and stages["transport_midpoint_grasped"]:
                    object_in_tcp = base.agent.tcp_pose.sp.inv() * source.pose.sp
                high_target = sapien.Pose(
                    [release_object.p[0], release_object.p[1], midpoint.p[2]], source.pose.sp.q
                )
                high_ok = bool(
                    midpoint_ok
                    and stages["transport_midpoint_grasped"]
                    and _move(planner, high_target * object_in_tcp.inv())
                )
                stages["transport_high_target_completed"] = high_ok
                stages["transport_high_target_grasped"] = bool(high_ok and _bool(base.agent.is_grasping(source)))
                if stages["transport_high_target_grasped"]:
                    object_in_tcp = base.agent.tcp_pose.sp.inv() * source.pose.sp
                stages["transport_command_completed"] = bool(
                    high_ok
                    and stages["transport_high_target_grasped"]
                    and _move(planner, release_object * object_in_tcp.inv())
                )
                recorder.current_phase = "expert_release"
                for _ in range(release_max_steps):
                    planner.open_gripper(t=1)
                    if _bool(base.evaluate()["success"]):
                        break
                stages["placed"] = _bool(base.evaluate()["success"])
    except Exception as exc:
        stages["oracle_error"] = repr(exc)
    finally:
        planner.close()
    evaluation = base.evaluate()
    return {
        "accepted": _bool(evaluation["success"]),
        "success": _bool(evaluation["success"]),
        "stages": stages,
        "evaluation": {
            key: bool(_array(value).reshape(-1)[0])
            if _array(value).dtype == bool
            else float(_array(value).reshape(-1)[0])
            for key, value in evaluation.items()
        },
        "max_source_z": max(float(record["object_pose"][2]) for record in recorder.records),
        "release_max_steps": release_max_steps,
        "lift_height": lift_height,
    }


def save_attempt(output: Path, index: int, recorder: RecordingEnv, sources: list[str], metadata: dict) -> dict:
    for name in ("data", "videos", "metadata", "actions", "timelines"):
        (output / name).mkdir(parents=True, exist_ok=True)
    stem = f"episode_{index:06d}"
    records = recorder.records
    actions = recorder.actions
    if len(records) != len(actions) + 1:
        raise RuntimeError("record/action boundary mismatch")
    proprio = np.stack(
        [encode_base_ee6d(record["tcp_base"], record["gripper_01"])[:10] for record in records[:-1]]
    )
    targets = np.stack(
        [encode_base_ee6d(record["tcp_base"], record["gripper_01"])[:10] for record in records[1:]]
    )
    h5_path = output / "data" / f"{stem}.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("images", data=np.asarray([record["rgb"] for record in records], dtype=np.uint8), compression="gzip")
        h5.create_dataset("proprio", data=proprio.astype(np.float32))
        h5.create_dataset("abs_action_6d", data=targets.astype(np.float32))
        h5.create_dataset("object_pose", data=np.asarray([record["object_pose"] for record in records[:-1]], dtype=np.float32))
        h5.create_dataset("target_pose", data=np.asarray([record["target_pose"] for record in records[:-1]], dtype=np.float32))
        h5.create_dataset("source_labels", data=np.asarray([value.encode("ascii") for value in sources], dtype="S6"))
        h5.attrs["seed"] = int(metadata["seed"])
        h5.attrs["success"] = bool(metadata["success"])
        h5.attrs["expert_control_start"] = (
            -1 if metadata["expert_control_start"] is None else int(metadata["expert_control_start"])
        )
        h5.attrs["expert_control_end"] = int(metadata["expert_control_end"])
        h5.attrs["split"] = metadata["split"]
    video_path = output / "videos" / f"{stem}.mp4"
    first = records[0]["rgb"]
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (first.shape[1], first.shape[0]))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    for record in records:
        writer.write(cv2.cvtColor(record["rgb"], cv2.COLOR_RGB2BGR))
    writer.release()
    action_path = output / "actions" / f"{stem}.npy"
    np.save(action_path, np.asarray(actions, dtype=np.float32))
    timeline_path = output / "timelines" / f"{stem}.json"
    timeline_path.write_text(json.dumps(metadata["timeline"], indent=2) + "\n", encoding="utf-8")
    metadata_path = output / "metadata" / f"{stem}.json"
    result = {
        **metadata,
        "data_path": str(h5_path),
        "video_path": str(video_path),
        "action_path": str(action_path),
        "timeline_path": str(timeline_path),
        "metadata_path": str(metadata_path),
        "num_observations": len(records),
        "num_actions": len(actions),
    }
    metadata_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("internal_pca", "diffdagger", "failure_recovery", "offline_bc"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--xvla-root", type=Path)
    parser.add_argument("--rlinf-root", type=Path, required=True)
    parser.add_argument("--task-module", type=Path, required=True)
    parser.add_argument("--multilayer-assets", type=Path)
    parser.add_argument("--gate-score-name", default="vlm_action_bridge_pca")
    parser.add_argument("--gate-threshold", type=float)
    parser.add_argument("--gate-patience", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-successes", type=int, default=100)
    parser.add_argument("--offline-id-target", type=int, default=50)
    parser.add_argument("--offline-ood-target", type=int, default=50)
    parser.add_argument("--id-seed-start", type=int, default=96300)
    parser.add_argument("--ood-seed-start", type=int, default=96400)
    parser.add_argument("--max-attempts", type=int, default=400)
    parser.add_argument("--max-policy-steps", type=int, default=50)
    parser.add_argument("--max-episode-steps", type=int, default=150)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--probe-steps", type=int, default=5)
    parser.add_argument("--diff-timesteps", type=int, default=16)
    parser.add_argument("--diff-noise-samples", type=int, default=1)
    parser.add_argument("--lift-height", type=float, default=0.35)
    parser.add_argument("--release-max-steps", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.method != "offline_bc" and not args.checkpoint:
        raise ValueError("policy-backed methods require --checkpoint")
    if args.method == "internal_pca" and (not args.multilayer_assets or args.gate_threshold is None):
        raise ValueError("internal_pca requires assets and threshold")
    if args.method == "diffdagger" and args.gate_threshold is None:
        raise ValueError("diffdagger requires a frozen threshold")
    if args.gate_patience < 1:
        raise ValueError("gate patience must be positive")
    args.output_dir.mkdir(parents=True)
    sys.path.insert(0, str(args.rlinf_root.resolve()))
    load_task_module(args.task_module)
    device = torch.device("cuda")
    policy = None
    probe = scorer = None
    if args.method != "offline_bc":
        policy = PandaXVLAPolicy(args.checkpoint, args.xvla_root, device, 20)
        if args.method == "internal_pca":
            probe = XVLAMultilayerProbe(policy.model, probe_seed=0, probe_steps=args.probe_steps)
            scorer = XVLAMultilayerScorer(args.multilayer_assets / "multilayer_detector_assets.pt", device="cuda", knn_k=10)
    accepted: list[dict] = []
    rows: list[dict] = []
    accepted_by_split = {"id": 0, "ood": 0}
    for attempt_index in range(args.max_attempts):
        if args.method == "offline_bc":
            if accepted_by_split["id"] >= args.offline_id_target and accepted_by_split["ood"] >= args.offline_ood_target:
                break
        elif len(accepted) >= args.target_successes:
            break
        split = "id" if attempt_index % 2 == 0 else "ood"
        if args.method == "offline_bc" and accepted_by_split[split] >= (args.offline_id_target if split == "id" else args.offline_ood_target):
            split = "ood" if split == "id" else "id"
        seed = (args.id_seed_start if split == "id" else args.ood_seed_start) + attempt_index // 2
        env = gym.make(
            ENV_IDS[split],
            obs_mode="rgb+segmentation",
            render_mode="rgb_array",
            sim_backend="physx_cpu",
            control_mode="pd_joint_pos",
            max_episode_steps=args.max_episode_steps,
        )
        recorder = RecordingEnv(env)
        try:
            raw_obs, _ = recorder.reset(seed=int(seed))
            recorder.current_phase = "policy"
            timeline: list[dict] = []
            expert_start: int | None = 0 if args.method == "offline_bc" else None
            gate_count = 0
            pinocchio = env.unwrapped.agent.robot.create_pinocchio_model()
            link_names = [link.name for link in env.unwrapped.agent.robot.get_links()]
            tcp_index = link_names.index("panda_hand_tcp")
            while expert_start is None and len(recorder.actions) < args.max_policy_steps:
                assert policy is not None
                torch.manual_seed(int(seed * 1000 + len(recorder.actions)))
                inputs = policy.inputs(recorder, raw_obs)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    encoding = policy.model.forward_vlm(inputs["input_ids"], inputs["image_input"], inputs["image_mask"])
                    generated = policy.generate(inputs, encoding, args.flow_steps)
                    score = None
                    if args.method == "internal_pca":
                        features, _ = probe.extract(inputs)
                        score = float(scorer.score(features)[args.gate_score_name])
                    elif args.method == "diffdagger":
                        score = policy.diff_score(inputs, encoding, generated, args.diff_timesteps, args.diff_noise_samples)
                alarm = False
                if args.method in ("internal_pca", "diffdagger"):
                    gate_count = gate_count + 1 if score >= args.gate_threshold else 0
                    alarm = gate_count >= args.gate_patience
                timeline.append({
                    "decision_index": len(timeline),
                    "env_step": len(recorder.actions),
                    "score": score,
                    "threshold": args.gate_threshold,
                    "alarm": alarm,
                    "method": args.method,
                })
                if alarm:
                    expert_start = len(recorder.actions)
                    break
                model_chunk = generated.float().cpu().numpy()[0]
                for model_action in model_chunk[:EXECUTE_HORIZON, :ACTIVE_ACTION_DIM]:
                    command = target_to_joint_action(recorder, model_action, pinocchio, tcp_index)
                    recorder.current_phase = "policy"
                    raw_obs, _reward, terminated, truncated, _info = recorder.step(command)
                    if _bool(terminated) or _bool(truncated) or len(recorder.actions) >= args.max_policy_steps:
                        break
            if expert_start is None and args.method == "failure_recovery":
                expert_start = len(recorder.actions)
            if expert_start is not None:
                expert_start = min(expert_start, len(recorder.actions))
                if args.method == "offline_bc" or expert_start < args.max_episode_steps:
                    oracle = expert_continuation(
                        recorder, lift_height=args.lift_height, release_max_steps=args.release_max_steps
                    )
                    strict_success = bool(oracle["success"])
                else:
                    oracle = {"success": False, "accepted": False, "stages": {}, "evaluation": {}}
                    strict_success = False
            else:
                oracle = {"success": False, "accepted": False, "stages": {}, "evaluation": {}}
                strict_success = False
            expert_end = len(recorder.actions)
            metadata = {
                **_task_metadata(env.unwrapped, split, seed),
                "attempt_index": attempt_index,
                "method": args.method,
                "strict_success": strict_success,
                "success": strict_success,
                "expert_control_start": expert_start,
                "expert_control_end": expert_end,
                "expert_action_steps": 0 if expert_start is None else expert_end - expert_start,
                "timeline": timeline,
                "oracle": oracle,
                "sources": (["policy"] * expert_start + ["expert"] * (expert_end - expert_start)) if expert_start is not None else ["policy"] * expert_end,
                "flow_steps": args.flow_steps,
                "execute_horizon": EXECUTE_HORIZON,
                "max_episode_steps": args.max_episode_steps,
                "admitted": bool(strict_success and expert_start is not None and expert_end > expert_start),
            }
            saved = save_attempt(args.output_dir, attempt_index, recorder, metadata["sources"], metadata)
            rows.append(saved)
            if saved["admitted"]:
                accepted.append(saved)
                accepted_by_split[split] += 1
            if (attempt_index + 1) % 5 == 0 or saved["admitted"]:
                print(json.dumps({"attempt": attempt_index + 1, "split": split, "accepted": len(accepted), "accepted_by_split": accepted_by_split, "strict_success": strict_success, "expert_start": expert_start}), flush=True)
        finally:
            env.close()
    if probe is not None:
        probe.close()
    if args.method == "offline_bc":
        target_met = accepted_by_split["id"] == args.offline_id_target and accepted_by_split["ood"] == args.offline_ood_target
    else:
        target_met = len(accepted) >= args.target_successes
    if not target_met:
        raise RuntimeError(f"collection target not met: accepted={len(accepted)} by_split={accepted_by_split}")
    (args.output_dir / "accepted_episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in accepted), encoding="utf-8"
    )
    summary = {
        "format": "xvla_panda_vegetable_basket_dagger_collection_v1",
        "method": args.method,
        "episodes": len(rows),
        "accepted": len(accepted),
        "accepted_by_split": accepted_by_split,
        "raw_by_split": {split: sum(row["split"] == split for row in rows) for split in ("id", "ood")},
        "target_met": target_met,
        "target_successes": args.target_successes,
        "offline_targets": {"id": args.offline_id_target, "ood": args.offline_ood_target},
        "protocol": {
            "strict_raw_id_ood_alternation": True,
            "admission": "strict_success_and_nonempty_expert_suffix",
            "expert_suffix": "all real observations/actions from expert_control_start; temporal mask at training",
            "max_policy_steps": args.max_policy_steps,
            "max_episode_steps": args.max_episode_steps,
            "flow_steps": args.flow_steps,
            "execute_horizon": EXECUTE_HORIZON,
        },
        "rows": rows,
    }
    (args.output_dir / "collection_provenance.json").write_text(json.dumps({
        "format": "xvla_panda_vegetable_basket_collection_provenance_v1",
        "method": args.method,
        "checkpoint": None if args.checkpoint is None else str(args.checkpoint.resolve()),
        "multilayer_assets": None if args.multilayer_assets is None else str(args.multilayer_assets.resolve()),
        "gate_score_name": args.gate_score_name,
        "gate_threshold": args.gate_threshold,
        "gate_patience": args.gate_patience,
        "raw_split_schedule": "strict ID/OOD alternation",
        "admission": "strict_success_and_nonempty_expert_suffix",
    }, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "COLLECTION_COMPLETE").write_text("complete\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
