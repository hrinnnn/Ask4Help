#!/usr/bin/env python3
"""Evaluate a trained X-VLA policy on controlled StackPyramid splits."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

# Set these before importing Torch or the simulator.  On the H20 runtime,
# their unrestricted CPU pools can raise SIGFPE during image preprocessing.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import imageio
import numpy as np
import torch
from PIL import Image


REAL_ACTION_DIM = 8
MODEL_ACTION_DIM = 20
ACTION_HORIZON = 10
TASK = "stack the red cube next to the green cube and place the blue cube on top"


def scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.size == 1:
        return array.reshape(-1)[0].item()
    return value


def bool_scalar(value: Any) -> bool:
    return bool(scalar(value))


def json_state(value: Any) -> list[float]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32).reshape(-1).tolist()


def validate_formal_reset(metadata: dict[str, Any]) -> dict[str, Any]:
    if metadata.get("ood_geometry") != "v4":
        raise RuntimeError(f"formal reset geometry mismatch: {metadata.get('ood_geometry')!r}")
    invariants = metadata.get("reset_invariants", {})
    if not metadata.get("reset_invariant_pass", False) or invariants.get("red_placed", False):
        raise RuntimeError(f"formal reset invariant failed: {invariants}")
    red = np.asarray(metadata["cube_poses"]["red"]["p"], dtype=np.float64)[:3]
    green = np.asarray(metadata["cube_poses"]["green"]["p"], dtype=np.float64)[:3]
    distance = float(np.linalg.norm((red - green)[:2]))
    if not 0.14 <= distance <= 0.18:
        raise RuntimeError(f"formal v4 red-green distance outside jitter range: {distance}")
    return {"geometry": "v4", "red_green_xy_distance": distance, "reset_invariants": invariants}


def image_array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3 or array.shape[-1] < 3:
        raise ValueError(f"expected HWC RGB image, got {array.shape}")
    return np.ascontiguousarray(array[..., :3].astype(np.uint8))


def frame_array(value: Any) -> np.ndarray | None:
    if isinstance(value, dict):
        for key in ("base_camera", "hand_camera", "rgb"):
            if key in value:
                frame = frame_array(value[key])
                if frame is not None:
                    return frame
        return None
    try:
        return image_array(value)
    except ValueError:
        return None


def make_policy(checkpoint: Path, xvla_root: Path, device: torch.device) -> Any:
    sys.path.insert(0, str(xvla_root))
    from models.configuration_xvla import XVLAConfig
    from models.modeling_xvla import XVLA
    from models.processing_xvla import XVLAProcessor

    # Keep inference's action contract identical to the training and reload
    # smoke paths: the policy predicts a 20-D latent chunk while only the
    # first 8 dimensions correspond to the environment action.
    config = XVLAConfig.from_pretrained(str(checkpoint))
    config.action_mode = "auto"
    config.real_action_dim = REAL_ACTION_DIM
    config.max_action_dim = MODEL_ACTION_DIM
    config.num_actions = ACTION_HORIZON
    # The standalone evaluator does not use Accelerate's mixed-precision
    # parameter management.  Loading the complete policy in FP32 avoids a
    # mixed-dtype matmul in the action expert's domain-conditioned layers.
    model = XVLA.from_pretrained(
        str(checkpoint), config=config, torch_dtype=torch.float32
    ).to(device).eval()
    processor = XVLAProcessor.from_pretrained(checkpoint)
    return model, processor


def prepare_inputs(model: Any, processor: Any, raw_obs: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    sensors = raw_obs["sensor_data"]
    images = [[
        Image.fromarray(image_array(sensors["base_camera"]["rgb"])),
        Image.fromarray(image_array(sensors["hand_camera"]["rgb"])),
    ]]
    inputs = {
        **processor.encode_image(images),
        **processor.encode_language([TASK]),
    }
    # StackPyramid exposes the Panda state as a flat 64-D observation rather
    # than the nested agent/qpos structure used by PickSingleYCB.
    qpos = raw_obs["state"]
    if isinstance(qpos, torch.Tensor):
        qpos = qpos.detach().cpu().numpy()
    qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
    proprio = np.zeros((1, MODEL_ACTION_DIM), dtype=np.float32)
    proprio[0, : min(REAL_ACTION_DIM, qpos.size)] = qpos[:REAL_ACTION_DIM]
    inputs.update({
        "domain_id": torch.zeros(1, dtype=torch.long),
        "proprio": torch.from_numpy(proprio),
    })
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }


@torch.inference_mode()
def predict(model: Any, inputs: dict[str, torch.Tensor], device: torch.device, seed: int, steps: int) -> np.ndarray:
    torch.manual_seed(seed)
    with torch.autocast(device_type=device.type, enabled=False):
        encoding = model.forward_vlm(inputs["input_ids"], inputs["image_input"], inputs["image_mask"])
        prior = torch.randn(
            1, ACTION_HORIZON, model.action_space.dim_action,
            device=device, dtype=inputs["proprio"].dtype,
        )
        action = torch.zeros_like(prior)
        for index in range(max(1, steps), 0, -1):
            time = torch.full((1,), index / float(max(1, steps)), device=device, dtype=prior.dtype)
            noisy = prior * time[:, None, None] + action * (1 - time[:, None, None])
            proprio, noisy = model.action_space.preprocess(inputs["proprio"], noisy)
            action = model.transformer(
                domain_id=inputs["domain_id"],
                action_with_noise=noisy,
                proprio=proprio,
                t=time,
                **encoding,
            )
        action = model.action_space.postprocess(action)
    return action.float().cpu().numpy()[0]


def details(env: Any) -> dict[str, Any]:
    base = env.unwrapped
    cubes = (base.cubeA, base.cubeB, base.cubeC)
    positions = [cube.pose.p.detach().cpu().numpy().reshape(-1, 3)[0] for cube in cubes]
    threshold = float(np.linalg.norm(2 * base.cube_half_size[:2].detach().cpu().numpy()) + 0.005)
    xy_ab = float(np.linalg.norm((positions[0] - positions[1])[:2])) <= threshold
    xy_cb = float(np.linalg.norm((positions[2] - positions[1])[:2])) <= threshold
    xy_ca = float(np.linalg.norm((positions[2] - positions[0])[:2])) <= threshold
    z_cb = abs(float(positions[2][2] - positions[1][2])) > 0.02
    z_ca = abs(float(positions[2][2] - positions[0][2])) > 0.02
    grasped = [bool_scalar(base.agent.is_grasping(cube)) for cube in cubes]
    evaluation = base.evaluate()
    return {
        "positions": [position.tolist() for position in positions],
        "xy_ab": xy_ab,
        "xy_cb": xy_cb,
        "xy_ca": xy_ca,
        "z_cb": z_cb,
        "z_ca": z_ca,
        "grasped": grasped,
        "success": bool_scalar(evaluation["success"] if isinstance(evaluation, dict) else evaluation),
    }


def stage_events(
    env: Any,
    initial_z: dict[str, float],
    completed_events: dict[str, bool] | None = None,
) -> dict[str, bool]:
    """Return stage events only after their preceding physical events."""
    base = env.unwrapped
    completed_events = completed_events or {}
    red = base.cubeA.pose.p.detach().cpu().numpy().reshape(-1, 3)[0]
    blue = base.cubeC.pose.p.detach().cpu().numpy().reshape(-1, 3)[0]
    green = base.cubeB.pose.p.detach().cpu().numpy().reshape(-1, 3)[0]
    threshold = float(np.linalg.norm(2 * base.cube_half_size[:2].detach().cpu().numpy()) + 0.005)
    red_contact = bool_scalar(base.agent.is_grasping(base.cubeA))
    blue_contact = bool_scalar(base.agent.is_grasping(base.cubeC))
    tcp = base.agent.tcp.pose.p.detach().cpu().numpy().reshape(-1, 3)[0]
    gripper_closed = bool(getattr(env, "gripper_closed", False))
    red_grasped = red_contact or (gripper_closed and float(np.linalg.norm(red - tcp)) <= 0.05)
    red_lifted = completed_events.get("red_grasped", False) and float(red[2]) > initial_z["red"] + 0.015
    blue_lifted = float(blue[2]) > initial_z["blue"] + 0.015
    red_placed = (
        completed_events.get("red_lifted", False)
        and
        float(np.linalg.norm((red - green)[:2])) <= threshold
        and not red_grasped
        and float(red[2]) <= initial_z["red"] + 0.03
    )
    blue_grasped = completed_events.get("red_placed", False) and (
        blue_contact or (gripper_closed and float(np.linalg.norm(blue - tcp)) <= 0.05)
    )
    return {
        "red_grasped": red_grasped,
        "red_lifted": red_lifted,
        "red_placed": red_placed,
        "blue_grasped": blue_grasped,
        "blue_lifted": completed_events.get("blue_grasped", False) and blue_lifted,
    }


STAGE_LOCALITY_CONTRACTS = {
    "stage1_ood": {"prefix": "red_grasped", "target": "red_lifted"},
    "stage2_ood": {"prefix": "red_lifted", "target": "red_placed"},
    "stage3_ood": {"prefix": "red_placed", "target": "blue_lifted"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("id", "stage1_ood", "stage2_ood", "stage3_ood"), required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--start-seed", type=int, required=True)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--execute-horizon", type=int, default=5)
    parser.add_argument("--flow-steps", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sim-backend", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--render-backend", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument(
        "--formal-evidence",
        action="store_true",
        help="Save per-episode actions and full state/event timelines for a formal gate.",
    )
    parser.add_argument("--geometry", choices=("v1", "v2", "v3", "v4"))
    parser.add_argument("--fresh-env-per-episode", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.formal_evidence and args.geometry != "v4":
        raise ValueError("formal evidence requires explicit --geometry v4")
    if args.geometry is not None:
        os.environ["STACKPYRAMID_OOD_GEOMETRY"] = args.geometry
    # The H20 runtime's CPU image preprocessing path can raise SIGFPE when
    # unrestricted Torch/OMP thread pools are combined with the simulator.
    # Evaluation is low-throughput by design, so keep the runtime deterministic
    # and bounded without changing the policy or simulator semantics.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    videos = args.output / "videos"
    videos.mkdir()
    (args.output / "config.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    from tools.stackpyramid_task import register_stackpyramid_splits, reset_metadata, stackpyramid_env_id, stackpyramid_geometry_version

    register_stackpyramid_splits()
    device = torch.device(args.device)
    model, processor = make_policy(args.checkpoint, args.xvla_root, device)
    def make_env() -> Any:
        return gym.make(
            stackpyramid_env_id(args.split),
            obs_mode="rgb+state",
            control_mode="pd_joint_pos",
            render_mode="rgb_array",
            sim_backend=args.sim_backend,
            render_backend=args.render_backend,
        )

    env = make_env()
    rows: list[dict[str, Any]] = []
    try:
        for episode_index in range(args.episodes):
            if args.fresh_env_per_episode and episode_index > 0:
                env.close()
                env = make_env()
            seed = args.start_seed + episode_index
            raw_obs, _ = env.reset(seed=seed)
            reset_meta = reset_metadata(env, split=args.split)
            formal_reset = validate_formal_reset(reset_meta) if args.formal_evidence else None
            initial_positions = env.unwrapped.cubeA.pose.p.detach().cpu().numpy().reshape(-1, 3)[0]
            initial_blue = env.unwrapped.cubeC.pose.p.detach().cpu().numpy().reshape(-1, 3)[0]
            initial_z = {"red": float(initial_positions[2]), "blue": float(initial_blue[2])}
            frames: list[np.ndarray] = []
            first = frame_array(env.render())
            if first is not None:
                frames.append(first)
            executed = 0
            ever_grasped = False
            ever_base = False
            ever_pyramid = False
            event_reached = {name: False for name in ("red_grasped", "red_lifted", "red_placed", "blue_grasped", "blue_lifted")}
            event_steps: dict[str, int | None] = {name: None for name in event_reached}
            final_info: dict[str, Any] = {}
            formal_actions: list[np.ndarray] = []
            formal_state_timeline: list[dict[str, Any]] = []
            if args.formal_evidence:
                formal_state_timeline.append({
                    "step": 0,
                    "state": json_state(raw_obs["state"]),
                    "details": details(env),
                    "stage_events": dict(event_reached),
                })
            while executed < args.max_episode_steps and not ever_pyramid:
                inputs = prepare_inputs(model, processor, raw_obs, device)
                chunk = predict(model, inputs, device, seed + executed, args.flow_steps)
                chunk = np.clip(
                    chunk[:ACTION_HORIZON],
                    np.asarray(env.action_space.low, dtype=np.float32),
                    np.asarray(env.action_space.high, dtype=np.float32),
                )
                for action in chunk[: args.execute_horizon]:
                    raw_obs, _, terminated, truncated, info = env.step(action.astype(np.float32))
                    executed += 1
                    if args.formal_evidence:
                        formal_actions.append(np.asarray(action, dtype=np.float32).copy())
                    final_info = info if isinstance(info, dict) else {}
                    current = details(env)
                    events = stage_events(env, initial_z, event_reached)
                    for name, reached in events.items():
                        if reached and not event_reached[name]:
                            event_reached[name] = True
                            event_steps[name] = executed
                    if args.formal_evidence:
                        formal_state_timeline.append({
                            "step": executed,
                            "state": json_state(raw_obs["state"]),
                            "details": current,
                            "stage_events": dict(event_reached),
                        })
                    ever_grasped |= any(current["grasped"])
                    ever_base |= bool(current["xy_ab"] and (current["z_cb"] or current["z_ca"]))
                    ever_pyramid |= bool(current["success"])
                    frame = frame_array(env.render())
                    if frame is not None:
                        frames.append(frame)
                    if bool_scalar(terminated) or bool_scalar(truncated) or executed >= args.max_episode_steps:
                        break
                if bool_scalar(terminated) or bool_scalar(truncated):
                    break
            final = details(env)
            events = stage_events(env, initial_z, event_reached)
            for name, reached in events.items():
                if reached and not event_reached[name]:
                    event_reached[name] = True
                    event_steps[name] = executed
            video_path = videos / f"{args.split}_{seed}.mp4"
            if frames:
                with imageio.get_writer(video_path, fps=10, codec="libx264", macro_block_size=None) as writer:
                    for frame in frames:
                        writer.append_data(frame)
            row = {
                "episode_index": episode_index,
                "seed": seed,
                "split": args.split,
                "steps": executed,
                "ever_grasped": bool(ever_grasped),
                "ever_base_completed": bool(ever_base),
                "strict_success": bool(ever_pyramid),
                "stage_events": event_reached,
                "stage_event_steps": event_steps,
                "final": final,
                "video": str(video_path),
            }
            if args.formal_evidence:
                row["reset_metadata"] = reset_meta
                row["formal_reset"] = formal_reset
                actions_path = args.output / "actions" / f"{args.split}_{seed}.npy"
                states_path = args.output / "states" / f"{args.split}_{seed}.json"
                actions_path.parent.mkdir(parents=True, exist_ok=True)
                states_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(actions_path, np.asarray(formal_actions, dtype=np.float32))
                states_path.write_text(json.dumps(formal_state_timeline) + "\n", encoding="utf-8")
                row["actions"] = str(actions_path)
                row["state_timeline"] = str(states_path)
            rows.append(row)
            print(json.dumps(row, ensure_ascii=True), flush=True)
    finally:
        env.close()

    summary = {
        "format": "stackpyramid_xvla_policy_eval_v1",
        "split": args.split,
        "episodes": len(rows),
        "ever_grasped": sum(int(row["ever_grasped"]) for row in rows),
        "ever_base_completed": sum(int(row["ever_base_completed"]) for row in rows),
        "strict_success": sum(int(row["strict_success"]) for row in rows),
        "stage_event_counts": {
            name: sum(int(row["stage_events"][name]) for row in rows)
            for name in ("red_grasped", "red_lifted", "red_placed", "blue_grasped", "blue_lifted")
        },
        "video_count": len(list(videos.glob("*.mp4"))),
        "formal_evidence": bool(args.formal_evidence),
        "geometry": stackpyramid_geometry_version(),
        "env_id": stackpyramid_env_id(args.split),
        "action_array_count": len(list((args.output / "actions").glob("*.npy"))) if args.formal_evidence else None,
        "state_timeline_count": len(list((args.output / "states").glob("*.json"))) if args.formal_evidence else None,
        "rows": rows,
    }
    if args.split in STAGE_LOCALITY_CONTRACTS:
        contract = STAGE_LOCALITY_CONTRACTS[args.split]
        summary["stage_locality_contract"] = contract
        summary["prefix_completion"] = summary["stage_event_counts"][contract["prefix"]]
        summary["target_stage_reached"] = summary["stage_event_counts"][contract["target"]]
        summary["prefix_completion_rate"] = summary["prefix_completion"] / float(len(rows))
        summary["target_stage_reached_rate"] = summary["target_stage_reached"] / float(len(rows))
    (args.output / "episodes.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8"
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n")
    if summary["episodes"] != args.episodes or summary["video_count"] != args.episodes:
        raise RuntimeError(f"incomplete evaluation artifacts: {summary}")
    if args.formal_evidence:
        evidence_errors = []
        for row in rows:
            action_path = Path(row["actions"])
            state_path = Path(row["state_timeline"])
            try:
                action_array = np.load(action_path)
                state_rows = json.loads(state_path.read_text(encoding="utf-8"))
                if action_array.shape[0] != row["steps"]:
                    evidence_errors.append(f"{row['seed']}: actions {action_array.shape} vs steps {row['steps']}")
                if len(state_rows) != row["steps"] + 1:
                    evidence_errors.append(f"{row['seed']}: states {len(state_rows)} vs steps+1 {row['steps'] + 1}")
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                evidence_errors.append(f"{row['seed']}: {exc}")
        summary["formal_evidence_errors"] = evidence_errors
        if (
            summary["action_array_count"] != args.episodes
            or summary["state_timeline_count"] != args.episodes
            or evidence_errors
        ):
            raise RuntimeError(f"formal evidence incomplete: {summary}")
        (args.output / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    (args.output / "EVAL_COMPLETE").write_text("complete\n")


if __name__ == "__main__":
    main()
