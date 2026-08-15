#!/usr/bin/env python3
"""Collect stage-localized StackPyramid gated-DAgger expert suffixes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import imageio
import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]

from tools.evaluate_stackpyramid_xvla import (  # noqa: E402
    ACTION_HORIZON,
    MODEL_ACTION_DIM,
    REAL_ACTION_DIM,
    frame_array,
    image_array,
    stage_events,
)
from tools.stackpyramid_task import (  # noqa: E402
    register_stackpyramid_splits,
    stackpyramid_env_id,
    stackpyramid_geometry_version,
    stackpyramid_reset_invariants,
)


TASK = "stack the red cube next to the green cube and place the blue cube on top"
EXECUTE_HORIZON = 5
MAX_EPISODE_STEPS = 250


def _scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    return array.reshape(-1)[0].item() if array.size == 1 else value


def _bool(value: Any) -> bool:
    return bool(_scalar(value))


def _record(raw_obs: dict[str, Any]) -> dict[str, np.ndarray]:
    sensors = raw_obs["sensor_data"]
    state = raw_obs["state"]
    if isinstance(state, torch.Tensor):
        state = state.detach().cpu().numpy()
    return {
        "base": image_array(sensors["base_camera"]["rgb"]),
        "wrist": image_array(sensors["hand_camera"]["rgb"]),
        "state": np.asarray(state, dtype=np.float32).reshape(-1).copy(),
    }


def _copy_record(record: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: value.copy() for key, value in record.items()}


class StepRecorder:
    """Proxy used so motion-planning expert actions enter the same timeline."""

    def __init__(self, env: Any):
        self.env = env
        self.records: list[dict[str, np.ndarray]] = []
        self.actions: list[np.ndarray] = []
        self.sources: list[str] = []
        self.frames: list[np.ndarray] = []
        self.current_source = "policy"
        self.gripper_closed = False
        self.initial_z: dict[str, float] = {}
        self.reset_invariants: dict[str, bool] = {}
        self.event_first_steps: dict[str, int] = {}
        self.event_history: list[dict[str, bool]] = []

    @property
    def unwrapped(self) -> Any:
        return self.env.unwrapped

    def reset(self, **kwargs: Any):
        raw_obs, info = self.env.reset(**kwargs)
        self.records = [_record(raw_obs)]
        self.actions = []
        self.sources = []
        self.frames = []
        self.gripper_closed = False
        base = self.env.unwrapped
        self.initial_z = {
            "red": float(base.cubeA.pose.p.detach().cpu().numpy().reshape(-1, 3)[0][2]),
            "blue": float(base.cubeC.pose.p.detach().cpu().numpy().reshape(-1, 3)[0][2]),
        }
        self.reset_invariants = stackpyramid_reset_invariants(self)
        if stackpyramid_geometry_version() == "v4" and any(self.reset_invariants.values()):
            raise RuntimeError(f"StackPyramid v4 reset invariant failed: {self.reset_invariants}")
        self.event_first_steps = {}
        self.event_history = []
        self._capture()
        self._update_events()
        return raw_obs, info

    def _capture(self) -> None:
        frame = frame_array(self.env.render())
        if frame is not None:
            self.frames.append(frame)

    def step(self, action: Any):
        raw_obs, reward, terminated, truncated, info = self.env.step(action)
        value = action.detach().cpu().numpy() if isinstance(action, torch.Tensor) else np.asarray(action)
        value = np.asarray(value, dtype=np.float32).reshape(-1).copy()
        self.actions.append(value)
        if value.size:
            self.gripper_closed = bool(value[-1] < 0.0)
        self.sources.append(self.current_source)
        self.records.append(_record(raw_obs))
        self._capture()
        self._update_events()
        return raw_obs, reward, terminated, truncated, info

    def _update_events(self) -> None:
        events = stage_events(self, self.initial_z)
        step = len(self.actions)
        self.event_history.append(events)
        for name, reached in events.items():
            if reached and name not in self.event_first_steps:
                self.event_first_steps[name] = step

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)


def _install_rrt_fallback() -> None:
    from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver

    original = PandaArmMotionPlanningSolver.move_to_pose_with_screw
    if getattr(original, "_stackpyramid_rrt_fallback", False):
        return

    def move_with_fallback(self, pose, dry_run=False, refine_steps=0):
        result = original(self, pose, dry_run=dry_run, refine_steps=refine_steps)
        if result != -1:
            return result
        return self.move_to_pose_with_RRTConnect(
            pose, dry_run=dry_run, refine_steps=refine_steps
        )

    move_with_fallback._stackpyramid_rrt_fallback = True
    PandaArmMotionPlanningSolver.move_to_pose_with_screw = move_with_fallback


class StackPyramidOracle:
    """Official Panda motion-planning recipe, continuing from the current state."""

    def __init__(self, recorder: StepRecorder):
        import sapien
        from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver

        self.recorder = recorder
        self.sapien = sapien
        self.planner = PandaArmMotionPlanningSolver(
            recorder,
            debug=False,
            vis=False,
            base_pose=recorder.unwrapped.agent.robot.pose,
            visualize_target_grasp_pose=False,
            print_env_info=False,
        )

    def run(self) -> None:
        from transforms3d.euler import euler2quat
        from mani_skill.examples.motionplanning.panda.solutions.stack_pyramid import (
            compute_grasp_info_by_obb,
            get_actor_obb,
        )

        base = self.recorder.unwrapped
        finger_length = 0.025
        approaching = np.array([0, 0, -1])
        moving_cube, target_cube = base.cubeA, base.cubeB
        self.recorder.current_source = "expert"

        obb = get_actor_obb(moving_cube)
        target_closing = base.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
        grasp_info = compute_grasp_info_by_obb(
            obb, approaching=approaching, target_closing=target_closing, depth=finger_length
        )
        closing, center = grasp_info["closing"], grasp_info["center"]
        moving_position = moving_cube.pose.p.detach().cpu().numpy().reshape(-1, 3)[0]
        target_position = target_cube.pose.p.detach().cpu().numpy().reshape(-1, 3)[0]
        distance = float(np.linalg.norm(moving_position - target_position))
        need_move_a_b = distance > 0.07
        if need_move_a_b:
            # Keep the official ManiSkill order: approach open, close at grasp,
            # then transport. The recorder only captures the same env.step calls.
            grasp_pose = base.agent.build_grasp_pose(approaching, closing, moving_cube.pose.sp.p)
            reach_pose = grasp_pose * self.sapien.Pose([0, 0, -0.05])
            self.planner.open_gripper()
            self.planner.move_to_pose_with_screw(reach_pose)
            self.planner.move_to_pose_with_screw(grasp_pose)
            self.planner.close_gripper()
            lift_pose = self.sapien.Pose([0, 0, 0.1]) * grasp_pose
            self.planner.move_to_pose_with_screw(lift_pose)
            target_position = target_cube.pose.p.detach().cpu().numpy().reshape(-1, 3)[0].copy()
            red_position = moving_cube.pose.p.detach().cpu().numpy().reshape(-1, 3)[0]
            direction = red_position[:2] - target_position[:2]
            direction /= max(float(np.linalg.norm(direction)), 1e-6)
            target_position[:2] += 0.040 * direction
            # Release above the tabletop so the open fingers do not sweep
            # through the green cube while descending to the contact plane.
            target_position[2] += 0.060
            goal_pose = self.sapien.Pose(target_position, lift_pose.q)
            self.planner.move_to_pose_with_screw(goal_pose)
            self.planner.open_gripper()
            self.planner.move_to_pose_with_screw(goal_pose * self.sapien.Pose([0, 0, 0.08]))

        moving_cube = base.cubeC
        obb = get_actor_obb(moving_cube)
        target_closing = base.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
        grasp_info = compute_grasp_info_by_obb(
            obb, approaching=approaching, target_closing=target_closing, depth=finger_length
        )
        closing, center = grasp_info["closing"], grasp_info["center"]
        # ``compute_grasp_info_by_obb`` returns a grasp center in the OBB
        # frame; use the actor pose for the world-space target expected by
        # Panda's grasp-pose helper.
        grasp_pose = base.agent.build_grasp_pose(
            approaching, closing, moving_cube.pose.sp.p
        )
        angles = np.arange(0, np.pi * 2 / 3, np.pi / 2)
        angles = np.repeat(angles, 2)
        angles[1::2] *= -1
        for angle in angles:
            grasp_pose2 = grasp_pose * self.sapien.Pose(q=euler2quat(0, 0, angle))
            if self.planner.move_to_pose_with_screw(grasp_pose2, dry_run=True) != -1:
                grasp_pose = grasp_pose2
                break
        blue_position = moving_cube.pose.p.detach().cpu().numpy().reshape(-1, 3)[0]
        safe_pose = self.sapien.Pose(
            [blue_position[0], blue_position[1], 0.20], grasp_pose.q
        )
        self.planner.move_to_pose_with_screw(safe_pose)
        reach_pose = grasp_pose * self.sapien.Pose([0, 0, -0.05])
        self.planner.move_to_pose_with_screw(reach_pose)
        if need_move_a_b:
            self.planner.open_gripper()
        self.planner.move_to_pose_with_screw(grasp_pose)
        self.planner.close_gripper()
        lift_pose = self.sapien.Pose([0, 0, 0.1]) * grasp_pose
        self.planner.move_to_pose_with_screw(lift_pose)
        goal_pose_a = base.cubeA.pose * self.sapien.Pose([0, 0, base.cube_half_size[2] * 2])
        goal_pose_b = base.cubeB.pose * self.sapien.Pose([0, 0, base.cube_half_size[2] * 2])
        goal_p = (goal_pose_a.p + goal_pose_b.p) / 2
        offset = (goal_p - base.cubeC.pose.p).cpu().numpy()[0]
        self.planner.move_to_pose_with_screw(self.sapien.Pose(lift_pose.p + offset, lift_pose.q))
        self.planner.open_gripper()
        self.planner.close()


def _load_asset(path: Path, device: torch.device):
    from rlinf.algorithms.vla_fail import PCAResidualStatistics

    payload = torch.load(path, map_location="cpu", weights_only=False)
    return PCAResidualStatistics.from_state_dict(payload["statistics"])


def _prepare(model: Any, processor: Any, raw_obs: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    sensors = raw_obs["sensor_data"]
    images = [[
        Image.fromarray(image_array(sensors["base_camera"]["rgb"])),
        Image.fromarray(image_array(sensors["hand_camera"]["rgb"])),
    ]]
    inputs = {**processor.encode_image(images), **processor.encode_language([TASK])}
    qpos = raw_obs["state"]
    if isinstance(qpos, torch.Tensor):
        qpos = qpos.detach().cpu().numpy()
    proprio = np.zeros((1, MODEL_ACTION_DIM), dtype=np.float32)
    proprio[0, : min(REAL_ACTION_DIM, np.asarray(qpos).size)] = np.asarray(qpos).reshape(-1)[:REAL_ACTION_DIM]
    inputs.update({"domain_id": torch.zeros(1, dtype=torch.long), "proprio": torch.from_numpy(proprio)})
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}


@torch.inference_mode()
def _predict(model: Any, processor: Any, raw_obs: dict[str, Any], device: torch.device, seed: int, steps: int):
    torch.manual_seed(seed)
    inputs = _prepare(model, processor, raw_obs, device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        encoding = model.forward_vlm(inputs["input_ids"], inputs["image_input"], inputs["image_mask"])
        prior = torch.randn(1, ACTION_HORIZON, model.action_space.dim_action, device=device, dtype=inputs["proprio"].dtype)
        action = torch.zeros_like(prior)
        for index in range(max(1, steps), 0, -1):
            time = torch.full((1,), index / float(max(1, steps)), device=device, dtype=prior.dtype)
            noisy = prior * time[:, None, None] + action * (1 - time[:, None, None])
            proprio, noisy = model.action_space.preprocess(inputs["proprio"], noisy)
            action = model.transformer(domain_id=inputs["domain_id"], action_with_noise=noisy, proprio=proprio, t=time, **encoding)
        action = model.action_space.postprocess(action)
    bridge = torch.cat([encoding["vlm_features"], encoding["aux_visual_inputs"]], dim=1).float().mean(dim=1)
    return action.float().cpu().numpy()[0], bridge, inputs, encoding


@torch.inference_mode()
def _diff_score(model: Any, inputs: dict[str, torch.Tensor], encoding: dict[str, torch.Tensor], generated: np.ndarray, timesteps: int) -> float:
    action = torch.as_tensor(generated, device=inputs["proprio"].device, dtype=torch.bfloat16).unsqueeze(0)
    action = model.action_space._pad_to_model_dim(action)
    score = torch.zeros(1, device=action.device, dtype=torch.float32)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for value in (torch.arange(timesteps, device=action.device) + 0.5) / timesteps:
            noise = torch.randn_like(action)
            time = value.to(action.dtype).expand(1)
            noisy = noise * time[:, None, None] + action * (1 - time[:, None, None])
            proprio, noisy = model.action_space.preprocess(inputs["proprio"], noisy)
            prediction = model.transformer(domain_id=inputs["domain_id"], action_with_noise=noisy, proprio=proprio, t=time, **encoding)
            score += (prediction[..., :REAL_ACTION_DIM].float() - action[..., :REAL_ACTION_DIM].float()).square().flatten(1).mean(1)
    return float((score / timesteps)[0].cpu())


def _summary(env: Any) -> dict[str, Any]:
    base = env.unwrapped
    cubes = (base.cubeA, base.cubeB, base.cubeC)
    positions = [cube.pose.p.detach().cpu().numpy().reshape(-1, 3)[0] for cube in cubes]
    threshold = float(np.linalg.norm(2 * base.cube_half_size[:2].detach().cpu().numpy()) + 0.005)
    evaluation = base.evaluate()
    success = _bool(evaluation["success"] if isinstance(evaluation, dict) else evaluation)
    return {
        "positions": [position.tolist() for position in positions],
        "xy_ab": float(np.linalg.norm((positions[0] - positions[1])[:2])) <= threshold,
        "success": success,
        "grasped": [_bool(base.agent.is_grasping(cube)) for cube in cubes],
    }


def _save_h5(path: Path, accepted: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        for index, item in enumerate(accepted):
            group = handle.create_group(f"traj_{index:06d}")
            records = item["records"]
            actions = np.asarray(item["actions"], dtype=np.float32)
            obs = group.create_group("obs")
            sensor = obs.create_group("sensor_data")
            sensor.create_group("base_camera").create_dataset("rgb", data=np.stack([r["base"] for r in records]))
            sensor.create_group("hand_camera").create_dataset("rgb", data=np.stack([r["wrist"] for r in records]))
            obs.create_dataset("state", data=np.stack([r["state"] for r in records]))
            group.create_dataset("actions", data=actions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("bridge_pca", "offline_oracle", "failure_recovery", "diffdagger"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--asset", type=Path)
    parser.add_argument("--pca-threshold", type=float)
    parser.add_argument("--diff-threshold", type=float)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("id", "stage1_ood", "stage2_ood", "stage3_ood"), required=True)
    parser.add_argument("--target", type=int, default=20)
    parser.add_argument("--id-seed", type=int, default=70000)
    parser.add_argument("--ood-seed", type=int, default=80000)
    parser.add_argument("--max-attempts", type=int, default=500)
    parser.add_argument("--min-ood-fraction", type=float)
    parser.add_argument("--flow-steps", type=int, default=5)
    parser.add_argument("--diff-timesteps", type=int, default=16)
    parser.add_argument("--diff-patience", type=int, default=2)
    parser.add_argument("--sim-backend", choices=("gpu", "cpu"), default="cpu")
    parser.add_argument("--render-backend", choices=("gpu", "cpu"), default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    sys.path.insert(0, str(args.xvla_root.resolve()))
    register_stackpyramid_splits()
    _install_rrt_fallback()
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    from models.modeling_xvla import XVLA
    from models.processing_xvla import XVLAProcessor

    device = torch.device("cuda")
    model = XVLA.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16).to(device).eval()
    processor = XVLAProcessor.from_pretrained(args.checkpoint)
    stats = _load_asset(args.asset, device) if args.asset else None
    if args.method == "bridge_pca" and (stats is None or args.pca_threshold is None):
        raise ValueError("bridge_pca needs --asset and --pca-threshold")
    if args.method == "diffdagger" and args.diff_threshold is None:
        raise ValueError("diffdagger needs --diff-threshold")
    env = StepRecorder(gym.make(
        stackpyramid_env_id(args.split), obs_mode="rgb+state", control_mode="pd_joint_pos",
        render_mode="rgb_array", sim_backend=args.sim_backend, render_backend=args.render_backend,
    ))
    accepted: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    accepted_counts = {"id": 0, args.split: 0}
    next_seed = {"id": args.id_seed, args.split: args.ood_seed}
    try:
        for attempt in range(args.max_attempts):
            if len(accepted) >= args.target:
                break
            split = "id" if attempt % 2 == 0 else args.split
            seed = next_seed[split]
            next_seed[split] += 1
            raw_obs, _ = env.reset(seed=seed)
            expert_start: int | None = 0 if args.method == "offline_oracle" else None
            ever_grasped = False
            ever_base = False
            success = False
            scores: list[dict[str, Any]] = []
            terminated = truncated = False
            if args.method != "offline_oracle":
                while len(env.actions) < MAX_EPISODE_STEPS and expert_start is None and not (terminated or truncated):
                    generated, bridge, inputs, encoding = _predict(model, processor, raw_obs, device, seed + len(env.actions), args.flow_steps)
                    score = None
                    threshold = None
                    alarm = False
                    if args.method == "bridge_pca":
                        from rlinf.algorithms.vla_fail import pca_residual_score
                        score = float(pca_residual_score(bridge.unsqueeze(1), stats)[0].item())
                        threshold = args.pca_threshold
                        alarm = score > threshold
                    elif args.method == "diffdagger":
                        score = _diff_score(model, inputs, encoding, generated, args.diff_timesteps)
                        threshold = args.diff_threshold
                        alarm = score > threshold
                    elif args.method == "failure_recovery":
                        alarm = len(env.actions) >= 50
                        threshold = 50
                    scores.append({"env_step": len(env.actions), "score": score, "threshold": threshold, "alarm": alarm})
                    if alarm:
                        expert_start = len(env.actions)
                        break
                    low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
                    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
                    env.current_source = "policy"
                    for action in np.clip(generated[:ACTION_HORIZON], low, high)[:EXECUTE_HORIZON]:
                        raw_obs, _, terminated, truncated, _ = env.step(action)
                        current = _summary(env)
                        ever_grasped |= any(current["grasped"])
                        ever_base |= current["xy_ab"]
                        success |= current["success"]
                        if terminated or truncated or len(env.actions) >= MAX_EPISODE_STEPS:
                            break
            oracle_error = None
            if expert_start is not None and not success:
                env.current_source = "expert"
                try:
                    StackPyramidOracle(env).run()
                except Exception as exc:
                    oracle_error = repr(exc)
            final = _summary(env)
            ever_grasped |= any(final["grasped"])
            ever_base |= final["xy_ab"]
            success |= final["success"]
            raw_row = {
                "attempt": attempt,
                "seed": seed,
                "split": split,
                "method": args.method,
                "success": bool(success),
                "ever_grasped": bool(ever_grasped),
                "ever_base_completed": bool(ever_base),
                "expert_start_step": expert_start,
                "expert_action_steps": 0 if expert_start is None else max(0, len(env.actions) - expert_start),
                "steps": len(env.actions),
                "timeline": scores,
                "final": final,
                "oracle_error": oracle_error,
            }
            video_path = args.output_dir / "raw_videos" / f"attempt_{attempt:06d}_seed_{seed:06d}.mp4"
            video_path.parent.mkdir(parents=True, exist_ok=True)
            with imageio.get_writer(video_path, fps=10, codec="libx264", macro_block_size=None) as writer:
                for frame in env.frames:
                    writer.append_data(frame)
            raw_row["video"] = str(video_path)
            raw_rows.append(raw_row)
            with (args.output_dir / "episodes.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(raw_row) + "\n")
            if success and expert_start is not None and len(env.actions) > expert_start:
                item = {
                    "seed": seed,
                    "split": split,
                    "records": [_copy_record(r) for r in env.records[expert_start:]],
                    "actions": [action.copy() for action in env.actions[expert_start:]],
                }
                accepted.append(item)
                accepted_counts[split] = accepted_counts.get(split, 0) + 1
                print(json.dumps({"attempt": attempt, "accepted": len(accepted), "split": split, "expert_actions": len(item["actions"])}), flush=True)
            else:
                print(json.dumps({"attempt": attempt, "accepted": len(accepted), "split": split, "success": bool(success), "expert_start": expert_start}), flush=True)
        if len(accepted) < args.target:
            raise RuntimeError(f"collection incomplete: {len(accepted)}/{args.target}")
    finally:
        env.close()

    _save_h5(args.output_dir / "accepted_suffixes.h5", accepted)
    (args.output_dir / "training_episodes.jsonl").write_text(
        "".join(json.dumps({"index": i, "seed": item["seed"], "split": item["split"], "expert_action_steps": len(item["actions"])}) + "\n" for i, item in enumerate(accepted)),
        encoding="utf-8",
    )
    summary = {
        "format": "stackpyramid_xvla_gated_collection_v1",
        "method": args.method,
        "split": args.split,
        "target_accepted": args.target,
        "accepted_total": len(accepted),
        "accepted_by_split": accepted_counts,
        "raw_attempts": len(raw_rows),
        "raw_successes": sum(int(row["success"]) for row in raw_rows),
        "expert_action_steps": sum(len(item["actions"]) for item in accepted),
        "dataset": str((args.output_dir / "accepted_suffixes.h5").resolve()),
        "selection_metrics": {
            "raw_attempts": len(raw_rows),
            "raw_attempts_by_split": {
                split: sum(int(row["split"] == split) for row in raw_rows)
                for split in ("id", args.split)
            },
            "raw_successes_by_split": {
                split: sum(int(row["split"] == split and row["success"]) for row in raw_rows)
                for split in ("id", args.split)
            },
            "alarm_count_by_split": {
                split: sum(int(row["split"] == split and any(item["alarm"] for item in row["timeline"])) for row in raw_rows)
                for split in ("id", args.split)
            },
            "query_count_by_split": {
                split: sum(int(row["split"] == split and row["expert_start_step"] is not None and args.method != "offline_oracle") for row in raw_rows)
                for split in ("id", args.split)
            },
            "takeover_count_by_split": {
                split: sum(int(row["split"] == split and row["expert_start_step"] is not None and args.method != "offline_oracle") for row in raw_rows)
                for split in ("id", args.split)
            },
            "assisted_success_by_split": {
                split: sum(int(row["split"] == split and row["success"] and row["expert_start_step"] is not None and args.method != "offline_oracle") for row in raw_rows)
                for split in ("id", args.split)
            },
            "accepted_by_split": accepted_counts,
            "alternating_stream": True,
            "selection_is_not_forced_50_50": True,
        },
    }
    ood_fraction = accepted_counts.get(args.split, 0) / max(1, len(accepted))
    selection_gate = {
        "required_ood_fraction": args.min_ood_fraction,
        "actual_ood_fraction": ood_fraction,
        "pass": None if args.min_ood_fraction is None else ood_fraction >= args.min_ood_fraction,
        "gate_scope": "method_specific",
    }
    summary["selection_gate"] = selection_gate
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if selection_gate["pass"] is False:
        raise RuntimeError(
            f"selection gate failed: required OOD fraction {args.min_ood_fraction}, "
            f"actual {ood_fraction}, accepted_by_split={accepted_counts}"
        )
    (args.output_dir / "COLLECTION_COMPLETE").write_text("complete\n", encoding="utf-8")


if __name__ == "__main__":
    main()
