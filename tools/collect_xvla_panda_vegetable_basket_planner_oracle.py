#!/usr/bin/env python3
"""Collect Panda motion-planning Oracle evidence for the basket task."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import gymnasium as gym
import h5py
import numpy as np

from panda_vegetable_basket_adapter import tcp_pose_world, world_pose_to_base


TASK = "put the vegetable into the yellow basket"
ENV_IDS = {
    "id": "XVLAPandaPutVegetableInBasketID-v1",
    "ood": "XVLAPandaPutVegetableInBasketOOD-v1",
}


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _bool(value: Any) -> bool:
    return bool(_array(value).reshape(-1)[0])


def _pose(actor: Any) -> np.ndarray:
    return _array(actor.pose.raw_pose).reshape(-1, 7)[0].astype(np.float32)


def _rgb(obs: dict[str, Any]) -> np.ndarray:
    value = _array(obs["sensor_data"]["3rd_view_camera"]["rgb"])
    return value[0].astype(np.uint8) if value.ndim == 4 else value.astype(np.uint8)


def _load_task_module(path: Path) -> None:
    spec = importlib.util.spec_from_file_location("panda_vegetable_basket_variants", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load task module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def _move(planner: Any, pose: Any) -> bool:
    result = planner.move_to_pose_with_screw(pose)
    if result != -1:
        return True
    return planner.move_to_pose_with_RRTConnect(pose) != -1


class RecordingEnv:
    """Expose an environment to the planner while recording every transition."""

    def __init__(self, env: Any):
        self.env = env
        self.records: list[dict[str, Any]] = []
        self.actions: list[np.ndarray] = []
        self.phases: list[str] = []
        self.current_phase = "reset"
        self.gripper_01 = 1.0

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __getattr__(self, name: str):
        return getattr(self.env, name)

    def _record(self, obs: dict[str, Any]) -> None:
        base = self.env.unwrapped
        self.records.append(
            {
                "rgb": _rgb(obs),
                "tcp_world": tcp_pose_world(self.env),
                "tcp_base": world_pose_to_base(self.env, tcp_pose_world(self.env)),
                "object_pose": _pose(base.objs[base.source_obj_name]),
                "target_pose": _pose(base.objs[base.target_obj_name]),
                "gripper_01": float(self.gripper_01),
            }
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.records = []
        self.actions = []
        self.phases = []
        self.current_phase = "reset"
        self.gripper_01 = 1.0
        self._record(obs)
        return obs, info

    def step(self, action):
        action_array = _array(action).reshape(-1).astype(np.float32)
        if action_array.size:
            self.gripper_01 = float(np.clip((action_array[-1] + 1.0) / 2.0, 0.0, 1.0))
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.actions.append(action_array.copy())
        self.phases.append(self.current_phase)
        self._record(obs)
        return obs, reward, terminated, truncated, info


def _save_episode(output: Path, index: int, recorder: RecordingEnv, metadata: dict[str, Any]) -> dict[str, Any]:
    data_dir = output / "data"
    video_dir = output / "videos"
    metadata_dir = output / "metadata"
    for path in (data_dir, video_dir, metadata_dir):
        path.mkdir(parents=True, exist_ok=True)
    stem = f"episode_{index:06d}"
    h5_path = data_dir / f"{stem}.h5"
    video_path = video_dir / f"{stem}.mp4"
    metadata_path = metadata_dir / f"{stem}.json"
    transitions = len(recorder.actions)
    if len(recorder.records) != transitions + 1:
        raise RuntimeError("record boundary mismatch")
    proprio = np.stack([record["tcp_base"][:10] for record in recorder.records[:-1]])
    actions = np.stack(
        [
            np.concatenate(
                [record["tcp_base"][:9], [record["gripper_01"]]]
            ).astype(np.float32)
            for record in recorder.records[1:]
        ]
    )
    source_states = np.stack([record["object_pose"] for record in recorder.records[:-1]])
    target_states = np.stack([record["target_pose"] for record in recorder.records[:-1]])
    frames = [record["rgb"] for record in recorder.records]
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("images", data=np.asarray(frames, dtype=np.uint8), compression="gzip")
        h5.create_dataset("proprio", data=proprio.astype(np.float32))
        h5.create_dataset("abs_action_6d", data=actions.astype(np.float32))
        h5.create_dataset("object_pose", data=source_states.astype(np.float32))
        h5.create_dataset("target_pose", data=target_states.astype(np.float32))
        h5.attrs["seed"] = int(metadata["seed"])
        h5.attrs["success"] = bool(metadata["success"])
        h5.attrs["split"] = metadata["split"]
        h5.attrs["robot"] = metadata["robot"]
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer: {video_path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    metadata = dict(metadata)
    metadata.update(
        {
            "data_path": str(h5_path),
            "video_path": str(video_path),
            "metadata_path": str(metadata_path),
            "num_observations": len(recorder.records),
            "num_actions": transitions,
            "phases": recorder.phases,
            "expert_control_start": 0,
            "expert_control_end": transitions,
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def run_episode(
    env: Any,
    split: str,
    seed: int,
    episode_index: int,
    output: Path,
    lift_height: float,
    release_max_steps: int,
    closing_axis_mode: str,
) -> dict[str, Any]:
    import sapien
    from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver

    recorder = RecordingEnv(env)
    recorder.reset(seed=int(seed))
    base = env.unwrapped
    source = base.objs[base.source_obj_name]
    target = base.objs[base.target_obj_name]
    source_start = _pose(source)
    target_start = _pose(target)
    source_sp = source.pose.sp
    target_sp = target.pose.sp
    object_matrix = np.asarray(source_sp.to_transformation_matrix(), dtype=np.float64)
    local_axis = np.array(
        [1.0, 0.0, 0.0] if closing_axis_mode == "object_local_x" else [0.0, 1.0, 0.0]
    )
    closing = (
        object_matrix[:3, :3] @ local_axis
        if closing_axis_mode.startswith("object_local")
        else np.array([0.0, 1.0, 0.0])
    )
    closing[2] = 0.0
    closing /= max(float(np.linalg.norm(closing)), 1e-8)
    approaching = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    grasp_pose = base.agent.build_grasp_pose(approaching, closing, source_sp.p)
    planner = PandaArmMotionPlanningSolver(
        recorder,
        debug=False,
        vis=False,
        base_pose=base.agent.robot.pose.sp,
        visualize_target_grasp_pose=False,
        print_env_info=False,
    )
    stages: dict[str, Any] = {}
    stable_grasp = False
    lifted = False
    placed = False
    try:
        recorder.current_phase = "open"
        planner.open_gripper(t=4)
        recorder.current_phase = "pregrasp"
        stages["pregrasp_reached"] = _move(planner, grasp_pose * sapien.Pose([0, 0, -0.06]))
        recorder.current_phase = "grasp"
        stages["grasp_reached"] = stages["pregrasp_reached"] and _move(planner, grasp_pose)
        recorder.current_phase = "close"
        planner.close_gripper(t=8)
        for _ in range(4):
            if _bool(base.agent.is_grasping(source)):
                stable_grasp = True
                break
        stages["stable_grasp"] = stable_grasp
        if stable_grasp:
            object_in_tcp = base.agent.tcp_pose.sp.inv() * source.pose.sp
            lift_object = sapien.Pose(
                source.pose.sp.p + np.array([0.0, 0.0, lift_height]), source.pose.sp.q
            )
            recorder.current_phase = "lift"
            stages["lift_command_completed"] = _move(planner, lift_object * object_in_tcp.inv())
            lifted = bool(
                stages["lift_command_completed"]
                and _bool(base.agent.is_grasping(source))
                and float(source.pose.p[0, 2].cpu()) - float(source_start[2]) >= 0.05
            )
            stages["lifted"] = lifted
            if lifted:
                release_object = sapien.Pose(
                    target_sp.p + np.array([0.0, 0.0, 0.08]), source.pose.sp.q
                )
                recorder.current_phase = "transport"
                stages["transport_command_completed"] = _move(
                    planner, release_object * object_in_tcp.inv()
                )
                recorder.current_phase = "release"
                for _ in range(release_max_steps):
                    planner.open_gripper(t=1)
                    if _bool(base.evaluate()["success"]):
                        break
                placed = _bool(base.evaluate()["success"])
                stages["placed"] = placed
    except Exception as exc:
        stages["oracle_error"] = repr(exc)
    finally:
        planner.close()
    evaluation = base.evaluate()
    success = _bool(evaluation["success"])
    metadata = {
        "episode_index": episode_index,
        "split": split,
        "seed": int(seed),
        "robot": type(base.agent).__name__,
        "source_object": base.source_obj_name,
        "target_object": base.target_obj_name,
        "source_model_scale": float(base.episode_model_scales[base.source_obj_name]),
        "target_model_scale": float(base.episode_model_scales[base.target_obj_name]),
        "instruction": TASK,
        "configured_source_pose": _array(base.xyz_configs[0, 0]).tolist(),
        "configured_target_pose": _array(base.xyz_configs[0, 1]).tolist(),
        "start_source_pose": source_start.tolist(),
        "start_target_pose": target_start.tolist(),
        "final_source_pose": _pose(source).tolist(),
        "final_target_pose": _pose(target).tolist(),
        "success": success,
        "evaluation": {
            key: bool(_array(value).reshape(-1)[0])
            if _array(value).dtype == bool
            else float(_array(value).reshape(-1)[0])
            for key, value in evaluation.items()
        },
        "stages": stages,
        "max_source_z": float(max(record["object_pose"][2] for record in recorder.records)),
        "lift_height": float(lift_height),
        "release_max_steps": int(release_max_steps),
        "closing_axis_mode": closing_axis_mode,
        "action_contract": "joint-position planner replay re-encoded as base-frame absolute EE6D targets",
    }
    return _save_episode(output, episode_index, recorder, metadata)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rlinf-root", type=Path, required=True)
    parser.add_argument("--task-module", type=Path, required=True)
    parser.add_argument("--split", choices=("id", "ood"), required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lift-height", type=float, default=0.18)
    parser.add_argument("--release-max-steps", type=int, default=30)
    parser.add_argument("--max-episode-steps", type=int, default=120)
    parser.add_argument(
        "--closing-axis-mode",
        choices=("object_local_y", "object_local_x", "world_y"),
        default="object_local_y",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.rlinf_root))
    _load_task_module(args.task_module)
    env = gym.make(
        ENV_IDS[args.split],
        obs_mode="rgb+segmentation",
        render_mode="rgb_array",
        sim_backend="physx_cpu",
        control_mode="pd_joint_pos",
        max_episode_steps=args.max_episode_steps,
    )
    rows = []
    try:
        for index in range(args.episodes):
            row = run_episode(
                env,
                args.split,
                args.seed_start + index,
                index,
                args.output,
                args.lift_height,
                args.release_max_steps,
                args.closing_axis_mode,
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        env.close()
    summary = {
        "format": "xvla_panda_vegetable_basket_planner_oracle_v1",
        "split": args.split,
        "episodes": len(rows),
        "successes": sum(int(row["success"]) for row in rows),
        "videos": len(list((args.output / "videos").glob("*.mp4"))),
        "rows": rows,
    }
    (args.output / "episodes.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output / f"{args.split.upper()}_ORACLE_COMPLETE").write_text("complete\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
