#!/usr/bin/env python3
"""Collect fixed-timing OpenDrawer expert suffixes on one OOD split.

The collector keeps the policy prefix and expert continuation in one fresh
episode.  It is deliberately separate from detector/gate collection: the
only intervention variable is ``--takeover-step``.  In addition to the
LeRobot suffix dataset, it writes full episode evidence and task-state
timelines so that D-path and phase-relative timing can be audited later.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RLINF_ROOT = Path(os.environ.get("ASK4HELP_RLINF_ROOT", ROOT / "RLinf"))
sys.path[:0] = [str(ROOT), str(RLINF_ROOT)]

from rlinf.envs.maniskill.open_drawer_retrieve_place_spec import (  # noqa: E402
    ENV_IDS,
    TASK_INSTRUCTION,
    reset_metadata,
    validate_reset_metadata,
)
from toolkits.lerobot.collect_maniskill_peg_lerobot_joint import (  # noqa: E402
    MAIN_CAMERA_CANDIDATES,
    WRIST_CAMERA_CANDIDATES,
    _build_frames,
    _camera_image,
    _convert_solver_action_to_joint_delta,
    _create_dataset,
    _extract_record,
    _joint_delta_arm_bounds,
    _select_camera,
    _to_numpy,
)
from toolkits.lerobot.collect_maniskill_pick_single_ycb_airplane_lerobot import (  # noqa: E402
    write_episode_video_durably,
)
from toolkits.lerobot.validate_open_drawer_retrieve_place_oracle import (  # noqa: E402
    PandaPosePlannerClient,
    continue_episode as legacy_continue_episode,
)
from tools.open_drawer_direct_takeover_oracle import (  # noqa: E402
    continue_episode as direct_continue_episode,
)
from tools.evaluate_open_drawer_id_pi05 import _model_obs  # noqa: E402
from tools.maniskill_pi05_vfd_online_awbc import _load_model  # noqa: E402


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _bool(value: Any) -> bool:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value).reshape(-1)
    return bool(array[0]) if array.size else False


def _vector(value: Any) -> list[float]:
    return _to_numpy(value).astype(np.float32).reshape(-1).tolist()


def _task_state(env: Any) -> dict[str, Any]:
    """Snapshot task-relative state directly from the live simulator."""

    base = env.unwrapped
    tcp = base.agent.tcp.pose
    obj = base.obj.pose
    target = base.target_tray.pose
    evaluation = base.evaluate()
    return {
        "tcp_position": _vector(tcp.p),
        "tcp_quaternion": _vector(tcp.q),
        "object_position": _vector(obj.p),
        "object_quaternion": _vector(obj.q),
        "target_position": _vector(target.p),
        "drawer_qpos": _vector(base.drawer.get_qpos()),
        "object_grasped": _bool(evaluation["is_grasped"]),
        "ever_drawer_opened": _bool(evaluation["ever_drawer_opened"]),
        "ever_grasped": _bool(evaluation["ever_grasped"]),
        "ever_lifted": _bool(evaluation["ever_lifted"]),
        "object_in_target": _bool(evaluation["object_in_target"]),
        "object_released": _bool(evaluation["object_released"]),
        "is_robot_static": _bool(evaluation["is_robot_static"]),
        "success": _bool(evaluation["success"]),
    }


def _phase(state: dict[str, Any]) -> str:
    if state["success"] or state["object_in_target"]:
        return "placement"
    if state["ever_lifted"]:
        return "transport"
    if state["ever_grasped"]:
        return "grasp_or_lift"
    if state["ever_drawer_opened"]:
        return "post_open_pre_grasp"
    return "open_drawer"


class _RecordingProxy:
    """Forward calls to the live env while retaining expert frames/actions."""

    def __init__(self, env: Any, records: list[Any], actions: list[np.ndarray], *, delta_bounds: tuple[np.ndarray, np.ndarray] | None = None):
        self._env = env
        self._records = records
        self._actions = actions
        self._delta_bounds = delta_bounds

    @property
    def unwrapped(self):
        return self._env.unwrapped

    def step(self, action, *args, **kwargs):
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        if self._delta_bounds is not None:
            lower, upper = self._delta_bounds
            current_qpos = np.asarray(self._env.unwrapped.agent.robot.get_qpos(), dtype=np.float32).reshape(-1)
            action_array = _convert_solver_action_to_joint_delta(
                current_qpos, action_array, lower, upper
            )
        observation, reward, terminated, truncated, info = self._env.step(action_array, *args, **kwargs)
        self._actions.append(action_array)
        self._records.append(_extract_record(observation))
        return observation, reward, terminated, truncated, info

    def __getattr__(self, name: str):
        return getattr(self._env, name)


def _build_env(split: str, *, max_episode_steps: int, sim_backend: str, image_size: int):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    import rlinf.envs.maniskill.open_drawer_retrieve_place  # noqa: F401

    return gym.make(
        ENV_IDS[split],
        robot_uids="panda_wristcam",
        num_envs=1,
        obs_mode="rgb",
        control_mode="pd_joint_delta_pos",
        reward_mode="sparse",
        render_mode="rgb_array",
        sim_backend=sim_backend,
        sim_config={"sim_freq": 100, "control_freq": 10},
        sensor_configs={"width": image_size, "height": image_size},
        max_episode_steps=max_episode_steps,
    )


class _TimingRecordingProxy(_RecordingProxy):
    """Record task state after every expert step as well as FrameRecord."""

    def __init__(self, env: Any, records: list[Any], actions: list[np.ndarray], task_states: list[dict[str, Any]], **kwargs: Any):
        super().__init__(env, records, actions, **kwargs)
        self._task_states = task_states

    def step(self, action, *args, **kwargs):
        result = super().step(action, *args, **kwargs)
        self._task_states.append(_task_state(self._env))
        return result


def _save_task_timeline(path: Path, *, seed: int, takeover_step: int, states: list[dict[str, Any]]) -> None:
    rows = []
    for step, state in enumerate(states):
        rows.append({"step": step, "phase": _phase(state), **state})
    path.write_text(
        json.dumps(
            {
                "format": "open_drawer_fixed_timing_task_state_timeline_v1",
                "seed": seed,
                "scheduled_takeover_step": takeover_step,
                "actual_takeover_step": min(takeover_step, len(rows) - 1),
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def collect_one(
    *,
    env: Any,
    model: Any,
    planner: PandaPosePlannerClient,
    seed: int,
    takeover_step: int,
    execute_horizon: int,
    max_episode_steps: int,
    oracle_mode: str,
) -> dict[str, Any]:
    low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    raw_obs, _info = env.reset(seed=seed)
    metadata = reset_metadata(env, split=env.unwrapped.rlinf_split)
    main_camera = _select_camera(
        _extract_record(raw_obs).obs,
        "",
        ("base_camera",) + MAIN_CAMERA_CANDIDATES,
        "main",
    )
    wrist_camera = _select_camera(
        _extract_record(raw_obs).obs,
        "",
        ("hand_camera",) + WRIST_CAMERA_CANDIDATES,
        "wrist",
    )
    metadata["camera"] = {
        "main": main_camera,
        "wrist": wrist_camera,
        "main_shape": list(_camera_image(_extract_record(raw_obs).obs, main_camera).shape),
        "wrist_shape": list(_camera_image(_extract_record(raw_obs).obs, wrist_camera).shape),
        "requested_size": [384, 384],
    }
    errors = validate_reset_metadata(metadata, split=env.unwrapped.rlinf_split)
    if errors:
        raise RuntimeError(f"reset metadata failed for seed {seed}: {errors}")
    prefix_records = [_extract_record(raw_obs)]
    task_states = [_task_state(env)]
    prefix_actions: list[np.ndarray] = []
    terminated = truncated = False
    while len(prefix_actions) < takeover_step and len(prefix_actions) < max_episode_steps:
        with torch.inference_mode():
            predicted, _ = model.predict_action_batch(
                env_obs=_model_obs(raw_obs, TASK_INSTRUCTION),
                mode="eval",
                compute_values=False,
            )
        remaining = min(execute_horizon, takeover_step - len(prefix_actions))
        chunk = np.clip(predicted.detach().float().cpu().numpy()[0][:remaining], low, high).astype(np.float32)
        for action in chunk:
            raw_obs, _reward, terminated, truncated, _info = env.step(
                torch.as_tensor(action, device=env.unwrapped.device).unsqueeze(0)
            )
            prefix_actions.append(action)
            prefix_records.append(_extract_record(raw_obs))
            task_states.append(_task_state(env))
            if terminated or truncated or task_states[-1]["success"]:
                break
        if terminated or truncated or task_states[-1]["success"]:
            break

    actual_takeover_step = len(prefix_actions)
    prefix_phase = _phase(task_states[-1])
    if actual_takeover_step != takeover_step or terminated or truncated or task_states[-1]["success"]:
        return {
            "seed": seed,
            "scheduled_takeover_step": takeover_step,
            "actual_takeover_step": actual_takeover_step,
            "accepted": False,
            "reason": "policy_terminated_before_scheduled_takeover",
            "success": bool(task_states[-1]["success"]),
            "prefix_phase": prefix_phase,
            "metadata": metadata,
            "prefix_actions": prefix_actions,
            "prefix_records": prefix_records,
            "task_states": task_states,
        }

    expert_records = [prefix_records[-1]]
    expert_actions: list[np.ndarray] = []
    proxy = _TimingRecordingProxy(
        env,
        expert_records,
        expert_actions,
        task_states,
        delta_bounds=_joint_delta_arm_bounds(env),
    )
    try:
        oracle_fn = direct_continue_episode if oracle_mode == "direct_grasp" else legacy_continue_episode
        expert_result = oracle_fn(proxy, planner, seed=seed)
    except Exception as exc:
        return {
            "seed": seed,
            "scheduled_takeover_step": takeover_step,
            "actual_takeover_step": actual_takeover_step,
            "accepted": False,
            "reason": "expert_continuation_exception",
            "success": False,
            "prefix_phase": prefix_phase,
            "metadata": metadata,
            "expert_error": repr(exc),
            "prefix_actions": prefix_actions,
            "expert_actions": [],
            "full_actions": prefix_actions,
            "prefix_records": prefix_records,
            "expert_records": expert_records,
            "full_records": prefix_records,
            "task_states": task_states,
        }
    full_records = prefix_records + expert_records[1:]
    full_actions = prefix_actions + expert_actions
    success = bool(expert_result.get("success", False))
    return {
        "seed": seed,
        "scheduled_takeover_step": takeover_step,
        "actual_takeover_step": actual_takeover_step,
        "accepted": bool(success and expert_actions),
        "reason": "accepted" if success and expert_actions else "expert_continuation_failed",
        "success": success,
        "oracle_mode": oracle_mode,
        "prefix_phase": prefix_phase,
        "metadata": metadata,
        "expert_result": expert_result,
        "prefix_actions": prefix_actions,
        "expert_actions": expert_actions,
        "full_actions": full_actions,
        "prefix_records": prefix_records,
        "expert_records": expert_records,
        "full_records": full_records,
        "task_states": task_states,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=tuple(ENV_IDS),
        default="grasp_ood",
        help="controlled task split used for the policy prefix and Oracle suffix",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pi05-base", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--takeover-step", type=int, required=True)
    parser.add_argument("--start-seed", type=int, default=78000)
    parser.add_argument("--target", type=int, default=20)
    parser.add_argument("--max-attempts", type=int, default=100)
    parser.add_argument("--execute-horizon", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=400)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--sim-backend", choices=("physx_cpu", "gpu"), default="physx_cpu")
    parser.add_argument(
        "--oracle-mode",
        choices=("legacy", "direct_grasp"),
        default="legacy",
        help="legacy full continuation or current-state direct-grasp continuation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.takeover_step < 0 or args.target <= 0 or args.max_attempts < args.target:
        raise ValueError("invalid takeover step/target/max-attempts")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "raw_attempts").mkdir()
    (args.output_root / "accepted").mkdir()
    (args.output_root / "raw_videos").mkdir()
    model = _load_model(args.checkpoint, args.norm_stats, args.pi05_base)
    planner = PandaPosePlannerClient()
    env = _build_env(
        args.split,
        max_episode_steps=args.max_episode_steps,
        sim_backend=args.sim_backend,
        image_size=args.image_size,
    )
    dataset = None
    accepted_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    try:
        for attempt in range(args.max_attempts):
            if len(accepted_rows) >= args.target:
                break
            seed = args.start_seed + attempt
            result = collect_one(
                env=env,
                model=model,
                planner=planner,
                seed=seed,
                takeover_step=args.takeover_step,
                execute_horizon=args.execute_horizon,
                max_episode_steps=args.max_episode_steps,
                oracle_mode=args.oracle_mode,
            )
            attempt_dir = args.output_root / "raw_attempts" / f"attempt_{attempt:06d}_seed_{seed:06d}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            full_actions = np.asarray(result.get("full_actions", result.get("prefix_actions", [])), dtype=np.float32)
            full_records = result.get("full_records", result.get("prefix_records", []))
            task_states = result.get("task_states", [])
            if full_records and full_actions.size and len(full_records) == len(full_actions) + 1:
                np.save(attempt_dir / "actions.npy", full_actions)
                np.save(attempt_dir / "states.npy", np.asarray([r.state for r in full_records], dtype=np.float32))
                (attempt_dir / "reset_metadata.json").write_text(json.dumps(_jsonable(result["metadata"]), indent=2) + "\n")
                _save_task_timeline(
                    attempt_dir / "task_state_timeline.json",
                    seed=seed,
                    takeover_step=args.takeover_step,
                    states=task_states,
                )
                main_camera = _select_camera(full_records[0].obs, "", ("base_camera",) + MAIN_CAMERA_CANDIDATES, "main")
                wrist_camera = _select_camera(full_records[0].obs, "", ("hand_camera",) + WRIST_CAMERA_CANDIDATES, "wrist")
                frames = _build_frames(
                    records=full_records,
                    actions=list(full_actions),
                    task=TASK_INSTRUCTION,
                    main_camera=main_camera,
                    wrist_camera=wrist_camera,
                )
                video = write_episode_video_durably(
                    frames,
                    video_dir=args.output_root / "raw_videos",
                    episode_index=attempt,
                    seed=seed,
                    fps=10,
                )
                result["video"] = str(video)
            raw_row = {
                "attempt": attempt,
                "seed": seed,
                "scheduled_takeover_step": args.takeover_step,
                "actual_takeover_step": result.get("actual_takeover_step"),
                "prefix_phase": result.get("prefix_phase"),
                "accepted": bool(result.get("accepted", False)),
                "success": bool(result.get("success", False)),
                "reason": result.get("reason"),
                "oracle_mode": args.oracle_mode,
                "expert_action_steps": len(result.get("expert_actions", [])),
                "video": result.get("video"),
                "evidence_dir": str(attempt_dir),
            }
            raw_rows.append(raw_row)
            (attempt_dir / "attempt.json").write_text(json.dumps(_jsonable({**raw_row, "metadata": result.get("metadata"), "expert_result": result.get("expert_result")}), indent=2) + "\n")
            with (args.output_root / "raw_attempts.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(_jsonable(raw_row), sort_keys=True) + "\n")
            if not result.get("accepted", False):
                print(json.dumps({"attempt": attempt, "accepted": len(accepted_rows), "success": bool(result.get("success", False)), "reason": result.get("reason")}), flush=True)
                continue

            accepted_index = len(accepted_rows)
            accepted_dir = args.output_root / "accepted" / f"episode_{accepted_index:06d}"
            accepted_dir.mkdir(parents=True, exist_ok=True)
            for name in ("actions.npy", "states.npy", "reset_metadata.json", "task_state_timeline.json"):
                source = args.output_root / "raw_attempts" / f"attempt_{attempt:06d}_seed_{seed:06d}" / name
                if source.exists():
                    (accepted_dir / name).write_bytes(source.read_bytes())
            suffix_records = result["expert_records"]
            suffix_actions = result["expert_actions"]
            suffix_main = _select_camera(suffix_records[0].obs, "", ("base_camera",) + MAIN_CAMERA_CANDIDATES, "main")
            suffix_wrist = _select_camera(suffix_records[0].obs, "", ("hand_camera",) + WRIST_CAMERA_CANDIDATES, "wrist")
            suffix_frames = _build_frames(
                records=suffix_records,
                actions=suffix_actions,
                task=TASK_INSTRUCTION,
                main_camera=suffix_main,
                wrist_camera=suffix_wrist,
            )
            if dataset is None:
                dataset = _create_dataset(
                    repo_id=str(args.output_root / "lerobot_dataset"),
                    image_shape=tuple(suffix_frames[0]["image"].shape),
                    wrist_image_shape=tuple(suffix_frames[0]["wrist_image"].shape),
                    fps=10,
                    image_writer_threads=4,
                    image_writer_processes=4,
                )
            for frame in suffix_frames:
                dataset.add_frame(frame)
            dataset.save_episode()
            accepted_row = {
                **raw_row,
                "episode_index": accepted_index,
                "accepted_dir": str(accepted_dir),
                "raw_video": result.get("video"),
            }
            accepted_rows.append(accepted_row)
            with (args.output_root / "accepted_experts.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(_jsonable(accepted_row), sort_keys=True) + "\n")
            print(json.dumps({"attempt": attempt, "accepted": len(accepted_rows), "success": True}), flush=True)
    finally:
        if dataset is not None and getattr(dataset, "image_writer", None) is not None:
            dataset.image_writer.wait_until_done()
        planner.close()
        env.close()
        del model
        torch.cuda.empty_cache()

    summary = {
        "format": "open_drawer_fixed_timing_collection_v1",
        "task": "OpenDrawerRetrievePlace",
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "scheduled_takeover_step": args.takeover_step,
        "target_accepted": args.target,
        "accepted": len(accepted_rows),
        "raw_attempts": len(raw_rows),
        "raw_successes": sum(int(row["success"]) for row in raw_rows),
        "expert_actions": sum(int(row["expert_action_steps"]) for row in accepted_rows),
        "episodes": str(args.output_root / "accepted_experts.jsonl"),
        "raw_attempt_manifest": str(args.output_root / "raw_attempts.jsonl"),
        "dataset": str(args.output_root / "lerobot_dataset"),
        "status": "complete" if len(accepted_rows) >= args.target else "incomplete_unrecoverable_candidate",
        "oracle_mode": args.oracle_mode,
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if len(accepted_rows) < args.target:
        (args.output_root / "COLLECTION_FAILED").write_text(
            f"accepted={len(accepted_rows)} target={args.target} attempts={len(raw_rows)}\n"
        )
        raise SystemExit(f"fixed timing collection incomplete: {len(accepted_rows)}/{args.target}")
    (args.output_root / "COLLECTION_COMPLETE").write_text("complete fixed-timing Grasp-OOD suffix collection\n", encoding="utf-8")


if __name__ == "__main__":
    main()
