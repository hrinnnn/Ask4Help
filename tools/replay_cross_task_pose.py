#!/usr/bin/env python3
"""Extract comparable Panda TCP pose timelines from existing task artifacts.

This is a diagnostic-only adapter for the cross-task ERD visualisation.  It
never loads a policy checkpoint and never writes to a formal experiment root.
It accepts either saved action summaries (replayed in the registered ManiSkill
environment), saved qpos episode directories, or LeRobot parquet expert data.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


def _configure_repo(repo_root: Path) -> None:
    root = str(repo_root.resolve())
    rlinf = str((repo_root / "RLinf").resolve())
    sys.path[:0] = [root, rlinf]


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _bool_scalar(value: Any) -> bool:
    return bool(_to_numpy(value).reshape(-1)[0])


def _build_env(task: str, split: str, *, repo_root: Path, max_episode_steps: int, control_freq: int, sim_backend: str):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    if task == "stackcube":
        from tools.stackcube_stage2_ood import register_stack_cube_splits, stack_cube_env_id

        register_stack_cube_splits()
        env_id = stack_cube_env_id(split)
    elif task == "ycb":
        from rlinf.envs.maniskill.pick_single_ycb_object_variation import (
            PICK_SINGLE_YCB_OBJECT_ID_ENV_ID,
            PICK_SINGLE_YCB_OBJECT_OOD_ENV_ID,
            register_controlled_pick_single_ycb_object_variants,
        )

        register_controlled_pick_single_ycb_object_variants()
        env_id = PICK_SINGLE_YCB_OBJECT_ID_ENV_ID if split == "id" else PICK_SINGLE_YCB_OBJECT_OOD_ENV_ID
    elif task == "opendrawer":
        import rlinf.envs.maniskill.open_drawer_retrieve_place  # noqa: F401
        from rlinf.envs.maniskill.open_drawer_retrieve_place_spec import ENV_IDS

        if split not in ENV_IDS:
            raise ValueError(f"unsupported OpenDrawer split: {split}")
        env_id = ENV_IDS[split]
    else:
        raise ValueError(f"unsupported task: {task}")

    return gym.make(
        env_id,
        robot_uids="panda",
        num_envs=1,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        reward_mode="sparse",
        render_mode=None,
        sim_backend=sim_backend,
        sim_config={"sim_freq": 100, "control_freq": control_freq},
        max_episode_steps=max_episode_steps,
    )


def _snapshot(env: Any, task: str, info: dict[str, Any] | None = None) -> dict[str, Any]:
    base = env.unwrapped
    tcp = base.agent.tcp.pose
    qpos = _to_numpy(base.agent.robot.get_qpos()).reshape(-1).astype(np.float32)
    if qpos.size < 9:
        raise RuntimeError(f"expected Panda qpos with >=9 values, got {qpos.shape}")
    result: dict[str, Any] = {
        "position": _to_numpy(tcp.p).reshape(-1)[:3].astype(np.float32).tolist(),
        "quaternion_wxyz": _to_numpy(tcp.q).reshape(-1)[:4].astype(np.float32).tolist(),
        "gripper_width": float(qpos[-2:].sum()),
    }
    if task == "stackcube":
        result["object_position"] = _to_numpy(base.cubeA.pose.p).reshape(-1)[:3].astype(np.float32).tolist()
        result["target_position"] = _to_numpy(base.cubeB.pose.p).reshape(-1)[:3].astype(np.float32).tolist()
        result["grasped"] = bool(base.agent.is_grasping(base.cubeA))
        result["on_cube"] = _bool_scalar((info or {}).get("is_cubeA_on_cubeB", False))
        result["success"] = _bool_scalar((info or {}).get("success", False))
    elif task == "ycb":
        result["object_position"] = _to_numpy(base.obj.pose.p).reshape(-1)[:3].astype(np.float32).tolist()
        result["target_position"] = _to_numpy(base.goal_site.pose.p).reshape(-1)[:3].astype(np.float32).tolist()
        result["grasped"] = bool(base.agent.is_grasping(base.obj))
        result["strict_success"] = _bool_scalar((info or {}).get("success", False))
    else:
        result["object_position"] = _to_numpy(base.obj.pose.p).reshape(-1)[:3].astype(np.float32).tolist()
        result["target_position"] = _to_numpy(base.target_tray.pose.p).reshape(-1)[:3].astype(np.float32).tolist()
        result["grasped"] = bool(base.agent.is_grasping(base.obj))
        result["strict_success"] = _bool_scalar((info or {}).get("success", False))
    return result


def _save_pose(output: Path, episode_index: int, seed: int, poses: list[dict[str, Any]], metadata: dict[str, Any]) -> str:
    pose_array = np.asarray(
        [
            [
                *item["position"],
                *item["quaternion_wxyz"],
                item["gripper_width"],
                float(item.get("grasped", False)),
                float(item.get("on_cube", item.get("strict_success", False))),
            ]
            for item in poses
        ],
        dtype=np.float32,
    )
    object_positions = np.asarray([item["object_position"] for item in poses], dtype=np.float32)
    target_positions = np.asarray([item["target_position"] for item in poses], dtype=np.float32)
    path = output / "pose" / f"episode_{episode_index:06d}_seed_{seed:06d}.npz"
    np.savez_compressed(
        path,
        pose=pose_array,
        position=pose_array[:, :3],
        quaternion_wxyz=pose_array[:, 3:7],
        gripper_width=pose_array[:, 7],
        phase_flag_1=pose_array[:, 8],
        phase_flag_2=pose_array[:, 9],
        object_position=object_positions,
        target_position=target_positions,
    )
    metadata["pose_file"] = str(path)
    return str(path)


def _action_rows(summary: Path, limit: int | None) -> list[dict[str, Any]]:
    if summary.suffix == ".jsonl":
        rows = [json.loads(line) for line in summary.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        payload = json.loads(summary.read_text(encoding="utf-8"))
        rows = payload.get("rows")
        if isinstance(rows, str) and rows == "episodes.jsonl":
            rows = [
                json.loads(line)
                for line in (summary.parent / rows).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"summary has no rows: {summary}")
    return rows if limit is None else rows[:limit]


def _replay_actions(args: argparse.Namespace, env: Any) -> list[dict[str, Any]]:
    import torch

    rows = _action_rows(args.summary, args.limit)
    output_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        action_path = Path(row["actions"])
        actions = np.load(action_path).astype(np.float32)
        seed = int(row.get("seed", args.seed_start + index))
        _obs, _info = env.reset(seed=seed)
        poses = [_snapshot(env, args.task)]
        terminated = truncated = False
        for action in actions:
            if terminated or truncated:
                break
            tensor = torch.as_tensor(action, dtype=torch.float32, device=env.unwrapped.device).reshape(1, -1)
            _obs, _reward, terminated, truncated, info = env.step(tensor)
            poses.append(_snapshot(env, args.task, info))
        record: dict[str, Any] = {
            "episode_index": int(row.get("episode_index", index)),
            "seed": seed,
            "source_steps": int(row.get("steps", len(actions))),
            "action_array_steps": int(len(actions)),
            "pose_steps": len(poses) - 1,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "source_row": row,
        }
        _save_pose(args.output, record["episode_index"], seed, poses, record)
        output_rows.append(record)
        print(f"POSE_REPLAY_COMPLETE task={args.task} episode={record['episode_index']} seed={seed} steps={len(poses)-1}", flush=True)
    return output_rows


def _qpos_from_parquet(path: Path) -> np.ndarray:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["state"])
    values = table["state"].to_pylist()
    return np.asarray(values, dtype=np.float32)


def _metadata_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _set_qpos(env: Any, qpos: np.ndarray) -> None:
    import torch

    tensor = torch.as_tensor(qpos, dtype=torch.float32, device=env.unwrapped.device).reshape(1, -1)
    env.unwrapped.agent.robot.set_qpos(tensor)


def _replay_qpos_files(args: argparse.Namespace, env: Any, paths: list[tuple[int, int, Path, dict[str, Any]]]) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    for index, seed, state_path, metadata in paths:
        _obs, _info = env.reset(seed=seed)
        states = np.load(state_path).astype(np.float32)
        poses: list[dict[str, Any]] = []
        for state in states:
            _set_qpos(env, state)
            poses.append(_snapshot(env, args.task))
        record = {
            "episode_index": int(index),
            "seed": int(seed),
            "source_state_steps": int(len(states)),
            "source_metadata": metadata,
        }
        _save_pose(args.output, int(index), int(seed), poses, record)
        output_rows.append(record)
        print(f"POSE_QPOS_COMPLETE task={args.task} episode={index} seed={seed} steps={len(states)}", flush=True)
    return output_rows


def _build_qpos_paths(args: argparse.Namespace) -> list[tuple[int, int, Path, dict[str, Any]]]:
    assert args.episode_root is not None
    paths: list[tuple[int, int, Path, dict[str, Any]]] = []
    for index, directory in enumerate(sorted(args.episode_root.glob("episode_*"))):
        state_path = directory / "states.npy"
        metadata_path = directory / "reset_metadata.json"
        if not state_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
        paths.append((index, args.seed_start + index, state_path, metadata))
        if args.limit is not None and len(paths) >= args.limit:
            break
    return paths


def _build_parquet_paths(args: argparse.Namespace) -> list[tuple[int, int, Path, dict[str, Any]]]:
    assert args.parquet_root is not None and args.metadata_jsonl is not None
    meta = _metadata_rows(args.metadata_jsonl)
    selected: list[tuple[int, int, Path, dict[str, Any]]] = []
    for row in meta:
        if args.task == "opendrawer":
            wanted_source = "id" if args.split == "id" else "ood"
            wanted_split = "id" if args.split == "id" else args.split
            if row.get("source") != wanted_source or row.get("split") != wanted_split:
                continue
        index = int(row.get("episode_index", len(selected)))
        path = args.parquet_root / f"episode_{index:06d}.parquet"
        if not path.is_file():
            continue
        seed = int(row.get("seed", args.seed_start + index))
        selected.append((index, seed, path, row))
        if args.limit is not None and len(selected) >= args.limit:
            break
    return selected


def replay(args: argparse.Namespace) -> dict[str, Any]:
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "pose").mkdir(exist_ok=True)
    env = _build_env(
        args.task,
        args.split,
        repo_root=args.repo_root,
        max_episode_steps=args.max_episode_steps,
        control_freq=args.control_freq,
        sim_backend=args.sim_backend,
    )
    try:
        if args.source == "summary":
            rows = _replay_actions(args, env)
        elif args.source == "qpos_dirs":
            rows = _replay_qpos_files(args, env, _build_qpos_paths(args))
        else:
            parquet_rows = _build_parquet_paths(args)
            rows = []
            for index, seed, parquet, metadata in parquet_rows:
                _obs, _info = env.reset(seed=seed)
                states = _qpos_from_parquet(parquet)
                poses = []
                for state in states:
                    _set_qpos(env, state)
                    poses.append(_snapshot(env, args.task))
                record = {"episode_index": index, "seed": seed, "source_parquet": str(parquet), "source_state_steps": len(states), "source_metadata": metadata}
                _save_pose(args.output, index, seed, poses, record)
                rows.append(record)
                print(f"POSE_PARQUET_COMPLETE task={args.task} episode={index} seed={seed} steps={len(states)}", flush=True)
    finally:
        env.close()
    payload = {
        "format": "cross_task_pose_replay_v1",
        "task": args.task,
        "split": args.split,
        "source": args.source,
        "episodes_replayed": len(rows),
        "rows": rows,
        "formal_pipeline_untouched": True,
    }
    (args.output / "pose_replay_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--task", choices=("stackcube", "ycb", "opendrawer"), required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--source", choices=("summary", "qpos_dirs", "parquet"), required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--episode-root", type=Path)
    parser.add_argument("--parquet-root", type=Path)
    parser.add_argument("--metadata-jsonl", type=Path)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-episode-steps", type=int, default=400)
    parser.add_argument("--control-freq", type=int, default=10)
    parser.add_argument("--sim-backend", choices=("physx_cpu", "gpu"), default="physx_cpu")
    args = parser.parse_args()
    if args.source == "summary" and args.summary is None:
        parser.error("--summary is required for --source summary")
    if args.source == "qpos_dirs" and args.episode_root is None:
        parser.error("--episode-root is required for --source qpos_dirs")
    if args.source == "parquet" and (args.parquet_root is None or args.metadata_jsonl is None):
        parser.error("--parquet-root and --metadata-jsonl are required for --source parquet")
    _configure_repo(args.repo_root)
    print(json.dumps(replay(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
