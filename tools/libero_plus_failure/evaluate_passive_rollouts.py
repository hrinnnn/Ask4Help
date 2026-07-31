#!/usr/bin/env python3
"""Record passive clean or LIBERO-Plus pi0.5 rollouts with raw features.

Run this through the existing Python 3.8 LIBERO client.  It intentionally
contains no detector threshold and never changes the action chosen by pi0.5:
this is a passive failure-detection benchmark, not an intervention loop.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import imageio
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client.websocket_client_policy import WebsocketClientPolicy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from libero_plus_failure.rollout_records import (  # noqa: E402
    absolute_eef_points,
    single_sample_overlap,
    velocity_normalized_acc,
    write_rollout,
)


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
RESOLUTION = 256
REPLAN_STEPS = 5
ACTION_HORIZON = 10
WAIT_STEPS = 10
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}


def quat_to_axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(denominator), 0.0):
        return np.zeros(3, dtype=np.float64)
    return quat[:3] * (2.0 * math.acos(float(quat[3]))) / denominator


def get_env(task: Any, seed: int) -> tuple[Any, str]:
    task_description = str(task.language)
    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file, camera_heights=RESOLUTION, camera_widths=RESOLUTION
    )
    env.seed(seed)
    return env, task_description


def policy_input(
    obs: Dict[str, Any], prompt: str, resize_size: int, action_variance_samples: int
) -> tuple[dict, np.ndarray]:
    base = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    base = image_tools.convert_to_uint8(image_tools.resize_with_pad(base, resize_size, resize_size))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, resize_size, resize_size))
    result = {
        "observation/image": base,
        "observation/wrist_image": wrist,
        "observation/state": np.concatenate((obs["robot0_eef_pos"], quat_to_axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])),
        "prompt": prompt,
    }
    if action_variance_samples:
        result["failure_probe/action_variance_samples"] = int(action_variance_samples)
    return result, base


def run_episode(
    *, client: WebsocketClientPolicy, task: Any, task_index: int, seed: int, output_dir: Path,
    suite: str, source: str, category: str | None, configuration_id: int | None, resize_size: int,
    action_variance_samples: int,
) -> None:
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite rollout " + str(output_dir))
    env, prompt = get_env(task, seed)
    env.reset()
    initial_states = benchmark.get_benchmark_dict()[suite]().get_task_init_states(task_index)
    if len(initial_states) == 0:
        raise RuntimeError("task has no initial state")
    obs = env.set_init_state(initial_states[0])
    plan: collections.deque = collections.deque()
    timeline: List[dict] = []
    feature_rows: Dict[str, List[np.ndarray]] = {"bridge": [], "action_expert_final": []}
    frames: List[np.ndarray] = []
    previous_points = None
    acc_ema = None
    steps = 0
    success = False
    done = False
    try:
        while steps < MAX_STEPS[suite] + WAIT_STEPS:
            if steps < WAIT_STEPS:
                obs, _reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                steps += 1
                continue
            if not plan:
                element, frame = policy_input(obs, prompt, resize_size, action_variance_samples)
                frames.append(frame)
                response = client.infer(element)
                chunk = np.asarray(response["actions"], dtype=np.float32)
                if tuple(chunk.shape) != (ACTION_HORIZON, 7):
                    raise ValueError("server did not return the official [10, 7] LIBERO action chunk")
                probes = response.get("failure_features")
                if not isinstance(probes, dict):
                    raise ValueError("instrumented policy response lacks failure_features")
                bridge = np.asarray(probes["bridge"], dtype=np.float32)
                final = np.asarray(probes["action_expert_final"], dtype=np.float32)
                if bridge.ndim != 2 or bridge.shape[0] != 1 or final.ndim != 2 or final.shape[0] != ACTION_HORIZON:
                    raise ValueError("feature/action shape mismatch from instrumented policy")
                points = absolute_eef_points(chunk[:ACTION_HORIZON], np.asarray(obs["robot0_eef_pos"], dtype=np.float64))
                acc_raw = acc_value = stac = None
                if previous_points is not None:
                    acc_raw, acc_ema = velocity_normalized_acc(
                        previous_points, points, execute_horizon=REPLAN_STEPS, previous_ema=acc_ema
                    )
                    acc_value = acc_ema
                    stac = single_sample_overlap(previous_points, points, execute_horizon=REPLAN_STEPS)
                action_total_var = response.get("failure_probe", {}).get("action_total_variance")
                if action_variance_samples:
                    if action_total_var is None or not math.isfinite(float(action_total_var)):
                        raise ValueError("server did not return a finite requested action total variance")
                timeline.append({
                    "decision_index": len(timeline), "env_step": steps, "eef_position": np.asarray(obs["robot0_eef_pos"]).tolist(),
                    "action_chunk": chunk.tolist(), "acc_raw": acc_raw, "acc": acc_value, "stac_single": stac,
                    "policy_ms": float(response.get("policy_timing", {}).get("infer_ms", float("nan"))),
                    "feature_ms": float(response.get("failure_probe", {}).get("feature_ms", float("nan"))),
                    "probe": response.get("failure_probe", {}),
                    "action_total_variance": action_total_var,
                })
                feature_rows["bridge"].append(bridge)
                feature_rows["action_expert_final"].append(final)
                previous_points = points
                plan.extend(chunk[:REPLAN_STEPS])
            action = np.asarray(plan.popleft(), dtype=np.float32)
            obs, _reward, done, info = env.step(action.tolist())
            steps += 1
            success = bool(info.get("success", done))
            if done:
                break
    finally:
        env.close()

    if not timeline:
        raise RuntimeError("rollout did not reach a policy decision")
    episode = {
        "suite": suite, "source": source, "category": category, "configuration_id": configuration_id,
        "task_index": task_index, "task_instruction": prompt, "seed": seed, "success": success,
        "done": bool(done), "steps": steps, "execute_horizon": REPLAN_STEPS,
        "action_horizon": ACTION_HORIZON, "timeline": timeline,
    }
    write_rollout(output_dir, episode=episode, features=feature_rows)
    video_path = output_dir / "rollout.mp4"
    imageio.mimwrite(video_path, frames, fps=10)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_10", choices=sorted(MAX_STEPS))
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--source", choices=("clean", "libero_plus"), required=True)
    parser.add_argument("--category")
    parser.add_argument("--configuration-id", type=int)
    parser.add_argument("--action-variance-samples", type=int, default=0)
    args = parser.parse_args()
    task_suite = benchmark.get_benchmark_dict()[args.suite]()
    if not 0 <= args.task_index < task_suite.n_tasks:
        raise ValueError("task-index is outside this installed benchmark suite")
    run_episode(
        client=WebsocketClientPolicy(args.host, args.port), task=task_suite.get_task(args.task_index),
        task_index=args.task_index, seed=args.seed, output_dir=args.output_dir, suite=args.suite,
        source=args.source, category=args.category, configuration_id=args.configuration_id,
        resize_size=args.resize_size, action_variance_samples=args.action_variance_samples,
    )


if __name__ == "__main__":
    main()
