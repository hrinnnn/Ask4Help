#!/usr/bin/env python3
"""Single-GPU StackCube VLA-FAIL rollout, calibration, and evaluation.

This is intentionally a passive failure detector.  It never changes policy
control or queries an expert: it records the exact LLMD, ACC, and fused alarm
that VLA-FAIL would emit while a normal π0.5 policy executes receding chunks.
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

from rlinf.algorithms.vla_fail import (  # noqa: E402
    LLMDStatistics,
    constant_split_conformal_threshold,
    failure_alert,
    llmd_score,
    velocity_normalized_acc,
)
from rlinf.envs.maniskill.stack_cube_variants import reset_metadata  # noqa: E402
from toolkits.lerobot.collect_maniskill_peg_lerobot_joint import (  # noqa: E402
    MAIN_CAMERA_CANDIDATES,
    WRIST_CAMERA_CANDIDATES,
    _build_frames,
    _extract_record,
    _select_camera,
)
from toolkits.lerobot.collect_maniskill_plug_lerobot_joint import (  # noqa: E402
    write_episode_video_durably,
)
from tools.maniskill_pi05_vfd_online_awbc import (  # noqa: E402
    _action_chunk,
    _bool,
    _build_env,
    _load_model,
    _wrap_obs,
)


class PandaEndEffectorProjector:
    """Kinematically map a joint-delta chunk to absolute EEF positions.

    ACC is defined on absolute end-effector positions in VLA-FAIL.  This class
    intentionally updates only the Panda's seven arm joints; the gripper
    command cannot move the hand frame and must not leak joint units into ACC.
    """

    def __init__(self, env: Any, *, requested_link: str | None) -> None:
        robot = env.unwrapped.agent.robot
        self._pinocchio = robot.create_pinocchio_model()
        names = [link.name for link in robot.get_links()]
        candidates = [requested_link, getattr(env.unwrapped.agent.tcp, "name", None), "panda_hand", "panda_link7"]
        self.link_name = next((name for name in candidates if name in names), None)
        if self.link_name is None:
            raise RuntimeError(
                "Could not resolve the Panda end-effector link for ACC. "
                f"available={names}; pass --eef-link-name explicitly."
            )
        self._link_index = names.index(self.link_name)

    def project(self, qpos: Any, action_chunk: np.ndarray) -> torch.Tensor:
        joints = np.asarray(qpos, dtype=np.float64).reshape(-1).copy()
        if joints.shape[0] < 7:
            raise ValueError(f"Panda qpos must include seven arm joints, got {joints.shape}")
        if action_chunk.ndim != 2 or action_chunk.shape[1] < 7:
            raise ValueError(f"expected pd_joint_delta_pos chunk [H,>=7], got {action_chunk.shape}")
        points = []
        for action in action_chunk:
            joints[:7] += action[:7]
            self._pinocchio.compute_forward_kinematics(joints)
            pose = self._pinocchio.get_link_pose(self._link_index)
            points.append(np.asarray(pose.p, dtype=np.float64).copy())
        return torch.from_numpy(np.stack(points))


def _load_statistics(path: Path) -> tuple[LLMDStatistics, torch.Tensor, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "vla_fail_llmd_statistics_v1":
        raise ValueError(f"Not a VLA-FAIL statistics file: {path}")
    return LLMDStatistics.from_state_dict(payload["statistics"]), payload["fixed_prior"], payload


def _load_thresholds(path: Path | None) -> dict[str, float] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"llmd_threshold", "acc_threshold", "delta"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Threshold file missing {sorted(missing)}")
    return {key: float(payload[key]) for key in required}


def _episode_metrics(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [episode for episode in episodes if episode["success"]]
    failures = [episode for episode in episodes if not episode["success"]]
    return {
        "episodes": len(episodes),
        "successes": len(successes),
        "failures": len(failures),
        "episode_false_positive_rate": float(np.mean([row["alarm"] for row in successes])) if successes else None,
        "episode_failure_recall": float(np.mean([row["alarm"] for row in failures])) if failures else None,
        "first_alarm_chunk": {
            str(row["seed"]): row["first_alarm_chunk"] for row in episodes if row["first_alarm_chunk"] is not None
        },
    }


def _run_episode(
    *,
    env: Any,
    model: Any,
    statistics: LLMDStatistics,
    prior: torch.Tensor,
    seed: int,
    execute_horizon: int,
    max_episode_steps: int,
    projector: PandaEndEffectorProjector,
    thresholds: dict[str, float] | None,
    min_velocity: float,
    ema_alpha: float,
) -> tuple[dict[str, Any], list[Any], list[np.ndarray]]:
    raw_obs, info = env.reset(seed=seed)
    metadata = reset_metadata(env, split="id" if "ID" in env.spec.id else "ood")
    records = [_extract_record(raw_obs)]
    executed_actions: list[np.ndarray] = []
    timeline: list[dict[str, Any]] = []
    previous_points = None
    previous_acc_ema = None
    success = False
    while len(executed_actions) < max_episode_steps and not success:
        env_obs = _wrap_obs(raw_obs, info, task="stack")
        with torch.inference_mode():
            features = model.extract_llmd_action_features(env_obs, prior)
            current_llmd = float(llmd_score(features, statistics)[0].item())
            predicted, _ = model.predict_action_batch(
                env_obs=env_obs, mode="eval", compute_values=False
            )
        full_chunk = _action_chunk(predicted, int(model.config.action_horizon))
        low = np.asarray(env.action_space.low).reshape(-1)
        high = np.asarray(env.action_space.high).reshape(-1)
        full_chunk = np.clip(full_chunk, low, high).astype(np.float32)
        qpos = raw_obs["agent"]["qpos"].detach().cpu().numpy()[0]
        current_points = projector.project(qpos, full_chunk)
        acc_raw = acc_ema = None
        if previous_points is not None:
            acc_raw, acc_ema = velocity_normalized_acc(
                previous_points,
                current_points,
                execute_horizon=execute_horizon,
                min_velocity=min_velocity,
                previous_ema=previous_acc_ema,
                ema_alpha=ema_alpha,
            )
        previous_points = current_points
        previous_acc_ema = acc_ema
        alarm = (
            failure_alert(
                llmd_value=current_llmd,
                llmd_threshold=thresholds["llmd_threshold"],
                acc_value=acc_ema,
                acc_threshold=thresholds["acc_threshold"],
            )
            if thresholds is not None
            else False
        )
        chunk = full_chunk[:execute_horizon]
        timeline.append(
            {
                "chunk_index": len(timeline),
                "env_step": len(executed_actions),
                "llmd": current_llmd,
                "acc_raw": acc_raw,
                "acc_ema": acc_ema,
                "alarm": alarm,
            }
        )
        for action in chunk:
            raw_obs, _reward, terminated, truncated, info = env.step(
                torch.as_tensor(action, device=env.unwrapped.device).unsqueeze(0)
            )
            executed_actions.append(action)
            records.append(_extract_record(raw_obs))
            success = _bool(info.get("success", False))
            if success or _bool(terminated) or _bool(truncated):
                break
    return (
        {
            "seed": seed,
            "success": success,
            "steps": len(executed_actions),
            "alarm": any(chunk["alarm"] for chunk in timeline),
            "first_alarm_chunk": next((chunk["chunk_index"] for chunk in timeline if chunk["alarm"]), None),
            "timeline": timeline,
            **metadata,
        },
        records,
        executed_actions,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pi05-base", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--llmd-statistics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("id", "ood"), required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--execute-horizon", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument(
        "--target-successes",
        type=int,
        default=20,
        help="Successful held-out ID rollouts required for strict calibration.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=200,
        help="Maximum calibration seeds to attempt while collecting target successes.",
    )
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--min-velocity", type=float, default=1e-3)
    parser.add_argument("--ema-alpha", type=float, default=0.9)
    parser.add_argument("--eef-link-name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.calibrate and args.thresholds is not None:
        raise ValueError("--calibrate and --thresholds are mutually exclusive")
    if args.calibrate and args.split != "id":
        raise ValueError("VLA-FAIL calibration must use successful ID rollouts")
    if args.episodes < 1 or args.target_successes < 1 or args.max_attempts < 1:
        raise ValueError("episodes must be positive")
    if args.calibrate and args.target_successes > args.max_attempts:
        raise ValueError("target-successes cannot exceed max-attempts")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    statistics, prior, statistics_payload = _load_statistics(args.llmd_statistics)
    model = _load_model(args.checkpoint, args.norm_stats, args.pi05_base)
    env = _build_env(args.max_episode_steps, task="stack", split=args.split)
    projector = PandaEndEffectorProjector(env, requested_link=args.eef_link_name)
    thresholds = _load_thresholds(args.thresholds)
    episodes = []
    try:
        attempts = args.max_attempts if args.calibrate else args.episodes
        for index in range(attempts):
            episode, records, actions = _run_episode(
                env=env,
                model=model,
                statistics=statistics,
                prior=prior.to("cuda"),
                seed=args.seed + index,
                execute_horizon=args.execute_horizon,
                max_episode_steps=args.max_episode_steps,
                projector=projector,
                thresholds=thresholds,
                min_velocity=args.min_velocity,
                ema_alpha=args.ema_alpha,
            )
            main_camera = _select_camera(records[0].obs, "", ("base_camera",) + MAIN_CAMERA_CANDIDATES, "main")
            wrist_camera = _select_camera(records[0].obs, "", ("hand_camera",) + WRIST_CAMERA_CANDIDATES, "wrist")
            frames = _build_frames(
                records=records,
                actions=actions,
                task="stack the red cube on the green cube",
                main_camera=main_camera,
                wrist_camera=wrist_camera,
            )
            write_episode_video_durably(frames, args.output_dir / "videos", index, args.seed + index, fps=10)
            episodes.append(episode)
            print(
                f"[vla-fail] {index + 1}/{args.episodes} seed={episode['seed']} "
                f"success={int(episode['success'])} alarm={int(episode['alarm'])}",
                flush=True,
            )
            if args.calibrate and sum(row["success"] for row in episodes) >= args.target_successes:
                break
    finally:
        env.close()
        del model
        torch.cuda.empty_cache()
    output = {
        "format": "stackcube_vla_fail_rollout_v1",
        "protocol": {
            "llmd": "final Action Expert pre-projection features; fixed Gaussian prior at pi0.5 t=1; max action-token score",
            "acc": "velocity-normalized EEF overlap MAE with EMA",
            "execute_horizon": args.execute_horizon,
            "action_horizon": int(prior.shape[1]),
            "fixed_prior_seed": statistics_payload["fixed_prior_seed"],
            "eef_link_name": projector.link_name,
        },
        "thresholds": thresholds,
        "metrics": _episode_metrics(episodes),
        "episodes": episodes,
    }
    if args.calibrate:
        successful = [row for row in episodes if row["success"]][: args.target_successes]
        successful_llmd = [[chunk["llmd"] for chunk in row["timeline"]] for row in successful]
        successful_acc = [
            [chunk["acc_ema"] for chunk in row["timeline"] if chunk["acc_ema"] is not None]
            for row in successful
            if any(chunk["acc_ema"] is not None for chunk in row["timeline"])
        ]
        if len(successful_llmd) != args.target_successes or len(successful_acc) != args.target_successes:
            raise RuntimeError(
                "Strict calibration requires the requested number of successful ID rollouts with ACC overlap; "
                f"got LLMD={len(successful_llmd)}/{args.target_successes}, "
                f"ACC={len(successful_acc)}/{args.target_successes} after {len(episodes)} attempts."
            )
        thresholds = {
            "delta": args.delta,
            "target_successes": args.target_successes,
            "attempts": len(episodes),
            "llmd_threshold": constant_split_conformal_threshold(successful_llmd, delta=args.delta),
            "acc_threshold": constant_split_conformal_threshold(successful_acc, delta=args.delta),
        }
        output["thresholds"] = thresholds
        (args.output_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2) + "\n")
    (args.output_dir / "episodes.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["metrics"], indent=2))


if __name__ == "__main__":
    main()
