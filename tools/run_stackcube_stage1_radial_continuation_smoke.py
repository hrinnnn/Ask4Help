#!/usr/bin/env python3
"""Run the radial StackCube OOD policy-to-expert-to-policy smoke.

The detector is calibrated from independent successful ID rollouts.  The
expert is allowed to control only until a stable red lift is observed; the
policy then resumes on the same simulator state.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "RLinf")]

from rlinf.envs.maniskill.stack_cube_privileged_oracle import (  # noqa: E402
    StackCubePrivilegedChunkOracle,
)
from tools.collect_stackcube_xvla_dagger import (  # noqa: E402
    MAIN_CAMERA_CANDIDATES,
    WRIST_CAMERA_CANDIDATES,
    _build_frames,
    _extract_record,
    _select_camera,
)
from tools.evaluate_stackcube_xvla import bool_scalar, clip_action_chunk  # noqa: E402
from tools.stackcube_stage1_radial_ood import (  # noqa: E402
    STACK_CUBE_RADIAL_OOD_SPLIT,
    STACK_CUBE_TASK,
    register_stackcube_stage1_radial_variants,
    radial_env_id,
    stage1_radial_reset_metadata,
    stage1_radial_state_record,
)
from tools.xvla_airplane_failure_detection import XVLAMultilayerProbe  # noqa: E402
from tools.xvla_airplane_runtime import XVLAAirplanePolicy  # noqa: E402
from toolkits.lerobot.collect_maniskill_plug_lerobot_joint import (  # noqa: E402
    write_episode_video_durably,
)


CHUNK_SIZE = 5
MAX_EPISODE_STEPS = 100
UPPER_QUANTILE = 0.95
LOWER_QUANTILE = 0.75
STABLE_LIFT_DECISIONS = 2
MIN_EXPERT_DWELL_CHUNKS = 1


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def make_env(split: str) -> Any:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    return gym.make(
        radial_env_id(split),
        robot_uids="panda_wristcam",
        num_envs=1,
        obs_mode="rgb",
        control_mode="pd_joint_delta_pos",
        reward_mode="sparse",
        render_mode="rgb_array",
        sim_backend="physx_cpu",
        sim_config={"sim_freq": 100, "control_freq": 10},
        sensor_configs={"width": 384, "height": 384},
        max_episode_steps=MAX_EPISODE_STEPS,
    )


class PCAResidual:
    def __init__(self, asset_path: Path, layer: str = "vlm_action_bridge") -> None:
        payload = torch.load(asset_path, map_location="cpu", weights_only=False)
        source = payload["layers"][layer]
        self.mean = source["mean"].float()
        self.eigenvectors = source["eigenvectors"].float()
        self.pca_dim = int(source["pca_dim"])
        self.layer = layer

    def score(self, feature: torch.Tensor) -> float:
        centered = feature.float().cpu() - self.mean
        coordinates = centered @ self.eigenvectors
        residual_dims = self.eigenvectors.shape[1] - self.pca_dim
        return float(coordinates[:, :residual_dims].norm(dim=-1)[0].item())


def policy_decision(
    policy: XVLAAirplanePolicy,
    probe: XVLAMultilayerProbe,
    raw_obs: dict[str, Any],
    *,
    seed: int,
    flow_steps: int,
    detector: PCAResidual,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> tuple[np.ndarray, float]:
    torch.manual_seed(int(seed))
    inputs = policy.prepare(raw_obs, STACK_CUBE_TASK)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        features, encoding = probe.extract(inputs)
        generated = policy._generate_from_encoding(inputs, encoding, steps=flow_steps)
    score = detector.score(features[detector.layer])
    return (
        clip_action_chunk(
            generated.float().cpu().numpy(), action_low, action_high, CHUNK_SIZE
        ),
        score,
    )


def save_video(
    records: list[Any],
    actions: list[np.ndarray],
    output: Path,
    index: int,
    seed: int,
) -> str:
    main_camera = _select_camera(
        records[0].obs, "", ("base_camera",) + MAIN_CAMERA_CANDIDATES, "main"
    )
    wrist_camera = _select_camera(
        records[0].obs, "", ("hand_camera",) + WRIST_CAMERA_CANDIDATES, "wrist"
    )
    frames = _build_frames(
        records=records,
        actions=actions,
        task=STACK_CUBE_TASK,
        main_camera=main_camera,
        wrist_camera=wrist_camera,
    )
    path = write_episode_video_durably(
        frames,
        video_dir=output / "videos",
        episode_index=index,
        seed=seed,
        fps=10,
    )
    return str(path)


def calibrate(
    policy: XVLAAirplanePolicy,
    probe: XVLAMultilayerProbe,
    detector: PCAResidual,
    *,
    seed_start: int,
    episodes: int,
    min_successes: int,
    flow_steps: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
    output: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index in range(episodes):
        seed = seed_start + index
        env = make_env("id")
        try:
            raw_obs, _ = env.reset(seed=seed)
            scores: list[float] = []
            actions: list[np.ndarray] = []
            success = False
            terminated = truncated = False
            decision = 0
            while len(actions) < MAX_EPISODE_STEPS and not success:
                chunk, score = policy_decision(
                    policy,
                    probe,
                    raw_obs,
                    seed=seed * 1000 + decision,
                    flow_steps=flow_steps,
                    detector=detector,
                    action_low=action_low,
                    action_high=action_high,
                )
                scores.append(score)
                decision += 1
                for action in chunk:
                    raw_obs, _, terminated, truncated, info = env.step(
                        torch.as_tensor(action, device=env.unwrapped.device).unsqueeze(0)
                    )
                    actions.append(np.asarray(action, dtype=np.float32))
                    success = bool_scalar(info.get("success", False))
                    if success or bool_scalar(terminated) or bool_scalar(truncated):
                        break
            rows.append(
                {
                    "episode_index": index,
                    "seed": seed,
                    "split": "id",
                    "success": bool(success),
                    "steps": len(actions),
                    "decision_scores": scores,
                    "max_score": max(scores) if scores else None,
                    "metadata": stage1_radial_reset_metadata(env, split="id", paired_seed=seed),
                }
            )
        finally:
            env.close()
        append_jsonl(output / "episodes.jsonl", rows[-1])

    successful = [row for row in rows if row["success"]]
    if len(successful) < min_successes:
        result = {
            "status": "CALIBRATION_FAILED",
            "attempts": len(rows),
            "successful_id_trajectories": len(successful),
            "required_successful_id_trajectories": min_successes,
            "rows": rows,
        }
        write_json(output / "summary.json", result)
        (output / "CALIBRATION_FAILED").write_text(
            "Independent successful-ID calibration did not reach the frozen minimum.\n",
            encoding="utf-8",
        )
        raise RuntimeError("independent ID calibration failed")

    maxima = sorted(float(row["max_score"]) for row in successful)
    decision_scores = [score for row in successful for score in row["decision_scores"]]
    rank = min(len(maxima), int(np.ceil((len(maxima) + 1) * UPPER_QUANTILE)))
    thresholds = {
        "status": "CALIBRATION_COMPLETE",
        "seed_manifest": list(range(seed_start, seed_start + episodes)),
        "successful_seed_selection": [row["seed"] for row in successful],
        "successful_id_trajectories": len(successful),
        "upper_threshold": float(maxima[rank - 1]),
        "upper_threshold_definition": "q=0.95 over successful-ID trajectory maxima",
        "lower_threshold": float(np.quantile(np.asarray(decision_scores), LOWER_QUANTILE)),
        "lower_threshold_definition": "q=0.75 over successful-ID decision scores",
        "upper_quantile": UPPER_QUANTILE,
        "lower_quantile": LOWER_QUANTILE,
        "stable_lift_decisions": STABLE_LIFT_DECISIONS,
        "minimum_expert_dwell_chunks": MIN_EXPERT_DWELL_CHUNKS,
        "rows": rows,
    }
    write_json(output / "thresholds.json", thresholds)
    write_json(output / "summary.json", thresholds)
    (output / "CALIBRATION_COMPLETE").write_text(
        "Successful-ID thresholds frozen for radial continuation smoke.\n",
        encoding="utf-8",
    )
    return thresholds


def run_continuation_episode(
    policy: XVLAAirplanePolicy,
    probe: XVLAMultilayerProbe,
    detector: PCAResidual,
    thresholds: dict[str, Any],
    *,
    seed: int,
    flow_steps: int,
    output: Path,
    episode_index: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> dict[str, Any]:
    env = make_env(STACK_CUBE_RADIAL_OOD_SPLIT)
    try:
        raw_obs, _ = env.reset(seed=seed)
        records = [_extract_record(raw_obs)]
        actions: list[np.ndarray] = []
        sources: list[str] = []
        state_timeline: list[dict[str, Any]] = [stage1_radial_state_record(env)]
        score_timeline: list[dict[str, Any]] = []
        state = "policy"
        oracle: StackCubePrivilegedChunkOracle | None = None
        expert_dwell = 0
        stable_lift = 0
        takeovers = 0
        returns = 0
        false_release = False
        success = False
        terminated = truncated = False
        decision = 0

        while len(actions) < MAX_EPISODE_STEPS and not success:
            policy_chunk, score = policy_decision(
                policy,
                probe,
                raw_obs,
                seed=seed * 1000 + decision,
                flow_steps=flow_steps,
                detector=detector,
                action_low=action_low,
                action_high=action_high,
            )
            before = state
            event: str | None = None
            if state == "policy" and score > float(thresholds["upper_threshold"]):
                state = "expert"
                takeovers += 1
                expert_dwell = 0
                stable_lift = 0
                oracle = StackCubePrivilegedChunkOracle(chunk_size=CHUNK_SIZE)
                oracle.initialize_from_state(env)
                event = "policy_to_expert"
            if state == "expert":
                assert oracle is not None
                plan = oracle.plan(env)
                candidate = np.asarray(
                    [
                        plan.action_at(raw_obs["agent"]["qpos"], index)
                        for index in range(CHUNK_SIZE)
                    ],
                    dtype=np.float32,
                )
                expert_dwell += 1
                controller = "expert"
                oracle_phase = plan.phase
                oracle_valid = plan.planning_succeeded
            else:
                candidate = policy_chunk
                controller = "policy"
                oracle_phase = None
                oracle_valid = None

            score_timeline.append(
                {
                    "decision_index": decision,
                    "env_step": len(actions),
                    "score": score,
                    "controller_before": before,
                    "controller_after": state,
                    "controller": controller,
                    "event": event,
                    "upper_threshold": thresholds["upper_threshold"],
                    "lower_threshold": thresholds["lower_threshold"],
                    "expert_dwell_chunks": expert_dwell,
                    "oracle_phase": oracle_phase,
                    "oracle_valid": oracle_valid,
                }
            )

            for action in candidate:
                raw_obs, _, terminated, truncated, info = env.step(
                    torch.as_tensor(action, device=env.unwrapped.device).unsqueeze(0)
                )
                actions.append(np.asarray(action, dtype=np.float32))
                sources.append(controller)
                records.append(_extract_record(raw_obs))
                current = stage1_radial_state_record(env, info)
                state_timeline.append(current)
                predicates = current["predicates"]
                success = bool_scalar(info.get("success", False))
                if state == "expert":
                    if predicates["red_grasped"] and predicates["red_lifted"]:
                        stable_lift += 1
                    else:
                        stable_lift = 0
                if state == "policy" and returns > 0:
                    if (
                        not predicates["red_grasped"]
                        and not predicates["red_placed"]
                        and not success
                    ):
                        false_release = True
                if success or bool_scalar(terminated) or bool_scalar(truncated):
                    break

            if (
                state == "expert"
                and expert_dwell >= MIN_EXPERT_DWELL_CHUNKS
                and stable_lift >= STABLE_LIFT_DECISIONS
            ):
                state = "policy"
                returns += 1
                event = "expert_to_policy"
                score_timeline[-1]["return_after_chunk"] = True
            decision += 1

        actions_path = output / "actions" / f"episode_{episode_index:03d}_seed_{seed}.npy"
        actions_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(actions_path, np.asarray(actions, dtype=np.float32))
        row = {
            "episode_index": episode_index,
            "seed": seed,
            "split": STACK_CUBE_RADIAL_OOD_SPLIT,
            "success": bool(success),
            "continuation_success": bool(success and returns > 0),
            "takeover_count": takeovers,
            "return_to_policy_count": returns,
            "real_return_to_policy": bool(returns > 0),
            "false_release": bool(false_release),
            "expert_action_steps": int(sum(source == "expert" for source in sources)),
            "policy_action_steps": int(sum(source == "policy" for source in sources)),
            "expert_action_ratio": float(
                sum(source == "expert" for source in sources) / max(1, len(actions))
            ),
            "steps": len(actions),
            "score_timeline": score_timeline,
            "state_timeline": state_timeline,
            "metadata": stage1_radial_reset_metadata(
                env, split=STACK_CUBE_RADIAL_OOD_SPLIT, paired_seed=seed
            ),
            "failure_reason": (
                None
                if success
                else "false_release"
                if false_release
                else "episode_not_successful"
            ),
            "actions": str(actions_path),
        }
        row["video"] = save_video(records, actions, output, episode_index, seed)
        return row
    finally:
        env.close()


def save_video(
    records: list[Any],
    actions: list[np.ndarray],
    output: Path,
    index: int,
    seed: int,
) -> str:
    main_camera = _select_camera(
        records[0].obs, "", ("base_camera",) + MAIN_CAMERA_CANDIDATES, "main"
    )
    wrist_camera = _select_camera(
        records[0].obs, "", ("hand_camera",) + WRIST_CAMERA_CANDIDATES, "wrist"
    )
    frames = _build_frames(
        records=records,
        actions=actions,
        task=STACK_CUBE_TASK,
        main_camera=main_camera,
        wrist_camera=wrist_camera,
    )
    path = write_episode_video_durably(
        frames,
        video_dir=output / "videos",
        episode_index=index,
        seed=seed,
        fps=10,
    )
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--internal-assets", type=Path, required=True)
    parser.add_argument("--local-output", type=Path, required=True)
    parser.add_argument("--persistent-output", type=Path, required=True)
    parser.add_argument("--calibration-seed-start", type=int, default=973000)
    parser.add_argument("--calibration-episodes", type=int, default=50)
    parser.add_argument("--calibration-min-successes", type=int, default=25)
    parser.add_argument("--smoke-seed-start", type=int, default=972000)
    parser.add_argument("--smoke-episodes", type=int, default=20)
    parser.add_argument("--flow-steps", type=int, default=10)
    args = parser.parse_args()
    if args.local_output.exists() or args.persistent_output.exists():
        raise FileExistsError("radial continuation output roots must be new")
    args.local_output.mkdir(parents=True)
    write_json(
        args.local_output / "protocol.json",
        {
            "task": "StackCube Stage1 radial-distance policy-to-expert-to-policy smoke",
            "checkpoint": str(args.checkpoint),
            "internal_assets": str(args.internal_assets),
            "ood_split": STACK_CUBE_RADIAL_OOD_SPLIT,
            "red_ood_rule": "red_id + 0.04 * normalize(red_id - green)",
            "calibration_seed_start": args.calibration_seed_start,
            "calibration_episodes": args.calibration_episodes,
            "calibration_min_successes": args.calibration_min_successes,
            "smoke_seed_start": args.smoke_seed_start,
            "smoke_episodes": args.smoke_episodes,
            "stable_lift_decisions": STABLE_LIFT_DECISIONS,
            "minimum_expert_dwell_chunks": MIN_EXPERT_DWELL_CHUNKS,
            "formal_ood_evaluation_started": False,
            "downstream_training_started": False,
        },
    )
    register_stackcube_stage1_radial_variants()
    policy = XVLAAirplanePolicy(args.checkpoint, args.xvla_root)
    policy.checkpoint = args.checkpoint
    probe = XVLAMultilayerProbe(policy.model, probe_seed=0, probe_steps=5)
    detector = PCAResidual(args.internal_assets)
    calibration_env = make_env("id")
    action_low = np.asarray(calibration_env.action_space.low, dtype=np.float32).reshape(-1)
    action_high = np.asarray(calibration_env.action_space.high, dtype=np.float32).reshape(-1)
    calibration_env.close()
    try:
        calibration_output = args.local_output / "calibration"
        calibration_output.mkdir()
        thresholds = calibrate(
            policy,
            probe,
            detector,
            seed_start=args.calibration_seed_start,
            episodes=args.calibration_episodes,
            min_successes=args.calibration_min_successes,
            flow_steps=args.flow_steps,
            action_low=action_low,
            action_high=action_high,
            output=calibration_output,
        )
        smoke_output = args.local_output / "continuation_smoke"
        smoke_output.mkdir()
        rows = []
        for index in range(args.smoke_episodes):
            row = run_continuation_episode(
                policy,
                probe,
                detector,
                thresholds,
                seed=args.smoke_seed_start + index,
                flow_steps=args.flow_steps,
                output=smoke_output,
                episode_index=index,
                action_low=action_low,
                action_high=action_high,
            )
            rows.append(row)
            append_jsonl(smoke_output / "episodes.jsonl", row)
            print(
                f"[radial-continuation] {index + 1}/{args.smoke_episodes} "
                f"seed={row['seed']} takeover={row['takeover_count']} "
                f"return={int(row['real_return_to_policy'])} "
                f"success={int(row['continuation_success'])}",
                flush=True,
            )
        summary = {
            "episodes": len(rows),
            "successes": sum(int(row["success"]) for row in rows),
            "continuation_successes": sum(int(row["continuation_success"]) for row in rows),
            "takeover_episodes": sum(int(row["takeover_count"] > 0) for row in rows),
            "return_to_policy_episodes": sum(int(row["real_return_to_policy"]) for row in rows),
            "false_release_episodes": sum(int(row["false_release"]) for row in rows),
            "mean_expert_action_ratio": float(np.mean([row["expert_action_ratio"] for row in rows])),
            "video_count": sum(int(bool(row.get("video"))) for row in rows),
            "action_count": sum(int(bool(row.get("actions"))) for row in rows),
            "thresholds": thresholds,
            "rows": rows,
        }
        write_json(smoke_output / "summary.json", summary)
        passed = (
            len(rows) == args.smoke_episodes
            and summary["continuation_successes"] >= 16
            and summary["return_to_policy_episodes"] > 0
            and summary["video_count"] == args.smoke_episodes
            and summary["action_count"] == args.smoke_episodes
        )
        marker = "CONTINUATION_SMOKE_PASSED" if passed else "CONTINUATION_SMOKE_FAILED"
        (args.local_output / marker).write_text(
            "Radial policy-to-expert-to-policy continuation decision.\n",
            encoding="utf-8",
        )
        if not passed:
            (args.local_output / "DIAGNOSTIC_STOPPED_CONTINUATION_FAILED").write_text(
                "Continuation smoke did not meet the pre-registered gate.\n",
                encoding="utf-8",
            )
    finally:
        probe.close()
        del policy
        torch.cuda.empty_cache()
    if args.persistent_output.exists():
        raise FileExistsError("refusing to overwrite persistent output")
    args.persistent_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(args.local_output, args.persistent_output)
    if not (args.local_output / "CONTINUATION_SMOKE_PASSED").exists():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
