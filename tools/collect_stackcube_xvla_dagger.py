#!/usr/bin/env python3
"""Collect four fair X-VLA StackCube DAgger/BC datasets."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "RLinf")]

from rlinf.envs.maniskill.stack_cube_privileged_oracle import (  # noqa: E402
    StackCubePrivilegedChunkOracle,
)
from rlinf.envs.maniskill.stack_cube_variants import STACK_CUBE_TASK  # noqa: E402
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
from tools.evaluate_stackcube_xvla import bool_scalar, clip_action_chunk  # noqa: E402
from tools.xvla_airplane_runtime import XVLAAirplanePolicy  # noqa: E402
from tools.xvla_airplane_failure_detection import XVLAMultilayerProbe  # noqa: E402
from tools.stackcube_stage2_ood import (  # noqa: E402
    STACK_CUBE_STAGE2_OOD_SPLIT,
    register_stack_cube_splits,
    stack_cube_env_id,
    stack_cube_reset_metadata,
)


METHODS = (
    "internal_pca",
    "input_pca",
    "bridge_pca",
    "action_pca",
    "vlm_bridge_pca",
    "offline_oracle",
    "failure_recovery",
    "diffdagger",
)
INTERNAL_PCA_METHODS = {"internal_pca", "input_pca", "action_pca"}
TIMING_CONDITIONS = (
    "immediate",
    "post_grasp",
    "post_lift",
    "failure_recovery",
    "fixed_timing",
)
EXECUTE_HORIZON = 5
TASK_HORIZON = 150
FAILURE_RECOVERY_STEP = 50
FAILURE_STALL_STEPS = 30
FAILURE_TIMEOUT_STEP = 75


@dataclass(frozen=True)
class InternalPCA:
    mean: torch.Tensor
    eigenvectors: torch.Tensor
    pca_dim: int

    @classmethod
    def load(
        cls, path: Path, device: torch.device, *, layer: str = "vlm_action_bridge"
    ) -> "InternalPCA":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        source = payload["layers"][layer]
        return cls(
            mean=source["mean"].to(device=device, dtype=torch.float32),
            eigenvectors=source["eigenvectors"].to(device=device, dtype=torch.float32),
            pca_dim=int(source["pca_dim"]),
        )

    @torch.inference_mode()
    def score(self, feature: torch.Tensor) -> float:
        centered = feature.to(self.mean.device, torch.float32) - self.mean
        coordinates = centered @ self.eigenvectors
        residual_dims = self.eigenvectors.shape[1] - self.pca_dim
        return float(coordinates[:, :residual_dims].norm(dim=-1)[0].item())


@dataclass
class FailureRecoveryState:
    ever_grasped: bool = False
    ever_lifted: bool = False
    best_stage: int = 0
    last_progress_step: int = 0
    reason: str | None = None
    currently_grasped: bool = False
    on_cube: bool = False
    cube_z: float = 0.0
    dropped_decision_boundaries: int = 0

    def update(
        self,
        *,
        env_step: int,
        currently_grasped: bool,
        on_cube: bool,
        success: bool,
        cube_z: float,
    ) -> None:
        self.currently_grasped = currently_grasped
        self.on_cube = on_cube
        self.cube_z = cube_z
        self.ever_grasped |= currently_grasped
        self.ever_lifted |= currently_grasped and cube_z >= 0.07
        stage = 3 if on_cube else 2 if cube_z >= 0.07 else 1 if currently_grasped else 0
        if stage > self.best_stage:
            self.best_stage = stage
            self.last_progress_step = env_step
        if success or self.reason is not None:
            return
        if self.ever_grasped and not currently_grasped and not on_cube and cube_z < 0.06:
            self.reason = "dropped_after_grasp"
        elif self.ever_grasped and env_step - self.last_progress_step >= FAILURE_STALL_STEPS:
            self.reason = "stalled_after_progress"
        elif env_step >= FAILURE_TIMEOUT_STEP:
            self.reason = "episode_timeout"

    def timing_trigger(self, condition: str) -> str | None:
        """Evaluate a controlled takeover condition at an action-chunk boundary."""
        if condition == "post_grasp" and self.currently_grasped:
            return "post_grasp"
        if condition == "post_lift" and self.currently_grasped and self.cube_z >= 0.07:
            return "post_lift"
        if condition == "failure_recovery":
            dropped = (
                self.ever_lifted
                and not self.currently_grasped
                and not self.on_cube
                and self.cube_z < 0.06
            )
            self.dropped_decision_boundaries = (
                self.dropped_decision_boundaries + 1 if dropped else 0
            )
            if self.dropped_decision_boundaries >= 2:
                return "dropped_after_lift_two_boundaries"
        return None


def alternating_split(attempt_index: int, ood_split: str = "ood") -> str:
    if attempt_index < 0:
        raise ValueError("attempt_index must be non-negative")
    if ood_split not in {"ood", STACK_CUBE_STAGE2_OOD_SPLIT}:
        raise ValueError(f"unsupported OOD split: {ood_split}")
    return "id" if attempt_index % 2 == 0 else ood_split


def exact_budget_subset(lengths: list[int], budget: int) -> list[int] | None:
    """Return a deterministic full-episode subset whose action count is exact."""
    reachable: dict[int, tuple[int, ...]] = {0: ()}
    for index, length in enumerate(lengths):
        for total, chosen in list(reachable.items())[::-1]:
            candidate = total + length
            if candidate <= budget and candidate not in reachable:
                reachable[candidate] = (*chosen, index)
        if budget in reachable:
            return list(reachable[budget])
    return None


def collection_complete(
    accepted_episodes: int,
    expert_actions: int,
    *,
    target_episodes: int,
    expert_action_budget: int | None,
) -> bool:
    if expert_action_budget is not None:
        return expert_actions >= expert_action_budget
    return accepted_episodes >= target_episodes


def consecutive_gate(score: float, threshold: float, count: int, patience: int) -> tuple[int, bool]:
    count = count + 1 if score > threshold else 0
    return count, count >= patience


def admitted_suffix(success: bool, expert_start: int | None, action_count: int) -> tuple[int, int] | None:
    """Retain every real expert action; temporal masking handles short tails."""
    if not success or expert_start is None or expert_start >= action_count:
        return None
    return expert_start, action_count


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _make_env(split: str):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    return gym.make(
        stack_cube_env_id(split),
        robot_uids="panda_wristcam",
        num_envs=1,
        obs_mode="rgb",
        control_mode="pd_joint_delta_pos",
        reward_mode="sparse",
        render_mode="rgb_array",
        sim_backend="physx_cpu",
        sim_config={"sim_freq": 100, "control_freq": 10},
        sensor_configs={"width": 384, "height": 384},
        max_episode_steps=TASK_HORIZON,
    )


def task_state(env: Any, *, grasped: bool = False, on_cube: bool = False) -> np.ndarray:
    base = env.unwrapped
    cube = base.cubeA.pose.p.reshape(-1, 3)[0].detach().cpu().numpy()
    target = base.cubeB.pose.p.reshape(-1, 3)[0].detach().cpu().numpy()
    tcp = base.agent.tcp.pose.p.reshape(-1, 3)[0].detach().cpu().numpy()
    qpos = base.agent.robot.get_qpos().reshape(-1).detach().cpu().numpy()
    gripper_width = float(qpos[-2:].sum())
    return np.concatenate(
        [cube, target, tcp, tcp - cube, cube - target,
         np.asarray([gripper_width, float(grasped), float(on_cube)], dtype=np.float32)]
    ).astype(np.float32)


def _run_attempt(
    *,
    method: str,
    split: str,
    seed: int,
    env: Any,
    policy: XVLAAirplanePolicy | None,
    internal_pca: InternalPCA | None,
    internal_layer: str,
    internal_probe: XVLAMultilayerProbe | None,
    pca_threshold: float,
    diff_threshold: float,
    diff_patience: int,
    diff_timesteps: int,
    flow_steps: int,
    failure_recovery_mode: str,
    controlled_timing: bool = False,
    fixed_timing_step: int | None = None,
) -> tuple[list[Any], list[np.ndarray], list[str], int | None, dict[str, Any]]:
    raw_obs, _ = env.reset(seed=seed)
    records = [_extract_record(raw_obs)]
    actions: list[np.ndarray] = []
    sources: list[str] = []
    timeline: list[dict[str, Any]] = []
    task_states = [task_state(env)]
    expert_start = 0 if method in {"offline_oracle", "immediate"} else None
    gate_count = 0
    failure_state = FailureRecoveryState()
    failure_recovery_event = "immediate" if method == "immediate" else None
    oracle = StackCubePrivilegedChunkOracle(chunk_size=EXECUTE_HORIZON)
    oracle_initialized = False
    success = grasped = on_cube = False
    terminated = truncated = False

    while len(actions) < TASK_HORIZON and not (success or bool_scalar(terminated) or bool_scalar(truncated)):
        step = len(actions)
        score = threshold = None
        alarm = expert_start is not None
        timing_reason = None
        policy_seed = seed * 1000 + len(timeline)
        if (
            fixed_timing_step is not None
            and expert_start is None
            and step >= fixed_timing_step
        ):
            expert_start, alarm = step, True
            timing_reason = f"fixed_step_{fixed_timing_step}"
            failure_recovery_event = timing_reason
        if controlled_timing and expert_start is None and method in {
            "post_grasp", "post_lift", "failure_recovery"
        }:
            timing_reason = failure_state.timing_trigger(method)
            if timing_reason is not None:
                expert_start, alarm = step, True
                failure_recovery_event = timing_reason
        if expert_start is None and method == "failure_recovery" and not controlled_timing:
            if failure_recovery_mode == "fixed_step" and step >= FAILURE_RECOVERY_STEP:
                expert_start, alarm = step, True
                failure_recovery_event = "fixed_step"
            elif failure_recovery_mode == "event" and failure_state.reason is not None:
                expert_start, alarm = step, True
                failure_recovery_event = failure_state.reason
        if expert_start is None:
            assert policy is not None
            if method in INTERNAL_PCA_METHODS:
                assert internal_probe is not None
                torch.manual_seed(policy_seed)
                inputs = policy.prepare(raw_obs, STACK_CUBE_TASK)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    features, encoding = internal_probe.extract(inputs)
                    generated = policy._generate_from_encoding(
                        inputs, encoding, steps=flow_steps
                    )
                predicted = generated.float().cpu().numpy()
                feature = features[internal_layer]
            else:
                predicted, feature, inputs, encoding = policy.predict(
                    raw_obs, STACK_CUBE_TASK, seed=policy_seed, steps=flow_steps
                )
            if method in {*INTERNAL_PCA_METHODS, "bridge_pca", "vlm_bridge_pca"}:
                assert internal_pca is not None
                score, threshold = internal_pca.score(feature), pca_threshold
                gate_count, alarm = consecutive_gate(score, threshold, gate_count, 1)
            elif method == "diffdagger":
                score = policy.diffdagger_score(
                    inputs,
                    encoding,
                    predicted,
                    num_timesteps=diff_timesteps,
                    num_noise_samples=1,
                )
                threshold = diff_threshold
                gate_count, alarm = consecutive_gate(
                    score, threshold, gate_count, diff_patience
                )
            if alarm:
                expert_start = step

        if expert_start is not None:
            if not oracle_initialized:
                stable_grasp_event = failure_recovery_event in {"post_grasp", "post_lift"}
                initialized_phase = oracle.initialize_from_state(
                    env,
                    grasped_hint=True if stable_grasp_event else None,
                )
                failure_recovery_event = failure_recovery_event or initialized_phase
                oracle_initialized = True
            plan = oracle.plan(env)
            candidate = plan.actions
            source = "expert"
        else:
            plan = None
            source = "policy"
            low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
            high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
            candidate = clip_action_chunk(predicted, low, high, EXECUTE_HORIZON)

        timeline.append(
            {
                "decision_index": len(timeline),
                "env_step": step,
                "controller": source,
                "score": score,
                "threshold": threshold,
                "alarm": bool(alarm),
            }
        )
        for local_step, action in enumerate(candidate):
            if plan is not None:
                action = plan.action_at(raw_obs["agent"]["qpos"], local_step)
            raw_obs, _, terminated, truncated, info = env.step(
                torch.as_tensor(action, device=env.unwrapped.device).unsqueeze(0)
            )
            actions.append(np.asarray(action, dtype=np.float32))
            sources.append(source)
            records.append(_extract_record(raw_obs))
            grasped |= bool_scalar(info.get("is_cubeA_grasped", False))
            on_cube |= bool_scalar(info.get("is_cubeA_on_cubeB", False))
            success |= bool_scalar(info.get("success", False))
            failure_state.update(
                env_step=len(actions),
                currently_grasped=bool_scalar(info.get("is_cubeA_grasped", False)),
                on_cube=bool_scalar(info.get("is_cubeA_on_cubeB", False)),
                success=bool_scalar(info.get("success", False)),
                cube_z=float(env.unwrapped.cubeA.pose.p.reshape(-1, 3)[0, 2].item()),
            )
            task_states.append(
                task_state(
                    env,
                    grasped=bool_scalar(info.get("is_cubeA_grasped", False)),
                    on_cube=bool_scalar(info.get("is_cubeA_on_cubeB", False)),
                )
            )
            if success or bool_scalar(terminated) or bool_scalar(truncated):
                break

    return records, actions, sources, expert_start, {
        "seed": seed,
        "split": split,
        "method": method,
        "success": success,
        "grasped_once": grasped,
        "on_cube_once": on_cube,
        "lifted_once": failure_state.ever_lifted,
        "max_stage": failure_state.best_stage,
        "steps": len(actions),
        "expert_start_step": expert_start,
        "expert_action_steps": 0 if expert_start is None else len(actions) - expert_start,
        "failure_recovery_mode": failure_recovery_mode,
        "failure_recovery_event": failure_recovery_event,
        "fixed_timing_step": fixed_timing_step,
        "timeline": timeline,
        "task_state_dim": int(task_states[0].shape[0]),
        "task_states": np.stack(task_states),
        **stack_cube_reset_metadata(env, split=split),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=tuple(dict.fromkeys((*METHODS, *TIMING_CONDITIONS))), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--internal-assets", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-id", type=Path, required=True)
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--expert-action-budget", type=int)
    parser.add_argument("--pool-action-target", type=int)
    parser.add_argument("--seed-manifest", type=Path)
    parser.add_argument("--controlled-timing", action="store_true")
    parser.add_argument(
        "--timing-step",
        type=int,
        help="Fixed action-step takeover for the fixed_timing condition.",
    )
    parser.add_argument("--consume-all-seeds", action="store_true")
    parser.add_argument("--offline-per-split", type=int, default=50)
    parser.add_argument(
        "--ood-split", choices=("ood", STACK_CUBE_STAGE2_OOD_SPLIT), default="ood"
    )
    parser.add_argument("--id-seed", type=int, default=70000)
    parser.add_argument("--ood-seed", type=int, default=80000)
    parser.add_argument("--max-attempts", type=int, default=5000)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--diff-timesteps", type=int, default=16)
    parser.add_argument("--diff-patience", type=int, default=2)
    parser.add_argument("--diff-threshold", type=float)
    parser.add_argument("--internal-layer", default="vlm_action_bridge")
    parser.add_argument("--probe-steps", type=int, default=5)
    parser.add_argument("--probe-seed", type=int, default=0)
    parser.add_argument(
        "--failure-recovery-mode", choices=("fixed_step", "event"), default="fixed_step"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.method == "fixed_timing":
        if args.timing_step is None or args.timing_step < 0 or args.timing_step >= TASK_HORIZON:
            raise ValueError("fixed_timing requires 0 <= --timing-step < task horizon")
    elif args.timing_step is not None:
        raise ValueError("--timing-step is only valid with --method fixed_timing")
    if args.expert_action_budget is not None and args.expert_action_budget <= 0:
        raise ValueError("expert-action-budget must be positive")
    if args.output_dir.exists() or args.repo_id.exists():
        raise FileExistsError("output and dataset paths must be new")
    args.output_dir.mkdir(parents=True)
    calibration_name = f"{args.internal_layer}_pca"
    calibration = (
        json.loads(args.calibration.read_text(encoding="utf-8"))
        if args.calibration is not None
        else {"methods": {}}
    )
    methods = calibration["methods"]
    pca_threshold = float(methods.get(calibration_name, {}).get("threshold", 0.0))
    diff_threshold = float(
        args.diff_threshold
        if args.diff_threshold is not None
        else methods.get("diffdagger", {}).get("threshold", 0.0)
    )
    device = torch.device("cuda")
    internal_pca = (
        InternalPCA.load(args.internal_assets, device, layer=args.internal_layer)
        if args.method in {*INTERNAL_PCA_METHODS, "bridge_pca", "vlm_bridge_pca"}
        and args.internal_assets is not None
        else None
    )
    policy = (
        None
        if args.method in {"offline_oracle", "immediate"}
        else XVLAAirplanePolicy(args.checkpoint, args.xvla_root)
    )
    internal_probe = (
        XVLAMultilayerProbe(
            policy.model, probe_seed=args.probe_seed, probe_steps=args.probe_steps
        )
        if args.method in INTERNAL_PCA_METHODS and policy is not None
        else None
    )
    _write_json(
        args.output_dir / "collection_provenance.json",
        {
            "format": "xvla_stackcube_four_group_collection_v1",
            "method": args.method,
            "checkpoint": str(args.checkpoint.resolve()),
            "internal_assets": (
                str(args.internal_assets.resolve()) if args.internal_assets else None
            ),
            "calibration": str(args.calibration.resolve()) if args.calibration else None,
            "internal_pca_threshold": pca_threshold,
            "internal_layer": args.internal_layer,
            "internal_calibration_method": calibration_name,
            "diffdagger_threshold": diff_threshold,
            "raw_split_schedule": (
                "offline_exact_50_50"
                if args.method == "offline_oracle" and args.expert_action_budget is None
                else "strict_id_ood_alternation"
            ),
            "ood_split": args.ood_split,
            "target_episodes": args.target,
            "expert_action_budget": args.expert_action_budget,
            "pool_action_target": args.pool_action_target,
            "seed_manifest": str(args.seed_manifest.resolve()) if args.seed_manifest else None,
            "controlled_timing": args.controlled_timing,
            "timing_step": args.timing_step,
            "task_horizon": TASK_HORIZON,
            "failure_recovery_step": FAILURE_RECOVERY_STEP,
            "failure_recovery_mode": args.failure_recovery_mode,
            "failure_event_definition": {
                "drop_height": 0.06,
                "lift_height": 0.07,
                "stall_steps": FAILURE_STALL_STEPS,
                "timeout_step": FAILURE_TIMEOUT_STEP,
            },
            "admission": "task_success_and_nonempty_full_expert_suffix",
            "tail_handling": "all_real_actions_saved; temporal action_valid_mask during training",
        },
    )
    register_stack_cube_splits()
    active_splits = ("id", args.ood_split)
    envs = {split: _make_env(split) for split in active_splits}
    dataset = None
    accepted = 0
    accepted_expert_actions = 0
    accepted_by_split = {split: 0 for split in active_splits}
    accepted_actions_by_split = {split: 0 for split in active_splits}
    raw_by_split = {split: 0 for split in active_splits}
    next_seed = {"id": args.id_seed, args.ood_split: args.ood_seed}
    frozen_seeds = None
    if args.seed_manifest is not None:
        seed_payload = json.loads(args.seed_manifest.read_text(encoding="utf-8"))
        frozen_seeds = [int(seed) for seed in seed_payload["seeds"]]
        if not frozen_seeds:
            raise ValueError("seed manifest contains no seeds")
    try:
        for attempt_index in range(args.max_attempts):
            target_actions = args.pool_action_target or args.expert_action_budget
            if not args.consume_all_seeds and collection_complete(
                accepted, accepted_expert_actions,
                target_episodes=args.target, expert_action_budget=target_actions
            ):
                break
            if frozen_seeds is not None and attempt_index >= len(frozen_seeds):
                break
            split = args.ood_split if frozen_seeds is not None else alternating_split(
                attempt_index, args.ood_split
            )
            if (
                args.method == "offline_oracle"
                and args.expert_action_budget is None
                and accepted_by_split[split] >= args.offline_per_split
            ):
                split = args.ood_split if split == "id" else "id"
            if (
                args.method == "offline_oracle"
                and args.expert_action_budget is None
                and accepted_by_split[split] >= args.offline_per_split
            ):
                break
            seed = frozen_seeds[attempt_index] if frozen_seeds is not None else next_seed[split]
            if frozen_seeds is None:
                next_seed[split] += 1
            raw_by_split[split] += 1
            records, actions, sources, expert_start, row = _run_attempt(
                method=args.method,
                split=split,
                seed=seed,
                env=envs[split],
                policy=policy,
                internal_pca=internal_pca,
                internal_layer=args.internal_layer,
                internal_probe=internal_probe,
                pca_threshold=pca_threshold,
                diff_threshold=diff_threshold,
                diff_patience=args.diff_patience,
                diff_timesteps=args.diff_timesteps,
                flow_steps=args.flow_steps,
                failure_recovery_mode=args.failure_recovery_mode,
                controlled_timing=args.controlled_timing,
                fixed_timing_step=args.timing_step if args.method == "fixed_timing" else None,
            )
            row["attempt_index"] = attempt_index
            task_states = row.pop("task_states")
            frames = _build_frames(
                records=records,
                actions=actions,
                task=STACK_CUBE_TASK,
                main_camera=_select_camera(
                    records[0].obs, "", ("base_camera",) + MAIN_CAMERA_CANDIDATES, "main"
                ),
                wrist_camera=_select_camera(
                    records[0].obs, "", ("hand_camera",) + WRIST_CAMERA_CANDIDATES, "wrist"
                ),
            )
            row["video"] = str(
                write_episode_video_durably(
                    frames,
                    video_dir=args.output_dir / "raw_archive/videos",
                    episode_index=attempt_index,
                    seed=seed,
                    fps=10,
                )
            )
            action_stem = args.output_dir / "raw_archive/actions" / f"episode_{attempt_index:06d}_seed_{seed:06d}"
            action_stem.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(action_stem) + ".npy", np.asarray(actions, dtype=np.float32))
            task_state_path = (
                args.output_dir / "raw_archive/task_states"
                / f"episode_{attempt_index:06d}_seed_{seed:06d}.npy"
            )
            task_state_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(task_state_path, task_states)
            row["task_states"] = str(task_state_path)
            Path(str(action_stem) + ".sources.json").write_text(
                json.dumps(sources) + "\n", encoding="utf-8"
            )
            admitted = admitted_suffix(row["success"], expert_start, len(actions))
            row["accepted"] = admitted is not None
            if admitted is not None:
                begin, end = admitted
                full_expert_actions = end - begin
                if args.expert_action_budget is not None and args.pool_action_target is None:
                    remaining = args.expert_action_budget - accepted_expert_actions
                    end = min(end, begin + remaining)
                selected_expert_actions = end - begin
                row["accepted_expert_action_steps"] = selected_expert_actions
                row["budget_truncated"] = selected_expert_actions < full_expert_actions
            _append_jsonl(args.output_dir / "episodes.jsonl", row)
            if admitted is not None:
                label_frames = _build_frames(
                    records=records[begin : end + 1],
                    actions=actions[begin:end],
                    task=STACK_CUBE_TASK,
                    main_camera=_select_camera(
                        records[0].obs, "", ("base_camera",) + MAIN_CAMERA_CANDIDATES, "main"
                    ),
                    wrist_camera=_select_camera(
                        records[0].obs, "", ("hand_camera",) + WRIST_CAMERA_CANDIDATES, "wrist"
                    ),
                )
                if dataset is None:
                    dataset = _create_dataset(
                        repo_id=str(args.repo_id),
                        image_shape=tuple(label_frames[0]["image"].shape),
                        wrist_image_shape=tuple(label_frames[0]["wrist_image"].shape),
                        fps=10,
                        image_writer_threads=4,
                        image_writer_processes=0,
                    )
                for frame in label_frames:
                    dataset.add_frame(frame)
                dataset.save_episode()
                _append_jsonl(
                    args.output_dir / "training_episodes.jsonl",
                    {
                        "dataset_episode_index": accepted,
                        "raw_attempt_index": attempt_index,
                        "seed": seed,
                        "split": split,
                        "expert_start_step": begin,
                        "expert_action_steps": end - begin,
                        "full_expert_action_steps": full_expert_actions,
                        "budget_truncated": end - begin < full_expert_actions,
                    },
                )
                accepted += 1
                accepted_expert_actions += end - begin
                accepted_by_split[split] += 1
                accepted_actions_by_split[split] += end - begin
            print(
                f"[xvla-stackcube-collect] method={args.method} raw={attempt_index + 1} "
                f"split={split} accepted={accepted}/{args.target} "
                f"expert_actions={accepted_expert_actions}/"
                f"{target_actions or 'episode-target'} by_split={accepted_by_split}",
                flush=True,
            )
    finally:
        if internal_probe is not None:
            internal_probe.close()
        if dataset is not None and getattr(dataset, "image_writer", None) is not None:
            dataset.image_writer.wait_until_done()
        for env in envs.values():
            env.close()

    target_actions = args.pool_action_target or args.expert_action_budget
    if not args.consume_all_seeds and not collection_complete(
        accepted, accepted_expert_actions,
        target_episodes=args.target, expert_action_budget=target_actions
    ):
        raise RuntimeError(
            f"collection incomplete: episodes={accepted}/{args.target}, "
            f"expert_actions={accepted_expert_actions}/{target_actions}"
        )
    _write_json(
        args.output_dir / "summary.json",
        {
            "method": args.method,
            "accepted_total": accepted,
            "accepted_expert_actions": accepted_expert_actions,
            "pool_action_target": args.pool_action_target,
            "accepted_by_split": accepted_by_split,
            "accepted_actions_by_split": accepted_actions_by_split,
            "raw_by_split": raw_by_split,
            "raw_total": sum(raw_by_split.values()),
            "dataset": str(args.repo_id.resolve()),
        },
    )


if __name__ == "__main__":
    main()
