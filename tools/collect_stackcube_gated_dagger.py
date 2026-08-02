#!/usr/bin/env python3
"""Collect equal-label StackCube oracle and detector-gated BC datasets.

The raw archive retains every attempted rollout.  The LeRobot training dataset
contains only the latching expert suffix, truncated on 10-step boundaries so
all experiment groups receive exactly the same number of action labels.
"""

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
RLINF_ROOT = ROOT / "RLinf"
sys.path[:0] = [str(ROOT), str(RLINF_ROOT)]

from rlinf.envs.maniskill.stack_cube_privileged_oracle import StackCubePrivilegedChunkOracle  # noqa: E402
from rlinf.envs.maniskill.stack_cube_variants import STACK_CUBE_TASK, reset_metadata  # noqa: E402
from rlinf.algorithms.diffdagger import DiffDAggerQueryGate, load_calibration_scores  # noqa: E402
from toolkits.lerobot.collect_maniskill_peg_lerobot_joint import (  # noqa: E402
    MAIN_CAMERA_CANDIDATES,
    WRIST_CAMERA_CANDIDATES,
    _build_frames,
    _create_dataset,
    _extract_record,
    _select_camera,
)
from toolkits.lerobot.collect_maniskill_plug_lerobot_joint import write_episode_video_durably  # noqa: E402
from tools.evaluate_stackcube_internal_detectors import (  # noqa: E402
    Detector,
    load_detectors,
    score_feature,
)
from tools.maniskill_pi05_vfd_online_awbc import (  # noqa: E402
    _action_chunk,
    _bool,
    _build_env,
    _load_model,
    _wrap_obs,
)


CHUNK_LABEL_HORIZON = 10
EXECUTE_HORIZON = 5


@dataclass(frozen=True)
class ExpertSuffix:
    start: int | None
    action_count: int

    @property
    def trainable_chunks(self) -> int:
        return self.action_count // CHUNK_LABEL_HORIZON

    @property
    def valid_10_step_anchors(self) -> int:
        return max(0, self.action_count - CHUNK_LABEL_HORIZON + 1)

    @property
    def has_full_horizon(self) -> bool:
        return self.valid_10_step_anchors > 0


def choose_split(
    collected: dict[str, int], targets: dict[str, int], *, prefer_id: bool
) -> str | None:
    """Choose the split with remaining action-label budget; alternate ties."""

    pending = [split for split in ("id", "ood") if collected[split] < targets[split]]
    if not pending:
        return None
    if len(pending) == 1:
        return pending[0]
    return "id" if prefer_id else "ood"


def alternating_split(attempt_index: int) -> str:
    """Return the controlled reset split for an alternating raw-rollout stream."""
    if attempt_index < 0:
        raise ValueError("attempt_index must be non-negative")
    return "id" if attempt_index % 2 == 0 else "ood"


def selected_suffix_steps(suffix: ExpertSuffix, remaining_chunks: int) -> int:
    """Return the exact, non-overlapping 10-step label budget admitted from a suffix."""

    if suffix.start is None or remaining_chunks <= 0:
        return 0
    return min(suffix.trainable_chunks, remaining_chunks) * CHUNK_LABEL_HORIZON


def admitted_suffix_steps(
    suffix: ExpertSuffix, *, remaining_chunks: int, fixed_episode_collection: bool,
    preserve_terminal_suffix: bool = False,
) -> int:
    """Keep an expert suffix, optionally including its terminal partial chunk."""
    if suffix.start is None or not suffix.has_full_horizon:
        return 0
    if preserve_terminal_suffix or fixed_episode_collection:
        return suffix.action_count
    return selected_suffix_steps(suffix, remaining_chunks)


def is_successful_expert_trajectory(suffix: ExpertSuffix, *, success: bool) -> bool:
    """A successful assisted trajectory retains every real expert action.

    The full suffix is retained, including its terminal action.  It still must
    contain one valid 10-step anchor; shorter suffixes cannot supervise this
    action-chunk policy on their own.
    """
    return bool(success and suffix.start is not None and suffix.has_full_horizon)


def should_latch_expert(
    method: str, *, action_step: int, score: float | None = None, threshold: float | None = None
) -> bool:
    """Centralize the three group intervention semantics for collection and tests."""
    if method == "offline_oracle":
        return True
    if method == "late_success":
        return action_step >= 50
    if method in {"bridge_knn", "bridge_llmd", "diffdagger"}:
        return score is not None and threshold is not None and score >= threshold
    if method == "failure_recovery":
        return False
    raise ValueError(f"unknown collection method: {method}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _resolve_detector(
    method: str,
    detectors: dict[str, Detector] | None,
    thresholds: dict[str, Any] | None,
) -> tuple[str, Detector, float] | None:
    if method not in {"bridge_knn", "bridge_llmd"}:
        return None
    if detectors is None or thresholds is None:
        raise ValueError(f"{method} requires detector assets and thresholds")
    expected = "vlm_bridge_final_mean__knn_k10" if method == "bridge_knn" else "vlm_bridge_final_mean__llmd"
    detector = detectors.get(expected)
    threshold = thresholds.get("detectors", {}).get(expected, {}).get("threshold")
    if detector is None or threshold is None:
        raise ValueError(f"threshold asset is missing {expected}")
    return expected, detector, float(threshold)


def _policy_chunk(model: Any, raw_obs: dict[str, Any], info: dict[str, Any], *, seed: int, step: int) -> np.ndarray:
    candidate, _ = _policy_prediction(model, raw_obs, info, seed=seed, step=step)
    return candidate


def _policy_prediction(
    model: Any, raw_obs: dict[str, Any], info: dict[str, Any], *, seed: int, step: int
) -> tuple[np.ndarray, torch.Tensor]:
    """Generate a policy chunk once and retain the full chunk for DiffDAgger."""
    env_obs = _wrap_obs(raw_obs, info, task="stack")
    # Keep policy actions identical across detector groups until the first gate.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed * 1000 + step)
        torch.cuda.manual_seed_all(seed * 1000 + step)
        with torch.inference_mode():
            predicted, _ = model.predict_action_batch(env_obs=env_obs, mode="eval", compute_values=False)
    return _action_chunk(predicted, EXECUTE_HORIZON), predicted


def _diffdagger_score(
    model: Any,
    raw_obs: dict[str, Any],
    info: dict[str, Any],
    predicted_actions: torch.Tensor,
    *,
    num_timesteps: int,
    num_noise_samples: int,
) -> float:
    env_obs = _wrap_obs(raw_obs, info, task="stack")
    with torch.inference_mode():
        score = model.compute_diffdagger_uncertainty(
            env_obs,
            predicted_actions,
            num_timesteps=num_timesteps,
            num_noise_samples=num_noise_samples,
        )
    return float(score.reshape(-1)[0].detach().cpu())


@dataclass
class FailureRecoveryState:
    """Deterministic simulator-only subgoal failure monitor.

    This is deliberately an oracle recovery baseline, not a vision deployable
    detector: it reads cube positions from ManiSkill and records its reason in
    the controller timeline.
    """

    initial_height: float
    best_goal_distance: float
    lifted_once: bool = False
    stagnant_chunks: int = 0


def _failure_recovery_reason(
    env: Any, state: FailureRecoveryState, *, action_step: int, min_policy_steps: int
) -> str | None:
    base = env.unwrapped
    cube = np.asarray(base.cubeA.pose.p.detach().cpu())[0]
    target = np.asarray(base.cubeB.pose.p.detach().cpu())[0]
    height = float(cube[2])
    goal_distance = float(np.linalg.norm(cube[:2] - target[:2]))
    lifted = height >= state.initial_height + 0.03
    state.lifted_once = state.lifted_once or lifted
    if goal_distance < state.best_goal_distance - 0.003:
        state.best_goal_distance = goal_distance
        state.stagnant_chunks = 0
    elif state.lifted_once:
        state.stagnant_chunks += 1

    if action_step < min_policy_steps:
        return None
    if not state.lifted_once:
        return "grasp_timeout_no_lift"
    if height <= state.initial_height + 0.012:
        return "dropped_after_lift"
    if state.stagnant_chunks >= 3:
        return "post_grasp_stagnation"
    return None


def _detector_score(
    model: Any,
    raw_obs: dict[str, Any],
    info: dict[str, Any],
    detector: Detector,
    prior: torch.Tensor,
) -> float:
    env_obs = _wrap_obs(raw_obs, info, task="stack")
    with torch.inference_mode():
        features = model.extract_multilayer_llmd_features(env_obs, prior)
        return float(score_feature(features[detector.layer], detector)[0].item())


def _run_attempt(
    *,
    env: Any,
    split: str,
    seed: int,
    method: str,
    model: Any | None,
    detector_spec: tuple[str, Detector, float] | None,
    prior: torch.Tensor | None,
    diffdagger_gate: DiffDAggerQueryGate | None,
    diffdagger_num_timesteps: int,
    diffdagger_num_noise_samples: int,
    failure_min_policy_steps: int,
    episode_index: int,
    raw_dir: Path,
) -> tuple[list[Any], list[np.ndarray], list[str], ExpertSuffix, dict[str, Any]]:
    raw_obs, info = env.reset(seed=seed)
    metadata = reset_metadata(env, split=split)
    records = [_extract_record(raw_obs)]
    actions: list[np.ndarray] = []
    sources: list[str] = []
    timeline: list[dict[str, Any]] = []
    oracle = StackCubePrivilegedChunkOracle(chunk_size=EXECUTE_HORIZON)
    cube_height = float(np.asarray(env.unwrapped.cubeA.pose.p.detach().cpu())[0, 2])
    cube_xy = np.asarray(env.unwrapped.cubeA.pose.p.detach().cpu())[0, :2]
    target_xy = np.asarray(env.unwrapped.cubeB.pose.p.detach().cpu())[0, :2]
    failure_state = FailureRecoveryState(cube_height, float(np.linalg.norm(cube_xy - target_xy)))
    expert_latched = should_latch_expert(method, action_step=0)
    expert_start: int | None = 0 if expert_latched else None
    success = False
    terminated = truncated = False
    while len(actions) < 100 and not (success or terminated or truncated):
        start = len(actions)
        score: float | None = None
        threshold: float | None = None
        failure_reason: str | None = None
        if not expert_latched and method in {"bridge_knn", "bridge_llmd"}:
            assert model is not None and detector_spec is not None and prior is not None
            name, detector, threshold = detector_spec
            score = _detector_score(model, raw_obs, info, detector, prior)
            if should_latch_expert(
                method, action_step=start, score=score, threshold=threshold
            ):
                expert_latched = True
                expert_start = start
        elif not expert_latched and method == "diffdagger":
            assert model is not None and diffdagger_gate is not None
            candidate, predicted_actions = _policy_prediction(model, raw_obs, info, seed=seed, step=start)
            score = _diffdagger_score(
                model, raw_obs, info, predicted_actions,
                num_timesteps=diffdagger_num_timesteps,
                num_noise_samples=diffdagger_num_noise_samples,
            )
            decision = diffdagger_gate.decide(torch.tensor([score], device="cuda"))
            threshold = decision.threshold
            if bool(decision.query_mask.item()):
                expert_latched = True
                expert_start = start
        elif not expert_latched and method == "failure_recovery":
            failure_reason = _failure_recovery_reason(
                env, failure_state, action_step=start, min_policy_steps=failure_min_policy_steps
            )
            if failure_reason is not None:
                expert_latched = True
                expert_start = start
        elif not expert_latched and should_latch_expert(method, action_step=start):
            expert_latched = True
            expert_start = start

        if expert_latched:
            plan = oracle.plan(env)
            candidate = plan.actions
            source = "expert"
        else:
            assert model is not None
            plan = None
            if method != "diffdagger":
                candidate = _policy_chunk(model, raw_obs, info, seed=seed, step=start)
            source = "policy"
        timeline.append(
            {
                "chunk_index": len(timeline),
                "env_step": start,
                "controller": source,
                "score": score,
                "threshold": threshold,
                "alarm": bool(score is not None and threshold is not None and score >= threshold),
                "failure_reason": failure_reason,
            }
        )
        for local_step, action in enumerate(candidate):
            if plan is not None:
                action = plan.action_at(raw_obs["agent"]["qpos"], local_step)
            raw_obs, _reward, terminated, truncated, info = env.step(
                torch.as_tensor(action, device=env.unwrapped.device).unsqueeze(0)
            )
            actions.append(np.asarray(action, dtype=np.float32))
            sources.append(source)
            records.append(_extract_record(raw_obs))
            success = _bool(info.get("success", False))
            if success or _bool(terminated) or _bool(truncated):
                break

    frames = _build_frames(
        records=records,
        actions=actions,
        task=STACK_CUBE_TASK,
        main_camera=_select_camera(records[0].obs, "", MAIN_CAMERA_CANDIDATES, "main"),
        wrist_camera=_select_camera(records[0].obs, "", WRIST_CAMERA_CANDIDATES, "wrist"),
    )
    video = write_episode_video_durably(
        frames, video_dir=raw_dir / "videos", episode_index=episode_index, seed=seed, fps=10
    )
    # OSSFS does not reliably support the random seek used by zip-based NPZ
    # writers.  Keep raw actions in simple durable sidecars instead.
    action_stem = raw_dir / "actions" / f"episode_{episode_index:06d}_seed_{seed:06d}"
    np.save(str(action_stem) + ".npy", np.asarray(actions, dtype=np.float32))
    (Path(str(action_stem) + ".sources.json")).write_text(
        json.dumps(sources) + "\n", encoding="utf-8"
    )
    suffix = ExpertSuffix(
        start=expert_start,
        action_count=(len(actions) - expert_start) if expert_start is not None else 0,
    )
    row = {
        "episode_index": episode_index,
        "seed": seed,
        "split": split,
        "method": method,
        "success": success,
        "steps": len(actions),
        "expert_start_step": expert_start,
        "expert_action_steps": suffix.action_count,
        "expert_trainable_10_step_chunks": suffix.trainable_chunks,
        "timeline": timeline,
        "video_path": str(video),
        **metadata,
    }
    return records, actions, sources, suffix, row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=(
            "offline_oracle", "bridge_knn", "bridge_llmd", "late_success",
            "diffdagger", "failure_recovery",
        ),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-id", type=Path, required=True, help="Absolute destination for the LeRobot training dataset.")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--pi05-base", type=Path)
    parser.add_argument("--norm-stats", type=Path)
    parser.add_argument("--detector-assets", type=Path)
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--diffdagger-calibration", type=Path)
    parser.add_argument("--diffdagger-alpha", type=float, default=0.95)
    parser.add_argument("--diffdagger-patience", type=int, default=2)
    parser.add_argument("--diffdagger-num-timesteps", type=int, default=16)
    parser.add_argument("--diffdagger-num-noise-samples", type=int, default=1)
    parser.add_argument("--failure-min-policy-steps", type=int, default=50)
    parser.add_argument(
        "--preserve-terminal-suffix",
        action="store_true",
        help="Keep all real expert actions; SFT masks only unavailable future targets.",
    )
    parser.add_argument("--id-episodes", type=int, default=50)
    parser.add_argument("--ood-episodes", type=int, default=50)
    parser.add_argument("--id-target-chunks", type=int)
    parser.add_argument("--ood-target-chunks", type=int)
    parser.add_argument(
        "--gated-fixed-episodes",
        action="store_true",
        help=(
            "For a DAgger-cost comparison, collect a fixed number of raw ID/OOD "
            "rollouts and retain only naturally requested expert suffixes."
        ),
    )
    parser.add_argument(
        "--gated-successful-expert-episodes",
        type=int,
        help=(
            "Stop after this many successful, expert-intervened trajectories. "
            "Raw ID/OOD attempts alternate and only accepted expert suffixes train."
        ),
    )
    parser.add_argument("--id-seed", type=int, default=13000)
    parser.add_argument("--ood-seed", type=int, default=23000)
    parser.add_argument("--max-attempts-per-split", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() or args.repo_id.exists():
        raise FileExistsError("output-dir and repo-id must be new experiment paths")
    offline = args.method == "offline_oracle"
    if args.gated_fixed_episodes and args.gated_successful_expert_episodes is not None:
        raise ValueError("choose either --gated-fixed-episodes or --gated-successful-expert-episodes")
    successful_expert_target = args.gated_successful_expert_episodes
    if successful_expert_target is not None and successful_expert_target <= 0:
        raise ValueError("--gated-successful-expert-episodes must be positive")
    if not offline and successful_expert_target is None and not args.gated_fixed_episodes and (
        args.id_target_chunks is None or args.ood_target_chunks is None
    ):
        raise ValueError(
            "gated collection requires chunk targets, --gated-fixed-episodes, or "
            "--gated-successful-expert-episodes"
        )
    if not offline and (args.checkpoint is None or args.pi05_base is None or args.norm_stats is None):
        raise ValueError("gated collection requires checkpoint, pi05-base, and norm-stats")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    raw_dir = args.output_dir / "raw_archive"
    (raw_dir / "actions").mkdir(parents=True)

    model = None if offline else _load_model(args.checkpoint, args.norm_stats, args.pi05_base)
    detectors = thresholds = prior = None
    if args.method in {"bridge_knn", "bridge_llmd"}:
        if args.detector_assets is None or args.thresholds is None:
            raise ValueError("bridge-gated collection needs detector assets and thresholds")
        detectors, asset, asset_sha = load_detectors(args.detector_assets)
        thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))
        if thresholds.get("detector_assets_sha256") != asset_sha:
            raise ValueError("threshold asset does not match detector assets")
        prior = torch.load(Path(asset["llmd_source_statistics"]), map_location="cpu", weights_only=False)["fixed_prior"].to("cuda")
    detector_spec = _resolve_detector(args.method, detectors, thresholds)
    diffdagger_gate = None
    if args.method == "diffdagger":
        if args.diffdagger_calibration is None:
            raise ValueError("diffdagger collection requires --diffdagger-calibration")
        diffdagger_gate = DiffDAggerQueryGate(
            load_calibration_scores(args.diffdagger_calibration),
            alpha=args.diffdagger_alpha,
            patience=args.diffdagger_patience,
        )

    fixed_episode_collection = offline or args.gated_fixed_episodes
    targets = (
        {"id": int(args.id_episodes), "ood": int(args.ood_episodes)}
        if fixed_episode_collection or successful_expert_target is not None
        else {"id": int(args.id_target_chunks), "ood": int(args.ood_target_chunks)}
    )
    collected = {"id": 0, "ood": 0}
    attempts = {"id": 0, "ood": 0}
    next_seed = {"id": int(args.id_seed), "ood": int(args.ood_seed)}
    rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    dataset = None
    envs = {split: _build_env(100, task="stack", split=split) for split in ("id", "ood")}
    prefer_id = True
    raw_attempt_index = 0
    successful_expert_collected = 0
    try:
        while True:
            if successful_expert_target is not None:
                if successful_expert_collected >= successful_expert_target:
                    break
                split = alternating_split(raw_attempt_index)
                raw_attempt_index += 1
            else:
                split = choose_split(collected, targets, prefer_id=prefer_id)
                if split is None:
                    break
            if attempts[split] >= args.max_attempts_per_split:
                raise RuntimeError(f"{split} exhausted {attempts[split]} attempts before its target")
            prefer_id = not prefer_id
            seed = next_seed[split]
            next_seed[split] += 1
            attempts[split] += 1
            if diffdagger_gate is not None:
                diffdagger_gate.reset()
            records, actions, _sources, suffix, row = _run_attempt(
                env=envs[split], split=split, seed=seed, method=args.method, model=model,
                detector_spec=detector_spec, prior=prior,
                diffdagger_gate=diffdagger_gate,
                diffdagger_num_timesteps=args.diffdagger_num_timesteps,
                diffdagger_num_noise_samples=args.diffdagger_num_noise_samples,
                failure_min_policy_steps=args.failure_min_policy_steps,
                episode_index=len(rows), raw_dir=raw_dir,
            )
            if offline:
                suffix = ExpertSuffix(start=0, action_count=len(actions))
                admitted_steps = (
                    suffix.trainable_chunks * CHUNK_LABEL_HORIZON if row["success"] else 0
                )
            elif successful_expert_target is not None:
                admitted_steps = (
                    suffix.action_count
                    if is_successful_expert_trajectory(suffix, success=bool(row["success"]))
                    else 0
                )
            else:
                admitted_steps = admitted_suffix_steps(
                    suffix,
                    remaining_chunks=targets[split] - collected[split],
                    fixed_episode_collection=args.gated_fixed_episodes,
                    preserve_terminal_suffix=args.preserve_terminal_suffix,
                )
            row["admitted_expert_action_steps"] = admitted_steps
            row["admitted_expert_10_step_chunks"] = admitted_steps // CHUNK_LABEL_HORIZON
            row["accepted_successful_expert_trajectory"] = bool(
                successful_expert_target is not None and admitted_steps > 0
            )
            rows.append(row)
            if admitted_steps:
                assert suffix.start is not None
                begin = suffix.start
                label_frames = _build_frames(
                    records=records[begin : begin + admitted_steps + 1],
                    actions=actions[begin : begin + admitted_steps],
                    task=STACK_CUBE_TASK,
                    main_camera=_select_camera(records[0].obs, "", MAIN_CAMERA_CANDIDATES, "main"),
                    wrist_camera=_select_camera(records[0].obs, "", WRIST_CAMERA_CANDIDATES, "wrist"),
                )
                if dataset is None:
                    dataset = _create_dataset(
                        repo_id=str(args.repo_id), image_shape=tuple(label_frames[0]["image"].shape),
                        wrist_image_shape=tuple(label_frames[0]["wrist_image"].shape), fps=10,
                        image_writer_threads=4, image_writer_processes=0,
                    )
                for frame in label_frames:
                    dataset.add_frame(frame)
                dataset.save_episode()
                train_rows.append({"dataset_episode_index": len(train_rows), "raw_episode_index": row["episode_index"], "seed": seed, "split": split, "start_step": begin, "action_steps": admitted_steps, "ten_step_chunks": admitted_steps // CHUNK_LABEL_HORIZON})
            if successful_expert_target is not None and admitted_steps:
                successful_expert_collected += 1
                collected[split] += 1
            elif fixed_episode_collection:
                collected[split] += 1
            elif admitted_steps:
                collected[split] += admitted_steps // CHUNK_LABEL_HORIZON
            _write_jsonl(args.output_dir / "episodes.jsonl", rows)
            _write_jsonl(args.output_dir / "training_episodes.jsonl", train_rows)
            print(
                f"[collect] method={args.method} split={split} "
                f"seed={seed} admitted_actions={admitted_steps} "
                f"collected={collected} successful_total={successful_expert_collected} "
                f"targets={targets} successful_target={successful_expert_target}",
                flush=True,
            )
    finally:
        if dataset is not None and getattr(dataset, "image_writer", None) is not None:
            dataset.image_writer.wait_until_done()
        for env in envs.values():
            env.close()
        if model is not None:
            del model
            torch.cuda.empty_cache()

    if dataset is None:
        raise RuntimeError("collection produced no trainable expert suffixes")
    if fixed_episode_collection or successful_expert_target is not None:
        # Fixed-rollout collection measures both task success and the natural
        # expert-query cost.  Its training budget is derived after collection.
        budget = {
            split: sum(row["action_steps"] // CHUNK_LABEL_HORIZON for row in train_rows if row["split"] == split)
            for split in ("id", "ood")
        }
    else:
        budget = collected
    summary = {
        "method": args.method,
        "target_unit": (
            "successful_expert_trajectories"
            if successful_expert_target is not None
            else "episodes" if fixed_episode_collection else "ten_step_chunks"
        ),
        "targets": {"total": successful_expert_target}
        if successful_expert_target is not None else targets,
        "collected": {"total": successful_expert_collected, **collected}
        if successful_expert_target is not None else collected,
        "training_chunk_budget": budget,
        "attempts": attempts,
        "dataset": str(args.repo_id),
        "raw_archive": str(raw_dir),
    }
    _write_json(args.output_dir / "summary.json", summary)
    _write_json(args.output_dir / "label_budget.json", budget)


if __name__ == "__main__":
    main()
