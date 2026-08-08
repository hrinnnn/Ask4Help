#!/usr/bin/env python3
"""Collect strict-success expert data for four airplane BC/DAgger groups."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RLINF_ROOT = ROOT / "RLinf"
sys.path[:0] = [str(ROOT), str(RLINF_ROOT)]

from rlinf.algorithms.diffdagger import DiffDAggerQueryGate  # noqa: E402
from rlinf.algorithms.vla_fail import PCAResidualStatistics, pca_residual_score  # noqa: E402
from rlinf.envs.maniskill.pick_single_ycb_airplane_variants import (  # noqa: E402
    PICK_SINGLE_YCB_AIRPLANE_TASK,
    register_controlled_pick_single_ycb_airplane_variants,
    reset_metadata,
)
from toolkits.lerobot.collect_maniskill_peg_lerobot_joint import (  # noqa: E402
    MAIN_CAMERA_CANDIDATES,
    WRIST_CAMERA_CANDIDATES,
    _build_frames,
    _convert_solver_action_to_joint_delta,
    _create_dataset,
    _extract_record,
    _joint_delta_arm_bounds,
    _select_camera,
    _to_numpy,
)
from toolkits.lerobot.collect_maniskill_pick_single_ycb_airplane_lerobot import (  # noqa: E402
    _build_env,
    write_episode_video_durably,
)
from toolkits.lerobot.diagnose_pick_single_ycb_airplane_oracle import (  # noqa: E402
    ORACLE_NECK_CANDIDATES,
    try_candidate,
)
from tools.evaluate_pick_single_ycb_airplane_pi05 import model_observation  # noqa: E402
from tools.maniskill_pi05_vfd_online_awbc import _load_model  # noqa: E402
from tools.pick_single_ycb_airplane_eval_common import clip_action_chunk  # noqa: E402

EXECUTE_HORIZON = 5
POLICY_HORIZON = 50
TASK_HORIZON = 250
DELTA_SERVO_TOLERANCE = 0.012
DELTA_SERVO_MAX_SUBSTEPS = 4
ORACLE_CLOSE_MAX_STEPS = 60
ORACLE_STABLE_GRASP_STEPS = 4


def alternating_split(attempt_index: int) -> str:
    if attempt_index < 0:
        raise ValueError("attempt_index must be non-negative")
    return "id" if attempt_index % 2 == 0 else "ood"


def should_query_bridge_pca(score: float, threshold: float) -> bool:
    return score > threshold


def admitted_expert_suffix(*, success: bool, expert_start: int | None, action_count: int) -> tuple[int, int] | None:
    """Return every real expert action; temporal masking handles the short tail."""
    if not success or expert_start is None or expert_start >= action_count:
        return None
    return expert_start, action_count


def delta_servo_complete(curr_qpos: np.ndarray, solver_action: np.ndarray, *, tolerance: float) -> bool:
    """Whether the seven arm joints have reached an absolute planner target."""
    curr = np.asarray(curr_qpos, dtype=np.float32).reshape(-1)
    target = np.asarray(solver_action, dtype=np.float32).reshape(-1)
    if curr.size < 7 or target.size < 7:
        raise ValueError("delta servo requires at least seven arm joints")
    return bool(np.max(np.abs(target[:7] - curr[:7])) <= tolerance)


def patience_gate_episode_score(scores: list[float], *, patience: int) -> float:
    """Reduce a score sequence to the threshold needed to trigger its gate."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if patience < 1:
        raise ValueError("patience must be at least one")
    if values.size < patience:
        return float("-inf")
    if not np.isfinite(values).all():
        raise ValueError("gate calibration scores must be finite")
    # A patience-length window triggers only when every score exceeds the
    # threshold.  Its limiting score is therefore the minimum in the window.
    return float(max(np.min(values[index : index + patience]) for index in range(values.size - patience + 1)))


def calibrate_patience_gate_threshold(
    score_sequences: list[list[float]], *, alpha: float, patience: int
) -> tuple[float, list[float]]:
    """Calibrate the complete temporal gate on successful ID trajectories."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    episode_scores = [
        patience_gate_episode_score(sequence, patience=patience)
        for sequence in score_sequences
    ]
    finite_scores = [score for score in episode_scores if np.isfinite(score)]
    if not finite_scores:
        raise ValueError("no calibration trajectory is long enough for the patience gate")
    return float(np.quantile(np.asarray(finite_scores), alpha)), episode_scores


def _bool(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu().reshape(-1)[0].item())
    return bool(np.asarray(value).reshape(-1)[0])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _env_args(split: str, args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        split=split,
        image_size=args.image_size,
        control_freq=args.control_freq,
        max_episode_steps=TASK_HORIZON,
        sim_backend=args.sim_backend,
    )


def _policy_prediction(model: Any, raw_obs: dict[str, Any], *, seed: int, step: int) -> tuple[np.ndarray, torch.Tensor]:
    env_obs = model_observation(raw_obs)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed * 1000 + step)
        torch.cuda.manual_seed_all(seed * 1000 + step)
        with torch.inference_mode():
            predicted, result = model.predict_action_batch(env_obs=env_obs, mode="train")
    model_actions = result["forward_inputs"]["model_action"]
    return predicted.detach().float().cpu().numpy(), model_actions


def _bridge_pca_score(model: Any, raw_obs: dict[str, Any], prior: torch.Tensor, stats: PCAResidualStatistics) -> float:
    with torch.inference_mode():
        features = model.extract_multilayer_llmd_features(model_observation(raw_obs), prior)
    return float(pca_residual_score(features["vlm_bridge_final_mean"], stats)[0].item())


def _diff_score(
    model: Any,
    raw_obs: dict[str, Any],
    model_actions: torch.Tensor,
    *,
    num_timesteps: int,
    num_noise_samples: int,
) -> float:
    with torch.inference_mode():
        score = model.compute_diffdagger_uncertainty(
            model_observation(raw_obs),
            model_actions,
            num_timesteps=num_timesteps,
            num_noise_samples=num_noise_samples,
        )
    return float(score.reshape(-1)[0].detach().cpu())


def _plan_and_execute_expert(
    policy_env: Any,
    solver_env: Any,
    *,
    seed: int,
    raw_obs: dict[str, Any],
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[list[Any], list[np.ndarray], dict[str, Any]]:
    """Execute planner targets through delta control in the exact policy env."""
    state = policy_env.unwrapped.get_state_dict()
    attempts: list[dict[str, Any]] = []
    for name, local_point in ORACLE_NECK_CANDIDATES:
        policy_env.unwrapped.set_state_dict(state)
        candidate_records: list[Any] = []
        candidate_actions: list[np.ndarray] = []
        original_step = policy_env.step

        servo_substeps = 0
        servo_saturated_steps = 0

        def step_hook(solver_action, *step_args, **step_kwargs):
            nonlocal servo_substeps, servo_saturated_steps
            result = None
            for _ in range(DELTA_SERVO_MAX_SUBSTEPS):
                qpos = policy_env.unwrapped.agent.robot.get_qpos()[0].detach().cpu().numpy()
                if result is not None and delta_servo_complete(
                    qpos, solver_action, tolerance=DELTA_SERVO_TOLERANCE
                ):
                    break
                action = _convert_solver_action_to_joint_delta(qpos, solver_action, lower, upper)
                if np.any(np.abs(action[:7]) >= 1.0 - 1e-6):
                    servo_saturated_steps += 1
                result = original_step(action, *step_args, **step_kwargs)
                candidate_actions.append(action)
                candidate_records.append(_extract_record(result[0]))
                servo_substeps += 1
            assert result is not None
            return result

        policy_env.step = step_hook  # type: ignore[method-assign]
        try:
            result = try_candidate(
                policy_env,
                seed=seed,
                name=name,
                local_point=local_point,
                close_steps=ORACLE_CLOSE_MAX_STEPS,
                complete_task=True,
                reset_before_attempt=False,
                force_planner_pd_joint_pos=True,
                closing_sign=-1.0 if name.endswith("_flip") else 1.0,
                stable_grasp_steps=ORACLE_STABLE_GRASP_STEPS,
            )
        finally:
            policy_env.step = original_step  # type: ignore[method-assign]
        result["delta_servo_substeps"] = servo_substeps
        result["delta_servo_saturated_steps"] = servo_saturated_steps
        attempts.append(result)
        if result["accepted"] and candidate_actions:
            return candidate_records, candidate_actions, {
                "accepted": True,
                "selected_candidate": name,
                "attempts": attempts,
                "planned_actions": len(candidate_actions),
                "executed_actions": len(candidate_actions),
                "execution_mode": "same_env_absolute_target_to_delta",
            }
    policy_env.unwrapped.set_state_dict(state)
    return [], [], {"accepted": False, "attempts": attempts, "execution_mode": "same_env_absolute_target_to_delta"}


def _save_raw_attempt(
    *,
    output_dir: Path,
    episode_index: int,
    seed: int,
    records: list[Any],
    actions: list[np.ndarray],
    sources: list[str],
    control_freq: int,
) -> str:
    video_records = records
    video_actions = actions
    if not actions:
        # A motion-planning attempt can fail before producing any command.
        # Preserve an honest empty action sidecar while encoding one static
        # diagnostic frame for the raw archive.
        video_records = [records[0], records[0]]
        video_actions = [np.zeros(8, dtype=np.float32)]
    frames = _build_frames(
        records=video_records,
        actions=video_actions,
        task=PICK_SINGLE_YCB_AIRPLANE_TASK,
        main_camera=_select_camera(records[0].obs, "", ("base_camera",) + MAIN_CAMERA_CANDIDATES, "main"),
        wrist_camera=_select_camera(records[0].obs, "", ("hand_camera",) + WRIST_CAMERA_CANDIDATES, "wrist"),
    )
    video = write_episode_video_durably(
        frames,
        video_dir=output_dir / "raw_archive" / "videos",
        episode_index=episode_index,
        seed=seed,
        fps=control_freq,
    )
    stem = output_dir / "raw_archive" / "actions" / f"episode_{episode_index:06d}_seed_{seed:06d}"
    stem.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(stem) + ".npy", np.asarray(actions, dtype=np.float32))
    Path(str(stem) + ".sources.json").write_text(json.dumps(sources) + "\n", encoding="utf-8")
    return str(video)


def _run_attempt(
    *,
    method: str,
    split: str,
    seed: int,
    model: Any | None,
    policy_env: Any,
    solver_env: Any,
    lower: np.ndarray,
    upper: np.ndarray,
    pca_prior: torch.Tensor | None,
    pca_stats: PCAResidualStatistics | None,
    pca_threshold: float,
    diff_gate: DiffDAggerQueryGate | None,
    args: argparse.Namespace,
) -> tuple[list[Any], list[np.ndarray], int | None, dict[str, Any]]:
    raw_obs, _info = policy_env.reset(seed=seed)
    records = [_extract_record(raw_obs)]
    actions: list[np.ndarray] = []
    sources: list[str] = []
    timeline: list[dict[str, Any]] = []
    expert_start: int | None = 0 if method == "offline_oracle" else None
    expert_latched = expert_start is not None
    success = False
    oracle_result: dict[str, Any] | None = None
    if diff_gate is not None:
        diff_gate.reset()

    while len(actions) < TASK_HORIZON and not success:
        step = len(actions)
        score = threshold = None
        predicted = model_actions = None
        if not expert_latched and step < POLICY_HORIZON:
            assert model is not None
            predicted, model_actions = _policy_prediction(model, raw_obs, seed=seed, step=step)
            if method == "bridge_pca":
                assert pca_prior is not None and pca_stats is not None
                score = _bridge_pca_score(model, raw_obs, pca_prior, pca_stats)
                threshold = pca_threshold
                expert_latched = should_query_bridge_pca(score, threshold)
            elif method == "diffdagger":
                assert diff_gate is not None and model_actions is not None
                score = _diff_score(
                    model,
                    raw_obs,
                    model_actions,
                    num_timesteps=args.diff_num_timesteps,
                    num_noise_samples=args.diff_num_noise_samples,
                )
                decision = diff_gate.decide(torch.tensor([score], device="cuda"))
                threshold = decision.threshold
                expert_latched = bool(decision.query_mask.item())
            if expert_latched:
                expert_start = step
        if method == "failure_recovery" and step >= POLICY_HORIZON and not success:
            expert_latched = True
            expert_start = step
        if method == "offline_oracle":
            expert_latched = True

        if expert_latched:
            timeline.append({"env_step": step, "controller": "expert", "score": score, "threshold": threshold, "alarm": expert_start == step})
            expert_records, expert_actions, oracle_result = _plan_and_execute_expert(
                policy_env, solver_env, seed=seed, raw_obs=raw_obs, lower=lower, upper=upper
            )
            records.extend(expert_records)
            actions.extend(expert_actions)
            sources.extend(["expert"] * len(expert_actions))
            success = bool(oracle_result["accepted"])
            break

        if step >= POLICY_HORIZON:
            break
        assert model is not None and predicted is not None
        low = np.asarray(policy_env.action_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(policy_env.action_space.high, dtype=np.float32).reshape(-1)
        chunk = clip_action_chunk(predicted, low, high, int(model.config.action_horizon))
        timeline.append({"env_step": step, "controller": "policy", "score": score, "threshold": threshold, "alarm": False})
        for action in chunk[:EXECUTE_HORIZON]:
            raw_obs, _reward, terminated, truncated, info = policy_env.step(
                torch.as_tensor(action, device=policy_env.unwrapped.device).unsqueeze(0)
            )
            actions.append(np.asarray(action, dtype=np.float32))
            sources.append("policy")
            records.append(_extract_record(raw_obs))
            success = _bool(info.get("success", False))
            if success or _bool(terminated) or _bool(truncated):
                break

    metadata = reset_metadata(policy_env, split=split)
    return records, actions, expert_start, {
        "seed": seed,
        "split": split,
        "method": method,
        "success": success,
        "steps": len(actions),
        "expert_start_step": expert_start,
        "expert_action_steps": 0 if expert_start is None else len(actions) - expert_start,
        "timeline": timeline,
        "oracle": oracle_result,
        "sources": sources,
        **metadata,
    }


def _load_diff_scores(path: Path) -> list[float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    scores = payload.get("scores")
    if not isinstance(scores, list) or not scores:
        raise ValueError("Diff calibration must contain a non-empty scores list")
    return [float(score) for score in scores]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("offline_oracle", "bridge_pca", "failure_recovery", "diffdagger"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--pi05-base", type=Path)
    parser.add_argument("--norm-stats", type=Path)
    parser.add_argument("--detector-assets", type=Path)
    parser.add_argument("--bridge-pca-threshold", type=float, default=0.6333393454551697)
    parser.add_argument("--diff-calibration", type=Path)
    parser.add_argument("--diff-alpha", type=float, default=0.95)
    parser.add_argument("--diff-patience", type=int, default=2)
    parser.add_argument(
        "--diff-threshold",
        type=float,
        help="Optional threshold calibrated for the complete patience gate.",
    )
    parser.add_argument("--diff-num-timesteps", type=int, default=16)
    parser.add_argument("--diff-num-noise-samples", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--target-expert-trajectories", type=int, default=100)
    parser.add_argument(
        "--only-split",
        choices=("id", "ood"),
        help="Diagnostic mode: draw every raw attempt from exactly one split.",
    )
    parser.add_argument("--offline-per-split", type=int, default=50)
    parser.add_argument("--offline-id-target", type=int)
    parser.add_argument("--offline-ood-target", type=int)
    parser.add_argument("--id-seed", type=int, default=70000)
    parser.add_argument("--ood-seed", type=int, default=80000)
    parser.add_argument("--max-attempts", type=int, default=5000)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--control-freq", type=int, default=10)
    parser.add_argument("--sim-backend", choices=("physx_cpu", "gpu"), default="physx_cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.repo_id)
    if args.output_dir.exists() or dataset_path.exists():
        raise FileExistsError("output and dataset paths must be new")
    args.output_dir.mkdir(parents=True)
    if args.method != "offline_oracle" and not all((args.checkpoint, args.pi05_base, args.norm_stats)):
        raise ValueError("gated methods require checkpoint, pi05-base, and norm-stats")

    model = None if args.method == "offline_oracle" else _load_model(args.checkpoint, args.norm_stats, args.pi05_base)
    pca_prior = None
    pca_stats = None
    if args.method == "bridge_pca":
        asset = torch.load(args.detector_assets, map_location="cpu", weights_only=False)
        pca_prior = asset["fixed_prior"].to("cuda")
        pca_stats = PCAResidualStatistics.from_state_dict(asset["statistics"]["bridge_pca_residual"])
    diff_gate = None
    if args.method == "diffdagger":
        diff_gate = DiffDAggerQueryGate(
            _load_diff_scores(args.diff_calibration),
            alpha=args.diff_alpha,
            patience=args.diff_patience,
        )
        if args.diff_threshold is not None:
            if not np.isfinite(args.diff_threshold):
                raise ValueError("DiffDAgger threshold override must be finite")
            diff_gate.threshold = float(args.diff_threshold)

    provenance = {
        "format": "pick_airplane_four_group_collection_v1",
        "method": args.method,
        "checkpoint": None if args.checkpoint is None else str(args.checkpoint.resolve()),
        "checkpoint_sha256": None if args.checkpoint is None else _sha256(args.checkpoint / "actor/model_state_dict/full_weights.pt"),
        "norm_stats": None if args.norm_stats is None else str(args.norm_stats.resolve()),
        "bridge_pca_threshold": args.bridge_pca_threshold if args.method == "bridge_pca" else None,
        "diff_threshold": args.diff_threshold if args.method == "diffdagger" else None,
        "diff_patience": args.diff_patience if args.method == "diffdagger" else None,
        "split_schedule": "strict_id_ood_alternation",
        "admission": "strict_success_and_nonempty_expert_suffix",
        "training_labels": "all_real_expert_suffix_actions_temporal_masked",
    }
    _write_json(args.output_dir / "collection_provenance.json", provenance)

    register_controlled_pick_single_ycb_airplane_variants()
    policy_envs = {split: _build_env(_env_args(split, args), control_mode="pd_joint_delta_pos") for split in ("id", "ood")}
    solver_envs = {split: _build_env(_env_args(split, args), control_mode="pd_joint_pos") for split in ("id", "ood")}
    lower, upper = _joint_delta_arm_bounds(policy_envs["id"])
    dataset = None
    rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    accepted = {"id": 0, "ood": 0}
    offline_targets = {
        "id": args.offline_per_split if args.offline_id_target is None else args.offline_id_target,
        "ood": args.offline_per_split if args.offline_ood_target is None else args.offline_ood_target,
    }
    next_seed = {"id": args.id_seed, "ood": args.ood_seed}
    try:
        for attempt_index in range(args.max_attempts):
            if args.method == "offline_oracle":
                if accepted == offline_targets:
                    break
            elif len(train_rows) >= args.target_expert_trajectories:
                break
            split = args.only_split or alternating_split(attempt_index)
            if args.method == "offline_oracle" and accepted[split] >= offline_targets[split]:
                continue
            seed = next_seed[split]
            next_seed[split] += 1
            records, actions, expert_start, row = _run_attempt(
                method=args.method,
                split=split,
                seed=seed,
                model=model,
                policy_env=policy_envs[split],
                solver_env=solver_envs[split],
                lower=lower,
                upper=upper,
                pca_prior=pca_prior,
                pca_stats=pca_stats,
                pca_threshold=args.bridge_pca_threshold,
                diff_gate=diff_gate,
                args=args,
            )
            sources = row.pop("sources")
            row["attempt_index"] = attempt_index
            row["video"] = _save_raw_attempt(
                output_dir=args.output_dir,
                episode_index=attempt_index,
                seed=seed,
                records=records,
                actions=actions,
                sources=sources,
                control_freq=args.control_freq,
            )
            admitted = admitted_expert_suffix(success=row["success"], expert_start=expert_start, action_count=len(actions))
            row["accepted"] = admitted is not None
            if admitted is not None:
                begin, end = admitted
                frames = _build_frames(
                    records=records[begin : end + 1],
                    actions=actions[begin:end],
                    task=PICK_SINGLE_YCB_AIRPLANE_TASK,
                    main_camera=_select_camera(records[0].obs, "", ("base_camera",) + MAIN_CAMERA_CANDIDATES, "main"),
                    wrist_camera=_select_camera(records[0].obs, "", ("hand_camera",) + WRIST_CAMERA_CANDIDATES, "wrist"),
                )
                if dataset is None:
                    dataset = _create_dataset(
                        repo_id=args.repo_id,
                        image_shape=tuple(frames[0]["image"].shape),
                        wrist_image_shape=tuple(frames[0]["wrist_image"].shape),
                        fps=args.control_freq,
                        image_writer_threads=4,
                        image_writer_processes=0,
                    )
                for frame in frames:
                    dataset.add_frame(frame)
                dataset.save_episode()
                accepted[split] += 1
                train_rows.append({
                    "dataset_episode_index": len(train_rows),
                    "raw_attempt_index": attempt_index,
                    "seed": seed,
                    "split": split,
                    "expert_start_step": begin,
                    "expert_action_steps": end - begin,
                })
            rows.append(row)
            _write_jsonl(args.output_dir / "episodes.jsonl", rows)
            _write_jsonl(args.output_dir / "training_episodes.jsonl", train_rows)
            print(f"[airplane-gated] method={args.method} attempt={attempt_index + 1} accepted={len(train_rows)} split_counts={accepted}", flush=True)
    finally:
        if dataset is not None and getattr(dataset, "image_writer", None) is not None:
            dataset.image_writer.wait_until_done()
        for env in (*policy_envs.values(), *solver_envs.values()):
            env.close()
        if model is not None:
            del model
            torch.cuda.empty_cache()

    expected = sum(offline_targets.values()) if args.method == "offline_oracle" else args.target_expert_trajectories
    if dataset is None or len(train_rows) != expected:
        raise RuntimeError(f"collected {len(train_rows)}/{expected} accepted expert trajectories")
    _write_json(args.output_dir / "summary.json", {
        "method": args.method,
        "raw_attempts": len(rows),
        "accepted_total": len(train_rows),
        "accepted_by_split": accepted,
        "dataset": str(dataset_path),
        "raw_archive": str(args.output_dir / "raw_archive"),
    })


if __name__ == "__main__":
    main()
