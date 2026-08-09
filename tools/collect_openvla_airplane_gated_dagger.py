#!/usr/bin/env python3
"""Collect four OpenVLA airplane BC/robot-gated DAgger datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "RLinf")]

from openvla_airplane.dataset import AIRPLANE_INSTRUCTION  # noqa: E402
from openvla_airplane.evaluate import _base_image  # noqa: E402
from openvla_airplane.gated import (  # noqa: E402
    METHODS,
    admitted_expert_suffix,
    alternating_split,
    update_patience_gate,
)
from openvla_airplane.model import load_rlinf_openvla, predict_action_rlinf  # noqa: E402
from openvla_airplane.runtime import DetectorBank  # noqa: E402
from rlinf.envs.maniskill.pick_single_ycb_airplane_variants import (  # noqa: E402
    PICK_SINGLE_YCB_AIRPLANE_TASK,
    register_controlled_pick_single_ycb_airplane_variants,
    reset_metadata,
)
from toolkits.lerobot.collect_maniskill_peg_lerobot_joint import (  # noqa: E402
    MAIN_CAMERA_CANDIDATES,
    WRIST_CAMERA_CANDIDATES,
    _build_frames,
    _create_dataset,
    _extract_record,
    _joint_delta_arm_bounds,
    _select_camera,
)
from toolkits.lerobot.collect_maniskill_pick_single_ycb_airplane_lerobot import _build_env  # noqa: E402
from tools.collect_pick_single_ycb_airplane_gated_dagger import (  # noqa: E402
    ORACLE_CLOSE_MAX_STEPS,
    ORACLE_STABLE_GRASP_STEPS,
    _plan_and_execute_expert,
    _save_raw_attempt,
)

TASK_HORIZON = 250
FAILURE_RECOVERY_STEP = 50


def _bool(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu().reshape(-1)[0].item())
    return bool(np.asarray(value).reshape(-1)[0])


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _env_args(args: argparse.Namespace, split: str) -> argparse.Namespace:
    return argparse.Namespace(
        split=split,
        image_size=args.image_size,
        control_freq=args.control_freq,
        max_episode_steps=TASK_HORIZON,
        sim_backend=args.sim_backend,
    )


@torch.inference_mode()
def _diffdagger_score(model, image, *, samples: int, seed: int, device: int) -> float:
    image_tensor = torch.as_tensor(np.array(image, copy=True), device=device).unsqueeze(0).repeat(samples, 1, 1, 1)
    device_index = torch.device(f"cuda:{device}").index
    with torch.random.fork_rng(devices=[device_index]):
        torch.manual_seed(seed)
        actions, _ = model.predict_action_batch(
            env_obs={
                "main_images": image_tensor,
                "task_descriptions": [AIRPLANE_INSTRUCTION] * samples,
            },
            calculate_logprobs=False,
            calculate_values=False,
            do_sample=True,
            max_new_tokens=8,
            use_cache=True,
        )
    values = actions[:, 0].float()
    return float(values.var(dim=0, unbiased=False).sum().item())


def _run_attempt(
    *,
    method: str,
    split: str,
    seed: int,
    model,
    processor,
    detector_bank: DetectorBank | None,
    policy_env,
    solver_env,
    lower: np.ndarray,
    upper: np.ndarray,
    pca_threshold: float,
    diff_threshold: float,
    patience: int,
    action_samples: int,
    device: int,
) -> tuple[list[Any], list[np.ndarray], int | None, dict[str, Any]]:
    raw_obs, _ = policy_env.reset(seed=seed)
    records = [_extract_record(raw_obs)]
    actions: list[np.ndarray] = []
    sources: list[str] = []
    timeline: list[dict[str, Any]] = []
    expert_start = 0 if method == "offline_oracle" else None
    gate_count = 0
    strict_success = False
    terminated = truncated = False

    while len(actions) < TASK_HORIZON and expert_start is None and not strict_success:
        step = len(actions)
        if method == "failure_recovery" and step >= FAILURE_RECOVERY_STEP:
            expert_start = step
            timeline.append({"env_step": step, "controller": "expert", "alarm": True, "reason": "fixed_failure_recovery"})
            break

        image = _base_image(raw_obs)
        action, inputs = predict_action_rlinf(model, processor, image, device, use_cache=True)
        score = threshold = None
        alarm = False
        if method == "siglip_pca":
            assert detector_bank is not None
            score = detector_bank.score(model, inputs, image, sample_seed=seed * 1000 + step)[0][
                "siglip_pooled_residual_pca"
            ]
            threshold = pca_threshold
            gate_count, alarm = update_patience_gate(score, threshold, gate_count, patience)
        elif method == "diffdagger":
            score = _diffdagger_score(
                model,
                image,
                samples=action_samples,
                seed=seed * 1000 + step,
                device=device,
            )
            threshold = diff_threshold
            gate_count, alarm = update_patience_gate(score, threshold, gate_count, patience)
        if alarm:
            expert_start = step
            timeline.append({
                "env_step": step,
                "controller": "expert",
                "score": score,
                "threshold": threshold,
                "alarm": True,
            })
            break

        low = np.asarray(policy_env.action_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(policy_env.action_space.high, dtype=np.float32).reshape(-1)
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), low, high)
        if action.size != 8:
            raise ValueError(f"OpenVLA returned {action.size} actions; expected 8")
        timeline.append({
            "env_step": step,
            "controller": "policy",
            "score": score,
            "threshold": threshold,
            "alarm": False,
        })
        raw_obs, _, terminated, truncated, info = policy_env.step(
            torch.as_tensor(action, device=policy_env.unwrapped.device).unsqueeze(0)
        )
        actions.append(action)
        sources.append("policy")
        records.append(_extract_record(raw_obs))
        strict_success = _bool(info.get("success", False))
        if terminated or truncated:
            break

    oracle_result = None
    if expert_start is not None:
        expert_records, expert_actions, oracle_result = _plan_and_execute_expert(
            policy_env,
            solver_env,
            seed=seed,
            raw_obs=raw_obs,
            lower=lower,
            upper=upper,
        )
        records.extend(expert_records)
        actions.extend(expert_actions)
        sources.extend(["expert"] * len(expert_actions))
        strict_success = bool(oracle_result["accepted"])

    return records, actions, expert_start, {
        "seed": seed,
        "split": split,
        "method": method,
        "strict_success": strict_success,
        "steps": len(actions),
        "expert_start_step": expert_start,
        "expert_action_steps": 0 if expert_start is None else len(actions) - expert_start,
        "timeline": timeline,
        "oracle": oracle_result,
        "sources": sources,
        **reset_metadata(policy_env, split=split),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--base-path", default="openvla/openvla-7b")
    parser.add_argument("--detector-assets", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-id", type=Path, required=True)
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--offline-per-split", type=int, default=50)
    parser.add_argument("--id-seed", type=int, default=70000)
    parser.add_argument("--ood-seed", type=int, default=80000)
    parser.add_argument("--max-attempts", type=int, default=5000)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--action-samples", type=int, default=10)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--control-freq", type=int, default=10)
    parser.add_argument("--sim-backend", choices=("physx_cpu", "gpu"), default="physx_cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() or args.repo_id.exists():
        raise FileExistsError("output and dataset paths must be new")
    args.output_dir.mkdir(parents=True)
    if args.method != "offline_oracle" and args.checkpoint is None:
        raise ValueError("gated methods require --checkpoint")
    if args.method in {"siglip_pca", "diffdagger"} and args.calibration is None:
        raise ValueError("score-gated methods require --calibration")
    if args.method == "siglip_pca" and args.detector_assets is None:
        raise ValueError("siglip_pca requires --detector-assets")

    calibration = {} if args.calibration is None else json.loads(args.calibration.read_text(encoding="utf-8"))
    pca_threshold = float(calibration.get("siglip_pca", {}).get("threshold", "nan"))
    diff_threshold = float(calibration.get("diffdagger", {}).get("threshold", "nan"))
    model = processor = None
    detector_bank = None
    if args.method != "offline_oracle":
        model, processor = load_rlinf_openvla(args.base_path, args.checkpoint, args.device)
    if args.method == "siglip_pca":
        detector_bank = DetectorBank(args.detector_assets, args.device, action_samples=1)

    _write_json(args.output_dir / "collection_provenance.json", {
        "format": "openvla_airplane_four_group_collection_v1",
        "method": args.method,
        "checkpoint": None if args.checkpoint is None else str(args.checkpoint.resolve()),
        "base_path": args.base_path,
        "detector_assets": None if args.detector_assets is None else str(args.detector_assets.resolve()),
        "calibration": None if args.calibration is None else str(args.calibration.resolve()),
        "split_schedule": "offline_exact_50_50" if args.method == "offline_oracle" else "strict_raw_id_ood_alternation",
        "policy_horizon": TASK_HORIZON,
        "failure_recovery_step": FAILURE_RECOVERY_STEP,
        "gate_patience": args.patience,
        "diffdagger_action_samples": args.action_samples if args.method == "diffdagger" else None,
        "oracle_close": {"max_steps": ORACLE_CLOSE_MAX_STEPS, "stable_steps": ORACLE_STABLE_GRASP_STEPS},
        "admission": "strict_oracle_success_and_nonempty_full_expert_suffix",
        "tail_handling": "all_real_actions_saved",
    })

    register_controlled_pick_single_ycb_airplane_variants()
    policy_envs = {split: _build_env(_env_args(args, split), control_mode="pd_joint_delta_pos") for split in ("id", "ood")}
    solver_envs = {split: _build_env(_env_args(args, split), control_mode="pd_joint_pos") for split in ("id", "ood")}
    bounds = {split: _joint_delta_arm_bounds(policy_envs[split]) for split in ("id", "ood")}
    dataset = None
    accepted = 0
    accepted_by_split = {"id": 0, "ood": 0}
    raw_by_split = {"id": 0, "ood": 0}
    next_seed = {"id": args.id_seed, "ood": args.ood_seed}
    try:
        for attempt_index in range(args.max_attempts):
            if accepted >= args.target:
                break
            split = alternating_split(attempt_index)
            if args.method == "offline_oracle" and accepted_by_split[split] >= args.offline_per_split:
                split = "ood" if split == "id" else "id"
            if args.method == "offline_oracle" and accepted_by_split[split] >= args.offline_per_split:
                break
            seed = next_seed[split]
            next_seed[split] += 1
            raw_by_split[split] += 1
            lower, upper = bounds[split]
            records, actions, expert_start, row = _run_attempt(
                method=args.method,
                split=split,
                seed=seed,
                model=model,
                processor=processor,
                detector_bank=detector_bank,
                policy_env=policy_envs[split],
                solver_env=solver_envs[split],
                lower=lower,
                upper=upper,
                pca_threshold=pca_threshold,
                diff_threshold=diff_threshold,
                patience=args.patience,
                action_samples=args.action_samples,
                device=args.device,
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
            admitted = admitted_expert_suffix(row["strict_success"], expert_start, len(actions))
            row["accepted"] = admitted is not None
            _append_jsonl(args.output_dir / "episodes.jsonl", row)
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
                        repo_id=str(args.repo_id),
                        image_shape=tuple(frames[0]["image"].shape),
                        wrist_image_shape=tuple(frames[0]["wrist_image"].shape),
                        fps=args.control_freq,
                        image_writer_threads=4,
                        image_writer_processes=0,
                    )
                for frame in frames:
                    dataset.add_frame(frame)
                dataset.save_episode()
                _append_jsonl(args.output_dir / "training_episodes.jsonl", {
                    "dataset_episode_index": accepted,
                    "raw_attempt_index": attempt_index,
                    "seed": seed,
                    "split": split,
                    "expert_start_step": begin,
                    "expert_action_steps": end - begin,
                })
                accepted += 1
                accepted_by_split[split] += 1
            print(
                f"[openvla-collector] method={args.method} attempt={attempt_index + 1} "
                f"accepted={accepted}/{args.target} accepted_by_split={accepted_by_split}",
                flush=True,
            )
    finally:
        if dataset is not None and getattr(dataset, "image_writer", None) is not None:
            dataset.image_writer.wait_until_done()
        for env in (*policy_envs.values(), *solver_envs.values()):
            env.close()

    if accepted != args.target:
        raise RuntimeError(f"collected {accepted}/{args.target} accepted trajectories")
    _write_json(args.output_dir / "summary.json", {
        "method": args.method,
        "accepted_total": accepted,
        "accepted_by_split": accepted_by_split,
        "raw_by_split": raw_by_split,
        "dataset": str(args.repo_id),
        "raw_archive": str(args.output_dir / "raw_archive"),
    })


if __name__ == "__main__":
    main()
