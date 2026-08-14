#!/usr/bin/env python3
"""Calibrate and evaluate ID-only pi0.5 failure detectors on OpenDrawer."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RLINF_ROOT = Path(os.environ.get("ASK4HELP_RLINF_ROOT", ROOT / "RLinf"))
sys.path[:0] = [str(ROOT), str(RLINF_ROOT)]

from rlinf.algorithms.vla_fail import (  # noqa: E402
    constant_split_conformal_threshold,
    knn_score,
    llmd_score,
    pca_residual_score,
)
from rlinf.envs.maniskill.open_drawer_retrieve_place_spec import (  # noqa: E402
    ENV_IDS,
    TASK_INSTRUCTION,
    reset_metadata,
)
from tools.maniskill_pi05_vfd_online_awbc import _load_model  # noqa: E402
from tools.evaluate_open_drawer_id_pi05 import _model_obs  # noqa: E402
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


def _bool(value: Any) -> bool:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return bool(np.asarray(value, dtype=bool).reshape(-1).any())


def _load_detectors(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "open_drawer_internal_detector_assets_v1":
        raise ValueError(f"unsupported detector asset: {path}")
    from rlinf.algorithms.vla_fail import (
        KNNStatistics,
        LLMDStatistics,
        PCAResidualStatistics,
    )
    for spec in payload["detectors"].values():
        kind = spec["kind"]
        if kind == "llmd":
            spec["statistics"] = LLMDStatistics.from_state_dict(spec["statistics"])
        elif kind == "knn":
            spec["statistics"] = KNNStatistics.from_state_dict(spec["statistics"])
        elif kind == "pca_residual":
            spec["statistics"] = PCAResidualStatistics.from_state_dict(spec["statistics"])
        else:
            raise ValueError(f"unknown detector kind {kind!r}")
    return payload["detectors"], payload


def _score(feature: torch.Tensor, spec: dict[str, Any]) -> float:
    kind = spec["kind"]
    statistics = spec["statistics"]
    if kind == "llmd":
        value = llmd_score(feature, statistics)
    elif kind == "knn":
        value = knn_score(feature, statistics)
    elif kind == "pca_residual":
        value = pca_residual_score(feature, statistics)
    else:
        raise ValueError(f"unknown detector kind {kind!r}")
    return float(value.max().item())


def _wilson(successes: int, total: int) -> list[float] | None:
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    scale = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / scale
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / scale
    return [float(max(0.0, center - radius)), float(min(1.0, center + radius))]


def _metrics(rows: list[dict[str, Any]], detector_names: list[str], thresholds: dict[str, float]) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, precision_recall_fscore_support, roc_auc_score

    labels = np.asarray([int(not row["success"]) for row in rows], dtype=np.int64)
    result: dict[str, Any] = {}
    for name in detector_names:
        scores = np.asarray([row["episode_scores"][name] for row in rows], dtype=np.float64)
        pred = scores >= thresholds[name]
        positive = labels == 1
        success = labels == 0
        tp = int(np.sum(pred & positive))
        fp = int(np.sum(pred & success))
        fn = int(np.sum(~pred & positive))
        tn = int(np.sum(~pred & success))
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, pred.astype(np.int64), average="binary", zero_division=0
        )
        result[name] = {
            "failure_definition": "failure = not success",
            "episodes": int(len(rows)),
            "failures": int(labels.sum()),
            "successes": int((labels == 0).sum()),
            "threshold": float(thresholds[name]),
            "tp_failure": tp,
            "fp_success_false_alarm": fp,
            "fn_failure_missed": fn,
            "tn_success_not_alarm": tn,
            "success_conditioned_false_alarm_rate": float(fp / max(1, int(success.sum()))),
            "success_conditioned_false_alarm_ci95": _wilson(fp, int(success.sum())),
            "failure_recall": float(recall),
            "precision": float(precision),
            "f1": float(f1),
            "balanced_accuracy": float(balanced_accuracy_score(labels, pred.astype(np.int64))),
            "auprc": float(average_precision_score(labels, scores)) if len(np.unique(labels)) > 1 else None,
            "auroc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) > 1 else None,
            "score_summary": {
                "min": float(scores.min()), "p50": float(np.quantile(scores, 0.5)),
                "p95": float(np.quantile(scores, 0.95)), "max": float(scores.max()),
            },
        }
    return result


def _env(split: str, max_steps: int):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    import rlinf.envs.maniskill.open_drawer_retrieve_place  # noqa: F401

    return gym.make(
        ENV_IDS[split], robot_uids="panda_wristcam", num_envs=1, obs_mode="rgb",
        control_mode="pd_joint_delta_pos", reward_mode="sparse", render_mode="rgb_array",
        sim_backend="physx_cpu", sim_config={"sim_freq": 100, "control_freq": 10},
        sensor_configs={"width": 384, "height": 384}, max_episode_steps=max_steps,
    )


def _run_episode(*, model: Any, prior: torch.Tensor, env: Any, detectors: dict[str, dict[str, Any]], split: str,
                 seed: int, execute_horizon: int, max_episode_steps: int,
                 thresholds: dict[str, float] | None, episode_index: int, video_dir: Path | None) -> dict[str, Any]:
    raw_obs, info = env.reset(seed=seed)
    metadata = reset_metadata(env, split=split)
    records = [_extract_record(raw_obs)]
    actions: list[np.ndarray] = []
    timeline: list[dict[str, Any]] = []
    success = False
    ever = {key: False for key in ("ever_drawer_opened", "ever_grasped", "ever_lifted", "object_in_target", "object_released", "is_robot_static")}
    low = np.asarray(env.action_space.low).reshape(-1)
    high = np.asarray(env.action_space.high).reshape(-1)
    while len(actions) < max_episode_steps and not success:
        env_obs = _model_obs(raw_obs)
        with torch.inference_mode():
            features = model.extract_multilayer_llmd_features(
                env_obs, prior, include_action_expert_final=True
            )
            predicted, _ = model.predict_action_batch(env_obs=env_obs, mode="eval", compute_values=False)
        scores = {name: _score(features[spec["layer"]], spec) for name, spec in detectors.items()}
        alarms = {name: bool(thresholds is not None and scores[name] >= thresholds[name]) for name in detectors}
        timeline.append({"decision_index": len(timeline), "env_step": len(actions), "scores": scores, "alarms": alarms})
        chunk = np.clip(predicted.detach().float().cpu().numpy()[0][:execute_horizon], low, high).astype(np.float32)
        for action in chunk:
            raw_obs, _reward, terminated, truncated, info = env.step(torch.as_tensor(action, device=env.unwrapped.device).unsqueeze(0))
            actions.append(action)
            records.append(_extract_record(raw_obs))
            for key in ever:
                ever[key] |= _bool(info.get(key, False))
            success = _bool(info.get("success", False))
            if success or _bool(terminated) or _bool(truncated) or len(actions) >= max_episode_steps:
                break
    video_path = None
    if video_dir is not None and actions:
        main = _select_camera(records[0].obs, "", ("base_camera",) + MAIN_CAMERA_CANDIDATES, "main")
        wrist = _select_camera(records[0].obs, "", ("hand_camera",) + WRIST_CAMERA_CANDIDATES, "wrist")
        frames = _build_frames(records=records, actions=actions, task=TASK_INSTRUCTION, main_camera=main, wrist_camera=wrist)
        video_path = write_episode_video_durably(frames, video_dir=video_dir, episode_index=episode_index, seed=seed, fps=10)
    return {
        "episode_index": episode_index, "seed": seed, "split": split, "success": bool(success),
        "steps": len(actions), "timeline": timeline,
        "episode_scores": {name: max((point["scores"][name] for point in timeline), default=float("nan")) for name in detectors},
        "first_alarm_decision": {name: next((point["decision_index"] for point in timeline if point["alarms"][name]), None) for name in detectors},
        "video": str(video_path) if video_path else None, **ever, **metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("calibrate", "evaluate"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pi05-base", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--detector-assets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=tuple(ENV_IDS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--execute-horizon", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=400)
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--target-successes", type=int, default=20)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--save-videos", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "calibrate" and args.split != "id":
        raise ValueError("calibration must use ID policy rollouts")
    if args.mode == "evaluate" and args.thresholds is None:
        raise ValueError("evaluation requires --thresholds")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_videos:
        (args.output_dir / "videos").mkdir()
    detector_specs, asset = _load_detectors(args.detector_assets)
    thresholds: dict[str, float] | None = None
    if args.thresholds:
        threshold_payload = json.loads(args.thresholds.read_text())
        thresholds = {name: float(spec["threshold"]) for name, spec in threshold_payload["detectors"].items()}
        if set(thresholds) != set(detector_specs):
            raise ValueError("threshold asset does not match detector asset")
    model = _load_model(args.checkpoint, args.norm_stats, args.pi05_base)
    # Construct the same deterministic fixed prior used when fitting the ID assets.
    from rlinf.algorithms.vla_fail import fixed_gaussian_prior
    prior = fixed_gaussian_prior(
        action_horizon=int(model.config.action_horizon),
        action_dim=int(model.config.action_dim),
        seed=int(asset["fixed_prior_seed"]),
        device="cuda",
    )
    env = _env(args.split, args.max_episode_steps)
    rows: list[dict[str, Any]] = []
    try:
        for index in range(args.episodes if args.mode == "evaluate" else args.episodes * 4):
            row = _run_episode(model=model, prior=prior, env=env, detectors=detector_specs, split=args.split, seed=args.seed + index,
                               execute_horizon=args.execute_horizon, max_episode_steps=args.max_episode_steps,
                               thresholds=thresholds, episode_index=index,
                               video_dir=(args.output_dir / "videos") if args.save_videos else None)
            rows.append(row)
            print(f"[open-drawer-detectors] {len(rows)} seed={row['seed']} success={int(row['success'])}", flush=True)
            if args.mode == "calibrate" and sum(int(row["success"]) for row in rows) >= args.target_successes:
                break
    finally:
        env.close()
        del model
        torch.cuda.empty_cache()

    if args.mode == "calibrate":
        successful = [row for row in rows if row["success"]]
        if len(successful) < args.target_successes:
            raise RuntimeError(f"calibration only found {len(successful)}/{args.target_successes} successful ID rollouts")
        thresholds = {name: constant_split_conformal_threshold(
            [[max(point["scores"][name] for point in row["timeline"])] for row in successful[:args.target_successes]],
            delta=args.delta,
        )
                      for name in detector_specs}
        payload = {
            "format": "open_drawer_internal_detector_threshold_v1",
            "checkpoint": str(args.checkpoint), "detector_assets": str(args.detector_assets),
            "split": "id", "delta": args.delta, "target_successes": args.target_successes,
            "successful_seeds": [row["seed"] for row in successful[:args.target_successes]],
            "detectors": {name: {"threshold": float(value)} for name, value in thresholds.items()},
        }
        (args.output_dir / "calibration.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        (args.output_dir / "episodes.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    assert thresholds is not None
    result = {
        "format": "open_drawer_failure_detector_rollouts_v1",
        "checkpoint": str(args.checkpoint), "detector_assets": str(args.detector_assets),
        "split": args.split, "episodes": len(rows), "metrics": _metrics(rows, list(detector_specs), thresholds),
        "episodes_path": str(args.output_dir / "episodes.json"),
    }
    (args.output_dir / "episodes.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
