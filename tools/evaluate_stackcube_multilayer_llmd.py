#!/usr/bin/env python3
"""Calibrate and passively evaluate cached multi-layer π0.5 LLMD probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
    llmd_score,
)
from rlinf.envs.maniskill.stack_cube_variants import reset_metadata  # noqa: E402
from tools.maniskill_pi05_vfd_online_awbc import (  # noqa: E402
    _action_chunk,
    _bool,
    _build_env,
    _load_model,
    _wrap_obs,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _score_summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "min": None, "p05": None, "p50": None, "p95": None, "max": None}
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "p05": float(np.quantile(array, 0.05)),
        "p50": float(np.quantile(array, 0.5)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def _wilson_interval(successes: int, total: int, *, z: float = 1.959963984540054) -> list[float] | None:
    if total == 0:
        return None
    proportion = successes / total
    scale = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / scale
    radius = z * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)) / scale
    return [float(max(0.0, center - radius)), float(min(1.0, center + radius))]


def _load_statistics(path: Path) -> tuple[dict[str, LLMDStatistics], torch.Tensor, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "stackcube_multilayer_llmd_statistics_v1":
        raise ValueError(f"not a multi-layer LLMD statistics asset: {path}")
    return (
        {name: LLMDStatistics.from_state_dict(spec["statistics"]) for name, spec in payload["layers"].items()},
        payload["fixed_prior"],
        payload,
    )


def _load_thresholds(path: Path | None, statistics_sha: str) -> dict[str, Any] | None:
    if path is None:
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("format") != "stackcube_multilayer_llmd_threshold_v1":
        raise ValueError(f"not a multi-layer threshold asset: {path}")
    if result.get("statistics_sha256") != statistics_sha:
        raise ValueError("threshold asset was calibrated from different multi-layer statistics")
    return result


def _metrics(episodes: list[dict[str, Any]], layer: str) -> dict[str, Any]:
    successes = [row for row in episodes if row["success"]]
    failures = [row for row in episodes if not row["success"]]
    success_alarm = sum(row["alarms"].get(layer, False) for row in successes)
    failure_alarm = sum(row["alarms"].get(layer, False) for row in failures)
    trace = [point["scores"][layer] for row in episodes for point in row["timeline"]]
    return {
        "episodes": len(episodes),
        "successes": len(successes),
        "failures": len(failures),
        "success_alarm_count": success_alarm,
        "failure_alarm_count": failure_alarm,
        "success_false_positive_rate": success_alarm / len(successes) if successes else None,
        "success_false_positive_rate_95_ci": _wilson_interval(success_alarm, len(successes)),
        "failure_recall": failure_alarm / len(failures) if failures else None,
        "failure_recall_95_ci": _wilson_interval(failure_alarm, len(failures)),
        "score_bounds": _score_summary(trace),
    }


def _run_episode(
    *, env: Any, model: Any, statistics: dict[str, LLMDStatistics], prior: torch.Tensor,
    seed: int, execute_horizon: int, max_episode_steps: int, thresholds: dict[str, Any] | None,
) -> dict[str, Any]:
    raw_obs, info = env.reset(seed=seed)
    metadata = reset_metadata(env, split="id" if "ID" in env.spec.id else "ood")
    timeline: list[dict[str, Any]] = []
    executed = 0
    success = False
    while executed < max_episode_steps and not success:
        env_obs = _wrap_obs(raw_obs, info, task="stack")
        with torch.inference_mode():
            features = model.extract_multilayer_llmd_features(env_obs, prior)
            scores = {name: float(llmd_score(feature, statistics[name])[0].item()) for name, feature in features.items()}
            predicted, _ = model.predict_action_batch(env_obs=env_obs, mode="eval", compute_values=False)
        chunk = _action_chunk(predicted, int(model.config.action_horizon))[:execute_horizon]
        chunk = np.clip(chunk, np.asarray(env.action_space.low), np.asarray(env.action_space.high)).astype(np.float32)
        alarms = {
            name: bool(thresholds is not None and score >= thresholds["layers"][name]["threshold"])
            for name, score in scores.items()
        }
        timeline.append({"chunk_index": len(timeline), "env_step": executed, "scores": scores, "alarms": alarms})
        for action in chunk:
            raw_obs, _reward, terminated, truncated, info = env.step(
                torch.as_tensor(action, device=env.unwrapped.device).unsqueeze(0)
            )
            executed += 1
            success = _bool(info.get("success", False))
            if success or _bool(terminated) or _bool(truncated):
                break
    return {
        "seed": seed,
        "success": success,
        "steps": executed,
        "alarms": {name: any(point["alarms"][name] for point in timeline) for name in statistics},
        "first_alarm_chunk": {
            name: next((point["chunk_index"] for point in timeline if point["alarms"][name]), None)
            for name in statistics
        },
        "timeline": timeline,
        **metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pi05-base", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("id", "ood"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--execute-horizon", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--target-successes", type=int, default=20)
    parser.add_argument("--max-attempts", type=int, default=200)
    parser.add_argument("--delta", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.calibrate == (args.thresholds is not None):
        raise ValueError("choose exactly one of --calibrate or --thresholds")
    if args.calibrate and args.split != "id":
        raise ValueError("calibration requires successful ID policy rollouts")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats, prior, payload = _load_statistics(args.statistics)
    statistics_sha = _sha256(args.statistics)
    thresholds = _load_thresholds(args.thresholds, statistics_sha)
    model = _load_model(args.checkpoint, args.norm_stats, args.pi05_base)
    env = _build_env(args.max_episode_steps, task="stack", split=args.split)
    episodes = []
    try:
        attempts = args.max_attempts if args.calibrate else args.episodes
        for index in range(attempts):
            episode = _run_episode(
                env=env, model=model, statistics=stats, prior=prior.to("cuda"),
                seed=args.seed + index, execute_horizon=args.execute_horizon,
                max_episode_steps=args.max_episode_steps, thresholds=thresholds,
            )
            episode["episode_index"] = index
            episodes.append(episode)
            print(f"[multilayer-llmd] {index + 1}/{attempts} seed={episode['seed']} success={int(episode['success'])}", flush=True)
            if args.calibrate and sum(row["success"] for row in episodes) >= args.target_successes:
                break
    finally:
        env.close()
        del model
        torch.cuda.empty_cache()
    result: dict[str, Any] = {
        "format": "stackcube_multilayer_llmd_rollout_v1",
        "statistics": str(args.statistics),
        "statistics_sha256": statistics_sha,
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "protocol": {"fixed_prior_seed": payload["fixed_prior_seed"], "execute_horizon": args.execute_horizon, "action_horizon": int(prior.shape[1])},
        "thresholds": thresholds,
        "metrics": {name: _metrics(episodes, name) for name in stats},
        "episodes": episodes,
    }
    if args.calibrate:
        successful = [row for row in episodes if row["success"]][: args.target_successes]
        if len(successful) != args.target_successes:
            (args.output_dir / "episodes.json").write_text(json.dumps(result, indent=2) + "\n")
            raise RuntimeError(f"strict calibration needs {args.target_successes} successful ID trajectories, got {len(successful)}")
        thresholds = {
            "format": "stackcube_multilayer_llmd_threshold_v1",
            "delta": args.delta,
            "conformal_target_coverage": 1.0 - args.delta,
            "target_successes": args.target_successes,
            "attempts": len(episodes),
            "checkpoint": str(args.checkpoint),
            "statistics": str(args.statistics),
            "statistics_sha256": statistics_sha,
            "fixed_prior_seed": payload["fixed_prior_seed"],
            "successful_seeds": [row["seed"] for row in successful],
            "layers": {},
        }
        rank = min(len(successful), math.ceil((len(successful) + 1) * (1.0 - args.delta)))
        for name in stats:
            maxima = [max(point["scores"][name] for point in row["timeline"]) for row in successful]
            thresholds["layers"][name] = {
                "threshold": constant_split_conformal_threshold([[value] for value in maxima], delta=args.delta),
                "order_statistic_rank": rank,
                "calibration_trajectory_maxima_bounds": _score_summary(maxima),
            }
        result["thresholds"] = thresholds
        (args.output_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2) + "\n")
    (args.output_dir / "episodes.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
