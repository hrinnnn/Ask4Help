#!/usr/bin/env python3
"""Generate shared airplane rollouts and raw independent-token PCA scores."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RLINF_ROOT = ROOT / "RLinf"
import sys

sys.path[:0] = [str(ROOT), str(RLINF_ROOT)]

from rlinf.envs.maniskill.pick_single_ycb_airplane_variants import (  # noqa: E402
    PICK_SINGLE_YCB_AIRPLANE_TASK,
    register_controlled_pick_single_ycb_airplane_variants,
    reset_metadata,
)
from toolkits.lerobot.collect_maniskill_pick_single_ycb_airplane_lerobot import (  # noqa: E402
    _build_env,
    write_episode_video_durably,
)
from toolkits.lerobot.collect_maniskill_peg_lerobot_joint import (  # noqa: E402
    MAIN_CAMERA_CANDIDATES,
    WRIST_CAMERA_CANDIDATES,
    _bool_scalar,
    _build_frames,
    _extract_record,
    _select_camera,
)
from tools.evaluate_pick_single_ycb_airplane_pi05 import model_observation  # noqa: E402
from tools.maniskill_pi05_vfd_online_awbc import _load_model  # noqa: E402
from tools.pick_single_ycb_airplane_eval_common import clip_action_chunk  # noqa: E402
from tools.pick_single_ycb_airplane_tokenwise_pca import (  # noqa: E402
    MAIN_METHODS,
    TokenwisePCAScorer,
    checkpoint_weights_path,
    sha256,
    sha256_path,
)
from rlinf.algorithms.vla_fail import LLMDStatistics, llmd_score  # noqa: E402


class FinalLLMDProbe:
    """Score the Action Expert's final tokens without altering policy actions."""

    def __init__(self, assets_path: Path, *, device: torch.device | str) -> None:
        payload = torch.load(assets_path, map_location="cpu", weights_only=False)
        statistics = LLMDStatistics.from_state_dict(payload["statistics"]["final_llmd"])
        target = torch.device(device)
        self.statistics = LLMDStatistics(
            mean=statistics.mean.to(target),
            precision=statistics.precision.to(target),
            ridge=statistics.ridge,
            num_observations=statistics.num_observations,
        )
        self.fixed_prior = torch.as_tensor(payload["fixed_prior"], device=target)

    def score(self, model: Any, env_obs: dict[str, Any]) -> float:
        features = model.extract_llmd_action_features(env_obs, self.fixed_prior)
        return float(llmd_score(features, self.statistics)[0].item())


def _grasped(env: Any) -> bool:
    """Read ManiSkill's instantaneous Panda grasp state without task labels."""

    value = env.unwrapped.agent.is_grasping(env.unwrapped.obj)
    return _bool_scalar(value)


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p05": None, "p50": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size), "min": float(array.min()), "p05": float(np.quantile(array, 0.05)),
        "p50": float(np.quantile(array, 0.50)), "p95": float(np.quantile(array, 0.95)), "max": float(array.max()),
    }


def _run_episode(
    *, env: Any, model: Any, scorer: TokenwisePCAScorer, final_llmd: FinalLLMDProbe | None,
    split: str, seed: int,
    execute_horizon: int, max_episode_steps: int, episode_index: int, video_dir: Path,
) -> dict[str, Any]:
    raw_obs, _info = env.reset(seed=seed)
    metadata = reset_metadata(env, split=split)
    low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    records, actions, timeline = [_extract_record(raw_obs)], [], []
    strict_success = ever_grasped = False
    executed = 0
    while executed < max_episode_steps and not strict_success:
        torch.cuda.synchronize()
        policy_start = time.perf_counter()
        with torch.inference_mode():
            env_obs = model_observation(raw_obs)
            predicted, result = model.predict_action_batch(
                env_obs=env_obs, mode="eval", compute_values=False, return_prefix_probes=True
            )
        torch.cuda.synchronize()
        policy_ms = (time.perf_counter() - policy_start) * 1000.0
        pca_start = time.perf_counter()
        scored = scorer.score_probe(result["prefix_probes"])
        torch.cuda.synchronize()
        pca_ms = (time.perf_counter() - pca_start) * 1000.0
        final_llmd_ms = 0.0
        if final_llmd is not None:
            final_llmd_start = time.perf_counter()
            scored["scores"]["final_llmd"] = final_llmd.score(model, env_obs)
            torch.cuda.synchronize()
            final_llmd_ms = (time.perf_counter() - final_llmd_start) * 1000.0
        chunk = clip_action_chunk(predicted.detach().float().cpu().numpy(), low, high, execute_horizon)
        timeline.append({
            "decision_index": len(timeline), "env_step": executed, "scores": scored["scores"],
            "modalities": scored["modalities"], "topk": scored["topk"], "policy_ms": policy_ms,
            "pca_score_ms": pca_ms, "final_llmd_score_ms": final_llmd_ms,
        })
        for action in chunk:
            raw_obs, _reward, terminated, truncated, info = env.step(
                torch.as_tensor(action, device=env.unwrapped.device).unsqueeze(0)
            )
            actions.append(action.copy())
            records.append(_extract_record(raw_obs))
            executed += 1
            ever_grasped |= _grasped(env)
            strict_success = _bool_scalar(info.get("success"))
            if strict_success or _bool_scalar(terminated) or _bool_scalar(truncated):
                break
    main_camera = _select_camera(records[0].obs, "", ("base_camera",) + MAIN_CAMERA_CANDIDATES, "main")
    wrist_camera = _select_camera(records[0].obs, "", ("hand_camera",) + WRIST_CAMERA_CANDIDATES, "wrist")
    frames = _build_frames(
        records=records, actions=actions, task=PICK_SINGLE_YCB_AIRPLANE_TASK,
        main_camera=main_camera, wrist_camera=wrist_camera,
    )
    video = write_episode_video_durably(
        frames, video_dir=video_dir, episode_index=episode_index, seed=seed, fps=10
    )
    return {
        "episode_index": episode_index, "seed": seed, "split": split, "steps": executed,
        "ever_grasped": bool(ever_grasped), "strict_success": bool(strict_success),
        "video": str(video), "timeline": timeline, **metadata,
    }


def _distributions(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    methods = tuple(episodes[0]["timeline"][0]["scores"])
    for method in methods:
        groups: dict[str, list[float]] = {}
        for episode in episodes:
            group = f"{episode['split']}_{'success' if episode['ever_grasped'] else 'failure'}"
            groups.setdefault(group, []).extend(float(point["scores"][method]) for point in episode["timeline"])
        result[method] = {name: _summary(values) for name, values in groups.items()}
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pi05-base", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--final-llmd-assets", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("both", "id", "ood"), default="both")
    parser.add_argument("--episodes-per-split", type=int, default=50)
    parser.add_argument("--id-seed", type=int, default=50000)
    parser.add_argument("--ood-seed", type=int, default=60000)
    parser.add_argument("--execute-horizon", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=250)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--control-freq", type=int, default=10)
    parser.add_argument("--sim-backend", choices=("physx_cpu", "gpu"), default="physx_cpu")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite rollout output: {args.output_dir}")
    if args.episodes_per_split != 50 or args.execute_horizon != 5 or args.max_episode_steps != 250:
        raise ValueError("registered protocol fixes 50 episodes/split, horizon=5, max_episode_steps=250")
    register_controlled_pick_single_ycb_airplane_variants()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.cuda.reset_peak_memory_stats()
    model = _load_model(args.checkpoint, args.norm_stats, args.pi05_base)
    try:
        # The independent bases must be resident beside the action model.  A
        # memory failure is a protocol outcome, not permission to silently
        # substitute a shared basis or a lower rank.
        preflight_start = time.perf_counter()
        scorer = TokenwisePCAScorer(args.assets_dir, device=args.device)
        final_llmd = (
            None
            if args.final_llmd_assets is None
            else FinalLLMDProbe(args.final_llmd_assets, device=args.device)
        )
        torch.cuda.synchronize()
        preflight = {
            "passed": True,
            "asset_load_ms": (time.perf_counter() - preflight_start) * 1000.0,
            "resident_pca_asset_bytes": scorer.resident_asset_bytes(),
            "cuda_allocated_bytes_after_load": int(torch.cuda.memory_allocated()),
            "cuda_reserved_bytes_after_load": int(torch.cuda.memory_reserved()),
        }
        (args.output_dir / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n", encoding="utf-8")
    except torch.OutOfMemoryError as error:
        preflight = {
            "passed": False,
            "error": str(error),
            "cuda_allocated_bytes": int(torch.cuda.memory_allocated()),
            "cuda_reserved_bytes": int(torch.cuda.memory_reserved()),
            "instruction": "Do not replace independent rank-1000 bases; provision sufficient GPU memory.",
        }
        (args.output_dir / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n", encoding="utf-8")
        del model
        torch.cuda.empty_cache()
        raise RuntimeError("independent token PCA resident-memory preflight failed") from error
    splits = (("id", args.id_seed), ("ood", args.ood_seed)) if args.split == "both" else (
        (("id", args.id_seed),) if args.split == "id" else (("ood", args.ood_seed),)
    )
    episodes: list[dict[str, Any]] = []
    try:
        for split, first_seed in splits:
            env_args = argparse.Namespace(
                split=split, image_size=args.image_size, control_freq=args.control_freq,
                max_episode_steps=args.max_episode_steps, sim_backend=args.sim_backend,
            )
            env = _build_env(env_args, control_mode="pd_joint_delta_pos")
            try:
                for offset in range(args.episodes_per_split):
                    episode = _run_episode(
                        env=env, model=model, scorer=scorer, final_llmd=final_llmd, split=split, seed=first_seed + offset,
                        execute_horizon=args.execute_horizon, max_episode_steps=args.max_episode_steps,
                        episode_index=len(episodes), video_dir=args.output_dir / "videos",
                    )
                    episodes.append(episode)
                    print(
                        f"[airplane-tokenwise-pca] split={split} episode={offset + 1}/50 "
                        f"seed={episode['seed']} grasp={int(episode['ever_grasped'])}", flush=True,
                    )
            finally:
                env.close()
    finally:
        peak_bytes = int(torch.cuda.max_memory_allocated())
        del model
        torch.cuda.empty_cache()

    result = {
        "format": "pick_single_ycb_airplane_tokenwise_pca_rollouts_v1",
        "task": "pick_single_ycb_airplane", "success_label": "ever_grasped",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(checkpoint_weights_path(args.checkpoint)),
        "norm_stats": str(args.norm_stats), "norm_stats_sha256": sha256_path(args.norm_stats),
        "assets_dir": str(args.assets_dir), "assets_manifest_sha256": sha256(args.assets_dir / "assets_manifest.json"),
        "protocol": {"requested_split": args.split, "id_seeds": [args.id_seed, args.id_seed + 49], "ood_seeds": [args.ood_seed, args.ood_seed + 49],
                     "execute_horizon": args.execute_horizon, "max_episode_steps": args.max_episode_steps,
                     "threshold_calibration": "none_posthoc_scan_only"},
        "runtime": {"resident_pca_asset_bytes": scorer.resident_asset_bytes(), "peak_cuda_bytes": peak_bytes},
        "grasp_rates": {
            split: (sum(row["ever_grasped"] for row in episodes if row["split"] == split) / count if count else None)
            for split in ("id", "ood")
            for count in (sum(row["split"] == split for row in episodes),)
        },
        "strict_success_rates": {
            split: (sum(row["strict_success"] for row in episodes if row["split"] == split) / count if count else None)
            for split in ("id", "ood")
            for count in (sum(row["split"] == split for row in episodes),)
        },
        "score_distributions": _distributions(episodes), "episodes": episodes,
    }
    (args.output_dir / "episodes.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in {"episodes", "score_distributions"}}, indent=2))


if __name__ == "__main__":
    main()
