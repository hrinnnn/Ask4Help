#!/usr/bin/env python3
"""Analyze expert-reference gripper-pose deviation for X-VLA OOD rollouts.

The score uses only end-effector position/orientation, gripper width, and
finite-difference velocity. Simulator phase labels are retained only for an
audit of pre-grasp/post-grasp timing and are never used in the ERD score.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


POSE_RE = re.compile(r"episode_(?P<episode>\d+)_seed_(?P<seed>\d+)\.npz$")
DECISION_STEPS = np.arange(0, 151, 5, dtype=np.int32)


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.maximum(norm, 1e-12)


def _quat_inverse(q: np.ndarray) -> np.ndarray:
    q = _quat_normalize(q)
    result = q.copy()
    result[..., 1:] *= -1.0
    return result


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def _quat_rotvec(q: np.ndarray) -> np.ndarray:
    """Return the shortest rotation vector for wxyz quaternions."""

    q = _quat_normalize(q)
    q = np.where(q[..., :1] < 0.0, -q, q)
    w = np.clip(q[..., 0], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    xyz = q[..., 1:]
    sin_half = np.linalg.norm(xyz, axis=-1)
    scale = np.divide(
        angle,
        sin_half,
        out=np.full_like(angle, 2.0),
        where=sin_half > 1e-9,
    )
    return xyz * scale[..., None]


def _pose_series(payload: dict[str, np.ndarray], fps: float) -> dict[str, np.ndarray]:
    position = np.asarray(payload["position"], dtype=np.float64)
    quaternion = _quat_normalize(np.asarray(payload["quaternion_wxyz"], dtype=np.float64))
    width = np.asarray(payload["gripper_width"], dtype=np.float64).reshape(-1, 1)
    if len(position) != len(quaternion) or len(position) != len(width):
        raise ValueError("pose arrays have inconsistent lengths")
    linear_velocity = np.vstack(
        [np.zeros((1, 3), dtype=np.float64), np.diff(position, axis=0) * fps]
    )
    relative_quaternion = _quat_multiply(quaternion[1:], _quat_inverse(quaternion[:-1]))
    angular_velocity = np.vstack(
        [
            np.zeros((1, 3), dtype=np.float64),
            _quat_rotvec(relative_quaternion) * fps,
        ]
    )
    return {
        "position": position,
        "quaternion": quaternion,
        "width": width,
        "linear_velocity": linear_velocity,
        "angular_velocity": angular_velocity,
    }


def _pose_residual(
    query: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    query_index: int,
    reference_index: int,
) -> np.ndarray:
    relative_quaternion = _quat_multiply(
        query["quaternion"][query_index],
        _quat_inverse(reference["quaternion"][reference_index]),
    )
    orientation_error = _quat_rotvec(relative_quaternion)
    return np.concatenate(
        [
            query["position"][query_index] - reference["position"][reference_index],
            orientation_error,
            query["width"][query_index] - reference["width"][reference_index],
            query["linear_velocity"][query_index]
            - reference["linear_velocity"][reference_index],
            query["angular_velocity"][query_index]
            - reference["angular_velocity"][reference_index],
        ]
    )


def _load_pose_root(root: Path) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for path in sorted((root / "pose").glob("*.npz")):
        match = POSE_RE.search(path.name)
        if match is None:
            continue
        with np.load(path) as payload:
            arrays = {key: np.asarray(payload[key]) for key in payload.files}
        if "object_position" not in arrays or "target_position" not in arrays:
            raise ValueError(f"missing geometry arrays in {path}")
        result[int(match.group("seed"))] = {
            "seed": int(match.group("seed")),
            "episode_index": int(match.group("episode")),
            "path": str(path),
            "arrays": arrays,
        }
    if not result:
        raise ValueError(f"no pose files found under {root / 'pose'}")
    return result


def _load_summary(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(row["episode_index"]): row for row in payload.get("rows", [])}


def _context(item: dict[str, Any]) -> np.ndarray:
    arrays = item["arrays"]
    return np.concatenate(
        [
            np.asarray(arrays["object_position"])[0].reshape(-1)[:3],
            np.asarray(arrays["target_position"])[0].reshape(-1)[:3],
        ]
    ).astype(np.float64)


def _context_scale(contexts: np.ndarray) -> np.ndarray:
    median = np.median(contexts, axis=0)
    scale = 1.4826 * np.median(np.abs(contexts - median), axis=0)
    scale[scale < 1e-4] = 1.0
    return scale


def _context_distance(left: np.ndarray, right: np.ndarray, scale: np.ndarray) -> float:
    return float(np.linalg.norm((left - right) / scale))


def _local_distance(residual: np.ndarray, scale: np.ndarray) -> float:
    # Position/orientation/width are used for phase matching.  Velocity is
    # intentionally excluded from the alignment objective to avoid matching a
    # fast but wrong learner state to a future expert state.
    return float(np.linalg.norm(residual[:7] / scale[:7]))


def _causal_align(
    query: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    scale: np.ndarray,
    *,
    max_forward_jump: int = 5,
    max_time_lead: int = 0,
) -> np.ndarray:
    indices: list[int] = []
    previous = 0
    query_length = len(query["position"])
    reference_length = len(reference["position"])
    for query_index in range(query_length):
        upper = min(
            reference_length - 1,
            previous + max_forward_jump,
            query_index + max_time_lead,
        )
        candidates = range(previous, upper + 1)
        current = min(
            candidates,
            key=lambda index: _local_distance(
                _pose_residual(query, reference, query_index, index), scale
            ),
        )
        indices.append(current)
        previous = current
    return np.asarray(indices, dtype=np.int32)


def _pairwise_expert_residuals(
    experts: list[dict[str, Any]],
    contexts: np.ndarray,
    context_scale: np.ndarray,
    fps: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    provisional: list[tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]] = []
    for index, item in enumerate(experts):
        features = _pose_series(item["arrays"], fps)
        distances = [
            _context_distance(contexts[index], contexts[j], context_scale)
            if index != j
            else float("inf")
            for j in range(len(experts))
        ]
        peer_index = int(np.argmin(distances))
        peer_features = _pose_series(experts[peer_index]["arrays"], fps)
        provisional.append(
            (
                features,
                peer_features,
                {
                    "seed": item["seed"],
                    "peer_seed": experts[peer_index]["seed"],
                    "context_distance": distances[peer_index],
                },
            )
        )
    raw_residuals: list[np.ndarray] = []
    pair_rows: list[dict[str, Any]] = []
    for features, peer_features, metadata in provisional:
        indices = _causal_align(
            features,
            peer_features,
            np.ones(13, dtype=np.float64),
        )
        usable = min(len(features["position"]), len(indices))
        raw_residuals.append(
            np.stack(
                [_pose_residual(features, peer_features, i, int(indices[i])) for i in range(usable)],
                axis=0,
            )
        )
        pair_rows.append({**metadata, "aligned_steps": usable, "alignment_tail": int(indices[-1])})
    return np.concatenate(raw_residuals, axis=0), pair_rows


def _robust_scales(residuals: np.ndarray) -> np.ndarray:
    scale = 1.4826 * np.median(np.abs(residuals), axis=0)
    # Floors are measurement/control tolerances, not simulator task values.
    floors = np.asarray(
        [1e-3] * 3 + [1e-2] * 3 + [1e-3] + [1e-2] * 3 + [1e-1] * 3,
        dtype=np.float64,
    )
    return np.maximum(scale, floors)


def _threshold_from_expert(
    experts: list[dict[str, Any]],
    context_scale: np.ndarray,
    feature_scale: np.ndarray,
    fps: float,
    *,
    decision_horizon: int,
) -> tuple[float, list[float]]:
    values: list[float] = []
    contexts = np.asarray([_context(item) for item in experts])
    for index, item in enumerate(experts):
        distances = [
            _context_distance(contexts[index], contexts[j], context_scale)
            if index != j
            else float("inf")
            for j in range(len(experts))
        ]
        peer = experts[int(np.argmin(distances))]
        query = _pose_series(item["arrays"], fps)
        reference = _pose_series(peer["arrays"], fps)
        indices = _causal_align(query, reference, feature_scale)
        for step in DECISION_STEPS:
            if step >= len(query["position"]):
                break
            residual = _pose_residual(query, reference, int(step), int(indices[min(step, len(indices) - 1)]))
            values.append(float(np.linalg.norm(residual / feature_scale)))
            if step >= decision_horizon:
                break
    threshold = float(np.quantile(np.asarray(values), 0.95))
    return threshold, values


def _first_persistent_crossing(
    distances: np.ndarray,
    *,
    threshold: float,
    persistence: int,
    horizon: int,
) -> int | None:
    steps = DECISION_STEPS[DECISION_STEPS <= horizon]
    values = distances[: len(steps)]
    for index in range(0, len(values) - persistence + 1):
        if np.all(values[index : index + persistence] > threshold):
            return int(steps[index])
    return None


def _phase_label(arrays: dict[str, np.ndarray], step: int, task: str) -> str:
    index = min(step, len(arrays["phase_flag_1"]) - 1)
    if np.any(np.asarray(arrays["phase_flag_1"][: index + 1]) > 0.5):
        return "post_grasp_or_recovery"
    return "pre_grasp"


def _irreversibility(arrays: dict[str, np.ndarray], task: str) -> dict[str, Any]:
    grasped = np.asarray(arrays["phase_flag_1"], dtype=np.float64) > 0.5
    second = np.asarray(arrays["phase_flag_2"], dtype=np.float64) > 0.5
    object_z = np.asarray(arrays["object_position"], dtype=np.float64)[:, 2]
    ever_grasped = False
    ever_lifted = False
    best_stage = 0
    last_progress = 0
    for step in range(len(grasped)):
        current = bool(grasped[step])
        on_target = bool(second[step]) if task == "stackcube" else bool(second[step])
        if task == "stackcube":
            stage = 3 if on_target else 2 if object_z[step] >= 0.07 else 1 if current else 0
            ever_grasped = ever_grasped or current
            ever_lifted = ever_lifted or (current and object_z[step] >= 0.07)
            if stage > best_stage:
                best_stage, last_progress = stage, step
            if ever_grasped and not current and not on_target and object_z[step] < 0.06:
                return {"step": step, "reason": "dropped_after_grasp", "censored": False}
            if ever_grasped and step - last_progress >= 30:
                return {"step": step, "reason": "stalled_after_progress", "censored": False}
            if step >= 75:
                return {"step": step, "reason": "episode_timeout", "censored": False}
        else:
            if ever_grasped and not current and object_z[step] <= object_z[0] + 0.01:
                return {"step": step, "reason": "grasp_lost", "censored": False}
            ever_grasped = ever_grasped or current
    return {"step": len(grasped) - 1, "reason": "right_censored", "censored": True}


def _detector_summary(
    path: Path | None,
    pose_by_episode: dict[int, dict[str, Any]],
    task: str,
) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    output: dict[str, Any] = {}
    for name, method in payload.get("methods", {}).items():
        detected: list[int] = []
        phase_counts = {"pre_grasp": 0, "post_grasp_or_recovery": 0, "unknown": 0}
        lead_times: list[int] = []
        late_alarms = 0
        for detail in method.get("episodes_detail", []):
            if not detail.get("alarm_observed"):
                continue
            step = int(detail["alarm_step"])
            detected.append(step)
            item = pose_by_episode.get(int(detail["episode_index"]))
            if item is None:
                phase_counts["unknown"] += 1
            else:
                label = _phase_label(item["arrays"], step, task)
                phase_counts[label] += 1
                irreversibility = _irreversibility(item["arrays"], task)
                if not irreversibility["censored"]:
                    lead = int(irreversibility["step"] - step)
                    lead_times.append(lead)
                    late_alarms += int(lead < 0)
        output[name] = {
            "episodes": int(method.get("episodes", 0)),
            "observed": len(detected),
            "miss": int(method.get("episodes", 0)) - len(detected),
            "mean_step": None if not detected else float(np.mean(detected)),
            "median_step": None if not detected else float(np.median(detected)),
            "p25_step": None if not detected else float(np.quantile(detected, 0.25)),
            "p75_step": None if not detected else float(np.quantile(detected, 0.75)),
            "phase_counts": phase_counts,
            "lead_time_observed_count": len(lead_times),
            "lead_time_median_observed": None if not lead_times else float(np.median(lead_times)),
            "late_alarms": late_alarms,
        }
    return output


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    expert_map = _load_pose_root(args.expert_root)
    learner_map = _load_pose_root(args.learner_root)
    expert = [expert_map[key] for key in sorted(expert_map)]
    learner = [learner_map[key] for key in sorted(learner_map)]
    expert_contexts = np.asarray([_context(item) for item in expert])
    context_scale = _context_scale(expert_contexts)
    expert_residuals, pair_rows = _pairwise_expert_residuals(
        expert, expert_contexts, context_scale, args.fps
    )
    feature_scale = _robust_scales(expert_residuals)
    expert_peer_context_distances = np.asarray(
        [row["context_distance"] for row in pair_rows], dtype=np.float64
    )
    context_support_threshold = float(np.quantile(expert_peer_context_distances, 0.95))
    threshold, threshold_values = _threshold_from_expert(
        expert,
        context_scale,
        feature_scale,
        args.fps,
        decision_horizon=args.horizon,
    )
    summary_rows = _load_summary(args.learner_summary)
    pose_by_episode = {item["episode_index"]: item for item in learner}
    learner_context_distances: list[float] = []
    rows: list[dict[str, Any]] = []
    for item in learner:
        arrays = item["arrays"]
        features = _pose_series(arrays, args.fps)
        context = _context(item)
        distances = [
            _context_distance(context, _context(ref), context_scale) for ref in expert
        ]
        nearest_index = int(np.argmin(distances))
        nearest_context_distance = float(distances[nearest_index])
        context_supported = nearest_context_distance <= context_support_threshold
        reference = _pose_series(expert[nearest_index]["arrays"], args.fps)
        match_indices = _causal_align(
            features,
            reference,
            feature_scale,
            max_forward_jump=args.max_forward_jump,
        )
        values = np.asarray(
            [
                float(
                    np.linalg.norm(
                        _pose_residual(
                            features,
                            reference,
                            min(int(step), len(features["position"]) - 1),
                            int(match_indices[min(int(step), len(match_indices) - 1)]),
                        )
                        / feature_scale
                    )
                )
                for step in DECISION_STEPS
            ],
            dtype=np.float64,
        )
        alarm = _first_persistent_crossing(
            values,
            threshold=threshold,
            persistence=args.persistence,
            horizon=args.horizon,
        )
        irreversibility = _irreversibility(arrays, args.task)
        episode_summary = summary_rows.get(item["episode_index"], {})
        alarm_current_grasped = (
            None
            if alarm is None
            else bool(float(arrays["phase_flag_1"][min(alarm, len(arrays["phase_flag_1"]) - 1)]) > 0.5)
        )
        alarm_ever_grasped = (
            None
            if alarm is None
            else bool(np.any(np.asarray(arrays["phase_flag_1"][: alarm + 1]) > 0.5))
        )
        row = {
            "episode_index": item["episode_index"],
            "seed": item["seed"],
            "context_distance_to_nearest_expert": distances[nearest_index],
            "context_supported": context_supported,
            "reference_status": "supported" if context_supported else "reference_unsupported",
            "nearest_expert_seed": expert[nearest_index]["seed"],
            "erd_alarm_step": alarm,
            "erd_alarm_observed": alarm is not None,
            "erd_alarm_phase": None if alarm is None else _phase_label(arrays, alarm, args.task),
            "grasped_at_alarm": alarm_current_grasped,
            "ever_grasped_before_alarm": alarm_ever_grasped,
            "irreversibility": irreversibility,
            "lead_time": None if alarm is None else int(irreversibility["step"] - alarm),
            "lead_time_censored": bool(irreversibility["censored"]),
            "first_grasp_step": next(
                (int(index) for index, value in enumerate(arrays["phase_flag_1"]) if float(value) > 0.5),
                None,
            ),
            "max_step": int(len(features["position"]) - 1),
            "success_label": episode_summary.get("success", episode_summary.get("ever_grasped")),
            "distance_at_decision_steps": values.tolist(),
            "match_indices_at_decision_steps": [
                int(match_indices[min(step, len(match_indices) - 1)]) for step in DECISION_STEPS
            ],
        }
        rows.append(row)
        learner_context_distances.append(distances[nearest_index])
    detected = [row["erd_alarm_step"] for row in rows if row["erd_alarm_step"] is not None]
    phase_counts = {"pre_grasp": 0, "post_grasp_or_recovery": 0}
    for row in rows:
        if row["erd_alarm_phase"] in phase_counts:
            phase_counts[row["erd_alarm_phase"]] += 1
    payload = {
        "format": "xvla_erd_pose_analysis_v1",
        "task": args.task,
        "reference": {
            "episodes": len(expert),
            "root": str(args.expert_root),
            "context_matching": "nearest initial object/target context",
            "feature_definition": "position, orientation-log, gripper-width, linear-velocity, angular-velocity",
            "feature_scale": feature_scale.tolist(),
            "context_scale": context_scale.tolist(),
            "expert_peer_context_distance_q95": context_support_threshold,
            "pair_rows": pair_rows,
        },
        "threshold": {
            "value": threshold,
            "rule": "q=.95 of leave-one-context-out expert residuals at decision steps",
            "calibration_values": threshold_values,
            "persistence_decisions": args.persistence,
        },
        "learner": {
            "episodes": len(learner),
            "root": str(args.learner_root),
            "horizon": args.horizon,
            "decision_stride": 5,
            "fps": args.fps,
            "nearest_context_distance_mean": float(np.mean(learner_context_distances)),
            "nearest_context_distance_median": float(np.median(learner_context_distances)),
            "unsupported_contexts": sum(not row["context_supported"] for row in rows),
            "unsupported_context_rate": float(
                np.mean([not row["context_supported"] for row in rows])
            ),
        },
        "erd_summary": {
            "observed": len(detected),
            "miss_or_censored": len(rows) - len(detected),
            "observed_supported_context": sum(
                row["erd_alarm_step"] is not None and row["context_supported"]
                for row in rows
            ),
            "unsupported_contexts": sum(not row["context_supported"] for row in rows),
            "mean_step_detected_only": None if not detected else float(np.mean(detected)),
            "median_step_detected_only": None if not detected else float(np.median(detected)),
            "p25_step_detected_only": None if not detected else float(np.quantile(detected, 0.25)),
            "p75_step_detected_only": None if not detected else float(np.quantile(detected, 0.75)),
            "phase_counts": phase_counts,
            "irreversibility_events": sum(not row["irreversibility"]["censored"] for row in rows),
            "lead_time_observed_count": sum(row["lead_time"] is not None and not row["lead_time_censored"] for row in rows),
            "lead_time_median_observed": (
                float(np.median([row["lead_time"] for row in rows if row["lead_time"] is not None and not row["lead_time_censored"]]))
                if any(row["lead_time"] is not None and not row["lead_time_censored"] for row in rows)
                else None
            ),
        },
        "detector_comparison": _detector_summary(args.detector_json, pose_by_episode, args.task),
        "rows": rows,
        "limitations": [
            "ERD score uses replayed pose features only; simulator phase flags are audit labels, not score inputs.",
            "Expert and learner OOD seeds are disjoint; nearest-context matching is used instead of exact pairing.",
            "Grab Plane episodes without an identifiable drop event are right-censored for lead time.",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "erd_pose_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("stackcube", "airplane"), required=True)
    parser.add_argument("--expert-root", type=Path, required=True)
    parser.add_argument("--learner-root", type=Path, required=True)
    parser.add_argument("--learner-summary", type=Path)
    parser.add_argument("--detector-json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--persistence", type=int, default=2)
    parser.add_argument("--max-forward-jump", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(analyze(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
