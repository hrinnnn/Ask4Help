#!/usr/bin/env python3
"""Compute a lightweight, auditable D-path summary for timing collections.

The primary value is the per-anchor distribution of expert-reference TCP
deviation.  This diagnostic intentionally does not infer a full policy-only
future after takeover; it reports only prefix observations available before
each scheduled anchor and marks the resulting critical-time estimate as
prefix-censored when the threshold has not yet been crossed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _context(reset: dict[str, Any]) -> np.ndarray:
    obj = reset.get("object_pose", {})
    target = reset.get("target_pose", {})
    obj_p = np.asarray(obj.get("p", [0, 0, 0]), dtype=np.float64).reshape(-1)[:3]
    target_p = np.asarray(target.get("p", [0, 0, 0]), dtype=np.float64).reshape(-1)[:3]
    yaw = float(obj.get("yaw_deg", 0.0))
    return np.asarray([*obj_p[:2], yaw / 90.0, *target_p[:2]], dtype=np.float64)


def _feature(row: dict[str, Any]) -> np.ndarray:
    # TCP position is the deployable core. Orientation is intentionally kept
    # out of this first diagnostic because quaternion sign/phase handling is
    # task-specific; contact/lifecycle channels remain separate audit fields.
    return np.asarray(row["tcp_position"], dtype=np.float64).reshape(-1)[:3]


def _nearest_reference(context: np.ndarray, references: list[tuple[np.ndarray, list[dict[str, Any]]]]) -> int:
    if not references:
        raise ValueError("empty expert reference bank")
    distances = [float(np.linalg.norm(context - item[0])) for item in references]
    return int(np.argmin(distances))


def _quantile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q)) if values else float("nan")


def _persistent_crossing(values: list[float], steps: list[int], threshold: float, persistence: int = 2) -> int | None:
    for i in range(0, len(values) - persistence + 1):
        window = values[i : i + persistence]
        if len(window) == persistence and all(np.isfinite(window)) and all(value > threshold for value in window):
            return int(steps[i])
    return None


def load_episode(path: Path) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    timeline = _read(path / "task_state_timeline.json")
    reset = _read(path / "reset_metadata.json")
    rows = timeline.get("rows", [])
    if not rows:
        raise ValueError(f"empty task timeline: {path}")
    return _context(reset), rows, reset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference-anchor", type=int, default=0)
    parser.add_argument("--anchors", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-only-root", type=Path)
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--persistence", type=int, default=2)
    args = parser.parse_args()

    ref_root = args.root / f"anchor_{args.reference_anchor}" / "accepted"
    reference_paths = sorted(ref_root.glob("episode_*/"))
    if not reference_paths:
        raise FileNotFoundError(f"no reference episodes under {ref_root}")
    references: list[tuple[np.ndarray, list[dict[str, Any]]]] = []
    for path in reference_paths:
        context, rows, _reset = load_episode(path)
        references.append((context, rows))

    # Leave-one-out natural expert variation at matched absolute decision
    # steps.  A robust q95 of these residuals is the frozen diagnostic scale.
    calibration: list[float] = []
    for i, (context, rows) in enumerate(references):
        peers = [(c, r) for j, (c, r) in enumerate(references) if j != i]
        if not peers:
            continue
        peer = peers[int(np.argmin([np.linalg.norm(context - c) for c, _r in peers]))][1]
        limit = min(len(rows), len(peer))
        for step in range(limit):
            calibration.append(float(np.linalg.norm(_feature(rows[step]) - _feature(peer[step]))))
    threshold = _quantile(calibration, args.quantile)

    anchor_results: list[dict[str, Any]] = []
    for anchor in args.anchors:
        episode_root = args.root / f"anchor_{anchor}" / "accepted"
        paths = sorted(episode_root.glob("episode_*/"))
        rows_out: list[dict[str, Any]] = []
        d_values: list[float] = []
        crossings: list[int | None] = []
        phases: list[str] = []
        for path in paths:
            context, rows, reset = load_episode(path)
            ref_index = _nearest_reference(context, references)
            ref_rows = references[ref_index][1]
            # Only use the policy prefix.  Expert continuation begins at the
            # scheduled anchor and must not leak into this timing score.
            prefix = rows[: min(anchor + 1, len(rows))]
            limit = min(len(prefix), len(ref_rows))
            steps = list(range(0, limit, 5))
            values = [float(np.linalg.norm(_feature(prefix[s]) - _feature(ref_rows[s]))) for s in steps]
            at_anchor = values[-1] if values else float("nan")
            d_values.append(at_anchor)
            crossing = _persistent_crossing(values, steps, threshold, args.persistence)
            crossings.append(crossing)
            phases.append(prefix[-1].get("phase", "unknown") if prefix else "unknown")
            rows_out.append({
                "seed": reset.get("seed"),
                "d_path_at_anchor": at_anchor,
                "persistent_crossing_step_within_prefix": crossing,
                "phase_at_anchor": phases[-1],
                "reference_index": ref_index,
            })
        anchor_results.append({
            "anchor": anchor,
            "episodes": len(rows_out),
            "d_path_mean": float(np.mean(d_values)) if d_values else None,
            "d_path_median": float(median(d_values)) if d_values else None,
            "d_path_p25": float(np.quantile(d_values, 0.25)) if d_values else None,
            "d_path_p75": float(np.quantile(d_values, 0.75)) if d_values else None,
            "threshold": threshold,
            "crossing_observed_rate": float(sum(value is not None for value in crossings) / len(crossings)) if crossings else None,
            "phase_counts": {phase: phases.count(phase) for phase in sorted(set(phases))},
            "rows": rows_out,
        })
    policy_only_summary: dict[str, Any] | None = None
    if args.policy_only_root is not None:
        policy_rows: list[dict[str, Any]] = []
        for path in sorted((args.policy_only_root / "episodes").glob("episode_*/")):
            context, rows, reset = load_episode(path)
            ref_index = _nearest_reference(context, references)
            ref_rows = references[ref_index][1]
            limit = min(len(rows), len(ref_rows))
            steps = list(range(0, limit, 5))
            values = [float(np.linalg.norm(_feature(rows[s]) - _feature(ref_rows[s]))) for s in steps]
            crossing = _persistent_crossing(values, steps, threshold, args.persistence)
            policy_rows.append({"seed": reset.get("seed"), "crossing_step": crossing, "steps": steps, "d_path": values, "reference_index": ref_index})
        crossings = [row["crossing_step"] for row in policy_rows if row["crossing_step"] is not None]
        policy_only_summary = {
            "episodes": len(policy_rows),
            "crossing_observed": len(crossings),
            "crossing_observed_rate": len(crossings) / len(policy_rows) if policy_rows else None,
            "crossing_median": float(median(crossings)) if crossings else None,
            "crossing_p25": float(np.quantile(crossings, 0.25)) if crossings else None,
            "crossing_p75": float(np.quantile(crossings, 0.75)) if crossings else None,
            "rows": policy_rows,
        }

    payload = {
        "format": "open_drawer_grasp_timing_d_path_summary_v1",
        "reference_anchor": args.reference_anchor,
        "reference_episodes": len(references),
        "calibration_quantile": args.quantile,
        "calibration_residual_count": len(calibration),
        "threshold": threshold,
        "persistence_decisions": args.persistence,
        "prefix_censored": True,
        "anchors": anchor_results,
        "policy_only": policy_only_summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("threshold", "calibration_residual_count", "reference_episodes")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
