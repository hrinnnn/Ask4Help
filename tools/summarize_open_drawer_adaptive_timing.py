#!/usr/bin/env python3
"""Reconcile the one-model-per-anchor OpenDrawer timing sweep.

This is deliberately separate from the legacy three-seed/2500-step report.
It validates the adaptive checkpoint protocol, the fixed 100-ID/100-Grasp-OOD
evaluation denominators, and the frozen timing evidence before emitting a
comparison table.  DCA/EAS/DCE are intervention-quality descriptors; the
post-training success rates remain the downstream utility outcome.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np


ANCHORS_DEFAULT = (0, 50, 80, 120, 160, 220)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"invalid JSONL rows: {path}")
    return rows


def _checkpoint_from_marker(marker: Path) -> Path:
    text = marker.read_text(encoding="utf-8")
    match = re.search(r"(?:^|\s)checkpoint=([^\s]+)", text)
    if match is None:
        raise ValueError(f"checkpoint path missing in {marker}")
    path = Path(match.group(1))
    return path if path.is_absolute() else (marker.parent / path)


def _timeline_rows(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    rows = payload.get("timeline")
    if rows is None:
        rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"timeline rows missing: {path}")
    return [row for row in rows if isinstance(row, dict)]


def _audit_eval(path: Path, episodes: int) -> dict[str, Any]:
    errors: list[str] = []
    if not (path / "EVAL_COMPLETE").is_file():
        errors.append("missing EVAL_COMPLETE")
    split_rows: dict[str, dict[str, Any]] = {}
    for split in ("id", "grasp_ood"):
        output = path / split
        summary_path = output / "summary.json"
        if not summary_path.is_file():
            errors.append(f"missing {summary_path}")
            continue
        payload = read_json(summary_path)
        rows = payload.get("rows", [])
        if payload.get("episodes") != episodes or len(rows) != episodes:
            errors.append(f"denominator mismatch in {summary_path}")
        videos = output / "videos"
        if len(list(videos.glob("*.mp4"))) != episodes:
            errors.append(f"video denominator mismatch in {videos}")
        for row in rows:
            if not isinstance(row, dict):
                errors.append(f"non-object row in {summary_path}")
                continue
            for key in ("actions", "states", "timeline", "reset_metadata", "video"):
                artifact = Path(str(row.get(key, "")))
                if not artifact.is_file():
                    errors.append(f"missing {key} artifact in {summary_path}")
            try:
                actions = np.load(Path(str(row["actions"])))
                states = np.load(Path(str(row["states"])))
                if actions.ndim != 2 or actions.shape[1] != 8:
                    errors.append(f"invalid action shape {actions.shape} in {summary_path}")
                if states.ndim != 2 or states.shape != (len(actions) + 1, 9):
                    errors.append(f"invalid state shape {states.shape} in {summary_path}")
                timeline = _timeline_rows(Path(str(row["timeline"])))
                if len(timeline) != len(actions) + 1:
                    errors.append(f"timeline/action mismatch in {summary_path}")
                json.loads(Path(str(row["reset_metadata"])).read_text(encoding="utf-8"))
            except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"artifact read failed in {summary_path}: {exc}")
        split_rows[split] = {
            "episodes": payload.get("episodes"),
            "successes": payload.get("successes"),
            "success_rate": payload.get("success_rate"),
            "drawer_opened_rate": payload.get("drawer_opened_rate"),
            "grasp_rate": payload.get("grasp_rate"),
            "lift_rate": payload.get("lift_rate"),
            "in_target_rate": payload.get("in_target_rate"),
            "seed_start": payload.get("seed_start"),
        }
    return {"path": str(path), "splits": split_rows, "errors": errors, "pass": not errors}


def _accepted_episode_dirs(root: Path, anchor: int) -> list[Path]:
    return sorted((root / f"anchor_{anchor}" / "accepted").glob("episode_*/"))


def _task_feature(row: dict[str, Any]) -> np.ndarray:
    def vector(name: str, size: int) -> list[float]:
        value = np.asarray(row.get(name, [0.0] * size), dtype=np.float64).reshape(-1)
        result = value[:size].tolist()
        return result + [0.0] * (size - len(result))

    feature = vector("tcp_position", 3) + vector("object_position", 3)
    feature += vector("target_position", 3) + vector("drawer_qpos", 1)
    return np.asarray(feature, dtype=np.float64)


def _context(rows: list[dict[str, Any]]) -> np.ndarray:
    return _task_feature(rows[0]) if rows else np.zeros(10, dtype=np.float64)


def _dtw_average(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) == 0 or len(right) == 0:
        return float("nan")
    table = np.full((len(left) + 1, len(right) + 1), np.inf, dtype=np.float64)
    counts = np.zeros((len(left) + 1, len(right) + 1), dtype=np.int64)
    table[0, 0] = 0.0
    for i in range(1, len(left) + 1):
        for j in range(1, len(right) + 1):
            options = ((table[i - 1, j], counts[i - 1, j]),
                       (table[i, j - 1], counts[i, j - 1]),
                       (table[i - 1, j - 1], counts[i - 1, j - 1]))
            previous, previous_count = min(options, key=lambda item: item[0])
            table[i, j] = previous + float(np.linalg.norm(left[i - 1] - right[j - 1]))
            counts[i, j] = previous_count + 1
    return float(table[-1, -1] / max(1, counts[-1, -1]))


def _quality_metrics(collection_root: Path, anchors: tuple[int, ...]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    reference_dirs = _accepted_episode_dirs(collection_root, 0)
    if not reference_dirs:
        return {}, ["missing anchor_0 accepted episodes for DCA/EAS"]
    reference_rows: list[list[dict[str, Any]]] = []
    reference_lengths: list[int] = []
    reference_contexts: list[np.ndarray] = []
    for path in reference_dirs:
        try:
            rows = _timeline_rows(path / "task_state_timeline.json")
            if not rows:
                raise ValueError("empty task timeline")
            reference_rows.append(rows)
            reference_lengths.append(int(len(np.load(path / "actions.npy"))))
            reference_contexts.append(_context(rows))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"reference artifact failed {path}: {exc}")
    if not reference_rows:
        return {}, errors or ["empty reference bank"]
    matrix = np.concatenate([np.stack([_task_feature(row) for row in rows]) for rows in reference_rows], axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-6] = 1.0
    nominal_median = float(median(reference_lengths))
    pending: list[tuple[int, int, float, float, str]] = []
    for anchor in anchors:
        dirs = _accepted_episode_dirs(collection_root, anchor)
        if not dirs:
            errors.append(f"missing accepted episodes for anchor {anchor}")
            continue
        try:
            metadata_rows = read_jsonl(collection_root / f"anchor_{anchor}" / "accepted_experts.jsonl")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"missing accepted metadata for anchor {anchor}: {exc}")
            continue
        if len(metadata_rows) != len(dirs):
            errors.append(f"accepted metadata/episode mismatch for anchor {anchor}: {len(metadata_rows)} vs {len(dirs)}")
            continue
        for index, path in enumerate(dirs):
            try:
                rows = _timeline_rows(path / "task_state_timeline.json")
                raw = metadata_rows[index]
                takeover = int(raw.get("actual_takeover_step", raw.get("scheduled_takeover_step", anchor)))
                ref_index = index if anchor == 0 and index < len(reference_rows) else int(np.argmin([np.linalg.norm(_context(rows) - context) for context in reference_contexts]))
                ref = reference_rows[ref_index]
                start = min(max(0, takeover), len(ref) - 1)
                query = np.stack([_task_feature(row) for row in rows[max(0, min(takeover, len(rows) - 1)):]])
                target = np.stack([_task_feature(row) for row in ref[start:]])
                distance = _dtw_average(query / scale, target / scale)
                expert_steps = int(raw.get("expert_action_steps", len(np.load(path / "actions.npy"))))
                eas = max(0.0, 1.0 - expert_steps / max(nominal_median, 1.0))
                pending.append((anchor, expert_steps, distance, eas, str(path)))
            except (OSError, ValueError, KeyError, IndexError, json.JSONDecodeError) as exc:
                errors.append(f"quality artifact failed {path}: {exc}")
    finite_distances = [distance for _a, _s, distance, _e, _p in pending if math.isfinite(distance) and distance > 1e-12]
    sigma = float(median(finite_distances)) if finite_distances else 1.0
    rows: list[dict[str, Any]] = []
    for anchor, expert_steps, distance, eas, path in pending:
        dca = float(math.exp(-distance / max(sigma, 1e-12))) if math.isfinite(distance) else 0.0
        dce = float(2.0 * dca * eas / (dca + eas + 1e-12))
        rows.append({
            "anchor": anchor,
            "expert_action_steps": expert_steps,
            "nominal_median_action_steps": nominal_median,
            "dtw_distance": distance,
            "dca": dca,
            "eas": eas,
            "dce": dce,
            "source": path,
        })
    by_anchor: dict[str, Any] = {}
    for anchor in anchors:
        current = [row for row in rows if row["anchor"] == anchor]
        by_anchor[str(anchor)] = {
            "episodes": len(current),
            "expert_actions_mean": float(mean([row["expert_action_steps"] for row in current])) if current else None,
            "dca_mean": float(mean([row["dca"] for row in current])) if current else None,
            "eas_mean": float(mean([row["eas"] for row in current])) if current else None,
            "dce_mean": float(mean([row["dce"] for row in current])) if current else None,
        }
    return {
        "format": "open_drawer_adaptive_intervention_quality_v1",
        "reference_anchor": 0,
        "reference_episodes": len(reference_rows),
        "nominal_median_action_steps": nominal_median,
        "dca_scale": sigma,
        "dca_scale_rule": "median positive normalized task-state DTW distance across all accepted timing episodes",
        "by_anchor": by_anchor,
        "rows": rows,
    }, errors


def _d_path_rows(path: Path, anchors: tuple[int, ...], errors: list[str]) -> dict[str, Any] | None:
    if not path.is_file():
        errors.append(f"missing D-path summary: {path}")
        return None
    try:
        payload = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"invalid D-path summary: {exc}")
        return None
    available = {int(row.get("anchor")): row for row in payload.get("anchors", []) if isinstance(row, dict) and "anchor" in row}
    result: dict[str, Any] = {
        "threshold": payload.get("threshold"),
        "calibration_quantile": payload.get("calibration_quantile"),
        "persistence_decisions": payload.get("persistence_decisions"),
        "prefix_censored": payload.get("prefix_censored"),
        "policy_only": payload.get("policy_only"),
        "anchors": {},
    }
    threshold = float(payload["threshold"]) if payload.get("threshold") is not None else float("nan")
    for anchor in anchors:
        row = available.get(anchor)
        if row is None:
            errors.append(f"missing D-path anchor {anchor}")
            continue
        median_value = row.get("d_path_median")
        result["anchors"][str(anchor)] = {
            "d_path_median": median_value,
            "d_path_p25": row.get("d_path_p25"),
            "d_path_p75": row.get("d_path_p75"),
            "d_over_tau": (float(median_value) / threshold if median_value is not None and math.isfinite(threshold) and threshold > 0 else None),
            "crossing_observed_rate": row.get("crossing_observed_rate"),
            "phase_counts": row.get("phase_counts", {}),
        }
    return result


def _checkpoint_rows(root: Path, anchors: tuple[int, ...], seed: int, errors: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    frozen_path = root / "adaptive_steps.json"
    if not frozen_path.is_file():
        errors.append(f"missing {frozen_path}")
        return rows
    frozen = read_json(frozen_path).get("frozen_steps")
    if not isinstance(frozen, int) or frozen < 5000:
        errors.append(f"invalid frozen_steps={frozen}")
        return rows
    for anchor in anchors:
        directory = root / "training" / f"anchor_{anchor}" / f"seed_{seed}" / f"steps_{frozen}"
        marker = directory / "SEGMENT_COMPLETE"
        try:
            checkpoint = _checkpoint_from_marker(marker)
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
            checkpoint = directory / "missing_checkpoint"
        full_weights = checkpoint / "actor/model_state_dict/full_weights.pt"
        dcp = checkpoint / "actor/dcp_checkpoint"
        if not full_weights.is_file() or not dcp.is_dir():
            errors.append(f"incomplete checkpoint for anchor {anchor}: {checkpoint}")
        rows.append({
            "anchor": anchor,
            "seed": seed,
            "frozen_steps": frozen,
            "segment_root": str(directory),
            "checkpoint": str(checkpoint),
            "full_weights": str(full_weights),
            "dcp": str(dcp),
            "complete": full_weights.is_file() and dcp.is_dir(),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="adaptive execution root")
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--budget-root", type=Path, required=True)
    parser.add_argument("--policy-only-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchors", type=int, nargs="+", default=list(ANCHORS_DEFAULT))
    parser.add_argument("--seed", type=int, default=9301)
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    anchors = tuple(args.anchors)
    errors: list[str] = []
    if not (args.root / "ADAPTIVE_TIMING_TRAINING_COMPLETE").is_file():
        errors.append("missing ADAPTIVE_TIMING_TRAINING_COMPLETE")
    if not (args.formal_root / "TIMING_COLLECTION_COMPLETE").is_file():
        errors.append("missing TIMING_COLLECTION_COMPLETE")
    if not (args.formal_root / "AUDIT_PASS").is_file():
        errors.append("missing formal AUDIT_PASS")
    if not (args.budget_root / "BUDGET_AUDIT_PASS").is_file():
        errors.append("missing BUDGET_AUDIT_PASS")
    budget_path = args.budget_root / "budget_manifest.json"
    if not budget_path.is_file():
        errors.append(f"missing {budget_path}")
        budget = None
    else:
        budget = read_json(budget_path)
        selected = budget.get("selected_expert_actions", {})
        if any(int(selected.get(f"anchor_{anchor}", -1)) != 5006 for anchor in anchors):
            errors.append("formal budget does not match frozen 5006 action budget")

    frozen_payload = read_json(args.root / "adaptive_steps.json") if (args.root / "adaptive_steps.json").is_file() else {}
    frozen_steps = frozen_payload.get("frozen_steps")
    checkpoint_rows = _checkpoint_rows(args.root, anchors, args.seed, errors)
    evaluation_rows: list[dict[str, Any]] = []
    utility: dict[str, Any] = {}
    for index, anchor in enumerate(anchors):
        step_label = str(frozen_steps) if frozen_steps is not None else "unknown"
        evaluation_root = args.root / "evaluation" / f"anchor_{anchor}" / f"seed_{args.seed}" / f"steps_{step_label}"
        audit = _audit_eval(evaluation_root, args.episodes)
        evaluation_rows.append({"anchor": anchor, "seed": args.seed, **audit})
        errors.extend(f"anchor={anchor}: {error}" for error in audit["errors"])
        if audit["pass"]:
            id_row = audit["splits"]["id"]
            ood_row = audit["splits"]["grasp_ood"]
            utility[str(anchor)] = {
                "id_sr": id_row.get("success_rate"),
                "grasp_ood_sr": ood_row.get("success_rate"),
                "id_successes": id_row.get("successes"),
                "grasp_ood_successes": ood_row.get("successes"),
                "episodes": args.episodes,
            }

    d_path = _d_path_rows(args.formal_root / "d_path_summary.json", anchors, errors)
    quality, quality_errors = _quality_metrics(args.formal_root, anchors)
    errors.extend(quality_errors)
    if d_path is None:
        d_path = {"anchors": {}}
    for anchor in anchors:
        row = utility.get(str(anchor), {})
        row.update(d_path.get("anchors", {}).get(str(anchor), {}))
        row.update(quality.get("by_anchor", {}).get(str(anchor), {}) if quality else {})
        utility[str(anchor)] = row

    baseline = None
    if args.policy_only_root is not None and (args.policy_only_root / "summary.json").is_file():
        payload = read_json(args.policy_only_root / "summary.json")
        baseline = {
            "episodes": payload.get("episodes"),
            "successes": payload.get("successes"),
            "success_rate": payload.get("success_rate"),
            "status": "diagnostic_only_not_formal_100_episode_comparison",
        }

    payload = {
        "format": "open_drawer_adaptive_timing_final_reconciliation_v1",
        "protocol": {
            "anchors": list(anchors),
            "training_seed": args.seed,
            "models_per_anchor": 1,
            "minimum_steps": 5000,
            "frozen_steps": frozen_steps,
            "ood20_rule": "continue at +2500 while strict rate <= 0.40; freeze first rate > 0.40",
            "formal_episodes_per_split": args.episodes,
            "formal_budget_actions": 5006,
        },
        "budget_manifest": budget,
        "checkpoints": checkpoint_rows,
        "evaluations": evaluation_rows,
        "d_path": d_path,
        "intervention_quality": quality,
        "policy_only_baseline": baseline,
        "utility_by_anchor": utility,
        "errors": errors,
        "pass": not errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    root = args.root
    if errors:
        (root / "INDEPENDENT_RECONCILIATION_FAILED").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        raise SystemExit(1)
    lines = [
        "# OpenDrawer adaptive Grasp-OOD timing sweep",
        "",
        "One model per anchor, shared training seed 9301; DCA/EAS/DCE are intervention-quality descriptors.",
        "",
        "| Timing | D/τ_D | EAS | DCA | ID SR | Grasp-OOD SR |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for anchor in anchors:
        row = utility[str(anchor)]
        fmt = lambda value: "NA" if value is None else f"{float(value):.3f}"
        lines.append(
            f"| {anchor} | {fmt(row.get('d_over_tau'))} | {fmt(row.get('eas_mean'))} | "
            f"{fmt(row.get('dca_mean'))} | {fmt(row.get('id_sr'))} | {fmt(row.get('grasp_ood_sr'))} |"
        )
    (root / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "INDEPENDENT_RECONCILIATION_COMPLETE").write_text(
        "all adaptive timing checkpoints, evaluations, D-path and denominators independently audited\n",
        encoding="utf-8",
    )
    print(json.dumps({"pass": True, "utility_by_anchor": utility}, indent=2), flush=True)


if __name__ == "__main__":
    main()
