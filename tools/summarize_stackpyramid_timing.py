#!/usr/bin/env python3
"""Summarize auditable DCA/EAS/DCE proxies for gated expert suffixes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def _groups(handle: h5py.File) -> list[str]:
    return sorted((name for name in handle if name.startswith("traj_")), key=lambda name: int(name.rsplit("_", 1)[1]))


def _states(path: Path) -> tuple[list[np.ndarray], list[int]]:
    sequences: list[np.ndarray] = []
    lengths: list[int] = []
    with h5py.File(path, "r") as handle:
        for name in _groups(handle):
            group = handle[name]
            state = np.asarray(group["obs/state"], dtype=np.float32)[:, :8]
            actions = np.asarray(group["actions"], dtype=np.float32)
            if state.shape[0] != actions.shape[0] + 1:
                raise ValueError(f"boundary mismatch in {path}:{name}")
            sequences.append(state)
            lengths.append(int(actions.shape[0]))
    return sequences, lengths


def _nearest_distance(query: np.ndarray, reference: np.ndarray, block: int = 256) -> np.ndarray:
    values = []
    for start in range(0, len(query), block):
        distances = np.linalg.norm(query[start : start + block, None] - reference[None], axis=-1)
        values.append(distances.min(axis=1))
    return np.concatenate(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nominal-h5", type=Path, required=True)
    parser.add_argument("--expert-h5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", default="bridge_pca")
    parser.add_argument("--split", required=True)
    args = parser.parse_args()

    nominal, nominal_lengths = _states(args.nominal_h5)
    expert, expert_lengths = _states(args.expert_h5)
    nominal_matrix = np.concatenate(nominal, axis=0)
    mean = nominal_matrix.mean(axis=0)
    std = nominal_matrix.std(axis=0)
    std[std < 1e-5] = 1.0
    nominal_z = [(sequence - mean) / std for sequence in nominal]
    expert_z = [(sequence - mean) / std for sequence in expert]
    reference = np.concatenate(nominal_z, axis=0)
    # Estimate natural nominal variation from adjacent anchors within each
    # nominal episode, avoiding artificial cross-episode jumps.
    adjacent = np.concatenate([np.linalg.norm(np.diff(sequence, axis=0), axis=1) for sequence in nominal])
    positive_adjacent = adjacent[adjacent > 1e-8]
    sigma_nom = float(np.median(positive_adjacent) if len(positive_adjacent) else 1.0)
    nominal_length = float(np.median(nominal_lengths))

    rows = []
    for index, (sequence, action_count) in enumerate(zip(expert_z, expert_lengths)):
        distances = _nearest_distance(sequence, reference)
        dca = float(np.exp(-float(distances.mean()) / max(sigma_nom, 1e-6)))
        eas = float(max(0.0, 1.0 - action_count / max(nominal_length, 1.0)))
        dce = float((2.0 * dca * eas) / (dca + eas + 1e-8))
        rows.append({
            "trajectory": index,
            "expert_actions": action_count,
            "mean_normalized_nearest_state_distance": float(distances.mean()),
            "dca": dca,
            "eas": eas,
            "dce": dce,
        })

    summary = {
        "format": "stackpyramid_timing_metrics_v1",
        "method": args.method,
        "split": args.split,
        "nominal_episodes": len(nominal),
        "expert_episodes": len(expert),
        "nominal_median_actions": nominal_length,
        "sigma_nom": sigma_nom,
        "metrics": {
            "dca_mean": float(np.mean([row["dca"] for row in rows])),
            "eas_mean": float(np.mean([row["eas"] for row in rows])),
            "dce_mean": float(np.mean([row["dce"] for row in rows])),
        },
        "rows": rows,
        "interpretation": "DCA uses normalized nearest nominal task-state distance; EAS compares suffix actions with nominal median; DCE is their harmonic mean. This is a diagnostic proxy until paired state-snapshot DTW is available.",
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "summary.md").write_text(
        "# StackPyramid timing metrics\n\n"
        f"- Method: `{args.method}`\n- Split: `{args.split}`\n"
        f"- DCA: `{summary['metrics']['dca_mean']:.4f}`\n"
        f"- EAS: `{summary['metrics']['eas_mean']:.4f}`\n"
        f"- DCE: `{summary['metrics']['dce_mean']:.4f}`\n\n"
        "These are diagnostic DCA/EAS/DCE proxies; they are not a substitute for post-training success rate.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
