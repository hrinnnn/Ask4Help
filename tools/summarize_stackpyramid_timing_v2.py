#!/usr/bin/env python3
"""Compute paired, stage-aware timing diagnostics for StackPyramid suffixes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def groups(handle: h5py.File) -> list[str]:
    return sorted((name for name in handle if name.startswith("traj_")), key=lambda name: int(name.rsplit("_", 1)[1]))


def load_h5(path: Path) -> tuple[list[np.ndarray], list[int]]:
    states: list[np.ndarray] = []
    lengths: list[int] = []
    with h5py.File(path, "r") as handle:
        for name in groups(handle):
            state = np.asarray(handle[name]["obs/state"], dtype=np.float32)[:, :8]
            actions = np.asarray(handle[name]["actions"], dtype=np.float32)
            if state.shape[0] != actions.shape[0] + 1:
                raise ValueError(f"boundary mismatch in {path}:{name}")
            states.append(state)
            lengths.append(int(actions.shape[0]))
    return states, lengths


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dtw_distance(query: np.ndarray, reference: np.ndarray) -> float:
    cost = np.linalg.norm(query[:, None, :] - reference[None, :, :], axis=-1)
    table = np.full((len(query) + 1, len(reference) + 1), np.inf, dtype=np.float64)
    table[0, 0] = 0.0
    for i in range(1, len(query) + 1):
        for j in range(1, len(reference) + 1):
            table[i, j] = cost[i - 1, j - 1] + min(table[i - 1, j], table[i, j - 1], table[i - 1, j - 1])
    return float(table[-1, -1] / max(len(query), len(reference), 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nominal-h5", type=Path, required=True)
    parser.add_argument("--nominal-training-jsonl", type=Path, required=True)
    parser.add_argument("--expert-h5", type=Path, required=True)
    parser.add_argument("--expert-training-jsonl", type=Path)
    parser.add_argument("--expert-source-training-jsonl", type=Path)
    parser.add_argument("--budget-manifest", type=Path)
    parser.add_argument("--expert-episodes-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--split", required=True)
    args = parser.parse_args()

    nominal, nominal_lengths = load_h5(args.nominal_h5)
    expert, expert_lengths = load_h5(args.expert_h5)
    nominal_meta = load_jsonl(args.nominal_training_jsonl)
    if args.expert_training_jsonl and args.expert_training_jsonl.is_file():
        expert_meta = load_jsonl(args.expert_training_jsonl)
    elif args.expert_source_training_jsonl and args.budget_manifest:
        source_meta = load_jsonl(args.expert_source_training_jsonl)
        manifest = json.loads(args.budget_manifest.read_text(encoding="utf-8"))
        selected_groups = manifest["conditions"][args.method]["selected_groups"]
        expert_meta = [source_meta[int(name.rsplit("_", 1)[1])] for name in selected_groups]
    else:
        raise ValueError("provide selected expert metadata or source metadata plus budget manifest")
    raw_meta = {row["seed"]: row for row in load_jsonl(args.expert_episodes_jsonl)}
    nominal_by_seed = {row["seed"]: index for index, row in enumerate(nominal_meta)}
    nominal_seeds = sorted(nominal_by_seed)
    if len(expert) != len(expert_meta):
        raise ValueError("expert H5 and training metadata have different episode counts")

    nominal_matrix = np.concatenate(nominal, axis=0)
    mean = nominal_matrix.mean(axis=0)
    std = nominal_matrix.std(axis=0)
    std[std < 1e-5] = 1.0
    nominal_z = [(sequence - mean) / std for sequence in nominal]
    expert_z = [(sequence - mean) / std for sequence in expert]
    nominal_length = float(np.median(nominal_lengths))
    rows = []
    for index, (sequence, action_count, meta) in enumerate(zip(expert_z, expert_lengths, expert_meta)):
        seed = meta["seed"]
        if seed not in raw_meta:
            raise ValueError(f"missing raw metadata for seed {seed}")
        nominal_seed = seed if seed in nominal_by_seed else min(nominal_seeds, key=lambda value: abs(value - seed))
        nominal_sequence = nominal_z[nominal_by_seed[nominal_seed]]
        raw = raw_meta[seed]
        start = min(int(raw["expert_start_step"] or 0), len(nominal_sequence) - 1)
        nominal_suffix = nominal_sequence[start:]
        start_error = float(np.linalg.norm(sequence[0] - nominal_sequence[start]))
        path_error = dtw_distance(sequence, nominal_suffix)
        start_score = 1.0 / (1.0 + start_error)
        path_score = 1.0 / (1.0 + path_error)
        dca = float(np.sqrt(start_score * path_score))
        eas = float(max(0.0, 1.0 - action_count / max(nominal_length, 1.0)))
        dce = float((2.0 * dca * eas) / (dca + eas + 1e-8))
        rows.append({
            "trajectory": index,
            "seed": seed,
            "nominal_seed": nominal_seed,
            "expert_actions": action_count,
            "nominal_start_step": start,
            "start_error": start_error,
            "completion_dtw_error": path_error,
            "dca": dca,
            "eas": eas,
            "dce": dce,
        })

    summary = {
        "format": "stackpyramid_timing_metrics_v2",
        "method": args.method,
        "split": args.split,
        "nominal_episodes": len(nominal),
        "expert_episodes": len(expert),
        "nominal_median_actions": nominal_length,
        "metrics": {
            "dca_mean": float(np.mean([row["dca"] for row in rows])),
            "eas_mean": float(np.mean([row["eas"] for row in rows])),
            "dce_mean": float(np.mean([row["dce"] for row in rows])),
        },
        "interpretation": "DCA combines paired takeover-state alignment and stage-aware completion DTW; EAS measures saved expert actions; DCE is their harmonic mean.",
        "rows": rows,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output / "summary.md").write_text(
        "# StackPyramid timing metrics v2\n\n"
        f"- Method: `{args.method}`\n- Split: `{args.split}`\n"
        f"- DCA: `{summary['metrics']['dca_mean']:.4f}`\n"
        f"- EAS: `{summary['metrics']['eas_mean']:.4f}`\n"
        f"- DCE: `{summary['metrics']['dce_mean']:.4f}`\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
