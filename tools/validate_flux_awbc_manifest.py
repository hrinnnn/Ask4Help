#!/usr/bin/env python3
"""Validate a GRM sidecar and report Flux-style AWBC weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from rlinf.algorithms.awbc import compute_flux_awbc_weights
from rlinf.data.awbc import AWBCProgressManifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress-threshold", type=float, default=0.01)
    args = parser.parse_args()

    records = list(AWBCProgressManifest.load(args.manifest))
    delta = torch.tensor([record.delta_phi for record in records])
    lengths = torch.tensor([record.episode_length_chunks for record in records])
    valid = torch.tensor([record.valid for record in records])
    result = compute_flux_awbc_weights(
        delta,
        lengths,
        valid=valid,
        progress_threshold=args.progress_threshold,
    )
    positive = int((result.weights > 0).sum().item())
    summary = {
        "records": len(records),
        "valid_records": int(valid.sum().item()),
        "positive_weight_records": positive,
        "zero_weight_records": int((result.weights <= 0).sum().item()),
        "weight_min": float(result.weights.min().item()),
        "weight_max": float(result.weights.max().item()),
        "weight_mean": float(result.weights.mean().item()),
        "effective_sample_size": float(result.effective_sample_size.item()),
        "delta_min": float(delta[valid].min().item()) if valid.any() else None,
        "delta_max": float(delta[valid].max().item()) if valid.any() else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if positive == 0:
        raise RuntimeError("Flux AWBC manifest has no positive-weight samples")


if __name__ == "__main__":
    main()
