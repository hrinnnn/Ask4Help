#!/usr/bin/env python3
"""Render frozen LIBERO failure-detection summaries as Markdown and CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping


METHOD_LABELS = {
    "bridge_llmd": "Bridge LLMD",
    "bridge_deep_knn": "Bridge Deep kNN",
    "bridge_pca_residual": "Bridge PCA residual",
    "final_llmd": "Action Expert final LLMD",
    "acc": "ACC",
    "stac_single": "STAC-Single",
    "vla_fail_final_or_acc": "VLA-FAIL (final LLMD OR ACC)",
}
METRICS = (
    "roc_auc",
    "aucpr",
    "aucpdt",
    "balanced_accuracy",
    "weighted_accuracy",
    "tpr",
    "tnr",
    "fpr",
    "precision",
    "recall",
    "f1",
    "mean_normalized_detection_time",
    "twa",
)


def percentage(value: Any) -> str:
    return "-" if value is None else "%.1f%%" % (100.0 * float(value))


def table_rows(summary: Mapping[str, Any], group: str = "all") -> list[dict[str, Any]]:
    source = summary["all"] if group == "all" else summary["groups"][group]
    rows = []
    for method, values in source.items():
        if method == "runtime_ms":
            continue
        rows.append({"group": group, "method": method, "method_label": METHOD_LABELS.get(method, method), **dict(values)})
    return rows


def markdown_table(rows: list[Mapping[str, Any]], *, title: str) -> str:
    columns = ("Method", "ROC-AUC", "AUCPR", "AUCPDT", "Bal. Acc.", "TPR", "TNR", "FPR", "F1", "T-det", "TWA")
    metric_keys = ("roc_auc", "aucpr", "aucpdt", "balanced_accuracy", "tpr", "tnr", "fpr", "f1", "mean_normalized_detection_time", "twa")
    body = ["## " + title, "", "| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        values = [str(row["method_label"])] + [percentage(row.get(metric)) for metric in metric_keys]
        body.append("| " + " | ".join(values) + " |")
    return "\n".join(body) + "\n"


def write_tables(summary: Mapping[str, Any], output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite " + str(output_dir))
    output_dir.mkdir(parents=True, exist_ok=False)
    all_rows = table_rows(summary)
    markdown = ["# LIBERO-Plus Passive Failure Detection", "", markdown_table(all_rows, title="Overall")]
    for group in sorted(summary.get("groups", {})):
        if group.startswith("category="):
            markdown.append(markdown_table(table_rows(summary, group), title=group))
    runtime = summary.get("runtime_ms", {})
    markdown.extend(
        [
            "## Runtime",
            "",
            "- Policy sampling: %s ms/decision" % runtime.get("policy_mean_ms"),
            "- Deterministic feature probe: %s ms/decision" % runtime.get("feature_probe_mean_ms"),
            "- Total: %s ms/decision" % runtime.get("total_mean_ms"),
            "- Feature probe overhead: %s%%" % runtime.get("feature_probe_overhead_percent"),
            "",
        ]
    )
    (output_dir / "leaderboard.md").write_text("\n".join(markdown), encoding="utf-8")
    fieldnames = ("group", "method", "method_label", "episodes", *METRICS)
    with (output_dir / "leaderboard.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for group in ["all", *sorted(summary.get("groups", {}))]:
            writer.writerows({key: row.get(key) for key in fieldnames} for row in table_rows(summary, group))
    (output_dir / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    write_tables(json.loads(args.summary.read_text(encoding="utf-8")), args.output_dir)


if __name__ == "__main__":
    main()
