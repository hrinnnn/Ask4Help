#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/Ask4Help-xvla-stackpyramid-v4-512}"
PY="${PY:-/root/.venvs/xvla-h20/bin/python}"
WORK="${WORK:-/root/ask4help_stage2_work/xvla_stackpyramid_oracle_repair_v3}"
PERSIST="${PERSIST:-/mnt/data/ask4help/results/xvla_stackpyramid_oracle_repair_v3}"
COLLECTION="$WORK/id_collection_additional_256_retry3"
AUDIT="$WORK/id_collection_additional_256_audit_full_planner"
OLD_H5="/root/ask4help_stage2_work/xvla_stackpyramid_oracle_repair_v1/id_collection_256_retry1/accepted_suffixes.h5"

mkdir -p "$WORK" "$PERSIST"
while [[ ! -f "$COLLECTION/COLLECTION_COMPLETE" ]]; do
  if ! pgrep -f "id_collection_additional_256_retry3" >/dev/null 2>&1; then
    printf 'collector exited before COLLECTION_COMPLETE\n' > "$WORK/COLLECTION_AUDIT_BLOCKED"
    cp "$WORK/COLLECTION_AUDIT_BLOCKED" "$PERSIST/COLLECTION_AUDIT_BLOCKED"
    exit 1
  fi
  sleep 60
done

if [[ ! -f "$AUDIT/ID_COLLECTION_AUDIT_PASS" ]]; then
  "$PY" "$ROOT/tools/audit_stackpyramid_id_collection.py" \
    --collection-root "$COLLECTION" --output "$AUDIT" \
    --task-spec "$ROOT/configs/stackpyramid_v4_task_spec.json" \
    --expected-episodes 256
  if [[ ! -e "$PERSIST/$(basename "$AUDIT")" ]]; then
    cp -a "$AUDIT" "$PERSIST/$(basename "$AUDIT")"
  fi
fi

"$PY" - "$OLD_H5" "$COLLECTION/accepted_suffixes.h5" "$WORK/action_diversity_report.json" "$PERSIST/action_diversity_report.json" "$WORK/training_exposure_report.json" "$PERSIST/training_exposure_report.json" <<'PY'
import json, sys
from pathlib import Path
import h5py
import numpy as np

old_h5, new_h5, local_div, durable_div, local_exp, durable_exp = map(Path, sys.argv[1:])

def inspect(path):
    arrays = []
    anchors = 0
    tail = 0
    with h5py.File(path, "r") as handle:
        for name in sorted(handle):
            if not name.startswith("traj_"):
                continue
            actions = np.asarray(handle[name]["actions"], dtype=np.float32)
            arrays.append(actions)
            anchors += int(actions.shape[0])
            tail += sum(min(10, actions.shape[0] - i) < 10 for i in range(actions.shape[0]))
    first = arrays[0] if arrays else None
    rows = []
    for index, value in enumerate(arrays[:5]):
        rows.append({
            "index": index,
            "shape": list(value.shape),
            "max_abs_diff_vs_first": float(np.max(np.abs(value - first))) if first is not None else None,
            "mean_abs_diff_vs_first": float(np.mean(np.abs(value - first))) if first is not None else None,
        })
    return {"episodes": len(arrays), "anchors": anchors, "tail_anchors": tail, "first_five": rows}

old = inspect(old_h5)
new = inspect(new_h5)
diversity = {
    "format": "stackpyramid_full_planner_action_diversity_v1",
    "status": "AUDITED",
    "historical_256": old,
    "additional_256": new,
    "additional_actions_not_template_only": any(
        row["max_abs_diff_vs_first"] not in (None, 0.0) for row in new["first_five"][1:]
    ),
    "merge_allowed": False,
    "training_allowed": False,
}
for path in (local_div, durable_div):
    path.write_text(json.dumps(diversity, indent=2) + "\n", encoding="utf-8")

anchors = old["anchors"] + new["anchors"]
batch = 8
steps = 20000
exposure = {
    "format": "stackpyramid_training_exposure_report_v2",
    "status": "TRAINING_EXPOSURE_REVIEW_REQUIRED",
    "training_started": False,
    "historical_anchors": old["anchors"],
    "additional_anchors": new["anchors"],
    "projected_merged_episodes": old["episodes"] + new["episodes"],
    "projected_merged_anchors": anchors,
    "projected_tail_anchors": old["tail_anchors"] + new["tail_anchors"],
    "batch_size": batch,
    "planned_optimizer_steps": steps,
    "planned_samples": batch * steps,
    "effective_dataset_passes": batch * steps / anchors if anchors else None,
    "freeze_steps": 1000,
    "warmup_steps": 2000,
    "freeze_fraction_of_steps": 1000 / steps,
    "warmup_fraction_of_steps": 2000 / steps,
    "domain_id_mapping_verified": False,
    "action_semantics_verified": False,
    "merge_allowed": False,
    "training_allowed": False,
    "ood_started": False,
}
for path in (local_exp, durable_exp):
    path.write_text(json.dumps(exposure, indent=2) + "\n", encoding="utf-8")
PY
printf 'training exposure report written; waiting for domain/action protocol decision\n' > "$WORK/TRAINING_EXPOSURE_REVIEW_REQUIRED"
cp "$WORK/TRAINING_EXPOSURE_REVIEW_REQUIRED" "$PERSIST/TRAINING_EXPOSURE_REVIEW_REQUIRED"
