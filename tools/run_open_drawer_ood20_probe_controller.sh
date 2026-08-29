#!/usr/bin/env bash
set -u

ROOT=${OPEN_DRAWER_ROOT:-/data/zhaozhixuan/Ask4Help-open-drawer}
PY=${OPEN_DRAWER_PYTHON:-/data/zhaozhixuan/Ask4Help-airplane-5090/RLinf/.venv/bin/python}
RUN=${OPEN_DRAWER_TIMING_ROOT:?set OPEN_DRAWER_TIMING_ROOT}
MODEL=${OPEN_DRAWER_TIMING_CHECKPOINT:?set OPEN_DRAWER_TIMING_CHECKPOINT}
PI05_BASE=${OPEN_DRAWER_TIMING_PI05_BASE:?set OPEN_DRAWER_TIMING_PI05_BASE}
NORM=${OPEN_DRAWER_TIMING_NORM:?set OPEN_DRAWER_TIMING_NORM}
GPU=${OPEN_DRAWER_OOD20_GPU:-2}
CPU_SET=${OPEN_DRAWER_OOD20_CPU_SET:-40-59}
SEED_START=${OPEN_DRAWER_OOD20_SEED_START:-79000}
PROBE_ROOT="$RUN/ood20_probe"
STATE="$RUN/ood20_probe_pipeline_state.json"
LOG="$RUN/ood20_probe_controller.log"
ANCHORS=(0 50 80 120 160 220)
SEEDS=(9301 9302 9303)

mkdir -p "$PROBE_ROOT" "$RUN/logs"
exec > >(tee -a "$LOG") 2>&1

write_state() {
  printf '%s\n' "{\"format\":\"open_drawer_grasp_timing_ood20_probe_v1\",\"stage\":\"$1\",\"status\":\"$2\",\"detail\":\"${3:-}\",\"updated_at\":\"$(date -Is)\"}" > "$STATE"
}

audit_probe() {
  local out=$1
  "$PY" - "$out" <<'PY'
import json
import sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1])
summary_path = root / "summary.json"
if not summary_path.is_file():
    raise SystemExit(f"missing summary: {summary_path}")
payload = json.loads(summary_path.read_text(encoding="utf-8"))
if payload.get("split") != "grasp_ood" or payload.get("episodes") != 20:
    raise SystemExit("probe split/denominator mismatch")
rows = payload.get("rows", [])
if len(rows) != 20 or len(list((root / "videos").glob("*.mp4"))) != 20:
    raise SystemExit("probe rows/video denominator mismatch")
for row in rows:
    for key in ("actions", "states", "timeline", "reset_metadata", "video"):
        path = Path(str(row.get(key, "")))
        if not path.is_file():
            raise SystemExit(f"missing {key}: {path}")
    actions = np.load(Path(row["actions"]))
    states = np.load(Path(row["states"]))
    if actions.ndim != 2 or actions.shape[1] != 8:
        raise SystemExit(f"bad action shape: {actions.shape}")
    if states.shape != (len(actions) + 1, 9):
        raise SystemExit(f"bad state shape: {states.shape}")
    timeline = json.loads(Path(row["timeline"]).read_text(encoding="utf-8"))
    if len(timeline.get("timeline", [])) != len(actions) + 1:
        raise SystemExit("timeline/action alignment mismatch")
    json.loads(Path(row["reset_metadata"]).read_text(encoding="utf-8"))
PY
}

gpu_is_idle() {
  local used util uuid apps
  read -r used util < <(nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $1, $2}')
  uuid=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader | tr -d " ")
  apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader | grep "$uuid" || true)
  [[ "$used" -le 100 && "$util" -le 5 && -z "$apps" ]]
}

wait_for_gpu() {
  while ! gpu_is_idle; do
    write_state "$1" waiting "GPU$GPU not idle"
    sleep 300
  done
}

write_state preflight running "20-OOD probe per completed timing model on GPU$GPU"
index=0
for anchor in "${ANCHORS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    condition="anchor_${anchor}"
    checkpoint="$RUN/training/$condition/seed_$seed/run/checkpoints/global_step_2500"
    out="$PROBE_ROOT/$condition/seed_$seed/grasp_ood"
    marker="$PROBE_ROOT/$condition/seed_$seed/OOD20_COMPLETE"
    probe_seed=$((SEED_START + index * 100))
    index=$((index + 1))
    if [[ -f "$marker" ]]; then
      continue
    fi
    while [[ ! -f "$checkpoint/actor/model_state_dict/full_weights.pt" ]]; do
      write_state "waiting_${condition}_seed_${seed}" waiting "checkpoint_pending"
      sleep 600
    done
    if [[ -f "$out/summary.json" ]]; then
      if audit_probe "$out"; then
        printf '%s\n' '20 Grasp-OOD probe artifacts independently audited' > "$marker"
        continue
      fi
      write_state "ood20_${condition}_seed_${seed}" failed "existing_probe_summary_failed_audit"
      printf '%s\n' "probe audit failed: $out" > "$RUN/OOD20_PROBE_FAILED"
      exit 1
    fi
    if [[ -e "$out" && -n "$(find "$out" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
      write_state "ood20_${condition}_seed_${seed}" failed "partial_probe_output_exists"
      printf '%s\n' "partial probe output: $out" > "$RUN/OOD20_PROBE_FAILED"
      exit 1
    fi
    wait_for_gpu "waiting_gpu_${condition}_seed_${seed}"
    mkdir -p "$out"
    write_state "ood20_${condition}_seed_${seed}" running "gpu=$GPU cpu_set=$CPU_SET episodes=20"
    env CUDA_VISIBLE_DEVICES="$GPU" \
      ASK4HELP_RLINF_ROOT="$ROOT/RLinf" PYTHONPATH="$ROOT:$ROOT/RLinf" \
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
      OMP_NUM_THREADS=20 MKL_NUM_THREADS=20 PYTHONUNBUFFERED=1 \
      taskset -c "$CPU_SET" "$PY" -u "$ROOT/tools/evaluate_open_drawer_id_pi05.py" \
      --checkpoint "$checkpoint" --pi05-base "$PI05_BASE" --norm-stats "$NORM" \
      --output-dir "$out" --episodes 20 --seed "$probe_seed" --split grasp_ood \
      --execute-horizon 5 --max-episode-steps 400 \
      > "$PROBE_ROOT/$condition/seed_${seed}/probe.log" 2>&1 || true
    if ! audit_probe "$out"; then
      write_state "ood20_${condition}_seed_${seed}" failed "probe_artifact_audit_failed"
      printf '%s\n' "probe artifact audit failed: $out" > "$RUN/OOD20_PROBE_FAILED"
      exit 1
    fi
    printf '%s\n' '20 Grasp-OOD probe artifacts independently audited' > "$marker"
  done
done
write_state all_ood20 complete "all completed timing models probed"
printf '%s\n' '20-OOD probes complete for all timing models' > "$RUN/OOD20_PROBE_COMPLETE"
echo OPEN_DRAWER_GRASP_TIMING_OOD20_PROBE_COMPLETE
