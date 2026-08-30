#!/usr/bin/env bash
set -u

# Formal Grasp-OOD suffix recollection using the reviewed current-state Oracle.
# This root is intentionally independent from the legacy-Oracle formal root.

ROOT=${OPEN_DRAWER_ROOT:-/data/zhaozhixuan/Ask4Help-open-drawer}
PY=${OPEN_DRAWER_PYTHON:-/data/zhaozhixuan/Ask4Help-airplane-5090/RLinf/.venv/bin/python}
PLANNER=${PANDA_PLANNER_PYTHON:-/data/zhaozhixuan/simplerenv_ms3/env/bin/python}
RUN=${OPEN_DRAWER_DIRECT_ORACLE_FORMAL_ROOT:-$ROOT/results/open_drawer_grasp_timing_sweep_v1_direct_oracle_formal_retry1}
FORMAL=$RUN/formal
CHECKPOINT=${OPEN_DRAWER_TIMING_CHECKPOINT:?set OPEN_DRAWER_TIMING_CHECKPOINT}
PI05_BASE=${OPEN_DRAWER_TIMING_PI05_BASE:?set OPEN_DRAWER_TIMING_PI05_BASE}
NORM=${OPEN_DRAWER_TIMING_NORM:?set OPEN_DRAWER_TIMING_NORM}
GPU=${OPEN_DRAWER_DIRECT_ORACLE_FORMAL_GPU:-7}
CPU_SET=${OPEN_DRAWER_DIRECT_ORACLE_FORMAL_CPU_SET:-140-159}
TARGET=${OPEN_DRAWER_DIRECT_ORACLE_FORMAL_TARGET:-30}
MAX_ATTEMPTS=${OPEN_DRAWER_DIRECT_ORACLE_FORMAL_MAX_ATTEMPTS:-80}
SEED_START=${OPEN_DRAWER_DIRECT_ORACLE_FORMAL_SEED_START:-78300}
ANCHORS=(0 50 80 120 160 220)
STATE=$RUN/direct_oracle_formal_collection_state.json
LOG=$RUN/direct_oracle_formal_collection.log

mkdir -p "$RUN" "$FORMAL" "$RUN/logs"
exec > >(tee -a "$LOG") 2>&1

write_state() {
  printf '%s\n' "{\"format\":\"open_drawer_direct_oracle_formal_collection_v1\",\"stage\":\"$1\",\"status\":\"$2\",\"detail\":\"${3:-}\",\"updated_at\":\"$(date -Is)\"}" > "$STATE"
}

fail() {
  write_state "$1" failed "${2:-}"
  printf '%s\n' "stage=$1 detail=${2:-}" > "$RUN/DIRECT_ORACLE_FORMAL_COLLECTION_FAILED"
  exit 1
}

gpu_is_idle() {
  local gpu=$1 used util uuid apps
  read -r used util < <(nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $1, $2}')
  uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader | tr -d ' ')
  apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader | grep "$uuid" || true)
  [[ "$used" -le 100 && "$util" -le 5 && -z "$apps" ]]
}

write_state preflight running "oracle_mode=direct_grasp target=$TARGET anchors=${ANCHORS[*]} gpu=$GPU"
gpu_is_idle "$GPU" || fail preflight "GPU $GPU is not genuinely idle"
[[ -s "$CHECKPOINT/actor/model_state_dict/full_weights.pt" ]] || fail preflight "missing checkpoint full_weights"
[[ -d "$CHECKPOINT/actor/dcp_checkpoint" ]] || fail preflight "missing checkpoint DCP"
[[ -d "$PI05_BASE" ]] || fail preflight "missing pi05 base"
[[ -d "$NORM" ]] || fail preflight "missing frozen norm"

export PYTHONPATH="$ROOT:$ROOT/RLinf"
export ASK4HELP_RLINF_ROOT="$ROOT/RLinf"
export PANDA_PLANNER_PYTHON="$PLANNER"
export PANDA_PLANNER_MODE=shortest_joint_path
export PANDA_OBJECT_GRASP_MODE=symmetric_shortest
export CUDA_VISIBLE_DEVICES="$GPU"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=20 MKL_NUM_THREADS=20 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1

for index in "${!ANCHORS[@]}"; do
  anchor=${ANCHORS[$index]}
  out="$FORMAL/anchor_${anchor}"
  if [[ -f "$out/COLLECTION_COMPLETE" ]]; then
    write_state "anchor_${anchor}" completed "existing collection marker"
    continue
  fi
  if [[ -e "$out" && -n "$(find "$out" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    fail "anchor_${anchor}" "refusing partial output: $out"
  fi
  mkdir -p "$out"
  write_state "anchor_${anchor}" running "takeover_step=$anchor gpu=$GPU"
  set +e
  taskset -c "$CPU_SET" "$PY" -u "$ROOT/tools/collect_open_drawer_fixed_timing.py" \
    --split grasp_ood \
    --checkpoint "$CHECKPOINT" --pi05-base "$PI05_BASE" --norm-stats "$NORM" \
    --output-root "$out" --takeover-step "$anchor" --start-seed "$((SEED_START + index * 100))" \
    --target "$TARGET" --max-attempts "$MAX_ATTEMPTS" --execute-horizon 5 \
    --max-episode-steps 400 --oracle-mode direct_grasp \
    > "$RUN/logs/anchor_${anchor}.log" 2>&1
  rc=$?
  set -e
  if [[ ! -f "$out/summary.json" ]]; then
    fail "anchor_${anchor}" "collector_rc=$rc; missing summary"
  fi
  accepted=$($PY -c 'import json,sys; print(int(json.load(open(sys.argv[1])).get("accepted",0)))' "$out/summary.json" 2>/dev/null || printf '0')
  [[ "$accepted" -ge "$TARGET" ]] || fail "anchor_${anchor}" "collector_rc=$rc; accepted=$accepted/$TARGET"
  [[ -f "$out/COLLECTION_COMPLETE" ]] || printf '%s\n' "direct Oracle formal anchor $anchor accepted=$accepted target=$TARGET" > "$out/COLLECTION_COMPLETE"
  write_state "anchor_${anchor}" completed "accepted=$accepted target=$TARGET"
done

write_state artifact_audit running "six anchors; target=$TARGET"
if ! "$PY" -u "$ROOT/tools/audit_open_drawer_grasp_timing.py" \
  --root "$FORMAL" --anchors "${ANCHORS[@]}" --target "$TARGET" \
  > "$RUN/formal_audit.log" 2>&1; then
  fail artifact_audit "audit_open_drawer_grasp_timing_failed"
fi

# The generic audit checks contract/evidence. This second pass confirms that
# accepted rows were generated by the reviewed direct-grasp continuation and
# that the old formal root was not silently reused.
if ! "$PY" - "$FORMAL" "$ROOT/results/open_drawer_grasp_timing_sweep_v1_formal/formal" <<'PY'
import json
import sys
from pathlib import Path

formal = Path(sys.argv[1])
old = Path(sys.argv[2]).resolve()
if formal.resolve() == old:
    raise SystemExit("new direct Oracle formal root resolves to legacy root")
for anchor in (0, 50, 80, 120, 160, 220):
    root = formal / f"anchor_{anchor}"
    summary = json.loads((root / "summary.json").read_text())
    if summary.get("split") != "grasp_ood" or summary.get("oracle_mode") != "direct_grasp":
        raise SystemExit(f"protocol mismatch in {root}: {summary.get('split')} {summary.get('oracle_mode')}")
    rows = [json.loads(line) for line in (root / "accepted_experts.jsonl").read_text().splitlines() if line]
    if len(rows) != 30:
        raise SystemExit(f"accepted denominator mismatch in {root}: {len(rows)}")
    for row in rows:
        attempt = root / "raw_attempts" / f"attempt_{int(row['attempt']):06d}_seed_{int(row['seed']):06d}" / "attempt.json"
        evidence = json.loads(attempt.read_text())
        expert = evidence.get("expert_result") or {}
        if expert.get("oracle_mode") != "direct_grasp_from_current_state":
            raise SystemExit(f"legacy Oracle row in {attempt}")
print("DIRECT_ORACLE_FORMAL_PROTOCOL_AUDIT_PASS")
PY
then
  fail artifact_audit "direct_oracle_protocol_audit_failed"
fi

printf '%s\n' 'six Grasp-OOD timing anchors collected with reviewed direct-grasp Oracle' > "$RUN/TIMING_COLLECTION_COMPLETE"
printf '%s\n' 'direct Oracle formal collection and independent artifact/protocol audit passed' > "$FORMAL/AUDIT_PASS"
write_state complete complete "six anchors; accepted=$((TARGET * ${#ANCHORS[@]})); oracle_mode=direct_grasp"
echo OPEN_DRAWER_DIRECT_ORACLE_FORMAL_COLLECTION_COMPLETE
