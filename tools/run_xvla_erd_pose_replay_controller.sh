#!/usr/bin/env bash
set -euo pipefail

# Diagnostic-only controller. It never changes the formal fixed-grid pipeline.
PYTHON_BIN="${PYTHON_BIN:-/data/zhaozhixuan/envs/xvla_official_5090/bin/python}"
WORKTREE="${WORKTREE:-/data/zhaozhixuan/xvla_fixedgrid_knee_work}"
RESULT_ROOT="${RESULT_ROOT:-/data/zhaozhixuan/Ask4Help-airplane-5090/results/xvla_fixedgrid_taskpolicy_knee_v1}"
DIAG_ROOT="${DIAG_ROOT:-${RESULT_ROOT}/erd_pose_replay_v1}"
RETRY_TAG="${RETRY_TAG:-full_retry1}"
PYTHONPATH_VALUE="${WORKTREE}:${WORKTREE}/RLinf"

STACK_SUMMARY="${RESULT_ROOT}/stage_c_gate_v1/stackcube/heldout_ood/summary.json"
AIRPLANE_SUMMARY="${RESULT_ROOT}/stage_c_gate_v1/airplane/heldout_ood/summary.json"
STACK_OUT="${DIAG_ROOT}/stackcube_ood_${RETRY_TAG}"
AIRPLANE_OUT="${DIAG_ROOT}/airplane_ood_${RETRY_TAG}"
LOG_ROOT="${DIAG_ROOT}/logs_${RETRY_TAG}"
STATE_PATH="${DIAG_ROOT}/pipeline_state_${RETRY_TAG}.json"

mkdir -p "${DIAG_ROOT}" "${LOG_ROOT}"

write_state() {
  local stage="$1"
  "${PYTHON_BIN}" - "$STATE_PATH" "$stage" "$STACK_OUT" "$AIRPLANE_OUT" <<'PY'
import json,sys,datetime
path,stage,stack_out,air_out=sys.argv[1:]
payload={
  "format":"xvla_erd_pose_replay_controller_v1",
  "stage":stage,
  "updated_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),
  "stackcube_output":stack_out,
  "airplane_output":air_out,
}
open(path,'w',encoding='utf-8').write(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
PY
}

audit_ok() {
  "${PYTHON_BIN}" - "$1/replay_audit.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
ok=(d.get('episodes_requested')==50 and d.get('episodes_replayed')==50
    and d.get('all_step_counts_match') is True
    and d.get('all_reset_metadata_match') is True
    and d.get('all_initial_rgb_match') is True)
raise SystemExit(0 if ok else 1)
PY
}

run_replay() {
  local task="$1" split="$2" summary="$3" output="$4" log="$5"
  if [[ -f "${output}/replay_audit.json" ]] && audit_ok "${output}"; then
    echo "REPLAY_SKIP task=${task} output=${output}"
    return
  fi
  PYTHONPATH="${PYTHONPATH_VALUE}" "${PYTHON_BIN}" "${WORKTREE}/tools/replay_xvla_pose_timeline.py" \
    --task "${task}" --split "${split}" --summary "${summary}" \
    --output "${output}" --limit 50 >"${log}" 2>&1
  audit_ok "${output}"
}

write_state "stackcube_replay_running"
run_replay stackcube ood "${STACK_SUMMARY}" "${STACK_OUT}" "${LOG_ROOT}/stackcube.log"
write_state "stackcube_replay_complete_airplane_replay_running"
run_replay airplane ood "${AIRPLANE_SUMMARY}" "${AIRPLANE_OUT}" "${LOG_ROOT}/airplane.log"
write_state "complete"
printf '%s\n' "ERD_POSE_REPLAY_COMPLETE" > "${DIAG_ROOT}/ERD_POSE_REPLAY_COMPLETE_${RETRY_TAG}"
