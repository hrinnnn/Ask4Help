#!/usr/bin/env bash
set -u
ROOT=$OPEN_DRAWER_ROOT
if [ -z "$ROOT" ]; then ROOT=/data/zhaozhixuan/Ask4Help-open-drawer; fi
PY=$OPEN_DRAWER_PYTHON
if [ -z "$PY" ]; then PY=/data/zhaozhixuan/Ask4Help-airplane-5090/RLinf/.venv/bin/python; fi
RL=$ROOT/RLinf
RUN=/sdd/ask4help-open-drawer/results/open_drawer_pi05_resume50k_from_v9_v1
CKPT=/sdd/ask4help-open-drawer/results/open_drawer_generic_pi05_fresh_v9/training/smoke_2step/smoke_2step/checkpoints/global_step_2
MERGED=/sdd/ask4help-open-drawer/results/open_drawer_canonical_id_recovery_v3/datasets/id_merged512
NORM=/sdd/ask4help-open-drawer/results/open_drawer_id_expansion_recovery_v4/norm_stats/id_merged256
MODEL=$ROOT/results/model_cache/pi05_base_pytorch_v1
LOG=$RUN/logs
PIDS=$RUN/pids
STATE=$RUN/pipeline_state.json
mkdir -p "$RUN" "$LOG" "$PIDS" "$RUN/provenance" /sdd/ray_od_resume50k /sdd/tmp_od_resume50k

log(){ printf '%s %s\n' "$(date -Is)" "$*" >> "$LOG/controller.log"; }
state(){ printf '%s\n' "{\"task\":\"OpenDrawer pi0.5 resume from v9 smoke\",\"stage\":\"$1\",\"status\":\"$2\",\"detail\":\"$3\",\"ood_started\":false,\"global_step_start\":2,\"global_step_target\":50000}" > "$STATE"; }
fail(){ log "FAILED $*"; state failed failed "$*"; printf '%s\n' "$*" > "$RUN/PIPELINE_FAILED"; exit 1; }
alive(){ [ -s "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }
waitpid(){ while alive "$1"; do sleep 30; done; }
choose_gpus(){
  while :; do
    apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | sed 's/ //g')
    idle=$(nvidia-smi --query-gpu=index,uuid,memory.free,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F, -v apps="$apps" '$3+0 >= 28672 && $5+0 <= 5 {gsub(/ /,"",$1); gsub(/ /,"",$2); if (index(apps,$2)==0) print $1 ":" $2}')
    G0=$(printf '%s\n' "$idle" | cut -d: -f1 | sed -n '1p'); G1=$(printf '%s\n' "$idle" | cut -d: -f1 | sed -n '2p')
    if [ -n "$G0" ] && [ -n "$G1" ]; then return 0; fi
    state waiting_for_idle_gpus waiting need_two_gpus_free_ge_28GiB_and_no_compute_apps
    sleep 300
  done
}
record_mapping(){
  label=$1; pid=$2
  {
    printf 'label=%s\n' "$label"
    printf 'launcher_pid=%s\n' "$pid"
    nvidia-smi --query-gpu=index,uuid,memory.free,memory.used,utilization.gpu --format=csv,noheader
    printf '%s\n' compute_apps:
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader 2>/dev/null || true
    printf '%s\n' launcher_cuda_visible_devices:
    tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep '^CUDA_VISIBLE_DEVICES=' || true
  } > "$RUN/provenance/gpu_mapping_$label.txt"
}
verify_visible(){
  pid=$1; expected=$2
  actual=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' | head -1)
  [ "$actual" = "$expected" ] || fail "cuda_visible_mismatch expected=$expected actual=$actual"
}
env_args(){
  printf '%s\n' CUDA_VISIBLE_DEVICES="$1" ASK4HELP_RLINF_PLACEMENT="$2" OPEN_DRAWER_RESUME_DIR="$CKPT" OPEN_DRAWER_ID_DATASET="$MERGED" OPEN_DRAWER_ID_NORM_STATS="$NORM" OPEN_DRAWER_PI05_MODEL_PATH="$MODEL" OPEN_DRAWER_RUN_ROOT="$3" OPEN_DRAWER_EXPERIMENT_NAME="$4" PYTHONPATH="$ROOT:$RL" EMBODIED_PATH="$RL/examples/sft" RAY_TMPDIR=/sdd/ray_od_resume50k TMPDIR=/sdd/tmp_od_resume50k PYTHONUNBUFFERED=1
}
audit_checkpoint(){
  "$PY" - "$RUN/provenance/checkpoint_load_audit.json" <<'PY'
import json,pickle
from pathlib import Path
p=Path('/sdd/ask4help-open-drawer/results/open_drawer_generic_pi05_fresh_v9/training/smoke_2step/smoke_2step/checkpoints/global_step_2')
m=pickle.loads((p/'actor/dcp_checkpoint/.metadata').read_bytes())
keys=list(getattr(m,'state_dict_metadata',{}).keys())
d={'checkpoint':str(p),'global_step_start':2,'full_weights':(p/'actor/model_state_dict/full_weights.pt').is_file(),'dcp_metadata':True,'dcp_key_count':len(keys),'optimizer_key_count':sum('optimizers' in k for k in keys),'scheduler_key_count':sum('scheduler' in k.lower() or 'lr_scheduler' in k.lower() for k in keys),'resume_policy':'load DCP optimizer/scheduler if runtime accepts; no fallback to old checkpoint; record any loader rejection'}
Path('/sdd/ask4help-open-drawer/results/open_drawer_pi05_resume50k_from_v9_v1/provenance/checkpoint_load_audit.json').write_text(json.dumps(d,indent=2)+'\n')
PY
}
run_smoke(){
  [ -f "$RUN/RESUME_SMOKE_PASSED" ] && return
  out="$RUN/resume_smoke"; [ ! -e "$out" ] || fail resume_smoke_output_exists; mkdir -p "$out"
  choose_gpus; audit_checkpoint
  env $(env_args "$G0,$G1" "$G0-$G1" "$out" resume_smoke) taskset -c 0-39 "$PY" "$RL/examples/sft/train_vla_sft.py" --config-path "$RL/examples/sft/config" --config-name open_drawer_resume_50000_from_v9_smoke runner.max_steps=4 runner.save_interval=2 > "$LOG/resume_smoke.log" 2>&1 < /dev/null &
  echo $! > "$PIDS/resume_smoke.pid"; sleep 5; alive "$PIDS/resume_smoke.pid" && { record_mapping resume_smoke "$(cat "$PIDS/resume_smoke.pid")"; verify_visible "$(cat "$PIDS/resume_smoke.pid")" "$G0,$G1"; }; waitpid "$PIDS/resume_smoke.pid"
  actual="$out/resume_smoke/checkpoints/global_step_4/actor/model_state_dict/full_weights.pt"
  [ -f "$actual" ] || fail resume_smoke_checkpoint_missing
  printf '%s\n' "$actual" > "$RUN/provenance/resume_smoke_checkpoint_path.txt"
  printf '%s\n' 'Resume smoke loaded global_step=2 and wrote global_step=4.' > "$RUN/RESUME_SMOKE_PASSED"
}
run_formal(){
  [ -f "$RUN/TRAINING_STARTED" ] && return
  out="$RUN/training/resume_50k"; mkdir -p "$out"; choose_gpus
  env $(env_args "$G0,$G1" "$G0-$G1" "$out" resume_50k) taskset -c 0-39 "$PY" "$RL/examples/sft/train_vla_sft.py" --config-path "$RL/examples/sft/config" --config-name open_drawer_resume_50000_from_v9_smoke > "$LOG/resume_50k.log" 2>&1 < /dev/null &
  echo $! > "$PIDS/resume_50k.pid"; sleep 5; alive "$PIDS/resume_50k.pid" && { record_mapping resume_50k "$(cat "$PIDS/resume_50k.pid")"; verify_visible "$(cat "$PIDS/resume_50k.pid")" "$G0,$G1"; }; printf '%s\n' 'Resume from v9 global_step=2; optimizer/scheduler DCP load requested; target total step=50000.' > "$RUN/TRAINING_STARTED"
  waitpid "$PIDS/resume_50k.pid"
  for step in 5000 10000 15000 20000 25000 30000 35000 40000 45000 50000; do [ -f "$out/resume_50k/checkpoints/global_step_$step/actor/model_state_dict/full_weights.pt" ] || fail missing_checkpoint_$step; done
  printf '%s\n' 'Resume training completed at total global_step=50000.' > "$RUN/TRAINING_COMPLETE"
}
main(){
  state provenance_audit running weights_dcp_optimizer_scheduler
  [ -f "$CKPT/actor/model_state_dict/full_weights.pt" ] && [ -f "$CKPT/actor/dcp_checkpoint/.metadata" ] && [ -f "$MERGED/meta/info.json" ] && [ -e "$NORM" ] || fail missing_resume_input
  run_smoke
  state resume_smoke_passed waiting verified_global_step_4_checkpoint
  run_formal
  state complete waiting_total_control_review resume_50k_complete_ood_locked
}
main "$@"

