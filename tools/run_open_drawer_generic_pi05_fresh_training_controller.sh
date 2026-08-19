#!/usr/bin/env bash
set -u
ROOT=$OPEN_DRAWER_ROOT
if [ -z "$ROOT" ]; then ROOT=/data/zhaozhixuan/Ask4Help-open-drawer; fi
PY=$OPEN_DRAWER_PYTHON
if [ -z "$PY" ]; then PY=/data/zhaozhixuan/Ask4Help-airplane-5090/RLinf/.venv/bin/python; fi
RL=$ROOT/RLinf
RUN=/sdd/ask4help-open-drawer/results/open_drawer_generic_pi05_fresh_v8
MODEL=$ROOT/results/model_cache/pi05_base_pytorch_v1
MERGED=/sdd/ask4help-open-drawer/results/open_drawer_canonical_id_recovery_v3/datasets/id_merged512
NORM=/sdd/ask4help-open-drawer/results/open_drawer_id_expansion_recovery_v4/norm_stats/id_merged256
LOG=$RUN/logs
PIDS=$RUN/pids
STATE=$RUN/pipeline_state.json
TMP=$RUN/tmp
RAY=$RUN/ray_tmp
mkdir -p "$RUN" "$LOG" "$PIDS" "$TMP" "$RAY" "$RUN/provenance"

log(){ printf '%s %s\n' "$(date -Is)" "$*" >> "$LOG/controller.log"; }
state(){ printf '%s\n' "{\"task\":\"OpenDrawer generic pi0.5 fresh ID training\",\"stage\":\"$1\",\"status\":\"$2\",\"detail\":\"$3\",\"ood_started\":false,\"model_family\":\"generic_openpi_pi05\",\"canonical_prompt\":\"open the drawer, retrieve the blue object, and place it in the green tray\"}" > "$STATE"; }
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
record_gpu_mapping(){
  label=$1; pid=$2
  {
    printf 'label=%s\n' "$label"
    printf 'launcher_pid=%s\n' "$pid"
    nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader
    printf '%s\n' 'compute_apps:'
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader 2>/dev/null || true
    printf '%s\n' 'launcher_cuda_visible_devices:'
    tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep '^CUDA_VISIBLE_DEVICES=' || true
  } > "$RUN/provenance/gpu_mapping_$label.txt"
}
verify_visible_mapping(){
  pid=$1; expected=$2
  actual=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' | head -1)
  [ "$actual" = "$expected" ] || fail "cuda_visible_mapping_mismatch expected=$expected actual=$actual pid=$pid"
}
common_env(){
  printf '%s\n' CUDA_VISIBLE_DEVICES="$1" ASK4HELP_RLINF_PLACEMENT="$2" OPEN_DRAWER_ID_DATASET="$MERGED" OPEN_DRAWER_ID_NORM_STATS="$NORM" OPEN_DRAWER_PI05_MODEL_PATH="$MODEL" PYTHONPATH="$ROOT:$RL" EMBODIED_PATH="$RL/examples/sft" RAY_TMPDIR=/sdd/ray_od_v8 TMPDIR=/sdd/tmp_od_v8 PYTHONUNBUFFERED=1
}
smoke(){
  [ -f "$RUN/TRAINING_SMOKE_PASSED" ] && return
  out="$RUN/training/smoke_2step"; [ ! -e "$out" ] || fail smoke_output_exists; mkdir -p "$out"
  env $(common_env "$G0" "0-0") OPEN_DRAWER_RUN_ROOT="$out" OPEN_DRAWER_EXPERIMENT_NAME=smoke_2step taskset -c 40-59 "$PY" "$RL/examples/sft/train_vla_sft.py" --config-path "$RL/examples/sft/config" --config-name open_drawer_canonical_id_sft_20000_openpi_pi05 runner.max_steps=2 runner.save_interval=2 > "$LOG/training_smoke.log" 2>&1 < /dev/null &
  echo $! > "$PIDS/training_smoke.pid"; sleep 5; if alive "$PIDS/training_smoke.pid"; then record_gpu_mapping smoke "$(cat "$PIDS/training_smoke.pid")"; verify_visible_mapping "$(cat "$PIDS/training_smoke.pid")" "$G0"; fi; waitpid "$PIDS/training_smoke.pid"
  actual="$out/smoke_2step/checkpoints/global_step_2/actor/model_state_dict/full_weights.pt"
  [ -f "$actual" ] || fail verified_smoke_checkpoint_missing
  printf '%s\n' "$actual" > "$RUN/provenance/verified_smoke_checkpoint_path.txt"
  printf '%s\n' '2-step fresh generic pi0.5 canonical-prompt training smoke passed with verified checkpoint path.' > "$RUN/TRAINING_SMOKE_PASSED"
}
train(){
  [ -f "$RUN/TRAINING_COMPLETE" ] && return
  out="$RUN/training/id_sft_20000"; mkdir -p "$out"
  if [ ! -f "$RUN/TRAINING_STARTED" ]; then
    env $(common_env "$G0,$G1" "0-1") OPEN_DRAWER_RUN_ROOT="$out" OPEN_DRAWER_EXPERIMENT_NAME=canonical_id_sft_20000 taskset -c 0-39 "$PY" "$RL/examples/sft/train_vla_sft.py" --config-path "$RL/examples/sft/config" --config-name open_drawer_canonical_id_sft_20000_openpi_pi05 > "$LOG/training_20000.log" 2>&1 < /dev/null &
    echo $! > "$PIDS/training_20000.pid"; sleep 5; if alive "$PIDS/training_20000.pid"; then record_gpu_mapping train "$(cat "$PIDS/training_20000.pid")"; verify_visible_mapping "$(cat "$PIDS/training_20000.pid")" "$G0,$G1"; fi; printf '%s\n' generic_pi05_fresh_training_started_no_resume > "$RUN/TRAINING_STARTED"
  fi
  waitpid "$PIDS/training_20000.pid"
  for step in 5000 10000 15000 20000; do [ -f "$out/canonical_id_sft_20000/checkpoints/global_step_$step/actor/model_state_dict/full_weights.pt" ] || fail missing_checkpoint_$step; done
  printf '%s\n' generic_pi05_training_complete > "$RUN/TRAINING_COMPLETE"
}
eval_step(){
  kind=$1; step=$2; episodes=$3; seed=$4
  out="$RUN/eval/$kind""_step_$step"; ckpt="$RUN/training/id_sft_20000/canonical_id_sft_20000/checkpoints/global_step_$step"
  [ -f "$out/summary.json" ] && return
  mkdir -p "$out"
  env $(common_env "$G0" "0-0") "$PY" "$ROOT/tools/evaluate_open_drawer_id_pi05.py" --checkpoint "$ckpt" --pi05-base "$MODEL" --norm-stats "$NORM" --output-dir "$out" --episodes "$episodes" --seed "$seed" --split id --execute-horizon 5 --max-episode-steps 400 > "$LOG/$kind""_step_$step.log" 2>&1 || true
  "$PY" - "$out" "$episodes" <<'PY'
import json,sys
from pathlib import Path
out=Path(sys.argv[1]); expected=int(sys.argv[2]); s=json.loads((out/"summary.json").read_text()); videos=len(list((out/"videos").glob("*.mp4"))); episodes=len(list((out/"episodes").glob("episode_*")))
if s.get("episodes") != expected or videos != expected or episodes != expected: raise SystemExit(f"incomplete evidence {s.get('episodes')} {videos} {episodes}")
PY
}
select_formal(){
  [ -f "$RUN/FORMAL_GATE_COMPLETE" ] && return
  selected=$("$PY" - "$RUN/eval" <<'PY'
import json,sys
from pathlib import Path
best=None
for p in Path(sys.argv[1]).glob("probe_step_*/summary.json"):
 s=json.loads(p.read_text()); step=int(p.parent.name.rsplit("_",1)[1]); key=(int(s.get("successes",0)),-step)
 if best is None or key>best[0]: best=(key,step)
if best is None: raise SystemExit(1)
print(best[1])
PY
  ) || fail no_probe_selection
  eval_step formal "$selected" 100 86000
  "$PY" - "$RUN/eval/formal_step_$selected/summary.json" "$RUN/eval/formal_step_$selected" "$selected" "$RUN" <<'PY'
import json,sys
from pathlib import Path
s=json.loads(Path(sys.argv[1]).read_text()); out=Path(sys.argv[2]); step=int(sys.argv[3]); run=Path(sys.argv[4])
p={"selected_step":step,"episodes":s.get("episodes"),"successes":s.get("successes"),"videos":len(list((out/"videos").glob("*.mp4"))),"action_state_timeline_episodes":len(list((out/"episodes").glob("episode_*")))}
p["pass"]=p["episodes"]==100 and p["videos"]==100 and p["action_state_timeline_episodes"]==100 and p["successes"]>=80
(run/"FORMAL_ID_GATE_RESULT.json").write_text(json.dumps(p,indent=2)+"\n")
(run/"ID_BASE_VALIDATED_PENDING_TOTAL_CONTROL_REVIEW" if p["pass"] else run/"ID_BASE_PROTOCOL_FAILED").write_text(json.dumps(p,indent=2)+"\n")
(run/"FORMAL_GATE_COMPLETE").write_text("100-ID formal gate complete; OOD/PCA/DAgger remain locked.\n")
PY
}
main(){
  state preflight running generic_base_resource_and_disk_check
  [ -e "$MODEL" ] && [ -e "$MERGED/meta/info.json" ] && [ -e "$NORM" ] || fail missing_input
  choose_gpus
  printf '%s\n' '{"model_family":"generic_openpi_pi05","base":"pi05_base_pytorch_v1","source_dataset":"canonical_v3_merged512","fresh_optimizer":true,"prompt":"open the drawer, retrieve the blue object, and place it in the green tray","steps":20000,"checkpoints":[5000,10000,15000,20000],"ood_started":false}' > "$RUN/provenance/run_manifest.json"
  state training_smoke running verified_checkpoint_writing
  smoke
  state training_20000 running fresh_generic_pi05_two_gpu
  train
  for step in 5000 10000 15000 20000; do state probe_$step running independent_20_id; eval_step probe "$step" 20 85000; done
  state formal_id_gate running independent_100_id
  select_formal
  state complete waiting_total_control_review id_only_complete_ood_locked
}
main "$@"
