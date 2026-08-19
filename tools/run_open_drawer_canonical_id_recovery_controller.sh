#!/usr/bin/env bash
set -u
ROOT=$OPEN_DRAWER_ROOT
if [ -z "$ROOT" ]; then ROOT=/data/zhaozhixuan/Ask4Help-open-drawer; fi
PY=$OPEN_DRAWER_PYTHON
if [ -z "$PY" ]; then PY=/data/zhaozhixuan/Ask4Help-airplane-5090/RLinf/.venv/bin/python; fi
RL=$ROOT/RLinf
RUN=/sdd/ask4help-open-drawer/results/open_drawer_canonical_id_recovery_v3
MODEL=$ROOT/results/model_cache/pi05_base_pytorch_v1
PLANNER=/data/zhaozhixuan/Ask4Help-open-drawer/oracle_compat_env_v1/bin/python
OLD=$RUN/../open_drawer_id_expansion_recovery_v4/datasets/id_merged256
NORM=$RUN/../open_drawer_id_expansion_recovery_v4/norm_stats/id_merged256
NEW=$RUN/datasets/id_extra256
MERGED=$RUN/datasets/id_merged512
LOG=$RUN/logs
PIDS=$RUN/pids
STATE=$RUN/pipeline_state.json
mkdir -p "$RUN" "$LOG" "$PIDS" "$RUN/provenance"
log(){ printf '%s %s\n' "$(date -Is)" "$*" >> "$LOG/controller.log"; }
state(){ printf '%s\n' "{\"task\":\"OpenDrawer canonical ID recovery\",\"stage\":\"$1\",\"status\":\"$2\",\"detail\":\"$3\",\"ood_started\":false}" > "$STATE"; }
fail(){ log "FAILED $*"; state failed failed "$*"; printf '%s\n' "$*" > "$RUN/PIPELINE_FAILED"; exit 1; }
alive(){ [ -s "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }
waitpid(){ while alive "$1"; do sleep 30; done; }
choose_gpus(){
  idle=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F, '$2+0 < 1000 && $3+0 == 0 {gsub(/ /,"",$1); print $1}' | head -2)
  G0=$(printf '%s\n' "$idle" | sed -n '1p'); G1=$(printf '%s\n' "$idle" | sed -n '2p')
  [ -n "$G0" ] && [ -n "$G1" ] || fail "fewer_than_two_idle_gpus"
}
collect(){
  name=$1; data=$2; out=$3; seed=$4; n=$5; max=$6; pid="$PIDS/$name.pid"
  [ -f "$out/summary.json" ] && return
  [ ! -e "$out" ] || fail "partial_output_$out"
  mkdir -p "$(dirname "$data")" "$(dirname "$out")"
  env CUDA_VISIBLE_DEVICES="$G0" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_LEROBOT_HOME="$RUN/lerobot_cache" PANDA_PLANNER_PYTHON="$PLANNER" PANDA_PLANNER_SITE_PACKAGES=/data/zhaozhixuan/Ask4Help-airplane-5090/RLinf/.venv/lib/python3.11/site-packages PYTHONPATH="$ROOT:$RL" taskset -c 0-19 "$PY" "$RL/toolkits/lerobot/collect_open_drawer_retrieve_place_lerobot.py" --repo-id "$data" --output-dir "$out" --video-dir "$out/videos" --num-episodes "$n" --seed "$seed" --max-attempts "$max" --image-size 384 --control-freq 10 --max-episode-steps 400 --save-videos > "$LOG/$name.log" 2>&1 < /dev/null &
  echo $! > "$pid"; waitpid "$pid"; [ -f "$out/summary.json" ] || fail "summary_missing_$name"
}
audit(){
  "$PY" "$ROOT/tools/audit_open_drawer_canonical_id_collection.py" --collection "$1" --merged-dataset "$2" --expected-new "$3" --expected-merged "$4" --output "$5" || fail audit_failed
}
train_smoke(){
  [ -f "$RUN/TRAINING_SMOKE_PASSED" ] && return
  out="$RUN/training/smoke_2step"; [ ! -e "$out" ] || fail training_smoke_partial; mkdir -p "$out"
  env CUDA_VISIBLE_DEVICES="$G0" ASK4HELP_RLINF_PLACEMENT="$G0-$G0" OPEN_DRAWER_ID_DATASET="$MERGED" OPEN_DRAWER_ID_NORM_STATS="$NORM" OPEN_DRAWER_PI05_MODEL_PATH="$MODEL" OPEN_DRAWER_RUN_ROOT="$out" OPEN_DRAWER_EXPERIMENT_NAME=smoke_2step PYTHONPATH="$ROOT:$RL" EMBODIED_PATH="$RL/examples/sft" taskset -c 0-19 "$PY" "$RL/examples/sft/train_vla_sft.py" --config-path "$RL/examples/sft/config" --config-name open_drawer_canonical_id_sft_20000_openpi_pi05 runner.max_steps=2 runner.save_interval=2 > "$LOG/training_smoke.log" 2>&1 < /dev/null &
  pid="$PIDS/training_smoke.pid"; echo $! > "$pid"; waitpid "$pid"
  [ -f "$out/canonical_id_sft_20000/checkpoints/global_step_2/actor/model_state_dict/full_weights.pt" ] || fail training_smoke_checkpoint_missing
  printf '%s\n' generic_base_2step_smoke_passed > "$RUN/TRAINING_SMOKE_PASSED"
}
train(){
  [ -f "$RUN/TRAINING_COMPLETE" ] && return
  out="$RUN/training/id_sft_20000"; mkdir -p "$out"
  if [ ! -f "$RUN/TRAINING_STARTED" ]; then
    env CUDA_VISIBLE_DEVICES="$G0,$G1" ASK4HELP_RLINF_PLACEMENT="$G0-$G1" OPEN_DRAWER_ID_DATASET="$MERGED" OPEN_DRAWER_ID_NORM_STATS="$NORM" OPEN_DRAWER_PI05_MODEL_PATH="$MODEL" OPEN_DRAWER_RUN_ROOT="$out" OPEN_DRAWER_EXPERIMENT_NAME=canonical_id_sft_20000 PYTHONPATH="$ROOT:$RL" EMBODIED_PATH="$RL/examples/sft" taskset -c 0-39 "$PY" "$RL/examples/sft/train_vla_sft.py" --config-path "$RL/examples/sft/config" --config-name open_drawer_canonical_id_sft_20000_openpi_pi05 > "$LOG/training_20000.log" 2>&1 < /dev/null &
    pid="$PIDS/training_20000.pid"; echo $! > "$pid"; printf '%s\n' generic_pi05_fresh_training_started > "$RUN/TRAINING_STARTED"
  fi
  waitpid "$PIDS/training_20000.pid"
  for step in 5000 10000 15000 20000; do [ -f "$out/canonical_id_sft_20000/checkpoints/global_step_$step/actor/model_state_dict/full_weights.pt" ] || fail missing_checkpoint_$step; done
  printf '%s\n' training_complete > "$RUN/TRAINING_COMPLETE"
}
evaluate(){
  kind=$1; step=$2; episodes=$3; seed=$4
  out="$RUN/eval/$kind""_step_$step"; ckpt="$RUN/training/id_sft_20000/canonical_id_sft_20000/checkpoints/global_step_$step"
  [ -f "$out/summary.json" ] && return
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="$G0" PYTHONPATH="$ROOT:$RL" "$PY" "$ROOT/tools/evaluate_open_drawer_id_pi05.py" --checkpoint "$ckpt" --pi05-base "$MODEL" --norm-stats "$NORM" --output-dir "$out" --episodes "$episodes" --seed "$seed" --split id --execute-horizon 5 --max-episode-steps 400 > "$LOG/$kind""_step_$step.log" 2>&1 || true
  count=$("$PY" -c "import json; print(json.load(open('$out/summary.json'))['episodes'])" 2>/dev/null || echo 0)
  [ "$count" = "$episodes" ] || fail incomplete_eval_$kind_$step
}
main(){
  state preflight running generic_base_resource_audit
  [ -e "$MODEL" ] && [ -e "$OLD" ] && [ -e "$NORM" ] || fail missing_input
  choose_gpus
  printf '%s\n' '{"model_family":"generic_openpi_pi05","instruction":"open the drawer, retrieve the blue object, and place it in the green tray","new_seed_range":"84000-84255","training_steps":20000,"ood_started":false}' > "$RUN/provenance/run_manifest.json"
  state collection_smoke running seeds_83900_83901
  collect collection_smoke "$RUN/datasets/id_smoke2" "$RUN/collection_smoke_v1" 83900 2 4
  audit "$RUN/collection_smoke_v1" "$OLD" 2 256 "$RUN/audit_collection_smoke.json"
  printf '%s\n' collection_smoke_passed > "$RUN/COLLECTION_SMOKE_PASSED"
  state collection_extra256 running seeds_84000_84255
  collect collection_extra256 "$NEW" "$RUN/collection_extra256" 84000 256 320
  audit "$RUN/collection_extra256" "$OLD" 256 256 "$RUN/audit_collection_extra256.json"
  printf '%s\n' collection_256_passed > "$RUN/COLLECTION_COMPLETE"
  state merge_id_dataset running merge_512
  [ -e "$MERGED" ] || "$PY" "$RL/toolkits/lerobot/merge_lerobot_datasets.py" --source-dir "$OLD" "$NEW" --output-dir "$MERGED" > "$LOG/merge_512.log" 2>&1 || fail merge_failed
  audit "$RUN/collection_extra256" "$MERGED" 256 512 "$RUN/audit_id_merged512.json"
  printf '%s\n' merged_512_audit_passed > "$RUN/DATA_AUDIT_PASSED"
  train_smoke
  state training_20000 running generic_pi05_fresh_optimizer_scheduler
  train
  for step in 5000 10000 15000 20000; do state probe_$step running independent_20_id; evaluate probe "$step" 20 85000; done
  state formal_id_gate running independent_100_id
  evaluate formal 20000 100 86000
  printf '%s\n' formal_100_id_complete_ood_locked > "$RUN/FORMAL_GATE_COMPLETE"
  state complete waiting_total_control_review id_only_complete_no_ood
}
main "$@"
