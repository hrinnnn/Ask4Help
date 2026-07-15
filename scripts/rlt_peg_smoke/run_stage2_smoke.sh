#!/usr/bin/env bash
set -euo pipefail

RLINF_ROOT=${RLINF_ROOT:-/root/Ask4Help/RLinf}
PYTHON=${PYTHON:-"${RLINF_ROOT}/.venv/bin/python"}
STAGE1_ACTOR=${STAGE1_ACTOR:?Set STAGE1_ACTOR to the saved Stage 1 actor directory}
NORM_STATS_PATH=${NORM_STATS_PATH:?Set NORM_STATS_PATH to the dataset norm_stats.json}
REPO_ID=${REPO_ID:?Set REPO_ID to the local LeRobot dataset repo id}
OUTPUT_DIR=${OUTPUT_DIR:?Set OUTPUT_DIR to the Stage 2 output directory}

cd "${RLINF_ROOT}"
export EMBODIED_PATH="${RLINF_ROOT}/examples/embodiment"
export PYTHONPATH="${RLINF_ROOT}:${PYTHONPATH:-}"

"${PYTHON}" examples/embodiment/train_embodied_agent.py \
  --config-path "${RLINF_ROOT}/examples/embodiment/config" \
  --config-name maniskill_rlt_stage2_ac_mlp \
  runner.logger.log_path="${OUTPUT_DIR}" \
  runner.max_epochs=1 \
  runner.val_check_interval=-1 \
  runner.save_interval=1 \
  env.train.total_num_envs=2 \
  env.eval.total_num_envs=2 \
  env.train.max_episode_steps=20 \
  env.train.max_steps_per_rollout_epoch=20 \
  env.eval.max_episode_steps=20 \
  env.eval.max_steps_per_rollout_epoch=20 \
  env.train.rlt_policy_switch.task_mode=critical_phase \
  env.train.rlt_policy_switch.expert_takeover.enable=False \
  algorithm.update_epoch=1 \
  algorithm.critic_actor_ratio=1 \
  algorithm.rlt_schedule.max_updates_per_train_step=4 \
  algorithm.rlt_schedule.warmup_min_size=16 \
  algorithm.rlt_schedule.warmup_post_collect_updates=4 \
  algorithm.rlt_schedule.train_every_transitions=1 \
  algorithm.replay_buffer.min_buffer_size=4 \
  actor.micro_batch_size=4 \
  actor.global_batch_size=4 \
  rollout.rlt_feature_model.model_path="${STAGE1_ACTOR}" \
  rollout.rlt_feature_model.openpi_data.repo_id="${REPO_ID}" \
  "+rollout.rlt_feature_model.openpi_data.norm_stats_path=${NORM_STATS_PATH}"
  "rollout.expert_model.model_path=${STAGE1_ACTOR}"
