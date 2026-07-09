#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def write_config(args: argparse.Namespace) -> None:
    text = f"""defaults:
- env/libero_spatial@env.train
- env/libero_spatial@env.eval
- model/pi0_5@actor.model
- training_backend/fsdp@actor.fsdp_config
- weight_syncer/patch_syncer@weight_syncer
- override hydra/job_logging: stdout

hydra:
  run:
    dir: .
  output_subdir: null
  searchpath:
  - file://${{oc.env:EMBODIED_PATH}}/config/

cluster:
  num_nodes: 1
  component_placement:
    actor,env,rollout,reward: '{args.train_gpu_rank}'

runner:
  task_type: embodied
  logger:
    log_path: ../results
    project_name: rlinf
    experiment_name: {args.experiment_name}
    logger_backends:
    - tensorboard
  max_epochs: {args.max_epochs}
  max_steps: {args.max_steps}
  only_eval: false
  val_check_interval: -1
  save_interval: 40
  resume_dir: null
  ckpt_path: null

algorithm:
  normalize_advantages: true
  kl_penalty: kl
  group_size: 1
  reward_coef: 1.0
  reward_type: chunk_level
  logprob_type: chunk_level
  entropy_type: token_level
  update_epoch: 1
  adv_type: gae
  loss_type: actor_critic
  loss_agg_func: token-mean
  kl_beta: 0.0
  entropy_bonus: 0
  clip_ratio_high: 0.2
  clip_ratio_low: 0.2
  clip_ratio_c: 3.0
  value_clip: 0.2
  huber_delta: 10.0
  gamma: 0.99
  gae_lambda: 0.95
  filter_rewards: false
  rewards_lower_bound: 0.1
  rewards_upper_bound: 0.9

env:
  group_name: EnvGroup
  train:
    rollout_epoch: {args.rollout_epoch}
    total_num_envs: {args.num_envs}
    max_episode_steps: {args.max_episode_steps}
    max_steps_per_rollout_epoch: {args.max_episode_steps}
{args.task_filter_yaml}  eval:
    rollout_epoch: 1
    total_num_envs: 500
    auto_reset: true
    ignore_terminations: true
    max_episode_steps: 240
    max_steps_per_rollout_epoch: 240
    group_size: 1
    is_eval: true
    video_cfg:
      save_video: true
      video_base_dir: ${{runner.logger.log_path}}/video/eval

rollout:
  group_name: RolloutGroup
  generation_backend: huggingface
  recompute_logprobs: false
  unnorm_key: libero_10
  enable_offload: false
  pipeline_stage_num: 1
  model:
    model_path: {args.pi05_model_path}
    precision: ${{actor.model.precision}}

actor:
  group_name: ActorGroup
  training_backend: fsdp
  micro_batch_size: {args.micro_batch_size}
  global_batch_size: {args.global_batch_size}
  seed: 42
  enable_offload: false
  model:
    model_path: {args.pi05_model_path}
    model_type: openpi
    num_action_chunks: 5
    num_steps: 3
    add_value_head: true
    openpi:
      value_after_vlm: true
  optim:
    lr: 5.0e-06
    value_lr: 0.0001
    adam_beta1: 0.9
    adam_beta2: 0.95
    adam_eps: 1.0e-08
    weight_decay: 0.01
    clip_grad: 1.0
    critic_warmup_steps: 0
  fsdp_config:
    strategy: fsdp
    sharding_strategy: no_shard
    gradient_checkpointing: false
    mixed_precision:
      param_dtype: ${{actor.model.precision}}
      reduce_dtype: ${{actor.model.precision}}
      buffer_dtype: ${{actor.model.precision}}

reward:
  use_reward_model: true
  group_name: RewardGroup
  reward_mode: history_buffer
  history_reward_assign: false
  env_reward_weight: 1.0
  reward_weight: {args.reward_weight}
  model:
    model_type: dopamine_grm
    model_name: {args.grm_model_name}
    grm_endpoint: {args.grm_endpoint}
    goal_bank_dir: {args.goal_bank_dir}
    modes:
    - incremental
    - forward
    - backward
    fusion: mean_valid
    gamma: 0.99
    grm_interval_chunks: {args.grm_interval_chunks}
    invalid_reward: 0.0
    phi_clip:
    - 0.0
    - 1.0
    request_timeout: 240
    max_tokens: 32
    temperature: 0.0
    num_envs: {args.num_envs}
    history_buffers:
      grm_window:
        history_size: 1
        min_history_size: 1
        input_interval: 1
        history_keys:
        - main_images
        - wrist_images
        - reference_start_main_images
        - reference_start_wrist_images
        - task_descriptions
        - task_ids
        input_on_done: true

critic:
  use_critic_model: false
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--pi05-model-path", required=True)
    parser.add_argument("--goal-bank-dir", required=True)
    parser.add_argument("--grm-model-name", required=True)
    parser.add_argument("--grm-endpoint", required=True)
    parser.add_argument("--train-gpu-rank", default="1")
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--rollout-epoch", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--max-episode-steps", type=int, required=True)
    parser.add_argument("--micro-batch-size", type=int, required=True)
    parser.add_argument("--global-batch-size", type=int, required=True)
    parser.add_argument("--reward-weight", type=float, default=0.1)
    parser.add_argument("--grm-interval-chunks", type=int, default=1)
    parser.add_argument("--task-id-filter", default="")
    args = parser.parse_args()
    if args.task_id_filter:
        args.task_filter_yaml = f"    task_id_filter: [{args.task_id_filter}]\n"
    else:
        args.task_filter_yaml = ""
    write_config(args)


if __name__ == "__main__":
    main()

