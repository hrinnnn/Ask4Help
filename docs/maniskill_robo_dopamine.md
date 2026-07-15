# ManiSkill Robo-Dopamine Integration

This integration adds chunk-level Robo-Dopamine potential shaping to the RLinf
ManiSkill RLT Stage 2 pipeline. Environment installation is intentionally out of
scope; run these commands only after the official RLinf environment passes its
own smoke checks.

## Goal Bank

Generate the reference goal from the terminal observation of a successful
official ManiSkill motion-planning replay:

```bash
cd /root/Ask4Help/RLinf
.venv/bin/python toolkits/lerobot/collect_maniskill_peg_lerobot_joint.py \
  --repo-id /tmp/maniskill_peg_grm_reference \
  --num-episodes 1 \
  --grm-goal-bank-dir /data/ask4help/assets/grm_goal_bank/maniskill_peg
```

The command must create `task_000/goal_main.png`, `goal_wrist.png`, and
`meta.json`. The reward model fails fast if this bank is missing or empty.

## Fake Endpoint Smoke

Set the same Stage 1 checkpoint and normalization inputs used by the normal RLT
Stage 2 smoke, then run:

```bash
export STAGE1_ACTOR=/data/ask4help/models/maniskill_rlt_stage1/actor
export NORM_STATS_PATH=/data/ask4help/datasets/maniskill_peg/norm_stats.json
export REPO_ID=local/rlt_maniskill_joint
export GRM_GOAL_BANK_DIR=/data/ask4help/assets/grm_goal_bank/maniskill_peg
export OUTPUT_DIR=/data/ask4help/results/rlt_peg_grm_fake_smoke
bash /root/Ask4Help/scripts/rlt_peg_smoke/run_stage2_grm_fake_smoke.sh
```

This launches a local OpenAI-compatible fake endpoint and runs one shortened
RLT actor-critic epoch through the same reward-worker path as the real model.

## Real Endpoint Smoke

Start the Robo-Dopamine endpoint separately, then use the same smoke with the
real endpoint:

```bash
export GRM_ENDPOINT=http://127.0.0.1:8000/v1/chat/completions
export GRM_MODEL_NAME=tanhuajie2001/Robo-Dopamine-GRM-2.0-4B-Preview
bash /root/Ask4Help/scripts/rlt_peg_smoke/run_stage2_grm_smoke.sh
```

Inspect `${OUTPUT_DIR}/grm_metrics.jsonl`. A valid smoke has at least three
non-terminal GRM calls, valid parsed mode scores, finite shaping rewards on more
than one chunk, and no reward repeated across all ten low-level steps in a
chunk. ManiSkill sparse success reward remains enabled with weight `1.0`; GRM
shaping uses weight `0.1`.
