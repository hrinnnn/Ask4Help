# ManiSkill pi0.5 AWBC

This pipeline trains the pi0.5 action expert with advantage-weighted behavior
cloning (AWBC). It combines expert demonstrations and rollouts from a shared
warm-start policy, uses Robo-Dopamine to estimate offline progress, and keeps a
uniform behavior-cloning run as the matched control.

## Data Flow

```text
expert demonstrations ----+
                           +--> Robo-Dopamine Phi sidecars --> 1:1 sampler
warm-start policy rollouts +                                  |
                                                              v
                                                 per-sample pi0.5 flow loss
                                                              |
                                                              v
                                                  ARM AWBC weighted mean
```

AWBC only weights the action-expert flow-matching loss. The RLT loss, when
enabled in another experiment, is not weighted by this implementation.

## Weight Definition

For each valid anchor pair, the exact ARM mode computes:

```text
delta_phi = phi_next - phi
gain = delta_phi * episode_length / mean_valid_episode_length
lower = mean_valid(gain) - 2 * std_valid(gain)
upper = mean_valid(gain) + 2 * std_valid(gain)
weight = clip((gain - lower) / (upper - lower + 1e-6), 0, 1)
```

Statistics are gathered across all distributed ranks. Invalid GRM outputs do
not become a synthetic `Phi=0`: they are excluded from statistics and sampling.
If there are too few valid samples or gain variance is effectively zero, valid
samples fall back to weight 1.

The modes are:

- `uniform`: same valid anchors and same 1:1 expert/policy sampler, all weights 1.
- `arm_paper_exact`: the formula above, including continuous negative progress.
- `robodopamine_robust`: exact mode plus optional negative-delta suppression,
  confidence scaling, a weight floor, and gain clipping.

The robust controls default to disabled so they cannot silently change the ARM
baseline.

## Persistent Layout

Large files belong on the OSS mount, not in Git:

```text
/mnt/data/ask4help/datasets/lerobot/local/awbc_peg/
  expert/
  policy_warmstart_25pct/
/mnt/data/ask4help/annotations/awbc_peg/
  goal_bank/
  expert_progress.jsonl
  policy_progress.jsonl
  combined_progress.jsonl
/mnt/data/ask4help/models/awbc_peg/
  warmstart_25pct/
/mnt/data/ask4help/results/awbc_peg/
  uniform/
  arm_exact/
  robodopamine_robust/
```

The combined manifest follows `ConcatDataset([expert, policy])` order. Every
dataset row has a record, while only rows whose current and next GRM anchors are
valid are eligible for AWBC sampling.

## 1. Choose The Shared Warm Start

Train at most 500 SFT steps and evaluate checkpoints on fixed seeds. Select the
checkpoint whose success rate is closest to 25 percent. Record the exact
checkpoint, evaluation seeds, and measured success rate. All three downstream
runs must start from this same checkpoint.

## 2. Collect Policy Episodes

The collection configuration disables RLT policy switching and retains both
successful and failed trajectories:

```bash
PI05_MODEL_PATH=/mnt/data/ask4help/models/awbc_peg/warmstart_25pct \
NORM_STATS_PATH=/mnt/data/ask4help/datasets/lerobot/local/rlt_peg_smoke/norm_stats.json \
OUTPUT_DIR=/mnt/data/ask4help/datasets/lerobot/local/awbc_peg/policy_collection \
bash scripts/awbc_peg/collect_policy_rollouts.sh
```

The default config uses 8 environments and 4 rollout epochs to target 32
episodes. The script merges rank shards and writes `successful_episodes.json`.

## 3. Annotate Progress

Run a Robo-Dopamine-compatible endpoint, then build the goal bank and both
sidecars. The goal image is the final frame of a successful expert episode.
Progress is queried every 5 environment steps and at the terminal frame using
incremental, forward, and backward modes with consistency-aware fusion.

```bash
EXPERT_DATASET=/mnt/data/ask4help/datasets/lerobot/local/awbc_peg/expert \
POLICY_DATASET=/mnt/data/ask4help/datasets/lerobot/local/awbc_peg/policy_collection/policy_rollouts \
POLICY_SUCCESS_EPISODES=/mnt/data/ask4help/datasets/lerobot/local/awbc_peg/policy_collection/successful_episodes.json \
GRM_ENDPOINT=http://127.0.0.1:8000/v1/chat/completions \
GRM_MODEL_NAME=/mnt/data/ask4help/models/Robo-Dopamine-GRM-2.0-4B-Preview \
OUTPUT_ROOT=/mnt/data/ask4help/annotations/awbc_peg \
bash scripts/awbc_peg/prepare_progress_manifests.sh
```

Annotation writes a resumable per-anchor cache. A parser failure neither updates
the previous progress state nor creates a false negative reward. Success sets
only the terminal anchor of that episode to `Phi=1`; the next episode resets to
`Phi=0`.

## 4. Run Matched Training

Uniform control:

```bash
EXPERT_DATASET=/mnt/data/ask4help/datasets/lerobot/local/awbc_peg/expert \
POLICY_DATASET=/mnt/data/ask4help/datasets/lerobot/local/awbc_peg/policy_collection/policy_rollouts \
PROGRESS_MANIFEST=/mnt/data/ask4help/annotations/awbc_peg/combined_progress.jsonl \
PI05_WARM_START=/mnt/data/ask4help/models/awbc_peg/warmstart_25pct \
NORM_STATS_PATH=/mnt/data/ask4help/datasets/lerobot/local/rlt_peg_smoke/norm_stats.json \
OUTPUT_DIR=/mnt/data/ask4help/results/awbc_peg/uniform \
MAX_STEPS=500 bash scripts/awbc_peg/run_uniform_bc.sh
```

ARM exact AWBC uses the same variables:

```bash
OUTPUT_DIR=/mnt/data/ask4help/results/awbc_peg/arm_exact \
AWBC_MODE=arm_paper_exact MAX_STEPS=500 \
bash scripts/awbc_peg/run_awbc_sft.sh
```

For fair comparison, hold the warm start, two datasets, manifest, sampling
ratio, batch size, learning rate, step count, and evaluation seeds fixed.

## Verification

Local/unit coverage includes:

- exact ARM hand calculations, episode scaling, invalid and zero-variance cases;
- distributed global statistics, weighted loss equivalence, and zero gradients;
- manifest alignment, duplicate/missing/out-of-range failures, and 1:1 sampling;
- success/reset isolation and invalid Robo-Dopamine parser behavior.

Server smoke order is:

1. OpenPI loader/model-forward smoke.
2. AWBC transform and batch metadata smoke.
3. ManiSkill Peg reset/step smoke.
4. Fake-GRM one-episode annotation smoke.
5. Two-step uniform BC.
6. Two-step exact AWBC.

Do not interpret fake-GRM or two-step runs as scientific results. Formal
comparison uses the real GRM annotations, 500 training steps, 50 fixed seeds for
iteration, then 3 training seeds and 100 evaluation episodes per method.
