# StackCube VLA-FAIL Reproduction

This is the first training-free failure-detection baseline. It replaces the
previous paired-model VFD gate: one pi0.5 checkpoint and one GPU are enough.

## Exact Method Mapping

| VLA-FAIL component | Implementation |
| --- | --- |
| Feature | Final Action Expert token immediately before `action_out_proj` |
| Noise | One Gaussian action prior sampled once and reused for dataset statistics and rollout queries |
| Flow time | Semantic Gaussian-prior time: paper `t=0`, local RLinf pi0.5 `t=1` |
| LLMD | Per-token mean and population covariance, ridge `1e-6`, max token-wise squared Mahalanobis score |
| ACC | Absolute Panda end-effector positions, old suffix versus new prefix, overlap-prefix velocity normalization, EMA `alpha=.9` |
| FAIL | Logical OR of LLMD and ACC alarms |
| Calibration | Separate constant split-conformal bands at `delta=.05` from trajectory maxima of 20 successful held-out ID rollouts |

The local policy has `H=10` and executes `R=5` actions. The original paper's
hardware pi0.5 task has a different horizon; this therefore reproduces the
method on the local policy contract rather than its numerical task result.

## Single-GPU Protocol

All stages run sequentially on one GPU. This baseline includes no ensemble,
VFD, expert controller, online AWBC, Robo-Dopamine, or learned detector.

```bash
MODE=stats CHECKPOINT=... PI05_BASE=... NORM_STATS=... DATASET_ROOT=... \
  bash scripts/stackcube_vla_fail/run_reproduction.sh
MODE=calibrate CHECKPOINT=... PI05_BASE=... NORM_STATS=... DATASET_ROOT=... \
  bash scripts/stackcube_vla_fail/run_reproduction.sh
MODE=id_eval CHECKPOINT=... PI05_BASE=... NORM_STATS=... DATASET_ROOT=... \
  bash scripts/stackcube_vla_fail/run_reproduction.sh
MODE=ood_eval CHECKPOINT=... PI05_BASE=... NORM_STATS=... DATASET_ROOT=... \
  bash scripts/stackcube_vla_fail/run_reproduction.sh
```

`stats` uses every original observation in the 128-demo SFT dataset by default.
It writes the statistics, fixed prior, manifest, and SHA256. `calibrate` tries
held-out ID seeds until it has 20 successful trajectories with ACC overlap, or
stops at 200 attempts without emitting a threshold.

## Reusable Detector Assets

The detector has two persistent, reusable assets. They are deliberately
separate: fitting the normal feature distribution is not threshold calibration.

1. `llmd_statistics.pt` plus `llmd_statistics.pt.json`: final Action Expert
   Gaussian statistics, the one fixed flow prior, SFT dataset identity, model
   path, sample count, and SHA256.
2. `calibration_id/thresholds.json`: LLMD and ACC conformal thresholds plus
   the source statistics SHA256, checkpoint, successful calibration seeds,
   horizon, and finite-sample calibration parameters.

For a later experiment with the *same* checkpoint, data contract, fixed-prior
seed, and `H/R`, pass the existing locations explicitly instead of refitting:

```bash
STATS_PATH=/mnt/data/ask4help/results/stackcube_vla_fail/vla_fail_step7000/llmd_statistics.pt \
THRESHOLDS_PATH=/mnt/data/ask4help/results/stackcube_vla_fail/vla_fail_step7000/calibration_id/thresholds.json \
MODE=ood_eval CHECKPOINT=... PI05_BASE=... NORM_STATS=... DATASET_ROOT=... \
  bash scripts/stackcube_vla_fail/run_reproduction.sh
```

The evaluator rejects a threshold whose recorded statistics SHA does not match
the supplied `STATS_PATH`. Refit both assets whenever the policy checkpoint,
normalization contract, action horizon, execution horizon, or fixed-prior seed
changes.

Every rollout writes `episodes.json`, per-chunk LLMD/ACC/FAIL timelines,
outcome summaries, and videos under
`/mnt/data/ask4help/results/stackcube_vla_fail/reproduction_v1/`.
Render comparable ID/OOD score traces with:

```bash
python tools/plot_stackcube_vla_fail.py \
  --id /mnt/data/ask4help/results/stackcube_vla_fail/reproduction_v1/id_eval/episodes.json \
  --ood /mnt/data/ask4help/results/stackcube_vla_fail/reproduction_v1/ood_eval/episodes.json \
  --output /mnt/data/ask4help/results/stackcube_vla_fail/reproduction_v1/id_ood_scores.png
```

## Tests

`RLinf/tests/unit_tests/test_vla_fail.py` tests the paper-critical invariants:
token-wise Gaussian fitting and max aggregation, fixed prior reproducibility,
finite-sample conformal trajectory-max calibration, EEF-only overlap ACC with
EMA, and logical-OR fusion. The evaluator also fails closed when it cannot
resolve a Panda end-effector link or calibration has insufficient success.
