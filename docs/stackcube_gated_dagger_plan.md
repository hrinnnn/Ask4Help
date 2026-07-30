# StackCube Robot-Gated DAgger Plan

## Current pilot

The current implementation compares only three ordinary-BC groups, all
initialized from the immutable `member0/global_step_7000` checkpoint:

1. `offline_oracle_bc`: 50 ID and 50 OOD oracle demonstrations from reset.
2. `bridge_knn_gated_bc`: Bridge Deep kNN latches the oracle at its first
   conformal alarm.
3. `late_success_dagger_bc`: the policy receives 50 low-level actions; if it
   has not succeeded, the oracle latches for the remaining episode.

Every rollout executes five actions per decision and every training label is a
complete 10-step expert segment.  Gated groups retain all raw attempts, but
only their expert suffixes become SFT data.  The offline group's 100 full
demos define the exact ID/OOD ten-step label budget; each gated group collects
until it matches that budget.  The original 128 ID demonstrations are fixed
replay.  Ordinary SFT uses `awbc.enabled=false` and a source-balanced batch:
two original-ID and two new-expert samples per micro-batch, yielding 32/32 in
the configured global batch of 64.

Each group first trains for 500 steps, saves `global_step_250` and
`global_step_500`, and supports an in-place resume to total step 1000.  After
the initial checkpoint, run ten disjoint OOD pure-policy rollouts.  Only if
the result is inadequate do we resume another 500 steps; no group is restarted
from base and no old checkpoint is overwritten.

Bridge Deep kNN calibration is rebuilt on a fresh successful-ID split that is
disjoint from collection and evaluation.  It uses the frozen ID feature bank,
trajectory-max score, and `q=.95` split conformal threshold.  The checkpoint,
normalization statistics, feature asset, threshold, seed manifests, label
budget, raw attempts, selected expert chunks, training logs, and videos are
all versioned in the experiment root.

## Deferred groups

The full planned comparison retains two deferred groups with the identical
label-accounting and SFT protocol:

- Bridge LLMD-gated BC.
- ACC-gated BC using the existing VLA-FAIL overlap/EMA/conformal implementation.

They are deliberately not started by the current pilot.  STAC remains out of
scope; no separate sampling semantics or action-distribution monitor is being
introduced for this experiment.

## Primary result

The primary metric is pure-policy success on the same 50 unseen OOD seeds for
all groups, with a Wilson 95% interval.  Collection cost is reported alongside
it: attempted episodes, expert takeover rate, first takeover step, total raw
expert actions, and admitted 10-step labels.  This keeps equal-label training
separate from the detector's actual label efficiency.
