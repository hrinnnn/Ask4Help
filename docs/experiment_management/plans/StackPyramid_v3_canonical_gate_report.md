# StackPyramid v3 Canonical Gate Report

Status: **BLOCKED; diagnostic only**

This report is the single cross-task gate record for the StackPyramid v3
timing-sweep and four-method comparison. No formal timing collection, mixed
stream collection, or training may use either task until the evidence below
is reproduced from one frozen source and one frozen gate manifest.

## Canonical Definition

- Benchmark: `stackpyramid_stage_localized_v3`
- Geometry implementation: `tools/stackpyramid_task.py`
- Canonical source revision: `50498f5`
- Timing manifest: `configs/stackpyramid_timing_v3_seed_manifest.json`
- Base checkpoint: `/root/stackpyramid_id_sft_10000_v1_from5090_local/ckpt-10000`
- Paired gate seeds: `83400` through `83499`
- Evaluation predicate: the strict task-success predicate emitted by
  `evaluate_stackpyramid_xvla.py`; prefix and target-stage predicates are
  split-specific and are recorded in the manifest.
- Runtime protocol: CPU simulator, 250 maximum environment steps, execution
  horizon 5, flow steps 5. For episode index `i`, the environment seed is
  `83400+i`; the action prediction uses the episode seed plus the executed
  action count.

The frozen OOD shifts are:

| Split | Changed factor | Shift | Prefix predicate | Target predicate |
|---|---|---|---|---|
| Stage1 OOD | red cube initial position | `[0.045, 0.045]` | `red_grasped` | `red_lifted` |
| Stage2 OOD | green cube / target position | `[0.060, 0.050]` | `red_lifted` | `red_placed` |
| Stage3 OOD | blue cube initial position | `[0.100, -0.120]` | `red_placed` | `blue_lifted` |

## Timing-Side Evidence

Root: `/root/stackpyramid_timing_sweep_v1_h20_v3_retry1`

The same base checkpoint was evaluated on the four paired splits:

| Split | Oracle | Base-policy strict success |
|---|---:|---:|
| ID | 100/100 | 94/100 |
| Stage1 OOD | 97/100 | 32/100 |
| Stage2 OOD | 97/100 | 0/100 |
| Stage3 OOD | 99/100 | 0/100 |

The prefix/locality audit is still incomplete. The completed portions report
`red_grasped=1/100` for Stage1 and `red_lifted=0/100` for Stage2. Therefore the
required prefix-completion gate is not currently satisfied, regardless of the
low final OOD success rates. Stage3 locality is still running and must be
audited before any conclusion.

## Four-Method-Side Evidence

The other task has reported the following preliminary values:

| Split | Reported base-policy strict success |
|---|---:|
| ID | 96/100 |
| Stage1 OOD | 40/100 |
| Stage2 OOD | 0/100 |
| Stage3 OOD | 0/100 |

Its exact output root, runtime command, norm path, source revision, manifest
path, and predicate evidence are not yet attached to this report. These values
must therefore remain **unreconciled diagnostics**, not a second official gate.

## Blocking Discrepancy

The timing and four-method results differ on the paired ID and Stage1 splits:
`94/100` versus `96/100`, and `32/100` versus `40/100`. The discrepancy cannot
be attributed to a collection/evaluation seed difference until both tasks show
that their gate runs used the same checkpoint, norm, source revision, task
implementation, manifest, simulator settings, action-sampling seed rule, and
strict-success predicate. The four-method task is prohibited from starting its
PCA pilot, mixed-stream collection, or training while any of these fields is
missing.

The canonical gate remains blocked until:

1. Stage3 locality is complete and all three prefix gates are explicitly
   audited.
2. Both tasks provide exact command and artifact evidence for every common
   gate field above.
3. The two tasks are rerun, if necessary, from one clean checkout at
   `50498f5` and the same manifest, or the discrepancy is explained by a
   documented deterministic setting difference and one side is declared
   canonical.
4. The final report records one authoritative ID/OOD base-policy table and
   labels all previous v1/v2 and unreconciled v3 results as diagnostic.

