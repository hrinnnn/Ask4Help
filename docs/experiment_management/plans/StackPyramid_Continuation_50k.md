# StackPyramid Continuation 50k Plan

## Current Checkpoint Provenance

The existing `ckpt-10000` is a custom StackPyramid adaptation, not an
official X-VLA fine-tuning recipe. It was initialized from
`/mnt/data/ask4help/models/X-VLA-Pt_from5090_v4` and trained on the repaired
256-ID dataset with:

| Parameter | Existing custom run |
|---|---:|
| Optimizer steps | 10,000 |
| Learning rate | `2.5e-5` |
| Learning coefficient | `0.1` |
| Batch size | `8` |
| Freeze steps | `1,000` |
| Warmup steps | `2,000` |
| Precision | `bf16` |
| Dataset anchors | `72,171` |

This checkpoint and its failed formal ID gate remain diagnostic.

## Continuation Decision

For a new canonical continuation, use the official X-VLA learning rate
`1e-4` and coefficient `0.1`, rather than silently preserving `2.5e-5`.
The `2.5e-5` setting is retained as a custom diagnostic comparison only.
The continuation target is a total of 50,000 optimizer steps from the
selected approved initialization, with a new output root and immutable
provenance. No continuation is launched by this document.

Before launch, the following gates are mandatory:

1. The completed 512-ID collection must pass raw-denominator, video, HDF5,
   action/state boundary, event-order and temporal-mask audits.
2. The actual merged anchor count must be used to calculate effective dataset
   passes; the old 72,171-anchor count cannot be reused.
3. The domain assignment must be resolved. Current `domain_id=0` is not yet
   established as the correct StackPyramid embodiment/soft prompt.
4. The action contract must be resolved. Current HDF5/env actions are 8D
   `pd_joint_pos`, while the pretrained checkpoint is native 20D EE6D.
   Padding 8D actions to 20D is a custom adapter and requires explicit
   semantic validation.
5. A fresh 2-step/reload smoke must pass under the approved domain and action
   configuration.

## Exposure Calculation

For (N_{512}) merged real action anchors, a 50,000-step run with batch 8
would expose (400{,}000/N_{512}) effective dataset passes. Freeze and warmup
fractions must be reported separately from the total-step exposure. Checkpoint
selection and the formal 100-ID gate remain independent; no OOD stage is
unlocked until the formal ID gate reaches the pre-registered threshold.

## Output Boundary

The continuation must use a new root under
`/mnt/data/ask4help/results/xvla_stackpyramid_oracle_repair_v3/` and must not
overwrite the existing 10k checkpoint, template-only partial, hybrid partial,
or full-planner collection diagnostics. Until the five launch gates pass, the
only permitted active work is collection audit and protocol analysis.
