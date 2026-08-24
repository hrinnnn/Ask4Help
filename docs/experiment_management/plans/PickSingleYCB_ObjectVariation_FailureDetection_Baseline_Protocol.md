# PickSingleYCB Object Variation Failure-Detection Baseline Protocol

This document is the execution contract for the passive failure-detection
comparison on the controlled PickSingleYCB object-variation task. It is read
by the Owner and any downstream controller before adding or running a detector.

## Task And Checkpoint

- Task: `PickSingleYCB Object Variation OOD` (`YCB-ObjectVar` in paper prose).
- ID object: `005_tomato_soup_can`.
- Object-OOD object: `008_pudding_box`.
- Only `object_model_id` changes between paired ID/OOD resets.
- Formal ID checkpoint: user-selected `global_step_4500`; the 15/20 probe is
  retained as diagnostic evidence and must not be replaced by a better-looking
  checkpoint.
- Frozen norm: `/data/.../datasets/id_v1_retry1/norm_stats.json`, computed only
  from the 128 ID demonstrations.
- Policy: pi0.5/OpenPI, Flow-SDE, action expert only, 8D action, 10-step chunk.

## Paper-Mandatory Baselines

These methods must be evaluated on the same passive policy rollouts before any
downstream DAgger training:

| Group | Method | Required implementation |
|---|---|---|
| Proposed/internal | Internal-Feature PCA | ID feature bank, fixed PCA residual threshold, strongest valid internal representation reported with layer/family provenance |
| Input-based | FIDeL | Official ResNet-18/euclidean FIDeL path, no semantic VLM filter; calibrate on successful ID only |
| Input/state-based | CRSAIL | Observable-state `k=5` implementation; any vision adaptation is a separate clearly named diagnostic, not silently merged |
| Output/generative | Diff-DAgger | Diffusion uncertainty/loss gate; fixed ID calibration and patience; canonical and threshold-sensitivity runs are separate |
| Output/action | ACC | Action-consistency score using the configured action-chunk sampling protocol |
| Output/action | STAC-Single | Single-sample action consistency; no extra action samples unless the protocol explicitly labels the variant |

Reference values from the current paper's Airplane table are context only and
must never be copied into the YCB results. The paper reports Airplane/pi0.5
AUPRC values of FIDeL `89.77`, CRSAIL `88.51`, ACC `73.11`, STAC `78.54`, and
Internal-Feature PCA `96.35`; these are not YCB measurements.

## Engineering-Extension Baselines

The following already-existing monitors are reported in the same YCB table as
extensions, with implementation and feature provenance:

- Bridge PCA (primary paper method in this codebase).
- Bridge Deep kNN, `k=10` when the HNSW/Deep-kNN asset is used.
- Bridge LLMD.
- Action Expert final LLMD.
- Action Expert final PCA.
- VLM input pooled PCA.
- VLM 50% PCA and Action Expert 50% PCA, when the layer cache is present.

An extension may not replace a paper-mandatory baseline. A missing method is
reported as `NOT_IMPLEMENTED` with its reason, not as a zero score.

## Common Passive Rollout Protocol

- Checkpoint and norm are identical for every detector.
- ID seeds: `14000--14099`; OOD seeds: `15000--15099`.
- Exactly 100 complete episodes per split.
- Policy-only execution; detector scores never change actions.
- `execute_horizon=5`, `max_episode_steps=200`.
- The same reset/task seed manifests, action arrays, state/timeline records, and
  videos are reused or matched across methods. No detector gets a separate
  policy rollout distribution.
- Failure label is `not strict task success` using the frozen PickSingleYCB
  predicate. Report policy success counts before detector metrics.
- Independent calibration uses successful ID rollouts only; OOD and failed
  rollouts are never used to fit thresholds or detector statistics.

## Required Metrics

Report both threshold-free and fixed-threshold results:

- Primary threshold-free metric: trajectory-max AUPRC.
- Also: trajectory-max AUROC and AUCPDT.
- Fixed threshold: balanced accuracy, failure recall, precision, F1,
  success-conditioned false-alarm rate, and first-alert lead time in low-level
  steps (count, mean, median).
- Report ID and OOD separately, with the complete denominator and video/action
  counts.
- Threshold-free metrics must be computed from the same per-episode score
  timelines and failure labels. Threshold sweeps may be diagnostic, but may not
  be selected on the evaluation labels and called a formal result.

## Diff-DAgger Threshold Sensitivity

The canonical Diff-DAgger calibration remains frozen at `q=.95`,
`patience=2`, and its ID-derived threshold. Because the canonical collection
did not reach 100 accepted trajectories, a lower-threshold run may be used only
as a separately named diagnostic sensitivity experiment:

- Preserve the canonical calibration and its failure evidence.
- Record the exact override threshold and multiplicative factor in the run
  manifest and collection summary.
- Use a new output root, never overwrite `diffdagger_retry1`.
- Do not merge the low-threshold data into the canonical baseline table without
  labeling it `Diff-DAgger-low-threshold-diagnostic`.
- If the low-threshold collection reaches 100 accepted trajectories, it may be
  used for an explicitly diagnostic matched training comparison; it does not
  erase the canonical Diff collection failure.

## Completion Gate

Before downstream training, verify for every available baseline:

- calibration artifact and threshold provenance;
- 100 ID + 100 OOD summary rows;
- 200 videos and action/state/timeline evidence;
- finite score timelines and no NaN/Inf;
- a metrics row in the comparison table.

Only after this passive baseline matrix is complete may the controller proceed
to collection audit, matched expert-action budgeting, four-way training, and
held-out ID/OOD policy evaluation. If a paper-mandatory method cannot be
reproduced, write `NEEDS_USER_DECISION` or `NOT_IMPLEMENTED` with evidence;
never silently skip it.
