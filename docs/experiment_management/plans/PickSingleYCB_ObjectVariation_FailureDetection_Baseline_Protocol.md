# PickSingleYCB Object Variation Failure-Detection And DAgger Protocol

This document is the execution contract for the passive failure-detection and
downstream robot-gated DAgger comparison on the controlled PickSingleYCB
object-variation task. It is the baseline inventory for the Owner and any
downstream controller before adding or running a detector, collector, trainer,
or final evaluator.

The paper has two distinct baseline families. The passive family measures
failure-detection reliability on unchanged policy rollouts. The downstream
family measures the utility of the supervision selected by each data-acquisition
method under matched expert-action budget. A method missing from one family is
not silently substituted by a method from the other family.

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

## Complete Experiment Matrix

The following matrix is the minimum reproduction set for this task. Every row
needs its own manifest entry, calibration/provenance record, complete
denominator, and metrics row. The same row is never counted twice under a
different name.

| Stage | Main rows that must be run | Fixed inputs and comparison rule | Main output |
|---|---|---|---|
| Passive failure detection | Internal-Feature PCA, FIDeL, CRSAIL, Diff-DAgger, ACC, STAC-Single | One frozen ID policy, one ID/OOD seed pair, policy actions unchanged by scores; calibration uses successful ID only | ID/OOD score timelines, threshold-free metrics, fixed-threshold metrics, 200 videos |
| Internal representation ablation | Bridge PCA, VLM input pooled PCA, VLM 50% PCA, Action Expert 50% PCA, Action Expert final PCA, Bridge kNN (`k=10`), Bridge LLMD, Action Expert final LLMD | Same ID feature bank and norm provenance; each representation is a separate named row, not a replacement for the paper PCA row | Layer/family provenance and the same passive metrics |
| Data acquisition | Internal-Feature/Bridge PCA, Failure-Recovery DAgger, Diff-DAgger, Offline BC | Same immutable ID base; gated methods use the same alternating stream and target 100 accepted expert suffixes; Offline BC uses complete OOD oracle demonstrations | Raw/accepted attempts, takeover timelines, suffix actions, videos, dataset audit |
| Matched update training | The four data-acquisition rows above | Independent training from the same ID checkpoint; original ID and new expert source-balanced 1:1; identical optimizer, steps, norm, temporal mask, and action budget | Checkpoints every 500 steps, reload smoke, budget audit, training manifest |
| End-to-end policy evaluation | The four independently updated policies | Common held-out ID/OOD seed manifests, same horizon and success predicate, no detector assistance during evaluation | ID/OOD policy success, supporting success fields, 200 videos per method, final comparison table |

### What is a paper baseline and what is an extension?

- **Paper-mandatory passive baselines:** Internal-Feature PCA, FIDeL,
  CRSAIL, Diff-DAgger, ACC, and STAC-Single. These six rows are required for
  the passive table. If one cannot be reproduced, record `NOT_IMPLEMENTED`
  with the concrete code/runtime reason; do not write a zero score.
- **Paper-mandatory downstream baselines:** Offline BC,
  Failure-Recovery DAgger, and Diff-DAgger. The fourth downstream row is the
  proposed Internal-Feature/Bridge PCA method. These four rows must use
  matched expert-action budget before their final success rates are compared.
- **Internal representation extensions:** Bridge kNN, Bridge LLMD, Action
  Expert final LLMD, Action Expert final PCA, VLM input pooled PCA, VLM 50%
  PCA, and Action Expert 50% PCA. They explain layer locality and are useful
  ablations, but they do not remove any paper-mandatory row.
- **Backbone extension:** If the paper makes a cross-backbone claim, repeat the
  passive matrix (and, when claimed, the downstream four-way matrix) for the
  same task using pi0.5, OpenVLA, and X-VLA. A pi0.5-only run must be reported
  as a single-backbone result, not as a backbone-general result.

### Canonical naming map

| Paper name | Runtime name in this repository | Notes |
|---|---|---|
| Internal-Feature PCA | `bridge_pca` / `bridge_pca_residual` | Report the exact representation and layer/family; `Bridge PCA` is the primary implementation here |
| FIDeL | `fidel_official` | Official ResNet-18/euclidean path; no semantic VLM filter |
| CRSAIL | `crsail_observable_state_k5` | Observable-state kNN is canonical; `crsail_vision_k5` is a separate adaptation row |
| Diff-DAgger | `diffdagger` | Fixed ID calibration and patience for canonical passive results |
| ACC | `acc` | Action-consistency score under the declared chunk sampling protocol |
| STAC-Single | `stac_single` | Single-sample action consistency |
| Offline BC | `offline_oracle` | Complete expert OOD demonstrations, no policy-triggered takeover |
| Failure-Recovery DAgger | `failure_recovery` | Policy runs until the frozen failure/recovery rule, then oracle completes the suffix |

Do not copy paper Airplane numbers into the PickSingleYCB table. Do not use a
low-threshold Diff run, a different checkpoint, or a different success rule as
the canonical row.

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

## Downstream Collection, Training, And Evaluation Protocol

The passive table and the downstream table are separate experiments. A good
passive detector score does not by itself qualify a dataset for training, and a
policy success rate does not replace the passive detector metrics.

### Collection

1. Freeze the selected ID checkpoint, original ID norm, prompt, simulator
   runtime, success predicate, action horizon, and paired ID/OOD seed manifest.
2. Run Internal-Feature/Bridge PCA, Failure-Recovery, and Diff-DAgger on the
   same ID/OOD alternating raw stream. Keep every raw attempt and video.
3. Accept only successful, non-empty expert suffixes with complete action,
   state, reset-metadata, timeline, and video evidence. The target is 100
   accepted trajectories per gated method; the accepted ID/OOD composition is
   natural and must be reported rather than forced.
4. Collect exactly 100 complete OOD oracle demonstrations for Offline BC. It is
   a data baseline, not a gated detector, and must not be mixed with the other
   three datasets.
5. Audit full suffix boundaries and temporal tails before training. Choose the
   largest common matched low-level expert-action budget at or below the frozen
   budget; never equalize only by trajectory count.

### Training

Each of the four methods starts independently from the same immutable ID
checkpoint with a fresh optimizer. The original ID demonstrations and that
method's new expert source are mixed 1:1. Use the same batch/micro-batch,
learning-rate/scheduler, total update steps, checkpoint interval, frozen ID
norm, Flow-SDE/action-expert settings, and temporal `action_valid_mask` for all
four methods. Every real episode observation remains an anchor; padded tail
actions are repeated only for shape and are excluded from the loss.

Before formal training, each branch must pass a 2-step train/reload/finite
forward smoke. During formal training, archive every common checkpoint and
record the source/target checkpoint identity, dataset manifest, norm, mask,
and training configuration. Checkpoint selection is shared by rule and cannot
be chosen independently for each method by looking at the final OOD labels.

### Final policy evaluation

Evaluate all four updated policies on the same held-out 100-ID and 100-OOD
seed manifests using pure policy execution. Report strict task success first,
then supporting fields such as ever-grasped, accepted lift, placement, and
episode completion when available. Include complete videos/actions/states and
the exact horizon. These results are downstream learning utility, not failure
detection scores.

### Required evidence checklist

For every passive baseline and every downstream method, the result root must
contain:

- a manifest naming checkpoint, norm, prompt, task/split, seeds, runtime,
  horizon, success predicate, method variant, and calibration source;
- calibration and threshold provenance, or an explicit `NOT_IMPLEMENTED`/
  `NEEDS_USER_DECISION` diagnostic;
- complete episode rows, score/action/state timelines, and no NaN/Inf;
- the expected video count and decodable videos;
- an auditable metric row with denominator and split labels;
- no use of old StackCube, StackPyramid, OpenDrawer, Airplane, OOD, or another
  method's training data.

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

The low-threshold diagnostic must also never be used to claim that the
canonical patience/threshold gate passed. Its report must include the
canonical result beside the override value, raw-attempt denominator,
accepted-ID/OOD counts, and the reason the sensitivity run was requested.

## Completion Gate

Before downstream training, verify for every available baseline:

- calibration artifact and threshold provenance;
- 100 ID + 100 OOD summary rows;
- 200 videos and action/state/timeline evidence;
- finite score timelines and no NaN/Inf;
- a metrics row in the comparison table.

Only after this passive baseline matrix is complete may the controller proceed
to collection audit, matched expert-action budgeting, four-way training, and
held-out ID/OOD policy evaluation. The complete canonical task therefore
requires: six paper passive rows, all declared internal extensions, four
collection rows, four independent training rows, and four final policy rows.
If a paper-mandatory method cannot be reproduced, write `NEEDS_USER_DECISION`
or `NOT_IMPLEMENTED` with evidence; never silently skip it. A partial matrix
is a diagnostic, not a completed benchmark.
