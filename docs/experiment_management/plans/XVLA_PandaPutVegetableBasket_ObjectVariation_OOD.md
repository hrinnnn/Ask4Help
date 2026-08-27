# X-VLA Panda Put Vegetable in Basket: Object-Variation OOD

## 1. Scope and scientific boundary

This is a new Panda experiment. It is not a continuation of the existing WidowX
`xvla_put_vegetable_basket_object_ood_v1` pipeline. The main comparison keeps
the Panda embodiment fixed and changes only the operated object between ID and
OOD. The old WidowX run remains diagnostic and cannot supply a checkpoint, norm,
dataset, detector asset, or result row.

The task uses the ManiSkill `PutEggplantInBasketScene-v1` BridgeData scene
template. The controlled environments are registered under new Panda-specific
IDs. The basket, sink, instruction, reset geometry and success predicate are
shared by both splits.

## 2. Frozen task contract

- Robot: `PandaBridgeDatasetFlatTable` with its Panda EE delta-pose controller.
- ID object: `eggplant`.
- OOD object: `eggplant` with fixed model scale `1.25` (object-size variation).
- Instruction: `put the vegetable into the yellow basket`.
- Scene: sink and yellow basket target from the BridgeData real-to-sim assets.
- Main camera: one fixed Panda experiment camera contract, frozen after smoke.
  The camera count and preprocessing must be identical for ID and OOD.
- Reset: paired seed generation; only the object asset identifier changes.
- Horizon: 120 environment steps, with all actual control and termination rules
  recorded in the manifest.
- Oracle profile: Panda planner, 35 cm lift with high-target two-stage transport,
  up to 30 one-step release waits;
  the profile is fixed before the formal gate.
- Success: after release, the object is inside the basket target region, above
  the target plane, static, and not grasped.
- X-VLA output: one active 10D EE6D arm block padded to the 20D model width;
  the Panda adapter maps this to the Panda controller. Directly putting Panda
  joint deltas into the X-VLA padding slots is invalid.
- Action chunk: 30 model actions, matching the foundation checkpoint contract.
- Training target: every real observation is an anchor. Tail actions repeat the
  final real action only for tensor shape and are excluded by the temporal mask.

The exact free X-VLA domain row, source commit, camera pose, and persistent
asset paths are not guessed here. They are frozen in the manifest only after
the preflight audits the foundation configuration and task runtime.

## 3. Stage-localized object OOD check

The object change is intended to affect grasp geometry and subsequent transport,
not to change the task meaning. Before formal collection, report:

1. Paired ID/OOD reset metadata and non-object equality checks.
2. RGB visibility of both objects at reset, contact, lift and release.
3. Oracle success and failure phase for each split.
4. Base-policy progress before the object-dependent failure stage.

If the Panda robot or camera change causes a general control failure on both
objects, that is a robot-adaptation failure, not evidence for object OOD. The
pipeline stops or enters a new diagnostic retry instead of relabeling it.

## 4. Required pipeline

### Stage A: registration and placement

Create the new manifest, task plan, seed manifest and Owner record before
resource allocation. Audit both candidate servers for actual idle GPUs, CPU/RAM,
disk headroom, native X-VLA runtime, model foundation, Panda source, BridgeData
assets and persistent output paths. Select the server jointly from these facts.

### Stage B: Panda task and Oracle

Implement the Panda-specific task wrapper, registration, camera contract,
metadata and EE6D adapter. Run a CPU or one-GPU reset/action-shape smoke, then
inspect representative RGB frames against object metadata. Run the fixed 20 ID
and 20 OOD Oracle gate; require at least 19 strict successes in each split.
Save all raw attempts, videos, actions, states, timelines and reset metadata.
Do not compensate for an Oracle failure by changing the success predicate.

### Stage C: ID demonstrations and base policy

Collect 128 successful ID Oracle demonstrations only. Audit anchor counts and
the temporal tail: the final observation must remain as an anchor with exactly
one valid target timestep, and padded timesteps/inactive dimensions must not
enter the loss. Compute and freeze the ID-only norm. Run the 2-step training,
reload and finite-forward smoke before formal training.

Train a fresh ID-adapted X-VLA-Pt policy for at most 10,000 steps, saving every
500 steps. Probe complete checkpoints on the fixed 20-episode ID selection
seeds. The first complete checkpoint meeting the 17/20 probe criterion is a
candidate; a separate 100-episode ID gate must reach at least 80/100 before
OOD detector or DAgger stages unlock. If no checkpoint qualifies, write
`ID_BASE_NOT_ACCEPTED` and preserve all evidence.

### Stage D: passive failure detection

Use the accepted ID policy and ID expert observations only for detector assets.
Calibrate fixed thresholds on independent successful ID policy rollouts, then
freeze them. Evaluate fixed held-out ID and object-OOD seeds with no detector
changing the policy action. Compare:

- Input pooled PCA;
- Panda Bridge/internal-feature PCA, LLMD and kNN;
- Action Expert final LLMD;
- ACC and STAC;
- official FIDeL-AD ResNet18/euclidean without a semantic VLM filter;
- CRSAIL observable-state k5 and clearly labeled vision adaptation when the
  implementation is compatible.

Report AUPRC, AUROC, AUCPDT, balanced accuracy, precision, recall, F1,
success-conditioned false alarms, score distributions and first-alarm timing.
Preserve per-decision scores and all rollout evidence. Failure labels and OOD
identity labels must not be conflated.

### Stage E: four data and training branches

Use the same accepted ID base, task contract and frozen norm for four
independent branches:

1. Internal Panda Bridge PCA gate;
2. Diff-DAgger gate;
3. Failure-Recovery after the frozen failure event;
4. Offline BC with complete OOD Oracle demonstrations.

The three gated collectors use a frozen raw mixed stream with strict ID/OOD
alternation and retain every raw attempt. Each gated branch targets 100
successful trajectories with a nonempty expert suffix; accepted split ratios
are reported as naturally produced, not forced to 50/50. Offline BC is kept as
a complete OOD Oracle baseline. All branches preserve actions, states, videos,
timelines, reset metadata, query/alarm times and expert-control intervals.

After collection, select a common matched low-level expert-action budget. Train
each method independently from the same immutable ID base with original-ID/new-
expert source balancing of 1:1, identical optimization settings and the correct
temporal mask. OOD data is never used to train the ID base.

### Stage F: final evaluation and registration

Evaluate each updated policy on independent held-out 100 ID and 100 object-OOD
seeds. Save videos, actions, states, timelines, reset metadata and summaries.
Report strict success, any supporting phase metrics, expert action cost and the
passive detector metrics. Register all paths in `comparison.json`, markdown and
the result index. Only after the evidence audit passes write `PIPELINE_COMPLETE`.

## 5. Evidence and recovery

Every stage has an independent marker, `pipeline_state.json`, command record,
PID, log and output directory. A persistent remote controller owns stage
transitions and resumes only from the first unverified stage. An engineering
failure gets a new retry directory; partial artifacts and logs remain intact.
Scientific failure writes the prescribed stop marker and does not silently
change task semantics, thresholds, seeds, horizon or model family.

The long-running Owner uses low-frequency `sleep`: 5--10 minutes during launch,
smoke, recovery and stage transitions; 20--30 minutes after stable throughput;
5--10 minutes again near checkpoint and gate completion. A heartbeat only
checks the controller and source-of-truth state; it must not replace the
controller's stage progression.

## 6. Robot-transfer interpretation

The main paper table is Panda-only object variation. A separate optional
ablation may compare the same eggplant task on WidowX and Panda, but that is a
robot-transfer axis and must not be presented as the object-OOD effect.
