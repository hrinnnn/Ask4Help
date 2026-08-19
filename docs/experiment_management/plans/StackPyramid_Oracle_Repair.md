# StackPyramid Oracle Repair

## Scope

Repair only the privileged StackPyramid Oracle action sequence. The v4 geometry, task instruction, success predicate, formal horizon, robot/camera setup, and frozen seed definitions remain unchanged. All old 128/256 demonstrations, norms, checkpoints, and timing/four-method outputs remain diagnostic and are excluded from the repaired dataset.

## Required sequence

The red-block plan must be strictly monotone:

`approach -> single grasp/close -> lift to safe height -> closed horizontal transport beside green -> vertical lower -> one release inside target tolerance -> retreat -> blue approach/grasp/lift/stack/release`.

The repaired Oracle must not release red before the target, return red to its initial region, or issue a second red grasp. The blue phase may start only after the red placement event has been verified.

## Read-only audit before editing

Audit representative current-v4 Oracle episodes, actions, state timelines and videos. Report phase, gripper command, red object pose, and action step for every grasp, release, red placement and blue transition. Confirm or reject the suspected `grasp -> early release/drop -> adjustment/regrasp -> re-transport` pattern before changing the implementation.

## Frozen gates

1. **Five-seed event smoke:** fixed v4 seeds, complete action/state evidence, and machine-checked event order.
2. **Twenty-seed audit:** complete videos/actions/state timelines, per-episode event counts, action duration, stall and truncation checks.
3. **Independent 100-seed Oracle gate:** use the frozen v4 Oracle gate seed definition, retain the complete denominator and failure reasons, and require the pre-registered Oracle success threshold.

Every gate must verify:

- exactly one valid red grasp;
- gripper remains closed throughout red transport;
- zero red releases before the target region;
- exactly one red release inside target distance/height tolerance;
- blue phase begins only after red placement is complete;
- no abnormal slow episode, unexplained pause, horizon truncation, or missing action/state/video evidence.

## Data and training boundary

No old demonstration may enter the repaired dataset. After the 100-seed Oracle gate passes, collect at least 128 and preferably 256 new ID demonstrations using the repaired Oracle, audit action/frame/state alignment and temporal-mask tail anchors, then train a fresh StackPyramid ID base from the original X-VLA base. Do not continue any old `ID_GATE_FAILED` checkpoint.

## Output and stop conditions

All outputs use the independent root `/mnt/data/ask4help/results/xvla_stackpyramid_oracle_repair_v1/`. Each gate writes its own marker, summary, videos, actions and state timelines. Any event-order, gripper, duration, evidence or Oracle-success failure stops the pipeline in diagnostic status; geometry, success predicate, horizon, seed semantics and thresholds must not be changed to obtain a pass.

## Formal ID Gate Boundary

The 256-data fresh ID training reached `TRAINING_COMPLETE` with the canonical original X-VLA base and all 20-episode probes remain diagnostic. The first formal 100-ID launcher failed before episode 1 because the optional evidence path attempted to convert a CUDA state tensor directly to NumPy. It is recorded as `FORMAL_GATE_FAILED_ENGINEERING_STATE_SERIALIZATION` under the formal gate root; no formal success/failure decision or `ID_BASE_VALIDATED` marker was written. The minimal fix serializes state through `detach().cpu().numpy()` and is committed in the source tree, but the formal evaluator is not relaunched in this phase without a new execution authorization.


## Final Formal ID Gate Closure

The repaired formal lifecycle smoke subsequently passed in
`formal_geometry_lifecycle_smoke_retry4`: three independent episodes used
fresh environments, the recorded environment was
`Ask4HelpStackPyramidID-v4`, all reset predicates were false, and the v4
red--green reset distance was within the frozen `[0.14, 0.18]` m range.

The canonical formal evaluation is
`formal_id_gate_100_retry4` with checkpoint `ckpt-10000` and frozen seeds
`84400--84499`. It contains `100/100` episodes, videos, action arrays, and
state timelines; all rows pass the formal evidence checks and
`formal_evidence_errors=[]`. The evaluation used geometry `v4`, environment
ID `Ask4HelpStackPyramidID-v4`, and a fresh-environment-per-episode
lifecycle. The scientific result is `ever_grasped=82/100`,
`red_lifted=41/100`, `red_placed=0/100`, and `strict_success=0/100`. It
therefore writes `FORMAL_ID_GATE_FAILED` and
`ID_BASE_NOT_ACCEPTED_FORMAL_100`.

Retry1 remains an engineering diagnostic for state serialization, retry2 is
invalid because the formal evaluator used the wrong geometry, and retry3 is
an engineering diagnostic for reset lifecycle contamination. None of these
directories is merged with retry4 or used to change its denominator. Because
the canonical ID base is not accepted, prefix/locality gates, OOD collection,
PCA calibration, timing sweeps, and downstream training remain locked.
