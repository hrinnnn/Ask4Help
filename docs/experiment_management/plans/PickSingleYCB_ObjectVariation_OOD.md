# PickSingleYCB Object-Variation OOD Pipeline

## Pipeline contract

- `pipeline_id`: `pick_single_ycb_object_variation_pi05_v1`
- `owner_thread`: `019ffbc4-f3a9-78f3-8684-e0b4cba3552a`
- `owner_label`: `codex-object-variation-pick-single-ycb`
- `authorized`: `true`
- `server`: `zhaozhixuan@111.198.58.150:12001`
- `run_root`: `/data/zhaozhixuan/Ask4Help-airplane-5090/results/object_variation_pick_single_ycb_v1/`
- `model`: pretrained pi0.5 Flow-SDE action-expert route; no StackCube, StackPyramid, OpenDrawer, or Airplane task checkpoint is allowed.

The task is a single-object pick-and-place. ID uses YCB model `005_tomato_soup_can`; OOD uses held-out YCB model `008_pudding_box`. The instruction remains role-based: `pick up the object and move it to the green goal`. The only intended split factor is `object_model_id`.

## Frozen task semantics

The Panda, table, base/wrist cameras, proprioception, control mode, action dimension, horizon, robot initialization noise, XY reset centers, XY jitter, object orientation, goal radius, and success predicate are shared. Each paired seed samples the same XY and goal-offset random values; only the model id changes. The object is placed flat with a fixed identity quaternion. Goal height is the same object-relative offset range so both shapes have a feasible target while preserving the same task semantics.

Strict success is the stock PickSingleYCB predicate: the object is within the fixed goal radius and the robot is static. Supporting fields include stable grasp, accepted lift, placement, and episode termination. A grasp-only episode is not strict success.

The event-driven oracle uses a top-down OBB grasp, stable-grasp confirmation, a real lift, transport to the goal, release, and final state evaluation. It must not use an OOD-specific candidate or alter the success rule.

## Seed and budget contract

| Purpose | Seeds | Denominator |
|---|---:|---:|
| ID demonstrations | 10000--10127 | 128 successful ID episodes |
| Oracle gate ID/OOD | 11000--11019 / 12000--12019 | 20 per split, at least 19 strict |
| Detector ID calibration | 13000--13024 | 25 independent successful ID rollouts |
| Passive detector ID/OOD | 14000--14099 / 15000--15099 | 100 per split |
| Mixed collection stream | 16000--16199 | fixed paired alternating attempts |
| Final held-out ID/OOD | 17000--17099 / 18000--18099 | 100 per split |

The ID base must reach at least `80/100` strict success on an independent formal ID gate before OOD collection, detector registration, or matched updates. If it fails, write `ID_BASE_NOT_ACCEPTED` and preserve all diagnostics.

The four data branches are Internal-Feature PCA, Diff-DAgger, Failure-Recovery, and Offline BC. Gated branches target 100 successful assisted trajectories with natural ID/OOD composition; raw attempts remain strictly paired/alternating and no source quota is imposed on accepted data. Offline BC collects complete OOD oracle demonstrations. Matched updates use the same original-ID/new-expert source balance, optimizer, training steps, and fixed total low-level expert-action budget of 12000 actions.

## Pipeline stages

`preflight -> oracle_smoke -> oracle_gate -> id_collection -> data_audit -> id_norm_and_sft -> id_checkpoint_selection -> id_formal_gate -> passive_failure_detection -> four_method_collection -> four_dataset_audit -> matched_training -> checkpoint_selection -> final_id_ood_eval -> result_registration -> PIPELINE_COMPLETE`

Every stage has an independent directory, PID/log, marker, manifest evidence, and retry path. Partial outputs are never reused by a new retry unless the audit explicitly accepts them. A controller must validate the current stage and start the next one in the same wake-up.

## Scientific and engineering gates

- Oracle gate: 19/20 strict success for both ID and OOD; no补试.
- Data audit: actions, states, videos, reset metadata, object model ids, full suffix/tail evidence, and split counts all match the manifest.
- Training: temporal tail anchors retained with `action_valid_mask`; 2-step train/reload/forward smoke before formal training.
- Detection: PCA and baselines use only independent ID calibration; passive rollout actions are unchanged by detector scoring.
- Final evaluation: held-out 100 ID + 100 OOD per method, complete denominator and evidence files.
- Terminal: only `PIPELINE_COMPLETE`, `NEEDS_USER_DECISION`, `ID_BASE_NOT_ACCEPTED`, or unrecoverable `PIPELINE_FAILED` ends ownership.

