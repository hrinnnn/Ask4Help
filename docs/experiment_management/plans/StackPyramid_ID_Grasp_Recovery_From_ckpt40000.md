# StackPyramid ID Grasp Recovery From ckpt-40000

## Status

Future recovery plan. The total-50k `ckpt-40000` is preserved as an
audited recovery baseline, not as an accepted ID base. Its final checkpoint
evaluation used 100 fresh v4 episodes and complete evidence, but strict
success was `45/100`, below the `80/100` ID gate.

## Baseline Evidence

- Baseline checkpoint:
  `/mnt/data/ask4help/results/xvla_stackpyramid_oracle_repair_v3/continuation_50k_from_ckpt10000_lr1e-4_retry1/training/ckpt-40000`
- Final videos:
  `/mnt/data/ask4help/results/xvla_stackpyramid_oracle_repair_v3/final_checkpoint_formal_id_gate_100_retry3/videos/`
- Final evidence: `100/100` videos, actions and states; no formal evidence errors.
- Event counts: red grasp `56/100`, red lift `56/100`, red place `52/100`,
  blue grasp `51/100`, blue lift `51/100`, strict success `45/100`.
- Formal evaluation remains diagnostic until the ID threshold is met.

## Hypothesis to Test

The dominant failure appears before red grasp: the policy often moves above
the red cube and then hovers or repeats a near-static action. The formal
evaluation uses `max_episode_steps=300` at 10Hz, approximately 30 seconds.
Timeout must remain a failure and must not be hidden by changing the formal
success predicate or denominator.

## Stage 1: Read-Only Failure Audit

For all 100 baseline episodes, classify the first terminal bottleneck:

- approach or hover above red;
- gripper close without contact;
- red grasp but no stable lift;
- red placement;
- blue grasp/lift;
- timeout or other evidence failure.

Record end-effector-to-red distance, gripper value and sign, action velocity,
repeated-chunk count, last valid timestep, and whether the 300-step horizon
was reached. Compare successful and failed timelines. Do not train or alter
the task during this stage.

## Stage 2: Horizon Diagnostic

Run a separate 20-episode diagnostic with the same checkpoint, seeds and
geometry but `450` steps. This is not a formal result. It answers whether a
longer horizon allows eventual grasp:

- if grasp improves only at 450 steps, the issue is policy efficiency or
  action timing;
- if hover persists, the issue is likely gripper/action adapter, state
  normalization, or missing pre-grasp coverage.

The official 300-step evaluation remains unchanged.

## Stage 3: Adapter and Data Recovery

If the audit confirms pre-grasp failure, verify the existing custom
`8D joint-action -> 20D X-VLA -> 8D execution` adapter, especially gripper
sign, normalization, action chunk repetition and temporal masking. If the
adapter is correct, collect a new same-ID recovery set of 128--256 successful
demonstrations focused on approach, contact, close, stable lift and red
placement. Keep geometry, instruction, success predicate, domain and norm
fixed. Do not include OOD or template-only data.

Train a separate recovery branch initialized from `ckpt-40000`, with a new
durable root, checkpoint markers and restart support. The baseline and all
previous diagnostics remain untouched.

## Current Execution State

The baseline failure audit, the 20-episode 450-step diagnostic, and the
adapter/gripper audit have completed. The 450-step diagnostic obtained
`11/20` strict successes and remains diagnostic because the formal horizon is
unchanged at 300 steps. The adapter audit confirms finite 8D action arrays and
records the task-specific `8D -> 20D -> 8D` interface; it does not claim native
EE6D semantics. A fresh ID-only recovery collection is currently running from
the same baseline checkpoint with disjoint seeds. Its accepted demonstrations
must be audited before recovery training is launched.

## Gate and Stop Rules

- Use independent 20-ID probes only for checkpoint selection.
- Run one fresh 100-ID formal gate after selection.
- Promote a recovery checkpoint only at `>=80/100` strict success with
  complete videos/actions/states/timelines.
- If the recovery remains below threshold, preserve the branch as diagnostic
  and stop before OOD/PCA/timing.
- Never solve the problem by increasing the formal horizon, changing the
  success predicate, or removing timeout failures from the denominator.
