# StackPyramid 512 X-VLA Adapter Audit

## Status

`TRAINING_EXPOSURE_REVIEW_REQUIRED` and adapter semantics are unresolved.
The current full-planner ID collection may finish, but it must not be merged
or used for training until this audit is accepted. OOD/PCA/timing remain
locked.

## Domain Conditioning

The canonical X-VLA checkpoint reports `num_domains=30`, base
`domain_id=0` is a valid embedding index, and the official README describes
`domain_id` as the identifier of the current robotic embodiment/domain. The
repository does not provide a StackPyramid-to-domain mapping or establish
that domain 0 is the correct embodiment for this custom task. The current
StackPyramid training, evaluation and collector all hard-code
`domain_id=torch.zeros(...)`. Therefore the current launch package does not
yet establish a valid new-embodiment soft prompt/domain assignment.

Required decision before training: either validate domain 0 for this
StackPyramid adaptation with an explicit provenance argument and smoke, or
allocate and validate a new domain id together with the corresponding soft
prompt/embedding path. No training result may be treated as canonical before
this decision.

## Action Contract

The base checkpoint is configured as `action_mode=ee6d`, `real_action_dim=20`,
`max_action_dim=20`, and `num_actions=30`. Its action hub therefore represents
the official 20-dimensional EE6D interface. The StackPyramid HDF5 actions and
ManiSkill `pd_joint_pos` environment action space are both 8-dimensional
joint-position targets. The current adapter overrides the loaded config to
`action_mode=auto`, `real_action_dim=8`, `max_action_dim=20`, and
`num_actions=10`; `AutoActionSpace` pads 8D targets to 20D for the model and
trims model outputs back to 8D for execution. The loss reduces only over the
first eight dimensions.

This adapter is shape-compatible but is not automatically semantically
equivalent to the official EE6D action space. In particular, padding joint
targets into a 20-dimensional EE6D-pretrained head does not prove that the
head's learned coordinates represent joint position. A proper training
decision must choose between validating this explicit `auto` joint adapter
and converting the dataset/environment contract to the official EE6D 20D
representation.

## Smoke Evidence

The read-only H20 contract smoke reports:

- environment action shape: `[8]`;
- HDF5 action shape: `[280, 8]` for the canonical trajectory;
- adapter preprocess: proprio `[1,20]`, action `[1,10,20]`;
- adapter postprocess: `[1,10,8]`;
- replay of the canonical action trajectory: successful;
- domain mapping: unresolved (`domain_id=0` is currently used).

The direct attempt to pass an 8D tensor through the unmodified base `ee6d`
action space fails because that space expects its 20D gripper/layout indices.
This confirms that the current adapter is a real protocol change rather than
a no-op reshape.

## Exposure Calculation

For any audited merged dataset with (N) real action anchors, the proposed
20,000-step, batch-8 run exposes (160{,}000) sampled anchors, or
(160{,}000/N) effective dataset passes. With 1,000 freeze steps and 2,000
warmup steps, the nominal fractions are 5% and 10% of optimizer steps. The
actual (N), tail-anchor count and valid-mask distribution must come from the
512-episode audit; the old 72,171-anchor 256-ID report must not be reused as
the new exposure value.

## Decision Boundary

The collection can finish and be audited. Merge, 2-step training smoke,
fresh training, checkpoint selection and formal ID evaluation are blocked
until the domain-conditioning and action-semantics decisions are recorded,
the 512-anchor exposure report is generated, and the updated launch protocol
is approved.
