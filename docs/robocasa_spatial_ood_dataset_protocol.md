# RoboCasa Spatial OOD Dataset Protocol

## Purpose

Build a controlled source-to-target dataset split for spatial generalization and
uncertainty-triggered expert help. The intended question is whether a policy
trained on one action distribution, such as opening a left drawer, can quickly
adapt when the corresponding object, fixture, or goal is on the right.

This protocol applies to released RoboCasa LeRobot demonstrations and to any
newly collected RoboCasa demonstrations.

## Candidate Tasks

Start with one task whose left/right variation changes the robot trajectory,
not only the language string:

- `DrawerUtensilSort`
- `GatherCuttingTools`
- `ToastOneSlotPair`
- `ClusterUtensilsInDrawer`

Use an atomic task such as `OpenDrawer` or `CloseDrawer` only as a diagnostic
smoke test. Use a composite task for the main uncertainty/help experiment.

## Do Not Label from Language Alone

Do not assign source or target using only the words `left` and `right` in an
instruction. Across kitchen layouts, a semantically left fixture need not be
on the same side of the robot coordinate frame. The split must be based on the
initial physical scene state and must be visually verified.

## Required Episode Inspection

For every candidate episode, read the RoboCasa dataset extras:

```text
extras/episode_<id>/ep_meta.json
extras/episode_<id>/model.xml.gz
extras/episode_<id>/states.npz
```

At reset (`t = 0`), replay or load the MuJoCo state and record:

- task name and instruction
- kitchen layout and style
- relevant fixture and object identifiers
- robot-base pose
- object, fixture, and goal poses in world coordinates
- poses expressed in the robot-base coordinate frame
- relevant yaw / orientation errors
- episode success and trajectory length

The authoritative spatial label is the signed lateral coordinate in the
robot-base frame. A small dead band around zero is excluded rather than
assigned to either side.

## Manifest

Create one immutable manifest per experiment, for example:

```json
{
  "episode_id": 42,
  "task": "DrawerUtensilSort",
  "instruction": "Open the left drawer and place the utensils inside it.",
  "layout": 12,
  "style": 27,
  "fixture": "drawer_2",
  "instruction_side": "left",
  "robot_frame_side": "left",
  "fixture_position_robot_frame": [0.31, -0.24, 0.81],
  "object_position_robot_frame": [0.10, -0.18, 0.85],
  "relative_yaw_rad": 0.08,
  "split": "source",
  "selection_version": "v1"
}
```

Store the manifest in Git. Keep raw demonstrations, videos, and rendered
replays in OSS rather than Git.

## Split Rules

For the first controlled study:

```text
source: only left-side episodes
target: only right-side episodes
```

Keep fixed, or tightly match, the following between source and target:

- task instruction template except the intended spatial variable
- robot embodiment and controller
- camera configuration
- fixture type and object category
- object count
- kitchen layout and style, initially limited to one to three layouts

Do not put target episodes in pi0.5 task adaptation, RLT Stage 1, action and
state normalization statistics, validation-driven checkpoint selection, or
offline replay initialization. Target episodes are reserved for zero-shot
evaluation and online adaptation.

## Verification Before Training

1. Render a random stratified sample of source and target episodes.
2. Check that labels match visible geometry and that trajectories succeed.
3. Report episode counts, lengths, fixture IDs, layouts, and spatial-coordinate
   distributions for both splits.
4. Confirm that source and target have no episode overlap.
5. Confirm that the target action distribution differs materially, for example
   through end-effector lateral trajectory and wrist-yaw statistics.

If layout, fixture identity, object type, and side all change together, the
experiment is confounded. Restrict the first benchmark to side as the only
intentional shift. Add cross-layout transfer as a separate harder setting.

## RLT Preparation

Materialize a source-only LeRobot subset after the manifest is approved.
Recompute normalization statistics using source episodes only. Convert or
adapt the selected RoboCasa observation/action fields to the RLT data contract
before training; a LeRobot layout alone does not make the data automatically
compatible with the current RLinf ManiSkill RLT implementation.

## Evaluation Sequence

1. Source-only policy on source and target: measure zero-shot generalization.
2. Source representation plus sparse online RL on target.
3. Add uncertainty-triggered expert takeover on target.
4. Add GRM potential-based shaping as a separate ablation.

For the help setting, log success, environment steps, number of help requests,
helped action steps, request timing relative to RoboCasa subtask annotations,
and success after policy control resumes.
