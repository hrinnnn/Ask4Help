# pi0.5 + VFD + PlugCharger Online AWBC

## Controlled task definition

The implementation uses two registered ManiSkill environments derived from
official `PlugCharger-v1`:

- `RLinfPlugChargerID-v1`: charger yaw relative to the official `goal_pose` is
  uniformly sampled from `[-15°, 15°]`.
- `RLinfPlugChargerOOD-v1`: that relative yaw is uniformly sampled from
  `[165°, 195°]`.

Only charger yaw is changed after the official reset. Object XY placement,
receptacle, Panda initialization, cameras, physics, and ManiSkill's success
criterion remain official. Every collected episode records reset poses,
relative yaw, split, seed, and source.

## Implementation status

- **2026-07-21 visual-data integrity fix.** The first 128-demo collection at
  `plug_charger_id_128` is invalid for VLA training: its simulator state and
  action streams moved, but its retained RGB observation buffers were static.
  Do not use that dataset or the two SFT checkpoints trained from it. RLinf
  commit `111d1500` snapshots camera tensors at every environment step and
  rejects a trajectory whose complete camera sequence has no visible motion.
  The repaired one-episode acceptance artifact is
  `datasets/lerobot/local/plug_charger_id_visual_gate_20260721_0500`; both
  LeRobot images and its MP4 contain 76 distinct frames. Future collections
  must first pass this one-episode check and use a new, timestamped OSS output
  directory rather than overwriting a previous experiment path.
- Peg SFT monitor and both 2,000-step Peg jobs were stopped on 2026-07-20;
  their partial logs were preserved under
  `/mnt/data/ask4help/results/online_awbc_peg/aborted_peg_sft_*` and are not
  inputs to this experiment.
- Controlled reset, Plug observation mapping, Plug privileged Phi, and the
  one-chunk oracle are implemented in RLinf.
- The oracle reuses the official ManiSkill Plug solver geometry:
  `reach -> grasp -> close -> pre-insert -> insert`. It replans from the live
  state at every expert-controlled 10-action chunk.
- The ID expert collector records solver trajectories, replays them in
  `pd_joint_delta_pos`, emits LeRobot data, videos, reset metadata, and an
  expert progress sidecar.
- VFD runner supports `--task plug --split id|ood`; one decision is made every
  10 low-level actions, using 5 VFD flow samples. A high VFD grants exactly one
  oracle chunk; the next chunk always recomputes VFD.

## Reproducible execution order

1. Collect 32 successful ID expert trajectories and compute ID-only norm stats.
2. Run member 0 (`seed=1000`, GPU 0) and member 1 (`seed=1001`, GPU 1)
   serially for 250 SFT steps. Checkpoints are saved at 50, 100, 150, 200, and
   250. The script stops existing Ray and constrains `CUDA_VISIBLE_DEVICES`
   before launch; confirm each job reports `world_size=1` and occupies only its
   assigned GPU before accepting it.
3. Evaluate each checkpoint on the fixed ID seeds `10000-10019`. Select the
   earliest common checkpoint with success in `[25%, 50%]`; otherwise earliest
   at least 25%, otherwise step 250.
4. Calibrate one fixed VFD threshold from 20 policy-successful ID rollouts,
   five evenly spaced chunks per episode, with the 95th percentile. Fail if
   fewer than 20 successes occur in 200 attempts.
5. Run ten OOD rounds. Each round collects two rollouts from seeds beginning at
   20000, adds its natural policy/expert data to the accumulated dataset list,
   merges manifests in that exact list order, then performs 50 AWBC steps per
   model. Run `MODE=uniform` from the same reference checkpoint and same
   manifests as the Uniform BC control.

## Artifacts

Use OSS paths under `/mnt/data/ask4help/`:

```text
datasets/lerobot/local/plug_charger_id/
results/online_awbc_plug/id_expert/
results/online_awbc_plug/members/
results/online_awbc_plug/calibration/
results/online_awbc_plug/round_<n>/
```

Do not treat an instance-local environment as persisted. The verified RLinf
environment archive remains at
`/mnt/data/ask4help/environments/rlinf-openpi-maniskill-h20x2-20260720/`.
