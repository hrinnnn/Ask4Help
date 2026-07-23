# StackCube Ask-for-Help + Online AWBC Smoke

## Scope

This smoke stops after proving the real closed loop:

1. Calibrate one fixed VFD threshold from five successful ID rollouts, with
   five uniformly spaced decision chunks per rollout.
2. Collect two OOD trajectories. Every decision executes five low-level
   actions; VFD uses five flow candidates and candidate zero is the policy
   action. A score above the threshold gives the current chunk to the
   current-state ManiSkill oracle.
3. Query the real Robo-Dopamine endpoint every five steps in incremental,
   forward, and backward modes with consistency-aware fusion.
4. Build training targets from `Phi(t + 10) - Phi(t)`. Invalid GRM results stay
   invalid and receive zero weight.
5. Run two independent two-step Flux-style AWBC updates from the two step-7000
   checkpoints, then reload each checkpoint and run one action forward pass.

No ten-round online experiment or post-update success evaluation is part of
this smoke.

## Flux Weight Rule

- `delta < 0`: zero
- `0 <= delta <= 0.01`: mean/std soft weight
- `delta > 0.01`: one
- invalid: zero

The base weight is multiplied by the episode-length factor and positive
weights are normalized to mean one within the batch. Training stops before
launch when the complete sidecar has no positive-weight records.

## Durable Outputs

All runtime artifacts are written under:

```text
/mnt/data/ask4help/results/stackcube_ask_for_help/
  vfd_robodopamine_awbc_smoke_step7000/
```

The directory contains calibration scores and threshold, assisted rollout
videos and controller timeline, Robo-Dopamine cache/sidecar and weight summary,
both AWBC checkpoints and logs, and both reload-forward summaries.
