# Robo-Dopamine Reproduction Status

This workspace targets a reproducible Robo-Dopamine-style experiment on RLinf pi0.5 with LIBERO.

## Paper-aligned pieces

- GRM receives the official eight-image ordering: reference start, reference goal, BEFORE front/left/right, AFTER front/left/right.
- Incremental, forward-anchored, and backward-anchored predictions reconstruct Phi using the hop equations.
- Consistency-aware mode uses the paper's Eq. 9-11 update when all three parser outputs are valid.
- Shaping uses `gamma * Phi_next - Phi_prev`; terminal transitions use a zero terminal potential and reset per-environment state immediately after the transition.
- One-shot adaptation uses the official Robo-Dopamine data preprocessing and fine-tuning scripts.

## Explicit deviations

- RLinf uses pi0.5 + Flow-SDE PPO. The paper reports PPO + OpenVLA-OFT and ReinFlow + pi0 in simulation.
- The paper's simulation result uses LIBERO-Goal; the earlier smoke runs used LIBERO-Spatial and are not formal paper comparisons.
- LIBERO provides one wrist camera in this setup, so the wrist image is duplicated to fill the official three-view schema.
- The automatic one-shot exporter initially creates uniformly spaced bootstrap segments. Strict paper reproduction requires replacing them with semantic keyframe annotations from one successful demonstration.
- The available `Robo-Dopamine-GRM-2.0-4B-Preview` checkpoint is a lightweight preview; paper tables report the 3B/8B variants.

## Formal comparison

Compare sparse reward and GRM-PBRS with identical pi0.5 initialization, PPO settings, tasks, seeds, and online simulator-step budget. Report true LIBERO success rate against environment interactions, success-curve AUC, interactions to fixed success thresholds, and mean +/- standard deviation across seeds.
