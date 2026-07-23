# StackCube Resident Online AWBC

This document records the first resident-worker version of the StackCube
ask-for-help smoke. It is an engineering validation, not a performance claim.

## Fixed Contract

- Two independent pi0.5 members start from their own `global_step_7000`
  weights and remain loaded on GPU0 for calibration, OOD rollout, update and
  forward/reload verification.
- GPU1 first trains a one-shot LoRA adapter for
  `Robo-Dopamine-GRM-2.0-4B-Preview`, then keeps a vLLM endpoint serving the
  base and `stackcube-grm-lora` adapter simultaneously.
- All controller decisions and all trainable anchors span exactly 10 low-level
  actions. Tail segments shorter than ten actions cannot enter the buffer.
- The initial VFD threshold is the 0.95 quantile of five scores from each of
  five successful ID policy trajectories. It remains fixed for the smoke.

## One-Shot GRM Adaptation

`prepare_grm_oneshot_adaptation.sh` exports one fixed, successful ID episode
to the official Robo-Dopamine raw directory schema. Base camera becomes
`cam_high`; hand camera becomes `cam_left_wrist`; `cam_right_wrist` is a
documented duplicate of hand camera. Privileged StackCube event records create
the start, grasp, lift, near-target and success keyframes.

The official preprocessing, pair generation and postprocessing modules run
unchanged. The upstream trainer already supports PEFT LoRA. A small, logged
patch only exposes its target module list so this engineering run can use
`q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`, rank 16, alpha 32,
dropout 0.05, two epochs and learning rate `1e-5`. The base checkpoint is never
written to; only the adapter is saved.

## Buffer And Loss

The raw online archive contains every policy and expert-takeover chunk. The
quality buffer admits only valid, full ten-action chunks whose persistent
Flux-style weight is at least 0.1. The weight is computed once from the fused
Robo-Dopamine potential and is not re-normalized by an individual batch.

For `N` admitted chunks, the round draws `K=min(N,32)` online chunks without
replacement and `K` expert anchors. Expert selection is hierarchical: choose a
complete expert episode uniformly, then choose one legal anchor within it.

```
loss = (sum(expert flow loss) + sum(weight_i * online flow loss_i))
       / (number of expert anchors + sum(weight_i))
```

The worker uses a known global denominator and backpropagates each microbatch
numerator divided by that denominator. This is exactly the formula above while
releasing each microbatch graph promptly; it is not separate expert/online
averaging.

## Running And Gates

`run_resident_adapted_awbc_smoke.sh` performs adaptation, starts the adapted
endpoint, starts the Unix-socket worker, runs preflight/calibration/collection,
annotates two OOD trajectories, admits the buffer, updates both members, and
runs checkpoint equivalence plus resident forward smoke.

The preflight fails hard if GPU0 cannot hold both eval models and perform one
minimal backward/optimizer step for each member. The implementation never
falls back to a reload-per-round path. All artifacts live in
`/mnt/data/ask4help/results/stackcube_ask_for_help/robodopamine_adapted_buffer_v1/`.
