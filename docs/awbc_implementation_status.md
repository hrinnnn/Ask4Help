# pi0.5 AWBC Implementation Status

Last updated: 2026-07-16

## Git State

- Ask4Help branch: `codex/pi05-awbc`
- Ask4Help implementation commit: `9a5be90`
- RLinf branch: `codex/pi05-awbc`
- RLinf commits:
  - `f197af8c`: ARM AWBC weighting and unit tests
  - `bf11398e`: OpenPI dataset/loss integration and tests
  - `27e7fb00`: ManiSkill policy-rollout collection config
  - `760f2616`: rank-specific deterministic AWBC sampling

The server uses `/root/Ask4Help-awbc` as an independent worktree. This avoids
overwriting an unrelated uncommitted change in the older
`/root/Ask4Help/scripts/rlt_peg_smoke/run_stage2_smoke.sh` checkout.

## Implemented

- ARM exact gain normalization and distributed global statistics.
- Optional Robo-Dopamine robust controls, disabled by default.
- Per-sample pi0.5 flow-loss reduction and weighted mean.
- AWBC isolation to the action-expert flow loss.
- Stable JSON/JSONL sidecar indexing and strict validation.
- Exact 1:1 expert/policy valid-anchor batch sampling.
- Multi-dataset OpenPI loading through `ConcatDataset`.
- Three-mode Robo-Dopamine offline annotation with consistency fusion.
- Episode start state, successful terminal `Phi=1`, invalid parser handling,
  resumable annotation cache, and final-demo goal-bank extraction.
- Policy rollout collection that retains successful and failed episodes.
- Uniform BC, ARM exact, and robust AWBC launch paths.

## Automated Tests

Local AWBC suite:

```text
24 passed
```

It covers exact hand calculations, episode scaling, invalid/zero-variance
fallbacks, robust controls, simulated multi-rank statistics, gradient behavior,
manifest alignment, source-balanced sampling, success/reset isolation, and
invalid parser behavior.

The broader server suite run before the final smoke reported:

```text
50 passed
```

This included the AWBC tests, existing Dopamine GRM tests, and the ManiSkill
reward contract. Ruff, Python compile, shell syntax, and Git diff checks passed.

## Server Smoke Results

Hardware visible during this run:

```text
1 x NVIDIA H20 97871 MiB
```

Goal-bank extraction from the existing 8-demo Peg dataset succeeded. The goal
for task 0 came from episode 0, final dataset frame 50, with both main and wrist
images.

Fake-GRM annotation succeeded end to end:

```text
expert rows:       565
policy rows:       565
combined rows:    1130
valid transitions: 114 per source
stride:              5 environment frames
```

The policy manifest reused the expert annotation cache without endpoint calls.
This duplicated dataset is only an engineering fixture, not experiment data.
Artifacts are persisted at:

```text
/mnt/data/ask4help/results/awbc_peg_smoke/fake_annotation
```

### Uniform BC, 2 Steps

The run completed without errors. All weights were 1 and weighted/unweighted
losses matched.

```text
step 1 loss: 0.0780
step 2 loss: 0.0836
ESS:         4.0 / 4
```

Checkpoint and TensorBoard output:

```text
/mnt/data/ask4help/results/awbc_peg_smoke/uniform_2step
```

### ARM Exact AWBC, 2 Steps

The run completed without errors and produced non-uniform weights.

```text
step 1 weighted loss:   0.0865
step 1 unweighted loss: 0.0885
step 1 weight range:    0.312 to 0.915
step 1 ESS:             3.2 / 4

step 2 weighted loss:   0.0785
step 2 unweighted loss: 0.0748
step 2 weight range:    0.280 to 0.885
step 2 ESS:             3.2 / 4
```

Checkpoint and TensorBoard output:

```text
/mnt/data/ask4help/results/awbc_peg_smoke/arm_exact_2step
```

Both final checkpoints occupy about 19 GB and include a distributed checkpoint
plus full weights.

### Real Robo-Dopamine, One Episode

The `robo-dopamine` Python environment was restored from the existing OSS
snapshot. Its verified runtime was:

```text
torch 2.8.0+cu128
transformers 4.57.0
vLLM 0.11.0
CUDA available: true
```

The real preview checkpoint started through the OpenAI-compatible vLLM server
and served the Qwen3-VL architecture successfully. A short one-episode smoke
used the start and terminal frames, issuing three real eight-image requests for
incremental, forward, and backward modes. All three outputs parsed as valid and
the successful terminal transition produced `Phi_next=1`.

```text
real endpoint requests: 3
dataset rows:           51
valid transitions:       1
```

After the endpoint was stopped, the same annotation was replayed against an
unreachable port and reconstructed all 51 rows from the cache. Results and the
vLLM log are persisted at:

```text
/mnt/data/ask4help/results/awbc_peg_smoke/real_annotation_1ep
```

This shortened smoke used a 50-frame stride to test the real model quickly.
Formal annotation still uses the specified 5-frame stride.

## Still Required For Scientific Results

- Select the shared warm-start checkpoint by measured success closest to 25%.
- Collect 32 real warm-start policy episodes, retaining failures.
- Annotate the complete expert and policy datasets with the real GRM endpoint
  at the formal 5-frame stride.
- Run matched 500-step Uniform, ARM exact, and selected robust AWBC experiments.
- Evaluate all methods with 50 fixed seeds, then 3 training seeds and 100
  evaluation episodes per method.
- Run the real two-GPU integration smoke on an instance exposing two GPUs. The
  current instance exposes only one H20; simulated multi-rank statistics are
  already unit-tested, but a real two-rank NCCL run is still pending.

Fake-GRM results above establish engineering correctness only. They must not be
used as evidence that AWBC improves policy success.
