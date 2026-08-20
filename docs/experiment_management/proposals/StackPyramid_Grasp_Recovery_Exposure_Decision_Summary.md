# StackPyramid Grasp Recovery: Training Decision Summary

## Decision Required

The 128-episode same-ID recovery collection has passed its data audit. No
training has been launched. Approval is required only for the recovery branch
configuration below; OOD, PCA, timing, and two-way experiments remain locked.

## Audited Evidence

- Existing 512-ID source: `138,979` anchors and `4,608` tail anchors.
- New recovery source: `128/128` raw attempts accepted, `33,293` anchors,
  `1,152` tail anchors, and `128/128` decodable videos.
- Proposed merged source: `640` episodes, `172,272` anchors, and `5,760` tail
  anchors.
- The recovery source is ID-only and contains no template-only or OOD data.

## Recommended Configuration

1. Continue from the audited `ckpt-40000` recovery baseline.
2. Use fixed source-balanced batches: `80%` existing 512-ID data and `20%`
   recovery data. This is close to the natural episode ratio while making the
   recovery exposure explicit.
3. Run `20,000` additional optimizer steps with batch size `8`, `lr=1e-4`,
   soft-prompt coefficient `0.1`, `bf16`, no new freeze period, and `2,000`
   warmup steps.
4. Save checkpoints at `5k/10k/15k/20k`. Evaluate each on the independent
   20-ID selection manifest `88500--88519`; then run one fresh 100-ID gate on
   `84400--84499` using the unchanged 300-step v4 formal protocol.

At 20k steps, the branch samples `160,000` anchors: `0.929` passes over the
combined source, approximately `0.921` passes over the existing source and
`0.961` over the recovery source. These figures describe the new branch only;
they do not replace the model's previous exposure history.

## Stop Rule

Only a complete `100/100` formal-evidence directory with at least `80/100`
strict successes can register an accepted ID base. Until then, preserve the
branch as diagnostic and do not start OOD, PCA, timing, or two-way evaluation.
