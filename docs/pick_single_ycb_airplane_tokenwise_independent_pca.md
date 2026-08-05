# Airplane Independent Token PCA TopK-16

This protocol evaluates the airplane `global_step_3000` checkpoint, whose
cumulative training history is the frozen five-thousand-step ID policy. It uses
only the 98 ID expert episodes (9,109 observations) and their original frozen
ID norm asset. StackCube data, OOD data, failure labels, and threshold
calibration are excluded from fitting.

For each valid prefix position `j`, we fit its own mean, rank-1000 PCA basis,
and ID residual scale. There is no shared token subspace:

```text
mu_j = mean_i h[i, j]
U_j = PCA_i(h[i, j] - mu_j)
r_j = ||(I - U_j U_j^T)(h[j] - mu_j)||
z_j = (r_j - mean_ID(r_j)) / (std_ID(r_j) + eps)
score = mean(TopK_16(z_j))
```

Tokens with fewer than 1,001 valid ID observations never enter fitting,
standardization, or TopK. The two image blocks and language/state block are
explicitly tracked as `base_camera`, `wrist_camera`, and `language_state`.

The model's real action-generating forward can return the VLM input prefix and
the final VLM bridge tokens together. The evaluator calls that forward once per
decision, then scores all four methods from that probe. It does not generate a
second action solely for detection.

The stages are deliberately separate and refuse to overwrite their output:

```bash
cd /root/Ask4Help
PI05_BASE=/path/to/pi05 GPU_ID=0 \
  scripts/pick_single_ycb_airplane/run_tokenwise_independent_pca_topk16.sh features

PI05_BASE=/path/to/pi05 GPU_ID=0 \
  scripts/pick_single_ycb_airplane/run_tokenwise_independent_pca_topk16.sh assets

PI05_BASE=/path/to/pi05 GPU_ID=0 \
  scripts/pick_single_ycb_airplane/run_tokenwise_independent_pca_topk16.sh evaluate

PI05_BASE=/path/to/pi05 \
  scripts/pick_single_ycb_airplane/run_tokenwise_independent_pca_topk16.sh scan

PI05_BASE=/path/to/pi05 \
  scripts/pick_single_ycb_airplane/run_tokenwise_independent_pca_topk16.sh render
```

`evaluate` runs the fixed shared rollout set: 50 ID seeds `50000..50049` and
50 OOD seeds `60000..60049`, action replan horizon 5, and at most 250 low-level
steps. The primary outcome is `ever_grasped`; strict task success is auxiliary.

The scanner intentionally uses the same rollout labels to scan candidate
thresholds. Its selected thresholds and metrics are therefore explicitly
post-hoc/oracle results, not calibrated deployment thresholds.

Independent rank-1000 bases are large. The evaluator loads model plus both
locations' assets on one GPU before the first rollout and writes `preflight.json`.
If this fails, it terminates without substituting shared PCA or reducing rank.
