# Original OpenVLA Airplane Failure Detection

This experiment is isolated from the pi0.5/RLinf checkout and from the X-VLA
5090 task. It uses the official OpenVLA LoRA recipe with a small LeRobot bridge
for the controlled 8D `pd_joint_delta_pos` airplane data.

## Reproducible order

1. Validate the 98 ID episodes and 9,109 observations.
2. Compute and freeze q01/q99 action bounds from ID expert actions only.
3. Run the action-token encode/decode round-trip smoke.
4. Run a two-step LoRA train/reload smoke.
5. Train the ID policy for 10,000 steps on two H20 GPUs, saving each 500-step adapter checkpoint.
6. Select a checkpoint using ID validation only.
7. Extract all visual, projector, Llama-layer, prompt, and action-token representations.
8. Fit detector representations using ID expert observations only. Calibration thresholds are
   stored separately from the fitted assets.
9. Evaluate the fixed 50 ID and 50 OOD seed manifests with `not ever_grasped` as failure.

The main detector is projector-output pooled residual PCA with rank 1,000. All
other requested baselines are retained in the same detector manifest. ACC and
STAC are explicitly inapplicable because original OpenVLA emits one action
vector per inference rather than an action chunk.

## Server layout

The durable result root is:

`/mnt/data/ask4help/results/pick_single_ycb_airplane/openvla_original_lora_r32_v1/`

The dataset manifest, run configuration, metrics JSONL, checkpoints, feature
cache, detector assets, rollout timelines, and videos live below this root.
The environment archive is separate under `/mnt/data/ask4help/environments/`.
