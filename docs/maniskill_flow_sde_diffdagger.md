# pi0.5 Flow-SDE PPO + DiffDAgger

The Chinese implementation record, including changed files, design decisions,
test results, and remaining validation work, is in
[`pi05_flow_sde_diffdagger_implementation_record.md`](pi05_flow_sde_diffdagger_implementation_record.md).

This branch connects DiffDAgger-style uncertainty intervention to RLinf's
pi0.5 Flow-SDE PPO path in ManiSkill. It does not download model checkpoints.

## Training path

1. pi0.5 samples an action chunk with Flow-SDE and records the denoising chain,
   old log probability, and value used by PPO.
2. The same action expert receives the generated normalized action at evenly
   spaced flow times. Its mean velocity reconstruction error is the uncertainty
   score:

   ```text
   x_t = t * noise + (1 - t) * action
   velocity_target = noise - action
   uncertainty = mean((velocity_prediction - velocity_target)^2)
   ```

3. An empirical CDF calibrated on in-distribution data converts that score into
   a threshold decision. The default is the original DiffDAgger setting:
   `alpha=0.99`, `patience=2`, and `patience_window=2`.
4. When a row crosses the gate, the expert model's full action chunk is executed.
   Non-intervened chunks remain ordinary on-policy Flow-SDE PPO samples.
5. Intervened chunks are excluded from PPO because their executed action came
   from another policy. They instead receive an auxiliary pi0.5 flow-matching
   SFT loss against the expert model action. With
   `openpi.train_expert_only=true`, this updates the action expert while leaving
   the VLM frozen.

The uncertainty definition, empirical CDF, and patience gate follow the public
DiffDAgger implementation. Flow velocity-MSE is the pi0.5 analogue of its DDPM
noise-prediction MSE. Combining expert SFT with PPO is an RLinf integration
choice, not a claim made by the original DiffDAgger paper.

## Configurations

- `maniskill_ppo_openpi_pi05_flow_sde.yaml`: Flow-SDE PPO without intervention.
- `maniskill_openpi_pi05_diffdagger_calibration.yaml`: writes uncertainty scores.
- `maniskill_ppo_openpi_pi05_flow_sde_diffdagger.yaml`: Flow-SDE PPO with the
  calibrated query gate and expert intervention.

The current environment defaults to RLinf's officially supported
`PutOnPlateInScene25Main-v3` pi0.5 ManiSkill pipeline. A downloaded pi0.5 base
checkpoint is enough for model-loading and engineering smoke tests, but a
task-adapted student and a successful task expert are required for meaningful
learning and intervention results.

## Calibration

For a paper-faithful experiment, build `DIFFDAGGER_CALIBRATION_PATH` from
uncertainty scores computed on the same in-distribution demonstration
observation/action pairs used to adapt the student. The calibration rollout
script is an engineering fallback that collects scores from ID environment
rollouts; it should not replace demonstration calibration in the final result.

```bash
export PI05_MODEL_PATH=/data/ask4help/models/pi05-student
export DIFFDAGGER_CALIBRATION_OUTPUT=/data/ask4help/results/diffdagger/id_scores.jsonl
bash scripts/maniskill_flow_sde/run_diffdagger_calibration.sh
```

When multiple rollout ranks are used, RLinf adds `_rank<N>` to the output file.
Merge the JSONL files before training.

## Smoke tests

Flow-SDE PPO only:

```bash
export PI05_MODEL_PATH=/data/ask4help/models/pi05-base-or-sft
bash scripts/maniskill_flow_sde/run_flow_sde_smoke.sh
```

DiffDAgger hybrid:

```bash
export PI05_MODEL_PATH=/data/ask4help/models/pi05-student
export DIFFDAGGER_EXPERT_MODEL_PATH=/data/ask4help/models/pi05-expert
export DIFFDAGGER_CALIBRATION_PATH=/data/ask4help/results/diffdagger/id_scores.jsonl
bash scripts/maniskill_flow_sde/run_diffdagger_smoke.sh
```

The expected diagnostics are `diffdagger/uncertainty_*`,
`diffdagger/cdf_mean`, `diffdagger/intervention_rate`,
`diffdagger/sft_samples`, and `diffdagger/sft_loss`.
