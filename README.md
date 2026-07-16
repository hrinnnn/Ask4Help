# VLA RL Workspace

This workspace tracks VLA RL code and experiments.

- RLinf: https://github.com/RLinf/RLinf
- RLT + pi0.5 + ManiSkill Peg reproduction: [docs/rlinf_rlt_pi05_maniskill_peg_reproduction.md](docs/rlinf_rlt_pi05_maniskill_peg_reproduction.md)

## pi0.5 LIBERO GRM Reproduction

The reproducible server workflow lives under `scripts/libero_grm/`.

On a prepared Aliyun DSW server with RLinf restored and `/mnt/data` mounted:

```bash
cd /root/Ask4Help

bash scripts/libero_grm/start_grm_endpoint.sh
bash scripts/libero_grm/download_libero_spatial_demos.sh
bash scripts/libero_grm/extract_libero_spatial_goal_bank.sh
bash scripts/libero_grm/run_task0_real_smoke.sh
bash scripts/libero_grm/run_short_train.sh
```

Default paths:

```bash
/mnt/data/ask4help/models/Robo-Dopamine-GRM-2.0-4B-Preview
/mnt/data/ask4help/models/RLinf-Pi05-LIBERO-SFT
/mnt/data/ask4help/libero_demos/libero_spatial
/mnt/data/ask4help/assets/grm_goal_bank/libero_spatial_demo_final
/mnt/data/ask4help/results
```

The default GPU split is `GRM_GPU=0` for the vLLM GRM endpoint and
`TRAIN_GPU_RANK=1` for RLinf. Override these environment variables if the
server layout changes.
