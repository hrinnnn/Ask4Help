# Active Pipelines

更新时间：2026-08-25

本文件是当前长实验的执行总表。Owner、Leader 和 Heartbeat 每次接力必须先读本文件，再读对应 manifest、plan 与远端 `pipeline_state.json`。聊天历史中的旧模型路线或旧阶段不得覆盖本文件。

## X-VLA Fixed-Grid Task-Policy Knee Validation

- `pipeline_id`: `xvla_fixedgrid_taskpolicy_knee_v1`
- `authorized`: `true`; 用户已要求记录并执行 StackCube/Grab Plane fixed-grid task-policy knee validation
- `owner_thread`: `current-thread`; `owner_label`: `codex-root-xvla-knee-validation`
- `server_preference`: `zhaozhixuan@111.198.58.150:12001`; H20 `root@39.101.70.188:1012` 为回退
- `current_stage`: `stage_c_gate_data_needs_user_decision_diffdagger_budget`
- `next_stage`: `user_decision_then_protocol-preserving_retry_or_ineligible-branch-reconciliation`
- `run_root`: `/data/zhaozhixuan/Ask4Help-airplane-5090/results/xvla_fixedgrid_taskpolicy_knee_v1/`
- `runtime_5090`: X-VLA `/data/zhaozhixuan/X-VLA`; Python `/data/zhaozhixuan/envs/xvla_official_5090/bin/python` (historical successful StackCube collection environment; RLinf `.venv` planner-init segfault retained as diagnostic); independent worktree `/data/zhaozhixuan/xvla_fixedgrid_knee_work`
- `manifest`: `configs/pipelines/xvla_fixedgrid_taskpolicy_knee_v1.json`
- `plan`: `docs/experiment_management/plans/XVLA_FixedGrid_TaskPolicy_Knee_Validation.md`
- `source_commit`: `2f5d725`; local branch `codex/xvla-fixed-grid-knee`; server worktree is detached at `055c436` (the source commit plus metadata-only reconciliation; Stage-B utility, Stage-C passive result, and Diff-DAgger stop recorded)
- `implementation`: fixed-step collector, task-state knee summarizer, restart-tolerant calibration controller, matched-budget Stage-B trainer/evaluator, temporal-mask training shim, durable Stage-B total supervisor, frozen gate-to-knee audit, durable Stage-C passive gate controller, whole-episode exact-budget gate selector, gate-selected Stage-C data/training/evaluation controller, independent final reconciliation, and contract/knee/controller tests
- `resource_preflight_2026-08-25`: calibration and smoke evidence passed; Stage-B training and its 54 formal evaluations completed on the explicitly selected 5090 GPU5/CPU `0-19`. Other protected GPUs and the H20 owner are untouched. Training controller PID `4087857`; total supervisor PID `2926040`; formal evaluation PID `1138192` completed; diagnostic partial-evaluation controller PID `3302716` completed on GPU4/CPU20-39; Stage-C retry5 controller PID `2270221` completed passive audit on GPU5/CPU0-19; Stage-C total supervisor PID `2210895` launched gate-data controller; gate-data controller PID `2620830` stopped at Diff-DAgger exact-budget selection and awaits user decision; supervisor interval `900s`.
- `stage_b_roots`: training=`/data/zhaozhixuan/Ask4Help-airplane-5090/results/xvla_fixedgrid_taskpolicy_knee_v1/stage_b_training_v1/`; evaluation=`/data/zhaozhixuan/Ask4Help-airplane-5090/results/xvla_fixedgrid_taskpolicy_knee_v1/stage_b_evaluation_v1/`; supervisor=`/data/zhaozhixuan/Ask4Help-airplane-5090/results/xvla_fixedgrid_taskpolicy_knee_v1/stage_b_total_supervisor_v1/`
- `stage_c_root`: `/data/zhaozhixuan/Ask4Help-airplane-5090/results/xvla_fixedgrid_taskpolicy_knee_v1/stage_c_gate_v1/`; passive controller PID `2210894`; it waits for `STAGE_B_UTILITY_COMPLETE`, builds a missing Airplane `vlm_input_pool` asset from ID metadata if needed, then runs validation-ID calibration and held-out OOD gate audit with a 900-second resource wait.
- `stage_c_downstream`: passive marker triggers total supervisor PID `2210895` at `/data/zhaozhixuan/Ask4Help-airplane-5090/results/xvla_fixedgrid_taskpolicy_knee_v1/stage_c_total_v1/` (900-second interval); data root is `/data/zhaozhixuan/Ask4Help-airplane-5090/results/xvla_fixedgrid_taskpolicy_knee_v1/stage_c_gate_data_v1/`. Gate-training pools are StackCube `154000--154399` and Airplane `164000--164399`; all ten task--method selections must equal the frozen `520/2820` budget before 30 training jobs and 60 utility evaluations can start.
- `diagnostic_partial_evaluation_2026-08-26`: 用户已授权在正式 Stage-B utility evaluation 前先做独立部分评估。GPU4 已独立核验为空闲，控制器 PID `3302716` 以 `--start-now`、CPU20-39 完成运行，不等待 Stage-B 完成；它使用两个已完成的 knee checkpoint（StackCube/Airplane `step_20/seed_17001`），每个 task 做 20-ID + 20-OOD，使用独立 seed `190000--191119` 和独立输出根 `diagnostic_partial_evaluation_v1`。四组 summary 均为 20/20 rows、20/20 videos、20/20 actions：StackCube ID/OOD success=`13/20`/`7/20`，Airplane ID/OOD ever_grasped=`15/20`/`6/20`（strict=`4/20`/`2/20`）。`PARTIAL_EVAL_COMPLETE` 已写入；日志末尾的 `free(): invalid pointer` 发生在完整 artifact 写出之后，仅保留为 cleanup engineering diagnostic。该诊断不计入正式 100-episode 分母，也不改变正式阈值、seed 或训练协议；Stage-B 总控 PID `2926040` 在该 marker 后才允许启动正式评估。
- `stage_b_utility_2026-08-27`: 54/54 formal evaluation summaries and utility summaries are complete and independently denominator-checked (StackCube 30/30, Airplane 24/24; each 100 episodes/100 rows). Under the frozen matched-budget protocol, StackCube OOD success is best at anchor `0` (`0.7233±0.0416`), while calibration knee set is `{10,20}`; Grab Plane OOD ever-grasped is best at anchor `0` (`0.7967±0.0569`), while calibration knee set is `{20}`. Both knee/utility overlaps are empty, so Stage-B currently gives a negative result for “calibration knee predicts downstream SR”; this is a scientific result, not an engineering stop. Full table is tracked in `docs/experiment_management/diagnostics/XVLA_FixedGrid_StageB_Utility_20260827.md`.
- `stage_c_retry_20260827`: first Stage-C passive attempt failed before writing detector artifacts because the Airplane metadata handler was not registered; a second attempt exposed the empty `feature_cache` scaffold left by that failure. Both tracebacks are retained in the remote build log. The handler registration, import-order, and empty-scaffold retry fixes are now in the GitHub branch; the exact builder import-order/data smoke passed (`BUILDER_IMPORT_ORDER_SMOKE_OK`), retry3 PID `1455702` is alive and waiting for GPU5, and no Stage-C scientific rows have been accepted yet.
- `h20_preflight_20260827`: H20 has two idle H100 GPUs and large persistent storage, but its recorded X-VLA runtime is the StackPyramid environment (`/root/X-VLA-stackpyramid-clean`, GPU1 smoke policy) and the standard persistent paths do not contain the current Airplane task checkpoint/metadata bundle. Keeping the 5090 placement avoids an unplanned checkpoint/data/runtime transfer; Stage-C remains waiting for the co-located 5090 GPU5 resource.
- `stage_c_passive_20260827`: Stage-C asset build and passive gate audit completed with `ASSETS_COMPLETE` (12,632 ID observations, 37 layers), 50 validation-ID and 50 held-out-OOD rows per task, frozen validation-ID q=.95 thresholds, and `stage_c_gate_audit_complete_data_pending`. StackCube ID/OOD strict successes were `43/50` and `0/50`; its KD/KHR were Input PCA `27.5/0.20`, Bridge PCA `5.2/0.76`, Action PCA `9.5/0.82`, Diff-DAgger `73.61/0`, fixed Recovery `30/0`. Grab Plane ID/OOD ever-grasped was `47/50` and `24/50` (strict OOD `0/50`); all methods had KHR `0` against knee `{20}` (Input KD `21.7`, Bridge `66.57`, Action `75.87`, Diff `68.86`, fixed Recovery `30`). This is passive timing alignment only; gate-selected policy utility is still pending.
- `stage_c_diffdagger_budget_stop_20260827`: StackCube Diff-DAgger consumed the complete pre-registered `400/400` OOD pool but admitted only `2` full episodes (`52` expert actions), so the frozen whole-episode exact `520`-action selector failed closed. The collection and selector traceback are preserved; no threshold, suffix, success predicate, or method substitution was made. A remote `NEEDS_USER_DECISION_DIFFDAGGER_EXACT_BUDGET` marker is written, and the remaining gate-data training is paused pending approval of a larger pool, a new admission protocol, or an ineligible-branch reconciliation.
- `forbidden`: do not use old Stage-2 timing as formal input; do not launch on any protected GPU; do not tune thresholds/anchors on OOD; do not claim completion from smoke or partial calibration

## OpenDrawer Grasp-OOD Controlled Timing Sweep

- `pipeline_id`: `open_drawer_grasp_timing_sweep_v1`
- `authorized`: `true`; 用户已要求使用现有 ID success rate `>50%` checkpoint，固定 Grasp-OOD takeover timing 并训练/比较 downstream SR
- `owner_thread`: `current-thread`; `owner_label`: `codex-open-drawer-grasp-timing-sweep`
- `server`: `zhaozhixuan@111.198.58.150:12001`; adaptive timing training audits GPU ownership at each job boundary and acquires only actually idle cards from the 8-GPU pool, with at most 2 concurrent Ray jobs (user-authorized upper bound 4; GPUs with other processes remain protected)
- `current_stage`: `adaptive_single_seed_training`
- `next_stage`: `100-ID/100-Grasp-OOD evaluation -> independent reconciliation`
- `run_root`: `/data/zhaozhixuan/Ask4Help-open-drawer/results/open_drawer_grasp_timing_sweep_v1/`
- `active_execution_root`: `/data/zhaozhixuan/Ask4Help-open-drawer/results/open_drawer_grasp_timing_sweep_v1_retry6_adaptive/`; retry5 multi-seed/2500-step results remain diagnostic; formal collection is reused read-only from `/data/zhaozhixuan/Ask4Help-open-drawer/results/open_drawer_grasp_timing_sweep_v1_formal/formal/`, policy-only D-path calibration from retry2, and exact 5006-action budget from retry3
- `auxiliary_ood20_probe`: `/data/zhaozhixuan/Ask4Help-open-drawer/results/open_drawer_grasp_timing_sweep_v1_retry6_adaptive/ood20_probe/`; 20 Grasp-OOD episodes after each adaptive checkpoint on the held audited GPU, diagnostic only and separate from formal 100-episode denominators
- `priority_ood20_gate_20260829`: 用户最新决定：每个新完成的 timing checkpoint 先完成对应 20 条 Grasp-OOD 诊断评测并通过独立产物审计，再启动下一个 checkpoint 训练；探针资源等待时训练停在边界，seed/anchor/budget/success predicate 与已登记 GPU 候选不变，正式100-ID/100-OOD顺序与分母不变
- `dynamic_gpu_policy_20260830`: 用户授权实时监测 GPU；最多使用 4 张实际空闲卡训练。adaptive 控制器逐 job 审计 8-GPU 池并持有一张卡完成训练/20-OOD 交替，严格单 job 以避免 `/dev/shm` 超额；任何有其他进程的 GPU 都保持保护。
- `adaptive_training_policy_20260830`: 用户新增规则：每个 checkpoint 至少 5000 步；20 条 Grasp-OOD 严格 SR `<=40%` 时每次追加 2500 步，直到首次 `>40%`；冻结该累计步数供后续所有 checkpoint 使用，并严格按“训练→20-OOD 审计→下一训练”交替。retry5 的 2500-step 结果仅 diagnostic。
- `one_model_per_anchor_20260830`: 用户进一步确定每个 anchor 只训练一个模型，不再重复 seed；当前采用所有 anchor 共用冻结 seed `9301` 的映射，最终比较为 6 个 anchor 模型。每个模型至少5000步，首个 `>40%` 的累计步数冻结给后续 anchor；旧多seed结果不混入。
- `manifest`: `configs/pipelines/open_drawer_grasp_timing_sweep_v1.json`
- `plan`: `docs/experiment_management/plans/OpenDrawer_Grasp_Timing_Sweep.md`
- `source_commit`: `244a1ed`; fixed-timing collector and timing inputs remain synced from the local GitHub branch; adaptive controller `run_open_drawer_adaptive_timing_controller.sh` is recorded in the manifest and synced to the server source tree
- `base_checkpoint`: `/sdd/ask4help-open-drawer/results/open_drawer_pi05_v9_recovery_from5000_v4/training/v9_full_prompt/checkpoints/global_step_5000`; its independent ID policy rollout is `61/100` with complete `100/100` videos/actions/states/timelines/reset metadata. It remains labeled preliminary (`>50%` diagnostic base), not `ID_BASE_VALIDATED`.
- `task`: OpenDrawer `grasp_ood` (object yaw 80-100 degrees), max episode steps 400, execute horizon 5; current pure-policy audit is `96/100` drawer-opened, `11/100` ever-grasped, `0/100` strict success.
- `timing_anchors`: provisional frozen diagnostic set `{0,50,80,120,160,220}` representing immediate, pre-open, post-open, object approach, grasp boundary, and post-failure recovery. These are not changed after observing downstream SR; any revision requires a new manifest.
- `diagnostic`: `t=0` single-episode smoke produced a complete expert suffix (`257` actions) and full task-state timeline; planner first failed under pi0.5 2.7/NumPy2 runtime, then succeeded with the isolated `simplerenv_ms3` planner environment. The final smoke process emitted `free(): invalid pointer` only after artifacts were written; retained as cleanup diagnostic.
- `forbidden`: do not reuse old prompt-mismatch checkpoint or q80 gate collections; do not alter Grasp-OOD geometry, ID norm, success predicate, timing anchors after SR, expert budget, or protected GPU ownership; do not claim `PIPELINE_COMPLETE` from the smoke.

## OpenDrawer pi0.5 Continuation From global_step_10000

- `pipeline_id`: `open_drawer_pi05_resume_from10000_v9`
- `authorized`: `true`
- `owner_thread`: `019fdb64-2df4-7a53-9a66-7a5c9b9fe97a`
- `server`: `zhaozhixuan@111.198.58.150:12001`
- `current_stage`: `interrupted_server_reboot_data_mount_missing` (v1 Ray socket; v2/v3 multi-GPU illegal-memory; v4 DCP-load; v5 controller-path; v6 step-offset; v7 disk-capacity; v8 smoke migration; v9 dual-GPU FSDP diagnostics retained)
- `next_stage`: restore `/data` mount and native runtime, reconcile latest complete checkpoint, then resume/evaluate without treating the interruption as scientific failure
- `controller`: no live process after the 5090 reboot; last log step `6484/10000`, loss `0.0164--0.0205` in the final window, valid-action-ratio about `0.98`; complete checkpoints `global_step_2500` and `global_step_5000` remain on `/sdd`; interruption marker `/sdd/ask4help-open-drawer/results/open_drawer_pi05_resume_from10000_v9/ENGINEERING_SERVER_RESTART_DATA_MOUNT_MISSING`
- `run_root`: `/sdd/ask4help-open-drawer/results/open_drawer_pi05_resume_from10000_v9/`
- `retry_diagnostic`: `/data/zhaozhixuan/Ask4Help-open-drawer/results/open_drawer_pi05_resume_from10000_v1/ENGINEERING_RAY_SOCKET_PATH_DIAGNOSTIC`
- `ray_tmp_root`: `/sdd/od_pi05_10k_v9`
- `retry_diagnostics`: v1 Ray socket; v2 CUDA illegal memory; v3 multi-GPU NCCL; v4 single-GPU DCP load hang; v5 controller config path; v6 step-offset mismatch; v7 disk capacity; v8 smoke migration
- `gpu_candidates`: physical GPU2, world-size 1; v9 reuses v8's audited smoke from `/sdd/ask4help-open-drawer/diagnostics/pi05_weights_smoke_v8/`, then saves native 2500/5000/7500/10000 (cumulative 12500/15000/17500/20000), estimated ~76GiB with ~46GiB current margin
- `historical_topology_audit`: prior successful OpenDrawer pi0.5 continuation logs used `FlexiblePlacementStrategy` with `[[2]]` and `local_world_size=1`; current v9 deliberately matches that proven single-GPU topology. The earlier dual-GPU failures are not evidence that the previous successful run used two cards.
- `disk_audit`: last verified approximately `928GB` free on `/data` and `356GB` free on `/sdd`; formal output is on `/sdd`, so the four-checkpoint budget plus safety margin remains satisfied.
- `parallel_gpu_probe`: pure NCCL 2-card all-reduce on physical GPU4/5 passed, but isolated RLinf FSDP 2-card probes failed across FSDP1/FSDP2/no-shard/full-shard variants: initialization/forward illegal-memory or empty-shard errors, and no-shard step1 checkpoint failed in DCP reduce-scatter. Consolidated diagnostic: `/sdd/ask4help-open-drawer/diagnostics/pi05_2gpu_fsdpp_runtime_summary_v1/ENGINEERING_2GPU_RLINF_FSDP_CHECKPOINT_UNSAFE`. Earlier 4-card placement conflict remains at `ENGINEERING_GPU_MAPPING_CONFLICT_OOM`. Formal v9 remains world-size1 on GPU2; no dual-GPU run is promoted.
- `ckpt2500_id_diagnostic`: independent ID-only probe completed on GPU4 with seeds `86000..86019`; strict success `1/20` (`5%`), drawer-opened `16/20` (`80%`), grasp `8/20` (`40%`), lift `6/20` (`30%`), in-target `1/20` (`5%`). Evidence audit passed with `20/20` videos, actions, states, timelines, and reset metadata. Output `/sdd/ask4help-open-drawer/diagnostics/eval_ckpt2500_id20_v1/`; evaluator cleanup emitted `free(): invalid pointer` only after complete artifacts, retained as engineering diagnostic and not treated as evidence failure.
- `checkpoint_watcher`: `/sdd/ask4help-open-drawer/tools/watch_open_drawer_pi05_checkpoints.sh`; waits for native 2500/5000/7500/10000, checks full_weights/DCP sizes and CPU-side meta reload, then writes checkpoint audit and cumulative markers.
- `post_training`: `/sdd/ask4help-open-drawer/tools/run_open_drawer_pi05_v9_post_training_controller.sh`; waits for `TRAINING_COMPLETE`, then runs four 20-ID checkpoint probes, selects by highest strict success with earliest tie, and runs one audited 100-ID gate; never starts OOD automatically.
- `resume_checkpoint`: `/data/zhaozhixuan/Ask4Help-open-drawer/results/open_drawer_failure_detection_v1/id_base_continuation_from4000_v7/training/sft_from4000_to10000/checkpoints/global_step_10000/`
- `checkpoint_evidence`: `full_weights.pt` and `actor/dcp_checkpoint` present
- `contract`: full canonical prompt for the continuation, 8D action, action horizon 10, temporal mask, Flow-SDE, train-expert-only, AWBC false, global batch 128, micro batch 32, frozen ID norm
- `history_warning`: source checkpoint used the older simplified prompt; all prior gates remain diagnostic and are not relabeled
- `smoke`: v7 exact new-step smoke passed; native checkpoint `global_step_2` maps to cumulative `10002`, full_weights+DCP present, loss `0.0216/0.0250` finite, valid_action_ratio `0.967/0.974`, optimizer/scheduler fresh-reset
- `reload_forward`: `/data/zhaozhixuan/Ask4Help-open-drawer/results/open_drawer_pi05_resume_from10000_v7/smoke_weights_10000_to_10002/reload_forward_eval/`; artifacts complete, cleanup abort recorded as `SIMULATOR_EXIT_AFTER_ARTIFACTS`
- `ood`: locked; no PCA/DAgger/OOD before an independent ID gate reaches `>=80/100`

## OpenDrawer pi0.5 Short-Prompt Parallel Continuation

- `pipeline_id`: `open_drawer_pi05_shortprompt_parallel_v1`
- `authorized`: `true` by the latest user decision; current v9 remains running and untouched
- `owner_thread`: `019fdb64-2df4-7a53-9a66-7a5c9b9fe97a`
- `server`: `zhaozhixuan@111.198.58.150:12001`
- `current_stage`: `training_to_20000`; next stage native checkpoint `2500` then continued short-prompt training
- `run_root`: `/sdd/ask4help-open-drawer/results/open_drawer_pi05_shortprompt_from10000_v3/`; v1 Ray path and v2 LR-type failures retained as diagnostics
- `source_checkpoint`: immutable pi0.5 `global_step_10000`; weights-only with fresh optimizer/scheduler
- `prompt`: `open the drawer and place the object in the tray`; this exact prompt must be consumed by both training and future evaluation
- `training_contract`: single GPU4, world size1, global batch128/micro32, 8D action, action horizon10, temporal mask, Flow-SDE, train-expert-only, AWBC=false, conservative lr `5e-6`, warmup100, native checkpoints every2500
- `placement_preflight`: checkpoint/dataset/norm/native runtime co-located on 5090; GPU4 selected idle; GPU2 is protected for v9; output is persistent `/sdd`; short Ray root `/sdd/odsp4`
- `current_stage`: `interrupted_server_reboot_data_mount_missing`; no live process after the 5090 reboot. Last log step `3408/10000`, finite loss about `0.014--0.018`, valid-action-ratio about `0.97`; complete checkpoint `global_step_2500` remains on `/sdd`; interruption marker `/sdd/ask4help-open-drawer/results/open_drawer_pi05_shortprompt_from10000_v3/ENGINEERING_SERVER_RESTART_DATA_MOUNT_MISSING`
- `controller`: `/data/zhaozhixuan/Ask4Help-open-drawer/tools/run_open_drawer_pi05_shortprompt_parallel_controller.sh`; last controller PID `743424`, training PID `834181`; state file is stale after reboot; 2-step smoke passed with full_weights+DCP; GPU4 was correctly isolated before interruption
- `forbidden`: stop or modify v9, touch GPU2/other-user processes, start OOD/PCA/DAgger, alter task/success/norm/mask

## OpenDrawer pi0.5 Recovery After Runtime Restoration

- `pipeline_id`: `open_drawer_pi05_recovery_v4`; `authorized=true`; owner remains A4H15
- `current_stage`: `id_checkpoint_gate_v9_native_5000`; next stage short-prompt native `5000` then `7500` ID gates
- `runtime`: `/sdd/ask4help-open-drawer/runtime/pi05_rlinf_v4/.venv`; torch `2.7.1+cu128`, `sm_120` and CUDA tensor smoke passed; archive `/sdd/ask4help-open-drawer/runtime_archives/pi05_rlinf_v4_20260824.tar.zst` validated
- `v9_full_prompt`: complete native `5000/5000`, target cumulative `20000`, GPU4/CPU80-99, full canonical prompt, original LR `2.5e-5`; training exited cleanly. The fixed-ID gate is complete at `10/20` strict (`drawer_opened=19/20`, `grasp=14/20`, `lift=14/20`, `in_target=10/20`), below the `16/20` qualification threshold. Evidence is complete at `20/20` videos/actions/states/timelines/reset metadata under `/sdd/ask4help-open-drawer/results/open_drawer_pi05_recovery_v4_checkpoint_id_gates/v9_full_prompt/native_step_5000/`; the prior empty waiting directory remains preserved as `native_step_5000_waiting_gpu_diagnostic_20260825`.
- `short_prompt`: native `7500/7500` complete, target cumulative `10000`, GPU5/CPU100-119, prompt `open the drawer and place the object in the tray`, conservative LR `5e-6`; native checkpoints `2500`, `5000`, and `7500` are complete. The command and log prove this recovery starts at native2500 and runs 7500 new steps, so the stale target cumulative `20000` was reconciled to `10000` in `ENGINEERING_STATE_RECONCILIATION_FINAL`. The native5000 fixed-ID gate is `6/20` strict and native7500 is `5/20`, both below threshold; native7500 evidence is complete at `20/20` videos/actions/states/timelines/reset metadata. The watcher-written native5000 gate JSON formatting defect remains preserved and independently reconciled.
- `data/norm/task/mask`: same immutable 128-ID dataset, frozen norm, 8D action, horizon10 temporal mask, Flow-SDE and train-expert-only contract
- `checkpoint_gate_watcher`: watcher PID `2313468` uses the fixed JSON writer and hard free-memory/no-compute-app GPU pool `[4,5,6,7]`; it has verified v9 native5000 and shortprompt native5000 and waits for shortprompt native7500. Fixed ID seeds are `88000..88019`, `max_episode_steps=400`, `execute_horizon=5`; `>=16/20` writes `ID_BASE_VALIDATED_*`, otherwise diagnostic. `/sdd` has about `27GB` free while short training continues, so no checkpoint cleanup or new copies are allowed without a task-owned manifest decision.
- `disk_risk_repair`: after verifying v9 `training_complete` and no v9 process, the finished v9 `smoke_2step` (~19GB) and `runtime_tmp` (~5.8GB) were migrated to `/data/zhaozhixuan/Ask4Help-open-drawer/archive/open_drawer_pi05_v9_recovery_from5000_v4_20260825/`; original `/sdd` paths remain symlinks. Migration manifest records source/destination byte totals and completion-only guard. `/sdd` free space is now about `44GB`; active short-prompt runtime/checkpoints were not touched.
- `forbidden`: GPUs1/2/3 are protected; no multi-GPU FSDP, OOD, PCA or DAgger before ID gate

## OpenDrawer v9 Formal Failure Detection and Three-Stage OOD Override

- `pipeline_id`: `open_drawer_pi05_v9_formal_failure_ood_v1`; `authorized=true` by the explicit 2026-08-25 user decision
- `owner_thread`: `019fdb64-2df4-7a53-9a66-7a5c9b9fe97a`; `server`: `zhaozhixuan@111.198.58.150:12001`
- `base_checkpoint`: v9 full-prompt native5000 at `/sdd/ask4help-open-drawer/results/open_drawer_pi05_v9_recovery_from5000_v4/training/v9_full_prompt/checkpoints/global_step_5000`; independent ID probe `10/20`, so `base_policy_status=preliminary_id_checkpoint` and not `ID_BASE_VALIDATED`
- `user_override`: Oracle gate is `20` independent resets per split with strict `>=18/20`; formal failure-detection rows include FIDeL, CRSAIL, ACC and STAC in addition to internal PCA/LLMD/kNN and Diff-DAgger. The user explicitly permits these benchmark and OOD-training rows to be registered as formal results under the preliminary base; policy qualification remains reported separately.
- `splits`: `handle_ood` handle offset `0.085`, `grasp_ood` yaw `80--100` degrees, `goal_ood` goal center `y=+0.30`; each split remains independent and paired with ID factors.
- `legacy_q95_diagnostic`: `/data/zhaozhixuan/Ask4Help-open-drawer/results/open_drawer_pi05_v9_formal_failure_ood_v1_serial_retry3/` was stopped before completion and is marked `diagnostic_q95_stopped_user_switched_to_q90`; its accepted rows and partial Grasp/PCA parts remain unchanged and are not mixed into the new run.
- `legacy_q90_diagnostic`: `/data/zhaozhixuan/Ask4Help-open-drawer/results/open_drawer_pi05_v9_formal_failure_ood_v1_serial_q90_retry1/` was stopped on Grasp/PCA after `308` raw, `308` videos and `4` accepted (`2 OOD / 2 ID`) because the q=.90 trigger rate was too low; it is marked `Q90_GRASP_PCA_STOPPED_FOR_LOWER_THRESHOLD_DIAGNOSTIC` and is not mixed into q=.80.
- `q80_retry_diagnostics`: retry1 failed because its calibration omitted the required `detectors` mapping; retry2 failed because the controller launch expanded an undefined `CTRL`. Both roots are preserved with engineering diagnostic markers and are excluded from retry3.
- `current_stage`: `grasp_pca_ratio_retry6_collection`; retry4 completed all 12 method/split collections with 100 accepted trajectories each, complete raw videos and datasets, and independent raw-stream audits. The retry4 Grasp/PCA gate failure (`26 ID / 74 OOD`, pilot `12/20` OOD) remains preserved as diagnostic. Retry5 used the same q=`.80` threshold and disjoint seeds but omitted planner environment variables; all 79 trigger attempts were `planner exited before becoming ready`, so retry5 is preserved as an engineering diagnostic. Retry6 restores the known-good planner environment with the same fixed threshold, geometry and seeds `240000/340000`; controller PID `1701330`, collector PID `1701344`, and output root `/data/zhaozhixuan/Ask4Help-open-drawer/results/open_drawer_pi05_v9_formal_failure_ood_v1_serial_q80_retry6_grasp_pca_plannerfix/`. It is accepted only if it reaches 100 accepted with both formal and first-20 pilot OOD ratios `>=80%`; otherwise it writes a final diagnostic and training remains locked. No training or downstream stage is active.
- `disk_risk_update`: the completed task-owned archive `pi05_rlinf_v4_20260824.tar.zst` was byte-checked and migrated from `/sdd/ask4help-open-drawer/runtime_archives/` to `/data/zhaozhixuan/Ask4Help-open-drawer/archive/runtime_archives/pi05_rlinf_v4_20260824/`; the old `/sdd` filename remains a symlink, and active runtime/checkpoints were not touched.
- `disk_risk_update_v2`: the completed, unreferenced `runtime/pi05_rlinf_v3` (`INSTALL_COMPLETE`, about 13G) was regular-file-byte/file-count/rsync-dry-run verified and migrated to `/data/zhaozhixuan/Ask4Help-open-drawer/archive/runtime/pi05_rlinf_v3_20260824/`; the old runtime path remains a symlink. Active `runtime/pi05_rlinf_v4` and all checkpoints were preserved.
- `combined_metrics_update`: independently generated `combined_id_ood_v1` contains 72 rows for `id+handle_ood`, `id+grasp_ood`, and `id+goal_ood` (200 episodes each), using the existing trajectory-max score and frozen ID calibration; separate OOD splits were not merged together.
- `next_stage`: explicit protocol decision for the Grasp/PCA OOD-ratio failure, then (only if the decision preserves a valid pre-registered collection) rebuild the affected collection before matched-budget training and final evaluation. Oracle20 and detector assets are already complete; do not copy Airplane/StackCube metric values or q95/q90 rows into the q80 run.
- `manifest`: `configs/pipelines/open_drawer_pi05_v9_formal_failure_ood_v1.json`
- `retry6_live_update`: planner-fixed Grasp/PCA retry is active at `513 raw / 513 videos / 38 accepted` (`10 ID / 28 OOD`, 73.7% OOD; first-20 pilot 75% OOD), with zero planner errors and strict raw ID/OOD alternation; controller PID `1701330`, collector PID `1701344`. Retry5 remains an engineering diagnostic and retry4's 74% OOD-ratio gate failure remains scientific diagnostic.
- `retry7_live_update`: q=.75 threshold was rebuilt from the same 61 successful ID trajectories, reproducing q=.80=`0.2992814481` and yielding q=.75=`0.2827334106`; ID asset audit passes with zero OOD calibration rows. The active retry7 Grasp/PCA controller is PID `1886504`, collector PID `1886515`, using disjoint seeds `250000/350000`; training remains locked until the formal and first-20 pilot OOD ratios both reach `>=80%`.
- `retry8_live_update`: q=.65 threshold was rebuilt from the same 61 successful ID trajectories, q=.80 reproduction remains `0.2992814481`, and q=.65=`0.2680910826`; ID asset audit passes with zero OOD calibration rows. The retry8 Grasp/PCA controller PID `1968260` and collector PID `1968298` used disjoint seeds `260000/360000` on physical GPU2 and stopped at `211 raw / 211 videos / 48 accepted` (`17 ID / 31 OOD`, 64.6% OOD; first-20 pilot 65%), with planner errors=0 and strict raw alternation. Training remains locked.

## OpenDrawer X-VLA Foundation Adaptation

- `pipeline_id`: `open_drawer_xvla_foundation_v1`
- `authorized`: `true`
- `owner_thread`: `019fdb64-2df4-7a53-9a66-7a5c9b9fe97a`
- `server`: `zhaozhixuan@111.198.58.150:12001`
- `current_stage`: `diagnostic_stopped_user_switched_to_pi05` (latest complete checkpoint `ckpt-35000` retained)
- `next_stage`: none; X-VLA is retained as diagnostic while the pi0.5 continuation is the active mainline
- `controller`: stopped by explicit user model switch; no X-VLA process remains
- `pipeline_state`: `/sdd/ask4help-open-drawer/results/xvla_opendrawer_foundation_adaptation_v8/pipeline_state.json`
- `design_root`: `/sdd/ask4help-open-drawer/results/xvla_opendrawer_foundation_adaptation_v8/`
- `foundation`: `/data/zhaozhixuan/Ask4Help-airplane-5090/results/xvla_airplane_v1/model_cache/X-VLA-Pt-local`
- `domain_id`: `19` (audited unused; no domain table resize)
- `data_root`: `/sdd/ask4help-open-drawer/results/xvla_opendrawer_foundation_adaptation_v2/dataset/xvla_id_512_adapter_space/` (512 episodes, 91,533 frames; CPU adapter complete)
- `plan`: `docs/experiment_management/plans/OpenDrawer_XVLA_Foundation_Adaptation.md`
- `manifest`: `configs/pipelines/open_drawer_xvla_foundation_v1.json`
- `completion`: v8 stopped at the latest complete `ckpt-35000` by explicit user decision; X-VLA selection/formal gate was not promoted; all v7/v8/retry4/retry5 artifacts remain diagnostic
- `execution_audit`: formal gate 前必须独立核对实际 X-VLA entrypoint、foundation/checkpoint、完整 prompt、512-ID manifest、domain id、8D-to-20D adapter、norm、temporal mask、effective batch/gradient accumulation、freeze/warmup、checkpoint reload，以及每个 selection/formal episode 的 video/actions/states/timeline/reset metadata。v7 中断写入 `ENGINEERING_RESOURCE_OR_RUNTIME_DIAGNOSTIC`；v8 smoke/reload-forward 已通过。任一 mismatch 写 `ENGINEERING_PROTOCOL_DIAGNOSTIC`，不得进入正式 gate。

不可变条件：X-VLA foundation、完整 canonical prompt、原 ID 任务/成功定义、512-ID 数据 provenance、temporal/action mask。禁止恢复任何 pi0.5 resume 路线，禁止使用 Airplane/StackCube task checkpoint，ID `<80/100` 禁止 OOD/PCA/DAgger/two-way。

当前推进顺序：

1. CPU-only FK/IK、20D action、inactive-arm mask、30-step temporal-mask 测试（已通过）；
2. foundation domain row 审计并冻结 OpenDrawer `domain_id=19`/soft-prompt（已通过）；
3. 生成并审计 X-VLA 512-ID manifest、anchor/tail/norm；
4. 创建 restart-tolerant Controller；
5. 同拓扑 2-step/reload/checkpoint smoke；
6. 50k fresh adaptation，每5k checkpoint；
7. 固定20-ID selection和独立100-ID gate；ckpt-15000 diagnostic retry5 已完成并通过完整 evidence audit（strict `0/20`，不解锁）；
8. `>=80/100` 后按已批准下游计划继续，否则执行 manifest 中的 ID recovery/科学停止。

## OpenDrawer pi0.5 Representative Detector Calibration Recovery

- `pipeline_id`: `open_drawer_representative_detector_calibration_v3`
- `authorized`: `true`; active Goal owner `019fdb64-2df4-7a53-9a66-7a5c9b9fe97a`
- `server`: `zhaozhixuan@111.198.58.150:12001`
- `current_stage`: `completed_timing_diagnostic_clean_all_layers_not_met`
- `next_stage`: `user_decision_for_more_successful_policy_ID_calibration_or_new_temporal_detector`
- `run_root`: `/sdd/ask4help-open-drawer/results/open_drawer_calibration_recovery_v3/`
- `checkpoint`: `/sdd/ask4help-open-drawer/results/open_drawer_pi05_v9_recovery_from5000_v4/training/v9_full_prompt/checkpoints/global_step_5000`; fixed prior seed=`0`; canonical prompt and ID norm are recorded in the manifest
- `ID_provenance`: expert-ID is the 128-demo full observation bank; policy-success-ID is 20 independent strict-success episodes from `open_drawer_policy_id_calibration_v1_retry3`, matched at video/action/state level
- `representative_layers`: VLM visual input, VLM bridge, VLM block 08, Action Expert block 08, Action Expert final/pre-output
- `detectors`: pooled PCA residual versus source-aware tokenwise PCA+OT; q=`.80/.95`; visual valid masks are retained and action tokens remain time ordered
- `live_processes`: none; expert/policy asset builders and all rescore processes exited after their artifacts were audited
- `asset_audit`: expert-ID tokenwise=`22,973` observations; policy-success raw-replay tokenwise=`4,853`; raw-replay qpos reproduction=`0`; pooled and phase-aligned assets also audited finite
- `smoke`: tokenwise expert smoke=`1000` observations; policy-success video smoke=`2` episodes (`329` matched observations); remote unit tests=`5 passed`
- `timing_sources`: fixed saved ID source `/sdd/ask4help-open-drawer/results/open_drawer_representative_pca_v1_id_control_retry1` and Grasp-OOD source `/sdd/ask4help-open-drawer/results/open_drawer_representative_pca_v1_retry4/grasp_ood_smoke`; passive replay only, no new policy action sampling
- `forbidden`: no formal OOD collection, DAgger or training in this Goal; old q=.80 outputs remain Bridge-PCA diagnostics and are not relabeled as Input PCA
- `plan`: `docs/experiment_management/plans/OpenDrawer_Representative_Detector_Calibration_Recovery.md`
- `manifest`: `configs/pipelines/open_drawer_representative_detector_calibration_v3.json`
- `results`: `/sdd/ask4help-open-drawer/results/open_drawer_calibration_recovery_v3/metrics_final_v2/comparison.{json,csv,md}`; final audit=`/sdd/ask4help-open-drawer/results/open_drawer_calibration_recovery_v3/FINAL_AUDIT.json`; annotated videos=`/sdd/ask4help-open-drawer/results/open_drawer_calibration_recovery_v3/annotated_videos/`
- `completion`: `OPEN_DRAWER_REPRESENTATIVE_CALIBRATION_COMPLETE` is present. This closes the approved diagnostic plan; it does not claim all five layers meet clean timing, and formal OOD/DAgger/training remain locked.

## StackPyramid Grasp Recovery

- `pipeline_id`: `stackpyramid_grasp_recovery_v1`
- `authorized`: `true`
- `owner_thread`: `019ff58e-8e47-7ca3-a028-07a2705e2c28`
- `server`: `root@39.101.70.188:1012`
- `current_stage`: `passive_pca_failure_detection_complete`
- `next_stage`: `needs_user_decision`
- `controller`: passive PCA retry8 complete; protocol audit passed; no active process
- `remote_pause`: `USER_PAUSED_TRAINING_FOR_HORIZON600_AUDIT` at recorded `global_step=32280`; durable `NEEDS_USER_DECISION` marker is present, so no restart is authorized without a new user decision
- `pipeline_state`: `/mnt/data/ask4help/results/xvla_stackpyramid_oracle_repair_v3/grasp_recovery_v1/pipeline_state.json`
- `baseline_checkpoint`: `/mnt/data/ask4help/results/xvla_stackpyramid_oracle_repair_v3/continuation_50k_from_ckpt10000_lr1e-4_retry1/training/ckpt-40000`
- `baseline_formal`: `/mnt/data/ask4help/results/xvla_stackpyramid_oracle_repair_v3/final_checkpoint_formal_id_gate_100_retry3/`
- `plan`: `docs/experiment_management/plans/StackPyramid_ID_Grasp_Recovery_From_ckpt40000.md`
- `manifest`: `configs/pipelines/stackpyramid_grasp_recovery_v1.json`
- `completion`: recovery checkpoint达到独立100-ID `>=80/100` 并注册为 ID base，或写入预注册科学停止 marker
- `execution_audit`: passive PCA 只读评测在每个 split 收口前独立核对固定 `ckpt-40000`、v4 geometry、green/blue-only shift、paired seeds、600-step horizon、ID-calibrated threshold、原 policy actions、完整分母及 video/actions/states/timeline/reset metadata。PCA 不得改变动作；任一 mismatch 写 `ENGINEERING_PROTOCOL_DIAGNOSTIC`，不得注册最终指标。

Baseline：strict `45/100`、red grasp `56/100`、red place `52/100`、blue lift `51/100`，证据100/100完整。该 checkpoint 是 recovery baseline，不是已接受 ID base。

已完成 baseline failure audit、horizon-450 diagnostic `20/20`（strict `11/20`）和 adapter/gripper audit。ID-only recovery collection 与 audit 已通过：`128/128` accepted、`33,293` anchors、`1,152` tail anchors、`128/128` 视频可解码。用户已于 2026-08-21 批准从 `ckpt-40000` 继续训练额外 `50,000` optimizer steps。沿用已审计的640条ID-only数据、80/20 source-balanced batches、`lr=1e-4`、soft-prompt coefficient `0.1`、`bf16`、batch 8、冻结norm/adapter/formal contract；原proposal中的20k暴露长度由本次用户明确授权覆盖。

当前推进顺序：

1. 对100条 baseline timeline 做 first-bottleneck、hover、gripper、action-repeat、timeout 分类；
2. 同 checkpoint 做独立20条 `450`-step diagnostic，正式300-step协议不变；
3. 审计 8D/20D adapter、gripper sign、normalization、chunk execution；
4. 若确认数据覆盖不足，新增128--256条同ID pre-grasp/contact/close/lift demonstrations；
5. passive PCA failure-detection evaluation 已完成：retry8 固定 `ckpt-40000`、v4、600 steps、ID q=.95 threshold、score-only policy rollout；Stage2/Stage3 ID/OOD 四组均 `100/100` evidence，paired reset/runtime metadata audit PASS，最终 metrics 位于 `/mnt/data/ask4help/results/xvla_stackpyramid_oracle_repair_v3/grasp_recovery_v1/failure_detection_pca_v1_retry8/`。旧 root 因 runtime metadata 缺失保留为 diagnostic；Oracle失败分支仍完全隔离；
6. 固定20-ID selection和独立100-ID gate；
7. `>=80/100` 后才允许进入下游，失败则写科学停止 marker。

## PickSingleYCB Object Variation OOD

- `pipeline_id`: `pick_single_ycb_object_variation_pi05_v1`
- `authorized`: `true`
- `owner_thread`: `019ffbc4-f3a9-78f3-8684-e0b4cba3552a`
- `owner_label`: `codex-object-variation-pick-single-ycb`
- `server`: `zhaozhixuan@111.198.58.150:12001`
- `current_stage`: `diff_collection_scientific_gate_failed`
- `next_stage`: `needs_user_decision` after passive detection, Bridge-PCA, Failure-Recovery and Offline-Oracle completed but Diff-DAgger failed its collection denominator
- `run_root`: `/data/zhaozhixuan/Ask4Help-airplane-5090/results/object_variation_pick_single_ycb_v1/`
- `manifest`: `configs/pipelines/pick_single_ycb_object_variation_pi05_v1.json`
- `plan`: `docs/experiment_management/plans/PickSingleYCB_ObjectVariation_OOD.md`
- `baseline_protocol`: `docs/experiment_management/plans/PickSingleYCB_ObjectVariation_FailureDetection_Baseline_Protocol.md` (mandatory passive baselines, internal extensions, four downstream data/training rows, and evidence gate)
- `task_contract`: generic PickSingleYCB instruction; ID=`005_tomato_soup_can`; OOD=`008_pudding_box`; paired reset differs only in `object_model_id`
- `placement`: 5090 selected because native RLinf/OpenPI/ManiSkill, pretrained pi0.5 base and YCB assets are co-located on persistent `/data`; H20 rejected because an existing Ray owner occupies its resource pool and root filesystem is full
- `runtime_repair`: shared NumPy 2.4.4 remains untouched; pipeline-only NumPy 1.26.4 overlay restores the official MPlib Panda planner, which otherwise exits during `mplib.Planner` construction
- `gpu_plan`: current resume uses independently idle GPU1 with CPU `20-39`; GPU0 is occupied by another user's PID `1023482` (`/ws/bench_fine.py`) and GPU2 belongs to existing PID `1198612`; neither is touched
- `next_action`: user explicitly selected complete `global_step_4500` as the formal ID checkpoint despite its `15/20` ID probe; retain that probe as diagnostic, then run passive failure detection before any OOD data collection
- `evidence_so_far`: Oracle gate `20/20 ID + 20/20 OOD`; ID collection `128/128` with `128/128` videos; data audit `6634` anchors / `1152` tail anchors; ID norm and resume/reload smoke passed; `step4500` probe `15/20` with `20/20` videos/actions; passive detection ID/OOD `100/100`; Bridge-PCA `100/100`, Failure-Recovery `100/100`, Offline-Oracle `100/100`; Diff-DAgger `10/100` after `600` raw attempts; `NEEDS_USER_DECISION` written and matched training/final comparison blocked
- `user_diagnostic_override`: 2026-08-25 user authorized separate Diff threshold-sensitivity collection at override threshold `0.05` with `q=.95`, `patience=2`, preserving canonical threshold `0.6456781893968583`; diagnostic root is `collections_diagnostic_v1/diffdagger_low_threshold_005_retry1`; while it runs, provisional Bridge-PCA, Failure-Recovery, and Offline-BC training may run in the separate `provisional_training_while_diff_collection_v1` root. These outputs are not canonical until the Diff branch is audited and all four methods are retrained under matched expert-action budget.
- `runtime_recovery`: `/data` was re-mounted from the existing `/dev/sdb1`; the original run root was re-audited on `2026-08-24 11:11+08:00`. The selected formal ID checkpoint is the complete `global_step_4500`; all earlier resume retry diagnostics and the four-GPU NCCL diagnostic remain untouched. The old single-GPU training may finish independently, but downstream experiments use only the frozen user-selected step4500 checkpoint and ID norm.
- `completion`: only `PIPELINE_COMPLETE`, `NEEDS_USER_DECISION`, `ORACLE_NOT_ACCEPTED`, `ID_BASE_NOT_ACCEPTED`, or unrecoverable `PIPELINE_FAILED`

## X-VLA Put Vegetable in Basket Object Variation OOD

- `pipeline_id`: `xvla_put_vegetable_basket_object_ood_v1`
- `authorized`: `true`
- `owner_label`: `codex-xvla-vegetable-basket-object-ood`
- `owner_thread`: `current-thread`
- `server`: `root@39.101.70.188:1012`
- `current_stage`: `id_gate_scientific_stop_visible_retry1`
- `next_stage`: `needs_user_decision`
- `run_root`: `/mnt/data/ask4help/results/xvla_put_vegetable_basket_object_ood_v1/`
- `manifest`: `configs/pipelines/xvla_put_vegetable_basket_object_ood_v1.json`
- `plan`: `docs/experiment_management/plans/XVLA_PutVegetableBasket_ObjectVariation_OOD.md`
- `task_contract`: controlled `PutEggplantInBasketScene-v1`; ID=`eggplant`; OOD=`bridge_carrot_generated_modified`; same WidowX/sink/basket/camera/instruction/reset/success; only object asset changes
- `model_contract`: X-VLA `widowx-air` domain id `4`, real action 10D, model max action 20D, action chunk 30, fresh adaptation from `/mnt/data/ask4help/models/X-VLA-Pt_from5090_v4`
- `placement`: H20 X-VLA source/model/runtime are present; GPU0/1 remain registered to existing Ray/RLinf PID `276925`, but the user authorized a scoped shared-idle exception. Repeated pre-launch audits must show zero utilization and stable memory; do not signal or reconfigure the existing PID/Ray session. Root overlay remains 100% full; `/tmp` was cleaned to about 20 GiB free and new outputs/caches go to `/mnt/data` and `/tmp`
- `retry`: `rgb_visible_retry1`; new run root=`/mnt/data/ask4help/results/xvla_put_vegetable_basket_object_ood_v1_rgb_visible_retry1/`; fixed code now follows the official greenscreen exclusion for ID and OOD. The old hidden-RGB dataset/checkpoint remain diagnostic and are forbidden as retry inputs.
- `short_term_goal`: visible-object smoke passed; formal Oracle ID/OOD=`20/20` each; fresh ID collection=`128/128`; temporal-mask/norm audit and 2-step reload smoke passed; fresh ID SFT reached `10000` with every 500-step checkpoint; independent ID gate completed `20/20` episodes and `20/20` videos but achieved `0/20` success. OOD remains locked.
- `scientific_stop`: visible-RGB retry has a complete denominator but failed the ID gate; evidence=`/mnt/data/ask4help/results/xvla_put_vegetable_basket_object_ood_v1_rgb_visible_retry1/provenance/id_gate_evidence_visible_retry1.json`; action diagnostic=`/mnt/data/ask4help/results/xvla_put_vegetable_basket_object_ood_v1_rgb_visible_retry1/diagnostics/policy_action_trace_ckpt10000_seed94000.json`.
- `completion`: only `PIPELINE_COMPLETE`, `NEEDS_USER_DECISION`, `ORACLE_NOT_ACCEPTED`, `ID_BASE_NOT_ACCEPTED`, or unrecoverable `PIPELINE_FAILED`

## X-VLA Panda Put Vegetable in Basket Object Variation OOD

- `pipeline_id`: `xvla_panda_put_vegetable_basket_object_ood_v1`
- `authorized`: `true` by the user-set active goal; this is a new Panda line, not the existing WidowX line
- `owner_thread`: `019ffbc4-f3a9-78f3-8684-e0b4cba3552a`; `owner_label`: `codex-xvla-panda-vegetable-basket-object-ood`
- `server_preference`: 5090 selected after live preflight; H20 rejected because both cards belong to PID `276925` and its root filesystem is full
- `current_stage`: `id_sft_10000_v2`
- `next_stage`: `checkpoint selection -> independent ID gate -> passive detection`
- `run_root`: `/data/zhaozhixuan/Ask4Help-airplane-5090/results/xvla_panda_put_vegetable_basket_object_ood_v1/`
- `manifest`: `configs/pipelines/xvla_panda_put_vegetable_basket_object_ood_v1.json`
- `plan`: `docs/experiment_management/plans/XVLA_PandaPutVegetableBasket_ObjectVariation_OOD.md`
- `task_contract`: Panda BridgeData basket task; ID=`eggplant`, OOD=`eggplant` at fixed scale `1.25`; same sink, basket, prompt, paired reset and success predicate; only object size changes within the main comparison
- `model_contract`: fresh X-VLA-Pt adaptation with a newly audited Panda domain row; active 10D EE6D block padded to 20D; Panda adapter and temporal mask required; WidowX domain 4 and old WidowX outputs forbidden
- `execution_contract`: `preflight -> task/oracle smoke -> 128 ID demos -> ID-only SFT and gate -> passive detection -> four data branches -> matched-budget training -> final evaluation -> result registration`
- `placement_preflight`: passed on 5090; GPU5--7 were idle, `/data` had about 4.7TB free, native X-VLA runtime/foundation/Panda source/BridgeData assets were co-located; controller PID is recorded in remote `pipeline.pid`
- `oracle_gate`: `oracle_gate_v11` passed ID `20/20` and OOD `20/20` strict; profile is lift `0.35m`, release wait `60`, horizon `150`, object-local-y closing axis, three fresh-environment retries retained under `raw_attempts/`
- `controller`: `tools/run_xvla_panda_vegetable_basket_full_pipeline.py`; remote PID is in `pipeline.pid`; current child is ID SFT on physical GPU5/7 with gloo, batch8 and accumulation8 (effective global batch128)
- `forbidden`: do not modify or reuse `xvla_put_vegetable_basket_object_ood_v1`; do not mix WidowX data/norm/checkpoint/results; do not unlock OOD before the independent ID gate
- `completion`: only `PIPELINE_COMPLETE`, `ORACLE_NOT_ACCEPTED`, `ID_BASE_NOT_ACCEPTED`, `NEEDS_USER_DECISION`, or unrecoverable `PIPELINE_FAILED` after evidence audit

## Heartbeat Watchdog Rule

Heartbeat 每次只做：读取本文件、读取两个 manifest、核对实际 PID/controller/marker/progress。如果 `authorized=true` 且 `next_stage` 非空，但没有健康进程或完成 marker，必须唤醒 Owner 启动/修复，不得返回“无变化”。

对已授权的完整 pipeline，Heartbeat/Owner 必须持续监测并推进到 `PIPELINE_COMPLETE`、预注册科学停止、`NEEDS_USER_DECISION` 或用户明确暂停；不得因为设计、smoke、Oracle、collection、checkpoint、partial evaluation 或任一中间阶段完成就停止。用户要求“持续完成这个任务，直到完成目标”时，该要求属于本 pipeline 的持续监督契约。
