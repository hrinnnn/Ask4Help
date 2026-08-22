# Active Pipelines

更新时间：2026-08-22

本文件是当前长实验的执行总表。Owner、Leader 和 Heartbeat 每次接力必须先读本文件，再读对应 manifest、plan 与远端 `pipeline_state.json`。聊天历史中的旧模型路线或旧阶段不得覆盖本文件。

## OpenDrawer pi0.5 Continuation From global_step_10000

- `pipeline_id`: `open_drawer_pi05_resume_from10000_v9`
- `authorized`: `true`
- `owner_thread`: `019fdb64-2df4-7a53-9a66-7a5c9b9fe97a`
- `server`: `zhaozhixuan@111.198.58.150:12001`
- `current_stage`: `training_retry_v9_sdd_disk_safe` (v1 Ray socket; v2/v3 multi-GPU illegal-memory; v4 DCP-load; v5 controller-path; v6 step-offset; v7 disk-capacity; v8 smoke migration; v9 dual-GPU FSDP diagnostics retained)
- `next_stage`: continue the already-passed smoke-backed training to cumulative `global_step_20000`, then checkpoint audit, checkpoint selection, and independent ID gate
- `controller`: v9 PID `1153921`; training PID `1153937`; checkpoint watcher PID `1820718`; repaired post-training controller PID `2124796`; state enricher PID `2133260`; last observed native additional step `2797/10000` at `2026-08-23T00:22+08:00`, cumulative `12797/20000`, loss `0.024`, valid-action-ratio `0.97`, marker `TRAINING_IN_PROGRESS`; `global_step_2500` checkpoint is present and under audit; next formal checkpoint is native `5000`; first checkpoint path `/sdd/ask4help-open-drawer/results/open_drawer_pi05_resume_from10000_v9/training/pi05_weights_from10000_to20000/pi05_weights_from10000_to20000/checkpoints/global_step_2500`
- `run_root`: `/sdd/ask4help-open-drawer/results/open_drawer_pi05_resume_from10000_v9/`
- `retry_diagnostic`: `/data/zhaozhixuan/Ask4Help-open-drawer/results/open_drawer_pi05_resume_from10000_v1/ENGINEERING_RAY_SOCKET_PATH_DIAGNOSTIC`
- `ray_tmp_root`: `/sdd/od_pi05_10k_v9`
- `retry_diagnostics`: v1 Ray socket; v2 CUDA illegal memory; v3 multi-GPU NCCL; v4 single-GPU DCP load hang; v5 controller config path; v6 step-offset mismatch; v7 disk capacity; v8 smoke migration
- `gpu_candidates`: physical GPU2, world-size 1; v9 reuses v8's audited smoke from `/sdd/ask4help-open-drawer/diagnostics/pi05_weights_smoke_v8/`, then saves native 2500/5000/7500/10000 (cumulative 12500/15000/17500/20000), estimated ~76GiB with ~46GiB current margin
- `historical_topology_audit`: prior successful OpenDrawer pi0.5 continuation logs used `FlexiblePlacementStrategy` with `[[2]]` and `local_world_size=1`; current v9 deliberately matches that proven single-GPU topology. The earlier dual-GPU failures are not evidence that the previous successful run used two cards.
- `disk_audit`: last verified approximately `928GB` free on `/data` and `356GB` free on `/sdd`; formal output is on `/sdd`, so the four-checkpoint budget plus safety margin remains satisfied.
- `parallel_gpu_probe`: pure NCCL 2-card all-reduce on physical GPU4/5 passed, but isolated RLinf FSDP 2-card probes failed across FSDP1/FSDP2/no-shard/full-shard variants: initialization/forward illegal-memory or empty-shard errors, and no-shard step1 checkpoint failed in DCP reduce-scatter. Consolidated diagnostic: `/sdd/ask4help-open-drawer/diagnostics/pi05_2gpu_fsdpp_runtime_summary_v1/ENGINEERING_2GPU_RLINF_FSDP_CHECKPOINT_UNSAFE`. Earlier 4-card placement conflict remains at `ENGINEERING_GPU_MAPPING_CONFLICT_OOM`. Formal v9 remains world-size1 on GPU2; no dual-GPU run is promoted.
- `checkpoint_watcher`: `/sdd/ask4help-open-drawer/tools/watch_open_drawer_pi05_checkpoints.sh`; waits for native 2500/5000/7500/10000, checks full_weights/DCP sizes and CPU-side meta reload, then writes checkpoint audit and cumulative markers.
- `post_training`: `/sdd/ask4help-open-drawer/tools/run_open_drawer_pi05_v9_post_training_controller.sh`; waits for `TRAINING_COMPLETE`, then runs four 20-ID checkpoint probes, selects by highest strict success with earliest tie, and runs one audited 100-ID gate; never starts OOD automatically.
- `resume_checkpoint`: `/data/zhaozhixuan/Ask4Help-open-drawer/results/open_drawer_failure_detection_v1/id_base_continuation_from4000_v7/training/sft_from4000_to10000/checkpoints/global_step_10000/`
- `checkpoint_evidence`: `full_weights.pt` and `actor/dcp_checkpoint` present
- `contract`: full canonical prompt for the continuation, 8D action, action horizon 10, temporal mask, Flow-SDE, train-expert-only, AWBC false, global batch 128, micro batch 32, frozen ID norm
- `history_warning`: source checkpoint used the older simplified prompt; all prior gates remain diagnostic and are not relabeled
- `smoke`: v7 exact new-step smoke passed; native checkpoint `global_step_2` maps to cumulative `10002`, full_weights+DCP present, loss `0.0216/0.0250` finite, valid_action_ratio `0.967/0.974`, optimizer/scheduler fresh-reset
- `reload_forward`: `/data/zhaozhixuan/Ask4Help-open-drawer/results/open_drawer_pi05_resume_from10000_v7/smoke_weights_10000_to_10002/reload_forward_eval/`; artifacts complete, cleanup abort recorded as `SIMULATOR_EXIT_AFTER_ARTIFACTS`
- `ood`: locked; no PCA/DAgger/OOD before an independent ID gate reaches `>=80/100`

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
- `current_stage`: `id_sft_formal_training_retry1`
- `next_stage`: `id_checkpoint_selection_and_formal_gate`
- `run_root`: `/data/zhaozhixuan/Ask4Help-airplane-5090/results/object_variation_pick_single_ycb_v1/`
- `manifest`: `configs/pipelines/pick_single_ycb_object_variation_pi05_v1.json`
- `plan`: `docs/experiment_management/plans/PickSingleYCB_ObjectVariation_OOD.md`
- `task_contract`: generic PickSingleYCB instruction; ID=`005_tomato_soup_can`; OOD=`008_pudding_box`; paired reset differs only in `object_model_id`
- `placement`: 5090 selected because native RLinf/OpenPI/ManiSkill, pretrained pi0.5 base and YCB assets are co-located on persistent `/data`; H20 rejected because an existing Ray owner occupies its resource pool and root filesystem is full
- `runtime_repair`: shared NumPy 2.4.4 remains untouched; pipeline-only NumPy 1.26.4 overlay restores the official MPlib Panda planner, which otherwise exits during `mplib.Planner` construction
- `gpu_plan`: reserve only independently idle GPU0/1/3/4; GPU2 belongs to existing PID `1198612`; never touch existing owners
- `next_action`: let single-GPU ID SFT retry1 finish; post-training controller then selects a checkpoint and runs the independent 100-ID gate
- `evidence_so_far`: Oracle gate `20/20 ID + 20/20 OOD`; ID collection `128/128` with `128/128` videos; data audit `6634` anchors / `1152` tail anchors; ID norm and 2-step reload/forward smoke passed; dual-GPU formal launch is preserved as engineering diagnostic
- `completion`: only `PIPELINE_COMPLETE`, `NEEDS_USER_DECISION`, `ORACLE_NOT_ACCEPTED`, `ID_BASE_NOT_ACCEPTED`, or unrecoverable `PIPELINE_FAILED`

## Heartbeat Watchdog Rule

Heartbeat 每次只做：读取本文件、读取两个 manifest、核对实际 PID/controller/marker/progress。如果 `authorized=true` 且 `next_stage` 非空，但没有健康进程或完成 marker，必须唤醒 Owner 启动/修复，不得返回“无变化”。

对已授权的完整 pipeline，Heartbeat/Owner 必须持续监测并推进到 `PIPELINE_COMPLETE`、预注册科学停止、`NEEDS_USER_DECISION` 或用户明确暂停；不得因为设计、smoke、Oracle、collection、checkpoint、partial evaluation 或任一中间阶段完成就停止。用户要求“持续完成这个任务，直到完成目标”时，该要求属于本 pipeline 的持续监督契约。
