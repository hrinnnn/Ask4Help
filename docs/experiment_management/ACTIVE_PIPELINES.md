# Active Pipelines

更新时间：2026-08-20

本文件是当前长实验的执行总表。Owner、Leader 和 Heartbeat 每次接力必须先读本文件，再读对应 manifest、plan 与远端 `pipeline_state.json`。聊天历史中的旧模型路线或旧阶段不得覆盖本文件。

## OpenDrawer X-VLA Foundation Adaptation

- `pipeline_id`: `open_drawer_xvla_foundation_v1`
- `authorized`: `true`
- `owner_thread`: `019fdb64-2df4-7a53-9a66-7a5c9b9fe97a`
- `server`: `zhaozhixuan@111.198.58.150:12001`
- `current_stage`: `training_50000` (corrected v7 gripper contract; execution audit passed)
- `next_stage`: training completion, final checkpoint selection, then formal 100-ID gate
- `controller`: remote PID `3735453`; training launcher PID `3739073`; post-training gate controller PID `3744600`
- `pipeline_state`: `/sdd/ask4help-open-drawer/results/xvla_opendrawer_foundation_adaptation_v6/pipeline_state.json`
- `design_root`: `/sdd/ask4help-open-drawer/results/xvla_opendrawer_foundation_adaptation_v7/`
- `foundation`: `/data/zhaozhixuan/Ask4Help-airplane-5090/results/xvla_airplane_v1/model_cache/X-VLA-Pt-local`
- `domain_id`: `19` (audited unused; no domain table resize)
- `data_root`: `/sdd/ask4help-open-drawer/results/xvla_opendrawer_foundation_adaptation_v2/dataset/xvla_id_512_adapter_space/` (512 episodes, 91,533 frames; CPU adapter complete)
- `plan`: `docs/experiment_management/plans/OpenDrawer_XVLA_Foundation_Adaptation.md`
- `manifest`: `configs/pipelines/open_drawer_xvla_foundation_v1.json`
- `completion`: X-VLA 50k adaptation、checkpoint selection、独立100-ID gate与证据注册完成并写 `PIPELINE_COMPLETE`
- `execution_audit`: formal gate 前必须独立核对实际 X-VLA entrypoint、foundation/checkpoint、完整 prompt、512-ID manifest、domain id、8D-to-20D adapter、norm、temporal mask、effective batch/gradient accumulation、freeze/warmup、checkpoint reload，以及每个 selection/formal episode 的 video/actions/states/timeline/reset metadata。任一 mismatch 写 `ENGINEERING_PROTOCOL_DIAGNOSTIC`，不得进入正式 gate。

不可变条件：X-VLA foundation、完整 canonical prompt、原 ID 任务/成功定义、512-ID 数据 provenance、temporal/action mask。禁止恢复任何 pi0.5 resume 路线，禁止使用 Airplane/StackCube task checkpoint，ID `<80/100` 禁止 OOD/PCA/DAgger/two-way。

当前推进顺序：

1. CPU-only FK/IK、20D action、inactive-arm mask、30-step temporal-mask 测试（已通过）；
2. foundation domain row 审计并冻结 OpenDrawer `domain_id=19`/soft-prompt（已通过）；
3. 生成并审计 X-VLA 512-ID manifest、anchor/tail/norm；
4. 创建 restart-tolerant Controller；
5. 同拓扑 2-step/reload/checkpoint smoke；
6. 50k fresh adaptation，每5k checkpoint；
7. 固定20-ID selection和独立100-ID gate；
8. `>=80/100` 后按已批准下游计划继续，否则执行 manifest 中的 ID recovery/科学停止。

## StackPyramid Grasp Recovery

- `pipeline_id`: `stackpyramid_grasp_recovery_v1`
- `authorized`: `true`
- `owner_thread`: `019ff58e-8e47-7ca3-a028-07a2705e2c28`
- `server`: `root@39.101.70.188:1012`
- `current_stage`: `passive_pca_failure_detection_complete`
- `next_stage`: `needs_user_decision`
- `controller`: passive PCA retry8 complete; protocol audit passed; no active process
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

## Heartbeat Watchdog Rule

Heartbeat 每次只做：读取本文件、读取两个 manifest、核对实际 PID/controller/marker/progress。如果 `authorized=true` 且 `next_stage` 非空，但没有健康进程或完成 marker，必须唤醒 Owner 启动/修复，不得返回“无变化”。
