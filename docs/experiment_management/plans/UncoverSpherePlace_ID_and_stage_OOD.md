# UncoverSpherePlace ID 与阶段 OOD 审计

**状态：formal candidate 的 3x3 Oracle gate、9/9 中间状态续接审计、三 split 正式采集与 temporal-mask 验收均已通过；ID base policy 已完成 10000 步训练，但 checkpoint selection 在 2000/4000/6000/8000/10000 均为 0/20 ID success，已写出 `BASE_POLICY_NOT_ACCEPTED`，因此下游 detector、gated collection、更新训练与最终评测尚未启动。**

## 任务定义

机器人先移走覆盖蓝色小球的棕色方形遮挡物，将遮挡物放入固定 parking zone；随后抓取暴露的小球，并将其放入绿色目标碗。正式成功必须同时满足：遮挡物曾进入 parking zone、小球曾被抓取、小球释放后位于碗内且静止。

## ID/OOD 条件

- ID：遮挡物朝向为零，目标碗位于固定中心附近；物体与遮挡物中心固定在同一中心位置。
- Handle OOD：只改变遮挡物 yaw，影响第一阶段的接近和抓取方式。
- Goal OOD：只改变目标碗位置，影响第二阶段的运输和放置方式。

paired reset 审计已在相同 seed 上完成：`handle_ood` 只改变 cover yaw，`goal_ood` 只改变 bowl 位置，sphere、机器人初始关节状态与其它记录的初始因素保持一致。目标碗碰撞在球体首次抓取前关闭，抓取后恢复，因此 goal OOD 不会改变遮挡移除和球体抓取阶段。

## 当前实现与门槛

环境与 oracle：`RLinf/rlinf/envs/maniskill/uncover_sphere_place.py`、`RLinf/rlinf/envs/maniskill/uncover_sphere_place_privileged_oracle.py`。formal candidate 固定物体中心与 Panda 初始关节状态，保留官方球体尺寸；当前 `outputs/oracle_gate_stage_localized_v1.json` 为 ID 3/3、Handle OOD 3/3、Goal OOD 3/3，`outputs/oracle_resume_audit_stage_localized_v2.json` 为 9/9 中间状态续接成功。对应实现 commit 为 RLinf `ca252667`，审计修复 commit 为外层仓库 `aeef149`。

LeRobot 运行依赖已在固定环境中完成导入 smoke。三 split collection smoke 位于 `outputs/collection_smoke_stage_localized_v1_retry4/`：ID、Handle OOD、Goal OOD 均为 3/3 成功，所有 episode 均保持 action/observation 对齐，9 个 side-by-side 视频均可解码。该 smoke 只用于验收 collector，不作为正式训练数据。

正式采集输出为 `/data/zhaozhixuan/Ask4Help-uncover-sphere-place/outputs/collection_formal_stage_localized_v1_retry5/`，三个 split 各 128 条成功轨迹。`collection_validation.json` 已验证 action/observation 对齐、双视角图像、视频可解码、完整 episode 尾部 anchor 与 temporal validity mask；每个 split 的最后 observation 均保留且只有一个有效 target。2-step SFT 与 checkpoint reload/forward smoke 输出为 `id_sft_smoke_2_retry1/`，均已通过。正式 ID base policy 从 X-VLA 预训练基座启动于 `id_sft_10000_retry4/`，使用物理 GPU4/5、每 500 步保存；训练 marker 和 `ckpt-500` 至 `ckpt-10000` 均已生成。原 `post_training_pipeline_v1/` 在完整写出 checkpoint-2000 的 20 条评测后因底层库退出清理的 `free(): invalid pointer` 返回 -6；该输出已保留追溯。修复后的 `post_training_pipeline_v1_retry1/` 对完整输出进行验收后继续完成五个 checkpoint 的 20 条 ID selection，结果全部为 `0/20`，最终写出 `checkpoint_selection/BASE_POLICY_NOT_ACCEPTED` 并按协议停止。该阶段没有启动 detector、gated suffix collection、matched-budget 更新训练或最终 OOD 评测。
