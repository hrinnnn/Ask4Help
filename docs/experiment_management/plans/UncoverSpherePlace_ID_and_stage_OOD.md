# UncoverSpherePlace ID 与阶段 OOD 审计

**状态：formal candidate 的 3x3 Oracle gate、9/9 中间状态续接审计与三 split collection smoke 均已通过；下一阶段为正式 ID/Handle-OOD/Goal-OOD 数据采集。**

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

正式采集仍需为三个 split 使用独立的新输出目录与固定 seed manifest；完成后再进行 ID base policy、failure detection 和 gated collection，不复用 smoke 数据。
