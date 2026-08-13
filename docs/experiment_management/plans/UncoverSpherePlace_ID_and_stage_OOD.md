# UncoverSpherePlace ID 与阶段 OOD 审计

**状态：环境 import/reset 与 20-seed paired-reset 审计已通过；当前用收窄的配对 nuisance 抖动重新进行 oracle gate，尚未进入正式采集。**

## 任务定义

机器人先移走覆盖蓝色小球的棕色方形遮挡物，将遮挡物放入固定 parking zone；随后抓取暴露的小球，并将其放入绿色目标碗。正式成功必须同时满足：遮挡物曾进入 parking zone、小球曾被抓取、小球释放后位于碗内且静止。

## ID/OOD 条件

- ID：遮挡物朝向为零，目标碗位于固定中心附近；物体中心只做小范围配对抖动。
- Handle OOD：只改变遮挡物 yaw，影响第一阶段的接近和抓取方式。
- Goal OOD：只镜像目标碗位置，影响第二阶段的运输和放置方式。

paired reset 审计已在 20 个相同 seed 上完成：`handle_ood` 只改变 cover yaw，`goal_ood` 只改变 bowl 位置，sphere 与其它记录的初始因素保持一致。正式冻结前仍需完成 oracle 门槛；若 Handle OOD 同时降低小球后续可见性或 Goal OOD 破坏第一阶段，则该条件不能进入 stage-localized timing study。

## 当前实现与门槛

环境与 oracle：`RLinf/rlinf/envs/maniskill/uncover_sphere_place.py`、`RLinf/rlinf/envs/maniskill/uncover_sphere_place_privileged_oracle.py`。import/注册、paired reset 和阶段 predicate smoke 已完成；前一版 20-seed oracle gate 为 ID 14/20、两个 OOD split 各 15/20，失败集中在小球抓取。当前恢复官方球体尺寸，并将仅用于配对审计的球/遮挡物中心抖动收窄到 $±0.02$ m，重新运行可复现 gate。ID、两个 OOD split 均须至少 19/20 严格成功，并验证 oracle 可以从“遮挡物已停入 parking zone”和“小球已抓住”两个中间状态继续完成任务。

未通过上述门槛前，不启动 base policy、failure detection、gated collection 或正式训练。
