# UncoverSpherePlace ID 与阶段 OOD 审计

**状态：已登记，环境骨架已实现；等待 ManiSkill import、paired-reset 与 oracle smoke。**

## 任务定义

机器人先移走覆盖蓝色小球的棕色方形遮挡物，将遮挡物放入固定 parking zone；随后抓取暴露的小球，并将其放入绿色目标碗。正式成功必须同时满足：遮挡物曾进入 parking zone、小球曾被抓取、小球释放后位于碗内且静止。

## ID/OOD 条件

- ID：遮挡物朝向为零，目标碗位于固定中心附近；物体中心只做小范围配对抖动。
- Handle OOD：只改变遮挡物 yaw，影响第一阶段的接近和抓取方式。
- Goal OOD：只镜像目标碗位置，影响第二阶段的运输和放置方式。

正式冻结前必须用 paired reset 证明非目标物体、机器人、相机、目标因素和随机变量保持一致。若 Handle OOD 同时降低小球后续可见性或 Goal OOD 破坏第一阶段，则该条件不能进入 stage-localized timing study。

## 当前实现与门槛

环境骨架：`RLinf/rlinf/envs/maniskill/uncover_sphere_place.py`。下一步依次运行：import/注册 smoke、固定 seed 的 ID/Handle/Goal reset 审计、阶段 predicate 记录、事件驱动 oracle。ID、两个 OOD split 均须在冻结的 20 个 seed 上至少 19/20 严格成功，并验证 oracle 可以从“遮挡物已停入 parking zone”和“小球已抓住”两个中间状态继续完成任务。

未通过上述门槛前，不启动 base policy、failure detection、gated collection 或正式训练。
