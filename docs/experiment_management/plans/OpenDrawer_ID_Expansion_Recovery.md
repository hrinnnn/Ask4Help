# OpenDrawer ID Expansion Recovery

## 目标

验证 OpenDrawer 当前失败是否主要来自 ID demonstration 数量与状态覆盖不足。该实验只修复并重新验收 ID base，不进入任何 OOD、PCA 或 DAgger 阶段。

## 不可变条件

- 保持 OpenDrawer 任务语义、success predicate、机器人、观察和 action space 不变。
- 保持 `execute_horizon=5`、`max_episode_steps=400`、官方 pi0.5 Flow-SDE 配置、`global batch=128 / micro=32` 和 temporal mask。
- 旧 v7 step10000 checkpoint、summary 和失败 marker 只作 diagnostic；canonical recovery 从 immutable step4000 checkpoint 开始。
- 新数据只来自原 ID 分布，不引入 Handle、Grasp 或 Goal OOD。

## 阶段 A：新增 ID demonstration

1. 先在同一 ID reset 分布上做 Oracle smoke，确认任务可以稳定完成。
2. 新增约 128 条不与旧 seed 重叠的成功 expert demonstrations，使合并数据达到约 256 条。
3. 新旧数据都必须通过 action/state/video/parquet 边界审计，并报告 episode length 的 median、p95、timeout 数量、action horizon 和 tail-anchor/mask 分布。
4. 若出现大量达到 `max_episode_steps` 仍未完成、视频显著异常变长或 action chunk 与任务 horizon 不匹配，停止训练并先修复数据或环境协议。
5. Oracle 正式 ID gate 使用独立 100 个 seed；要求严格成功率至少 `95/100`，并且不能依赖截断状态判定成功。

## 阶段 B：重新训练 ID base

- 在合并后的 ID 数据上重新计算并冻结 norm stats。
- 从 immutable step4000 checkpoint 开始训练，最多 10000 steps，每 500 steps 保存 checkpoint。
- 训练前报告 anchor 总数、tail-anchor 数量、mask-validity 分布，并完成 2-step reload/forward smoke。
- 在 step6000、8000、10000 做独立 100-ID gate。预注册选择规则为：选择第一个达到 `80/100` 的 checkpoint；如果没有任何 checkpoint 达到门槛，则整个 recovery 失败，不得挑选一个较高但低于门槛的结果解锁下游。

## 通过条件与停止条件

只有同时满足以下条件才允许后续 OOD：Oracle ID gate 通过、数据审计通过、temporal-mask 审计通过、独立 ID policy gate 至少 `80/100`、100 个视频和 summary 完整。任一条件失败都保留 diagnostic，并写出原因排序；不得改变任务语义或 success predicate 来挽救结果。

## 预期解释

- 若增加 ID 数据后成功率显著提升并通过 `80/100`，支持“数据覆盖/训练覆盖不足”假设。
- 若 Oracle 和数据协议正常，但增加数据后仍长期低于门槛，则应优先怀疑长时序闭环、抓取/抬升/放置控制、action horizon 或模型适配，而不是继续盲目增加 OOD 实验。
