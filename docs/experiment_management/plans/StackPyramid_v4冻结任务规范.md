# StackPyramid v4 冻结任务规范

## 目的

v1、v2、v3 全部只保留为诊断结果。v3 的根因是 ID 红绿块中心相距约 `0.04 m`，小于 ManiSkill `next_to` 阈值约 `0.0616 m`，导致 `red_placed` 在 reset 后已经成立，Stage1/Stage2 不再表示真实的动作阶段。v4 必须先修复任务几何，再重新训练唯一的 ID policy；不能复用 v3 的 `ckpt-10000`、ID demonstrations、norm 或 detector asset。

## 唯一共享来源

Timing sweep 与四方法 gated-DAgger 必须从同一 Git 提交读取：

- 实现：`tools/stackpyramid_task.py`
- 运行选择器：`STACKPYRAMID_OOD_GEOMETRY=v4`
- task spec：`configs/stackpyramid_v4_task_spec.json`
- seed manifest：`configs/stackpyramid_timing_v4_seed_manifest.json`

不得由两个任务各自维护另一份 v4 几何、predicate 或 reset 逻辑。服务器上的 experiment root、checkpoint、ID H5、norm 和 calibration 都必须使用新的 v4 目录。

## 几何

ID 中心为：

| 物体 | xy 中心（m） |
|---|---|
| red | `[-0.080, -0.080]` |
| green | `[0.080, -0.080]` |
| blue | `[0.000, 0.120]` |

每个物体独立加入 `[-0.008, 0.008] m` 的 xy jitter。OOD 只移动指定物体：

| split | 物体 | shift（m） |
|---|---|---|
| Stage1 | red | `[0.120, 0.100]` |
| Stage2 | green | `[-0.120, 0.100]` |
| Stage3 | blue | `[0.100, -0.120]` |

按最大独立 jitter 计算，所有 split 的最小预计物体中心距离约为 `0.074 m`，高于此前测得的 `next_to` 阈值约 `0.0616 m`。这个静态裕量不是成功证明；必须用实际 ManiSkill reset smoke 读取 live threshold，并确认 reset 时所有 stage predicate 均为 false。

## 阶段契约

| split | prefix | target |
|---|---|---|
| Stage1 OOD | `red_grasped` | `red_lifted` |
| Stage2 OOD | `red_lifted` | `red_placed` |
| Stage3 OOD | `red_placed` | `blue_lifted` |

正式门禁要求事件顺序真实呈现为 `red_grasped -> red_lifted -> red_placed -> blue_grasped -> blue_lifted`。Oracle、base policy 和 locality gate 都必须记录 event step；reset 后 `red_placed`、`red_grasped`、`red_lifted`、`blue_lifted` 均不得提前为真。

## 执行顺序

1. 只用 v4 source 做 20 条 reset smoke，逐 split 检查物体距离、predicate 初值和事件顺序。
2. 通过 smoke 后，按 v4 manifest 完成 ID 与 Stage1/2/3 Oracle gate；每 split 至少 `90/100` 严格成功。
3. 重新采集 v4 ID demonstrations，重新计算并冻结 v4 ID norm，从预训练 base 重新训练唯一 v4 ID policy。禁止使用 v3 checkpoint。
4. 用新 v4 ID checkpoint 做 100 条 ID 与三种 OOD base-policy gate：ID 至少 `80/100`，每个 OOD 不超过 `50/100`，并通过 prefix/locality gate。
5. 门禁全通过后，Stage1/2/3 从同一个 immutable v4 ID base 独立收集、匹配专家动作预算、训练和评测；三个 stage 不串行续训、不混合数据。

所有失败阶段只保留 diagnostic，不自动进入下一阶段；Timing 任务与四方法任务必须共享同一 v4 source commit、manifest、checkpoint、norm 和 success predicate。
