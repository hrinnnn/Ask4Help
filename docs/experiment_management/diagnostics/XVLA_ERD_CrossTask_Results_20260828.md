# 三任务 ERD-Pose replay 与可视化结果

日期：2026-08-28（北京时间）  
范围：StackCube Stage 2 target-position OOD、PickSingleYCB object variation、
OpenDrawer（ID/Handle/Grasp/Goal）。StackPyramid 按用户决定排除。  
性质：diagnostic-only；没有修改 formal pipeline、checkpoint、seed、success predicate、
训练或 OOD 解锁状态。

## 1. 做了什么

1. 使用现有 summary/action/state/parquet 产物，重新得到 Panda TCP position、quaternion、
   gripper width 和有限差分 velocity；没有重新运行 policy inference。
2. 对每个 task 用 expert reference 的 leave-one-context-out residual 计算 robust feature
   scale，再按 q=.925 和 q=.95 计算 task-specific threshold。
3. 用 causal monotonic alignment 对齐 learner/reference，在 decision stride=5、
   persistence=2 下计算第一次持续越阈的 onset step。
4. 对 ID/OOD policy 分别计算 score 的中位数与 P25--P75 分布，并保留已有 gate 的报警时刻
   作为旁证，不把 PCA/Diff 分数和 ERD 分数混为一谈。

## 2. pose replay 证据

| 数据 | expert reference | policy ID | policy OOD | finite pose files |
|---|---:|---:|---:|---:|
| StackCube Stage 2 | 100 OOD expert | 100 | 100 | 300 |
| PickSingleYCB object variation | 20 ID expert | 100 | 100 | 220 |
| OpenDrawer | ID 19；Handle 20；Grasp 17；Goal 20 | 100 | 每 split 100 | 476 |

所有生成的 pose arrays 均通过 finite-value 检查。StackCube Stage 2 的 ID policy 是用
OOD expert reference 做 exploratory comparison，因为该 timing root 没有同样完整的
ID expert action/state reference；因此它不应被写成正式 ID calibration 结果。

## 3. ERD threshold/timing sweep

`alarm step` 是持续越阈区间的第一个点；实际控制器确认时间约为 `alarm step + 5`。

| 数据组 | q=.925 threshold | q=.925 alarm | q=.95 threshold | q=.95 alarm |
|---|---:|---:|---:|---:|
| StackCube Stage 2 ID（exploratory） | 20.806 | 83/100，median 25 | 43.989 | 61/100，median 30 |
| StackCube Stage 2 OOD | 20.806 | 61/100，median 20 | 43.989 | 43/100，median 40 |
| YCB object ID | 12.565 | 90/100，median 5 | 17.275 | 83/100，median 25 |
| YCB object OOD | 12.565 | 100/100，median 15 | 17.275 | 100/100，median 25 |
| OpenDrawer ID | 23.530 | 100/100，median 50 | 29.042 | 100/100，median 50 |
| OpenDrawer Handle-OOD | 31.608 | 100/100，median 35 | 48.692 | 100/100，median 40 |
| OpenDrawer Grasp-OOD | 26.591 | 98/100，median 55 | 76.911 | 93/100，median 70 |
| OpenDrawer Goal-OOD | 22.720 | 100/100，median 50 | 26.662 | 100/100，median 50 |

### 解释

- StackCube Stage 2 OOD 上，q=.925 比 q=.95 更敏感：median 从 40 提前到 20，覆盖率从
  43/100 增加到 61/100；这与“更早发现 target-position 偏离”的直觉一致，但 ID 对照
  仍缺同分布 expert reference。
- YCB object variation 上，q=.925 对 OOD 全部报警，median 15；q=.95 仍全部报警但
  median 25。ID 曲线在 step 5 出现早期 crossing，说明 object-only 任务的正常 expert
  residual/短时运动需要额外的 transient 处理，不能仅凭更早就称为最佳。
- OpenDrawer 的 q=.925/q=.95 差异依赖 OOD stage。Grasp-OOD 的 q=.95 threshold 突然
  增大到 76.9，主要因为只有 17 条 OOD expert suffix 且 horizon=400；这说明该 split 的
  reference contract 还不够稳定，不能支持一个统一 threshold 的强结论。

## 4. 已有 gate 报警时间旁证

这些是历史 detector 的 first-alert，不是 ERD-Pose：

| 实验 | 代表 gate | ID | OOD |
|---|---|---:|---:|
| X-VLA StackCube/Grab Plane Stage-C（50+50） | Input / Bridge / Action / Diff | StackCube median 75/45/50/30 | StackCube median 35/5/20/105 |
| X-VLA StackCube/Grab Plane Stage-C（50+50） | Input / Bridge / Action / Diff | Grab Plane median 95/120/80/65 | Grab Plane median 0/100/95/90 |
| StackPyramid passive PCA | q=.95 PCA | 0/100 | Stage 2: 2/100（180,530）；Stage 3: 0/100 |
| YCB object variation | Bridge PCA | median 60 | step 0 |
| OpenDrawer action-block-08 PCA | q=.95 PCA | median 290 | Handle 220 / Grasp 295 / Goal 275 |

StackCube Stage 2 的 400-episode timing study本身只有固定介入 anchors 和 downstream
utility，没有同一批 policy 的 learned gate score timeline；因此不能把它直接当作 alarm
分布。

## 5. 可视化视频

视频按五个片段播放：StackCube Stage 2、YCB object variation、OpenDrawer Handle-OOD、
OpenDrawer Grasp-OOD、OpenDrawer Goal-OOD。每段上方同时显示 expert reference、ID policy
和 OOD policy 的代表性视频；下方显示 expert residual、ID/OOD 的 ERD median/P25--P75
曲线、q=.925/q=.95 threshold 和 q=.95 的 onset/confirmation 信息。

[播放三任务 ERD 可视化视频](</Users/zhaozhixuan/.codex/visualizations/2026/08/25/01a037ef-1a71-7ad3-9594-a7dcfe06ad7b/erd-pose/xvla-cross-task-erd-evidence-v1.mp4>)

## 6. 当前判断

现在可以把 q=.925 作为一个跨任务的**候选 calibration quantile**：在三个任务上它通常
比 q=.95 更早，且在 StackCube Stage 2 OOD/YCB OOD 上覆盖更高。但当前结果不能把它称为
已经证明的“全任务最佳 threshold”，原因是：

1. StackCube Stage 2 缺同分布 ID expert reference；
2. YCB OOD 的 reference 仍是 ID expert object geometry；
3. OpenDrawer expert 是 takeover suffix，且 Grasp-OOD 的 expert 数量只有 17；
4. 当前没有把 ERD-triggered takeover 与固定 0/10/15/20/30 takeover 做 matched-budget
   downstream utility 对照。

因此最小论文方案是：主分析报告 q=.925，q=.95 作为敏感性分析；同时把 raw threshold、
onset/confirmation、reference 类型和缺失/右删失分开报告。若要写“最佳”，还需在独立
held-out expert reference 上冻结 q，并做一次控制介入 utility 验证。
