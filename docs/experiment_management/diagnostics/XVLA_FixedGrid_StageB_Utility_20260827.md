# X-VLA Fixed-Grid Stage-B Utility Result

日期：2026-08-27（北京时间）  
pipeline：`xvla_fixedgrid_taskpolicy_knee_v1`  
性质：正式 Stage-B matched-budget utility evaluation；不是部分诊断。

## 审计范围

- StackCube：5 anchors × 3 training seeds × 2 splits = 30 summaries。
- Grab Plane：4 anchors × 3 training seeds × 2 splits = 24 summaries。
- 每个 summary：100 episodes、100 episode rows；独立读取核对结果为 StackCube `30/30`、Grab Plane `24/24`，无 denominator mismatch。
- Stage-B utility 使用 3 个训练 seed 的 mean/std；主 OOD endpoint 为 StackCube strict success、Grab Plane ever grasped。Grab Plane strict completion 保留为辅助列。
- 原始 summary root：`/data/zhaozhixuan/Ask4Help-airplane-5090/results/xvla_fixedgrid_taskpolicy_knee_v1/stage_b_evaluation_v1/`。

## 结果

### StackCube

| Anchor | ID success mean±std | OOD success mean±std | OOD strict | OOD per seed |
|---:|---:|---:|---:|---|
| 0 | 0.9067±0.0404 | **0.7233±0.0416** | 0.7233 | 0.71, 0.77, 0.69 |
| 10 | 0.8600±0.0346 | 0.6000±0.0794 | 0.6000 | 0.51, 0.66, 0.63 |
| 20 | 0.7867±0.0764 | 0.4467±0.1026 | 0.4467 | 0.42, 0.36, 0.56 |
| 30 | 0.8767±0.0351 | 0.6400±0.0608 | 0.6400 | 0.68, 0.57, 0.67 |
| 45 | 0.8933±0.0321 | 0.6500±0.0656 | 0.6500 | 0.66, 0.71, 0.58 |

- utility-best anchor set：`{0}`。
- calibration knee set：`{10,20}`。
- 95%-of-best utility set：`{0}`。
- knee/utility overlap：empty。

### Grab Plane

| Anchor | ID ever-grasped mean±std | OOD ever-grasped mean±std | OOD strict mean | OOD per seed |
|---:|---:|---:|---:|---|
| 0 | 0.8133±0.0757 | **0.7967±0.0569** | 0.0467 | 0.75, 0.78, 0.86 |
| 10 | 0.8067±0.0702 | 0.6133±0.1097 | 0.1467 | 0.55, 0.74, 0.55 |
| 20 | 0.8167±0.1159 | 0.1800±0.2138 | 0.0233 | 0.42, 0.11, 0.01 |
| 30 | 0.8333±0.0551 | 0.5567±0.3600 | 0.0067 | 0.55, 0.20, 0.92 |

- utility-best anchor set：`{0}`。
- calibration knee set：`{20}`。
- 95%-of-best utility set：`{0}`。
- knee/utility overlap：empty。

## 科学解释边界

在当前冻结的 matched-budget、2500-step、3-seed 协议下，两个 task 的 downstream OOD utility 都由 `t=0` anchor 取得最高均值，而不是由 Stage-A 的 time–deviation knee 取得。因此，当前 Stage-B 不支持“由 calibration knee 直接预测 downstream SR 最优 timing”这一假设。

这不是 pipeline 失败：所有 Stage-B 分支、checkpoint、100-ID/100-OOD summaries 和 utility summaries 均完整。它是一个有明确边界的负结果，可能意味着在当前任务和预算下，较早收集的 expert supervision 比 knee 附近的恢复数据更有利于学习。

不能从这张表直接推出 task-universal 结论，也不能把它解释成 gate method 的结果。Stage-C 仍需完成 passive gate-to-knee audit 和 gate-selected matched-budget utility，才能判断 gate timing closeness 是否与 method-level utility 有关。
