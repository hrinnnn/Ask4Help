# Phase-aware `D_path` 跨任务验证

日期：2026-08-28（北京时间）  
性质：diagnostic-only；没有重新训练 policy，也没有修改任何 formal pipeline、
checkpoint、seed、success predicate、detector threshold 或 expert-action budget。

## 1. 验证问题

验证 position-only、phase-aware 的末端路径偏离是否能在不同任务上提供稳定的
critical deviation onset：

\[
D_{\mathrm{path}}(t)=
\left\|
\frac{p_t^{\pi}-p_{\hat j_t}^{E}}{s_p}
\right\|_2.
\]

匹配索引 \(\hat j_t\) 使用从同任务 expert-to-expert DTW 估计的双侧归一化 phase
band，再做 monotone local alignment。每个 task 单独估计 phase band 和 robust position
scale，不跨 task 复用 raw threshold。

当存在足够的同分布成功 policy trajectories 时，将其按 seed 排序后交替划分为
calibration/held-out 两组。阈值是 calibration 组每条轨迹的 two-point persistent
maximum 的 q=.95；`t_PD` 是两个连续 5-step decision points 越阈区间的第一个点。
没有成功 policy calibration 的任务只能使用 expert pointwise q=.95，并标为
OOD-only diagnostic。

## 2. 合法性清单

| 任务 | 数据契约 | 本次强度 |
|---|---|---|
| PickSingleYCB object variation | 20 ID expert；100 ID policy（55 success）；100 OOD policy（1 success）；完整 pose/action/video | 可验证 ID false crossing 与 OOD onset；OOD 仍使用 ID object expert geometry |
| StackCube Stage2 target-OOD | 100 同分布 OOD expert；100 OOD policy（52 success/48 failure）；完整 pose/action/video | 当前最完整的 same-distribution validation |
| Fixed-grid StackCube OOD | 20 OOD expert；50 OOD policy，全部失败；pose replay 已审计 | 只能验证 failure onset，不能验证成功 policy false crossing |
| Fixed-grid Grab Plane OOD | 20 OOD expert；50 OOD policy；24/50 `ever_grasped`、strict 0/50；horizon=50 pose 完整 | endpoint/metric 边界验证；`ever_grasped` 不是完整任务成功 |
| OpenDrawer ID/Handle/Grasp/Goal | policy trajectory 完整，但 expert 是 takeover suffix，且 base preliminary | 不允许把 suffix-to-full-rollout 对齐包装成 reset-to-task `t_PD` |
| StackPyramid | canonical ID base 未通过；policy pose 有、同分布逐步 expert pose reference 不完整 | 排除 |
| Vegetable-basket / UncoverSpherePlace | ID gate 未通过或 OOD 尚未解锁 | 排除 |

## 3. 统一结果

| Task | threshold source | `tau_path` | held-out success false crossing | failure crossing | failure `t_PD` median (P25--P75) |
|---|---|---:|---:|---:|---:|
| PickSingleYCB ID / object-OOD | successful ID persistent-max q=.95 | 11.49 | 1/27 (3.7%) | ID 45/45；OOD 99/99 | ID 5 (5--10)；OOD 5 (5--10) |
| StackCube Stage2 target-OOD | successful OOD persistent-max q=.95 | 25.95 | 2/26 (7.7%) | 39/48 (81.3%) | 40 (35--52.5) |
| Fixed-grid StackCube OOD | expert pointwise q=.95 | 7.77 | not available | 50/50 | 5 (0--5) |
| Fixed-grid Grab Plane OOD | `ever_grasped` persistent-max q=.95 | 12.48 | 1/12 (8.3%) | 0/26 | censored |

前 0--40 steps 的 per-episode median `D_path`：

| Task/group | median `D_path` |
|---|---:|
| PickSingleYCB ID success | 3.15 |
| PickSingleYCB ID failure | 52.36 |
| PickSingleYCB OOD failure | 83.80 |
| StackCube Stage2 OOD success | 3.89 |
| StackCube Stage2 OOD failure | 11.81 |
| Fixed-grid StackCube OOD failure | 23.26 |
| Fixed-grid Grab Plane `ever_grasped` | 6.39 |
| Fixed-grid Grab Plane not grasped | 3.67 |

## 4. 研究判断

### 4.1 支持的范围

`D_path/t_PD` 在具有明显空间路径分叉的任务上成立：

- PickSingleYCB object variation：成功 ID 路径回到 expert 分布附近，ID failure 与
  OOD failure 均在 step 5 左右持续越阈；held-out false crossing 为 3.7%。
- StackCube Stage2 target-OOD：成功 OOD policy 与 failure policy 的路径分布分离，
  81.3% failure 在 horizon 内越阈，median onset 为 step 40。

### 4.2 明确反例

position-only `D_path` 在 Grab Plane 上失败：26 条 `ever_grasped=false` trajectories
没有一条持续越阈，且它们前 0--40 steps 的位置路径偏离反而低于 `ever_grasped=true`
trajectories。进一步检查 orientation、gripper width、7D pose、13D pose+velocity 和
task-relative `tcp-object`/`goal-object` 后，failure crossing 仍为 0--1/26。

这不是简单的 threshold 问题。Grab Plane policy 可以沿近似 nominal 的 gripper path
运动，却因为抓取接触/闭合结果失败；成功 trajectory 反而因真正抬起并移动物体而产生
更大的 position-path change。因此 gripper `D_path` 不是所有 failure mode 的充分指标。

### 4.3 论文边界

不能写成“`D_path` 在所有合法 task 上给出最佳 takeover timing”。当前证据支持：

> `D_path` is an expert-calibrated timing reference for spatial trajectory
> divergence, not a universal failure or contact-success oracle.

正文应报告 raw `D_path(t)`、persistent onset `t_PD`、success false crossing、failure
coverage 和 censored rate。最终 learning utility 仍由 matched-budget OOD SR 验证。

对于 grasp/contact-dominated shifts，需要额外的 task-relative contact/progress signal，
或把 `D_path` 与可观测 grasp state 分开报告；不能通过放宽/收紧 position threshold
修复 Grab Plane 反例。

## 5. 最小补充实验

1. 为 fixed-grid StackCube 增加同一 OOD distribution 下的成功 policy/reference
   validation trajectories，才能注册 trajectory-level false crossing。
2. Grab Plane 保留为反例；若论文希望覆盖该 failure mode，需预注册一个可部署的
   grasp/contact progress channel，再用 held-out ID 校准，不能在 OOD 上调参。
3. OpenDrawer 需要 reset-to-terminal expert references；现有 takeover suffix 只可做
   suffix-quality diagnostic。
4. StackPyramid、vegetable-basket 与 UncoverSpherePlace 在 ID base/reference gate
   通过前继续排除。
