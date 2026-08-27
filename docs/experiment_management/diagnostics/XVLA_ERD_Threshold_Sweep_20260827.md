# ERD-Pose threshold sweep diagnostic

日期：2026-08-27（北京时间）  
性质：diagnostic-only；不修改 formal X-VLA Stage-C 的 threshold、seed、预算、
success predicate 或 completion contract。

## 1. 这次实际比较的是什么

这里把两个概念分开：

1. `D` threshold 是 expert residual 的分位数，决定多大的 pose deviation 才算异常；
2. `alarm step` 是 fail trajectory 第一次连续两个 decision point 超过该 threshold 的时间。

因此 Grab Plane 的 `6.412` 小于 StackCube 的 `9.826`，并不意味着两个 task 的
绝对距离可以直接比较。真正可比较的是每个 task 内部改变 threshold 后，alarm 的覆盖率、
时刻和是否晚于不可逆事件。

所有候选都使用已经生成的 50 条 OOD pose timelines，decision stride=5，
persistence=2，horizon=50；threshold 只从各 task 的 Stage-A expert residual
calibration values 计算。没有用 downstream Success Rate 反向调参。

## 2. StackCube

支持的 OOD context 为 48/50。`q=.95` 的阈值为 9.826。

| expert quantile | D threshold | observed | supported | median alarm | pre/post | late alarms |
|---:|---:|---:|---:|---:|---:|---:|
| .800 | 4.622 | 50/50 | 48/48 | 5 | 48/2 | 2 |
| .850 | 5.407 | 50/50 | 48/48 | 5 | 48/2 | 2 |
| .900 | 6.108 | 50/50 | 48/48 | 5 | 48/2 | 2 |
| .925 | 6.583 | 50/50 | 48/48 | 5 | 48/2 | 2 |
| **.950** | **9.826** | **48/50** | **46/48** | **20** | **46/2** | **2** |
| .975 | 13.736 | 46/50 | 44/48 | 20 | 44/2 | 2 |
| .990 | 43.635 | 36/50 | 35/48 | 20 | 34/2 | 2 |

`q<=.925` 会把一个早期 transient spike 当成持续偏离，导致中位报警突然变成
step 5；它虽然更早，但不符合“允许的 expert pointwise exceedance 不超过 5%”
的校准约束。`q=.975` 没有把报警中位数再推迟，却降低了 supported coverage
（46/48 到 44/48）。所以在当前数据上，`q=.95` 是比经验 step-20 更有依据的
operating point；它不是为了拟合 step-20 而改出来的。

## 3. Grab Plane

支持的 OOD context 为 50/50。`q=.95` 的阈值为 6.412。

| expert quantile | D threshold | observed | supported | median alarm | pre/post | late alarms |
|---:|---:|---:|---:|---:|---:|---:|
| .800 | 5.276 | 50/50 | 50/50 | 10 | 50/0 | 0 |
| .850 | 5.524 | 50/50 | 50/50 | 10 | 50/0 | 0 |
| .900 | 6.091 | 50/50 | 50/50 | 10 | 50/0 | 0 |
| .925 | 6.224 | 50/50 | 50/50 | 10 | 50/0 | 0 |
| **.950** | **6.412** | **50/50** | **50/50** | **15** | **50/0** | **0** |
| .975 | 7.127 | 50/50 | 50/50 | 20 | 50/0 | 0 |
| .990 | 7.499 | 50/50 | 50/50 | 20 | 50/0 | 0 |

Grab Plane 的 q=.975/q=.990 并没有提高覆盖率，也没有减少晚报，因为 q=.95
已经是 50/50、全部 pre-grasp、且没有 late alarm。它们只是把报警整体推迟到
step 20。若目标是“尽早发现可靠的偏离”，`q=.95` 比更高 threshold 更合适；若
部署中明确更偏好保守、减少 expert intervention，则 q=.975 可以作为一个单独的
安全 operating point，而不应称为由当前数据证明的最优阈值。

## 4. 当前建议

在没有新的 held-out validation set 之前，不建议为了追上经验 timing 而直接改掉
q=.95。当前最可辩护的规则是：

- expert calibration 的 pointwise exceedance allowance 设为 5%（q=.95）；
- persistence 保持两个 decision points；
- StackCube 记录 `q=.95 -> step 20`；
- Grab Plane 记录 `q=.95 -> step 15`；
- 将 q=.975 作为“更保守的敏感性分析”，而不是替代主结果。

这个选择满足一个简单的约束式判据：在当前 reference 下保持不超过 5% 的 expert
pointwise exceedance，同时优先选择仍能覆盖绝大多数 supported fail episodes、且不
产生额外 late alarm 的最低可接受 threshold。它仍然只是 diagnostic operating
point，不等价于 Success Rate 最优点。

## 5. 视频中展示的内容

合成视频按 StackCube、Grab Plane 两段播放。每段上方同时显示 nearest-context
expert reference 和同 context 的 OOD fail；下方显示 50 条 fail 的 P25--P75
score band、median、选中的 fail score，以及 q=.925/q=.95/q=.975 三条 threshold
线。右侧 histogram 显示 expert calibration residual 与当前时刻的 fail-score
snapshot。

视频使用的配对为：

- StackCube：expert seed `150000`，fail seed `151000`，ERD crossing `step 20`；
- Grab Plane：expert seed `160016`，fail seed `161000`，ERD crossing `step 15`。

它们是可视化示例，不是把一条视频误当成 50 条轨迹的统计结论。

## 6. 仍然需要的验证

若论文要把 threshold 称为“optimal”，还需要把 expert calibration、threshold
selection 和 downstream utility 分开：在独立 held-out expert/reference 上冻结
threshold，再做固定 takeover steps（例如 0/10/15/20/30）和 ERD-triggered
takeover 的 matched-budget evaluation。当前 Stage-B 中固定 timing 的 utility
仍然不能被这次 ERD sweep 替代。
