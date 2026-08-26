# X-VLA Fixed-Grid Stage-C Passive Gate Audit

日期：2026-08-27（北京时间）  
pipeline：`xvla_fixedgrid_taskpolicy_knee_v1`  
性质：正式 passive gate-to-knee audit；下游 gate-selected utility 尚未完成。

## 固定协议与证据

- Airplane 缺失的 multilayer asset 已从冻结 checkpoint 和 ID metadata 成功构建：12,632 observations、37 layers，包含 `vlm_input_pool`；`ASSETS_COMPLETE` 已写入。
- 每个 task 使用 50 validation-ID + 50 held-out-OOD episodes。
- threshold 只由 validation-ID 的 q=.95 calibration 得到，之后冻结用于 OOD。
- knee tolerance 为 5 env steps；Failure-Recovery 是 fixed step 50 的比较基线，不是 detector。
- 原始根目录：`/data/zhaozhixuan/Ask4Help-airplane-5090/results/xvla_fixedgrid_taskpolicy_knee_v1/stage_c_gate_v1/`。

## 被动 rollout endpoint

| Task / split | Endpoint | Episodes | Successes |
|---|---|---:|---:|
| StackCube / validation ID | strict success | 50 | 43 |
| StackCube / held-out OOD | strict success | 50 | 0 |
| Grab Plane / validation ID | ever grasped | 50 | 47 |
| Grab Plane / held-out OOD | ever grasped | 50 | 24 |
| Grab Plane / held-out OOD | strict completion (auxiliary) | 50 | 0 |

## Gate-to-knee timing

### StackCube，knee set `{10,20}`

| Method | Alarms observed | Miss rate | KD mean | KHR (all episodes) |
|---|---:|---:|---:|---:|
| Input PCA | 50/50 | 0.00 | 27.50 | 0.20 |
| Bridge PCA | 50/50 | 0.00 | **5.20** | 0.76 |
| Action PCA | 50/50 | 0.00 | 9.50 | **0.82** |
| Diff-DAgger | 18/50 | 0.64 | 73.61 | 0.00 |
| Failure-Recovery (fixed 50) | 50/50 | 0.00 | 30.00 | 0.00 |

### Grab Plane，knee set `{20}`

| Method | Alarms observed | Miss rate | KD mean | KHR (all episodes) |
|---|---:|---:|---:|---:|
| Input PCA | 50/50 | 0.00 | 21.70 | 0.00 |
| Bridge PCA | 35/50 | 0.30 | 66.57 | 0.00 |
| Action PCA | 46/50 | 0.08 | 75.87 | 0.00 |
| Diff-DAgger | 44/50 | 0.12 | 68.86 | 0.00 |
| Failure-Recovery (fixed 50) | 50/50 | 0.00 | 30.00 | 0.00 |

## 解释边界

StackCube 的 Bridge/Action PCA 报警最接近 calibration knee；Diff-DAgger 的 OOD 报警 miss 较多且观测到的报警远离 knee。Grab Plane 中所有方法均没有一次报警落入 `t=20±5`，说明当前 detector timing 与该 task 的 calibration knee 不一致。

这些结果只描述 passive alarm timing，不代表 gate 产生的 expert data 已经训练出更好的 policy。Stage-C data controller 随后才会在相同 520/2820 action budget 下完成 10 个 method--task 数据选择、30 个训练和 60 个 ID/OOD utility evaluation。
