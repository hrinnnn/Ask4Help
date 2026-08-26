# X-VLA Fixed-Grid Partial Evaluation Diagnostic

日期：2026-08-26（北京时间）  
pipeline：`xvla_fixedgrid_taskpolicy_knee_v1`  
性质：工程可行性/初步正确率诊断，不是正式 utility 结果。

## 目的与冻结条件

在 Stage-B 的全部 27 个 matched-budget training 分支完成前，独立验证两个已完成的 knee checkpoint 能否被正式 evaluator 加载并产生可解释的成功率。诊断不改变正式 timing anchors、threshold、success predicate、training seed 或 100-episode evaluation denominator。

- StackCube checkpoint：`stage_b_training_v1/stackcube/step_20/seed_17001/train/ckpt-2500`
- Grab Plane checkpoint：`stage_b_training_v1/airplane/step_20/seed_17001/train/ckpt-2500`
- 每个 task：20 ID + 20 OOD
- 独立 seeds：StackCube `190000/190100`，Grab Plane `191000/191100`
- evaluator：与正式 Stage-B 相同；`max_episode_steps=150`、`flow_steps=10`、`execute_horizon=5`
- 资源：GPU4、CPU20-39；Stage-B 继续使用 GPU5、CPU0-19

## 结果

| Task / split | Endpoint | Episodes | Successes | Rate | Videos | Actions |
|---|---|---:|---:|---:|---:|---:|
| StackCube / ID | strict success | 20 | 13 | 0.65 | 20/20 | 20/20 |
| StackCube / OOD | strict success | 20 | 7 | 0.35 | 20/20 | 20/20 |
| Grab Plane / ID | ever grasped | 20 | 15 | 0.75 | 20/20 | 20/20 |
| Grab Plane / OOD | ever grasped | 20 | 6 | 0.30 | 20/20 | 20/20 |
| Grab Plane / ID | strict completion (auxiliary) | 20 | 4 | 0.20 | 20/20 | 20/20 |
| Grab Plane / OOD | strict completion (auxiliary) | 20 | 2 | 0.10 | 20/20 | 20/20 |

Raw report：`/data/zhaozhixuan/Ask4Help-airplane-5090/results/xvla_fixedgrid_taskpolicy_knee_v1/diagnostic_partial_evaluation_v1/partial_report.json`  
Completion marker：`.../diagnostic_partial_evaluation_v1/PARTIAL_EVAL_COMPLETE`

## 判断边界

四组 rollout 均成功完成，summary、逐 episode rows、video 和 action artifacts 均齐全；因此 evaluator、checkpoint reload、task success extraction 和 ID/OOD split 均具备可行性。两项任务的 OOD rate 均低于 ID rate，方向上是合理的初步 sanity check。

但每个 split 只有 20 条且只使用一个训练 seed，不能用于证明 knee 与最终 SR 的相关性，也不能替代正式的 100-ID/100-OOD Stage-B 或 Stage-C utility evaluation。评估进程在完整 artifacts 写出后出现 `free(): invalid pointer`，不影响本诊断的 evidence 完整性，作为工程清理诊断保留。
