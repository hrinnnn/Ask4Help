# X-VLA Fixed-Grid Stage-C Diff-DAgger Exact-Budget Stop

日期：2026-08-27（北京时间）  
pipeline：`xvla_fixedgrid_taskpolicy_knee_v1`  
性质：协议/科学 stop，等待用户决策；不是 evaluator 工程崩溃。

## 证据

StackCube Diff-DAgger 使用冻结 validation-ID q=.95 threshold，在预注册的 OOD training pool `154000--154399` 上消费完 `400/400` seeds：

- `raw_total=400`；
- `accepted_total=2`；
- `accepted_expert_actions=52`；
- frozen exact budget=`520` actions；
- accepted episodes 均为完整 episode，没有 suffix slicing 或 padding。

collection summary：
`/data/zhaozhixuan/Ask4Help-airplane-5090/results/xvla_fixedgrid_taskpolicy_knee_v1/stage_c_gate_data_v1/collections/stackcube/diffdagger/summary.json`

selector traceback：
`/data/zhaozhixuan/Ask4Help-airplane-5090/results/xvla_fixedgrid_taskpolicy_knee_v1/stage_c_gate_data_v1/logs/select_stackcube_diffdagger.log`

核心错误：

```text
RuntimeError: no whole-episode subset reaches exact budget=520; eligible_episodes=2 eligible_actions=52
```

## 判定

这是 gate feasibility 在冻结协议下未建立：即使继续消费全部预注册 seeds，也没有足够完整 expert suffix 组成 520-action dataset。已经完成的 StackCube Input/Bridge/Action selections 不受影响，但 Diff-DAgger branch 不能进入训练。后续 6 个 task--method branches 也不能在不改变 completion contract 的情况下自动跳过该 branch。

禁止的自动修复包括降低 OOD threshold、从 episode 中切出 suffix、修改 success predicate、用其他 method 替代 Diff-DAgger，或把 52 actions 当作 520 actions。

## 需要用户选择

1. 保持冻结协议，把 Diff-DAgger 记录为“ineligible under frozen exact budget”，并修改最终 reconciliation/论文表格以明确缺失该 branch；
2. 批准新 manifest revision，扩大 Diff-DAgger 的预注册 pool，只重跑该 branch；
3. 批准新的 admission/threshold protocol，重新做 ID calibration 并只重跑该 branch。

在得到选择前，远端 marker `NEEDS_USER_DECISION_DIFFDAGGER_EXACT_BUDGET` 阻止 gate training/utility 继续，所有已有 artifacts 保留。
