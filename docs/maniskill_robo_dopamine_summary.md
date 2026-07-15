# RLinf ManiSkill Robo-Dopamine 接入总结

## 目标

在 RLinf 的 `RLT + pi0.5 + ManiSkill` 管线中接入 Robo-Dopamine，为
`PegInsertionSideWideClearance-v1` 提供 chunk-level dense reward。

最终奖励为：

```text
r_total = r_sparse + 0.1 * (gamma^L * Phi_next - Phi_prev)
```

其中 `L` 是当前 action chunk 实际执行的 ManiSkill step 数。

## 已完成

- 接入 Robo-Dopamine 的 `incremental`、`forward`、`backward` 三种 progress mode。
- 三种 mode 均有效时使用 consistency-aware fusion；部分有效时使用 valid mean。
- parser 全部无效时 reward 为 0，且不更新 `prev_phi` 和 previous observation。
- 每个并行环境独立维护 start reference、previous observation 和 `prev_phi`。
- ManiSkill observation 增加 task ID 和 episode reset reference 图像。
- 保留完整 `[B, action_chunk]` done 信息，并根据首次 done 计算实际 chunk 长度。
- PBRS shaping 只写入 `[B, 10]` 的第 0 列，避免被重复广播或额外折扣。
- success 的诊断 `Phi_next` 强制为 1；terminal shaping potential 使用 0，随后清空该环境状态。
- 扩展官方 ManiSkill expert collector，从成功 replay 的 terminal observation 生成 goal bank。
- 增加 GRM Stage 2 配置、fake endpoint、运行脚本和 JSONL diagnostics。

## 验证结果

- 本地核心 GRM 单元测试：`20 passed`。
- 服务器完整目标测试：`26 passed`。
- 成功创建 ManiSkill RGB 环境，并验证主视角、腕部视角和 reset reference。
- 使用官方 motion-planning solver 成功生成一条 Peg expert replay 及 goal bank。
- fake GRM 对真实 ManiSkill observation 连续完成 3 个 action chunk：

```text
shaping rewards: 0.3024, -0.0357, -0.1662
```

三次调用均成功解析三个 mode，并使用 consistency-aware fusion。Shaping reward
只出现在 chunk 第 0 列。

## 持久化位置

```text
Goal bank:
/mnt/data/ask4help/assets/grm_goal_bank/maniskill_peg

Fake GRM smoke metrics:
/mnt/data/ask4help/results/maniskill_dopamine_grm_contract_smoke/grm_metrics.jsonl
```

## 当前未完成项

- 尚未运行完整 RLT Stage 2 rollout/train epoch，因为服务器没有现成的 ManiSkill
  Stage 1 actor checkpoint。
- 尚未运行真实 `Robo-Dopamine-GRM-2.0-4B-Preview` smoke，因为模型 endpoint
  当前没有启动。

上述两项属于外部运行条件，不需要继续修改当前接入代码。

## Git 分支

```text
Ask4Help: codex/maniskill-dopamine-grm
RLinf:    codex/maniskill-dopamine-grm
```

下一步是在 Stage 1 actor checkpoint 就绪后运行 fake endpoint 的完整 Stage 2
epoch，确认 reward worker、trajectory、replay buffer 和 actor-critic update 闭环；随后替换为真实
GRM endpoint，完成最终 smoke。
