# DiffDAgger 接入 RLinf RLT + pi0.5 + ManiSkill 调研

更新时间：2026-07-15

## 1. 结论先行

可以接，而且 RLinf 已经具备大部分底层接口，不需要把 DiffDAgger 的训练框架整体搬进来。

RLinf 当前已经有：

- ManiSkill 并行环境与 action-chunk rollout；
- student/expert 双模型推理；
- `intervene_flags` 和 expert action 覆盖；
- intervention trajectory 提取；
- replay buffer、demo buffer 和在线 LeRobot 数据集；
- OpenPI 的 DAgger SFT worker；
- RLT actor-critic 对 expert intervention action 做 BC、对所有 transition 做 Q-learning 的能力。

真正缺少的是：

1. 用 pi0.5 flow-matching loss 计算不确定性的 `DiffDAggerGate`；
2. 根据训练数据 loss 分布标定 CDF 和 `alpha` 分位阈值；
3. 连续 `K` 次超阈值后，在当前 chunk 执行前立即切换 expert；
4. 每个并行 env 独立维护 violation history、expert latch 和 episode reset；
5. 如果要完全复现原论文，还需要在累计 `N_d` 次 intervention 后更新 pi0.5 student 并重新标定阈值。

推荐分两层推进：

- **第一层：Flow-DiffDAgger faithful smoke。** 暂时不同时训练 RLT，只验证 pi0.5 student、flow-loss gate、expert takeover、intervention SFT、重新标定的闭环。
- **第二层：DiffDAgger-gated RLT。** gate 负责 Ask for Help；RLT actor-critic 同时从 sparse/dense reward 和 expert intervention 学习。这是我们的目标系统，但应明确称为 DiffDAgger adaptation，而不是原论文逐项等价复现。

## 2. DiffDAgger 原论文和代码究竟做了什么

论文将 diffusion policy 自己的训练损失用作 action-conditioned uncertainty：

```text
U(o, a) = E_(epsilon,t)[L_policy(o, a, epsilon, t)]
```

在训练数据上计算该 loss 的经验分布，并取 `alpha` 分位数作为阈值。部署时，policy 先生成候选 action sequence，再计算该候选 action 在当前 observation 下的 expected diffusion loss。连续 `K` 个状态超过阈值才请求 expert，以减少单次噪声造成的误触发。

官方代码的实际循环位于 [`sim_train.py`](https://github.com/sean1295/DiffDAgger/blob/main/diffdagger/sim_train.py)：

1. 先收集 `N_i` 条 expert demonstrations；
2. 用累计 expert 数据训练 diffusion policy；
3. 在训练数据 action 上重新计算 loss CDF 和阈值；
4. student rollout，每次生成 action chunk 后计算 diffusion loss；
5. 连续 `K` 次超阈值或 rollout timeout 时，切换 expert；
6. expert 持续控制到任务成功，而不是只纠正一个 action；
7. 累计新的 expert 数据后重新训练 policy 并重新标定阈值。

核心实现见 [`diffusion_policy.py`](https://github.com/sean1295/DiffDAgger/blob/main/diffdagger/agents/diffusion_policy.py)：

- `get_action(..., dagger=True)` 生成 action 后计算 uncertainty；
- `get_avg_diffusion_loss_ndata` 对 timestep/noise 求平均；
- `get_stats_from_dataset` 在训练数据上建立经验 CDF；
- `diffusion_loss > threshold` 连续达到 `patience` 次后 query。

需要特别记录一个“论文描述与 release code 的差别”：论文算法概念上在 expert 控制时聚合 `(observation, expert action)`；发布的 PushT 实现只提交最终成功、expert 段长度不少于 10 step 的 intervention episode，失败的 expert episode、纯自动成功 episode和过短 intervention 都不加入训练集。

论文仿真使用 ManiSkill 的 stacking、pushing、plugging。Plugging 的公开实验参数是 `N_i=20`、`N_f=100`、`N_d=8`、`alpha=0.99`、`K=2`、batch size 512、prediction horizon 64、action horizon 8、64 diffusion steps。论文与代码来源：[Diff-DAgger paper](https://arxiv.org/abs/2410.14868)、[official repository](https://github.com/sean1295/DiffDAgger)。

## 3. RLinf 当前管线

### 3.1 总体调度

入口 [`train_embodied_agent.py`](../RLinf/examples/embodiment/train_embodied_agent.py) 根据 `algorithm.loss_type` 只创建一种 actor worker：

```text
rlt_ac             -> RLTACFSDPPolicy
embodied_dagger    -> EmbodiedDAGGERFSDPPolicy
```

随后创建 rollout、env 和可选 reward worker。每个训练 step 的主循环位于 [`embodied_runner.py`](../RLinf/rlinf/runners/embodied_runner.py)：

```text
sync actor weights to rollout
  -> EnvWorker 与 RolloutWorker 交互收集 rollout
  -> optional RewardWorker 计算奖励
  -> ActorWorker 接收 trajectory/replay data
  -> ActorWorker 更新模型
  -> eval / checkpoint
```

因此，当前框架不能通过一个 YAML 同时选择 `rlt_ac` 和 `embodied_dagger` 两个 actor worker。这是完整联合训练的主要结构性缺口。

### 3.2 RLinf 已有的通用 DAgger

RLinf 官方已支持 embodied DAgger。官方流程是按 `beta` 概率选择 expert 执行动作，并在 classic DAgger 模式下对 student 访问的状态额外调用 expert relabel；随后把 expert label 放入 replay buffer，对 student 做 SFT。[RLinf DAgger documentation](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/dagger.html)

相关接口如下：

- [`huggingface_worker.py`](../RLinf/rlinf/workers/rollout/hf/huggingface_worker.py)：加载 student/expert、随机 `beta` 路由和 expert relabel；
- [`fsdp_dagger_policy_worker.py`](../RLinf/rlinf/workers/actor/fsdp_dagger_policy_worker.py)：接收 intervention trajectories 或在线 LeRobot episodes，并做 SFT；
- [`dataset.py`](../RLinf/rlinf/data/datasets/dagger/dataset.py)：rolling in-memory LeRobot dataset，可只采样全部为 intervention 的 action windows；
- [`openpi_action_model.py`](../RLinf/rlinf/models/embodiment/openpi/openpi_action_model.py)：`prepare_dagger_sft_batch`、`prepare_lerobot_sft_batch` 和 OpenPI SFT loss；
- [`embodied_io_struct.py`](../RLinf/rlinf/data/embodied_io_struct.py)：`intervene_flags`、expert action 覆盖和 intervention trajectory 提取。

RLinf 官方列出的组合包括 ManiSkill + MLP、LIBERO + pi0、RoboTwin + pi0；当前没有现成的“ManiSkill + pi0.5 + DiffDAgger uncertainty”配置。

### 3.3 当前 ManiSkill RLT 管线

当前分支的 RLT rollout 位于：

- [`rollout.py`](../RLinf/rlinf/algorithms/rlt/rollout.py)
- [`route.py`](../RLinf/rlinf/algorithms/rlt/route.py)
- [`expert.py`](../RLinf/rlinf/algorithms/rlt/expert.py)

数据流是：

```text
pi0.5 feature model
  -> z_rl + proprio + pi0.5 reference action chunk
  -> small RLT MLP actor proposes action chunk
  -> route chooses base pi0.5 / RLT actor / expert pi0.5
  -> EnvWorker executes selected chunk
  -> RLT replay buffer stores macro transition
  -> actor-critic performs Q update + BC update
```

`SimulatorRLTRoute` 已经支持 sticky expert takeover 和逐 chunk 的 `intervene_flags`。当前 takeover 请求由 [`maniskill_rlt_env.py`](../RLinf/rlinf/envs/maniskill/maniskill_rlt_env.py) 的 privileged simulator progress heuristic 生成，例如 peg 是否接近孔、连续几 chunk 没有 x/yz progress。这不是 uncertainty，也依赖 task-specific simulator state。

RLT actor 的 BC 逻辑已经适合 Ask for Help：[`fsdp_rlt_ac_policy_worker.py`](../RLinf/rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py) 在 `intervene_flags=True` 时，以实际 expert action 为 BC target；普通 transition 则以 pi0.5 reference chunk 为 BC target。若启用 `demo_buffer`，每个 batch 默认一半 online replay、一半 expert/intervention demo。

## 4. pi0.5 上如何定义 DiffDAgger uncertainty

原 DiffDAgger 是离散 diffusion timestep；pi0.5 是 continuous flow matching，不能复用原仓库的 DDPM scheduler 和绝对 threshold。

OpenPI 的正式训练公式是：

```text
noise ~ N(0, I)
t ~ 0.001 + 0.999 * Beta(1.5, 1.0)
x_t = t * noise + (1 - t) * action
u_t = noise - action
loss = MSE(v_theta(observation, x_t, t), u_t)
```

RLinf 的 [`openpi_action_model.py`](../RLinf/rlinf/models/embodiment/openpi/openpi_action_model.py) 直接继承上游 `PI0Pytorch`，现有 SFT 也使用这个逐 action、逐维度 MSE。因此对应的 Flow-DiffDAgger score 应定义为：

```text
U_flow(o, a) = mean_(m=1..M)[
  mean_valid_action_dim MSE(v_theta(o, x_t_m, t_m), noise_m - a)
]
```

实现时必须满足：

- `a` 先经过与 pi0.5 SFT 完全相同的 action normalization；
- observation 使用相同图像、state、prompt transform；
- 只聚合有效的 `action_chunk × action_env_dim`，不让 padding 维度影响分数；
- calibration 和 online scoring 使用相同的 `M`、time/noise sampling、dtype 和 action mask；
- threshold 必须在当前 pi0.5 checkpoint 和当前 task 数据上重新标定，不能照抄论文数值。

建议用固定的 common-random-number bank 降低 gate 抖动：按 OpenPI 的 Beta time distribution 做分层采样，并固定一组 Gaussian noise。第一版可用 `M=16` 做工程 smoke，再用 `M=16/32/64` 的 score rank correlation 和 query F1 选择正式值。原论文的小 diffusion policy 可以一次批量评估大量 timestep/noise；pi0.5 更大，直接照搬其计算量会明显拖慢 rollout。

## 5. 推荐的接入位置

不要把 gate 放进 ManiSkill env，也不要把 DiffDAgger 的 forked ManiSkill 覆盖到 RLinf。最合适的位置是 `SimulatorRLTRoute.route()`：student candidate 已生成、expert action 尚未调用、当前 chunk 尚未执行。

推荐新增：

```text
rlinf/algorithms/diffdagger/
  flow_uncertainty.py     # pi0.5 flow loss scorer
  calibration.py          # empirical CDF / quantile artifact
  gate.py                 # per-env K-consecutive and expert latch
  diagnostics.py          # metrics / JSONL records
```

候选动作定义为“如果 expert 不介入，本 chunk 真正会执行的动作”：

```text
candidate = RLT actor action, if actor_switch
candidate = pi0.5 reference action, otherwise
```

然后：

```text
score candidate with frozen/current pi0.5
  -> cdf(score) > alpha for K consecutive chunks
  -> query expert before executing current chunk
  -> expert action replaces candidate
  -> mark entire executed expert chunk as intervene
```

这个选择最符合 Ask for Help 的实际问题，因为 gate 判断的是“当前系统原本要执行的动作是否在 pi0.5 所学支持集内”。但它和原论文存在一项明确偏差：RLT actor 是 MLP，uncertainty 来自 pi0.5，而不是来自 RLT actor 自己的训练 loss。因此它应称为 **pi0.5 flow-loss-gated RLT**。

## 6. 并行环境和接管状态

每个 env 至少维护：

```text
violation_history[K]
consecutive_violations
expert_active
intervention_id
last_score
last_cdf
```

不能只按 rollout batch 下标保存状态，因为不同 EnvWorker 的 batch 可能合并。应由 EnvWorker 传稳定的 `(worker_rank, local_env_id, episode_id)`；gate 以该 key 保存状态。episode_id 改变时立即清空 history 和 expert latch。

对于第一版 Peg task，采用论文式 sticky takeover：一旦 query，expert 一直控制到 success/failure/timeout，episode 结束才释放。不要只接管一个 chunk，否则会把“请求帮助”变成短暂 action replacement，且 expert correction 未必能把系统带回可恢复状态。

当前 RLT 环境把上一 chunk 计算出的 `intervene_flag` 送回 rollout，因此存在一 chunk 延迟。DiffDAgger gate 应在 route 内对当前 candidate 立即判断并覆盖当前 chunk，不能复用这个延迟路径。

## 7. 数据如何进入训练

### 7.1 DiffDAgger faithful 路线

使用 `loss_type: embodied_dagger`：

- pi0.5 是 student；
- expert pi0.5 是 teacher；
- uncertainty gate 替换 beta-random gate；
- intervention action 写入现有 LeRobot/replay dataset；
- `EmbodiedDAGGERFSDPPolicy` 更新 pi0.5；
- 每累计 `N_d` 条有效 intervention episode，重新计算 training-data CDF。

这最接近原论文，但没有 RLT actor-critic。

### 7.2 RLT 联合路线

使用 `loss_type: rlt_ac`：

- pi0.5 feature/reference model 暂时冻结；
- DiffDAgger gate 触发 expert；
- 所有 transition 进入 RL replay，供 critic 学 reward；
- successful intervention transition 额外进入 `demo_buffer`；
- RLT actor 在 intervention step 直接 BC 到 expert action，其余 step 保持对 pi0.5 reference 的 BC；
- actor 同时优化 `-Q + BC`。

需要修正一项当前数据语义：`record_transition` 至少应为 `actor_switch OR expert_takeover`，否则 critical phase 外的 expert transition 会被丢弃。

为了匹配 release code，建议 demo buffer 默认只接收最终成功的 intervention episode；失败 episode仍可保留在 RL replay 中学习失败回报，但不作为 expert BC 数据。该过滤必须在 episode 完成后进行，不能在每个 chunk 到达时立刻提交 demo buffer。

## 8. 为什么第一版不建议同时在线更新 pi0.5 和 RLT

当前 RLinf 每次只创建一个 actor worker，且 RLT rollout 中：

- `hf_model` 是可同步更新的小 RLT actor；
- `rlt_feature_model` 是 frozen pi0.5；
- `expert_model` 是 frozen expert pi0.5。

若要在同一次 run 中同时更新 RLT actor-critic 和 pi0.5 DAgger student，需要第二个 trainable actor group、第二套 optimizer/checkpoint/weight sync，以及 rollout 对 `rlt_feature_model` 的热更新。这不是简单加一个 loss，而是 runner 级双 actor 训练。

更稳妥的顺序是：

1. 先完成 frozen-pi0.5 的 DiffDAgger-gated RLT；
2. 验证 uncertainty 能预测失败、expert intervention 能降低失败并改善 sample efficiency；
3. 再做交替更新：每累计 `N_d` 个成功 intervention episode，暂停 rollout，调用现有 DAgger SFT worker 更新 pi0.5，重新标定 CDF，再恢复 RLT；
4. 只有交替版本吞吐成为瓶颈时，再设计双 actor 并行 runner。

这种交替更新也更接近 DiffDAgger 原代码的“收集一批 -> 重新训练 -> 重新标定 -> 继续 rollout”，而不是强行异步同时更新。

## 9. 两阶段实施方案

### Stage A：Flow-DiffDAgger faithful smoke

目标：证明 RLinf 中的 pi0.5 flow loss 能产生有效 query，并跑通 DAgger 数据闭环。

1. 为 OpenPI 增加无梯度 `score_action_flow_loss` API；
2. 实现 calibration artifact，离线用 20 条成功 expert demos 生成 CDF；
3. 将 rollout 的 beta-random gate 替换为 score/quantile/K gate；
4. 单 env 验证 query 后 expert 在当前 chunk 接管并 latch 到 done；
5. 多 env 验证 state 不串线、reset 正确；
6. 累计 intervention 后运行一次现有 OpenPI DAgger SFT；
7. 更新 checkpoint 后重新 calibration；
8. 比较 SFT 前后同一固定 validation set 的 score、query F1 和 autonomous success。

Stage A 验收不是只看“程序没报错”，而是：

- in-distribution expert action 的 score 显著低于破坏/扰动 action；
- threshold 对 held-out failure 的 F1 高于随机和 state-progress heuristic；
- query 发生在 failure 前，而不是只在 episode 已经不可恢复时出现；
- intervention 数据确实进入 SFT，更新后 autonomous success 不下降。

### Stage B：DiffDAgger-gated RLT 闭环

目标：Ask for Help + expert correction + BC/RL 更新在同一 RLT run 中闭环。

1. 把 gate 注入 `SimulatorRLTRoute`，对最终 candidate action 打分；
2. 传入稳定 env/episode identity，维护 per-env K/latch；
3. expert action 当前 chunk 即时覆盖并写 `intervene_flags`；
4. 启用 RLT `demo_buffer`，只提交 successful intervention episode；
5. 修正 `record_transition`，保留 expert takeover transition；
6. 保持 eval 完全无 expert，只测 autonomous policy；
7. 与当前 privileged `stalled_progress` gate 做相同 expert budget 对照；
8. 最后再与 Robo-Dopamine dense reward 组合，形成 uncertainty query + dense reward + RLT actor-critic。

## 10. 测试清单

### 单元测试

- flow score 与直接调用 OpenPI per-element SFT loss 的手工平均一致；
- action normalization、padding mask、chunk/action dim 正确；
- empirical CDF、quantile、overflow 和空 calibration fail-fast；
- `K=2` 时一次超阈值不 query，两次连续才 query；
- 中间低于阈值时 consecutive count 重置；
- query 后 sticky takeover；
- done/reset 后 history 和 latch 清空；
- 多 env 合并、拆分和 reset 不串状态；
- expert action 覆盖发生在当前 chunk；
- intervention flag 与实际执行 action 完全对齐；
- failed intervention 不进入 BC demo buffer。

### 集成测试

- fake scorer：固定指定 env 在第 2 个 chunk query，验证 routing/replay；
- real pi0.5 scorer：3 个 in-distribution 和 3 个扰动 chunk 均输出 finite score；
- shortened Stage A：收集、SFT、重新 calibration 各运行一次；
- shortened Stage B：至少一次 autonomous chunk、一次 expert takeover、一次 RLT actor/critic update；
- checkpoint/resume 后 calibration version、gate state 和 replay data 一致。

## 11. 正式实验应比较什么

DiffDAgger 的核心不是“带 expert 时成功率更高”，而是在相同 expert supervision budget 下更高效。因此至少报告：

- autonomous success rate，eval 时禁止 expert；
- success vs expert-labeled steps；
- success vs intervention episodes；
- success vs environment steps；
- query precision/recall/F1：query 后若无 expert 是否会失败；
- intervention rate、平均首次 query 时间、false-positive rate；
- wall-clock 与 flow scorer latency；
- 最终 OOD split success。

推荐对照：

```text
SFT only
RLT sparse reward only
RLT + random/beta expert query
RLT + privileged stalled-progress query
RLT + Flow-DiffDAgger query
RLT + Flow-DiffDAgger + Robo-Dopamine reward
```

所有 Ask for Help 方法必须固定 expert-labeled step budget或 intervention episode budget；否则“更多请求 expert”会天然获得更高成功率，无法证明 query 方法有效。

## 12. 预计改动文件

第一版主要改动：

```text
RLinf/rlinf/models/embodiment/openpi/openpi_action_model.py
RLinf/rlinf/algorithms/diffdagger/{flow_uncertainty,calibration,gate}.py
RLinf/rlinf/algorithms/rlt/route.py
RLinf/rlinf/algorithms/rlt/rollout.py
RLinf/rlinf/data/embodied_io_struct.py
RLinf/rlinf/workers/rollout/hf/huggingface_worker.py
RLinf/rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py
RLinf/examples/embodiment/config/maniskill_rlt_stage2_ac_mlp_diffdagger.yaml
RLinf/tests/algorithms/diffdagger/
```

另需一个 calibration CLI，把 expert LeRobot dataset、pi0.5 checkpoint 和采样设置固化为带 manifest 的 artifact。artifact 至少记录模型 hash、dataset hash、task、normalizer、action horizon、`M`、time/noise seed、alpha 和 threshold；配置不匹配时 fail fast。

## 13. 当前最重要的设计判断

1. **能否接入：能。** RLinf 的 expert routing、intervention data、SFT 和 RLT demo BC 都已存在。
2. **是否直接搬 DiffDAgger 仓库：不应该。** 只移植算法，继续使用 RLinf 的 ManiSkill、worker 和数据结构。
3. **是否与论文完全一致：Stage A 可以接近；Stage B 不是完全一致。** Stage B 的 student 是 RLT actor，而 uncertainty model 是 pi0.5。
4. **第一版是否在线更新 pi0.5：不建议。** 先冻结 pi0.5，验证 gate + RLT；再做每 `N_d` 次 intervention 的交替 SFT/recalibration。
5. **expert 用什么：优先用已经训练好的全分布 pi0.5 expert checkpoint。** ManiSkill motion planner 在 env 内部，无法直接通过当前 rollout worker 的 observation-only expert API无缝调用；要用 planner需另做 EnvWorker RPC/expert-provider 接口。
6. **首个任务：继续 PegInsertionSideWideClearance-v1 做工程闭环。** 若正式复现论文的 Plug setting，再迁移到 `PlugCharger-v1` 并采用任务专属 calibration。
