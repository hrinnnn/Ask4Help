# RLinf pi0.5 VFD Uncertainty 接入计划

## 目标

将 SAVE（Uncertainty Quantification for Flow-Based Vision-Language-Action
Models）提出的 Velocity-Field Disagreement（VFD）接入当前的 RLinf +
OpenPI pi0.5 + ManiSkill 管线。

第一版实现：

```text
两个独立微调的 pi0.5 ensemble members
  -> vfd_oneway uncertainty
  -> ID 成功轨迹标定
  -> rollout monitor-only 记录
  -> uncertainty gate
  -> expert takeover
```

第一版不实现 SAVE 的完整多任务主动选数循环，也不将 VFD 当作 reward 或
progress estimator。

## Git 与代码边界

使用两个同名分支：

```text
Ask4Help: codex/pi05-vfd-uncertainty
RLinf:    codex/pi05-vfd-uncertainty
```

`uq_vla` 是修改过的 LeRobot 框架，不整体并入 RLinf。仅参考并移植：

- `Pi05Adapter` 的 observation prefix cache 与 velocity 接口；
- `VfdOneway` / `Vfd` 的数学实现；
- `CrossBayesianSampler` 的 action sampling 与 ensemble scoring 数据流。

## 模型角色

```text
member 0: 执行动作的 pi0.5 student，同时是 VFD reference model
member 1: 只用于 VFD 的 pi0.5 comparison/scorer model
expert:   uncertainty 触发后真正接管环境的 expert policy
```

member 1 不能当作 expert。两个 ensemble members 从同一个 pi0.5 base
checkpoint 出发，用相同数据和超参数、不同 seed 和 shuffle 分别训练：

```text
pi05_base
  |-- seed=1000, shuffle=1000 -> member_0
  `-- seed=1001, shuffle=1001 -> member_1
```

第一版保持 `openpi.train_expert_only: true`，先验证 action-expert-only
ensemble 是否已经能区分 ID、OOD 和失败状态。

## OpenPI VFD Adapter

在 `OpenPIActionModel` 中提供三个内部能力。

### prepare_vfd_conditioning

每个 observation、每个 ensemble member 只执行一次图像与语言 prefix 编码，
并保存 prefix attention mask 和 transformer KV cache。随后同一 observation 下
所有 flow timestep 和 action sample 复用该 conditioning。

### make_vfd_velocity_fn

暴露统一的 velocity 接口：

```text
v = velocity_fn(x_t, ode_t)
```

其中 `x_t` 是 flow 中间 action state。必须处理 pi0.5 与通用 ODE 的时间方向
差异：

```python
pi05_t = 1.0 - ode_t
ode_velocity = -pi05_velocity
```

### compute_vfd_uncertainty

接收两个模型、同一 observation 和共同的初始噪声，返回每个并行环境的 VFD
score，shape 为 `[batch_size]`。

## 第一版 VFD 算法

优先实现发布配置默认使用、计算成本更低的 `vfd_oneway`：

1. 为每个 observation 采样 `C` 个共同高斯初始噪声。
2. 使用 member 0 完成 flow ODE integration，并保存中间状态 `x_s`。
3. 在 member 0 的同一批 `x_s` 上查询两个模型的 velocity。
4. 计算每个 flow time 的 squared L2 disagreement：

   ```text
   d_s = ||v_member0(x_s, o) - v_member1(x_s, o)||^2
   ```

5. 使用 `kappa_s = s / (1 - s)` 和 flow interval `ds` 加权。
6. 对 flow time、action samples、action horizon 和 action dimensions归约，得到
   每个 observation 的 scalar VFD。

工程 smoke 从以下配置开始：

```yaml
uncertainty:
  type: vfd
  mode: oneway
  num_action_samples: 1
  velocity_eval_times: [0.0, 0.25, 0.5, 0.75, 0.9]
```

确认延迟可接受后切换到 `num_action_samples: 5`。不在 `s=1.0` 计算，避免
`kappa_s` 发散。完整双向 `vfd` 作为后续可选模式，不作为第一版默认路径。

## Rollout 接入

接入 `rlinf/workers/rollout/hf/huggingface_worker.py`：

```text
observation
  -> member 0 生成 candidate action chunk
  -> member 0 / member 1 计算 VFD
  -> 写入 trajectory 和 metrics
  -> gate 判断是否请求 expert
  -> 执行 student action 或 expert action
```

第一阶段只启用 monitor-only：计算并记录 VFD，但不允许触发 expert。完成 ID / OOD
曲线和延迟检查后再启用 intervention。

trajectory 至少记录：

```text
vfd_score
vfd_threshold
vfd_violation
intervene_flag
member_0_action
expert_action
```

## Calibration

从约 10 条 ID 成功轨迹计算逐 chunk VFD，使用 one-sided conformal 或固定分位数
生成 threshold artifact：

```json
{
  "method": "vfd_oneway",
  "member_0_hash": "...",
  "member_1_hash": "...",
  "task": "PegInsertionSide",
  "num_action_samples": 5,
  "velocity_eval_times": [0.0, 0.25, 0.5, 0.75, 0.9],
  "quantile": 0.95,
  "threshold": 0.0
}
```

模型 hash、task、normalizer、action horizon 或 VFD 配置不匹配时 fail fast，不能
静默复用旧阈值。

## Ask for Help Gate

复用现有 DiffDAgger gate 的 per-env episode state 与 sticky takeover：

```text
VFD > threshold -> violation
连续 K 次 violation -> expert takeover
expert 持续控制到 success / failure / timeout
episode reset -> 清空 violation history 和 expert latch
```

第一版建议：

```yaml
quantile: 0.95
patience: 2
sticky_takeover: true
```

## 测试计划

### 单元测试

- 两个 velocity 完全相同时 VFD 为 0；
- 人工 velocity 差异与手算公式一致；
- flow-time scaling 和 `ds` 权重正确；
- `s -> 1` 不产生 NaN / Inf；
- 两个模型使用相同初始噪声；
- batch、sample、horizon、action dimension 归约正确；
- `C=1` 与 `C=5` shape 正确；
- pi0.5 时间翻转和 velocity 符号正确；
- inference 全程无梯度；
- calibration 配置或模型 hash 不匹配时明确报错；
- episode reset 后 gate 状态清空；
- monitor-only 模式永远不触发 expert。

### 集成测试

1. fake velocity models 完成 end-to-end VFD 手算验证；
2. 单个真实 pi0.5 observation 完成双模型 VFD；
3. 对 ID、OOD、失败 observation 导出 VFD 曲线；
4. ManiSkill monitor-only rollout；
5. calibration threshold 加载与 violation 检测；
6. expert 在当前 action chunk 执行前接管；
7. intervention trajectory 正确写入并可供后续 SFT / AWBC 使用。

## 性能与有效性验收

- 单张 H20 能稳定加载两个 pi0.5 inference members；
- prefix 只编码一次，flow timestep 不重复运行图像和语言 backbone；
- 报告普通 action inference、`C=1` VFD 和 `C=5` VFD 的延迟；
- ID 成功状态 VFD 显著低于 OOD / failure 状态；
- VFD 与 episode success 呈稳定负相关；
- 报告 failure detection AUROC、TPR、TNR 和 detection time；
- expert intervention 主要落在真正困难或危险的状态。

## 实施顺序

```text
Stage 1: 移植 VFD 数学核心并完成单元测试
Stage 2: 实现 pi0.5 KV-cache velocity adapter
Stage 3: 两个 checkpoint 的单 observation VFD smoke
Stage 4: 独立训练两个 action-expert ensemble members
Stage 5: 使用 ID 成功轨迹生成 calibration artifact
Stage 6: Peg monitor-only rollout 并绘制 VFD 曲线
Stage 7: 接通 uncertainty gate 与 expert takeover
Stage 8: 对比 VFD、DiffDAgger 和 privileged gate
```

第一轮工程验证优先回答：VFD 是否能区分 Peg 的 ID / OOD，以及单张 H20 上的
实时延迟是否可接受。通过后再与 Robo-Dopamine progress 和 AWBC 数据更新闭环
组合。
