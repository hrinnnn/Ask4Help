# pi0.5 Flow-SDE PPO + DiffDAgger 实施记录

## 目标

本次工作的目标是在 RLinf 的 ManiSkill 管线中直接对 pi0.5 action expert
进行 Flow-SDE PPO，并将 DiffDAgger 的 uncertainty 检测和 expert intervention
接入同一个在线 RL 闭环。

代码在以下两个仓库的同名分支中维护：

```text
branch: codex/flow-sde-diffdagger
Ask4Help commit: eb3191c
RLinf commit:    37b4d6dc
```

后续顶层仓库可能包含其他 agent 的追加提交，因此以上哈希用于定位本次实现的
原始提交。没有下载或写入 ManiSkill checkpoint。

## 最终数据流

```text
ManiSkill observation
  -> pi0.5 Flow-SDE 生成 student action chunk
  -> 在多个 flow timestep 上计算 velocity reconstruction MSE
  -> 使用 ID calibration empirical CDF 得到 uncertainty quantile
  -> patience gate 判断是否请求 expert

未触发 intervention:
  -> 执行 student chunk
  -> 作为 on-policy 样本进入 Flow-SDE PPO

触发 intervention:
  -> 执行 expert chunk
  -> 从 PPO loss 中排除该 chunk
  -> 使用 expert model action 计算辅助 flow-matching SFT loss
  -> 更新 pi0.5 action expert
```

Flow-SDE PPO 保持的关键配置为：

```yaml
openpi:
  noise_method: flow_sde
  noise_level: 0.5
  joint_logprob: false
  train_expert_only: true
algorithm:
  entropy_bonus: 0.0
```

`train_expert_only: true` 表示训练 action expert 和 value head，不更新 VLM。

## DiffDAgger uncertainty

原版 DiffDAgger 对 diffusion policy 生成的 action 重新加噪，并以 diffusion
training loss 作为 uncertainty。pi0.5 是 flow-matching VLA，因此本次使用数学上
对应的 velocity reconstruction loss：

```text
x_t = t * noise + (1 - t) * action
velocity_target = noise - action
uncertainty = mean((velocity_prediction - velocity_target)^2)
```

实现会在多个均匀分布的 flow timestep 上计算每个并行环境各自的 MSE，再对
timestep 和 noise sample 求平均。默认参数：

```yaml
alpha: 0.99
uncertainty_num_timesteps: 16
uncertainty_num_noise_samples: 1
patience: 2
patience_window: 2
```

以下部分与公开 DiffDAgger 代码语义一致：

- 使用 in-distribution uncertainty 构造 empirical CDF。
- 使用 CDF quantile 作为查询阈值。
- 使用滑动窗口和 patience，避免一次偶然高 loss 直接触发 expert。
- 在线 uncertainty 在 policy 自己生成的 action 上计算。

以下部分是接入 RLinf 后的工程设计，不是原 DiffDAgger 论文的原始结论：

- 用 flow velocity-MSE 替换 DDPM noise-prediction MSE。
- 将非 intervention 数据交给 PPO。
- 将 intervention 数据从 PPO 中排除，并用于辅助 SFT。
- 在同一轮更新中组合 PPO loss 和 expert SFT loss。

## 主要代码改动

### DiffDAgger 核心

文件：`RLinf/rlinf/algorithms/diffdagger.py`

实现内容：

- JSON、JSONL、NPY、NPZ calibration score 加载。
- dependency-free empirical CDF 和 quantile。
- 每个并行环境独立的 patience history。
- episode reset 时按环境清空历史。
- flow noisy action、velocity target 和 reconstruction MSE。
- intervention PPO mask。
- student/expert action 与 expert label 的按行融合。

### pi0.5 uncertainty 前向

文件：`RLinf/rlinf/models/embodiment/openpi/openpi_action_model.py`

新增 `compute_diffdagger_uncertainty`：

- 复用 pi0.5 observation processor。
- 只构建一次 VLM prefix KV cache。
- 在多个 flow timestep 上调用 action expert velocity prediction。
- 只统计环境实际 action chunk 和 action dimension。
- 返回 `[num_envs]` 的独立 uncertainty score。

### Rollout 与 expert intervention

文件：`RLinf/rlinf/workers/rollout/hf/huggingface_worker.py`

实现内容：

- 初始化 calibration CDF 和 query gate。
- student rollout 后计算 uncertainty。
- 只为 query mask 中的环境执行 expert action。
- 保存 uncertainty、CDF、threshold 和 intervention flags。
- calibration-only 模式输出 JSONL score。
- 将 classic DAgger 的随机 beta 调度与 DiffDAgger 完全隔离。
- bootstrap value 只使用 student，不触发 expert intervention。

### PPO 与辅助 SFT

文件：`RLinf/rlinf/workers/actor/fsdp_actor_worker.py`

实现内容：

- expert 执行的 chunk 不属于 student on-policy action，因此从 PPO loss mask 中排除。
- 选出 intervention row，读取已保存的 expert `model_action`。
- 调用 OpenPI 原有 DAgger SFT batch processor。
- 将辅助 SFT loss 以 `sft_loss_coef` 加入总 loss。

### 数据传输和指标

相关文件：

```text
RLinf/rlinf/data/embodied_io_struct.py
RLinf/rlinf/workers/env/env_worker.py
RLinf/rlinf/utils/metric_utils.py
```

新增并贯通以下字段：

```text
diffdagger_scores
diffdagger_cdf_values
diffdagger_thresholds
intervene_flags
```

EnvWorker 会在 DiffDAgger 启用时把 `dones` 传给 rollout worker，确保 patience
状态不会跨 episode。新增日志包括：

```text
diffdagger/uncertainty_mean
diffdagger/uncertainty_min
diffdagger/uncertainty_max
diffdagger/cdf_mean
diffdagger/intervention_rate
diffdagger/sft_samples
diffdagger/sft_loss
```

## 配置与脚本

新增配置：

```text
RLinf/examples/embodiment/config/experiment/maniskill_pi05_flow_sde_base.yaml
RLinf/examples/embodiment/config/maniskill_ppo_openpi_pi05_flow_sde.yaml
RLinf/examples/embodiment/config/maniskill_openpi_pi05_diffdagger_calibration.yaml
RLinf/examples/embodiment/config/maniskill_ppo_openpi_pi05_flow_sde_diffdagger.yaml
```

新增脚本：

```text
scripts/maniskill_flow_sde/run_flow_sde_smoke.sh
scripts/maniskill_flow_sde/run_diffdagger_calibration.sh
scripts/maniskill_flow_sde/run_diffdagger_smoke.sh
```

当前默认任务沿用 RLinf 官方 pi0.5 ManiSkill 管线的
`PutOnPlateInScene25Main-v3`，而不是额外下载一个 ManiSkill policy checkpoint。

## 测试记录

新增测试文件：

```text
RLinf/tests/unit_tests/test_diffdagger.py
```

覆盖范围：

- calibration 文件解析和非法数值。
- empirical CDF 和原版 quantile indexing。
- 多环境 patience、滑动窗口和 episode reset。
- flow interpolation、velocity target 和逐样本 MSE。
- expert 只替换 query row。
- expert chunk 从 chunk-level PPO mask 中排除。
- uncertainty 诊断字段在 trajectory 转换和筛选中不丢失。
- 三个 Hydra 配置能够组合，并保持 Flow-SDE 关键不变量。
- fake student/expert 的 rollout worker 集成测试。

执行结果：

```text
本地 macOS:
  23 passed
  Ruff check passed
  Ruff format check passed
  Python py_compile passed
  shell syntax check passed

Aliyun H20 server:
  相关回归套件 22 passed
  最终 DiffDAgger 测试文件 12 passed
```

## 当前未完成项

代码完成时，用户启动的 `pi05_base` 下载仍位于：

```text
/root/ask4help_model_downloads/openpi/openpi-assets/checkpoints/pi05_base.partial
```

当时约为 7GB，因此遵照要求没有启动完整模型 smoke，也没有另行下载
ManiSkill checkpoint。

完整验证还需要：

1. 下载完成的 pi0.5 base 或 task-adapted student。
2. 能成功完成目标任务的 expert checkpoint 或 expert policy。
3. 在 student 的 ID demonstrations 上计算出的 calibration score。
4. 先运行 Flow-SDE PPO smoke，再运行 calibration 和 DiffDAgger hybrid smoke。

需要特别注意：仓库提供的 calibration rollout 脚本用于工程管线验证。正式复现
DiffDAgger 时，CDF 应来自 student adaptation 使用的 ID demonstration
observation/action pairs，而不能只使用普通环境 rollout 代替。
