# pi0.5 AWBC 实现状态

最后更新：2026-07-17

## 当前结论

RLinf 中的 pi0.5 AWBC 训练链路已经完成实现，并通过单元测试、fake GRM
标注、真实 Robo-Dopamine 标注、Uniform BC 两步训练和 ARM Exact AWBC
两步训练。当前结果证明工程闭环可以工作，但尚不能证明 AWBC 能提高任务成功率。

正式实验仍需完成：选择约 25% 成功率的共享 warm-start、收集真实 policy
轨迹、以 5 帧间隔完成真实 GRM 标注，并运行统一预算下的 500-step 对照实验。

## Git 与服务器状态

- Ask4Help 分支：`codex/pi05-awbc`
- Ask4Help 核心实现提交：`9a5be90`
- RLinf 分支：`codex/pi05-awbc`
- RLinf 提交：
  - `f197af8c`：ARM AWBC 权重计算与单元测试。
  - `bf11398e`：OpenPI 数据接口、flow loss 加权及测试。
  - `27e7fb00`：ManiSkill policy rollout 采集配置。
  - `760f2616`：不同 FSDP rank 使用不同的确定性采样 seed。

服务器使用独立 worktree：

```text
/root/Ask4Help-awbc
```

这样做是为了保留旧工作区中尚未提交的修改：

```text
/root/Ask4Help/scripts/rlt_peg_smoke/run_stage2_smoke.sh
```

AWBC 分支的验证没有覆盖、还原或提交这项无关修改。

## 已实现功能

### 1. ARM Exact AWBC 权重

实现了 ARM 论文对应的 gain 和权重计算：

```text
delta_phi = phi_next - phi
gain = delta_phi * episode_length / mean_valid_episode_length
lower = mean_valid(gain) - 2 * std_valid(gain)
upper = mean_valid(gain) + 2 * std_valid(gain)
weight = clip((gain - lower) / (upper - lower + 1e-6), 0, 1)
```

具体行为：

- 均值和标准差覆盖所有 FSDP rank。
- invalid 样本不参与均值、标准差和训练采样。
- 有效样本过少或方差接近 0 时，有效样本权重回退为 1。
- exact 模式保留负 progress 对连续权重的影响。
- 支持可选的 negative-delta、confidence、weight floor 和 gain clip。
- 鲁棒参数默认关闭，不会静默改变 ARM Exact 基线。

### 2. pi0.5 Flow Loss 加权

- 先把 flow-matching loss 按 batch、action horizon 和 action dimension 归约为
  每个样本一个 loss。
- 再使用 AWBC 权重计算加权均值。
- AWBC 只作用于 pi0.5 action expert 的 flow loss。
- RLT loss 不受 AWBC 权重影响。
- `uniform` 模式使用相同有效样本和相同 1:1 sampler，但所有权重为 1。
- 全 1 权重时，数值与原始 `loss.mean()` 一致。
- 零权重样本不会产生梯度贡献。

### 3. Expert 与 Policy 混合数据

- 使用 `ConcatDataset([expert, policy])` 组合两个 OpenPI 数据集。
- 通过稳定的 dataset index 读取 JSON/JSONL progress sidecar。
- batch 内固定使用 `expert:policy = 1:1`。
- sampler 只采样存在完整有效 `phi -> phi_next` 的锚点转移。
- manifest 缺失、重复、越界或长度不一致时立即明确报错。
- 不同 FSDP rank 使用不同但可复现的 seed，避免完全重复采样。

### 4. Robo-Dopamine 离线标注

实现了 ManiSkill LeRobot 数据集的离线标注工具：

- 从成功 expert demo 的最后一帧提取 goal reference。
- episode 第一帧作为 reference start，并设置 `phi=0`。
- 支持 incremental、forward、backward 三种 mode。
- 使用 consistency-aware fusion 得到融合后的 `Phi`。
- 默认正式标注间隔为 5 个环境 step。
- terminal 帧即使不落在固定间隔上也会被标注。
- success 只把当前 episode 的终点设置为 `Phi=1`。
- reset 后重新从 `phi=0` 开始，不污染下一个 episode。
- parser invalid 时不生成伪造的 `Phi=0`，也不更新前一个有效状态。
- 每个锚点写入可恢复 cache，支持中断后继续标注。

### 5. Policy 轨迹采集与训练入口

- 提供 pi0.5 ManiSkill rollout 采集配置。
- 关闭 RLT policy switching，确保采集的是指定 warm-start policy。
- 同时保留成功和失败 episode。
- 默认以 8 个环境、4 个 rollout epoch 采集约 32 条轨迹。
- 提供 Uniform BC、ARM Exact AWBC 和 Robo-Dopamine Robust AWBC 启动脚本。

## 自动测试

### 本地 AWBC 测试

```text
24 passed
```

覆盖内容：

- ARM 公式手算结果。
- episode length scaling。
- invalid、全 invalid、单有效样本和零方差 fallback。
- robust 模式的 negative delta、confidence 和 weight floor。
- 模拟多 rank 全局统计。
- 全 1 权重数值等价性和零权重梯度。
- manifest 对齐、缺失、重复和越界检测。
- expert/policy 平衡采样。
- success/reset 隔离。
- parser invalid 时不生成伪 `Phi=0`。

### 服务器完整测试

```text
50 passed, 2 warnings
```

包含：

- AWBC 权重、数据和标注测试。
- 现有 Dopamine GRM reward model 测试。
- ManiSkill Dopamine reward contract 测试。

两个 warning 分别来自 SAPIEN deprecated API 和 Vulkan ICD 探测；测试本身全部
通过。Ruff、Python compile、Shell 语法和 Git diff 检查也已通过。

## 服务器 Smoke 结果

本次服务器实际暴露的硬件为：

```text
1 x NVIDIA H20, 97871 MiB
```

### 1. Goal Bank 提取

从现有的 8 条 Peg expert demo 中成功提取 goal bank：

```text
task id:              0
source episode:       0
source dataset frame: 50
views:                main + wrist
```

### 2. Fake GRM 完整标注

```text
expert rows:                565
policy rows:                565
combined rows:             1130
valid transitions/source:   114
annotation stride:            5 frames
```

policy manifest 使用 expert 标注 cache 重建，没有再次调用 endpoint。该阶段把同一
数据集分别作为 expert 和 policy，只用于验证索引、缓存、采样和训练链路，不能作为
实验数据。

持久化路径：

```text
/mnt/data/ask4help/results/awbc_peg_smoke/fake_annotation
```

### 3. Uniform BC 两步训练

训练完成，无 Traceback、ERROR 或 OOM。所有权重均为 1，加权 loss 与未加权
loss 一致。

```text
step 1 loss: 0.0780
step 2 loss: 0.0836
ESS:         4.0 / 4
```

checkpoint、TensorBoard 和训练日志：

```text
/mnt/data/ask4help/results/awbc_peg_smoke/uniform_2step
```

### 4. ARM Exact AWBC 两步训练

训练完成，无 Traceback、ERROR 或 OOM，并产生了非均匀权重。

```text
step 1 weighted loss:   0.0865
step 1 unweighted loss: 0.0885
step 1 weight range:    0.312 - 0.915
step 1 ESS:             3.2 / 4

step 2 weighted loss:   0.0785
step 2 unweighted loss: 0.0748
step 2 weight range:    0.280 - 0.885
step 2 ESS:             3.2 / 4
```

checkpoint、TensorBoard 和训练日志：

```text
/mnt/data/ask4help/results/awbc_peg_smoke/arm_exact_2step
```

Uniform 和 AWBC 最终 checkpoint 各约 19 GB，均包含 distributed checkpoint
和 full weights。

### 5. 真实 Robo-Dopamine 单 Episode 标注

从 OSS 快照恢复了 `robo-dopamine` 环境，验证版本为：

```text
torch:        2.8.0+cu128
transformers: 4.57.0
vLLM:         0.11.0
CUDA:         available
```

真实 `Robo-Dopamine-GRM-2.0-4B-Preview` checkpoint 已通过兼容 OpenAI API
的 vLLM endpoint 成功启动，模型架构识别为 Qwen3-VL。

为了快速验证真实模型，该 smoke 只选取一条 demo 的 start 和 terminal，实际执行：

```text
incremental: 1 次 8 图推理
forward:     1 次 8 图推理
backward:    1 次 8 图推理
```

结果：

```text
real endpoint requests: 3
dataset rows:           51
valid transitions:       1
phi_incremental:         1.0
phi_forward:             1.0
phi_backward:            1.0
phi_next:                1.0
```

关闭 endpoint 后，使用不可达端口重新运行标注，仍从 cache 恢复出完整 51 行，证明
缓存复用有效。

结果和 vLLM 日志：

```text
/mnt/data/ask4help/results/awbc_peg_smoke/real_annotation_1ep
```

该快速 smoke 使用 50 帧间隔；正式标注仍使用 5 帧间隔。

### 6. 真实 Peg 完整 Episode 曲线

在同一条 Peg expert episode 上按正式的 5 帧间隔重新运行真实 GRM：

```text
episode frames:          51
GRM anchor transitions:  10
three-mode requests:     30
positive delta_phi:       7
negative delta_phi:       3
```

逐段 progress 为：

```text
frame  0 ->  5: +0.0805
frame  5 -> 10: +0.1952
frame 10 -> 15: -0.0168
frame 15 -> 20: -0.0049
frame 20 -> 25: +0.0173
frame 25 -> 30: +0.1136
frame 30 -> 35: +0.1858
frame 35 -> 40: +0.1073
frame 40 -> 45: -0.0748
frame 45 -> 50: +0.3968
```

最后一个转移包含环境 success 将终点覆盖为 `Phi=1`，因此不能解释为纯 GRM
预测。前 9 个转移使用三模式 consistency-aware 融合结果。

曲线、sidecar、摘要、cache、annotation log 和 vLLM log 均保存在：

```text
/mnt/data/ask4help/results/awbc_peg_smoke/real_episode0_stride5
```

这条轨迹是已有 expert demonstration，不是当前 policy 在线采集的 episode。它验证
了完整单 episode 的曲线导出能力；正式 online AWBC 仍需在 policy rollout 上重复
相同流程。

## 与正式实验的差距

当前尚未完成以下科学实验步骤：

1. 训练并评估不超过 500 steps 的共享 warm-start，选择实测成功率最接近 25%
   的 checkpoint。
2. 使用该 checkpoint 收集 32 条真实 policy 轨迹，并完整保留成功和失败样本。
3. 使用真实 GRM endpoint，以 5 帧间隔标注完整 expert 和 policy 数据集。
4. 在完全一致的数据、batch、学习率、训练步数和随机种子下分别训练：
   - Uniform BC。
   - ARM Exact AWBC。
   - 选定参数的 Robo-Dopamine Robust AWBC。
5. 先用 50 个固定 evaluation seeds 比较趋势，再运行 3 个训练 seed，每种方法
   评估 100 个 episode。
6. 在真正暴露两张 GPU 的实例上运行双 rank NCCL 集成测试。当前实例只有一张
   H20；多 rank 数学统计已经通过模拟测试，但真实两卡通信仍待验证。

## 结果解释边界

当前 smoke 已证明：

- 数据能够对齐并进入 pi0.5。
- AWBC 权重能够按预期改变 flow loss。
- 梯度、checkpoint 和 TensorBoard 输出正常。
- fake 与真实 Robo-Dopamine 都能生成可用 sidecar。
- invalid、success、reset 和 cache 的关键边界行为符合设计。

当前 smoke 尚未证明：

- AWBC 的任务成功率高于 Uniform BC。
- Robo-Dopamine progress 比其他 progress estimator 更有效。
- 当前 robust 参数优于 ARM Exact。

因此，fake GRM 数值和两步训练 loss 只能作为工程正确性证据，不能写成论文中的
性能结论。
