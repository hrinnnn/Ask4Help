# RLinf + RLT + pi0.5 + ManiSkill Peg 复现记录

## 目标

本次工作的目标是先跑通 RLinf 中的 RLT token-style online RL 基础闭环：

```text
pi0.5 base checkpoint
  -> Stage 1 SFT / RLT feature model
  -> ManiSkill Peg rollout
  -> replay buffer
  -> RLT actor-critic update
```

这份记录描述的是工程 smoke 和中短训练，不代表 Peg insertion 已经取得有效任务成功率。Robo-Dopamine/GRM 还没有接入这条 RLT 管线。

## 代码与服务器

- 本地工作区：`Ask4Help_agent_workspace`
- Git 分支：`codex/flow-sde-diffdagger`
- 代码源：`https://github.com/hrinnnn/Ask4Help`
- 服务器工作区：`/root/Ask4Help`
- RLinf：`/root/Ask4Help/RLinf`
- RLinf Python 环境：`/root/Ask4Help/RLinf/.venv`
- GPU：NVIDIA H20，约 96 GiB 显存
- 验证环境：PyTorch `2.6.0+cu124`，CUDA 可用

服务器只作为运行环境使用。代码和脚本在本地修改后提交到 GitHub；模型、数据、checkpoint 和日志写入 OSS 挂载目录 `/mnt/data/ask4help`。

## 模型准备

下载的是官方 OpenPI pi0.5 base checkpoint：

```text
/root/ask4help_model_downloads/openpi/openpi-assets/checkpoints/pi05_base
```

下载完成标志：

```text
params/commit_success.txt
```

随后使用 RLinf 官方转换器：

```text
RLinf/rlinf/utils/ckpt_convertor/convert_openpi_jax_to_python.py
```

转换时使用 OpenPI 原生配置 `pi05_droid`。这里不能使用 RLinf 注册的
`pi05_rlt_maniskill_joint`，因为它不是 OpenPI converter 能识别的原生配置。

PyTorch 权重位置：

```text
/root/ask4help_model_downloads/pi05_base_torch
/mnt/data/ask4help/models/pi05_base_torch
```

OSS 中的持久化副本约为 `6.8G`。

### 期间修复的问题

1. 转换脚本使用相对路径时，在 `/root` 下找不到转换器，改为使用绝对 RLinf 路径。
2. 将转换配置从 RLinf 专用名称改为 OpenPI 原生 `pi05_droid`。
3. 服务器没有 `rsync`，持久化脚本增加了 `cp -a` fallback。

## 数据准备

本次使用本地 LeRobot 格式的 Peg smoke 数据：

```text
/mnt/data/ask4help/datasets/lerobot/local/rlt_peg_smoke
```

已验证：

- 8 个 parquet 数据文件
- `norm_stats.json` 存在
- 数据包含 `actions`、`image`、`wrist_image`、`state` 和 task 字段
- RGB 图像尺寸为 `384 x 384`

## Smoke 流程

### 1. 基础环境和 ManiSkill

脚本：

```bash
bash scripts/rlt_peg_smoke/verify_rlinf_rlt_peg.sh
```

验证内容：

- `torch.cuda.is_available()`
- RLinf、OpenPI、Ray、Transformers、ManiSkill import
- 注册 Peg insertion side variant
- GPU backend 创建环境
- `reset()` 和 `step()`
- camera、qpos、success 字段

SAPIEN 输出过 Vulkan ICD warning，但 GPU ManiSkill reset/step smoke 成功，因此该 warning 在当前环境中不是阻断问题。

### 2. Loader、forward 和数据 transform

已验证 RLinf 可以加载转换后的 pi0.5 checkpoint，并读取 Peg 数据的 norm stats。
LeRobot 样本经过 RLinf transform 后可以生成 RLT 所需的 state/action 输入。

### 3. Stage 1

启动脚本：

```bash
bash scripts/rlt_peg_smoke/run_stage1_smoke.sh
```

设置：

- 8 条 demo
- `runner.max_steps=2`
- pi0.5 base 作为初始化模型
- 保存间隔为 1 step

结果：

```text
/mnt/data/ask4help/results/rlt_peg_stage1_smoke_20260715_122932
```

Stage 1 成功生成 `global_step_1` 和 `global_step_2` actor checkpoint。Stage 2 使用：

```text
.../global_step_2/actor
```

关键日志指标包括：

```text
train/loss=4.29
train/rlt_loss=4.21
train/vla_loss=0.083
train/grad_norm=2.34
```

### 4. Stage 2 rollout smoke

启动脚本：

```bash
bash scripts/rlt_peg_smoke/run_stage2_smoke.sh
```

设置：

- 2 个 ManiSkill 环境
- 每个 episode 20 步
- `algorithm.update_epoch=1`
- RLT warmup 和 replay buffer 使用小规模 smoke 设置
- expert takeover 关闭
- RLT feature model 和 expert model 都指向 Stage 1 actor

结果目录：

```text
/mnt/data/ask4help/results/rlt_peg_stage2_smoke_20260715_131527
```

Stage 2 完整执行了 rollout epoch，收集到 2 条轨迹。由于样本量还没有达到 replay warmup 阈值，没有触发 actor/critic update。

最终 smoke 指标：

```text
episode_len=20
num_trajectories=2
reward=0
success_once=0
actor_updates_run=0
critic_updates_run=0
```

## 中短 RLT 训练

在 Stage 2 smoke 成功后，运行了有限的 8 epoch 中短训练，而不是无限训练：

```text
runner.max_epochs=8
env.train.total_num_envs=2
env.train.max_episode_steps=20
algorithm.rlt_schedule.warmup_min_size=16
algorithm.rlt_schedule.warmup_post_collect_updates=4
algorithm.rlt_schedule.max_updates_per_train_step=4
```

结果目录：

```text
/mnt/data/ask4help/results/rlt_peg_short_train_20260715_134011
```

最终训练结果：

- 8/8 rollout epochs 完成
- replay buffer 收集 16 条轨迹
- actor update：4 次
- critic update：4 次
- `actor_loss=0.349`
- `critic_loss=0.015`
- `actor/grad_norm=2.898`
- `critic/grad_norm=2.440`
- 保存 4 个 actor checkpoint

checkpoint 位于：

```text
/mnt/data/ask4help/results/rlt_peg_short_train_20260715_134011/maniskill_rlt_stage2_ac_mlp/checkpoints/
```

### W&B 运行方式

第一次中短训练因为新 SSH 进程没有继承离线配置而退出：

```text
wandb.errors.errors.UsageError: No API key configured. Use `wandb login` to log in.
```

重新启动时设置：

```bash
export WANDB_MODE=offline
```

之后训练成功完成。今后的无网络服务器训练应显式设置该变量，或者预先完成 W&B login。

## 当前结论

已经验证的链路是：

```text
pi0.5 checkpoint
  -> RLinf loader
  -> RLT data transform
  -> ManiSkill GPU environment
  -> rollout
  -> replay buffer
  -> RLT actor-critic update
  -> checkpoint
```

这证明工程闭环可以运行，但还不能作为 Peg 任务效果结论。当前 sparse reward 下：

```text
reward=0
success_once=0
```

因此下一步不能直接把这次结果解释为“RLT 已经学会 Peg insertion”。需要先验证 Stage 1 专家/初始化策略在该任务上的成功率，再进行更长训练和独立评估。

## 下一步建议

1. 用相同环境单独评估 Stage 1 actor，确认专家动作和任务 success 逻辑正确。
2. 增大 rollout budget，至少让 replay buffer 收集足够多的成功和失败轨迹。
3. 固定 seed、环境数量、episode steps 和 interaction budget，记录 success rate、episode return 和 checkpoint 曲线。
4. 在 sparse baseline 与 RLT 之间使用相同初始化、任务、seed 和环境交互预算。
5. 基础 RLT 结果稳定后，再接入 Robo-Dopamine GRM 的 potential-based shaping；GRM 应作为后续 reward 模块，不与本次 RLT smoke 结果混在一起。
