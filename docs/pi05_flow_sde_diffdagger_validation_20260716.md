# pi0.5 Flow-SDE + DiffDAgger 前六步验证记录

## 记录时间

2026-07-16。验证服务器：`root@39.101.70.188 -p 1020`。

本次只做环境、模型和代码路径验证，没有下载 ManiSkill policy checkpoint，也没有启动正式训练。

## 1. Base pi0.5 检查

已确认 base 模型下载完成并存在于 OSS 挂载目录：

```text
/mnt/data/ask4help/models/pi05_base_torch
```

其中 `model.safetensors` 约 7.23GB。原始 OpenPI checkpoint 也保存在：

```text
/root/ask4help_model_downloads/openpi/openpi-assets/checkpoints/pi05_base
```

## 2. RLinf 环境与代码检查

服务器上的代码分支为：

```text
Ask4Help: codex/flow-sde-diffdagger
RLinf:    codex/flow-sde-diffdagger
```

已验证以下模块可以导入：

```text
torch
ray
transformers
openpi
mani_skill
```

GPU CUDA 可用。Hydra 配置能够成功 compose，且保留 pi0.5 Flow-SDE 关键设置。

RLinf DiffDAgger 相关单元测试结果：

```text
18 passed
```

测试命令：

```bash
cd /root/Ask4Help/RLinf
.venv/bin/python -m pytest -q \
  tests/unit_tests/test_diffdagger.py \
  tests/unit_tests/test_maniskill_dopamine_reward_contract.py
```

## 3. pi0.5 action expert 与 ManiSkill 检查

使用 `pi05_rlt_maniskill_joint` 数据配置时，pi0.5 action expert 成功加载并输出：

```text
action shape: (1, 10, 8)
```

这与 Peg/RLT 管线的 8 维 joint action 一致。服务器此前已经成功完成 RLT Peg Stage 2 smoke，结果位于：

```text
/mnt/data/ask4help/results/rlt_peg_short_train_20260715_134011
```

其中包含 actor checkpoint 和 replay/update 指标，说明 RLT ManiSkill 环境本身可以运行。

### 暴露出的配置问题

当前 DiffDAgger 示例配置仍默认：

```text
PutOnPlateInScene25Main-v3
action_dim: 7
```

而本项目当前目标是 Peg/RLT：

```text
PegInsertionSide...
action_dim: 8
```

因此不能直接把当前 DiffDAgger 配置当作 Peg hybrid smoke 配置使用。此前使用临时 override 验证了 8 维 action forward，但旧 PutOnPlate reward 配置会访问 Peg 环境不存在的字段。这是配置适配问题，不是 pi0.5 或 CUDA 故障。

## 4. Student / expert / calibration 资产检查

当前 OSS 中可以找到：

- pi0.5 base 模型；
- RLT Peg Stage 1/Stage 2 actor checkpoint；
- Robo-Dopamine 模型；
- RLT Peg smoke 数据和 norm stats。

没有找到可以直接用于 DiffDAgger 的独立 ManiSkill expert checkpoint，也没有找到已有的 ID uncertainty calibration 文件。

因此目前不能把 RLT actor、base 模型或同一个 student checkpoint 无依据地当作正式 expert。

## 5. DiffDAgger uncertainty 检查

已使用 RLT Peg smoke 数据和 pi0.5 action expert 启动 flow reconstruction uncertainty smoke。模型成功读取：

```text
/mnt/data/ask4help/datasets/lerobot/local/rlt_peg_smoke/norm_stats.json
```

但在实际执行 uncertainty 时，单个样本、单个 flow timestep 也超过 180 秒没有完成。GPU 没有 OOM，进程主要消耗计算资源；日志没有出现 Python traceback。说明当前实现路径可以进入 uncertainty 计算，但 pi0.5 的 flow velocity reconstruction 成本较高，需要进一步做 profiling 或降低 smoke 参数。

## 6. Hybrid DiffDAgger smoke 状态

完整 hybrid smoke 暂未启动，原因是：

1. 缺少独立 expert checkpoint；
2. 缺少 ID calibration score 文件；
3. 当前配置仍使用 PutOnPlate/7 维 action，需要先改为 Peg/8 维 action；
4. uncertainty 单次计算耗时过长，尚未完成性能验证。

因此本次没有伪造 intervention 结果。当前已完成的是：

```text
base 模型存在
环境依赖可用
pi0.5 action forward 可用
ManiSkill RLT Peg 管线此前可运行
DiffDAgger 单元测试通过
```

尚未完成的是：

```text
Peg 专用 DiffDAgger 配置
独立 expert rollout
ID calibration
真实 student -> uncertainty -> expert intervention 闭环
```

## 下一步

下一步应先把 DiffDAgger 配置改成 Peg 专用配置，并明确 student/expert checkpoint 来源；然后用较少 flow timestep 做可完成的 uncertainty calibration，最后再启动 hybrid smoke。正式实验前还需要单独报告 uncertainty 延迟和 intervention 率。
