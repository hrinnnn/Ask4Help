# pi0.5 StackCube ID SFT 执行记录

## 目标

在 ManiSkill `StackCube-v1` 上先建立一个容易学习、可重复的窄分布 ID
任务，完成以下闭环：

```text
恢复 OSS 环境和模型
-> 受控 ID reset
-> 128 条成功 expert demonstrations
-> 数据与视频全量校验
-> 两个独立 pi0.5 成员各训练 2000 steps
-> held-out ID 成功率和视频评测
```

本阶段不做 OOD、VFD、Ask for Help、AWBC 或 Robo-Dopamine。

## ID 定义

- `cubeA`：红色待抓取方块。
- `cubeB`：绿色目标底座。
- ID 只包含一个相对方向模式，不使用“左边或前面”的双模态混合。
- 在机器人/桌面坐标系中，`cubeA` 位于 `cubeB` 左侧窄扇区：
  - 相对方向角：`[-10 deg, +10 deg]`；
  - 中心距离：`[0.08 m, 0.10 m]`；
  - `cubeB` 在桌面中心附近独立抖动：x/y 各 `[-0.02 m, +0.02 m]`。
- 两方块 yaw、机器人初始关节噪声使用固定且可复现的小幅随机分布。
- 训练、held-out ID 共享同一几何分布，但 seed 不重叠。
- reset manifest 必须记录 seed、两个方块 pose、相对距离和相对方向角。

## 数据

- 收集 128 条成功 expert trajectories。
- 使用 ManiSkill 官方 StackCube motion-planning solution 作为 expert 主体；只适配受控
  reset 和当前控制接口，不自行手写替代任务逻辑。
- 只接受满足官方 `success=true` 的轨迹。
- 导出与当前 pi0.5 输入一致的 LeRobot 数据：base camera、wrist camera、机器人状态和动作。
- 训练前必须通过以下门禁：
  - 恰好 128 条 parquet、128 条视频、128 个 unique seed；
  - 每条轨迹均有官方成功终点；
  - 所有 reset metadata 均落在 ID 区间；
  - 每个视频可解码、有多帧且 RGB 确实变化；
  - 状态和动作不是常量，帧数、action 数和 manifest 严格对齐；
  - 随机抽取至少 2 条视频人工检查；
  - 使用该数据单独计算并保存 `norm_stats_id.json`。

## 训练

- 两个成员从相同 pi0.5 初始权重开始，但使用独立 seed `1000`、`1001`。
- 每个成员独占一张 GPU，`world_size=1`，防止误变成一个两卡 FSDP 模型。
- 第一阶段各训练 2000 steps，`awbc.enabled=false`。
- 每 250 steps 保存 checkpoint，至少保留 250/500/750/1000/1250/1500/1750/2000。
- checkpoint 必须包含恢复训练所需的 actor、optimizer、scheduler、global step 和 RNG 状态。
- 训练输出直接写入 OSS；关键 checkpoint 额外做不可覆盖归档和 SHA-256 校验。
- 若 2000-step ID 效果不足，从完整 checkpoint 续训，不使用 weights-only 假装恢复。

## 评测

- 两成员各使用 20 个固定、与训练不重叠的 held-out ID seeds。
- 纯 policy rollout，不调用 expert、VFD 或 dense privileged information。
- 保存逐 episode success、阶段事件、动作统计和视频。
- 首轮重点报告：
  - 官方 task success rate；
  - grasp rate；
  - on-cube rate；
  - release-and-static rate；
  - 平均 episode 长度。
- ID 目标是先达到至少 80% success；2000 steps 是首个评测点，不是假设一定收敛。

## 环境与持久化

- 新实例优先恢复：
  `/mnt/data/ask4help/environments/rlinf-openpi-maniskill-h20x2-20260720/`。
- 代码以 GitHub 为真源；服务器只拉取已 push 的分支。
- 模型使用 OSS 中的 `pi05_base_torch`。
- 数据、日志、checkpoint 和视频均写到 `/mnt/data/ask4help/`。
- 环境恢复后先执行 import/GPU/ManiSkill render smoke，再开始采集。

## 监测策略

- 环境恢复、首条 demo、数据门禁、训练启动和失败诊断：现场处理，不依赖延迟监测。
- 短任务和阶段切换：约 5 分钟检查。
- 稳定采集：根据吞吐使用 10-20 分钟检查。
- 稳定训练：先 10 分钟确认，再根据每 250-step 周期放宽到 30-60 分钟。
- 发现错误时立即处理；只有确认恢复健康后才重新挂延迟监测。

