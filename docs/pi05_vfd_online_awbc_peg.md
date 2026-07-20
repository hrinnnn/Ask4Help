# pi0.5 VFD Online AWBC: PegInsertion 工程闭环

本实验不使用 RLT、Robo-Dopamine 或动态阈值。Peg 只用于验证完整工程链路，不能作为 Generalization 的正式结论。

## 固定设置

- 初始数据：32 条 ManiSkill privileged-state oracle demonstration。
- VFD ensemble：同一 `pi05_base` 和同一初始数据，分别用 seed `1000`、`1001` 各 SFT 2,000 steps。
- 阈值：20 条 ID policy-success rollout；每条均匀抽 5 个 action chunk，共 100 个 VFD score；取全局 `q=0.95` 分位数，写入 `fixed_vfd_threshold.json`。
- 一个 chunk：10 个低层环境 action；每个 chunk 前计算一次 `C=5` 的 one-way VFD。
- ManiSkill：使用 `physx_cpu` 物理和 CUDA 图像渲染。当前 ManiSkill 官方 Panda motion planner 在本环境的 GPU-physics 变体无法稳定规划；CPU 物理是已有 expert collector 已验证的官方兼容路径，并不把图像或 pi0.5 推理移到 CPU。
- Online：每轮 2 条 rollout，累计 10 轮；每轮两个模型各进行 50 steps ARM exact AWBC 更新。
- Uniform BC：从相同初始 reference checkpoint、使用完全相同累计数据，运行 500 steps 作为 matched control。

## 执行语义

每个 chunk 都重新计算 VFD。若 `VFD > T_fixed`，ManiSkill official motion planner 从当前 simulator state 规划并只执行一个 expert chunk；若不超过阈值，执行 policy chunk。下一个 chunk 无条件重新仲裁，因此可以出现 `policy -> expert -> policy`。

固定阈值在首轮 online 过程中不更新。由于两个 VFD 模型仍持续 AWBC 更新，该阈值是有意保留的工程启发式；每轮必须记录 VFD 分布与 trigger rate，不将它表述为 online conformal coverage。

## 初始数据与阈值

```bash
REPO_ID=local/online_awbc_peg/initial_expert \
OUTPUT_DIR=/mnt/data/ask4help/results/online_awbc_peg/initial_expert \
bash scripts/online_awbc_peg/collect_initial_oracle.sh

MEMBER_0=/mnt/data/ask4help/models/online_awbc_peg/member_0 \
MEMBER_1=/mnt/data/ask4help/models/online_awbc_peg/member_1 \
NORM_STATS=/mnt/data/ask4help/datasets/lerobot/local/online_awbc_peg/norm_stats.json \
OUTPUT_DIR=/mnt/data/ask4help/results/online_awbc_peg/calibration \
bash scripts/online_awbc_peg/calibrate_fixed_threshold.sh
```

## 单轮采集

```bash
MEMBER_0=/mnt/data/ask4help/models/online_awbc_peg/member_0 \
MEMBER_1=/mnt/data/ask4help/models/online_awbc_peg/member_1 \
NORM_STATS=/mnt/data/ask4help/datasets/lerobot/local/online_awbc_peg/norm_stats.json \
THRESHOLD_PATH=/mnt/data/ask4help/results/online_awbc_peg/calibration/fixed_vfd_threshold.json \
REPO_ID=local/online_awbc_peg/round_00 \
OUTPUT_DIR=/mnt/data/ask4help/results/online_awbc_peg/round_00 \
bash scripts/online_awbc_peg/collect_online_round.sh
```

每个 round 产出独立 LeRobot dataset、`progress.jsonl`、`online_chunks.jsonl`、视频和摘要。累积数据集必须按追加顺序合并，且合并后的 AWBC manifest 使用同一顺序；训练配置将 `awbc.expert_sampling_ratio: null`，因此不会强制 1:1 批采样。
