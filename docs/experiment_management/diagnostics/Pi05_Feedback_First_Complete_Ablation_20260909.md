# 首轮 Timing / TASR 消融：完整结果与下一步

所有数字均为真实新采集数据，不是后SFT成功率。TASR下表用既有×3口径，×1/×2仍保留在各原始报告，不按结果选倍数。

| 任务/版本 | OOD配对报警：提前/相同/推迟 | 共同保留专家动作预算 | TASR off→on | 实际专家动作总成本 off→on |
|---|---:|---:|---:|---:|
| StackCube，原displacement_v1 | 2/8/0 | 296 | 122/296→123/296（41.22%→41.55%） | 392→397 |
| PickPlane，原displacement_v1 | 0/7/0 | 828 | 104/828→104/828（12.56%→12.56%） | 1758→1758 |
| OpenDrawer Grasp-OOD，原displacement_v1，原任务eval模式 | 0/7/0 | 325 | 60/325→60/325（18.46%→18.46%） | 670→670 |
| OpenDrawer Goal-OOD，原displacement_v1，原任务eval模式 | 0/10/0 | 708 | 203/708→203/708（28.67%→28.67%） | 812→812 |
| PickPlane，独立commitment_v2候选，stream1772000 | 6/2/0 | 1327 | 200/1327→345/1327（15.07%→26.00%） | 2247→2497 |
| PickPlane，commitment_v2候选，stream1773000 | 5/1/0 | 167，仅ID | 无法检验OOD TASR：匹配集没有OOD后缀 | 1292→1430 |

每个旧pilot每arm20raw，ID/OOD各10。配对时间只统计双方都报警的轨迹，不能把无报警当作一个很晚的时间。OOD无报警：原Plane3→3；Drawer Grasp3→3；Goal0→0；Plane候选两stream分别2→0、4→3。

预算是用于训练的完整成功专家后缀动作数，不等于实际执行的专家总动作，更不是包含规划/等待的人工墙钟时间。Plane旧Oracle有候选回退分支，成本包含所有实际模拟执行的专家尝试。

## OpenDrawer 的零结果不是省略或舍入

根目录为：
`/mnt/data/ask4help/results/pi05_timing_feedback_ablation_v1/opendrawer_stage_feedback_eval_v2`。
两个stage均已生成paired_timing_report、pilot_TASR和独立collection audit。

额外逐项检查全部成功训练后缀：Grasp4对、Goal11对的actions、qpos、main RGB、wrist RGB完全相同。因而没有理由把这两份相同输入各训练一次后，将优化噪声导致的SR差异解释成反馈收益。

OpenDrawer此前通用接口误用train/SDE所得50-ID10-success不用于这些结果。上表使用原任务eval模式重新完成的32-demo/50-ID校准，33/50 strict成功，独立审计通过。该校准成功率不是新的正式基座资格评测。

## 明确定位到的支持度问题

| 条件 | 已有至少2条历史反馈时的query数 | 满足原局部2邻居要求 | 第二近邻距离中位数 | 原截断半径 |
|---|---:|---:|---:|---:|
| Grasp | 1019 | 1 | 0.966 | 0.476 |
| Goal | 811 | 0 | 1.011 | 0.476 |

在一个Gaussian带宽处截断时，边界权重仍为exp(-1/2)≈0.607。原规则丢弃很多仍有一定相似度的样本，在稀疏、每次intervention仅1条cue的记忆中几乎未激活。这个诊断不能证明反馈方向本身正确，也不能证明更长收集流下原方法仍不会生效。

## 新实验已经启动，但没有提前认定有用

另存soft_mass候选：Gaussian带宽不变，取消硬截断，但总权重必须达到原两条邻居保证的下限2exp(-1/2)，避免仅凭很多极远点凑数。其它PCA、初始阈值、cue、lambda、beta和5步deadline不变。新Grasp/Goal seeds为1784000/1785000，各arm固定40raw，并从空记忆开始。

该候选是根据上述零结果提出的探索性修订。新任务会分别报告实际Timing、TASR和成本，不能把固定历史上的离线支持度增加当成闭环效果。

仍需将“记忆变多”与“支持规则改变”分开：在读取任何新的soft on/off结果之前，登记同样40新seed的原hard-radius反馈补充对照，复用同一fixed控制数据，不再重复生成fixed。若原规则在40raw已经奏效，应如实报告，不能把所有增益归因于软邻域。

## 训练准备与仍缺的证据

PickPlane候选第一组1327/1327数据已原生导出、通过末尾mask和实际归一化/1:1混合batch检查；SFT入口显式只优化真实8维动作。还没有真实模型两步训练/重载或正式SFT，不能填写任何新post-SFT SR。

用户对本轮保留自然ID/OOD比例、使用历史preliminary OpenDrawer基座的例外尚待确认。这个问题不影响继续收集和诊断Timing/TASR，但必须在正式训练准入时明确。
