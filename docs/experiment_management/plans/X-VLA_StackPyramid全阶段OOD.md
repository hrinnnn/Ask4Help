# X-VLA StackPyramid 全阶段 OOD 计划

**状态：H20 X-VLA 原生环境与 StackPyramid reset/step/render smoke 已完成；官方 motion-planning 单轨迹已保存但 seed0 success=false。正式 oracle、ID/OOD 收集和训练仍须先完成任务审计与 oracle 验收。**

## 1. 目标

将 StackPyramid 建成典型的多阶段 OOD benchmark。每个阶段 OOD 都是正式实验条件，用于检验 gate 能否在不同任务进度上定位缺失能力，而不是从多个阶段中挑选最容易得到好结果的一项。

## 2. 冻结任务语义与阶段

本计划采用三块积木的金字塔任务：机器人先抓取红块并把它放到绿块旁边，使红块和绿块形成底座；随后抓取蓝块，将其放到红绿底座上方，完成金字塔。三个有序阶段定义为：

1. **Stage 1：获取红块**。机器人接近、抓取并抬起红块。
2. **Stage 2：构建红绿底座**。机器人把红块运输到绿块旁边并正确放置。
3. **Stage 3：放置蓝块**。机器人抓取蓝块并将其放到红绿底座上方。

正式实现前必须用实际 ManiSkill 环境和 oracle smoke 核实颜色、成功几何和上述执行顺序。核实后的物体名称、ID/OOD 位置范围、阶段 predicate、成功条件、failure reason、oracle 接口、base checkpoint 和 norm 全部写入 `task_spec.json`。当前仓库尚无已核实的 StackPyramid 正式实现，因此本节是需要实现和验证的冻结设计目标，不能跳过 smoke 直接采集。

## 3. 固定 ID 分布

ID demonstrations 中红、绿、蓝三块的初始位置都限制在各自狭窄、互不冲突的区域，只保留完成训练所需的小幅位置扰动。相机、机器人初始状态、物体朝向、材质、任务指令和成功条件保持固定。ID policy 应先在该集中分布上掌握完整三阶段任务，再用于后续 OOD 和 robot-gated DAgger 实验。

三个物体的 ID 位置范围必须在采集前明确写入 `task_spec.json`。不得使用覆盖整个工作空间的宽随机分布训练 base policy，否则无法把后续失败归因于某个阶段的单一位置变化。

## 4. 三类单变量 Stage-Localized OOD

对同一个 paired seed，先生成完整 ID reset，再只把指定颜色积木的位置替换到与其 ID 区域不重叠的 OOD 区域。另外两块积木及所有其他随机变量必须与 paired ID reset 完全一致。

| 条件 | 唯一变化 | 对应能力缺口 | 必须保持不变 |
|---|---|---|---|
| Stage 1 OOD | 红块初始位置 | 从新位置接近、抓取并抬起红块 | 绿块、蓝块及全部非目标因素 |
| Stage 2 OOD | 绿块初始位置 | 把已抓取的红块运输到新的绿块位置旁边 | 红块、蓝块及全部非目标因素 |
| Stage 3 OOD | 蓝块初始位置 | 从新位置获取蓝块并完成顶部放置 | 红块、绿块及全部非目标因素 |

Stage 2 的绿块变化会改变红块应到达的底座位置，但不应改变红块本身的获取难度。Stage 3 的蓝块变化只在红绿底座完成后引入新的获取路径；底座几何和最终堆叠定义保持不变。

主实验分别测试这三个单变量 OOD，不把三块同时移动。三者可以在最终 mixed deployment stream 中随机交替出现，但每个 episode 仍只激活一个 OOD 因素。三块同时变化属于复合 OOD，只能作为额外泛化实验，不能替代主 timing study。

### 可见性与阶段相关性

如果三块积木从 reset 开始都出现在相机视野中，某个位置变化可能在 episode 开始时已经能够被 detector 观察到，但它对正确动作的影响只在对应阶段发生。因此，本 benchmark 中的“stage-localized”指**能力缺口和专家动作在阶段上局部化**，不等价于视觉 OOD 信号只在该阶段才首次出现。

正式结果应同时记录首次 detector 报警和受影响阶段的进入时刻。这一差异正好用于检验 gate 是否会仅因提前看见 OOD 物体而过早接管，还是能在该变化真正影响当前动作时选择更有学习价值的 takeover timing。不得把 reset 时已可见的位置变化描述成阶段中途才出现的分布变化。

## 5. 每个 Stage 的验证门槛

- 同 seed 的 ID/OOD reset 可复现，并且非目标因素一致。
- oracle 在 ID 和 OOD 上均至少 95% 成功。
- base policy 在受影响阶段之前的 prefix completion rate 至少 80%。
- base policy 在 OOD 上存在明确但非退化的能力缺口；目标范围为完整 SR 不高于 50%，同时能稳定到达目标阶段。
- 每个阶段建立互不重叠的 design、collection、evaluation seeds。
- Stage 2 OOD 不得显著降低红块的 grasp/lift rate；Stage 3 OOD 不得显著降低红绿底座的完成率。
- paired-reset 审计必须证明另外两块的位置和所有非目标 metadata 没有变化。

未满足门槛时只允许修复 oracle 或减少非目标扰动；不得根据 detector 或训练结果事后改变 OOD 分布。

## 6. 正式实验链路

对 Stage 1/2/3 分别执行：

1. 纯 policy rollout 与 failure-detection benchmark。
2. 记录任务阶段、首次报警和失败阶段。
3. 运行该阶段的受控 timing sweep。
4. 使用 matched expert-action budget 训练各 timing condition。
5. 在 100 ID、100 对应 Stage OOD 上评测。
6. 最后增加混合 ID/Stage-1/Stage-2/Stage-3 部署流，检验 gate 的实际数据选择。

## 7. 产物结构

建议根目录：`xvla_stackpyramid_stage_ood_v1/`，其下按 `stage1_grasp`、`stage2_transport`、`stage3_assembly` 分开保存：

- `task_spec.json` 与 seed manifests；
- base-policy rollouts、状态 snapshots 和 nominal oracle trajectories；
- detector、collection、training 与 evaluation；
- 每阶段 summary，以及跨阶段总表。

该任务完成后，才能启动完整的 StackPyramid timing sweep 汇总。

## 8. 运行位置

优先复用 5090 上已核实的 X-VLA 环境。若 5090 在我们的配额内没有足够空闲卡，不等待卡位，直接按《环境与存储》的回退规则在阿里云 H20 建立原生 X-VLA 环境、同步冻结资产并完成 smoke。两处运行必须使用相同源码版本、数据、base checkpoint、norm、seed manifest 和成功定义。
