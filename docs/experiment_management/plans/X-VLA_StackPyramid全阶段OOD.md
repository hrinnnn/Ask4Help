# X-VLA StackPyramid 全阶段 OOD 计划

**状态：协议审计中。H20 X-VLA 原生环境、任务审计、oracle gate、ID base policy，以及旧版 Stage-1/2/3 Internal-Feature PCA 结果已保存；旧四方法总控因 Diff 阈值未通过审计而停止，旧结果只作诊断，不能进入正式训练。正式设计要求 ID base 只训练一次，Stage-1/2/3 各自从同一 immutable ID ckpt-10000 独立收集、匹配预算和训练。**

## 1. 目标

将 StackPyramid 建成典型的多阶段 OOD benchmark。每个阶段 OOD 都是正式实验条件，用于检验 gate 能否在不同任务进度上定位缺失能力，而不是从多个阶段中挑选最容易得到好结果的一项。

## 2. 冻结任务语义与阶段

本计划采用三块积木的金字塔任务：机器人先抓取红块并把它放到绿块旁边，使红块和绿块形成底座；随后抓取蓝块，将其放到红绿底座上方，完成金字塔。三个有序阶段定义为：

1. **Stage 1：获取红块**。机器人接近、抓取并抬起红块。
2. **Stage 2：构建红绿底座**。机器人把红块运输到绿块旁边并正确放置。
3. **Stage 3：放置蓝块**。机器人抓取蓝块并将其放到红绿底座上方。

正式实现前必须用实际 ManiSkill 环境和 oracle smoke 核实颜色、成功几何和上述执行顺序。核实后的物体名称、ID/OOD 位置范围、阶段 predicate、成功条件、failure reason、oracle 接口、base checkpoint 和 norm 全部写入 `task_spec.json`。本轮已完成 paired-reset 审计、20/20 四 split oracle gate 和正式采集；旧的官方单轨迹失败记录只保留为诊断证据。

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
3. 使用同一 base policy 和冻结的该 Stage OOD seeds，分别收集四组 OOD expert 数据：
   `Internal-Feature PCA`、`Diff-DAgger`、`Failure-Recovery`、`Offline BC`。
4. 前三种 gated 方法在冻结的 `ID / 当前 Stage OOD` 严格交替 raw stream 上运行，各自停止于
   100 条成功且发生真实接管的 trajectory；不设置 50/50 accepted 配额。由于 ID base 已掌握 ID，
   正式 accepted 数据应显著 OOD-dominant。Offline BC 不走 mixed gate，直接收集完整当前 Stage
   OOD oracle trajectory。不得把 PCA suffix 复用于 Diff-DAgger，也不得用完整 oracle trajectory
   冒充 gated suffix。
5. 以低层 expert actions 选择四组共同可达预算；四组分别使用 `128 ID + 本组 OOD expert`
   的 `1:1 source-balanced` 数据，从相同 `ckpt-10000` 独立更新。
6. 在共同 checkpoint 上，对四个更新后策略分别运行 100 ID、100 对应 Stage OOD 评测。
7. 最后增加混合 ID/Stage-1/Stage-2/Stage-3 部署流，检验 gate 的实际数据选择。
8. 另行运行该阶段的受控 timing sweep，并使用 matched expert-action budget 训练各 timing
   condition；该 sweep 不替代上述四方法比较。

### 每个 Stage 的方法完成矩阵

| 方法 | OOD 收集 | 独立训练 | 100 ID + 100 OOD 评测 | 当前状态 |
|---|---|---|---|---|
| Internal-Feature PCA | 成功 expert suffix | ID + PCA suffix | 必须完成 | Stage 1/2/3 已完成 |
| Diff-DAgger | Diff 信号触发后的成功 expert suffix | ID + Diff suffix | 必须完成 | 待完成 |
| Failure-Recovery | 语义失败后的成功 recovery suffix | ID + recovery suffix | 必须完成 | 待完成 |
| Offline BC | 完整 OOD oracle demonstrations | ID + offline OOD demos | 必须完成 | 正式 oracle 数据已有，预算选择、独立训练和评测待完成 |

只有三个 Stage 的四行全部完成，StackPyramid 的“四方法 robot-gated DAgger 比较”才算完成。
当前已有的三组 Bridge-PCA policy 只是矩阵第一行在三个 Stage 上的结果，不是完整任务。

## 7. 产物结构

当前正式采集根目录：`/mnt/data/ask4help/results/xvla_stackpyramid_formal_collection_v2/`；其下按 `id`、`stage1_ood`、`stage2_ood`、`stage3_ood` 保存：

- `task_spec.json` 与 seed manifests；
- base-policy rollouts、状态 snapshots 和 nominal oracle trajectories；
- detector、collection、training 与 evaluation；
- 每阶段 summary，以及跨阶段总表。

本轮采集验收：ID `128/128`，三个 Stage OOD 均 `100/100`，所有 split raw attempts 均严格成功，视频分别为 `138/110/110/110`。H5 loader 已验证所有 action-bearing observation 均作为 anchor，ID 128 条轨迹共 `10579` anchors，尾部 anchors `1152`，最终 observation 恰有 1 个有效 target；2-step checkpoint reload/forward smoke 已通过。

本轮只完成了 Internal-Feature PCA 这一方法分支在三个 Stage 上的 stage-localized gated
DAgger。Diff-DAgger、Failure-Recovery 和 Offline BC 的对应分支仍需补齐；完整多介入时机
sweep 也需单独建立 matched timing conditions，不能由本轮 DCA/EAS/DCE 诊断代理替代。

## 8. 当前基线结果

固定 `ckpt-10000` 的纯 policy 评测使用 100 条 seeds per split、
`execute_horizon=5`、`max_episode_steps=250`。结果为：ID
`ever_grasped=100/100, strict_success=95/100`；Stage-1 OOD
`52/100, 0/100`；Stage-2 OOD `99/100, 0/100`；Stage-3 OOD
`0/100, 0/100`。四个 split 的 400 个评测视频和完整 summary 保存在
`/data/zhaozhixuan/xvla_stackcube_data/stackpyramid_baseline_eval_ckpt10000_v1/`。

## 9. Internal-Feature PCA 分支的修正版结果

当前已完成分支使用固定 ID base policy `ckpt-10000`、`assets_retry2/bridge_pca.pt` 和 ID25
校准阈值 `0.95207279920578`。三个 stage 各收集 100 条非空 expert suffix；收集根为
`/data/zhaozhixuan/xvla_stackcube_data/stackpyramid_gated_dagger_v1/bridge_pca_collection_v3/`。

| 条件 | Ever grasped | Base completed | Strict success |
|---|---:|---:|---:|
| Stage 1 ID | 100/100 | 92/100 | 90/100 |
| Stage 1 OOD | 28/100 | 14/100 | 11/100 |
| Stage 2 ID | 100/100 | 99/100 | 95/100 |
| Stage 2 OOD | 95/100 | 9/100 | 4/100 |
| Stage 3 ID | 100/100 | 100/100 | 97/100 |
| Stage 3 OOD | 100/100 | 100/100 | 90/100 |

三组 source-balanced 训练均完成 `ckpt-500/1000/1500/2000`，最终 loss 分别为
Stage 1 `0.0249633`、Stage 2 `0.1409179`、Stage 3 `0.0281246`。修正版训练输出为
`/data/zhaozhixuan/xvla_stackcube_data/stackpyramid_gated_dagger_v2/bridge_pca_training_v2/`，
正式评测输出为
`/data/zhaozhixuan/xvla_stackcube_data/stackpyramid_gated_dagger_v2/bridge_pca_postprocess_v2/`。
六份 100-episode summary、600 个评测视频、`comparison.json` 和
`PIPELINE_COMPLETE` 均已验收。

修正版训练的 temporal-mask 审计显示：每组为 `228=128 ID+100 expert` 条轨迹；ID 有
`10576` 个 anchors，其中 `1152` 个是尾部 anchors；expert 尾部各有 `900` 个 anchors。
最后观测保留且恰有 1 个有效 target timestep，真实动作维度为 8，未过滤 episode 尾部。

旧目录 `bridge_pca_training_v1/bridge_pca_postprocess_v1/` 仅作诊断保留：其训练误把
原始 H5 中额外的 10 条 ID 轨迹纳入，使用了 138 条而非 PCA 资产对应的 128 条 ID，不能作为主结果。
Timing 目录中的 DCA/EAS/DCE 仍是诊断代理；完整 paired state-snapshot timing utility 和多时机
对照仍需单独实验。

以上结果只代表四方法矩阵中的 Internal-Feature PCA 分支。它可以作为已核实的中间结果，
但在 Diff-DAgger、Failure-Recovery、Offline BC 三个分支补齐前，不得称为 StackPyramid
完整主实验或完整 baseline comparison。

## 10. 运行位置

优先复用 5090 上已核实的 X-VLA 环境。若 5090 在我们的配额内没有足够空闲卡，不等待卡位，直接按《环境与存储》的回退规则在阿里云 H20 建立原生 X-VLA 环境、同步冻结资产并完成 smoke。两处运行必须使用相同源码版本、数据、base checkpoint、norm、seed manifest 和成功定义。

## 11. 四方法 gated-DAgger 对照执行记录

本节是文章和后续 agent 共用的执行记录。该对照由
`codex-stackpyramid-four-method-comparison` 接管；另一个 agent 的独立多介入 timing sweep 不属于本节，不能互相替代。

当前持久化总控输出根：
`/data/zhaozhixuan/xvla_stackcube_data/stackpyramid_gated_dagger_four_method_comparison_v1_retry1_retry2/`。
它按 `calibration -> 每个 Stage 的四组收集 -> common expert-action budget -> 2-step smoke -> 2000-step training -> 100 ID/100 对应 Stage OOD evaluation -> summary` 顺序推进；总控状态在 `pipeline_state.json`，活动阶段日志在 `logs/`。

冻结协议：

- 同一 immutable ID base `stackpyramid_id_sft_10000_v1/formal_id_sft/ckpt-10000`，ID 数据为已验收的 128 条 asset subset。
- 每个 Stage 使用同一组交替 raw attempt seed：偶数 attempt 为 ID，奇数 attempt 为该 Stage OOD。前三种 gated 方法各收 100 条成功且发生真实接管的轨迹，只保留各自 gate 触发后的成功 suffix，不设 ID/OOD accepted quota；预期结果必须显著 OOD-dominant。Offline BC 单独收集 100 条当前 Stage 的完整 OOD oracle trajectory，不参加 mixed-stream data-selection 统计。
- Internal-Feature Bridge-PCA 使用已验收的 PCA asset 与固定阈值 `0.95207279920578`；Diff-DAgger 使用独立成功 ID calibration 的 `q=.95` trajectory-maximum threshold、16 个 diff timesteps；Failure-Recovery 固定在 50 个低层 policy steps 后接管。
- 四种方法收集完成后，以四组各自可达的最大共同 expert action 总量选择子集；训练仍保持 `128 ID + 本组 expert` 的 1:1 source-balanced sampler、正确 temporal action mask、batch 8、2000 steps、每 500 保存。
- 评测统一使用 checkpoint-2000、纯 policy、100 条固定 ID 与 100 条对应 Stage OOD、`execute_horizon=5`、`max_episode_steps=250`；主表同时保留 ever grasped、base completion 和 strict success。

当前状态（2026-08-15）：历史正确 mixed-stream 结果显示 Internal-kNN=96 OOD/4 ID、Diff=69/31、Failure-Recovery=75/25；而当前 PCA=49 ID/51 OOD，不能视为正常 data selection。旧 Stage 1 Diff retry 使用未通过审计的阈值，142 次 raw attempts 无 accepted，已停止并写入 diagnostic marker；所有旧 collection、日志和 partial H5 均保留，禁止进入预算对齐和训练。修正版必须先完成 ID、Stage-1/2/3 OOD 各 100 次 Oracle 审计，确认 ID base policy 已掌握且三个 OOD 存在受控缺口；PCA asset 只能由固定 ID expert 构建，PCA/Diff 都必须用独立成功 ID policy rollout 做 q=.95 校准。之后每个 Stage 才能按相同交替 raw stream、无 accepted split quota 收满 100 条，且 accepted 明显 OOD-dominant；否则在审计门停止。三组训练均从同一 ID `ckpt-10000` 独立重置 optimizer，不得串行续训。四方法收集、训练、评测和 `comparison.json/.md` 完成前，本任务仍标记为“进行中”。

后续代码审计已定位到更具体的 calibration 缺口：现有 PCA asset 来自固定 ID expert，但其正式阈值
没有在独立且成功的 ID policy rollout 上重新校准；旧 Diff retry 也出现大量 raw attempts 而没有
真实触发。旧 Diff 采集及自动 retry 必须停止并保留为 diagnostic。修正版只有在四 split
oracle/base-policy gate、PCA 独立成功-ID calibration、Diff 独立成功-ID calibration 和逐 split
funnel 输出全部通过后，才允许启动正式 collection；PCA pilot 若仍非明显 OOD-dominant，不得训练。
