# X-VLA StackCube 目标位置 OOD 与受控介入计划

**状态：首轮 cohort 已完成审计；retry1 因服务器 CPU affinity 启动包装问题保留为故障记录；retry2 正在正式 collection。** 本计划记录已审核的第一组 stage-localized OOD takeover-timing study。

## 1. 研究问题

固定已掌握 ID 抓取能力的 X-VLA StackCube base policy，只改变抓取后的目标位置，比较不同专家介入时机如何改变专家数据及更新后的 OOD 成功率：

`Takeover Timing → Expert Data Characteristics → Downstream OOD Success`

## 2. Target-Position OOD

- 新增独立环境 `RLinfStackCubeTargetOOD-v1`，不修改现有 ID 和旧 OOD 环境。
- 红块为被抓取的 cube A，绿块为目标 cube B。
- 对同一 paired seed，OOD 中红块位姿与 ID 完全相同；绿块关于红块镜像到相反目标区域。
- 红绿块距离仍保持 8--10 cm，只改变抓取后的运输方向和放置目标。
- metadata 记录 paired seed、两块位姿、相对方向、距离和 `ood_factor=target_position`。

实现时用镜像关系直接验证“只移动目标”，不要依赖容易混淆的角度正负号。若 oracle 无法在该固定区域达到要求，停止并报告，不静默缩小 OOD。

## 3. 冻结资产与 Seeds

- 唯一 base checkpoint：
  `/data/zhaozhixuan/Ask4Help-airplane-5090/results/xvla_stackcube_v1/temporal_mask_v2/id_sft_from3500_to10000_official_2gpu_retry1/ckpt-7500`
- 原始 ID 数据：
  `/data/zhaozhixuan/xvla_stackcube_data/stackcube_id_128_visual_v1_20260722`
- 已知 base ID 成功率：88/100。
- 分别冻结 benchmark-design、collection、evaluation 三组互不重叠的 seed manifest。
- 四个 timing condition 必须使用顺序完全一致的 collection seeds。

## 4. 四种受控介入时机

1. **Immediate**：reset 后立即接管。
2. **Post-Grasp**：首次满足 `is_cubeA_grasped` 后，在下一个 action-chunk 边界接管。
3. **Post-Lift / Boundary**：保持抓取且红块高度首次达到 `z >= 0.07 m` 后接管。
4. **Failure-Recovery**：完成 lift 后又丢下红块，且未完成堆叠；该状态连续两个决策边界成立后接管。

接管后专家持续控制至任务结束。先用纯 policy rollout 建立共同 seed cohort，要求这些 seeds 均已完成稳定 grasp/lift，随后发生目标相关失败；未到达受影响阶段的 rollout 不进入 cohort。四种条件只在该 cohort 上比较。

privileged oracle 必须支持从未抓取、已抓取、已抬起和已放置四类中间状态继续完成任务。

## 5. 数据、预算与训练

- 保存完整 controller timeline、阶段 predicate、动作、状态、双相机视频、reset metadata、介入步数和失败原因。
- 每个 OOD seed 额外保存 full-oracle nominal demonstration，供 DCA 对齐。
- 每组先建立共同成功 seed pool，再以确定性 subset selection 选取完整 suffix。
- 四组统一使用 `B=2002` 个低层 expert actions；不得截断单条 suffix。该值是最接近原定 2000、同时能由 11-step Post-Lift suffix 与主要 26-step suffix 精确组成的共同可达预算，偏差为 0.1%。首轮 400 条 screen 中有 286 条满足 clean stage-localized failure 条件，因此 retry2 使用其中确定性选出的 250 条共同 cohort，不改变 OOD 定义或专家预算。
- 训练数据为原始 128 条 ID demonstrations 与该组 OOD expert suffix，1:1 source-balanced。
- 从同一 ckpt-7500 初始化并重置 optimizer；action chunk 10，temporal mask。
- per-device batch 8，gradient accumulation 4，最大训练 10000 steps，每 500 steps 保存；默认在 2k、4k、6k、8k、10k 进行共同 checkpoint-selection evaluation。
- episode 尾部必须使用 temporal mask，并通过《训练与持续运行协议》规定的 anchor audit 和 2-step/reload smoke。
- 四个 timing condition 必须在相同 step 上共同评测和停止；满足《训练与持续运行协议》的预设共同早停条件时可以提前结束，否则训练至 10k。不得分别选择各自最佳 checkpoint。

## 6. 评测与验收

- checkpoint 由共同 validation seeds 选定；随后每组使用相同且未参与选择的 100 ID 和 100 OOD held-out test seeds。
- 主结果：DCA、EAS、DCE、OOD SR、相对 base 的 OOD SR 提升、ID SR。
- oracle 在 100 OOD 上成功率至少 95%。
- base policy 在 OOD 上 grasp/lift rate 至少 80%，完整任务成功率不高于 50%。
- 同 seed 满足 `Immediate < Post-Grasp <= Post-Lift < Recovery`。
- 四组均保留完整数据 manifest、训练日志、checkpoint、200 条评测 summary 和视频。

结果根目录使用全新名称 `stackcube_target_ood_timing_v1`，不覆盖现有 X-VLA StackCube 产物。

## 7. 运行位置

优先在 5090 服务器复用现有环境、ckpt-7500 和 ID 数据。如果 5090 在我们的资源配额内没有足够空闲卡，Agent 直接按《环境与存储》的回退规则在阿里云两张 H20 配置 X-VLA、同步已有资产并完成 smoke，然后继续本计划，不等待 5090 释放。当前阿里云持久盘尚未发现 X-VLA 环境或 checkpoint；**不得复制 5090 venv，也不得在阿里云重新训练 base checkpoint。**

## 8. 实现状态

`tools/run_xvla_stackcube_stage2_pipeline.py` 已对齐四个 timing condition，并以共同 OOD cohort、完整 suffix 的 `B=2002`、10k 上限、每 500-step checkpoint、共同 checkpoint selection，以及最终每组 100 ID/100 OOD 评测推进整条流水线。首轮 screen 的整体 lift rate 不能代替 cohort admission；正式 cohort 只按稳定 lift 后的目标相关失败候选筛选。启动前物理 smoke 已验证 Post-Grasp 与 Post-Lift 能从 policy 中间状态继续完成任务；Failure-Recovery 的迟触发样本允许因剩余 horizon 不足而失败，并只将成功恢复轨迹纳入共同 seed intersection。retry1 仅保留启动失败记录，retry2 使用按实际可用 CPU 自动分片的调度器重新启动。
