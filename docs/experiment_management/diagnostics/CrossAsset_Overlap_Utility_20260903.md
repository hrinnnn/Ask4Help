# 跨资产轨迹重合度与 Downstream Utility

后续更新：当时缺少的12个π0.5三任务分数现已全部恢复并完成独立审计，见 [Pi05_TASR_Final_20260903.md](Pi05_TASR_Final_20260903.md)。下文是较早的全资产盘点快照，不能继续将这些π0.5条目标为未计算。

日期：2026-09-03；执行：本地CPU、两服务器只读导出；没有训练、GPU回放或原pipeline修改。

## 交付与范围

主产物：`artifacts/cross_asset_overlap_utility_20260903/完整结果与视频.md`。
机器可读全表：`all_results.json`；详细匹配/相关性：`comparison.json`、`stage2_overlap.json`。

本轮定位43条件/方法：21条件计算Q×1/×2/×3并与SR连接；22历史gate条件已核对SR，但缺动态物体位置/接触记录，Q保留为空而非0。未声称全量指标验证完成。

已有StackCube/Grab Plane固定网格274条轨迹复用原参考、对齐、接触、分母，只按1/2/3倍放宽原半径；OpenDrawer沿用262条敏感性分析。另导出673条Stage2数值轨迹（含250条immediate成功参考池）。匹配单位保留位置0.02m、姿态15deg、开合0.01m；不按SR再拟合。

Stage2目标是现有Oracle完成close/lift后的transport与release，明确不复用Stage1的grasp片段。由真实动作符号、5步chunk边界、既有0.07m lift转换阈值重建phase。坐标为当前绿色目标中心+本条稳定抓取TCP轴，不冒充完整物体姿态测量。采用一个nominal参考、局部单调DP，当前接触条件必须相同；每个查询seed从参考/校准中排除。新Stage2定义本身也是探索性扩展，非事先冻结协议。

## 条件内相关性

| 分组 | 条件数 | ×1 Pearson / Spearman | ×2 | ×3 |
|---|---:|---:|---:|---:|
| X-VLA StackCube固定网格 | 5 | 0.570 / 0.667 | 0.703 / 1.000 | 0.850 / 1.000 |
| X-VLA Grab Plane固定网格 | 4 | 0.689 / 0.800 | 0.672 / 0.800 | 0.616 / 0.800 |
| OpenDrawer 5000-step probe | 6 | 0.552 / 0.429 | 0.686 / 0.486 | 0.726 / 0.771 |
| X-VLA StackCube Stage2时刻 | 4 | 0.731 / 0.200 | -0.475 / -0.800 | -0.800 / -0.800 |

Stage2 gate：Internal PCA Q×3=115/434=0.264977、OOD SR=74/100；Diff Q×3=82/434=0.188940、OOD SR=45/100。只有2种方法，不把两点相关当作证据。

## 反例与解释边界

Stage2时刻各组实际新增预算1968。×3时immediate/post-grasp/post-lift/recovery的Q依次0.2226/0.3095/0.4345/0.1946，但OOD SR为0.52/0.10/0.04/0.30。逐episode复核显示，曾抓住方块分别为92/100、13/100、15/100、85/100。后两种post-*模型多数失败早于transport片段。这支持前置技能不足是潜在限制，但不证明具体训练因果。高后段重合度不保证完整任务SR；禁止继续通过事后挑阈值/删条件隐藏这个反例。

固定网格每条件3训练seed×100-ID/100-OOD、2500步；以条件均值计算相关，不把同一Q重复三次当作独立条件。Plane保留旧ever_grasped端点，不能改名为稳定机身抓取SR。OpenDrawer已独立读取6份5000步20-probe逐轨迹summary，SR是Grasp-OOD条件下端到端成功；各条件测试seed块不同，并非配对测试。额外t0/7500步probe6/20不纳入统一5000步比较；正式evaluation根无summary。

PickSingleYCB Object四组的5954专家点、5000训练步和100-ID/100-OOD已有结果；OOD：PCA52、Diff42、Recovery49、Offline48。Diff0.05是低阈值诊断分支，不是原冻结阈值。原收藏缺动态物体/接触状态，仅9D关节状态、动作、视频与reset元信息，不足以恢复同版本Q；没有给出伪精确数值。

## 独立审计

`independent_audit.json`：822次fixedgrid版本逐轨迹核对；673条Stage2的4192个目标点以旋转矩阵角度独立重算，最大距离差1.724e-7；预算1968/434核对通过；111份逐episode评测summary分母及成功数重数；15个着色单视频帧数与pre-action对齐核对通过。Stage2正常参考检查点保留率×1/×2/×3为0.8981/0.9969/1.0000。

导出15个新着色单视频与4合辑；附4个未着色YCB原专家后缀示例。人工查看StackCube、Plane及Stage2关键帧。数值与工程审计通过不等于科学假设通过。

## 待补证据

22条件：旧X-VLA StackCube/Plane四组、π0.5 StackCube四组、π0.5 Plane四组、OpenVLA Plane两组、PickSingleYCB Object四组。

需要恢复**原训练选择清单内专家轨迹**的动态状态。优先逐条小规模验证动作回放与原qpos一致，再补物体/接触记录；此前Plane回放已出现漂移，本轮未盲目批量重跑。无法一致回放时需要原状态快照；新收集轨迹不能冒充旧模型的训练数据。此步骤本身不必重训旧模型。

复现入口：`fetch_cross_asset_utility_evidence.py`、`fetch_stage2_overlap_assets.py`、`inventory_cross_asset_overlap_sources.py`、`analyze_stage2_overlap.py`、`analyze_cross_asset_overlap_utility.py`、`render_cross_asset_overlap_videos.py`、`audit_cross_asset_overlap.py`、`build_cross_asset_overlap_report.py`（均在tools/）。
