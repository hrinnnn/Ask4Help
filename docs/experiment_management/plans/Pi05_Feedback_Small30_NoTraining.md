# Small30：灵敏反馈与检测能力诊断

用户2026-09-09批准，当前明确不训练。精确执行规格见
`configs/pipelines/pi05_feedback_small30_v1.json`，旧父pipeline的SFT next-stage不再适用。

## 固定比较

StackCube原任务OOD、OpenDrawer Goal-OOD分别运行fixed、old、sensitive三组。
每组从空记忆开始，严格ID/OOD交替，最多30条成功且实际接管的后缀；400原始尝试达到仍不足则如实报告，不补样改变分布。
失败干预仍可能给反馈，因此必须同时列出成功后缀数、全部干预数、有效反馈事件数和全部专家动作成本，不能声称只看了30次专家。

旧组保持hard_radius、minimum_support2、vote0.25、5步一次性deadline。
新版连续Gaussian权重，不设置计数/质量/vote硬门槛，单条反馈即有小幅影响；
保留lambda2、beta0.5、ID校准带宽乘2，不设额外等待deadline。
不强制时机移动10步。两任务沿用各自旧cue定义，StackCube commitment_v2，OpenDrawer displacement_v1；只在任务内比较。

## 独立评测

固定新的collection与evaluation seed段，不使用9.1M旧保留测试池。
每任务100条完整自主轨迹，50ID/50OOD；不接管、不更新memory，使用与collection相同原策略运行模式。
按第5/10/20/30条accepted后的历史前缀重建memory，对同一批纯policy特征/分数回放报警，
成功/失败标签采用strict任务成功而不是ID/OOD。报告BA、FPR、FNR与分子分母、变化不确定性；缺任一类别则BA不可计算。
这些是固定memory的被动报警结果，不能冒充真实介入。实际介入另外比较共同raw seed前缀并保留新增/丢失报警和两组未配对尾部。
不根据held-out BA决定更换阈值或选择最好checkpoint。

置信区间按完整reset seed簇重采样，保留同seed的ID/OOD配对；区间条件于本次已学到的memory，不等价于多次独立收集重复。
初始部署47349a02的分析器使用episode bootstrap，最终登记前必须用新版分析器另写
`timing_BA_report_seed_clustered.json`，保留初始报告；不得仅凭旧区间做BA不降低结论。

## 执行与验收

H20 native27相关测试通过；空闲显存仅原PID276925占用338/328MiB，保留不动。
StackCube GPU0/CPU0-3，OpenDrawer GPU1/CPU4-7；原模型/norm/calibration均同机。
5090全部占用，拒绝分配。OSS保存逐episode数据、状态、manifest；临时盘保存可追加运行日志。
控制器是`tools/run_pi05_feedback_small30.py`，重启时读取已有审计完成chunk，不自动覆盖partial。
达到`TIMING_BA_COMPLETE`后挑选提前、推迟、保持、变差的真实视频供人工审查；
只有数据、独立检测结果和定性视频齐备才算本轮诊断完成。禁止自动启动SFT或训练smoke。

OpenDrawer当前基座存在Goal-OOD前置抓取失败，因此本轮只验证gate行为，不宣称干净post-grasp因果机制。
