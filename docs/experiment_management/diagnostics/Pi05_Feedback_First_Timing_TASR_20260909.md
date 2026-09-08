# π0.5 Timing Feedback：首批真实 Timing / TASR

本表为开发pilot，不是最终消融结论。两个arm从相同空记忆/原checkpoint开始，各20个严格ID/OOD交替raw episode。每task最终独立SFT与SR尚未完成。

## StackCube

| 指标 | 无反馈 | 有反馈 |
|---|---:|---:|
| OOD episodes | 10 | 10 |
| OOD平均接管步数 | 25.5 | 21.0 |
| OOD无接管 | 0 | 0 |
| OOD辅助完成 | 9/10 | 9/10 |
| OOD全部实际专家动作 | 254 | 259 |
| ID全部实际专家动作 | 138 | 138 |
| 成功后缀自然组成 | 3 ID + 9 OOD | 3 ID + 9 OOD |
| TASR比较的完整后缀共同预算 | 296 | 296 |
| TASR ×1 | 94/296 = 31.76% | 94/296 = 31.76% |
| TASR ×2 | 115/296 = 38.85% | 114/296 = 38.51% |
| TASR ×3（现有主容差） | 122/296 = 41.22% | 123/296 = 41.55% |
| 新后训未辅助SR | 未运行 | 未运行 |

2/10 OOD发生提前：seed1761008的35→10，seed1761009的40→20；其余8条不变。ID接管4条、未接管6条，在两arm中完全一致。每一配对样本在首次实际接管之前的自主动作最大差均为0，因此不是不同随机轨迹造成的表面时间变化。

TASR使用原同task成功OOD参考、同一物理尺度和冻结容差；ID只计分母。两组共同完整后缀预算296，不截短episode；选中相同11个episode。重新由逐点距离/接触记录计数，分母与三个分子均通过独立重数。未评分点0。

**解释**：时机确实可改变，但主容差仅多1个相容anchor，较严格容差没有一致改善，专家成本还增加5动作。因此不能写“已经验证稳定提升”，也不能把辅助9/10写成后训练成功率。仍需要其他条件、较完整collection和真正同预算SFT。

### 训练门槛

75% OOD低于旧80%经验门槛。已经请求用户决定：本次ablation是否允许保留自然composition、匹配expert-action budget并完整披露。不改阈值凑比例，不擅自放开旧pipeline。未收到例外授权前，SC训练保持未启动；Timing/TASR和其他task诊断继续。

## Grab Plane

ID校准32专家demo+50policy reset完成、独立校验通过。正常t0专家smoke两条均成功。首个真实gate pilot发现旧planner会在晚接管时继续越过250-action episode horizon；另有未到达goal时的inf诊断值导致JSON写出失败。原root完整保留为无效诊断。

新版本明确执行成功/终止/时长边界，只在planner诊断字段把inf保留为显式文本。late-smoke t140后只执行110专家动作，250终止且不伪报成功，原始数组审计通过。新配对pilot在独立development_v2_horizon根运行，仍沿用原seed、checkpoint、PCA、反馈参数和任务时长。

## OpenDrawer

原native5000权重、norm已恢复至H20。环境/spec源码与5090和本地主工程逐字一致，未修改几何、相机或success。Grasp-OOD的ID/OOD五动作smoke均通过：384双图、10×8动作、2048Bridge、重复生成差0；已查看RGB。它只说明恢复链路可运行，不是新的ID资格或反馈效果。

尚需恢复ID/PCA参考、接入原expert并做stage-specific collection。原preliminary ID61/100状态保持，不把它改名为ID>=80%的正式base。

## 证据入口

- H20总根：`/mnt/data/ask4help/results/pi05_timing_feedback_ablation_v1/`。
- SC：`development_v1/stackcube_legacy_ood/{paired_timing_report.json,pilot_TASR.json}`及两arm的独立数组审计。
- Plane当前：`development_v2_horizon/airplane_yaw_ood/pipeline_state.json`；旧development_v1不混入。
- OpenDrawer：`opendrawer_runtime_smoke_v1/grasp/`。
- 本地：`artifacts/stackcube_feedback_paired_timing.json`、`artifacts/stackcube_feedback_pilot_TASR.json`及`artifacts/opendrawer_feedback_weight_restore/`。

本回合分类为progress：取得首批真实配对Timing/TASR，修复并重启Plane边界问题，完成OpenDrawer模型/源码恢复smoke。全Goal仍未完成，没有SFT/最终SR。
