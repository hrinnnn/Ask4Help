# 过度推迟的机制修订

用户明确要求继续改进，目标是在20–30条收集范围内保持有用提前、避免无依据的长延迟，并最终检查BA；当前仍不训练。
Owner沿用当前任务。旧Small30冻结完成，不改变现有controller或旧测试结果。

## 已核实问题

1. sensitive episode17首次baseline报警在10步，阈值只提高1.48%；15步再次阻止报警。20步后score回落。
   25步direction转为Earlier但score低，原实现遗忘被暂缓的报警，55步才接管。
2. episode7的10–40步主要由历史episode1、3两条Later支持。它们只测开头重叠窗口。
3. 新的完整expert片段前向审计为H20
   `/mnt/data/ask4help/results/pi05_timing_feedback_ablation_v1/later_segment_audit_v1/segment_audit.json`，
   4cases已完成，代码142810d8，未训练。
   episode3 offset10整体MSE0.06207<0.18138，但夹爪MSE0.49471且20%方向不一致；
   episode1 offset10整体MSE0.10138<0.18138，但夹爪MSE0.80031且20%方向不一致。
   这是平均误差掩盖子动作分歧的直接例证，不证明所有Later无效。

## 已实现的可选原型（尚未接入新采集）

- remember_deferred_alarm：保留首次被暂缓的报警，Later支持消失即接管，即使score已回落。
- use_later_duration：每个Later事件只在其已测连续一致步数内有效，从首次暂缓时刻计时，不续期。
- observed_agreement_steps：逐个完整、不重叠5动作窗口检查，整体误差通过且夹爪方向无分歧才累加；遇差异停止。
- 所有新开关默认关闭；25项相关测试通过。不是已完成的闭环修订效果。
- 只加pending报警、保留旧memory的prefix反事实：episode17 55→25；episode7 50→45，
  另两条15→10、35→25。该第一项不充分，因此继续时长证据修订。

## 接下来执行

1. 把4case审计扩展到已有开发collection的全部Later事件，保存每个实际完整专家片段的独立arm/gripper误差与连续一致长度。
2. 用同协议ID参考检查夹爪方向判定的正常切换情形，明确chunk跨开合边界是否应按首个分歧步截断；不要根据旧held-out BA挑阈值。
3. 将原型接入独立collector模式、memory序列化/恢复和独立审计，保存完整的continuation证据。不能只修改gate而忘记生成later_valid_steps。
4. 冻结候选后用全新collection seeds1870000起，先StackCube固定/候选各20raw从空memory做真实配对（工程验证不调整任务）。
5. 若机制契约通过，按用户授权再到每组30成功接管上限；新独立BA seeds1880000起，固定100raw。
   旧Small30 BA仅作旧版本结果，不能充当新版本独立验证。不强制每个task提前，不用GT阶段给gate。
6. 报告实际时机、漏报/误报、全部成本、成功和失败视频；不足或变差照报。未验证前不得称问题已解决。
