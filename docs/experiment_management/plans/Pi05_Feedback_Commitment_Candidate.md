# 饱和夹爪纠正线索：独立候选，不覆盖v1

## 观察与边界

原v1 Grab Plane pilot无时机或TASR变化。12个intervention中11个cue=Keep；569个query仅15个有至少两条邻居，只有2个非零调整。逐动作检查显示：多数上一块gripper command约−0.97、实际宽度变化为0，专家第一块却张开约79mm，且自由动作MSE约0.5。旧规则依赖上一块宽度变化方向，因此闭合已饱和后无法识别重新张开这一纠正。这是已观察到的测量盲点；它是否解释性能不足仍待验证。

## 候选与竞争解释

- 候选A：用之前的关闭指令和专家真实张开量，可以补回纠正线索。预测：在未参与开发的新seed上，部分原Keep变Earlier，产生真实接管变化。
- 竞争解释B：邻域支持太少，即使改cue也不会改变gate。检查support和实际decision flip，而不只报告更多Earlier标签。
- 竞争解释C：提前不是更有价值的时机。即使有flip，TASR、真实专家成本或后训SR也可能不升反降；这些结果必须保留。

## 预先固定的新增量

新增opening score为
`max(0, -mean(previous policy gripper commands)) * max(0, mean(first expert gripper commands)) * max(0, expert width after 5 actions - takeover width)`。
只有当前自由生成动作与专家的差异超过既有ID参考时才使用该线索。阈值为成功ID示范上该联合量的q95，并设10mm下限（原gripper归一化尺度的一单位）。不读取OOD训练/测试来选择阈值。

旧实际位移反转线索保持优先；普通低差异释放不自动标为Earlier。新增Earlier仍归因到真实的上一policy query。PCA、原gamma、lambda2、beta0.5、radius×2、至少2条支持和wait最多5动作均不变。

## 验证

- 新候选和旧v1分别命名、分别保存。旧StackCube的微弱结果和Plane零结果均保留。
- 新冷启动比较使用之前未采集的1772000/1773000系列seed，各stream固定/候选两arm20个ID/OOD交替episode。参数在这些新结果产生前冻结。
- 先验收实际时机和TASR，成功/失败专家成本均报告。若没有改善，不把这个候选选作已验证的新方法。
- 这是基于development失败分析提出的探索性修订，不冒充事先预测，也不改变仍待用户确认的训练composition门槛。

本记录按Hypothesis Generation skill区分观察、候选、竞争解释与可反驳预测。没有额外失败标签或人工最佳时间进入gate。
