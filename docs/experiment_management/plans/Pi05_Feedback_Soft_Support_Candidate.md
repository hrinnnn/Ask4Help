# Soft-support候选：在新seed验证前固定

## 已观察事实

原displacement_v1的OpenDrawer两stage各20raw/arm完整结束、通过独立审计，全部真实接管时间相同。Grasp matched325的TASR×3=60/325两组相同；Goal matched708为203/708两组相同。

源 `opendrawer_stage_feedback_eval_v2/support_diagnostic.json`：至少已有2条历史cue时，Grasp1019个query中仅1个有两个半径内邻居，Goal811个query中为0。第二近邻距离中位数为0.966/1.011，而原截断半径为0.47564。可见反馈方向并非完全没有生成，但原支持条件极少让它影响gate。

## 候选与竞争解释

- 候选：在一个Gaussian带宽处硬截断过于苛刻，丢弃了仍有相似度的历史反馈。放开硬截断、但保持原最低总相似度要求，可能提高反馈覆盖。
- 竞争解释：即使反馈能够激活，Earlier/Later线索仍可能不准确，特别是失败专家段也可能产生Later；当前论文明确允许所有eligible段产生反馈，本候选不偷偷修改这项规则。
- 另一边界：20raw中后期才积累足够反馈，冷启动可能掩盖变化；本次固定40raw/arm，留出更多后半段，不根据效果提前停止。

这些是根据原零结果提出的探索性修订，不是事先预测，更不是已经证明有效。

## 唯一更新变化

保持原ID校准带宽sigma=2*radius、gamma_B、PCA、M2、q_e/q_R、lambda2、beta0.5、vote0.25和一次最多5-action postponement不变。

对所有过去的独立intervention使用相同Gaussian权重，不在distance=sigma处硬截断；至少两条不同intervention，并要求总权重
`sum(w) >= 2*exp(-1/2) = 1.2130613194252668`。
该下限直接取自原规则“两条半径内邻居”的最低总权重，不根据TASR/SR重新拟合。单条完全相同的反馈加很多极远反馈仍不能达到下限；远状态还受Gaussian衰减和原ridge正则抑制。

固定历史上的支持诊断显示可激活query数约23/52，但历史反馈内容在新闭环中会变化，因此不能把此数当作新timing/TASR收益。

## 独立验证合同

- Backbone：pi0.5，OpenDrawer原eval模式，两stage同一已审计ID gate。
- 原failed/zero结果全部保留，候选另命名support_mode=soft_mass。
- Grasp新stream seed1784000、Goal新stream1785000，各arm固定40raw，严格ID/OOD交替；每组从空记忆开始。它们不与已用1782000/1783000的旧on/off、1781000–49 ID校准或1786000–29 Goal nominal参考重合。
- fixed对照仍不使用反馈；候选保持原displacement_v1 cue，不叠加Plane的commitment_v2。
- 同样的whole-successful-suffix预算规则、TASR×1/×2/×3；报告misses、实际专家成本、ID/OOD组成，不删除负例。
- 首先判断真实timing是否改变；再看TASR是否朝有利方向变化。若仅激活增加而TASR不升，不能说反馈有效。最终学习结论仍需独立post-SFT SR。

### 区分冷启动与支持规则

在读取新soft反馈结果之前，追加登记同样1784000/1785000、40raw的hard_radius反馈控制。复用各stage的新fixed数据，不额外生成fixed；各反馈arm仍各自空记忆。这样可区分“40raw本身使原规则开始有效”与“soft support提供额外作用”。不得只报告较好的反馈版本；最终SR验证仍须使用明确冻结的版本。

假设与检验按Hypothesis Generation skill分开记录；工具归属已在Scientific_Skills_Use_20260909.md记录，不作本方法效果证据。
