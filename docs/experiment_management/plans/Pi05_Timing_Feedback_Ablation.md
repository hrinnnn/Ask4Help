# π0.5 Timing Feedback 消融

Owner：`01a07faa-682a-7e01-9d5b-eeac5b96864d`。用户于2026-09-09启动新Goal。

2026-09-09追加决定：用户明确允许用OpenDrawer的不同OOD stage作为两项实验条件；交付顺序调整为先实际Timing、再新采集数据的TASR，然后把同预算训练持久化后台运行并补SR。前两项不能替代最终SR目标。现有5090全部GPU仍占用，H20无OpenDrawer输入，需只恢复缺失的同任务输入，不使用其他任务checkpoint替代。Goal-OOD若前置抓取能力不足，报告该问题，不把它称为干净的后段timing因果实验。

## 要回答的问题

在相同任务、基础策略、PCA和初始阈值下，使用已完成专家接管的时机反馈，能否改变后续接管数据，并在相同保留专家动作预算下提高更新后策略的OOD成功率？新实验不预设会提高，也不把原固定gate的主表改名为有反馈结果。

当前本地Method依据为 `outputs/method_minimal_revision_20260908/完整修订稿.tex`：正常生成动作与专家的下一执行块比较，纠正前序运动提供Earlier，低差异前缀提供Later，即时差异而无纠正提供Keep；局部kernel方向调整阈值，一次wait最多5动作，episode间提交记忆。旧X-VLA诊断结果不进入本次π0.5数据或效应估计。

## 任务与对照

主任务先固定为StackCube原任务OOD和Grab Plane yaw OOD。旧Stage2 controlled-timing无效结果禁止使用。StackCube原OOD不宣称是已通过阶段局部性的单因素任务。飞机主要终点沿用用户指定的ever_grasped，另外保留strict完成率，采集成功规则与原任务保持一致。

OpenDrawer作为第三项资格候选，不按效应大小选择。现有v9记录存在preliminary ID base和Grasp/PCA采集门禁失败；不能因本次想要正结果就绕过。先核实新资产和现有owner的真实完成状态，再决定能否采用合法Goal-OOD或Grasp-OOD。不修改或重启其他owner的pipeline。

两组为fixed Bridge-PCA与feedback Bridge-PCA；不是再次跑四种主实验方法。两组同初始gate，feedback空记忆，因此第一episode必须完全一致。固定组也记录同样的诊断信号但不提交或使用反馈，避免日志开销差异影响比较。

## 分阶段执行

1. **资产/运行时预检**：独立核对两服务器checkpoint、norm、数据、PCA、仿真和训练入口。H20根盘已满，不能默认训练日志可写；先验证独立/tmp与OSS路径。禁止重新训练已有base或删除他人文件。
2. **配对smoke**：新小规模ID/OOD场景，正常生成、oracle、图片、动作缩放、随机数和状态恢复验收。只验证实现，不据其SR选超参数。
3. **ID校准**：PCA来自原ID示范；初始阈值来自独立成功ID policy trajectory maxima的q95。动作差异与纠正尺度分别以ID示范q95校准。成功ID校准不足就补ID，不读取OOD测试来选阈值。反馈只需完整5动作目标；SFT仍是完整10步chunk加有效mask，两个窗口不可混淆。
4. **开发检查**：先每task两组20个alternating episodes，检查cue数、支持覆盖、是否实际改变takeover和是否错误重复wait。修复只针对实现契约；改变方法需要新版本和新的独立确认数据，不覆盖原结果。
5. **正式采集**：3个独立stream replicate，每arm严格ID/OOD交替至100个成功真实接管后缀或2000raw上限。每stream独立空记忆；实验独立重复单位是stream/更新后的policy，而非一个stream中的所有帧。两组保留自然composition；执行专家总成本包含失败和丢弃段。
6. **同预算SFT**：每task在所有arm/replicate间用完整成功后缀求共同可达动作预算，规则在看SR前固定，不截episode、不按时机挑选。原ID:new专家1:1，原norm，所有真实anchor和尾部mask。StackCube固定2000更新步、Plane固定2500更新步；每500保存，最终共同步数为主，不逐组挑最好checkpoint。先每arm完成2-step/reload/mask检查。
7. **独立评测**：每更新策略100ID+100OOD，同task配对seed。报告3个独立运行的SR和范围、每组分子分母、ID保留、成本和真实timing变化。旧历史SR只是背景，不作同期对照。
8. **论文交付**：一张task × feedback off/on的简表，主列OOD SR、ID SR和相同专家预算；timing/专家总成本放辅助列或补充表。只据完整结果写LaTeX。零/负结果照报；未显示优势时不写“证明有效”。

## 关键判定

- 当前query不能读取本episode稍后的专家动作；专家段结束后才产出一个cue，供后续episode使用。
- 纠正证据优先；attribution是实际存在的上一个policy query，不是虚构更早状态。
- 同初始阈值/同budget下的反馈增益不等于全局最佳takeover已被证明。
- 小规模采集仅是机制诊断，不能替代后训未辅助SR。
- 保存所有arm和所有replicate。不可因negative结果删除task或反复挑测试seed。
- `PIPELINE_COMPLETE`要求至少两task的采集、同预算训练、正式评测和论文段落均完成；设计或smoke不算结束。
