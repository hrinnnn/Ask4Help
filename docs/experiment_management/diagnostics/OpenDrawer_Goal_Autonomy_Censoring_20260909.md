# Goal-OOD自主复跑：当前批次确有前置抓取失败

使用与40raw gated实验完全相同的20个reset seed（1785000–1785019），分别运行ID和Goal-OOD；同checkpoint、完整prompt、norm和eval推理模式，不使用gate或expert，直到原任务终止或400动作上限。

| 条件 | 完整episode | 自主抓起 | 自主抬起 | 严格任务完成 |
|---|---:|---:|---:|---:|
| ID | 20 | 11/20 | 11/20 | 7/20 |
| Goal-OOD | 20 | 0/20 | 0/20 | 0/20 |

独立审计验证：每条复跑在原gate边界之前的动作与gated数据相同，最大差为0。Goal-OOD不是因为被提前接管后看不到后续抓取；即使完全不接管，到episode结束也没有抓起。

## 能说与不能说

- 能说：在当前基座及这20个配对开发seed上，目标位置OOD已经伴随上游抓取能力损失，因此不能把当前结果解释为干净的“完成抓取之后，再比较放置阶段takeover timing”。
- 不能说：所有Goal-OOD任务或所有policy都存在这个问题；也不能把这20条开发复跑叫作最终独立SR评测。
- 这些数据是真的，不删除；保留其原Timing/TASR、专家成本和失败模式。受限制的是阶段因果解释，而不是原始记录的真实性。
- 当前Goal TASR只计transport/place/release，不能据此声称覆盖了这一模型实际需要的全部OOD纠正行为；本轮不事后扩大target定义来制造收益。

最终9,100,000系列测试seed没有使用。后续继续在原主要任务StackCube和PickPlane上复现已冻结的commitment_v2规则；不因本结果临时修改OpenDrawer checkpoint、几何条件、阈值或成功规则。

证据根：`/mnt/data/ask4help/results/pi05_timing_feedback_ablation_v1/opendrawer_goal_autonomy_diagnostic_v1/`，包含全部ID/OOD逐episode轨迹和`autonomy_censoring_audit.json`。
