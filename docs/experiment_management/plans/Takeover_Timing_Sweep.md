# Takeover Timing Sweep 计划

**状态：已接管未完成的完整 timing sweep。共性协议已定义；具体 seeds、预算和 stage boundary 随任务冻结。第一轮 Bridge-PCA 在线 timing 诊断已存在，但不替代受控的四条件 sweep。**

**接管审计（2026-08-15）：** 原 StackPyramid timing 控制器及历史 `bridge_pca_postprocess_v2`、训练和采集 PID 均已退出；服务器未发现该任务的活动进程。现有 `timing/stage*_ood/summary.json` 只记录在线 Bridge-PCA 采集的 DCA/EAS/DCE 诊断，尚未生成 Immediate、Previous-Stage Complete、Capability Boundary、Failure-Recovery 四种受控条件及其 matched-budget 更新结果。后续启动前必须重新完成 stage boundary、paired seed、共同预算和 temporal-mask smoke。

## 1. 目的

在同一个 OOD、同一个 base policy 和相同专家动作预算下，单独改变接管时机，验证：

1. timing 是否改变专家数据的形态；
2. DCA/EAS/DCE 是否能诊断这些差异；
3. 这些差异是否对应更新后策略的 learning utility。

这些指标目前好像还没有实现得非常好，你也需要实现
## 2. 受控条件

每个 stage 至少包含四个核心条件：

- **Immediate**：完整专家演示，代表明显偏早介入。
- **Pre/Previous-Stage Complete**：完成前一已掌握阶段后介入。
- **Capability Boundary**：即将进入目标 OOD 阶段时介入。
- **Failure-Recovery**：发生可识别的语义失败后介入。

若环境支持 state snapshot/fork，再增加 boundary 前后各一个 action-chunk 的局部 sweep。Internal-PCA 和 Diff-DAgger 作为在线 gate 单独映射到上述受控 timing 曲线上，不与人为固定时机混为同一种条件。

## 3. 公平比较

- 所有条件使用同一组 paired OOD seeds 和同一 base checkpoint。
- 从共同的纯 policy rollout 或 snapshot 分叉，确保介入前状态一致。
- 接管后专家持续到任务成功；数据只保留成功 expert suffix。
- 以低层 expert actions 匹配预算，不按轨迹条数匹配。
- 使用完整 suffix 做确定性 subset selection，不截断 episode。
- 所有更新从 base checkpoint 重新开始，并使用完全相同的训练配置。

## 4. 自动化介入诊断

对每个 OOD seed 保存一条或多条 nominal oracle trajectory。将 expert takeover 起点匹配到 nominal trajectory 的最近任务状态，再对 completion suffix 做 task-state DTW。

连续 Direct-Completion Alignment 定义为：

\[
A_i = \exp\!\left(-D_i^{\mathrm{nom}}/\sigma_{\mathrm{nom}}\right),
\]

其中 `D_i^{nom}` 是与同 seed nominal completion 的最小归一化 DTW 距离，`sigma_nom` 由成功专家轨迹之间的自然差异确定。

Expert-Action Saving 定义为：

\[
E_i = \max\!\left(0,1-\frac{N_i^E}{N_i^\star}\right).
\]

综合指标暂定为 Direct-Completion Efficiency：

\[
\mathrm{DCE}_i = \frac{2A_iE_i}{A_i+E_i+\epsilon}.
\]

正文必须同时报告 DCA、EAS 和 DCE。DCE 只用于概括 intervention quality，不能替代训练后的 OOD Success Rate。

## 5. 主结果

每个 stage 生成一张表：

| Timing Condition | DCA ↑ | EAS ↑ | DCE ↑ | OOD SR ↑ | ΔOOD SR ↑ | ID SR ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Immediate |  |  |  |  |  |  |
| Previous Stage Complete |  |  |  |  |  |  |
| Capability Boundary |  |  |  |  |  |  |
| Failure-Recovery |  |  |  |  |  |  |

额外报告各诊断指标与 `ΔOOD SR` 的 Spearman correlation。样本点是“任务 × stage × timing condition”，不能只用单个任务的四个点得出强相关结论。

## 6. 启动门槛

- stage-localized OOD 已通过 oracle、prefix competence 与 base-gap 验证；
- stage boundary 可由环境 predicate 稳定重现；
- snapshot/fork 或 deterministic replay 已验证；
- nominal trajectories、DCA state representation 和归一化方式已冻结；
- 所有训练配置和专家动作预算已预注册。
