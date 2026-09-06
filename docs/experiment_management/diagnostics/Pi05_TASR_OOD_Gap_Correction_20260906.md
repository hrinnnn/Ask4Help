# TASR的OOD分子纠正

用户明确：gap只定义在OOD样本中，ID点不应因几何相似获得有效监督计数。此前实现虽用OOD参考和校准，却将ID/OOD查询均计入分子。这是指标域条件的语义错误；之前的数值/回放审计不能证明这一定义正确。

修正公式：

\[
\operatorname{TASR}_{\mathrm{OOD}}(\mathcal D_E)=
\frac{\sum_{\tau\in\mathcal D_E}\sum_t
\mathbf 1[\tau\text{ from OOD}]
\mathbf 1[s_t\text{ matches designated OOD target segment}]}
{\sum_{\tau\in\mathcal D_E}|\tau|}.
\]

ID全部计分母、分子0。OOD也只有目标相容点计分子；OOD非目标及不相容点仍在分母。本轮只改变域条件，不改变参考、校准阈值、对应、阶段、接触或训练选择清单。

| 任务 | BC | Recovery | Diff | 内部gate |
|---|---:|---:|---:|---:|
| 旧StackCube | 0.2913 | 0.4153 | 0.3691 | 0.5582（Deep-kNN） |
| Grab Plane | 0.2340 | 0.2088 | 0.2002 | 0.2358（Bridge-PCA） |
| YCB Object | 0.4271 | 0.2545 | 0.2051 | 0.4177（Bridge-PCA） |

以上统一3倍半径、q=.925；1/2倍结果同时保留。SR未改变：SC49/29/34/81，Plane0/63/84/81，YCB48/49/42/52。Plane为历史ever-grasped，SC81非PCA，YCB Diff0.05为诊断。

原始107019个专家点全部保留，移除ID兼容点对分子的贡献；1175条原split、selected、对应与几何参数逐项核对通过。独立从原point labels对12组×3倍数重数，ID绿色贡献为0；生成6个真实源视频的新旧标记对比，验证pre-action帧数并查看关键帧。

目标阶段核对：三个任务当前都是approach/alignment及稳定close；SC不覆盖transport/place，Plane不含额外close hold/lift/transport，YCB不含lift/transport。它们是任务相关阶段代理，尚未证明阶段内每一点属于base-policy的未掌握技能。不能把本次域修正称作完整skill-gap识别。YCB BC全部训练点来自OOD，纠正后不变；Plane及SC的排序反例仍保留。

3倍within-task Pearson/Spearman：SC0.666/0.200，Plane-0.485/-0.400，YCB0.718/0.400，每组4方法。定义纠正有必要，不等于普遍SR相关性成立。

产物根 `artifacts/pi05_tasr_ood_gap_20260906/`：`结果与目标片段核对.md`、`summary.json`、`audit.json`、两版LaTeX、6个视频、逐轨迹新标签。旧数值保持历史可追溯，但论文应替换旧的无域限制TASR；不混用旧版敏感性相关系数。
