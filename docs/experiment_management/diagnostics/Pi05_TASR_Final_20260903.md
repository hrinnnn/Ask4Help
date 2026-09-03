# π0.5三任务TASR最终结果

已完成用户要求的12组真实方法/任务结果（其中StackCube内部gate是Deep-kNN，不是Bridge-PCA）。没有重新训练任何模型。

主结果固定使用此前选定的3倍匹配半径；1倍、2倍同时保留在JSON中，没有按SR选择参数。

| 任务 | 方法 | OOD SR | TASR ×3 | 兼容目标点/实际专家点 |
|---|---|---:|---:|---:|
| 旧StackCube | BC | 49/100 | 0.5337 | 1449/2715 |
| 旧StackCube | Failure-Recovery | 29/100 | 0.5427 | 1422/2620 |
| 旧StackCube | Diff-DAgger | 34/100 | 0.5343 | 1371/2566 |
| 旧StackCube | Internal Deep-kNN | 81/100 | 0.5797 | 1539/2655 |
| Grab Plane | BC | 0/100 | 0.3344 | 5958/17819 |
| Grab Plane | Failure-Recovery | 63/100 | 0.3005 | 5488/18261 |
| Grab Plane | Diff-DAgger | 84/100 | 0.2968 | 5336/17978 |
| Grab Plane | Bridge-PCA | 81/100 | 0.3028 | 5628/18589 |
| YCB Object | BC | 48/100 | 0.4271 | 2543/5954 |
| YCB Object | Failure-Recovery | 49/100 | 0.2568 | 1529/5954 |
| YCB Object | Diff-DAgger, raw threshold 0.05 diagnostic | 42/100 | 0.3574 | 2128/5954 |
| YCB Object | Bridge-PCA | 52/100 | 0.4221 | 2513/5954 |

这里YCB的0.05是gate原始分数阈值，不是TASR校准分位数。TASR的校准分位数固定为0.925。

## 数据及计算定义

- 输入为模型实际使用的1158条新增专家轨迹、107019个专家轨迹点；另含17条未用于matched-budget训练的nominal reference-only轨迹。共恢复/检查1175条，目标计分缺口为0。
- SC kNN/BC使用其full_v2完整后缀重建数据，未使用full_v1旧截短数据；YCB严格按原5954点matched-budget选择清单。
- 分子是目标阶段内与同任务成功OOD参考相容的轨迹点；分母是全部新增专家后缀点，包括灰色非目标段。原始ID replay不加入这个分母。
- 目标段是approach/alignment及close至稳定抓取，lift/transport及额外静态闭合保持不计入分子；SC保持5步close规则，其他任务以4步稳定抓取确认结束close。
- 位置、姿态、开合尺度保持2cm、15度、1cm，参考与查询使用当前物体中心和稳定抓取TCP轴。TCP及开合来自原训练qpos；物体位置与瞬时接触来自经过验证的历史回放。
- 一个查询对应一条参考，分阶段单调对齐，reference每步允许推进0至5；全部query点保留。成功OOD offline示范以seed mod 5划分reference/calibration/check，查询seed从reference及calibration移除；每阶段q=.925半径再统一乘3。

## 回放恢复的关键修复

1. H20 Vulkan入口显式指定已有ICD；CUDA可见，scratch/cache放到`/tmp/pi05_tasr_reconstruction_v1`，避开已满的系统盘。没有重装环境、删除文件或停止旧PID276925。
2. Plane在接管时恢复snapshot的操作必须重现；仅有raw动作不足以复现原流程。对7条多候选异常轨迹，还恢复其原始废弃候选尝试，然后执行原文件中的accepted动作。旧候选动作步数和成功/失败标签均对应；这7条最终qpos及末帧物体pose误差均为0。废弃候选动作不计入TASR分母。
3. StackCube原采集复用ID/OOD两个环境；full_v2重建还记录过重复尝试次数。单条fresh-env回放会遗漏接触缓存历史。恢复1892个原始顺序条目（含未被训练选中的历史条目）后，余下10条异常轨迹全部qpos误差为0。训练数据始终还是原1158条，未增加或替换。
4. 验证对象是所有实际被TASR读取的状态，包括稳定抓取坐标轴和确认帧。1175条所需状态最大qpos误差为9.50098e-5，小于原1e-4标准；没有放宽阈值。40条unused tail的完整回放审计仍记录非PASS，后段不用于几何/接触匹配，只按原数据计分母；不宣称整段每个状态都逐位一致。

## 独立审计及边界

独立脚本通过：Object 9824、StackCube 6145、Plane 27011个目标点，总计42980；以旋转矩阵角度独立重算距离，最大差3.829e-7；同一seed排除、接触条件、半径分类、非目标不计分、分母均通过。原20份评测文件已读取；表中12份OOD结果分别按100条逐episode重数，与SR完全对应。加速DP与原实现逐项比对通过。

本表不能用来宣称TASR普遍预测SR：Plane BC的TASR最高但SR为0；YCB BC与PCA的TASR排序也不等于SR排序。指标衡量目标监督构成，不覆盖优化失败、前置能力和数据覆盖问题。保留这些反例，不后验调参。

Plane SR仍为历史ever-grasped，不是新复核的稳定机身抓取率；其Diff为84而非用户原表74。SC81明确是Deep-kNN，所以Bridge-PCA归属严格版表中的Ours/SC仍为空，并在脚注报告这组真实Deep-kNN结果；全资产表将该列写为Internal gate，不冒充Bridge-PCA。

## 产物

本地根：`artifacts/pi05_table_tasr_20260903/`

- `active_learning_results_filled.tex`：Bridge-PCA归属严格版，SC内部kNN结果在脚注。
- `active_learning_all_actual_gates.tex`：所有真实内部gate结果都在表内，列名Internal gate、SC明确注明Deep-kNN。
- `TASR结果与填表说明.md`、`pi05_tasr_table_values.json`：全精度数据、1/2/3倍敏感性与SR出处。
- `tasr_scores.json`、`tasr_object.json`、`tasr_stackcube.json`、`tasr_airplane.json`：逐轨迹匹配及选择来源。
- `tasr_independent_audit.json`、`PI05_TASR_COMPLETE.json`：独立验收。
- `training_assets/manifest.json`、`replay_bundle.json.gz`：实际训练输入与便携回放包。

持久结果根：`/mnt/data/ask4help/results/pi05_tasr_reconstruction_v1/`；`full_v1`保留初轮，`retry_stack_warm1`、`retry_plane_candidates`、`stack_history_gpu0/1`保留修复轨迹。最终分析归档于`analysis_final`，便携输入于`inputs`。旧失败诊断全部保留。
