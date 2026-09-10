# TNS 模型学习收益：已启动并得到第一批结果

2026-09-10。用户已明确授权直接执行，允许局部改进，不要求所有任务完美符合。

本轮已完成36次真实X-VLA缓存上的100步更新，以及独立Luna审计。计算仅更新最后的1025×8 affine action head，backbone与transformer冻结，不是完整VLA短训练，也没有新闭环SR。

新候选：在原生loss单位下，按ID:OOD=1:1同时计算ID保持和OOD改善。OOD内部等权正常/恢复探针：

\[
\mathrm{TNS}(D)=\frac{0.5\Delta L_{\mathrm{ID}}+0.25\Delta L_{\mathrm{nominal}}+0.25\Delta L_{\mathrm{recovery}}}
{0.5L^0_{\mathrm{ID}}+0.25L^0_{\mathrm{nominal}}+0.25L^0_{\mathrm{recovery}}}.
\]

每个\(\Delta L\)是更新前减更新后的独立探针损失。分数是相对混合风险下降，可能为负，不是成功概率。ID数据自身原loss较低，所以不能直接把各域相对百分比相加；本候选先按共同loss单位加权，再整体归一化。

## 已完成的第一批数值

两种reset划分×三个更新随机种子，Adam lr1e-4、100步、每步16ID+16专家anchor，所有真实tail与mask保留。每种timing方法只有原选择清单的前8条缓存，本轮的子集结果不能等同于整个原数据集质量。

| 条件 | 新候选均值 | 旧TASR×3 | 已有SR |
|---|---:|---:|---:|
| immediate | 0.1413 | 0.2226 | 0.52 |
| Recovery | 0.1320 | 0.1946 | 0.30 |
| post-grasp | 0.1209 | 0.3095 | 0.10 |
| post-lift | 0.0931 | 0.4345 | 0.04 |

100步时六次重复均为Spearman=1；旧TASR=-0.8。20步的新候选rho为0.8/1.0/0.4/0.4/0.4/0.4，不能声称任意更新长度都可靠。ID权重0.25的六次rho为0.8/1/0.8/0.4/0.4/0.4；权重0.75为1/0.8/1/0.8/1/1。0.5来自原ID:OOD配比，敏感性只是事后检查，不按SR选最优权重。

为什么有改善：只看OOD误差下降，post-lift得分最高、仍与SR冲突；但它造成的ID误差上升约1.535（原loss约0.312），明显大于其他三组的0.651—0.729。引入ID保持后，后段数据的局部改善与遗忘代价可以同时体现。仅ID保持与SR相关也达到0.8，因此不能把所有改善都归因于新组合。

另外两种gate数据独立比较（其原预算434，不与1968的timing组混算相关）：PCA新分数0.1204，Diff0.1094，六次配对均PCA>Diff，与已有74%/45%排序相符。旧TASR在这两组原本也排序正确，不能声称这里新指标已经优于旧指标。

## 已明确保留的研究边界

- ID保持公式是在首轮pilot发现遗忘后提出，属于事后候选；没有隐藏最初OOD-only分数的失败。
- 两个fold和三个采样种子不是六套独立采集的数据。所有后训SR此前已知。
- ID检查数据来自原基座训练集，衡量保持能力，不是新ID泛化测试。
- 缓存基座224bicubic/ImageNet normalization、BGR解码、auto8→20动作和tail mask均由Luna与本地handler核对。FP32 affine重建native BF16损失平均/最大绝对差0.00678/0.1495，记录为数值近似。
- fixed noisy inputs和冻结transformer限制了更新解释。100步head收益不能直接当作全模型训练收益。

## 更严格的独立探针已启动

服务器H20 GPU0/1忙于另一Owner的budget30SFT。新任务只占CPU8—11，nice10；原GPU workers分别CPU0—3/4—7，无重叠，无CUDA显存。native runtime与原checkpoint/data都在该机。

CPU5样本smoke已完成18.17秒，5条finite、1条tail。与旧GPUsmoke同样本、同末层权重、同mask；图像bridge特征cosine在0.99995—0.999996。CPU/CUDA的torch随机数生成不同，相同整数seed不表示相同噪声，不能把noisy-head损失差当作数值等价测试。

独立批次共1326个样本：沿用每方法128个原训练集合随机anchor、20个未被任何6组训练选择的正常probe reset；其中14个有对应成功Recovery，可加入恢复probe，缺少6个明确登记；ID更新用原episode0—3，ID检查用108—127的96个随机anchor，互不混用。原20个reference reset仍单独保留。

持久controller PID2098868，worker2098873；源代码已推GitHub分支`codex/tns-model-utility-v1`，commit1ebc0bc4。服务器直接GitHub fetch出现TLS错误，改部署该已推送commit的git archive；没有在服务器修改源码。

实际状态根：`/mnt/data/ask4help/results/tns_model_utility_v1/`。控制器自动执行cache提取→6组×3种子100步head更新→COMPUTE_COMPLETE，之后由Owner独立审计并解释，不把中间cache完成当终点。

最新启动确认：241/1326，208.2秒，无failure marker、GPU memory0。按当前吞吐估计该阶段约二十分钟量级；估计不是承诺。

本地入口：`tools/run_tns_cpu_pipeline.py`；协议`configs/pipelines/tns_model_utility_v1.json`。审计完成前不覆盖论文指标；若新探针推翻当前排序，保留反例并继续有依据的修订。

监测已创建：`tns`，每10分钟检查阶段与错误；健康时保持安静。完成独立结果审计与报告后自动删除。
