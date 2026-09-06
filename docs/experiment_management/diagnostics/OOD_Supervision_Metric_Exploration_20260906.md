# OOD监督选择与轨迹质量：60候选探索

用户要求扩大SC目标范围、突出ID/OOD选择、理解YCB差异，并对照OpenDrawer。已执行本地原始轨迹分析，无新训练或仿真。所有候选是看到SR后的研究探索，未替换正式OOD-only TASR。

## 候选

P=OOD专家点/全部新增专家点；G=匹配目标片段的点（硬半径或高斯连续权重）；U=原指标不评价的后续OOD点；O=OOD专家点。

探索 `P^alpha (G+beta U)/O`，factor∈{1,3,6}、beta∈{0,.5,1}、alpha∈{1,2,4}、kernel∈{hard,soft}，共54组合。beta>0是假定未评分后续片段有用的乐观消融，不是已核验非Recovery标签。alpha不等于1时不再解释为字面轨迹点占比。

另算6个全路径候选：直接用全部原训练qpos恢复TCP的位置/姿态/开合，机器人基坐标系、无物体中心或自身稳定抓取轴归一化；参考为同task成功OOD offline全轨迹；一条查询选一条参考，0..5步单调DP，同seed排除，q=.925校准，硬/软核×{1,3,6}。它评价完整几何相似度，没有接触标签约束，属于新候选，不冒充旧TASR的简单改阈值。

## 关键发现

| 候选 | SC Spearman | Plane | YCB | Drawer |
|---|---:|---:|---:|---:|
| 既有OOD目标段×3 | .2 | -.4 | .4 | .771 |
| 全路径×1 | .2 | -.8 | .4 | -.143 |
| 全路径×3 | .2 | -.4 | .4 | -.143 |
| 全路径×6 | .2 | .8 | .2 | -.143 |

SC原本排除约41%–43%的OOD后续点。全路径×3时，kNN/BC/Diff/Recovery分数为.9567/.4954/.6461/.7240，更符合用户“多数OOD轨迹都可用”的语义，但仍不能解释BC高于Recovery的SR。

YCB全路径×3 BC/PCA/Diff/Recovery为.9997/.9802/.5413/.9691（Pearson约.904，Spearman.4）。Recovery原局部TASR低分主要受目标片段长度与局部距离惩罚影响。不能据此宣布其训练数据全都有效，宽容全路径分数趋近OOD预算比例。

OpenDrawer是全部来自同一OOD条件的timing比较，P恒为1。把已掌握的开柜/搬运也计有效会奖励t0冗余监督，破坏原有时机区分。因而目标范围需要task-condition-specific定义；全任务学习与局部OOD适应不可无区别套相同片段。

60候选在其余三task选择参数、留出一个task的Spearman分别为SC.2、Plane-.4、YCB.4、Drawer-.143。全数据最好组合为原target×3、P指数4，平均Spearman约.443；该指数缺少轨迹占比解释，不能因为训练拟合更好就推荐为正式指标。

## YCB与相机/统计

人工查看四个实际选中专家后缀的reset/接触图：固定外部相机和腕部相机，未见方法切换相机布局的证据。TASR读取运动学与状态，无RGB或camera extrinsics输入；相机位置不直接影响其值。它可能影响policy训练，但当前数据不能归因为相机故障。自身稳定抓取轴归一化会删除部分全局姿态差异；新full-path候选作为这一因素和范围的联合消融，不能分别归因。

YCB相同100测试seed配对bootstrap（20000次，固定seed20260906）：PCA-BC 4pp，95%[-7,15]pp；PCA-Recovery 3pp，[-10,15]pp；PCA-Diff10pp，[-2,22]pp。区间仅反映该checkpoint条件下评测episode抽样，不含训练seed不确定性，不能强求轨迹指标复现微小SR排序。

Plane旧BC的ID ever-grasped41/100，PCA92、Diff87、Recovery93，提示需单独检查训练适配/能力保持，不应把OOD SR差距全部归于监督几何。

## 产物

`artifacts/ood_supervision_exploration_20260906/`：最终讨论报告、全部54组合及6全路径版本、留一任务检验、配对bootstrap、真实YCB相机接触图、分布图及全路径独立数值审计。计算脚本：`tools/explore_ood_supervision_metrics.py`、`explore_full_ood_path.py`、`summarize_ood_metric_exploration.py`、`audit_full_ood_path_exploration.py`。

建议继续采用统一的“全部新增专家预算中，OOD且属于待学目标区域的兼容监督占比”解释；按任务定义w_task(s,a)，SC范围可更宽，Drawer Grasp应保留局部。需要参考多样性与未参与设计的验证数据，而非每任务按SR选最优半径。此轮未发现已验证的通用强相关指标。
