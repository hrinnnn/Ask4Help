# StackCube / Grab Plane 目标监督占比离线诊断

2026-08-31；owner=codex-root-stackcube-airplane-target-ratio。输出 `artifacts/stackcube_airplane_target_ratio_20260831/`。

同一Q定义：目标抓取片段内与成功参考相容的新增expert anchors，占全部选中expert anchors的比例。StackCube采用原始抓取OOD，不混Stage2放置；Grab Plane参考为已接受neck_center专家，不沿用瞬时ever_grasped作为可靠抓取。目标截止稳定close，不含lift/运输/放置。

| Task | Takeover | 目标点 / 固定预算 | Q |
|---|---:|---:|---:|
| StackCube | 0 | 264/520 | 50.77% |
| StackCube | 10 | 204/520 | 39.23% |
| StackCube | 20 | 204/520 | 39.23% |
| StackCube | 30 | 194/520 | 37.31% |
| StackCube | 45 | 208/520 | 40.00% |
| Grab Plane | 0 | 1498/2820 | 53.12% |
| Grab Plane | 10 | 540/2820 | 19.15% |
| Grab Plane | 20 | 424/2820 | 15.04% |
| Grab Plane | 30 | 349/2820 | 12.38% |

274条数值资产包含预算选择样本和每任务60条t0参考；源parquet state/actions与原raw actions核验一致。每任务36参考/12校准/12检查，所有组统一排除同seed参考与校准。FK与记录TCP的最大位置差<0.6微米。

旧日志无逐帧物体四元数。补录smoke的StackCube完全复现，但Airplane TCP漂移2.76cm，故停止且不采用。当前坐标轴取每条轨迹实际稳定抓取TCP姿态、原点为当前物体中心；不是实测物体frame，需与OpenDrawer版本区分。原paired-nominal-axis诊断已保留。

匹配采用2cm/15deg/10mm尺度、分任务/分阶段q=.925及下限0.13017；单调DP。正常专家检查：StackCube接近108/130、close59/60；Airplane接近870/916、close39/48。后者仍有合法闭合被拒，结果必须保持诊断标签，不能宣称通用学习价值或SR预测。

9个视频以每task全预算组共同最小seed展示（StackCube150001、Airplane160000）；显示逐点颜色、误差/阈值、参考索引，严格按源frame t=动作前state t对齐。StackCube0.5倍速。

5项逻辑测试与独立分母/逐点/视频审计通过。只完成本轮离线任务，不修改原训练和54个Stage-B评测，不写原pipeline完成标志。临时GPU4仿真上下文已退出，未启动批量回放或新训练。
