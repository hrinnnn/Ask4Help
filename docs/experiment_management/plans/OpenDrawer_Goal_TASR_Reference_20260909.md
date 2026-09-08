# Goal-OOD TASR参考：在反馈对照采集之前固定

用途仅为用户授权的OpenDrawer不同OOD阶段的TASR评价；不训练VLA，不向gate输入OOD参考、任务阶段或未来专家动作。当前Goal-OOD仅做过四条运行时/Oracle smoke，没有新的feedback on/off结果。

- 固定30个nominal专家raw seeds：1786000–1786029，从reset直接由同一个direct-current-state专家执行，400动作上限/首次真实strict成功终止。失败也保存，不补挑容易seed；成功者才可进入名义参考。
- 固定按seed模5分组：0/1/2为reference，3为tolerance calibration，4为checking。与1781000系列ID gate校准、1781300/1781420 smoke和拟定1783000系列Goal gate对照隔离。至少8 reference、3 calibration、3 checking；不足则报告参考不足，不根据结果重新分配。
- Grasp-OOD仍使用原18参考/6检查和原抓取target。Goal-OOD target为完成必要抓取及lift之后的transport、place、release。开抽屉、抓取和lift均保留在专家动作分母，不进入Goal target分子。此定义依据目标位置变化影响的操作阶段，不依据on/off收益选择。
- Goal参考坐标：TCP相对目标托盘的位置，TCP相对托盘（固定世界朝向）的旋转，夹爪宽度和真实物体夹持状态。不能使用物体相对TCP坐标消除整个搬运位移。
- 沿用既有物理尺度2cm、15deg、1cm，单条连贯参考选择、分阶段单调对应（每query参考索引前进0–5）及接触一致条件。每phase阈值取nominal calibration误差q0.925，下限沿用1mm/1deg/1mm组成的归一化半径。统一报告×1/×2/×3，不按on/off效果选择倍数。
- 这是“专家采集之后、SFT之前”的数据评价，不是无专家轨迹即可获得的在线最优时机标签。对有/无反馈的推断单位为独立collection stream，不能把一条轨迹的很多点充当独立实验重复。

实验设计skill用于提前固定划分、目标和复现单位；既有工具归属记录位于diagnostics/Scientific_Skills_Use_20260909.md，不把工具文献作为方法有效性的证据。
