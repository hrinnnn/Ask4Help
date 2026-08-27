# Expert-reference deviation takeover：指标提案与现有数据诊断

日期：2026-08-27（北京时间）  
性质：diagnostic/proposal；不覆盖原 fixed-grid formal protocol。

## 1. 动机

原始 fixed-grid knee 是对“完整 expert suffix 的 cost--deviation 曲线”做离散
几何选择。它回答的是一个 retrospective trade-off 问题，但没有直接回答：

> learner trajectory 什么时候第一次离开了同任务、同 OOD 条件下的 expert
> trajectory manifold？

新的目标是把 takeover time 定义为 failed/policy-only trajectory 上的
deviation onset，而不是从所有 fixed anchors 的全局曲率间接推回一个时刻。

## 2. 建议指标：Expert-Reference Deviation (ERD)

给定同一 OOD 条件下的 expert demonstration bank
`E={e^(k)}`，以及 learner trajectory `x_0:T`：

### 2.1 Task-relative state

先将状态变换为不受绝对物体位置直接影响的 task-relative representation
`z_t=psi(x_t)`：

- StackCube：`tcp-cubeA`、`cubeA-cubeB`、gripper width、`grasped`、`on_cube`，
  可选地加入相对速度；
- Grab Plane：`tcp-object`、`goal-object`、gripper/proprioception、`grasped`、
  `strict_success`，不把 raw image novelty 直接当作 timing deviation。

每个维度的尺度由 expert bank 的 robust MAD 或 shrinkage covariance 得到。

### 2.2 Causal phase alignment

在每个 learner prefix 上只允许匹配到当前或此前的 expert phase，避免用未来
expert state 产生 hindsight leakage。记匹配 phase 为 `phi_t`，expert phase
统计量为 `mu_phi`、`Sigma_phi`，则：

\[
D_t^{\mathrm{ERD}} =
\sqrt{(z_t-\mu_{\phi_t})^\top
(\Sigma_{\phi_t}+\lambda I)^{-1}(z_t-\mu_{\phi_t})}.
\]

样本很少时可使用 leave-one-out expert median/MAD 的 diagonal 版本。

### 2.3 Threshold and takeover time

阈值只使用成功 validation prefixes 校准，例如：

\[
\tau_D(\phi)=Q_{0.95}
\left(D_t^{\mathrm{ERD}}\mid \text{successful validation prefixes},\phi_t=\phi\right).
\]

为了避免单个 noisy frame 触发，定义两次连续越阈值的交接时间：

\[
T_D=\min\{t:D_t^{\mathrm{ERD}}>\tau_D(\phi_t),
D_{t+1}^{\mathrm{ERD}}>\tau_D(\phi_{t+1})\}.
\]

`T_D` 应称为 **critical deviation time** 或 **ERD takeover time**，不要在
没有 downstream update 验证前直接称为 globally best timing。它表示“再不接管
就开始进入不可接受偏离”的时间。

### 2.4 Broad-OOD robustness：paired/conditional reference

不能把所有 OOD expert trajectories 直接混成一个全局均值。对象初始位姿、目标
距离、物体形状和执行速度造成的 between-context variation 可能比真正的 policy
deviation 还大，从而让全局 covariance 过宽、阈值失去敏感性。

推荐使用三层 reference：

1. **paired reference（首选）**：为每个 OOD scene/object context 保存一条同初始
   条件的 expert demo；learner prefix 只和它比较；
2. **conditional prototype**：没有完全配对 demo 时，按可观测 context（初始
   object/goal 相对位姿、物体类别、初始 TCP 距离）检索最近的 expert prototypes，
   只在相同 context cluster 内估计 `mu_phi` 和 `Sigma_phi`；
3. **unsupported-context abstention**：若当前 context 离 reference bank 太远，
   不把它误判成“没有偏离”，而是输出 `reference_unsupported`，要求补充 expert
   demo 或单独报告该 OOD 子分布。

在 real robot 上，建议把状态分成两种版本：

- `ERD-Pose`（可部署）：末端位置、末端姿态、速度、gripper width 和可获得的
  object/goal relative pose；姿态误差使用 SO(3) geodesic distance，不依赖仿真
  专有变量；
- `ERD-State`（仿真诊断）：加入完整 simulator task state，只作为 upper-bound
  ablation，不能冒充 real-robot metric。

若不能稳定获得 object pose，至少使用 robot-centric pose：以初始 TCP frame 或
可观测的场景 frame 表示末端轨迹，并明确把 object-relative 部分标记为缺失，而
不是用 raw image novelty 填补。阈值应在 context-stratified validation 上冻结，
并报告 macro-average 与 worst-context false-alarm/miss。

## 3. 用现有 Stage-A 数据做的 preliminary check

现有 Stage-A 已经有：

- 每个 task 的 20 条 OOD expert `step=0` trajectories，可作为初始 expert bank；
- 每个 fixed anchor 的 policy prefix 和完整 `task_states`；
- StackCube/Grab Plane 的相同 seed pairing。

用相对状态和 expert leave-one-out 的全局 q=.95 variation 作为诊断阈值，得到：

| Task | expert variation threshold | first clear ERD crossing | crossing episodes |
|---|---:|---:|---:|
| StackCube | `2.80` | `t=10` | `20/20` at `step=10` |
| Grab Plane | `3.56` | `t=20` | `19/20` at `step=20`; `0/20` at `step=10` |

对应的 mean relative-state deviation 为：

| Task | `t=10` | `t=20` |
|---|---:|---:|
| StackCube | `5.11` | `6.40` |
| Grab Plane | `1.56` | `5.14` |

这给出一个比原始几何 knee 更符合任务语义的初步结果：StackCube 在 `t=10`
附近开始明显偏离，Grab Plane 在 `t=20` 才出现统计上清晰的偏离；两者都属于
150-step episode 的早期阶段。

该结果仍然是 preliminary：这些 fixed-anchor prefixes 会在指定 anchor 处被
expert 接管，不是完整 policy-only failure trajectories；正式 ERD 需要在无接管
的 OOD rollout 或动作 replay 中保存完整 task-state timeline。

补充敏感性检查显示，若直接使用每个 phase 的 leave-one-out MAD threshold，
StackCube 的所有 `step_100` prefixes 会在 `step=3` 左右越阈值；这表明逐 phase
方差估计在早期动作阶段过于敏感。正式版本应优先使用成功 validation-policy
prefixes 校准的 threshold，并报告 raw crossing 与 debounced crossing 两者，
而不是直接采用这个 step-3 结果。

## 4. 与现有 detector alarm 的关系

在 `failure-recovery horizon=50` 的右删失诊断中：

- StackCube Bridge PCA：`50/50` 在 horizon 内报警，中位数 `5`；
- StackCube Action PCA：`44/50` 在 horizon 内报警，中位数 `20`；
- Grab Plane Input PCA：`43/50` 在 `step=0` 报警，属于明显的 OOD novelty/早报；
- Grab Plane Bridge/Action/Diff：主要晚于 50 或漏报。

因此，ERD 可以作为 task-state ground-truth timing reference：StackCube 的
Bridge PCA 接近 ERD onset，但 Grab Plane 现有 detector 并没有稳定逼近 ERD onset。
这比直接把任意 PCA alarm 当作 task knee 更容易解释。

## 5. 论文中的定位

建议同时报告三种不同对象：

1. `raw cost--deviation knee`：完整 expert suffix 的 retrospective 几何点；
2. `ERD takeover time T_D`：failed/policy-only prefix 第一次越过 expert manifold
   的安全交接点；
3. `downstream utility optimum`：经过 matched-budget policy update 后测得的真实
   学习最优点。

其中第 2 项是 safety/timing reference，第 3 项才是 learning utility；两者不能
在论文中混写成同一个“best timing”。

## 6. 推荐的下一步实验（不立即改变 formal pipeline）

1. 从 Stage-A `step=0` OOD expert bank 做 paired/context-stratified reference split；
2. 对 Stage-C 已保存的 50 条 OOD policy-only action trajectories 做 deterministic
   replay，补出 task-state timeline；若 replay 不能通过 state/video alignment，
   再做一次只保存 state 的 50-episode diagnostic rollout；
3. 用 validation successful prefixes 冻结 phase- and context-conditioned `tau_D`，同时报告
   raw crossing 和 two-step debounced crossing；
4. 比较 ERD `T_D` 与 Input/Bridge/Action/Diff 的 alarm time、miss/censor、
   pre-grasp/post-grasp phase 和 unsupported-context rate；
5. 只在 ERD timing 稳定后，选择 `t=0,10,20` 等少量 candidate 做 downstream
   utility check，不先启动完整 Stage-C 的 30 个 gate-selected models。

本提案不修改原 Stage-C threshold、seed、success predicate、expert budget 或
completion contract。
