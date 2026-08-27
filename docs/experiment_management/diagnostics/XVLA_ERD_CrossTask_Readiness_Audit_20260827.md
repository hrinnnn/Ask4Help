# ERD-Pose 跨任务数据就绪性与历史 gate 审计

日期：2026-08-27（北京时间）  
范围：StackCube Stage 2、StackPyramid、PickSingleYCB object variation、OpenDrawer，
并与当前 StackCube/Grab Plane ERD-Pose diagnostic 对照。  
性质：只读审计与方案建议；不改变任何 formal pipeline、checkpoint、seed、threshold、
success predicate 或 OOD 解锁状态。

## 1. 审计标准

要复用同一 ERD-Pose 分析，至少需要：

1. 一个可追溯的 expert reference（最好是同一 OOD context 或可做 context matching）；
2. policy 的 ID/OOD 逐步 state/action/video timeline；
3. 能还原 gripper position/orientation/width/velocity 的记录或可验证 replay；
4. 明确的 gate score timeline 与 first-alarm 定义；
5. 独立记录 denominator、checkpoint、task/success 定义，不能把 diagnostic 和 formal 主表混合。

当前 ERD-Pose 的主统计是 expert residual 的 q=.925/q=.95 等 pointwise quantile，
并在 OOD policy trajectory 上寻找第一次持续越阈。它不是已有 PCA gate 的同义替代物。

## 2. 真实已完成程度

| 任务/实验 | 已核对的真实产物 | ERD-Pose 就绪性 | 不能直接声称的内容 |
|---|---|---|---|
| 当前 StackCube/Grab Plane fixed-grid diagnostic | StackCube 与 Grab Plane 各 50 条 OOD pose replay、20 条 expert reference、actions/videos/reset metadata；当前 ERD summary 与视频已通过独立核对 | **已就绪** | 这是 task-policy-knee diagnostic，不是 downstream SR 最优性证明 |
| X-VLA StackCube Stage 2 target-OOD timing | `oracle_validation/collection` 有 100 条 Stage-2 OOD expert action/video/task-state；`cohort/policy_screen` 有 400 条 OOD policy action/video；最终各 timing method 有 100 ID + 100 OOD 视频与 actions；根目录有 `PIPELINE_COMPLETE_RESUMED` | **可 replay，但没有现成 EE pose**；需要按 reset metadata/action 重新记录 gripper pose | 400 条 policy screen 本身不是 gate alarm 数据；不能把固定 anchor timing 当成 learned alarm timing |
| X-VLA StackPyramid | Stage 2/3 passive PCA 各有 100 ID + 100 OOD 的 actions、videos、states、score timeline；threshold=`1.2141989`；当前 checkpoint 为 `ckpt-40000` | **policy 曲线已就绪，expert OOD pose 未就绪**；已有 OOD Oracle gate 主要只有 summary，不能替代逐步 expert pose reference | 当前 base 是 recovery diagnostic；不能称为已完成的四方法或 ERD 跨任务结果 |
| π0.5 PickSingleYCB object variation | Oracle ID/OOD 各 20/20；passive detection ID/OOD 各 100/100；ID expert dataset 128 episodes、6634 frames、state=9D Panda qpos；Bridge/Failure-Recovery/Offline-Oracle 各 100 accepted；Diff canonical 仅 10/100 | **policy 视频/action 已就绪；需要补 OOD expert action/state replay**。现有 OOD Oracle gate summary 只有视频，不能直接生成 expert pose | Diff partial/low-threshold diagnostic 不能冒充完整四方法 matched-budget 结果 |
| OpenDrawer π0.5 q=.95 diagnostic | ID、Handle-OOD、Grasp-OOD、Goal-OOD 各 100 条 policy rollout 均有 actions、9D Panda qpos states、reset metadata、score timeline；Oracle20 每个 split 有 17--20 条 OOD expert accepted LeRobot episodes，含 9D state/actions/video | **可 replay diagnostic**；需要用 Panda FK/环境 replay 将 qpos 转为 EE pose，并对 expert suffix 的起始阶段做对齐 | 该 checkpoint 是 preliminary ID checkpoint；q=.95 retry4 是 failure-detection diagnostic，不是已通过 ID base 的 DAgger 主结果 |
| X-VLA WidowX/Panda vegetable object variation | WidowX visible-RGB retry 的独立 ID gate 为 0/20，OOD 锁定；Panda 新线尚未完成 Oracle/ID gate | **暂不纳入** | 不能把候选 smoke 或未完成 pipeline 当成已做过的跨任务证据 |

## 3. 已完成 gate 的报警时间（历史 detector，不是 ERD）

以下只列出当前能从真实 score timeline/summary 直接复算的代表性结果；时间单位为
environment step。它们不能和 ERD 的 (D_t) 数值直接横向比较。

### X-VLA StackCube/Grab Plane Stage-C passive gate（50 ID + 50 OOD）

使用冻结 ID q=.95 trajectory-maximum threshold，first alert 为第一个越阈 decision point：

| Task/split | Input PCA | Bridge PCA | Action block-01 PCA | Diff-DAgger |
|---|---:|---:|---:|---:|
| StackCube ID | 8/50，median 75 | 9/50，median 45 | 9/50，median 50 | 4/50，median 30 |
| StackCube OOD | 50/50，median 35 | 50/50，median 5 | 50/50，median 20 | 18/50，median 105 |
| Grab Plane ID | 5/50，median 95 | 5/50，median 120 | 4/50，median 80 | 3/50，median 65 |
| Grab Plane OOD | 50/50，median 0 | 35/50，median 100 | 46/50，median 95 | 44/50，median 90 |

这张表说明不同 representation 的报警阶段差异很大；其中 Grab Plane Input PCA 的
step-0 报警不能直接解释为合理 takeover timing。

### StackPyramid passive PCA recovery gate

| split | episodes | strict success | threshold | first alarm |
|---|---:|---:|---:|---:|
| Stage 2 ID | 100 | 55/100 | 1.2142 | 0/100 |
| Stage 2 OOD | 100 | 0/100 | 1.2142 | 2/100（steps 180, 530） |
| Stage 3 ID | 100 | 36/100 | 1.2142 | 0/100 |
| Stage 3 OOD | 100 | 0/100 | 1.2142 | 0/100 |

### PickSingleYCB object variation passive gate

冻结 q=.95 thresholds、execute horizon=5；从每条 score trace 的第一个越阈点换算为
environment step：

| split | Bridge PCA | Diff-DAgger |
|---|---:|---:|
| ID | 48/100，mean 53.9，median 60 | 0/100 |
| OOD | 100/100，step 0 | 0/100 |

Failure-Recovery 是固定 step 50 的 intervention，不是 detector alarm；Offline-Oracle
从 step 0 接管，也不应被当作 learned alarm。

### OpenDrawer q=.95 failure-detection diagnostic

该根的 4 个 split 均为 100 episodes、完整 qpos/action/timeline evidence。代表性
`action_expert_block_08__pca_residual` first-alert 统计为：

| split | alerts | median step | failure recall |
|---|---:|---:|---:|
| ID | 24/100 | 290 | 0.538 |
| Handle-OOD | 83/100 | 220 | 0.830 |
| Grasp-OOD | 49/100 | 295 | 0.490 |
| Goal-OOD | 95/100 | 275 | 0.950 |

同一 root 的 VLM block-08 PCA 分别为 ID `33/100, median 275`、Handle `99/100, 215`、
Grasp `81/100, 250`、Goal `99/100, 220`。这些是 π0.5 action/feature detector
时间，不能包装成 ERD-Pose timing。

## 4. q=.925 的跨任务定位

可以把 q=.925 作为一个统一的**校准规则**，但不能把一个 raw threshold 数字用于
所有任务：


\[
\tau_k(q)=\operatorname{Quantile}_q
  \left(\{D^{\mathrm{expert}}_{k,e,t}\}\right),
\qquad q=0.925,
\]

其中 (k) 是 task/checkpoint，(	au_k) 仍然是 task-specific。跨任务比较应比较
alarm onset/confirmation、coverage、late alarm 和 expert-intervention cost，而不是
比较 (	au_k) 的绝对大小。

## 5. 建议的最小后续方案（等待确认）

### A. 先做现有数据可完成的 replay

1. **StackCube Stage 2 target-OOD**：用已有 OOD expert action/video/task-state 与
   policy action/video/reset metadata，重新 replay 并保存 EE pose、width、velocity；
   同时从 final evaluation 取 ID/OOD policy curves。
2. **PickSingleYCB object variation**：对 ID expert dataset、OOD Oracle 和 100/100
   policy rollouts做同样的 Panda FK/replay；只保留 object-model split 的合法配对。
3. **OpenDrawer**：使用现有 9D qpos state 和 17--20 条 OOD expert suffix，做 FK、stage
   alignment 和 ID/Handle/Grasp/Goal 的 ERD diagnostic；明确标注 preliminary base。

这三项都不需要重新训练，只产生 diagnostic pose timelines、q=.925/q=.95 sweep、
ID/OOD 曲线视频和历史 gate alarm 表。

### B. StackPyramid 单独补一项 reference capture

StackPyramid policy 的 ID/OOD state/video 已有，但 OOD expert 的逐步 action/state
reference 不完整。因此要纳入同一 ERD 主比较，需先用既有 v4 Oracle 在 Stage 2/3
各记录约 20 条 OOD expert action + state/pose trajectory；不需要重训 base。若不批准
这项 capture，StackPyramid 只报告已有 PCA timing，不与 ERD 数值合并。

### C. 统一输出口径

所有任务都同时输出：

- `q=.925` 主候选、`q=.95` 敏感性分析；
- `t_onset`（连续越阈区间的第一个点）和 `t_confirm`（第二个点）；
- ID false-alarm、OOD detection、pre/post-stage、late/censored；
- 一条 representative expert 曲线、一条 OOD fail 曲线，以及 50/100 条轨迹的分位带；
- ERD timing 与 downstream SR 分开，不声称 q=.925 已经证明 SR 最优。

当前推荐先执行 A，再决定是否为 StackPyramid 做 B。这样不会把缺 reference 的
StackPyramid 或未通过 ID gate 的 OpenDrawer 结果伪装成同等强度的正式证据。
