# X-VLA Fixed-Grid Task-Policy Knee Validation

**状态：Stage A 已在 StackCube 与 Airplane 完成，严格 extension 后两个 task 的
exact matched budget 均已恢复；Stage B 2500-step training 正由持久化总控运行，
training 完成后将自动进入 100-ID/100-OOD evaluation、utility summary、Stage-C
passive gate audit，以及其后的 gate-selected matched-budget data/training/evaluation
和独立 reconciliation。**

历史成功的 StackCube X-VLA collection 使用
`/data/zhaozhixuan/envs/xvla_official_5090/bin/python`。固定网格首次 smoke
在 RLinf 自带 `.venv` 中于 `mplib` planner 初始化时发生段错误；切换到历史
环境后 step 0 已写出完整 collection evidence，但在 SAPIEN teardown 阶段返回
`-6`。因此 controller 只在完整 summary/episode denominator 与 dataset 已写出
时记录并接受该 teardown abort；任何采样中途异常仍然失败并保留 diagnostic root。

Stage A 的独立 audit 显示 StackCube 与 Airplane held-out 的 calibration knee
均为 step 20。首轮 cohort 曾触发 `BUDGET_RESOLUTION_FAILED`；在不改变 anchors、
recoverability gate、whole-suffix 规则或 budget 公式的前提下增加预注册 extension
seeds 后，两个 task 均通过 exact-budget gate，因此解锁 Stage B。calibration knee
仍不被预先表述为 downstream SR 最优，需由 Stage B utility evaluation 独立验证。

为严格执行原 exact-budget 规则，新增一轮 calibration extension：StackCube
使用 OOD seeds `150020--150059`，Airplane 使用 OOD seeds `160020--160059`。
这只扩大共同 recoverable seed pool，不改变八个 timing anchors、checkpoint、
endpoint、recoverability gate、suffix 不切分规则或 budget 公式；extension 完成
后重新运行独立 audit。若仍低于 80%，才保留 `BUDGET_RESOLUTION_FAILED` 并停止
Stage-B。

StackCube extension 已完成：合并 cohort 的 valid anchors 为 `{0,10,20,30,45}`，
共同 recoverable seeds 为 59，exact common budget 已恢复为 `520/520`；step 60
仍因 `53/60` 低于 gate 而不进入 frontier。Airplane extension 随后执行，完成后
按相同规则重新判定其 budget gate。

Airplane extension 完成后，合并 cohort 的 valid anchors 为 `{0,10,20,30}`，
共同 recoverable seeds 为 55，exact common budget 为 `2820/2820`。两个 task
的 Stage-B 2-step/reload smoke 均通过，且日志确认 ID replay 与 selected expert
meta 同时加载、`action_valid_mask` 实际参与 loss；正式 2500-step training 已进入
controller 阶段。持久化 Stage-B total supervisor 以 900 秒间隔等待 training marker，
随后自动启动 evaluation controller，并在 54 个评测作业完成后生成两 task utility
summary。Stage-C passive gate controller 已预先启动并等待 Stage-B utility marker；
它固定使用 validation-ID seeds `{149000--149049,159000--159049}` 生成 q=.95
calibration，使用 held-out OOD seeds `{151000--151049,161000--161049}` 做 detector
rollout，再由 `summarize_xvla_gate_to_knee.py` 输出 KD/KHR。该阶段完成标记只表示
passive gate audit 完成，不能替代后续 gate-selected dataset/training。为使后续阶段
可在无人值守条件下继续，已预注册 Stage-C gate-training pool：StackCube
`154000--154399`、Airplane `164000--164399`；对应 utility evaluation 使用
StackCube ID/OOD `155000/156000`、Airplane ID/OOD `165000/166000` 各 100 episodes。
每个 gate method 先收集完整 expert suffix pool，再用确定性的 whole-episode exact
subset 选择 `520` 或 `2820` actions；只有所有 10 个 task--method 分支的预算审计
通过后，才启动 5 methods × 3 seeds × 2 tasks 的 2500-step training 和 100-ID/
100-OOD evaluation。若历史 Airplane detector asset 不含 `vlm_input_pool`，控制器
会在 Stage-B utility marker 之后、任何 validation/OOD rollout 之前，用同一
checkpoint 和 ID metadata 构建新的 ID-only multilayer asset；不得用其他层冒充
input，也不得从 OOD rows 拟合该 asset。

## 1. 目标

在不修改现有 Ground Truth task 语义的前提下，使用 canonical X-VLA StackCube 与 PickSingleYCB-Airplane，验证固定 takeover time 是否形成可重复的 time--deviation Pareto knee，并验证该 knee 是否落入 matched-budget downstream utility 的 near-optimal window。

本 pipeline 不使用旧 StackCube Stage-2 timing 作为正式输入；旧 Stage-2 结果保持 diagnostic/main-result 边界不变。

## 2. 冻结任务与模型

### StackCube

- task：现有 canonical X-VLA StackCube OOD split；固定 task、success predicate、camera、action contract 和 horizon。
- base：`/data/zhaozhixuan/Ask4Help-airplane-5090/results/xvla_stackcube_v1/temporal_mask_v2/id_sft_from3500_to10000_official_2gpu_retry1/ckpt-7500`。
- endpoint：strict task success。
- timing anchors：`env_step={0,10,20,30,45,60,80,100}`。

### PickSingleYCB-Airplane

- task：现有 canonical X-VLA Airplane OOD split；固定 object/goal/success predicate、camera、action contract 和 horizon。
- base：`/data/zhaozhixuan/Ask4Help-airplane-5090/results/xvla_airplane_v1/id_sft_10000_official_2gpu/ckpt-2500`。
- endpoint：`ever_grasped`，同时保留 strict completion 作为辅助结果。
- timing anchors：与 StackCube 相同的 `env_step={0,10,20,30,45,60,80,100}`。
- calibration cohort：在任何 anchor collection 前，按完整 base policy-only OOD rollout 冻结 `ever_grasped=false` 的 failure cohort；不得按某个 anchor 的结果动态筛选。

## 3. Calibration、budget 与指标

- 每个 task--base-policy 组合单独建立 knee；不同 backbone、checkpoint、OOD factor 不复用 knee。
- calibration 只需要 expert counterfactual completions，不更新 policy。
- 每个 anchor 使用相同 reset seeds 和同一 policy prefix；到达 anchor 后切换 privileged expert，保存 actions、18D task states、timeline、video、reset metadata 和 continuation outcome。
- recoverability gate：每个 anchor 的 calibration continuation 至少 `18/20` successful expert completions；否则写 `UNRECOVERABLE_REGION`，不移动 anchor、不补试、不进入 Pareto frontier。
- expert-time cost 不截断：`C=N_expert/N_nominal`，允许 `C>1`。
- deviation：task-state DTW 与 ID/OOD successful nominal reference 对齐，按 calibration set 固定尺度。
- 归一化 nominal-equivalent budget：

  `requested_budget = 20 * median(nominal_expert_actions)` per task。

  实际 budget 为共同 recoverable seed intersection 上不切 suffix 的最大可达完整-action budget；若低于 requested budget 的 `80%`，写 `BUDGET_RESOLUTION_FAILED` 并阻塞该 task 的 formal update。

- Pareto：删除双维被支配 anchors；用 bootstrap anchor-selection probability 建立离散 knee confidence set，不使用二维 convex hull。primary knee test：

  `P_bootstrap(knee ∈ W*_B,0.05) >= 0.80`。

## 4. Seed manifests

### StackCube

- threshold-validation ID：`149000--149049`。
- calibration OOD：`150000--150019`。
- held-out gate OOD：`151000--151049`。
- final ID：`153000--153099`。
- final OOD：`152000--152099`。
- Stage-C gate-training pool：`154000--154399`；gate utility ID/OOD：
  `155000--155099` / `156000--156099`。

### Airplane

- threshold-validation ID：`159000--159049`。
- calibration OOD failure cohort：`160000--160019`。
- held-out gate OOD：`161000--161049`。
- final ID：`163000--163099`。
- final OOD：`162000--162099`。
- Stage-C gate-training pool：`164000--164399`；gate utility ID/OOD：
  `165000--165099` / `166000--166099`。

All ranges must pass a server-side collision audit immediately before launch; a collision stops the launch and requires a new manifest revision.

## 5. Stages

### Stage A：fixed-grid timing calibration

1. run paired policy-prefix/expert-fork smoke on 2 seeds and all 8 anchors;
2. run 20-seed calibration for each task;
3. write Pareto frontier, bootstrap knee set, nominal scales and continuation audit;
4. independently verify calibration artifacts before any training.

### Stage B：controlled timing utility

- For each of 8 anchors, select whole suffixes only from the common recoverable seed intersection under the resolved task budget.
- Train three independent policies per anchor from the same base with the existing temporal-mask and source-balanced training contract, fixed at 2500 update steps.
- Evaluate 100 ID and 100 OOD episodes using frozen final seeds.
- Define `t*_B` and the 5-point near-optimal window from the mean OOD SR across the three training seeds.

### Stage C：gate-to-knee utility

- Freeze thresholds using validation-ID FAR `<=5%`; never retune on OOD.
- Evaluate Input PCA, VLM-action Bridge PCA, Action Block 01 PCA, Diff-DAgger and fixed-step Failure-Recovery on held-out OOD rollouts.
- Fork expert from each actual gate alarm state and compute KD/KHR against the frozen task knee.
- After the passive audit marker, collect `input_pca`, `bridge_pca`, `action_pca`,
  `diffdagger`, and fixed-step-50 `failure_recovery` suffixes using the frozen
  validation-ID thresholds and the pre-registered Stage-C training seed pools.
- Select only complete episodes at the exact resolved budget (`520` StackCube,
  `2820` Airplane); no expert suffix or temporal tail may be sliced.
- Train three independent 2500-step policies per task--method from the same base,
  with the existing source-balanced temporal-mask contract, then evaluate frozen
  100-ID/100-OOD splits and summarize endpoint SR (Airplane primary endpoint:
  `ever_grasped`, strict completion retained as an auxiliary row).
- A durable Stage-C total supervisor launches these sub-stages and finally runs an
  independent denominator/checkpoint/budget reconciliation before writing
  `PIPELINE_COMPLETE`.

### Stage D：Airplane confirmation

- Execute the same calibration and utility procedure after StackCube Stage A/B artifacts are frozen, without changing anchors, knee algorithm, threshold rule, or budget rule.
- A negative StackCube scientific result is retained; it does not authorize changing the procedure before Airplane.

## 6. Completion and forbidden actions

`PIPELINE_COMPLETE` requires both task branches to have complete calibration, gate audit, matched-budget training, 3-seed utility evaluation, independent reconciliation, and final report. A detector score, alarm timing table, finite loss, checkpoint, or partial task branch is not completion.

Forbidden: Stage-2 timing artifacts as formal input; test/OOD threshold selection; changing anchors after observing outcomes; slicing suffixes; mixing old task/backbone/protocol data; substituting Offline BC for a gate; claiming task-universal knee from one task--policy calibration.
