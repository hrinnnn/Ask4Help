# X-VLA Fixed-Grid Task-Policy Knee Validation

**状态：已认领；当前阶段为 StackCube calibration 的运行时诊断与重试。**

历史成功的 StackCube X-VLA collection 使用
`/data/zhaozhixuan/envs/xvla_official_5090/bin/python`。固定网格首次 smoke
在 RLinf 自带 `.venv` 中于 `mplib` planner 初始化时发生段错误；切换到历史
环境后 step 0 已写出完整 collection evidence，但在 SAPIEN teardown 阶段返回
`-6`。因此 controller 只在完整 summary/episode denominator 与 dataset 已写出
时记录并接受该 teardown abort；任何采样中途异常仍然失败并保留 diagnostic root。

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

### Airplane

- threshold-validation ID：`159000--159049`。
- calibration OOD failure cohort：`160000--160019`。
- held-out gate OOD：`161000--161049`。
- final ID：`163000--163099`。
- final OOD：`162000--162099`。

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
- Train/evaluate gate-selected datasets at the same resolved task budget only after Stage B artifacts pass.

### Stage D：Airplane confirmation

- Execute the same calibration and utility procedure after StackCube Stage A/B artifacts are frozen, without changing anchors, knee algorithm, threshold rule, or budget rule.
- A negative StackCube scientific result is retained; it does not authorize changing the procedure before Airplane.

## 6. Completion and forbidden actions

`PIPELINE_COMPLETE` requires both task branches to have complete calibration, gate audit, matched-budget training, 3-seed utility evaluation, independent reconciliation, and final report. A detector score, alarm timing table, finite loss, checkpoint, or partial task branch is not completion.

Forbidden: Stage-2 timing artifacts as formal input; test/OOD threshold selection; changing anchors after observing outcomes; slicing suffixes; mixing old task/backbone/protocol data; substituting Offline BC for a gate; claiming task-universal knee from one task--policy calibration.
