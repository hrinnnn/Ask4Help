# OpenDrawer Grasp-OOD Controlled Timing Sweep

**状态：用户已审核并批准修复后的 direct-grasp Oracle。现在用它重新收集六个正式 timing anchor
（`0/50/80/120/160/220`）的 Grasp-OOD expert suffix；旧 Oracle formal 数据不复用。收集与
共同预算审计已完成，当前按统一训练步数规则运行最多四卡的 adaptive training。**

本 pipeline 只研究 OpenDrawerRetrievePlace 的 `grasp_ood` 条件：保持 handle、drawer、
goal、robot、camera、prompt、norm、action contract 和 success predicate 不变，只把
物体 yaw 改为 `80--100` degrees。另一个 agent 的 detector/gate 研究不属于本 pipeline；
这里唯一改变的是 expert takeover step。

## 1. 科学问题

Grasp-OOD 的纯 policy rollout 应能够完成开抽屉前缀，但在抓取阶段暴露物体方向能力缺口。
本 sweep 检验：在相同 expert-action budget 下，立即接管、阶段边界接管和进入失败恢复后
接管是否产生不同的 expert data quality 与 downstream policy SR。

主要假设为：

1. `D_path` 在同一 Grasp-OOD expert reference 上的临界越阈时间 (T_D) 落在接近物体/抓取
   的阶段，而非 reset 或开抽屉阶段；
2. (T_D) 附近的接管相比立即接管使用更少的每轨迹专家动作，并保持接近的 OOD SR；
3. 晚于 (T_D) 的接管更容易产生 recovery-heavy suffix，并降低 OOD SR。

## 2. 已核验的 immutable base

- checkpoint：`/sdd/ask4help-open-drawer/results/open_drawer_pi05_v9_recovery_from5000_v4/training/v9_full_prompt/checkpoints/global_step_5000`
- portable weights：`actor/model_state_dict/full_weights.pt`
- task：`open the drawer, retrieve the blue object, and place it in the green tray`
- ID policy evidence：`61/100` strict success，100/100 episodes、videos、actions、states、
  timelines、reset metadata；这满足本 diagnostic 的 `>50%` base 条件，但不重新标记为
  `ID_BASE_VALIDATED`（正式 80% 门槛仍单独记录）。
- norm：`/data/zhaozhixuan/Ask4Help-open-drawer/results/id_policy_training_v1/norm_stats_open_drawer_id_raw_v1`
- pi0.5 base：`/data/zhaozhixuan/Ask4Help-open-drawer/results/model_cache/pi05_base_pytorch_v1`
- runtime：`/data/zhaozhixuan/Ask4Help-airplane-5090/RLinf/.venv/bin/python`；planner 使用
  `/data/zhaozhixuan/simplerenv_ms3/env/bin/python`；训练控制器在每个 job 启动前实时审计并锁定
  实际空闲 GPU（训练池避让预留给 OOD20 的 GPU2/5），最多同时运行两个 world-size-1 job（每个 Ray 实例对象存储 200 GB；在 504 GB
  `/dev/shm` 通过四-job 审计前保持两-job上限）。用户授权的最多四卡只作为安全上限，绝不占用有其他
  进程的 GPU；CPU sets 按 GPU 固定为 `0--19,20--39,...,140--159`。所有 seed、anchor、预算和训练步数保持不变。

## 3. 现有 OOD evidence

`grasp_ood` 的 100 条纯 policy rollout 已完成且 evidence 完整：

- final strict success：`0/100`；
- drawer opened：`96/100`；
- ever grasped：`11/100`；
- ever lifted：`0/100`。

这说明该 split 主要在抓取阶段暴露能力缺口，适合进行 timing sweep。20 条 Grasp-OOD
expert Oracle reference 也已通过完整成功和动作/视频审计。

## 4. Fixed timing conditions

诊断和正式阶段暂定使用以下 anchor steps；这些点在 downstream SR 之前冻结：

| Step | 语义 |
|---:|---|
| 0 | immediate start |
| 50 | pre-drawer-open |
| 80 | post-drawer-open |
| 120 | object-approach |
| 160 | near-grasp boundary |
| 220 | post-failure-recovery |

这些 anchor 来自已有 policy event distribution 与 expert stage duration，而不是观察
最终 SR 后挑选。若 diagnostic 明确显示某个点不属于对应阶段，只能写入新的 manifest revision，
不能静默移动原 anchor。

每个 fixed anchor 从同一 Grasp-OOD base policy rollout 前缀 fork；到 scheduled step 后，
privileged expert 从当前 simulator state 继续到严格成功或不可恢复。提前终止、planner
失败、空 suffix 和不成功 continuation 均保留在 raw denominator，不进入 accepted expert
dataset。

## 5. D-path and timing metrics

对每条完整 evidence 保存 task-relative TCP pose/orientation、gripper width、drawer/object/
target state 和 lifecycle events。给定同 split expert bank，使用 causal monotone phase
alignment 计算：

\[
D_i(t)=\operatorname{dist}\left(z^{\pi}_{i,t},\mathcal M^{E}_{\mathrm{grasp\text{-}OOD}}\right).
\]

阈值只来自成功 ID calibration，使用连续两个 decision points 越阈：

\[
T^D_i=\min\{t:D_i(t)>\tau_D,\;D_i(t+1)>\tau_D\}.
\]

对 anchor (a) 报告 (D_i(a))、(D_i(a)/\tau_D)、(a-T^D_i)、expert suffix length、
EAS、DCA、success 和 failure phase。Grasp/contact failure 若不能由 pose-only D 识别，保留
pose-only 结果并单独报告 contact/progress auxiliary channel，不调整 pose threshold 凑 coverage。

## 6. Diagnostic stage

- 每个 anchor 接受 5 条成功 expert continuations，最多 10 次 raw attempts；
- 使用同一 OOD seed start `78100`，保留 raw attempt、failure reason、video、actions、
  states、task-state timeline；
- 先核对 prefix phase、planner continuation、D-path crossing、dataset shape 和
  action/state/video 对齐；
- diagnostic 通过后才扩展到每个 anchor 30 条 accepted episodes；不足的 anchor 保留为
  `UNRECOVERABLE_REGION`，不得为了凑齐分母改变 policy prefix 或 planner 协议。

## 7. Matched-budget training and evaluation

正式阶段每个 anchor（本轮重新收集，Oracle 为 `direct_grasp`）：

- 30 accepted Grasp-OOD suffixes，完整 episode only；
- 六个 anchor 的共同专家动作预算重新由新 Oracle 数据的完整 episode length 计算最大共同可达
  exact sum，并在训练前冻结；不复用旧 Oracle 的 `5006`，也不根据 SR 或 D-path 结果调整；
- 与固定 128-ID dataset 组合，source-balanced `1:1`；
- 在所有 anchor 之间选择同一 exact whole-episode expert-action budget；所有 checkpoint 使用
  同一套新 Oracle budget-selected 数据；
- 每个 anchor 只训练一个模型，全部使用共同 seed `9301`；每个模型至少训练 `5000` 步，
  每次 Grasp-OOD20 严格成功率 `<=40%` 时追加 `2500` 步；首个超过 `40%` 的累计步数冻结
  并用于所有 anchor；
- 使用冻结 ID 与 Grasp-OOD seeds，各 100 episodes 评测；
- 主 endpoint 为 strict OpenDrawer success，辅助报告 drawer-opened、grasp、lift、
  in-target rates。

在训练过程中，独立的 OOD20 probe controller 会对每个已经完成的 timing checkpoint
运行 20 条 Grasp-OOD episode（在 GPU2/5 中选择第一张经审计的空闲卡，CPU40--59，seed 从 79000 起）。该 probe 只用于
提前观察已训练模型的 OOD 行为，拥有独立目录和分母，不替代正式的 100-ID/100-OOD
评估，也不参与 timing anchor 或 D-path 阈值选择。

自 2026-08-29 起，用户明确要求“出现新的 checkpoint 后先评估，不要急着启动下一个
checkpoint 的训练”。因此训练控制器在每个最终 `global_step_2500` checkpoint 写出后，
必须等待对应的 `ood20_probe/{condition}/seed_{seed}/OOD20_COMPLETE`，通过该 20 条
诊断评测的独立产物审计后才可启动下一个 timing job。若探针资源暂时不可用，训练保持
等待而不改变 seed、anchor、预算、success predicate 或资源候选；若探针失败则 fail closed，
不把中间 checkpoint 当作可报告的 SR。正式 100-ID/100-Grasp-OOD 评估顺序和分母不变。
当前正在运行的旧版控制器在该决定后由一次性的边界控制器安全接管：它只暂停训练父
shell，让正在计算的子进程完成当前 checkpoint，随后替换总控/并行 helper 为包含等待门
的版本；不终止正在计算的子进程，不改变已登记 GPU 候选。该接管过程写入独立状态和日志，
若边界 checkpoint 或旧 PID 校验失败则 fail closed。

### 2026-08-30：自适应训练步数与交替评测

用户进一步规定：每个 timing checkpoint 至少训练 `5000` 步；若 20 条 Grasp-OOD
诊断评测的严格成功率仍为 `<=40%`，就在同一模型上继续增加 `2500` 步并再次评测。
第一个严格成功率超过 `40%` 的模型所用的累计训练步数记为冻结步数 `S^*`；之后所有
anchor/seed 都训练到相同的 `S^*`，每个 checkpoint 评测完成后才启动下一个训练作业。
该规则在观察新结果前预先冻结；旧 retry5 的 2500-step 结果只作 diagnostic，不混入
adaptive formal comparison。20-OOD 仍不替代最终 100-ID/100-Grasp-OOD 分母。

每个 anchor 只训练一个模型，不再对同一 anchor 重复多个 seed。所有 anchor 使用同一个
冻结训练 seed `9301`，以避免 seed 差异混入 timing 效应；因此最终比较的单位是六个
anchor 模型，而不是 18 个 seed replicate。旧 retry5 的多 seed 结果仅作 diagnostic。

retry6 的训练没有进入模型更新：Ray 在初始化时因临时目录过长而触发 AF_UNIX 路径限制。
该失败根与空的 partial output 均保留为工程 diagnostic；retry7 仅把 Ray/TMP scratch
迁移到短的服务器路径，科学变量（checkpoint、seed、anchor、5000/+2500 规则、预算和
success predicate）全部不变。

用户随后允许最多四张实际空闲 GPU 并行训练，以缩短 wall time。retry7 已在修复后的
短路径上启动 anchor 0，但在产生正式 checkpoint 前因调度策略切换而停止；其已写出的
500/1000/1500/2000 partial checkpoints 不进入正式结果。retry8 使用同一 seed 9301
为六个 anchor 排队，最多四个模型同时训练到 5000 步；每个完成的 checkpoint 仍先做
20 条 OOD 审计，pilot anchor 0 的首个严格 SR `>40%` 仍唯一决定冻结步数。

retry8 在训练启动前发现正式 collection marker 位于 formal 根目录的父目录，因预检路径
不一致而退出；没有产生训练或评测产物。retry9 将 marker 检查同时兼容根目录与父目录，
其余科学变量保持不变。

retry9 已启动两个单卡作业，但每个 Ray 实例默认申请约 200 GiB object store；在
`/dev/shm≈504 GiB` 上继续扩展到四卡存在确定的共享内存风险，因此在正式 checkpoint
前停止并保留为 diagnostic。retry10 为本任务注入 100 GiB/作业的 Ray object-store
上限（四作业约 400 GiB），并先通过独立 Ray 初始化 smoke；这只改变运行时资源配置，
不改变训练数据、seed、anchor、步数规则或成功定义。

自适应训练完成后，持久化的 `run_open_drawer_adaptive_formal_eval_controller.sh` 会按
anchor 顺序对每个模型运行冻结的 `100` 条 ID 与 `100` 条 Grasp-OOD rollout；每一对
评测在一张重新审计的空闲 GPU 上完成，ID/OOD 使用不重叠的 seed 区间。评测结束后由
`summarize_open_drawer_adaptive_timing.py` 独立核对 checkpoint、视频/actions/states/
timeline 分母、D-path 与 intervention-quality 指标，只有 reconciliation 通过才写入
`INDEPENDENT_RECONCILIATION_COMPLETE` 和最终报告。

主表为：

| Timing | (D/\tau_D) | (T_g-T_D) | EAS | DCA | ID SR | Grasp-OOD SR |
|---|---:|---:|---:|---:|---:|---:|

同时保留 `No takeover` policy-only baseline。anchor-level Pearson/Spearman 只作探索性
描述；正式论点依赖 paired fixed-budget SR、专家动作成本和 late-vs-boundary 对照。

## 8. Direct-grasp Oracle repair validation

旧 continuation 在 drawer 已打开时仍会执行固定的 handle retreat，再进入物体抓取，
并且默认 planner 优先使用 screw path，容易造成不必要的抬升和姿态翻转。新增的
`direct_grasp` 模式从当前 takeover 状态直接规划 object pre-grasp；只有 drawer 尚未
打开时才执行必要的 handle opening。object/lift/transport 使用 shortest-joint-path，
并在物体 closing axis 的正反两个候选中选择较短路径。

修复后的 diagnostic root 为
`open_drawer_grasp_timing_sweep_v1_direct_oracle_retry4`，六个 anchor 各有 3/3
accepted success（18 条 accepted、24 条 raw videos）。这些视频只用于用户审核，
不覆盖旧 formal collection，也不自动解锁 adaptive training。

### 8.1 已抓住把手的继续拉动分支与扩展验证

当 takeover 时抽屉尚未达到开启阈值，但双指已经同时接触 `drawer_link` 且 TCP 到当前黄色把手
位置的距离不超过 `0.075\,m`，Oracle 将记录
`handle_grasped_before_takeover=true`，保持闭合夹爪并从当前 TCP 直接规划拉动；它不会重新执行
handle pregrasp/reach，也不会先抬升把手。这个几何门控用于排除抽屉前板的偶然接触。该分支已在
现场仿真状态中验证：继续拉动后抽屉 qpos 从约 `0` 到 `-0.363`，TCP 的 z 变化约 `0.006\,m`，
且 `direct_handle_pregrasp_steps=0`。

为检查 Oracle 不依赖单一 Grasp-OOD 分布，新增 diagnostic video sweep 覆盖三个受控 OOD split
（`grasp_ood`、`handle_ood`、`goal_ood`）和五个 takeover step（`0,80,160,220,300`），每个条件
目标接受 2 条成功 continuation，最多 6 条 raw attempts；实际得到 30 条 accepted、33 条 raw 视频，
每个条件均达到 2 条 accepted。每个条件单独保存 summary、raw attempts、task-state timeline、
actions/states 与视频；不改变 formal 分母或训练协议。用户审核这些视频后，才决定是否解锁
adaptive training。

### 8.2 正式 expert 重收集与训练切换

用户已批准将 `direct_grasp` Oracle 作为正式 expert。正式重收集使用同一个 ID checkpoint、同一
组 Grasp-OOD policy-prefix seed 和原冻结 anchor `{0,50,80,120,160,220}`；每个 anchor 目标
30 条成功 continuation，最多 80 条 raw attempts。输出写入新的
`open_drawer_grasp_timing_sweep_v1_direct_oracle_formal_retry1/formal`，并先通过独立的
denominator/evidence audit，再由 exact whole-episode selector 生成新的 `formal_budget`。
旧 `/open_drawer_grasp_timing_sweep_v1_formal/formal` 只作 diagnostic，不进入新训练。
本次 selector 已得到并冻结共同预算 `2413` 个 expert actions；六个 anchor 的 selected sum
完全相同，规则为最大共同可达整轨迹和。

训练使用每个 anchor 一个模型、共同 seed `9301`、最多四张实时核验为空闲的 GPU。每个模型至少
训练 5000 步；每个 checkpoint 先做 20 条 Grasp-OOD 诊断评测，严格成功率 `<=40%` 时追加
2500 步，首个 `>40%` 的累计步数冻结，并让所有 anchor 使用同一冻结步数。训练、OOD20、后续
100-ID/100-Grasp-OOD evaluation 继续保持交替和独立产物审计。

第一次 direct-Oracle adaptive 启动在 Ray 初始化时因 scratch 路径仍导致 AF\_UNIX socket 超过
107 字节而退出；该日志保留为 engineering diagnostic。重试只把运行时 scratch 缩短为
`/sdd/r_od1` 与 `/sdd/t_od1`，不改变 checkpoint、数据、seed、anchor、budget、训练步数或
成功定义。

## 9. Completion

只有以下内容全部存在才写入 `PIPELINE_COMPLETE`：

1. 所有 anchor 的 raw/accepted denominator 与 evidence audit；
2. exact budget selection manifest；
3. 六个 timing anchor 的单模型 training checkpoints 和 reload/forward audit；
4. 所有 ID/OOD evaluation 的 100/100 video/action/state/timeline evidence；
5. D-path、EAS、DCA、SR 汇总和独立 reconciliation report。

cleanup 阶段的 `free(): invalid pointer` 只在完整 artifacts 已写出后记录为 engineering
diagnostic，不改变科学分母。
