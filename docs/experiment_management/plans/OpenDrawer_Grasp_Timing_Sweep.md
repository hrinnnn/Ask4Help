# OpenDrawer Grasp-OOD Controlled Timing Sweep

**状态：已授权，正在进行 checkpoint/asset audit 后的 diagnostic 阶段。**

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
  `/data/zhaozhixuan/simplerenv_ms3/env/bin/python`；matched-budget training uses at most two concurrent
  world-size-1 jobs on audited idle GPU4/CPU80--99 and GPU0/CPU0--19。每个 Ray 实例的对象存储配置为
  200 GB；在 504 GB `/dev/shm` 下不再增加并发卡数。所有 seed、anchor、预算和训练步数保持不变。

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

正式阶段每个 anchor：

- 30 accepted Grasp-OOD suffixes，完整 episode only；
- 六个 anchor 的共同专家动作预算冻结为 `5006`，由完整 episode length 的最大共同可达
  exact sum 决定，不根据 SR 或 D-path 结果调整；
- 与固定 128-ID dataset 组合，source-balanced `1:1`；
- 在所有 anchor 之间选择同一 exact whole-episode expert-action budget；
- 从同一 immutable checkpoint 独立训练 3 个 seeds，2500 update steps；
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

主表为：

| Timing | (D/\tau_D) | (T_g-T_D) | EAS | DCA | ID SR | Grasp-OOD SR |
|---|---:|---:|---:|---:|---:|---:|

同时保留 `No takeover` policy-only baseline。anchor-level Pearson/Spearman 只作探索性
描述；正式论点依赖 paired fixed-budget SR、专家动作成本和 late-vs-boundary 对照。

## 8. Completion

只有以下内容全部存在才写入 `PIPELINE_COMPLETE`：

1. 所有 anchor 的 raw/accepted denominator 与 evidence audit；
2. exact budget selection manifest；
3. 所有 timing policy 的 3-seed training checkpoints 和 reload/forward audit；
4. 所有 ID/OOD evaluation 的 100/100 video/action/state/timeline evidence；
5. D-path、EAS、DCA、SR 汇总和独立 reconciliation report。

cleanup 阶段的 `free(): invalid pointer` 只在完整 artifacts 已写出后记录为 engineering
diagnostic，不改变科学分母。
