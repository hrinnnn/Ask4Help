# X-VLA ERD-Pose takeover timing diagnostic

日期：2026-08-27（北京时间）  
性质：diagnostic-only。该结果不覆盖原 fixed-grid formal Stage-C，也不修改其
threshold、seed、anchor、success predicate、expert budget 或 completion contract。

## 1. 目标

沿 learner 的 OOD policy-only trajectory，使用真机可获得的 gripper position、
orientation、gripper width 和 finite-difference velocity，判断第一次显著偏离
OOD expert reference 的时间。模拟器 phase flags 仅用于事后标注 pre-grasp/
post-grasp，不参与 ERD 分数。

## 2. Replay 证据

### Stage-C held-out OOD policy-only replay

| Task | action/video rows | action count | reset metadata | first RGB after codec resize |
|---|---:|---:|---:|---:|
| StackCube | 50/50 | 50/50 | 50/50 | 50/50 |
| Grab Plane | 50/50 | 50/50 | 50/50 | 50/50 |

RGB source video 为 `384x784`，replay camera frame 为 `384x772`；按相同视频编码
缩放后，首帧 MAE 均低于预设 `5.0`，因此没有发现 reset/state/video mismatch。

### Stage-A OOD expert reference replay

| Task | expert rows | reset metadata | first RGB | full action-count match | 备注 |
|---|---:|---:|---:|---:|---|
| StackCube | 20/20 | 20/20 | 20/20 | 20/20 | 完整 reference |
| Grab Plane | 20/20 | 20/20 | 20/20 | 7/20 | 原 expert source 在环境终止后仍保留尾部 actions |

Grab Plane 的每条 replay 仍覆盖至少 101 steps，因此 `horizon=50` 的 ERD timing
分析有完整 pose coverage；但超过实际 replay termination 的 expert tail 不注册
为 exact full-trajectory reconstruction。

## 3. ERD-Pose 定义

对每个 pose：

\[
z_t=[p_t^{ee},\ R_t^{ee},\ g_t,\ v_t^{ee},\omega_t^{ee}].
\]

姿态误差使用：

\[
\theta(R,R^E)=
\left\|\log\left((R^E)^\top R\right)\right\|_2.
\]

给定 nearest-context expert prototype 和 causal monotonic phase alignment，
对 learner/reference 的残差做 robust expert scaling：

\[
D_t^{\mathrm{ERD}}=
\left\|W_c
\begin{bmatrix}
p_t-p^E_{\hat\phi_t}\\
\log((R^E_{\hat\phi_t})^\top R_t)\\
g_t-g^E_{\hat\phi_t}\\
v_t-v^E_{\hat\phi_t}\\
\omega_t-\omega^E_{\hat\phi_t}
\end{bmatrix}
\right\|_2.
\]

phase index 只允许单调前进，并默认不看未来 reference phase；因此不是对整条
expert trajectory 做无约束全局最近距离。

阈值使用 expert leave-one-context-out residual 的 `q=.95` 作为本次 diagnostic
calibration，报警需要连续两个 5-step decision points 越阈值，且只在
`horizon=50` 内统计。

## 4. Context support

context 使用 episode 初始 object position 和 target position，按 nearest expert
prototype 匹配；support threshold 是 expert nearest-peer context distance 的
95th percentile。

| Task | support threshold | supported learner contexts | unsupported |
|---|---:|---:|---:|
| StackCube | 2.6845 | 48/50 | 2/50 |
| Grab Plane | 2.0331 | 50/50 | 0/50 |

unsupported context 不应被解释为“没有偏离”；本报告中的 ERD 数值仍保留，但
正式部署应将其标记为 `reference_unsupported` 并要求补充 prototype。

## 5. ERD-Pose 结果

### StackCube

- frozen ERD threshold：`9.8257`；
- first persistent crossing：`48/50`，其中 supported context 为 `46/48`；
- alarm step：mean `19.06`，median `20`，P25--P75=`15--25`；
- phase：`46` 条 pre-grasp，`2` 条 post-grasp/recovery；
- irreversibility：47 条 timeout，3 条 dropped-after-grasp；
- identifiable-event lead time：median `55` steps；2 条报警晚于已记录的早期 drop。

### Grab Plane

- frozen ERD threshold：`6.4120`；
- first persistent crossing：`50/50`；
- alarm step：mean `14.10`，median `15`，P25--P75=`10--20`；
- phase：`50/50` pre-grasp；
- irreversibility：24 条 grasp-lost，26 条 right-censored；
- 24 条 identifiable-event 的 lead time：median `61` steps；没有负 lead time。

这两个 task 的 ERD crossing 都位于 150-step episode 的早期；StackCube 约为
2.0 s，Grab Plane 约为 1.5 s（控制频率 10 Hz）。

## 6. 与现有 detector 的比较

以下为 Stage-C raw first-alarm timing；`lead` 只在可识别 irreversibility 的
episode 上统计，`late` 表示报警晚于该事件。

### StackCube

| Method | observed | mean step | median | pre / post | lead median | late |
|---|---:|---:|---:|---:|---:|---:|
| ERD-Pose | 48/50 | 19.06 | 20 | 46 / 2 | 55 | 2 |
| Input PCA | 50/50 | 40.90 | 35 | 48 / 2 | 35 | 10 |
| Bridge PCA | 50/50 | 5.60 | 5 | 48 / 2 | 70 | 2 |
| Action PCA | 50/50 | 28.50 | 20 | 47 / 3 | 55 | 3 |
| Diff-DAgger | 18/50 | 93.61 | 105 | 16 / 2 | -35 | 14 |
| Failure-Recovery | 50/50 | 50.00 | 50 | 47 / 3 | 25 | 2 |

StackCube 上 Bridge PCA 最早、覆盖最好；Action PCA 最接近 ERD median；ERD
本身提供了一个更直接的 task-state/pose timing reference。

### Grab Plane

| Method | observed | mean step | median | pre / post | lead median | late |
|---|---:|---:|---:|---:|---:|---:|
| ERD-Pose | 50/50 | 14.10 | 15 | 50 / 0 | 61 | 0 |
| Input PCA | 50/50 | 6.10 | 0 | 49 / 1 | 71 | 1 |
| Bridge PCA | 35/50 | 75.14 | 100 | 26 / 9 | -4 | 9 |
| Action PCA | 46/50 | 95.87 | 95 | 25 / 21 | -14 | 19 |
| Diff-DAgger | 44/50 | 88.86 | 90 | 25 / 19 | -14 | 16 |
| Failure-Recovery | 50/50 | 50.00 | 50 | 47 / 3 | 21 | 3 |

Grab Plane 上 Input PCA 的 step-0 报警不是合理的 pose deviation timing；其他
detectors 多数已经进入 post-grasp/recovery 或发生漏报。ERD-Pose 在当前 reference
下全部于 pre-grasp 报警，但仍需 real-robot validation。

## 7. Stage-B utility 交叉检查

ERD-Pose 是安全/时序 reference，不直接等于 learning utility optimum。现有
Stage-B OOD utility 为：

| Task | `t=0` | `t=10` | `t=20` | Stage-B best |
|---|---:|---:|---:|---:|
| StackCube OOD success | **0.7233** | 0.6000 | 0.4467 | 0 |
| Grab Plane OOD ever-grasped | **0.7967** | 0.6133 | 0.1800 | 0 |

因此 ERD-Pose 可以说明“何时开始偏离 expert pose manifold”，但不能单独证明
该时间训练出来的 policy 最优。两者必须分别报告。

## 8. 当前结论与限制

1. 现有 action/video/seed replay 足以在两个 task 上生成完整的 50 条 OOD pose
   timelines，并通过 reset metadata 与首帧 RGB 审计。
2. ERD-Pose 在 StackCube 给出约 `step=20` 的稳定早期 crossing，在 Grab Plane
   给出约 `step=15` 的稳定早期 crossing；所有有效 Grab Plane crossings 和
   绝大多数 StackCube crossings 都发生在 pre-grasp。
3. StackCube 有 2 个 context unsupported、2 个报警晚于早期 drop；这些不能被
   包装成完美安全指标。
4. Grab Plane expert reference 的完整尾部 replay 不可逐条复现，但 horizon=50
   的 pose coverage 完整；后续若研究长时段 lead time，需要重新记录或修复 expert
   source termination contract。
5. 本结果使用仿真 replay 验证方法链路；真机版本应直接记录 FK/encoder pose，
   并在同一 OOD distribution 上重新校准 tracking tolerance。

## 9. 产物

- replay implementation：`tools/replay_xvla_pose_timeline.py`
- replay controller：`tools/run_xvla_erd_pose_replay_controller.sh`
- analysis implementation：`tools/analyze_xvla_erd_pose.py`
- remote output root：`/data/zhaozhixuan/Ask4Help-airplane-5090/results/xvla_fixedgrid_taskpolicy_knee_v1/erd_pose_replay_v1/`
- ERD analysis root：`/data/zhaozhixuan/Ask4Help-airplane-5090/results/xvla_fixedgrid_taskpolicy_knee_v1/erd_pose_analysis_v1/`
- visualization：`xvla-erd-pose-final.html` in the thread visualization directory。
