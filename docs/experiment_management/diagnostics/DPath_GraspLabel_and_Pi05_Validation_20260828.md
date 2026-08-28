# `D_path` grasp-label correction and pi0.5 validation

日期：2026-08-28（北京时间）  
性质：diagnostic-only；没有修改任何 formal checkpoint、training、gate threshold、
success predicate 或 downstream result。

## 1. Fixed-grid Grab Plane 标签纠正

旧分析直接使用 `ever_grasped` 作为 success label。逐轨迹 pose 和关键帧审计显示，
这个标签把瞬时接触、错误部位抓取和稳定中段抓取混在一起：

| Visual/pose phenotype | Episodes | Definition |
|---|---:|---|
| Stable mid-body grasp | 2 | 连续 grasp `>=20` frames，且 grasp 时 object lift `>2 cm` |
| Transient/wrong grasp | 22 | 曾出现 grasp flag，但不满足稳定抓持和抬升 |
| No grasp | 26 | 全程没有 grasp flag |

旧 `ever_grasped=true` 的 24 条中，只有 seeds `161008`、`161034` 满足 stable
criterion；原视频使用的 seed `161000` 只有两帧 grasp，应归入 transient group。

纠正分组后的前 0--40 step median position `D_path`：

| Group | Median `D_path` |
|---|---:|
| Stable mid-body grasp | 1.88 |
| Transient/wrong grasp | 6.60 |
| No grasp | 3.67 |

因此旧图中“成功曲线比失败曲线高”的主要原因是 success label 过宽。纠正后稳定抓持
轨迹最接近 expert path，transient grasp 偏离最大；但 no-grasp 仍可沿 nominal TCP
位置路径运动，很多点低于 expert q95=`7.33`，所以 position-only `D_path` 仍不是完整
grasp/contact oracle。

纠正视频：

`/Users/zhaozhixuan/.codex/visualizations/2026/08/25/01a037ef-1a71-7ad3-9594-a7dcfe06ad7b/dpath-cross-task/fixedgrid-grabplane-dpath-relabel-audit.mp4`

## 2. pi0.5 PickSingleYCB-Airplane

### 2.1 Evidence

- policy：pi0.5 Internal-PCA experiment 的 frozen step-5000 detector policy；
- expert：原 ID expert bank 中 20 条完整 parquet states；
- policy replay：50 ID + 50 yaw-OOD saved actions；
- replay output：20/50/50 pose files，`REPLAY_COMPLETE`；
- 所有轨迹先截取共同前 80 environment steps，再估计双侧 phase band；
- stable grasp 仍定义为连续 `>=20` grasp frames 且 grasp 时 lift `>2 cm`。

远端 replay root：

`/mnt/data/ask4help/results/dpath_pi05_airplane_v1/full_replay_v2/`

### 2.2 Result

| Metric | Result |
|---|---:|
| Stable ID trajectories | 24/50 |
| Stable OOD trajectories | 0/50 |
| Calibration / held-out stable ID | 12 / 12 |
| Held-out stable-ID false crossing | 0/12 |
| `tau_path` | 34.99 |
| OOD persistent crossing | 32/50 |
| OOD `t_PD` median | 50 |
| OOD `t_PD` P25--P75 | 45--55 |
| OOD censored | 18/50 |

pi0.5 Airplane 的 OOD position-path curve 从约 step 20 开始上升，32/50 在 horizon 80
内持续越阈；相比 X-VLA Grab Plane 的 no-grasp 0/26 crossing，这个 pi0.5 policy 的
yaw-OOD failure 更容易形成可见空间路径分叉。但 18/50 仍然 censored，不能称为完整
failure detector。

视频：

`/Users/zhaozhixuan/.codex/visualizations/2026/08/25/01a037ef-1a71-7ad3-9594-a7dcfe06ad7b/dpath-cross-task/pi05-airplane-dpath-validation.mp4`

## 3. pi0.5 StackCube

### 3.1 Evidence

- policy：historical Internal-kNN downstream policy，checkpoint `global_step_2000`；
- expert：20 条 OOD Offline-Oracle full demonstrations；
- 新 diagnostic evaluation：20 ID + 20 OOD，保存 actions/videos/summary；
- policy result：ID `17/20` strict success，OOD `18/20` strict success；
- expert state episodes 只有 20 或 30 samples，因此本次仅比较共同前 15 steps。

远端 roots：

- expert pose：`/mnt/data/ask4help/results/dpath_pi05_stackcube_v1/expert_ood20/`
- policy evaluation：`/mnt/data/ask4help/results/dpath_pi05_stackcube_v1/policy_eval_v1/`
- policy pose：`/mnt/data/ask4help/results/dpath_pi05_stackcube_v1/policy_pose_v1/`

### 3.2 Result

| Metric | Result |
|---|---:|
| Calibration / held-out successful ID | 9 / 8 |
| Held-out successful-ID false crossing | 0/8 |
| `tau_path` | 27.29 |
| OOD success / failure | 18 / 2 |
| OOD failure crossing | 0/2 |

该 policy 已经在 18/20 OOD resets 上成功，只有两个失败样本；success/failure position
curve 在共同前 15 steps 内基本重叠。当前资产不足以验证 StackCube failure onset，不能把
`0/2` 解释为指标失败或成功。

视频：

`/Users/zhaozhixuan/.codex/visualizations/2026/08/25/01a037ef-1a71-7ad3-9594-a7dcfe06ad7b/dpath-cross-task/pi05-stackcube-dpath-small-sample.mp4`

## 4. Judgment

1. Grab Plane 旧视频和旧 success distribution 必须由 corrected grasp phenotype 替换。
2. pi0.5 Airplane 提供了比 X-VLA Grab Plane 更强的 `D_path` separation，但 coverage
   只有 64%，仍需把 no-crossing 作为 censored 报告。
3. pi0.5 StackCube 当前 policy 太强、failure denominator 太小，无法完成 onset validation。
4. `D_path` 应继续定位为 spatial trajectory divergence reference；grasp/contact outcome
   必须由独立稳定抓持或 object-progress channel 描述。
