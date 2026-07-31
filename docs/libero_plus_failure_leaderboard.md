# LIBERO-Plus 无训练 Failure Detection 榜单

## 目的

在官方 `pi05_libero` checkpoint 上，评测 detector 对 policy failure 的被动检测能力；detector 不改变动作、不请求专家、不更新模型。主榜只使用成功 expert reference 与成功 policy calibration，绝不使用失败 trajectory 拟合分布或设置阈值。

## 冻结协议

| 项目 | 固定设置 |
| --- | --- |
| 任务 | LIBERO-10 与 LIBERO-Plus 官方 `Camera Viewpoints`、`Robot Initial States`、`Objects Layout` 三类 |
| 策略 | 官方 `pi05_libero`，action horizon 10，执行/重规划 horizon 5 |
| expert reference | 每 task 10 条成功官方 demo，每 demo 10 个均匀合法决策锚点，共 1,000 个锚点 |
| calibration | 100 条与 reference/test 不重叠的 clean policy 自身成功 trajectory，split conformal `q=0.95` |
| 主方法 | bridge LLMD、bridge Deep kNN、bridge PCA residual |
| 基线 | Action Expert final LLMD、ACC、VLA-FAIL `final LLMD OR ACC`、STAC-Single |
| 附录 | Action Total Variance，`C=10`，单列额外采样延迟 |
| 分数 | 每个 decision point 原始分数、固定阈值与首次报警均持久保存 |

`bridge` 指 final VLM prefix 表示经 valid-token mean pooling 后、送入 Action Expert 的 2048 维表示；`final` 指 action token 经过 Action Expert final block、进入 action projection 前的 1024 维表示。特征 probe 在固定高斯 action prior 与固定 flow timestep 上运行，和 policy 原始 action sampling 分离，因此 detector 不会改变动作。

## 产物与恢复

实验根目录为 `/data/zhaozhixuan/libero_plus_failure/results/`：

```text
libero10_expert_feature_bank_v1/
  expert_selection_manifest.json
  expert_feature_cache.pt
libero10_reference_assets_v1/
  reference_assets.pt
  reference_assets.json
clean_calibration_v1/
  episodes/.../{episode.json,features.npz,rollout.mp4}
  batch_summary.json
calibration_thresholds_v1/
  thresholds.json
libero_plus_passive_v1/ and clean_controls_v1/
  episodes/.../{episode.json,features.npz,rollout.mp4}
scored_v1/
  scored_episodes.json
  summary.json
  videos/<TP|TN|FP|FN>/*.mp4
```

每个 episode 目录是不可覆盖的完成单元，必须同时有 `episode.json`、`features.npz`、`rollout.mp4` 才能被批量启动器视为完成。`run_passive_batch.py` 会跳过完整单元、拒绝覆盖不完整目录，并为每次启动留下 `launcher_events.jsonl` 与独立日志。

## 运行顺序

1. 下载/校验官方 LIBERO-Plus assets；该资源固定到官方仓库 commit 与压缩包 SHA256。
2. 由官方 `physical-intelligence/libero` 选择 manifest 指定的 100 条成功 demo，提取一次 feature cache。
3. 用 `fit_reference_assets.py` 从同一个 cache 建立全部主榜 distribution/reference bank。
4. 用 clean LIBERO client 调用 `run_passive_batch.py --mode calibration`，直到得到严格的 100 条 policy 自身成功轨迹；不足则明确失败，不伪造阈值。
5. 用 `score_passive_rollouts.py --mode calibrate` 固化 threshold JSON 与 reference assets SHA。
6. 分别在隔离的 official LIBERO-Plus client 和 clean LIBERO client 上，以同一 official manifest 跑 Plus 扰动与匹配 clean controls。两套环境不能修改彼此的 `LIBERO_CONFIG_PATH` 或 assets；评测器会兼容原版的 `libero.libero` 与 Plus 的顶层 `libero` 官方包结构。
7. 离线 replay 已存的 raw features，输出表格、分任务/类别/成功失败分层、bootstrap 95% CI，并用 `render_score_video.py` 输出包含视频、多个分数曲线、阈值与 first alert 的案例。

## 解释约束

- 轨迹失败标签只用于最后的排行榜 metric，不能回流到 feature bank、阈值或 detector 参数。
- `AUCPR` 等同 JSON 中的 `average_precision`；同时写两个字段以避免表格歧义。
- `feature_probe_mean_ms` 是增加的一次确定性 PaliGemma probe 的开销；`policy_mean_ms` 是官方 action sampling。两者相加报告为主榜每个 decision 的总延迟。
- Action Total Variance 需要 `C=10` action sampling，必须另跑/另报开销，不能与单样本主榜比较时隐瞒额外计算。
- 一旦环境在成功后自然 `done`，episode 停止；模型从未观察 `done`、success、剩余时长或评测标签。该 simulator 生命周期限制记录在最终报告中。
- 本榜单选定的三类 Plus 扰动没有 image corruption。非 root 服务器若 ImageMagick ABI 无法稳定加载，可显式设置 `LIBERO_PLUS_DISABLE_UNUSED_WAND=1`；此时仅替换官方 import-time 的未使用 Wand backend，任何 motion-blur 调用都会 fail-closed。所有该模式下的 episode 会写入 `environment_compat.unused_wand_motion_blur_stub=true`。
