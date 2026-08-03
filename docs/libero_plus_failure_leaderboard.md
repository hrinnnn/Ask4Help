# LIBERO-Plus 无训练 Failure Detection 榜单

## 目的

在官方 `pi05_libero` checkpoint 上，评测 detector 对 policy failure 的被动检测能力；detector 不改变动作、不请求专家、不更新模型。主榜只使用成功 expert reference 与成功 policy calibration，绝不使用失败 trajectory 拟合分布或设置阈值。

## 冻结协议

| 项目 | 固定设置 |
| --- | --- |
| 任务 | LIBERO-10 与 LIBERO-Plus 官方 `Camera Viewpoints`、`Robot Initial States`、`Objects Layout` 三类 |
| 策略 | 官方 `pi05_libero`，action horizon 10，执行/重规划 horizon 5 |
| expert reference | 主榜：LIBERO-10 全部 379 条官方 expert trajectory 的所有 observation；旧 `1,000 anchor / t=1` 资产仅作 preliminary 诊断 |
| calibration | 100 条与 reference/test 不重叠的 clean policy 自身成功 trajectory，split conformal `q=0.95` |
| 主方法 | bridge LLMD、bridge Deep kNN、bridge PCA residual |
| 基线 | Action Expert final LLMD、ACC、VLA-FAIL `final LLMD OR ACC`、STAC-Single |
| 附录 | Action Total Variance，`C=10`，单列额外采样延迟 |
| 分数 | 每个 decision point 原始分数、固定阈值与首次报警均持久保存 |

`bridge` 指 final VLM prefix 表示经 valid-token mean pooling 后、送入 Action Expert 的 2048 维表示；`final` 指 action token 经过 Action Expert final block、进入 action projection 前的 1024 维表示。特征 probe 在固定高斯 action prior 与固定 flow timestep `t=0` 上运行，和 policy 原始 action sampling 分离，因此 detector 不会改变动作。末尾 observation 也进入 reference bank；官方 action-horizon padding 以 `action_indices`、`action_is_pad` 和 `tail_padding_count` 保存用于审计，而不是被丢弃。

## 产物与恢复

实验根目录为 `/data/zhaozhixuan/libero_plus_failure/results/`：

```text
libero10_all_observation_reference_v1/
  reference_bank_request.json
  shards/worker_{00,01}/episode_*.{npz,json}
  validation.json
  assets/reference_assets.{pt,json}
  calibration/thresholds.json
  camera_difficulty5/ and object_layout_difficulty5/
  scored/<category>/{scored_episodes.json,summary.json}
  videos/<TP|TN|FP|FN>/*.mp4
libero10_expert_feature_bank_v1/ and libero10_reference_assets_v1/
  # deprecated preliminary 1,000-anchor / t=1 assets; never use for main claims
```

每个 episode 目录是不可覆盖的完成单元，必须同时有 `episode.json`、`features.npz`、`rollout.mp4` 才能被批量启动器视为完成。`run_passive_batch.py` 会跳过完整单元、拒绝覆盖不完整目录，并为每次启动留下 `launcher_events.jsonl` 与独立日志。

## 运行顺序

1. 下载/校验官方 LIBERO-Plus assets；该资源固定到官方仓库 commit 与压缩包 SHA256。
2. 用两张空闲 GPU 运行 `build_all_observation_feature_bank.py`，按 episode-id 分片提取全部 LIBERO-10 observation 的 feature-only probe；固定 prior 和 `t=0` 必须由 request manifest 绑定。
3. 用 `validate_all_observation_feature_bank.py` 验证无漏帧、无重复、十个 task 完整覆盖和所有 terminal tail；随后用 `fit_all_observation_reference_assets.py` 流式拟合统计量并持久化精确 Bridge kNN bank。
4. 用 clean LIBERO client 调用 `run_passive_batch.py --mode calibration --successes-per-task 10 --required-successes 100`，直到每个 task 都得到 10 条 policy 自身成功轨迹；不足则明确失败，不伪造阈值。
5. 用 `score_passive_rollouts.py --mode calibrate` 固化 threshold JSON 与 reference assets SHA。
6. 重新在隔离的 official LIBERO-Plus client 上跑 Camera Viewpoints difficulty-5 与 Objects Layout difficulty-5。此前 `t=1` rollout 不可和新资产混用；评测器会兼容原版的 `libero.libero` 与 Plus 的顶层 `libero` 官方包结构。
7. 离线 replay 已存的 raw features，输出表格、分任务/类别/成功失败分层、bootstrap 95% CI，并用 `render_score_video.py` 输出包含视频、多个分数曲线、阈值与 first alert 的案例。

## 解释约束

- 轨迹失败标签只用于最后的排行榜 metric，不能回流到 feature bank、阈值或 detector 参数。
- `AUCPR` 等同 JSON 中的 `average_precision`；同时写两个字段以避免表格歧义。
- `feature_probe_mean_ms` 是增加的一次确定性 PaliGemma probe 的开销；`policy_mean_ms` 是官方 action sampling。两者相加报告为主榜每个 decision 的总延迟。
- Action Total Variance 需要 `C=10` action sampling，必须另跑/另报开销，不能与单样本主榜比较时隐瞒额外计算。
- 全量 Bridge Deep kNN bank 必须在离线评分进程中只加载一次并常驻 CPU/GPU；不得在每个 decision 重建 reference tensors。
- 一旦环境在成功后自然 `done`，episode 停止；模型从未观察 `done`、success、剩余时长或评测标签。该 simulator 生命周期限制记录在最终报告中。
- 本榜单选定的三类 Plus 扰动没有 image corruption。非 root 服务器若 ImageMagick ABI 无法稳定加载，可显式设置 `LIBERO_PLUS_DISABLE_UNUSED_WAND=1`；此时仅替换官方 import-time 的未使用 Wand backend，任何 motion-blur 调用都会 fail-closed。所有该模式下的 episode 会写入 `environment_compat.unused_wand_motion_blur_stub=true`。
