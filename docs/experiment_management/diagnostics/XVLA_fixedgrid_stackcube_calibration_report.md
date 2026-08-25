# X-VLA StackCube fixed-grid calibration report

这是 StackCube canonical OOD policy 的 fixed-step calibration，使用 20 个冻结
seed `150000--150019` 和 anchors `0,10,20,30,45,60,80,100`。旧 Stage-2
timing 数据未作为输入。

## Evidence

- raw calibration root：
  `/data/zhaozhixuan/Ask4Help-airplane-5090/results/xvla_fixedgrid_taskpolicy_knee_v1/formal_calibration_retry1/calibration/`
- independent audit：`calibration_audit.json`
- recoverable-frontier knee summary：`knee_summary_recoverable.json`
- runtime：`/data/zhaozhixuan/envs/xvla_official_5090/bin/python`
- 所有 anchor 的 20 条 raw episodes、actions、task states 和 videos 均存在；
  `free(): invalid pointer` 只发生在证据写完后的 SAPIEN teardown，并单独记录为
  accepted teardown abort。

## Independent audit

| fixed step | recoverable expert continuations | recoverability | frontier status |
|---:|---:|---:|---|
| 0 | 20/20 | 1.00 | valid |
| 10 | 20/20 | 1.00 | valid |
| 20 | 20/20 | 1.00 | valid |
| 30 | 20/20 | 1.00 | valid |
| 45 | 19/20 | 0.95 | valid |
| 60 | 18/20 | 0.90 | valid, boundary |
| 80 | 14/20 | 0.70 | `UNRECOVERABLE_REGION` |
| 100 | 9/20 | 0.45 | `UNRECOVERABLE_REGION` |

## Knee summary

在通过 recoverability gate 的 anchors `{0,10,20,30,45,60}` 上，用共同成功轨迹、
task-state DTW deviation、expert-action cost 和 5,000 次 bootstrap 计算离散 knee：

- knee anchor：`env_step=20`；
- bootstrap selection probability：`0.9048`；
- confidence set（`p>=0.10`）：`{20}`；
- 该结果是 task-policy-specific calibration，不代表 downstream training utility
  已经验证，也不支持跨 task 复用。

因此，StackCube 当前可以报告为：存在一个稳定的可恢复 timing window，较晚的
`80--100` 已不可恢复；`20` 是当前 time--deviation trade-off 的 calibration knee。
下一步若要声称“对学习最优”，仍需 matched-budget timing utility 或至少一次
独立 downstream update 验证。

## Matched-budget stop marker

按冻结规则 `20 * median(nominal expert actions)`，StackCube requested budget 为
520 actions；在所有有效 anchors 的共同 recoverable seed intersection 上，
不切任何完整 expert suffix 时的最大共同 exact budget 只有 234（45.0%），
低于 80% resolution gate。因此 Stage-B matched-budget training 同样被标记为
`BUDGET_RESOLUTION_FAILED`，当前 calibration 不能直接推出 downstream SR 提升。
