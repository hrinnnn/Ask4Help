# X-VLA PickSingleYCB-Airplane fixed-grid calibration report

这是 canonical X-VLA Grab Plane/airplane OOD task 的 held-out fixed-step
calibration。使用冻结 OOD seeds `160000--160019`，成功 endpoint 为
`ever_grasped`，但只有在 expert suffix 非空时才计入 recoverability；没有使用
detector alarm timing 作为 timing ground truth。

## Evidence

- raw calibration root：
  `/data/zhaozhixuan/Ask4Help-airplane-5090/results/xvla_fixedgrid_taskpolicy_knee_v1/airplane_formal_calibration_retry1/airplane_calibration/calibration/`
- independent audit：`calibration_audit.json`
- recoverable-frontier knee summary：`knee_summary_recoverable.json`
- all eight anchors have 20 raw episodes, task states, videos and action records;
  native teardown `-6` is recorded only after these artifacts are flushed.

## Independent audit

| fixed step | recoverable expert continuations | recoverability | frontier status |
|---:|---:|---:|---|
| 0 | 20/20 | 1.00 | valid |
| 10 | 20/20 | 1.00 | valid |
| 20 | 20/20 | 1.00 | valid |
| 30 | 19/20 | 0.95 | valid |
| 45 | 19/20 | 0.95 | valid |
| 60 | 15/20 | 0.75 | `UNRECOVERABLE_REGION` |
| 80 | 5/20 | 0.25 | `UNRECOVERABLE_REGION` |
| 100 | 7/20 | 0.35 | `UNRECOVERABLE_REGION` |

## Knee summary

在有效 anchors `{0,10,20,30,45}` 上，使用共同成功轨迹、task-state DTW、
expert-action cost 和 5,000 次 bootstrap：

- knee anchor：`env_step=20`；
- bootstrap selection probability：`0.9812`；
- confidence set（`p>=0.10`）：`{20}`；
- 共同成功 seed 数：18。

因此 Airplane held-out calibration 与 StackCube 都支持 task-policy-specific
的 `step=20` time--deviation knee，并显示更晚 timing 进入不可恢复区域。但这
仍然是 calibration evidence，不是 downstream policy utility 的证明。

## Matched-budget stop marker

按冻结规则 `20 * median(nominal expert actions)`，Airplane requested budget 为
2820 actions；在所有有效 anchors 的共同 recoverable seed intersection 上，
不切任何完整 expert suffix 时的最大共同 exact budget 只有 1561（55.4%），
低于 80% resolution gate。因此 Stage-B matched-budget training 被明确标记为
`BUDGET_RESOLUTION_FAILED`，不能用当前数据声称 knee 带来 downstream SR 提升。
