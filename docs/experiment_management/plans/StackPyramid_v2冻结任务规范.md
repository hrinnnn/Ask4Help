# StackPyramid v2 冻结任务规范

**唯一实现入口：** `tools/stackpyramid_task.py`，运行时必须设置
`STACKPYRAMID_OOD_GEOMETRY=v2`。Timing sweep 与四方法 gated-DAgger
必须使用同一实现、同一 stage predicate 和同一 paired-reset 语义，不得在
各自脚本中复制或修改几何定义。

## 几何

ID 的三个 cube xy 中心为：

```text
red   (-0.020, -0.040)
green ( 0.020, -0.040)
blue  ( 0.000,  0.040)
```

每个 reset 仍使用相同的独立 ±0.008 m jitter、机器人初始化和非目标
cube 状态。每个 OOD 只移动一个目标 cube：

| Split | 受影响 cube | v2 xy shift |
|---|---|---|
| `stage1_ood` | red | `(0.045, 0.045)` |
| `stage2_ood` | green | `(0.060, 0.050)` |
| `stage3_ood` | blue | `(0.060, -0.080)` |

因此三个 split 的非目标因素保持 paired；v1 的更大 shift 只作诊断，不能
进入正式收集、训练或主表。

## 阶段谓词与门

| Split | Prefix | Target stage |
|---|---|---|
| `stage1_ood` | `red_grasped` | `red_lifted` |
| `stage2_ood` | `red_lifted` | `red_placed` |
| `stage3_ood` | `red_placed` | `blue_lifted` |

正式运行前必须重新完成 v2 Oracle、ID base policy、prefix/target-stage
locality audit 和 PCA OOD-dominant pilot。审计报告必须同时保存每个 split
的实际成功数、prefix completion rate 和 target-stage reach rate。任何
不满足协议门的结果只能标记为 diagnostic，不能启动正式 gated collection
或 training。

**共同源码版本：** 当前 v2 protocol recovery 使用
`codex/xvla-stackcube-stage2-ood` 的 commit `1522213`；后续 timing 与
四方法流程应从同一源码版本拉取，并在各自 manifest 记录该 commit。
