# X-VLA Fixed-Grid Stage-C Airplane Asset Retry

日期：2026-08-27（北京时间）  
pipeline：`xvla_fixedgrid_taskpolicy_knee_v1`  
性质：工程修复记录；不构成科学结果。

## 失败根因

Stage-C 被动 gate audit 在建立 Airplane 缺失的 `vlm_input_pool` multilayer asset 时失败。Airplane metadata 明确使用 `robot_type=panda_airplane`，但 builder 在迭代 `InfiniteDataReader` 前没有注册该 handler，触发：

```text
KeyError: "No handler registered for dataset 'panda_airplane'"
```

第一次失败还留下了空的 `feature_cache` scaffold，第二次 retry 因控制器将该空目录判定为 partial output 而停止。两次原始日志均保留在远端：

`/data/zhaozhixuan/Ask4Help-airplane-5090/results/xvla_fixedgrid_taskpolicy_knee_v1/stage_c_gate_v1/logs/build_airplane_multilayer_assets.log`

## 修复与验证

- 在 Ask4Help 源码中加入可审计的 `panda_airplane` handler 注册模块；
- 修正 builder 的 `sys.path` 顺序，保证导入的是 Ask4Help handler；
- 增加注册断言，防止在第一 batch 才暴露错误；
- 允许仅含空 `feature_cache` 的失败 scaffold 安全重试，但对真实 feature/asset 文件仍 fail-closed；
- 远端按 builder 的真实导入顺序读取第一条 Airplane sample，验证通过：`BUILDER_IMPORT_ORDER_SMOKE_OK`；sample 包含双视角 image、20D proprio、10×20 action 和 10-step temporal mask；
- retry3 控制器 PID `1455702` 已启动，当前等待 GPU5 上另一个用户进程释放；未修改 task、checkpoint、threshold、anchor、success predicate 或 Stage-B 结果。

## 当前边界

修复 smoke 只证明 dataset handler 和 builder import path 正确，仍需要实际 GPU asset build、四组 passive gate rollout、q=.95 calibration 和 gate-to-knee summary 全部完成后，Stage-C 才能通过其完成标记。Stage-C 的任何科学 rows 在此之前均不接受。
