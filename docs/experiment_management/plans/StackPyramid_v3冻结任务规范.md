# StackPyramid v3 冻结任务规范

> **已废止：diagnostic only。** v3 的 ID 红绿块初始距离小于 `next_to` 阈值，导致 `red_placed` 在 reset 后立即成立。不得继续使用 v3 的 checkpoint、数据、norm、calibration、collection 或 training；唯一后续设计见《StackPyramid v4 冻结任务规范》。

## 唯一几何与来源

> **已失效，仅作 diagnostic。** v3 的 ID 红绿块初始距离小于 `next_to` 阈值，导致 `red_placed` 在 reset 后立即成立，Stage 1/2 时序无效。不得继续使用 v3 demonstrations、norm、checkpoint 或门禁结果开展正式实验；正式后续转入 v4 重建。

StackPyramid v3 原共享 seed/config 文件为
`configs/stackpyramid_timing_v3_seed_manifest.json`，实现为 `tools/stackpyramid_task.py`；Timing
Sweep 与四方法流程必须使用同一份文件、同一源码提交和 `STACKPYRAMID_OOD_GEOMETRY=v3`。

v3 保留 v2 的 paired reset、Stage1/Stage2 定义、stage predicate 和 seed manifest，只把 Stage3 的
blue-block shift 改为 `[0.100, -0.120]`，避免 v2 的 `[0.060, -0.080]` 与红/绿块发生物理干涉。
v1 与 v2 产物只作 diagnostic，不得混入正式采集、预算匹配或训练。

## 门禁顺序

必须从同一 immutable ID checkpoint 重新完成：

1. ID、Stage1 OOD、Stage2 OOD、Stage3 OOD 各 100 条 Oracle；四个严格成功率均至少 90%。
2. 同一 checkpoint 的 base-policy ID/OOD 复测；ID 至少 90%，三个 OOD 均低于 90%，并保存实际成功数。
3. 三个 OOD 的 prefix completion 至少 80%，target-stage reach 必须大于 0。
4. Internal-PCA 只用固定 ID expert 资产，并用独立成功 ID rollout 校准；20 条 mixed-stream pilot 收满后，accepted OOD 比例至少 80%。

门禁失败时只保留 audit/preflight 诊断，禁止启动正式四方法采集或训练。门禁通过后，Stage1/2/3
分别从同一 immutable ID base 独立收集、预算匹配和训练；不串行续训，也不把三个 stage 混成一批。

## 审计记录

审计必须写出 `audit.json`、每个 split 的 summary/video、实际 Oracle/base 成功数、prefix/target
reach、PCA calibration、pilot 的 accepted ID/OOD 与 method-specific pass/fail。正式阶段还必须保留
raw attempts、alarm/query、takeover、assisted success、accepted counts、完整 suffix、训练 checkpoint
和评测结果。
