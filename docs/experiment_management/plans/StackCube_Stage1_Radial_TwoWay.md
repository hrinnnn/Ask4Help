# StackCube Stage1 Radial-Distance Two-Way Gate

## 目标

验证一个只在 StackCube 早期红块接近、抓取和抬升阶段产生能力缺口，并允许专家修正后回到原策略的 OOD 条件。该版本与旧 diagonal red-position v1/v2 完全独立；旧产物全部保留为 diagnostic，不进入本实验。

## 冻结任务定义

- 使用已有 immutable X-VLA StackCube ID base checkpoint `ckpt-7500`。
- ID reset 复用原 StackCube ID 分布。
- 对每个 paired seed，先采样同一个 `green_xy` 和 `red_id_xy`，令 `u = normalize(red_id_xy - green_xy)`，再设置 `red_ood_xy = red_id_xy + 0.04 * u`。
- OOD 只改变红块初始 XY；绿块 pose、机器人状态、相机、instruction、其他随机性和 success predicate 保持不变。
- OOD 只用于构造早期 red approach/grasp/lift 的能力缺口；不要求 OOD policy 自主抓取成功。
- StackCube 原始最终 success predicate 和 formal horizon 保持不变。

## 独立资产与 manifest

正式版本使用新的 environment IDs、task spec、paired reset manifest 和输出根。旧 diagonal task spec、seed、数据、detector asset、checkpoint 和 two-way 结果均不得复用或覆盖。

## 前置门

1. 本地 paired-reset smoke：验证 20 个 paired seeds 中 green pose、robot/camera/instruction 和非目标随机性相等，只有 red XY 按固定 4 cm radial rule 改变，reset event predicates 全为 false。
2. H20 真实 runtime reset smoke：验证 metadata 来自环境实例而不是纯 sampler。
3. ID/OOD Oracle smoke：固定同一环境生命周期，完整保存 state/action/video 和失败分母；两 split 都必须达到预注册 Oracle 门槛。
4. 使用 immutable `ckpt-7500` 做 OOD-only policy-to-expert continuation smoke：专家从真实 OOD reset 接管，完成 red approach/grasp/stable lift 后交还原 Policy，验证剩余 transport/stacking。

## Continuation smoke

固定 20 条 OOD seeds，记录 first takeover、真实 return-to-policy、continuation success、false release、expert action ratio，以及 red grasp/lift、red-on-green、最终 success 等 event timeline。保存完整 score、state、action 和视频文件。

通过条件是至少 `16/20` continuation success，并且存在真实 `Policy -> Expert -> Policy` 事件。该版本只要求一个 return window，不强制 second takeover；如果没有真实 return，停止为 diagnostic。

## 后续解锁

只有 continuation smoke 通过后，才在同一 immutable base 上分别运行 Internal PCA、Diff-DAgger、Failure-Recovery 和 Offline BC。gated 方法使用严格交替 raw ID/OOD stream、自然 accepted 分布和相同 return 规则；各自收集 100 条成功且发生真实接管的轨迹，按真实 expert low-level actions 匹配共同预算，独立训练并在共同 held-out 100 ID/100 OOD seeds 上评测。

该阶段不使用 StackCube Stage2 timing 数据，也不把旧 diagonal Stage1 结果混入。StackPyramid 只有在本 radial two-way smoke 稳定收口后才进入下一优先级。

## 停止条件

- paired reset 不是 red-only 改变；
- Oracle success、视频、actions、state timeline 或分母不完整；
- immutable base、norm、success predicate 或 seed provenance 不一致；
- 专家无法在 OOD reset 后完成接近/抓取/稳定抬升；
- return-to-policy 不真实、continuation success 低于 `16/20` 或出现 false release；
- 任何失败均保留新目录为 diagnostic，不调整 radial distance、阈值或成功定义来追求通过率。

## v1 Diagnostic Closure

H20 real-runtime reset smoke and ID/OOD Oracle smoke passed. The independent ID calibration used 47/50 successful rollouts and froze upper=`1.6427710056` and lower=`1.0735905170`. The OOD-only continuation smoke then produced 14/20 takeovers, 11/20 real returns, but only 3/20 continuation successes and 5/20 false releases. A read-only handoff audit found that all 11 returned episodes still had the carried red object outside the ID object-green distance range `[0.08, 0.10]` m at the stable-lift handoff. The result is `RADIAL_CONTINUATION_GATE_FAILED_PERSISTENT_CARRIED_OOD`; no downstream collection or training is unlocked. The complete evidence is retained under `/mnt/data/ask4help/results/xvla_stackcube_v1/temporal_mask_v2/stackcube_stage1_radial_two_way_v1/continuation_smoke_retry1/`.
