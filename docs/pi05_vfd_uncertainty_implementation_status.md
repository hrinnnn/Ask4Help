# pi0.5 VFD Uncertainty Stage 1-3 实现状态

## 当前结论

截至 2026-07-17，SAVE / `learnsyslab/uq_vla` 的 `vfd_oneway` 已接入
RLinf OpenPI pi0.5，并完成以下三阶段：

```text
Stage 1: VFD 数学核心和单元测试                 完成
Stage 2: pi0.5 prefix KV-cache velocity adapter 完成
Stage 3: 双 checkpoint 单 observation GPU smoke 完成
```

这证明当前 RLinf checkpoint、ManiSkill observation、两个 pi0.5 模型和 VFD
计算链路在单张 H20 上可以工作。它还不能证明 VFD 已经能有效区分 ID/OOD；该结论
需要后续独立 ensemble 训练、标定和 rollout 实验。

## 对原仓库的复现边界

实现优先复用 `learnsyslab/uq_vla` 的原始语义，没有重新设计另一套 uncertainty：

- `make_sampling_time_grid`、`select_ode_states` 和显式 Euler integration 来自原仓库
  `policies/common/flow_matching/ode_solver.py`。
- `VfdOneway` 的计算来自原仓库
  `uncertainty/uncertainty_scoring/scoring_metrics.py`：

  ```text
  sum_s [s / (1 - s)] * ||v_ref(x_s, s) - v_cmp(x_s, s)||^2 * ds
  ```

- action samples 的归约与原 `CrossBayesianSampler` 一致：先计算每个 action sample
  的 VFD，再对同一 observation 的 sample 维求均值。
- adapter 的 prefix cache、时间翻转和 velocity 符号与原 `Pi05Adapter` 一致：

  ```python
  pi05_t = 1.0 - ode_t
  ode_velocity = -pi05_velocity
  ```

只做了 RLinf 必需适配：

- LeRobot observation 处理替换为 RLinf 的 `obs_processor`、`input_transform` 和
  `_preprocess_observation`。
- LeRobot `denoise_step` 替换为 RLinf 已有的 `get_velocity`。
- 模型加载复用 RLinf `get_model`，不另写 checkpoint loader。
- 第一版只实现原仓库默认使用的 `vfd_oneway`，没有提前实现双向 `vfd`、gate 或
  rollout intervention。

## 代码位置

RLinf：

- `rlinf/algorithms/vfd.py`
- `rlinf/models/embodiment/openpi/openpi_vfd_adapter.py`
- `OpenPi0ForRLActionPrediction.compute_vfd_uncertainty`
- `tests/unit_tests/test_vfd.py`
- `tests/unit_tests/test_openpi_vfd_adapter.py`

Ask4Help：

- `tools/maniskill_pi05_vfd_smoke.py`
- `scripts/vfd_pi05/run_stage3_single_observation_smoke.sh`

Git：

```text
RLinf commit:    5ce49aa0 feat: add pi0.5 VFD uncertainty adapter
Ask4Help commit: ac8aa83  feat: add pi0.5 VFD stage 3 smoke
Ask4Help fix:    f2a93be  fix: complete standalone VFD model config
Branch:          codex/pi05-vfd-uncertainty
```

## 测试结果

本地回归：

```text
tests/unit_tests/test_diffdagger.py
tests/unit_tests/test_vfd.py
tests/unit_tests/test_openpi_vfd_adapter.py

22 passed
```

服务器定向测试：

```text
tests/unit_tests/test_vfd.py
tests/unit_tests/test_openpi_vfd_adapter.py

10 passed
```

测试覆盖：

- 相同 velocity field 的 VFD 为 0；
- 人工常量 velocity 与手算 VFD 一致；
- `s / (1-s)` 和 `ds` 权重正确；
- time grid 合并、去重和 requested state 选择正确；
- action sample 归约保持 observation batch；
- observation 扩展到多个 action samples；
- 每个模型每个 observation 只构建一次 prefix cache；
- pi0.5 时间方向翻转和 velocity 取负；
- Gaussian prior 的 shape、dtype 和随机种子可复现。

## Stage 3 真实 GPU Smoke

服务器：

```text
root@39.101.70.188 -p 1020
GPU: NVIDIA H20 96 GB
PyTorch: 2.6.0+cu124
```

输入：

```text
ManiSkill PegInsertionSideWideClearance observer reset, seed=0
member 0: Uniform BC 2-step checkpoint
member 1: ARM-exact AWBC 2-step checkpoint
C=1
velocity_eval_times=[0.0, 0.25, 0.5, 0.75, 0.9]
```

结果：

```json
{
  "status": "passed",
  "action_candidates_shape": [1, 1, 10, 32],
  "vfd_scores": [0.005903115961700678],
  "finite": true,
  "load_seconds": 111.34880182996858,
  "vfd_seconds": 0.6843298199819401,
  "cuda_memory_allocated_gib_after_load": 13.972984313964844,
  "cuda_memory_peak_gib": 14.131409168243408
}
```

OSS 结果：

```text
/mnt/data/ask4help/results/pi05_vfd_stage3/20260717_112115/result.json
```

说明：模型加载的约 111 秒主要是两个 checkpoint 的 CPU 读取、模型构建和首次迁移，
不是每个 rollout step 的开销。`0.684 s` 是首次 `C=1` 单 observation VFD smoke
时间，包含 prefix encoding、Euler trajectory 和两个模型在 evaluation times 的
velocity forward；正式性能评估仍需要 warmup 后重复计时。

## Smoke 中修正的问题

第一次独立脚本调用 RLinf `get_model` 时缺少 worker 通常补齐的字段：

```text
rollout.model.is_lora
```

脚本现已显式设置：

```text
is_lora=false
load_to_device=true
```

没有修改 RLinf 官方 loader，也没有绕过 checkpoint 或 norm stats 加载。

## 后续边界

当前两个 checkpoint 只适合验证工程链路：它们分别来自 2-step Uniform BC 和
2-step AWBC，不是 SAVE 实验所需的“相同训练数据和超参数、不同 seed/shuffle”的
独立 ensemble members。因此当前 VFD 数值不能用于论文结论。

下一步应从同一个 pi0.5 warm-start 训练两个独立 action-expert members，再执行：

```text
Stage 4: 两个独立 ensemble member
Stage 5: ID 成功轨迹 calibration
Stage 6: Peg ID/OOD monitor-only rollout 和 VFD 曲线
```

