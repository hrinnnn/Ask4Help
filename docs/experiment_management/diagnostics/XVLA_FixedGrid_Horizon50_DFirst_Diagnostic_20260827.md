# X-VLA fixed-grid horizon-50 与 D-first knee 诊断

日期：2026-08-27（北京时间）  
性质：diagnostic-only；不覆盖原 Stage-A/B/C formal protocol，也不改变原始结果。

## 1. 目的与重算规则

用户提出将 fixed Failure-Recovery 的 `50 env steps` 作为可接受 takeover horizon：
报警发生在 `step <= 50` 才计为 observed alarm；`step > 50` 统一视为 censored/miss。
`step=50` 保留为 observed alarm。报警时间的均值、分位数只在 censored 后的
observed alarms 上计算，漏报单独保留在分母中。

该重算只用于比较报警 timing；原始 gate threshold、success predicate、Stage-C
exact-budget 和训练分支均不变。

## 2. Horizon=50 后的报警时间

### StackCube（50 条 held-out OOD）

| method | observed | miss after censoring | mean step | median | P25--P75 | KHR |
|---|---:|---:|---:|---:|---:|---:|
| Input PCA | 35/50 | 15 | 23.00 | 25 | 0--45 | 0.20 |
| Bridge PCA | 50/50 | 0 | 5.60 | 5 | 5--5 | 0.76 |
| Action PCA | 44/50 | 6 | 21.14 | 20 | 20--20 | 0.82 |
| Diff-DAgger | 4/50 | 46 | 40.00 | 40 | 30--50 | 0 |
| Failure-Recovery | 50/50 | 0 | 50.00 | 50 | 50--50 | 0 |

在“必须在 recovery horizon 内报警”的指标下，Bridge PCA 的覆盖率最好
（50/50），且报警最早；但在“报警要贴近当前 StackCube knee set
`{10,20}`”的 KHR 指标下，Action PCA 仍略高（0.82 vs. 0.76）。因此，
不能不加限定地说 Bridge PCA 在所有 timing 定义下都是最优；它是
`early-and-before-horizon` 意义下的最佳 method。

### Grab Plane（50 条 held-out OOD）

| method | observed | miss after censoring | mean step | median | P25--P75 | KHR |
|---|---:|---:|---:|---:|---:|---:|
| Input PCA | 48/50 | 2 | 2.92 | 0 | 0--0 | 0 |
| Bridge PCA | 11/50 | 39 | 3.64 | 0 | 0--0 | 0 |
| Action PCA | 0/50 | 50 | -- | -- | -- | 0 |
| Diff-DAgger | 4/50 | 46 | 37.50 | 35 | 35--35 | 0 |
| Failure-Recovery | 50/50 | 0 | 50.00 | 50 | 50--50 | 0 |

Grab Plane 的 Input PCA 虽然有较高 horizon 内覆盖率，但 `43/50` 在
`step=0` 报警，不能解释为有用的“接近抓取前” timing。Bridge PCA 的
报警主要是 `step=0` 或被 horizon censor 掉；Action PCA 和 Diff-DAgger
几乎全部晚于 50。当前没有 method 命中 `t=20±5`。

## 3. 对原始 knee 的审计

Stage-A raw calibration 的单点 knee 都为 `t=20`，但它是对 cost--deviation
曲线做离散 curvature/knee 选择，不包含 pre-grasp 语义约束。

一个更基本的问题是 StackCube 的均值点中，`t=0` 同时具有最低 cost 和最低
deviation：

```text
StackCube: t=0 (C=1.000, D=0.000) dominates every later anchor in (C,D).
```

因此 StackCube 并不存在一个非平凡的 cost--deviation Pareto trade-off；原始
Kneedle 返回 `t=20` 可能是非单调 cost 曲线上的几何结果，而不是一个有效的
Pareto knee。Grab Plane 则存在真实的“cost 下降、deviation 上升”关系，
但 `t=20` 已经处于 D 快速上升后的区域。

## 4. 建议的 D-first 指标（论文讨论版）

不再把 C、D 直接相乘，而采用约束式定义：

\[
D^{\uparrow}(t_i)=\max_{j\le i}D(t_j),
\]

\[
t_{D\text{-safe}}(D_{\max})=
\max\{t_i:\ R(t_i)\ge r_{\min},\ D^{\uparrow}(t_i)\le D_{\max}\}.
\]

其中：

- `D_max` 是预先声明的最大可接受轨迹偏差，而不是看完 OOD utility 后调出来的权重；
- `R(t)` 是 recoverability，`r_min` 沿用现有 `18/20` gate；
- 先满足 D 约束，再在安全集合内取尽可能晚的 timing，以节省 expert time；
- 使用 `D↑` 防止某个晚 anchor 的均值偶然下降而掩盖此前已经发生的偏移。

如果没有可信的物理 `D_max`，应报告 sensitivity curve，而不是强行给一个唯一
的 knee。作为辅助量，可以定义 deviation-onset 为第一个稳定出现显著正偏差的
anchor；当前两个 task 的 first positive onset 都是 `t=10`。

## 5. 现有 Stage-A 数据上的敏感性

令 `D_max = alpha * max_t D(t)`，并用 `D↑` 及现有 recoverability gate 选择
latest safe anchor：

| alpha | StackCube | Grab Plane |
|---:|---:|---:|
| 0.4 | 0 | 0 |
| 0.5 | 0 | 10 |
| 0.6 | 10 | 10 |
| 0.7 | 10 | 10 |
| 0.8 | 10 | 10 |
| 0.9 | 45 | 10 |

这说明 `t=10` 在中间一段 D-cap 范围内比 raw `t=20` 稳定得多；但 `alpha`
仍需在 validation-ID 或物理任务容差上预注册，不能根据 Stage-B OOD SR 反向选择。

Stage-B utility 也给出方向一致但尚不充分的交叉检查：

- StackCube：`t=10` OOD SR `0.6000`，高于 `t=20` 的 `0.4467`，但低于 `t=0`
  的 `0.7233`；
- Grab Plane：`t=10` ever-grasped `0.6133`，显著高于 `t=20` 的 `0.1800`，
  但仍低于 `t=0` 的 `0.7967`。

所以 D-first `t=10` 比 raw `t=20` 更符合现有 utility 排序，但还不能被称为
downstream-optimal timing。

## 6. 推荐的论文表述边界

建议把当前结果拆成三个概念：

1. `raw cost--deviation knee`：数学曲线的候选点；
2. `D-safe/latest timing`：满足预先声明 D 上限的最晚介入点；
3. `downstream utility optimum`：需要独立 policy update 后才能确认。

StackCube 还应先通过 Pareto-dominance check；若 `t=0` 支配所有后续点，报告
`no non-trivial trade-off`，不要把 `t=20` 直接包装成最佳 takeover timing。

## 7. 外部方法学依据

Kneedle 原论文明确把 knee 视为复杂系统中 cost--benefit 分析的近似，并强调
系统特定的 cost/benefit 定义；它并不保证几何 knee 等于任务语义最优点。
安全 RL 文献通常先把 safety constraint 作为可行域，再在可行域内优化性能，
这支持这里的“D 约束优先、时间作为次级目标”设计。

## 8. 状态

本报告和 horizon=50 图均为诊断产物。除非用户明确批准新的 formal protocol，
不修改原 Stage-C threshold、seed pool、exact budget、success predicate 或
completion contract。
