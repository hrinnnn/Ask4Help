# OpenDrawerRetrievePlace ID 与阶段 OOD

## 当前冻结定义

- 任务：打开抽屉，取出蓝色物体，放入绿色托盘。
- 严格成功：抽屉曾打开、物体进入托盘、夹爪释放、机器人静止。
- ID：把手居中，物体 yaw 为 ±10°，托盘位于 y=-0.30 m；其余因素只做小范围配对抖动。
- 阶段 OOD：把手横向偏移（Handle）、物体 yaw 80°–100°（Grasp）、托盘镜像到 y=+0.30 m（Goal）。
- 当前阶段只采集 ID expert，不把 OOD 混入 ID base-policy 训练。

## Oracle 验收

- reset 起点固定验收：ID、Handle OOD、Grasp OOD、Goal OOD 各 20/20 严格成功。
- 中间 takeover 验收：在抽屉已打开和物体已抓住两个真实 simulator state 接管，2/2 均成功完成后续任务。
- Oracle 支持已验证的阶段边界接管；任意损坏状态仍需单独定义，不作无条件保证。

## 已完成数据阶段

- ID demonstrations：128 条，seed 75000–75131。
- 数据：`/data/zhaozhixuan/Ask4Help-open-drawer/results/id_oracle_collection_v1/lerobot_datasets/open_drawer_retrieve_place/id_oracle_128_retry1_v1`
- 视频：`/data/zhaozhixuan/Ask4Help-open-drawer/results/id_oracle_collection_v1/formal_128_retry1/videos`
- manifest/summary：`/data/zhaozhixuan/Ask4Help-open-drawer/results/id_oracle_collection_v1/formal_128_retry1`
- 审计：128 Parquet episodes、22973 real actions、9D state、8D delta action，全部成功。

## 下一阶段

1. 生成并冻结仅由这 128 条 ID expert 计算的 norm stats。
2. 做 2-step SFT、checkpoint reload 和有限值 forward smoke。
3. 正式训练 fresh π0.5 ID base policy，默认上限 2000 steps、每 250 保存。
4. 用独立 ID 与三个 stage-localized OOD seeds 做 base-policy 阶段性验收。
5. 冻结 base policy 后，先用共同纯 policy rollouts 完成 failure-detection comparison。
6. 对 Handle、Grasp、Goal 三个 OOD 分别运行四个独立数据采集分支：Internal-Feature PCA、
   Diff-DAgger、Failure-Recovery、Offline BC。前三种 gated 方法分别使用 `ID / 当前 OOD` 严格
   交替的固定 raw stream，停止于 100 条成功且发生真实接管的轨迹，不设 accepted-ID/OOD 配额；
   在 ID base 已通过验收的前提下，结果应显著 OOD-dominant。Offline BC 直接收集 100 条当前
   OOD 的完整 oracle demonstrations。每个方法、每个 OOD 形成自己的 expert dataset。
7. 对每个 OOD 按低层专家动作数匹配四组预算，分别使用 `128 ID + 本组 OOD expert`
   的 `1:1 source-balanced` 数据从同一 ID base checkpoint 独立训练。
8. 在共同 checkpoint 和相同 held-out ID/OOD seeds 上评测四个更新后策略。只有三个 OOD
   的四组收集、训练和评测全部完成，OpenDrawer 完整任务才算完成。

完成 ID SFT、完成 detector assets、完成一组 gate 收集，均只是中间阶段，不得提前把整条
OpenDrawer pipeline 标记为完成。

当前运行中的“补齐 50 ID + 50 OOD”控制器不符合正式 data-selection 协议。其全部产物仅保留为
diagnostic，不得进入 matched-budget 训练。正式重跑前必须先报告 ID、Handle OOD、Grasp OOD、
Goal OOD 的 oracle/base-policy 成功率，以及每种 gate 在各 split 上的 attempts、alarm/query、
takeover、assisted success 和 accepted counts。

## 2026-08-15 协议纠正：先修 ID base，再做 mixed-stream data selection

- 当前任务固定为三个相互独立的 stage-localized OOD split：`handle_ood`、`grasp_ood`、
  `goal_ood`。三种 OOD 不合并，亦不从前一个 OOD 更新 checkpoint 继续训练。
- Oracle 必须在 `id`、`handle_ood`、`grasp_ood`、`goal_ood` 各自至少 100 次独立 reset
  上达到至少 90% 严格成功；现有 20/20 只作为 smoke，不是正式 gate。
- 128 条冻结 ID demonstration 重新训练得到新的 ID base；在未参与训练的独立 100-ID
  validation 上严格成功率至少 80% 后才冻结。旧 step-2000 base 的 10/100 结果和其所有
  detector/calibration/collection/training 只作 diagnostic。
- Internal-feature PCA 资产只用新的 ID expert observation 建模；PCA 阈值和 DiffDAgger
  阈值只用独立成功 ID rollout 校准，校准集、norm、checkpoint、split provenance 必须
  固定后再开始 OOD collection。
- 每个 gated 方法在每个 OOD split 使用相同的 ID/OOD 严格交替 raw deployment stream，
  只以总计 100 条成功且发生真实 expert takeover 的 accepted trajectory 停止。不得设置
  `accepted_id=50` 或 `accepted_ood=50` 的配额；最终 ID/OOD accepted 数是结果，不是输入。
  若进程异常，恢复必须从上次 raw attempt offset 接续，不能重复 seed 或重排 stream。
- 三个 OOD split 各自收集、各自按 expert action budget 与其它方法匹配、各自从同一 immutable
  ID base 独立训练；只有三个 split 的四方法结果全部完成后，才进入主结果表。

2026-08-15 协议审计进一步确认，当前被 collection 使用的 ID base checkpoint 在固定 100 条 ID
评测上严格成功率只有 `10/100`，未达到“已掌握 ID”的基本前提。这是 ID 大量进入 accepted 数据的
首要根因；在该 base 上得到的 PCA/Diff/Failure collection 均不得作为正式数据。下一步必须先使用
冻结的 128 条 ID demonstrations 重新训练或选择 ID base，并在独立 100-ID validation 上达到至少
80% 后冻结；随后才重新评估三个 OOD split、重建 ID-only calibration，并在三个 OOD 上分别重跑。

正式训练启动前必须报告所有 episode anchors、tail anchors、mask validity 分布，并确认最后 observation 保留且只有一个有效 target timestep。
