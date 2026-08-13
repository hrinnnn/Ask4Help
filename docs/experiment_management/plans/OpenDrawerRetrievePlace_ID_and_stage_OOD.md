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
5. 冻结 base policy 后再做 failure detection、gated collection 和 matched-budget 更新。

正式训练启动前必须报告所有 episode anchors、tail anchors、mask validity 分布，并确认最后 observation 保留且只有一个有效 target timestep。
