# π0.5三任务论文表：出处核对与TASR恢复准备

用户明确要求只填真实、合格且与表内π0.5模型对应的TASR。不能拿之前21个X-VLA/固定时刻/OpenDrawer条件的Q填到这些格子。

## 当前结论

还没有可填入该表的三任务TASR数值。原记录缺少逐帧动态物体及接触状态，现存原始关节状态可恢复TCP姿态/开合，但不足以计算此前完整版本的物体相对几何与接触约束。不能用accepted-OOD比例或qpos-only相似度冒充TASR。

本轮重新读取并重数20份评测summary，确认：

| π0.5任务 | BC | failure-recovery（用户表HG列） | Diff | Bridge-PCA |
|---|---:|---:|---:|---:|
| 旧StackCube配置 | 49/100 | 29/100 | 34/100 | 未找到对应结果 |
| Grab Plane | 0/100 | 63/100 | **84/100** | 81/100 |
| YCB Object | 48/100 | 49/100 | 42/100（阈值0.05诊断） | 52/100 |

SC原表81/100已确认来自internal Deep-kNN，不是Bridge-PCA，不能保留为Ours-PCA的结果。用户所指SC--Green是旧配置，不是X-VLA绿色目标Stage2；校对稿暂命名SC--Legacy以免混淆。Plane原表74应为84，最高值应随之放在Diff列。

未核实的OD、SC--Red、Eggplant等行不在这份三任务校对稿中；不暗示那些用户数字已经得到验证。

## 实际训练数据核对

正确原始数据已导出到 `artifacts/pi05_table_tasr_20260903/training_assets/`，12集合、1158条被选轨迹、107019个专家轨迹点。每一条parquet动作均与对应raw动作的expert suffix逐项比对通过；1158不包含reference-only的额外未选offline轨迹。

| 数据 | 新增专家点 |
|---|---:|
| SC Deep-kNN full_v2 rebuild | 2655 |
| SC Offline full_v2 rebuild | 2715 |
| SC Diff full_v5 | 2566 |
| SC Recovery full_v5 | 2620 |
| Plane Offline | 17819 |
| Plane Bridge-PCA | 18589 |
| Plane Diff gatecal retry1 | 17978 |
| Plane Recovery | 18261 |
| YCB Object每方法matched budget | 5954 × 4 |

SC81与BC49模型的实际SFT数据分别来自full_v2的完整后缀重建版；不能用full_v1旧截短版算分母。对应 `full_v3/temporal_mask_original_id_norm_5k_retry1/run_manifest.json` 已现场核对。YCB按matched-budget清单选择原episode，未把整个100条collection直接替代其训练子集。

## 回放阻塞

H20已有runtime路径`/root/.venvs/xvla-h20/bin/python`实际解析到`/opt/conda/envs/robo-dopamine/bin/python3.10`，不是仅凭旧笔记可确认的原环境。先试纯CPU（physx_cpu、render_backend=none、CUDA不可见、2线程）回放。原计划包括YCB/Plane的offline及recovery共4条；执行在第1条YCB seed16001的环境初始化处退出255，无恢复状态产物，因此**未验证任何一条回放一致性**。错误日志保存在 `artifacts/pi05_table_tasr_20260903/replay_probe.stderr`；不是数据漂移已被证明，也不是TASR算法数值失败。

第一次探针的Python输入序列化`inf`问题已修复后重试；当前阻塞是上面的环境初始化退出。没有重训、没有占新GPU、没有停止既有进程。

资源现场：5090所有卡有工作；H20两卡仅有PID276925（32天、CPU0.0、GPU上下文338/328MiB），未停止或复用。已向用户询问是否允许在其中一卡使用少量显存验证回放。获准后先确认render/device与原环境协议，再用原qpos/动作验证；若不一致，禁止批量伪重建和TASR填表。

当前交付是**SR纠错和输入就绪，不是TASR完成**。校对LaTeX为 `artifacts/pi05_table_tasr_20260903/active_learning_results_verified_pending_tasr.tex`，所有尚未验证的TASR仍为`--`。
