# π0.5 Timing Feedback 消融：首次资产与实现预检

检查时间：2026-09-09 00:26–00:40北京时间；Owner `01a07faa-682a-7e01-9d5b-eeac5b96864d`。

## 当前证据

| 检查 | StackCube | Grab Plane |
|---|---|---|
| 同任务原ID checkpoint | 原global_step_7000 | 原ID2000之后native_step3000，即total5000 |
| full_weights.pt存在性/大小 | 存在，8,526,557,589 bytes | 存在，8,526,557,589 bytes |
| 原ID dataset | 128 episodes / 5741 frames | 98 episodes / 9109 frames |
| norm | 原norm_stats_id.json存在 | 原id_expert_norm_stats/norm_stats.json存在 |
| π0.5 base | 共用pi05_base_torch，config与model.safetensors存在 | 同左 |
| 当前新模型前向 | 尚未运行 | 尚未运行 |
| 当前新采集/训练 | 尚未运行 | 尚未运行 |

精确checkpoint与输入路径见 `configs/pipelines/pi05_timing_feedback_ablation_v1.json`。目录名不是新结果：上述checkpoint均为既有不可变输入。

H20运行时 `/root/Ask4Help-online-awbc/RLinf/.venv/bin/python` 已实际导入Torch与NumPy：Torch2.6.0+cu124、CUDA build12.4、NumPy1.26.4；openpi、mani_skill、mplib、lerobot、safetensors模块均可定位。尚未把模块存在当作planner/仿真/model forward验收。该恢复目录没有Git元数据，运行时来源需继续结合环境归档和关键源文件记录核对。

H20两卡各只有原PID276925约338/328MiB上下文，CPU0%，必须保留；根盘0可用，独立/tmp13GiB可用，/dev/shm256GiB可用，RAM约249GiB可用，/mnt/data为OSSFS。下一阶段先单卡前向，独立TMPDIR，长期输出写OSS，训练checkpoint临时空间另验收。

5090八卡实际显存28–32GiB全部被占，不启动任何新GPU作业；/data约902GiB可用，根盘约191MiB，不适合当前直接启动。未停止、迁移、清理任何原任务。

## 已实现而非性能证据

新增独立 `tools/pi05_timing_feedback.py`，按当前本地Method实现：5个实际动作的MSE、纠正运动、单条intervention的三值cue、lambda2/beta0.5局部方向、至少两段支持、0.25方向门槛、非重复5-action等待上限。没有读取任务OOD标签或成功标签。

`python -m unittest tests/test_pi05_timing_feedback.py`：9/9通过。只验证数学和状态逻辑，不能证明timing或SR提高。

## 下一项实际工作

在已登记新pipeline内接入π0.5 collector与原task/oracle，做配对RGB/forward smoke；随后独立ID校准，开发检查，再正式双臂采集、同预算SFT、独立ID/OOD评测。完成条件不缩减。旧X-VLA1040诊断、无效Stage2时机结果、SC kNN81和历史Plane81都不作为本次反馈消融的结果。

前一目标回合分类：首次实验目标回合；本回合为 **progress**，新增实验契约/认领、双服务器真实资产证据和通过的核心实现测试。不是verified wait，目前没有新运行作业。
