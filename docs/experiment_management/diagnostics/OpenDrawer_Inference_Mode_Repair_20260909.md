# OpenDrawer 推理模式协议修复

## 直接证据与决定

原OpenDrawer `collect_open_drawer_fixed_timing.py:246–249`、`evaluate_open_drawer_id_pi05.py:180–181`（5090原任务源码）和本地gated collector均调用mode="eval"。PickPlane历史gated collector则明确使用mode="train"。当前通用反馈接口先前对所有任务都使用train，错误地把Plane的采样模式带入OpenDrawer。

原生模型源码说明：train会选择一个去噪步骤加入SDE采样；eval使用ODE步骤，仍保留初始latent随机采样。因此二者不是可以忽略的日志开关，而是不同的执行策略。不能把SDE模式下50条ID仅10条成功直接当作原OpenDrawer基座的性能。

修复仅针对OpenDrawer：使用原任务的eval模式。Plane/StackCube既有实验模式和结果保持不变。新collector要求OpenDrawer校准provenance显式为eval，禁止误用旧SDE校准。

## 保留与重做

- 原 `opendrawer_native_reference_pipeline_v1/ID_calibration_native_v2` 的50条SDE ID及32条SDE动作差异保留，标为不同推理策略的诊断；不补成100来充当标准OpenDrawer校准。
- 原128-ID/22973-feature/rank1000 PCA只依赖VLM prefix，不依赖去噪模式。新校准对每条demo首帧验证eval预测后的Bridge与该参考一致，再复用。
- 原Goal nominal30条由expert从reset执行，没有加载VLA，故不受该模式修复影响，可以保留其18/6/6参考划分。
- v5 Oracle t120记录来自SDE policy前缀，只说明那些具体状态下的专家接续，不作为eval策略的接管效果。
- 重新计算完整32条示范的所有连续5-action窗口、M=2动作差异，以及同一50条ID seed1781000–49的eval-mode自主rollout。按两个GPU分片并行执行，不缩减窗口或示范数量。首20成功要求、q95、rank、物理尺度不变。
- 新目录：`opendrawer_eval_ID_calibration_v4`；配对20raw×2arms×2stage随后写入`opendrawer_stage_feedback_eval_v2`。

这是通过原任务代码识别的实现/协议修复，尚未读取新的OpenDrawer反馈开关比较结果，不是按照效果挑选推理模式。若正确模式下仍不足20个成功ID校准样本，再按原计划补ID，不降低标准。
