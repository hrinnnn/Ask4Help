# π0.5 Feedback SFT接口预检

## 已验证

使用新候选PickPlane stream1772000作为工程测试数据，不将其冒充正式100-accepted/多stream对照。两组共同整后缀预算1327，已使用原生LeRobot writer导出，各8个完整专家后缀；首/末样本的RGB、状态和动作与源trace逐项相同。

H20数据根：
`/mnt/data/ask4help/results/pi05_timing_feedback_ablation_v1/sft_data_engineering_v2/airplane_candidate_stream1772000`。

- `NATIVE_DATA_MASK_AUDIT.json`：真实原生reader保留两组各1327 anchors，所有16个episode的最后真实action anchor均只有1个有效目标，末10个anchor均检查。补齐时间和补齐动作维度被置为极大测试值时，真实8维有效目标loss仍为1。
- `NATIVE_BATCH_AUDIT.json`：原ID源9109个anchors、专家源1327个anchors；每microbatch恰好2 ID+2 expert。真实OpenPI预处理输出actions[4,10,32]、mask[4,10]，仅前8维非padding；实际batch观察到了2/6-valid-step的尾部样本。两有效RGB视图为224模型输入，第三dummy视图mask=false；原仿真/源图仍为384，没有改变相机或场景。
- 实际完整任务指令为 **pick up the toy airplane and move it to the green goal**。真实batch audit从采集provenance读取该完整指令；一次通用配置语法dry-run使用了短占位指令，没有启动训练，不能复制该占位字符串作为真实启动参数。

## 必须显式处理的原生入口细节

1. H20原生时间mask存在，但普通SFT默认不传`use_action_chunk_loss=True`，因此还会优化32维中的padding。新增task-scoped worker只计算10×8真实动作区域，再应用原生时间mask；旧RLinf源码和旧结果未修改。
2. 原生mask-loader缺少供SFT runner使用的长度和epoch sampler接口，task-scoped wrapper提供这些接口，并保留原生采样/归一化。
3. OpenPI PyTorch data loader仍调用JAX的进程信息；为它设置`JAX_PLATFORMS=cpu`，防止无GPU的CPU检查报错以及实际Torch训练被JAX额外占显存。未卸载或修改任何环境依赖。
4. 该H20原生RLinf源码中的`Cluster`先尝试`ray.init(address="auto")`，不读取旧脚本里的`RLINF_RAY_ADDRESS`。新的训练入口在创建Cluster之前仅对本进程的启动调用指定`address="local"`和16GiB object-store默认上限；不得连接/停止其他pipeline的Ray集群。两项mock调用测试通过，实际Ray启动仍须在两步smoke里验证。
5. 全部原始checkpoint/norm保持不变；OpenDrawer恢复权重是单独pt文件，正式SFT需要一个符合原生目录布局的只读checkpoint视图，不得重训或换base。

## 尚未完成

没有执行真实模型的两步反向传播、checkpoint重载或正式2500/5000-step SFT；上述只是数据和入口契约验收。GPU优先用于已启动的OpenDrawer真实Timing/TASR配对实验。完整训练之前仍需完成实际GPU/存储隔离、两步smoke和训练准入决策。
