# π0.5反馈消融：真实运行检查与当前阶段

## 已完成的工程证据

H20使用各task原checkpoint、norm和历史源码。StackCube source9a9f55c/RLinf a2a68e73；Plane sourcedc4c4c1/RLinf116bcdd22。Python为原生H20环境，未复制5090虚拟环境，也未动原PID276925。

首次真实检查发现仅seed Torch不足：RLinf Flow-SDE用Python random选择加入随机性的denoising step。旧smoke重复动作最大差SC0.2003、Plane0.4068。已在本实验adapter内隔离并恢复Python/NumPy/Torch随机状态，不改共享runtime的推理算法。另修复flattened model_action作为prior时的shape接口错误。

`runtime_smoke_v3_prior`两task各ID/OOD均完成：相同输入/RNG的预测最大差0；动作10×8，Bridge2048维；384×384双相机；各split实际执行5个动作。reset和OOD metadata一致，已查看SC与Plane主视角图片。该检查不构成任务成功率，终点false只表示5步未完成，不能记作0%SR。

代码单元测试10/10通过；工程失败v1/v2保留在独立目录。全部数值/图像副本位于本worktree `artifacts/pi05_feedback_runtime_smoke_v3/`；远端原始根为`/mnt/data/ask4help/results/pi05_timing_feedback_ablation_v1/runtime_smoke_v3_prior/`。

## 当前活跃阶段

ID-only校准已由持久controller1253424启动，workers1253425/1253429已确认存活且真实demo计数推进。每task固定32个ID demonstration，之后50个独立ID policy reset；q95、lambda2、beta0.5保持冻结。PCA复用原同task ID资产，并逐选中demo首帧检查缓存与新提取Bridge一致。动作反转使用URDF FK，与真实simulator TCP核对后才继续。

GitHub本机直连曾两次连接超时；随后使用任务临时SSH SOCKS通路成功从本地push至GitHub，服务器正常pull。临时通路已关闭，未更改全局代理或共享运行环境。

## 用户最新报告顺序

先报告实际Timing与新数据TASR，训练可持久化后台继续，再补SR。允许OpenDrawer多个OOD stage作为实验条件。5090仍八卡占用；OpenDrawer原8,526,557,492-byte native5000权重和原norm/轨迹可读，H20未发现其现有副本。OpenDrawer的任务/参考资产还需恢复与资格核对，没有用SC/Plane模型代替它，也未声称它已开始运行。

本回合为progress（真实smoke、RNG故障定位修复、校准启动并验证进展）；不是已完成消融。尚无新有/无feedback Timing、TASR或post-SFT SR比较结果。
