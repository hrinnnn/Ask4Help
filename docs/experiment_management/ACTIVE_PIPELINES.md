# Active Pipelines

## 2026-09-10 用户最新授权：挂后台训练最终Stage2两组

- 用户在最终30反馈结果报告后明确授权挂起训练；按上下文执行后台训练，覆盖此前“不训练”的未来动作限制，旧诊断结果不改。主owner不变，Luna继续30分钟只读监测。
- 已认领`pi05_budget30_sft_v1`：最终budget30的fixed vs sensitive两组，从同一OpenDrawer v9 native5000基座出发，各2500步；先完整后缀匹配专家动作预算，原ID:new专家1:1，原norm，保留全部真实anchor与tail mask。
- 使用用户已看过的自然ID/OOD组成，不补配额。该明确训练请求针对这两份数据；作为paired ablation，不冒充旧100accepted/80%OOD主实验。
- H20两卡空闲（仅保护276925），tmp8.7GiB、shm238GiB可用、RAM230GiB可用；5090所有卡有占用且data97%，不分配。模型/原始ID/norm均应保持当前同任务输入。
- 流程：预算索引→原生导出与mask/batch审计→视觉/两步重载→两组2500步→各100ID100OOD独立评测→报告。长任务由持久controller推进，不在smoke完成后停止。

## 2026-09-10 30反馈预算确认与报告完成

- `pi05_evidence_wait_budget30_v1`两组和100独立passive已完成full审计；主agent另复核标签、分母、无专家动作、fresh seed、记忆始终<=30和逐轨迹配对。固定/修订BA79.43/82.58%，FP2/4（22成功），FN25/13（78失败）；ΔBA95%CI[-4.74,10.00]pp，单流正向点估计而非显著普遍收益。
- 共同120raw：52双方接管、51双方未接管、17新增接管；52配对中32提前18不变2推迟，30条提前>=10步、最长推迟10步。共同60OOD辅助成功12→22，60ID30→32。收30成功后缀cost5075→3770，但共同120raw cost2593→3770，完整报告两种口径。
- 结果报告、LaTeX及正负配对视频在主工作区`artifacts/final_feedback_report_20260910/`，正负终帧已检查。此前StackCube、Goal-OOD负结果及76反馈版本保留。当前达到已授权的冻结策略诊断报告交付，未训练。
- 完成标记`DIAGNOSTIC_COMPLETE.json`；当前无后续训练授权，完成交付后删除空闲heartbeat以节省token，不创建Goal。

## 2026-09-10 用户指定：Luna max监测，主agent决策

- 心跳保持每30分钟，不创建Goal。有界监测子agent `/root/luna_monitor` 使用gpt-5.6-luna/max；每次只读状态、PID/progress/日志/marker，返回简短异常或完成提示。
- 主agent仍为唯一pipeline owner，负责必要复核、工程修复、科学决策、阶段推进和结果报告。Luna不改阈值/seed、不终止进程、不启动训练、不创建独立长期pipeline。
- 当前pipeline仍为`pi05_evidence_wait_budget30_v1`，主根与并行修订根不变。自动任务`small30-ba`已改为委派Luna检查，健康时主agent不重复远端轮询。

## 2026-09-10 Stage2正向结果与真正30反馈预算确认

- evidence_wait_v1 Stage2三组及100passive已完成，full审计与独立confusion复核通过。固定/上一版/修订BA75.10/77.89/80.49%，FP1/3/3（23成功），FN35/24/20（77失败）。修订相对固定ΔBA5.39pp，95%reset-seed簇CI[-1.96,11.96]pp，单流结果，非已证明显著普遍收益。
- 共同142raw的71OOD辅助完成14→22，10改善2变差；ID42→41。收30成功后缀cost4810→4254，但共同raw成本3545→4254，需同时报告。报告和正/负/推迟视频在主工作区`artifacts/evidence_wait_review_20260909/opendrawer_stage2/`。
- 关键预算限制：30成功后缀用了76条反馈（含失败）。按用户20–30反馈的实际目标，新冻结`configs/pipelines/pi05_evidence_wait_budget30_v1.json`：前30条有效反馈后记忆冻结，其余参数不变；新collection1910000/eval1920000。此为独立预算协议，不能称原76反馈版本复现。
- 用户已授权持续改进；新root`evidence_wait_budget30_v1`、并行root`evidence_wait_budget30_parallel_v1`，source27a7b980（本地f19e0dd3）；固定/独立评测GPU1，修订GPU0，同一既有资产，原任务完成，不训练。下一阶段发布后启动、首批审计、完整结果报告。

## 2026-09-09 晚间持续推进：双卡并行与增量审计

- 2026-09-09 23:16北京时间确认迁移完成：主controller1759813，GPU1 old worker1760294已133raw/25accepted；GPU0并行修订worker1763511已23raw/3accepted。原collector分块完整保留，新source5af98de7继续原seed和memory，暂无error；不再把旧controller暂停状态当作当前状态。

- 已实际启动：并行修订controller1755267、worker1755290（GPU0），首条轨迹落盘；迁移器1755721等原child1753234完成，旧controller1650915暂停属预期。主根`parallel_migration_state.json`为迁移状态源。跨GPU首条空memory整轨迹动作差0、reset RGB相同，验证通过。
- 增量审计在原old/chunk0080真实100条数据上与既有full审计逐行结果相同；复用80条，仅检查20条，用时44.8秒。整组末保留full审计。heartbeat已更新到当前并行/迁移流程，不创建Goal。

- 用户授权今晚持续推进、用heartbeat、不要Goal；结果不足继续有依据的新版本，直到可交付ablation报告，仍无训练。
- 性能修订source5af98de7：增量审计复用已审计prefix，每组结束保留一次full audit；新增only-arm/gpu/cpu/external-sensitive-root执行支持，科学参数不变。
- 新修订组将独立在GPU0/CPU0-3运行，root`/mnt/data/ask4help/results/pi05_timing_feedback_ablation_v1/evidence_wait_parallel_sensitive_v1`；主对照/独立passive继续GPU1/CPU4-7，原root不变。
- 主控制器在当前child结束后迁移，暂停的仅是旧控制器而非collector；`parallel_migration_state.json`记录被保留child及新owner。迁移若中断须先查此state及PID，不重复启动。主控制器会引用并行组完成artifact，不复制/重采同一组。

## 2026-09-09 最新追加：OpenDrawer Stage2优先，与StackCube修订并行

- 2026-09-09 22:28北京时间：SSH已恢复，OpenDrawer Stage2 controller正常自动推进至old/chunk0040审计（1742739），无error；未因短暂连接中断重启任何进程。

- 2026-09-09 22:16北京时间：本次两次SSH连接均被H20远端关闭，未取得新的进程或产物状态。上次已确认old组正常推进；不能将连接异常当作实验失败或重启依据。保留现有controller，后续先恢复只读连通再判断状态，不重启或改参数。

- 2026-09-09 22:05北京时间检查：OpenDrawer Stage2 fixed已205raw/30accepted（10ID20OOD）、4810全部专家动作，独立audit PASS；controller1650915自动进入old连续反馈组，首20条审计中（1727889）。仍无Stage2最终比较或新训练，按冻结序列继续。

- 2026-09-09 20:17北京时间：evidence_wait_v1 StackCube三组与100passive已完成并独立复核。固定/上一版连续/修订BA95.50/87.16/90.19%，FP1/7/5（33成功），FN4/3/3（67失败）；修订尚未超固定。共同44raw中两反馈均12条提前>=10、2条推迟>=10，最大推迟45→10；具体ep35 60→25（fixed15），ep43 45→30（fixed20）。报告与三路视频在主工作区`artifacts/evidence_wait_review_20260909/stackcube/`，代表帧已检查。OpenDrawer Stage2 fixed111raw/14accepted、worker1692500健康，继续全部冻结条件，无训练。

- 2026-09-09 19:33北京时间检查：evidence_wait_v1 StackCube fixed52raw/30accepted、old44raw/30accepted完成；修订sensitive首20条独立audit PASS，已进入chunk0020 worker1667614，确认remember_deferred_alarm/use_later_duration启用。OpenDrawer grasp fixed前40条完成审计，chunk0040 worker1667584正常初始化。两controller健康，无训练。

- 已实际启动并产出轨迹：controller1650914/1650915，fixed workers1650947/1650948；SC6raw/4accepted、OD Stage2首条完成。source74dced4d，native27+7测试通过。各task状态文件位于新evidence_wait_v1根；后续以此推进，不重复启动。
- 初步阶段核对：旧Stage2 7条已报警OOD在56–94步打开抽屉，135–350步接管，6条接管前未抓起，1条在195接管时首次grasp。没有初始化即报警的证据；继续区分用户旧图版本与相对操作阶段的“早”。
- 旧Small30 Goal独立BA已完整：fixed/old/sensitive92.00/92.00/87.33%，FP1/1/2（25成功），FN9/9/13（75失败）；保留为旧负结果，新Stage2不会据此调测试阈值。

- 用户明确授权继续StackCube，同时重点验证OpenDrawer Stage2抓取位姿OOD；目标至少一个任务相对固定Bridge-PCA有改善，保留全部结果。仍无训练。
- 修订已接入collector/memory恢复/独立审计与被动评测；新manifest `configs/pipelines/pi05_evidence_wait_v1.json`，GitHub部署74dced4d。三组fixed、old（Small30 continuous）、sensitive（pending+实测Later时长）；各30 accepted/400raw cap，首20逐组审计，后100独立passive。不要把此old和历史hard-radius组混淆。
- StackCube GPU0/CPU0-3，collection1870000/eval1880000；OpenDrawer grasp GPU1/CPU4-7，collection1890000/eval1900000。输出根`/mnt/data/ask4help/results/pi05_timing_feedback_ablation_v1/evidence_wait_v1`。旧Small30两任务采集/BA均已完成，OpenDrawer报告已重算seed-clustered但尚待完整视频交付。
- 旧Stage2两批共30OOD未出现t0或前25步报警；已触发时机135–385步。需按drawer-open/grasp事件核对用户说的相对阶段提前，不能直接断言初始化即报警或静默替换旧PCA版本。

## 2026-09-09 用户授权：修正过度推迟机制（仍不训练）

- 4case完整expert片段前向审计已完成，root`later_segment_audit_v1`、code142810d8，明确发现offset10夹爪方向20%不一致仍被整体MSE判为一致。新可选pending报警＋measured Later时长原型25测试通过，尚未接collector。后续执行按独立worktree计划`docs/experiment_management/plans/Pi05_Later_Repair_20260909.md`：全部开发Later审计、ID开合边界检查、接入/审计、新seed1870000配对、1880000独立BA；用户已授权继续改进，无训练。旧Small30继续完成，不能结束旧诊断后遗忘本修订。

- 同owner认领新的局部诊断与修订。旧Small30继续完成，不改原运行参数。
- 已核实episode17第10步阈值仅从0.7149提高到0.7255；暂缓首次crossing后，分数下降导致原alarm被遗忘，第25步direction已转正仍未接管，直到55。保持旧memory/实际prefix的离线反事实中pending-alarm修订使55→25；episode7仅50→45，说明此项修订不充分。
- 已本地实现可选remember_deferred_alarm，默认关闭，22项相关测试通过。下一项为开发数据4条完整expert suffix的机械臂/夹爪分项误差前向审计；不改变gate、不启动训练。
- 计划：根据实际连续动作一致范围定义Later有效片段；再冻结新版本并使用新seed验证。旧独立BA已看过，仅作为旧版结果，不用于宣称新版本泛化效果。

## 2026-09-09 最新用户决定：小样本灵敏反馈，仅收集与检测，不训练

- 2026-09-09 15:16北京时间检查：OpenDrawer old已88raw/30accepted（6ID24OOD）、3101专家动作，audit PASS；已自动进入sensitive worker1581872（continuous，无额外deadline），当前8raw/1accepted。fixed78raw/30accepted（6ID24OOD）、2786动作。没有BA新结果、没有训练；保持完整冻结序列，不因阶段成本变化调参数。

- 2026-09-09 14:42北京时间检查：OpenDrawer old49raw/16accepted，worker1569391健康，前40条audit已通过。StackCube补充导出独立成功轨迹seed1830019（passive episode38，fixed无报警/sensitive30在20步报警）原视频`artifacts/small30_review_20260909/stackcube/passive_success_extra_alarm_step20.mp4`，10fps、38frames、终帧已检查；这是被动误报例，不是实际专家接管。另保留fixed/sensitive episode0不变且失败原视频。无新训练或阈值修改。

- 2026-09-09 14:28-14:31北京时间：StackCube全部100 passive与三组采集完成；seed-clustered BA报告已另写并独立复核100个seed/split、50ID50OOD、零专家动作、confusion counts。30accepted节点fixed/old/sensitive BA=96.97/95.45/86.36%，FP=2/3/9（33成功），FN均0（67失败）。新版ΔBA=-10.61pp，条件于本次memory的reset-seed bootstrap95%区间[-18.33,-4.17]pp。该候选不满足用户BA不降低目标，禁止在测试集上回调参数。actual共同46raw中新版提前>=10步6条、推迟>=10步8条；被动24/69移动>=10步不得混称真实接管。
- StackCube已导出提前35→5与推迟10→55的全局时间对齐视频，主工作区`artifacts/small30_review_20260909/stackcube/`，顶部fixed/底部sensitive、半速、绿条专家，已检查代表帧。OpenDrawer old仍健康采集（1564665），不停止其冻结实验，整轮未完成、无训练。

- 2026-09-09 14:17北京时间检查：OpenDrawer fixed78 raw/30 accepted完成且audit PASS，自动进入old组worker1557040。StackCube independent passive79/100、worker1556699正常，尚未生成完整BA报告。两任务均无工程失败/训练，下一阶段仍由健康controller自动推进。

- 2026-09-09 14:05北京时间检查：StackCube三组各30 accepted已全部完成并审计，fixed/old/sensitive raw分别58/56/46，全部专家动作995/955/856。已自动进入独立pure-policy检测，32/100，首20条audit PASS且零专家动作。OpenDrawer fixed65 raw/24 accepted，worker1549543健康。暂无独立BA结果，不把采集成本差当作检测或后训SR改善。

- 2026-09-09 13:54北京时间检查：StackCube old已完成56 raw/30 accepted，audit PASS；sensitive首20条audit PASS，确认continuous且max_wait_blocks=null、无deadline，当前30 raw/18 accepted，worker1542648。OpenDrawer fixed当前49 raw/17 accepted，worker1540226，前40条分块审计通过。控制器正常自动推进；尚无独立BA，禁止据部分采集宣称改善。

- 2026-09-09 13:42北京时间检查：StackCube fixed已完成58 raw→30 accepted（3ID/27OOD），995专家动作，独立audit PASS；自动进入old组，worker1534985已23 raw/10 accepted。OpenDrawer fixed worker1532403已28 raw/7 accepted，先前20条chunk audit PASS。两controller持续健康，未启动训练、无新BA结论，继续按冻结方案运行。

- 已启动：GitHub部署`47349a02`（本地`44273c27`同树）；H20两任务controller为StackCube `1524113` / OpenDrawer Goal `1524114`，初始fixed workers `1524146` / `1524145`。输出根`/mnt/data/ask4help/results/pi05_timing_feedback_ablation_v1/small30_v1`；各task的`controller_state.json`为真实状态源。服务器原生27项相关测试通过；当前初始化/首批采集验收中，尚无新结果。
- 控制器依次执行fixed/old/sensitive各30成功接管上限（每arm400raw上限），每20个新episode独立进程并审计；然后100条独立纯policy轨迹（50ID+50OOD），再按5/10/20/30 accepted节点重建冻结memory计算BA和时机。不会调用任何训练入口。
- 监督：先确认真实episode落盘与视觉一致后再进入低频监测。工程异常保留原chunk，不改变模型/阈值/seed；结束于`TIMING_BA_COMPLETE`后仍需视频审查，不得当作整轮已完成。

- 已认领：同一 owner `01a07faa-682a-7e01-9d5b-eeac5b96864d`；独立 worktree `Ask4Help_feedback_ablation`。
- 用户批准 StackCube legacy OOD / OpenDrawer Goal-OOD，固定阈值、旧反馈5步上限、新版连续单事件反馈无额外延迟上限三个组，每组最多30个成功接管后缀；5/10/20/30节点检查独立BA与实际timing。
- 本轮禁止新训练（包括训练smoke）、自动SFT与训练后SR。旧pipeline中指向训练的历史next_stage不再授权；先前工程smoke保留，不继续。
- 当前阶段：新协议冻结与实现；尚未启动新一轮采集。新 manifest：`configs/pipelines/pi05_feedback_small30_v1.json`；所有旧结果保留。
- 新版是探索性修订，不保证BA不降、不强制移动10步、不依据held-out结果再调阈值。OpenDrawer不宣称为干净post-grasp因果实验。

## π0.5 Timing Feedback Ablation（2026-09-09）

- **训练工程预检准备**：当前StackCube fixed40成功后缀为20OOD+3ID（不是旧pilot的75%）；是否正式可用仍待完整arm/协议验收。已准备独立2-update训练+重载流程，在StackCube本轮paired完成、GPU0实际空闲后，使用已验收的PickPlane1327样本工程数据验证SFT链路。只做2步，不作为正式SFT/效应数据，不绕过正式训练准入；原reserved测试seed不使用。checkpoint先写独立/dev/shm任务目录，重载通过后复制到OSS，保留scratch。
- **08:45自主诊断完成**：20配对seed完全无接管复跑，ID抓起/抬起11/20、strict7/20；Goal-OOD均0/20，所有旧gate之前的动作差为0。不是单纯接管截断，当前基座/批次存在前置能力问题，不作为干净post-grasp timing验证；原数据全部保留。报告 `OpenDrawer_Goal_Autonomy_Censoring_20260909.md`。
- 已自动启动冻结commitment_v2的跨任务复现：controller1470507，StackCube seed1774000、PickPlane1775000，各arm40raw，hard_radius及其余参数不变。root `commitment_replication40_v1`，source5c960d57。StackCube opening只用32-ID得到0并沿用10mm下限。不是新的阈值搜索，不混入Soft支持修改；正式训练仍待准入决定。
- **08:35三组完成**：同40seed hard控制和三组共同预算已全部完成。Grasp B221时Fixed/Hard/Soft TASR×3均19/221；Goal B919均305/919。Goal Hard推迟4条、Soft10条，各5动作；没有TASR收益。成功专家成本Goal982→983，失败432→415，不把总成本小降称为成功数据效率提高。报告 `Pi05_Feedback_Three_Arm_40_Results_20260909.md`。
- 当前是同seed无接管开发复跑：controller1464038，ID/OOD workers1464544/1464545，root `opendrawer_goal_autonomy_diagnostic_v1`，source622ef779。目标为区分被接管截断与前置抓取失败；不是post-SFT或reserved测试。最终测试9,100,000系列种子已冻结未用。StackCube的commitment候选ID参考已算好（32demo/q95=0/floor10mm），尚未新采集。正式SFT仍待准入决定。
- **Soft40完整结果**：Grasp共同B221，TASR×3=19/221两组相同，新增1条失败报警导致专家成本510→540。Goal10条OOD配对接管推迟5步，但共同B920的TASR×3=304/920→303/920（33.04→32.93%）；成功专家动作982→983，失败段432→415，所以总成本减少不能包装成成功数据效率提升。hard40控制已自动运行，尚待三组同预算报告。
- 已登记后续同seed无接管开发复跑：Goal-ID/OOD各20，seed1785000–19，原base/eval模式，等待hard40完成后运行。目的区分“被提前接管截断”与“确实无法自主抓起”；不是reserved最终SR。接管前0/20抓起不能直接推断没有自主抓取能力。最终测试已另行冻结9,100,000系列seed，未使用。
- **07:40恢复已推进**：新root `opendrawer_soft_support_v2_chunked` controller1422325，固定组两stage均补齐40并通过审计，原24条只引用、无覆盖。新16条过程中FD计数稳定91；不把底层原因夸称已完全解释。当前feedback首chunk workers1429579/1429478，最多每进程20新episode，后续从已完成记录恢复精确memory。source949827ef（本地07151958）。
- 同40新seed原hard-radius对照已持久化等待：controller1426109，source98af1d4a，root `opendrawer_hard40_control_v1`。Soft完成后自动复用同fixed数据、收集hard feedback，最后重新求fixed/hard/soft三者共同整后缀预算，避免把不同二元预算下的TASR直接横比。正式SFT仍未启动，准入例外仍待用户确认。
- **渲染生命周期恢复**：soft_support_v1两fixed worker均在已完成24条后，创建第25个环境时出现vk::createInstanceUnique/ErrorIncompatibleDriver并退出；没有soft反馈结果。保存原24条，恢复将采用每进程最多20个新episode、episode后GC、显式恢复过去cue记忆；新root `opendrawer_soft_support_v2_chunked`。旧原始episode只被引用，不复制/覆盖，不改变seed或算法。文件描述符计数会写入进度，以核实资源累积，尚不把它断言为驱动不支持Vulkan。
- **首轮结果/下一候选**：eval-mode Grasp/Goal各20raw×2arms已完整通过timing/TASR审计；所有takeover不变，Grasp60/325=18.46%、Goal203/708=28.67%均不变。原支持条件覆盖分别仅1/1019、0/811 eligible queries。保留零结果；已按诊断冻结soft_mass候选，带宽不变、取消硬截断、保留2*exp(-.5)总权重下限。新seed1784000/1785000，各40raw/arm，独立新root `opendrawer_soft_support_v1`；未获得新结果前不声称改进。
- **06时段当前执行**：正确eval校准已完整结束，32 demos/50 ID policies，33 strict成功，独立审计通过（gamma0.3362330496311188，q_e0.20073064610519734）。controller1368627已自动启动两个stage的真实paired20：Grasp controller1387638，Goal controller1387639；fixed均完成20，feedback workers1396060/1395890继续运行。实际state在 `opendrawer_stage_feedback_eval_v2`，旧SDE50不参与。
- SFT工程数据两组1327完整样本已导出，原生reader尾mask/真实8维loss调用及实际1:1混合batch均通过。报告 `diagnostics/Pi05_Feedback_SFT_Interface_20260909.md`。尚未真实两步GPU训练或正式SFT；pending自然composition/preliminary基座例外未收到回答。
- **模式协议修复，优先于以下旧状态**：旧OpenDrawer通用接口误用了Plane的train/SDE采样；原任务gated/fixed-timing/eval源码均为eval。旧50-ID仅10成功和32-demo动作误差保留为不同策略诊断，不再尝试直接补100。原VLM-only PCA和expert-only Goal nominal bank可保留；重新全量计算32 dense demos+50 policy的eval校准，双卡分片，新root `opendrawer_eval_ID_calibration_v4`。随后配对root为`opendrawer_stage_feedback_eval_v2`。原controllers1307826/1349662已终止失败，不重启；无任何OD on/off结果已被用于选择模式。
- 训练工程接口已准备：native时间mask之外显式仅计算8维动作；pilot SFT数据导出v1因重复解压NPZ效率问题终止，源数据未变，修复后将在新目录重试。尚无SFT。自然composition/历史preliminary OD基座用于本次消融的例外已再次简明询问用户，未收到新决定。
- **04:35后续**：v5 Grasp/Goal oracle smoke均完成并通过独立raw/reset/action审计，完成数分别3/4与4/4；Grasp t120的一条direct-grasp失败在修正planner后仍保留，不改专家来消除负例。Goal主相机遮挡后方tray，但wrist reset/terminal可见绿盘与最终蓝块，已人工查看。Goal nominal TASR bank（fresh1786000–29）worker1343725已完成至少19个raw/19成功，GPU1；只做评价参考，不进入gate。
- 已写好stage controller：nominal bank→独立Goal metric calibration→ID gate audit→Grasp/Goal各20raw×2arms并行→各自TASR。种子1782000/1783000，原displacement_v1，均空记忆；不把新commitment候选偷偷并入OD。源码和运行记录使用GitHub部署分支，原正式工作分支HTTPS push仍不稳定。最终训练/SR仍未启动。
- **04:21更新**：128-ID/22973-anchor/rank1000 native PCA已完成，controller1307826自动启动ID校准worker1336534（GPU0）。修复direct-oracle初始planner_mode与reset记录时点后，smoke v5/grasp worker1336653在GPU1。v3的t120 OOD失败仍属错误planner初始化的工程诊断，不计入gate比较；v4因发布确认失败在模型加载时停止。现用GitHub已核实同树部署commit25a7f0cc（API发布分支codex/pi05-feedback-deployment，等价本地79df1bbc），正式工作分支推送网络暂不稳定。
- Grasp-OOD TASR原18-reference/6-check小包已导出；6条旧检查轨迹数值回归通过，未改变阈值。Goal-OOD不得借用抓取target定义。没有新OD反馈配对结果，也没有SFT启动。
- **04时段真实更新**：原Plane v1配对20raw/arm已完成，实际时机完全相同，matched828 TASR×3=104/828两组相同。独立饱和夹爪候选commitment_v2两条新stream均完成；1772000 matched1327为200/1327→345/1327（15.07→26.00%），成本2247→2497；1773000 OOD时机中位数提前15，但共同整后缀预算167只剩ID、TASR=0→0，不作为OOD验证成功。完整结果见候选报告；没有启动SFT。
- **当前OpenDrawer**：全部128条原始ID、22973 anchors恢复marker已通过。GPU0 reference controller1307826/worker1315075正用同一原始ID集在H20重建rank1000参考；原cached跨机器maxdiff0.0625的失败单独保留。当前state位于 `opendrawer_native_reference_pipeline_v1/controller_state.json`，不是下方历史PID。之后同controller自动ID校准与独立审计。GPU1仅用于本Goal新direct-current-state oracle smoke，先检查400步端点及真实RGB；不改变其他owner任务。

### 较早pilot阶段记录（下述Plane运行PID已经结束）

- **最新**：SC两arm各20raw完成及独立审计/TASR完成，OOD两条35→10和40→20，10OOD其余8不变；matched296动作TASR×3=122/296→123/296，×1相同、×2略降，不称稳定有效。两组accepted=3ID+9OOD=75%，已异步询问本次消融能否保留自然比例；未收到例外授权，不启动SC训练。
- **当前进程**：Plane原development_v1因旧oracle越过250-action时限和诊断inf写JSON失败保持无效。修复后late-smoke严格140+110=250、独立审计通过；`development_v2_horizon/airplane_yaw_ood` controller1282003、feedback worker1285539真实运行，source35f1e4e1。state以该root内文件为准，不再读已结束calibration的全局state作为当前进度。
- OpenDrawer原8,526,557,492-byte权重和norm已恢复到H20；原样环境source与5090/本地RLinf一致。Grasp ID/OOD双RGB、10×8动作、2048Bridge和重复预测差0检查通过，尚未做OD feedback collection或SFT。task源码快照e0269103。
- 代码source-of-truth在本地独立worktree和GitHub分支；网络失败时只部署已在GitHub核实存在的精确commit archive。TASR source55f0489f已通过GitHub API核实，未因CLI报错假定提交未到达。

### 之前阶段记录（以下PID/阶段不代表当前运行状态）

- 当前真实阶段为`ID_calibration_running`：H20 controller1253424，SC worker1253425/GPU0/CPU0–3，Plane worker1253429/GPU1/CPU4–7；state=`/mnt/data/ask4help/results/pi05_timing_feedback_ablation_v1/pipeline_state.json`。32-ID demo动作差异/纠正校准正在推进，随后各50独立ID policy；source`ab3a6aa6`已push/pull。下一阶段为真实expert接入、配对Timing/TASR数据、同预算后台SFT，阶段完成不结束Goal。

- 用户追加授权：OpenDrawer不同OOD stage也可作为两项条件；先报真实Timing与TASR，再持久后台训练，最终仍补SR。新信息已写入manifest/plan。
- 真实前向`runtime_smoke_v3_prior`已完成：SC与Plane分别ID/OOD同query配对动作差0，均10×8 actions与2048维Bridge，384×384双RGB，实际每split执行5动作。v1发现Python random未配对，v2发现prior扁平维度，修复后通过；这些是工程smoke，不是rollout SR。当前无smoke进程，下一步ID校准/真实专家采集接入，并盘点OpenDrawer。

- 新授权完整消融pipeline=`pi05_timing_feedback_ablation_v1`，owner=`01a07faa-682a-7e01-9d5b-eeac5b96864d`；manifest=`configs/pipelines/pi05_timing_feedback_ablation_v1.json`，plan=`docs/experiment_management/plans/Pi05_Timing_Feedback_Ablation.md`。
- 主任务StackCube/Grab Plane，固定gate与有反馈gate；完成新采集、同预算SFT、独立ID/OOD SR和论文LaTeX。旧X-VLA探针不替代新π0.5结果，OpenDrawer需先核实资格。
- 首次预检：H20原任务weights/norm/ID数据存在，原生Torch/NumPy导入通过；反馈核心9测试通过。当前`asset_runtime_preflight`，下一阶段`paired_smoke_then_ID_calibration`；尚无新GPU作业，不报告运行中。
- 5090全卡占用；H20保留PID276925，根盘满，使用独立/tmp与OSS前必须验收存储。所有正/零/负结果保留，不能提前写提升。

更新时间：2026-08-25

本文件是当前长实验的执行总表。Owner、Leader 和 Heartbeat 每次接力必须先读本文件，再读对应 manifest、plan 与远端 `pipeline_state.json`。聊天历史中的旧模型路线或旧阶段不得覆盖本文件。

## X-VLA Fixed-Grid Task-Policy Knee Validation

- `pipeline_id`: `xvla_fixedgrid_taskpolicy_knee_v1`
- `authorized`: `true`; 用户已要求记录并执行 StackCube/Grab Plane fixed-grid task-policy knee validation
- `owner_thread`: `current-thread`; `owner_label`: `codex-root-xvla-knee-validation`
- `server_preference`: `zhaozhixuan@111.198.58.150:12001`; H20 `root@39.101.70.188:1012` 为回退
- `current_stage`: `stage_c_gate_data_needs_user_decision_diffdagger_budget`
- `next_stage`: `user_decision_then_protocol-preserving_retry_or_ineligible-branch-reconciliation`
- `run_root`: `/data/zhaozhixuan/Ask4Help-airplane-5090/results/xvla_fixedgrid_taskpolicy_knee_v1/`
- `runtime_5090`: X-VLA `/data/zhaozhixuan/X-VLA`; Python `/data/zhaozhixuan/envs/xvla_official_5090/bin/python` (historical successful StackCube collection environment; RLinf `.venv` planner-init segfault retained as diagnostic); independent worktree `/data/zhaozhixuan/xvla_fixedgrid_knee_work`
- `manifest`: `configs/pipelines/xvla_fixedgrid_taskpolicy_knee_v1.json`
- `plan`: `docs/experiment_management/plans/XVLA_FixedGrid_TaskPolicy_Knee_Validation.md`
- `source_commit`: `2f5d725`; local branch `codex/xvla-fixed-grid-knee`; server worktree is detached at `055c436` (the source commit plus metadata-only reconciliation; Stage-B utility, Stage-C passive result, and Diff-DAgger stop recorded)
- `implementation`: fixed-step collector, task-state knee summarizer, restart-tolerant calibration controller, matched-budget Stage-B trainer/evaluator, temporal-mask training shim, durable Stage-B total supervisor, frozen gate-to-knee audit, durable Stage-C passive gate controller, whole-episode exact-budget gate selector, gate-selected Stage-C data/training/evaluation controller, independent final reconciliation, and contract/knee/controller tests
- `resource_preflight_2026-08-25`: calibration and smoke evidence passed; Stage-B training and its 54 formal evaluations completed on the explicitly selected 5090 GPU5/CPU `0-19`. Other protected GPUs and the H20 owner are untouched. Training controller PID `4087857`; total supervisor PID `2926040`; formal evaluation PID `1138192` completed; diagnostic partial-evaluation controller PID `3302716` completed on GPU4/CPU20-39; Stage-C retry5 controller PID `2270221` completed passive audit on GPU5/CPU0-19; Stage-C total supervisor PID `2210895` launched gate-data controller; gate-data controller PID `2620830` stopped at Diff-DAgger exact-budget selection and awaits user decision; supervisor interval `900s`.
- `stage_b_roots`: training=`/data/zhaozhixuan/Ask4Help-airplane-5090/results/xvla_fixedgrid_taskpolicy_knee_v1/stage_b_training_v1/`; evaluation=`/data/zhaozhixuan/Ask4Help-airplane-5090/results/xvla_fixedgrid_taskpolicy_knee_v1/stage_b_evaluation_v1/`; supervisor=`/data/zhaozhixuan/Ask4Help-airplane-5090/results/xvla_fixedgrid_taskpolicy_knee_v1/stage_b_total_supervisor_v1/`
- `stage_c_root`: `/data/zhaozhixuan/Ask4Help-airplane-5090/results/xvla_fixedgrid_taskpolicy_knee_v1/stage_c_gate_v1/`; passive controller PID `2210894`; it waits for `STAGE_B_UTILITY_COMPLETE`, builds a missing Airplane `vlm_input_pool` asset from ID metadata if needed, then runs validation-ID calibration and held-out OOD gate audit with a 900-second resource wait.
- `stage_c_downstream`: passive marker triggers total supervisor PID `2210895` at `/data/zhaozhixuan/Ask4Help-airplane-5090/results/xvla_fixedgrid_taskpolicy_knee_v1/stage_c_total_v1/` (900-second interval); data root is `/data/zhaozhixuan/Ask4Help-airplane-5090/results/xvla_fixedgrid_taskpolicy_knee_v1/stage_c_gate_data_v1/`. Gate-training pools are StackCube `154000--154399` and Airplane `164000--164399`; all ten task--method selections must equal the frozen `520/2820` budget before 30 training jobs and 60 utility evaluations can start.
- `diagnostic_partial_evaluation_2026-08-26`: 用户已授权在正式 Stage-B utility evaluation 前先做独立部分评估。GPU4 已独立核验为空闲，控制器 PID `3302716` 以 `--start-now`、CPU20-39 完成运行，不等待 Stage-B 完成；它使用两个已完成的 knee checkpoint（StackCube/Airplane `step_20/seed_17001`），每个 task 做 20-ID + 20-OOD，使用独立 seed `190000--191119` 和独立输出根 `diagnostic_partial_evaluation_v1`。四组 summary 均为 20/20 rows、20/20 videos、20/20 actions：StackCube ID/OOD success=`13/20`/`7/20`，Airplane ID/OOD ever_grasped=`15/20`/`6/20`（strict=`4/20`/`2/20`）。`PARTIAL_EVAL_COMPLETE` 已写入；日志末尾的 `free(): invalid pointer` 发生在完整 artifact 写出之后，仅保留为 cleanup engineering diagnostic。该诊断不计入正式 100-episode 分母，也不改变正式阈值、seed 或训练协议；Stage-B 总控 PID `2926040` 在该 marker 后才允许启动正式评估。
- `stage_b_utility_2026-08-27`: 54/54 formal evaluation summaries and utility summaries are complete and independently denominator-checked (StackCube 30/30, Airplane 24/24; each 100 episodes/100 rows). Under the frozen matched-budget protocol, StackCube OOD success is best at anchor `0` (`0.7233±0.0416`), while calibration knee set is `{10,20}`; Grab Plane OOD ever-grasped is best at anchor `0` (`0.7967±0.0569`), while calibration knee set is `{20}`. Both knee/utility overlaps are empty, so Stage-B currently gives a negative result for “calibration knee predicts downstream SR”; this is a scientific result, not an engineering stop. Full table is tracked in `docs/experiment_management/diagnostics/XVLA_FixedGrid_StageB_Utility_20260827.md`.
- `stage_c_retry_20260827`: first Stage-C passive attempt failed before writing detector artifacts because the Airplane metadata handler was not registered; a second attempt exposed the empty `feature_cache` scaffold left by that failure. Both tracebacks are retained in the remote build log. The handler registration, import-order, and empty-scaffold retry fixes are now in the GitHub branch; the exact builder import-order/data smoke passed (`BUILDER_IMPORT_ORDER_SMOKE_OK`), retry3 PID `1455702` is alive and waiting for GPU5, and no Stage-C scientific rows have been accepted yet.
- `h20_preflight_20260827`: H20 has two idle H100 GPUs and large persistent storage, but its recorded X-VLA runtime is the StackPyramid environment (`/root/X-VLA-stackpyramid-clean`, GPU1 smoke policy) and the standard persistent paths do not contain the current Airplane task checkpoint/metadata bundle. Keeping the 5090 placement avoids an unplanned checkpoint/data/runtime transfer; Stage-C remains waiting for the co-located 5090 GPU5 resource.
- `stage_c_passive_20260827`: Stage-C asset build and passive gate audit completed with `ASSETS_COMPLETE` (12,632 ID observations, 37 layers), 50 validation-ID and 50 held-out-OOD rows per task, frozen validation-ID q=.95 thresholds, and `stage_c_gate_audit_complete_data_pending`. StackCube ID/OOD strict successes were `43/50` and `0/50`; its KD/KHR were Input PCA `27.5/0.20`, Bridge PCA `5.2/0.76`, Action PCA `9.5/0.82`, Diff-DAgger `73.61/0`, fixed Recovery `30/0`. Grab Plane ID/OOD ever-grasped was `47/50` and `24/50` (strict OOD `0/50`); all methods had KHR `0` against knee `{20}` (Input KD `21.7`, Bridge `66.57`, Action `75.87`, Diff `68.86`, fixed Recovery `30`). This is passive timing alignment only; gate-selected policy utility is still pending.
- `stage_c_diffdagger_budget_stop_20260827`: StackCube Diff-DAgger consumed the complete pre-registered `400/400` OOD pool but admitted only `2` full episodes (`52` expert actions), so the frozen whole-episode exact `520`-action selector failed closed. The collection and selector traceback are preserved; no threshold, suffix, success predicate, or method substitution was made. A remote `NEEDS_USER_DECISION_DIFFDAGGER_EXACT_BUDGET` marker is written, and the remaining gate-data training is paused pending approval of a larger pool, a new admission protocol, or an ineligible-branch reconciliation.
- `forbidden`: do not use old Stage-2 timing as formal input; do not launch on any protected GPU; do not tune thresholds/anchors on OOD; do not claim completion from smoke or partial calibration

## OpenDrawer Grasp-OOD Controlled Timing Sweep

- `pipeline_id`: `open_drawer_grasp_timing_sweep_v1`
- `authorized`: `true`; 用户已要求使用现有 ID success rate `>50%` checkpoint，固定 Grasp-OOD takeover timing 并训练/比较 downstream SR
- `owner_thread`: `current-thread`; `owner_label`: `codex-open-drawer-grasp-timing-sweep`
- `server`: `zhaozhixuan@111.198.58.150:12001`; current leadership cap is at most 2 actually idle GPUs/cards, with GPU ownership audited at each job boundary; GPUs with other processes remain protected
- `current_stage`: `direct_oracle_fixed5000_transition`
- `next_stage`: `stop pilot continuation at safe boundary -> audit fixed5000 anchors -> resume anchors 80/120/160/220 to cumulative 5000 -> formal_100_ID_OOD_evaluation -> independent_reconciliation`
- `run_root`: `/data/zhaozhixuan/Ask4Help-open-drawer/results/open_drawer_grasp_timing_sweep_v1/`
- `active_execution_root`: `/data/zhaozhixuan/Ask4Help-open-drawer/results/open_drawer_grasp_timing_sweep_v1_direct_oracle_adaptive_retry1/`; retry5 multi-seed/2500-step results, retry6 Ray-socket failure, retry7 scheduling-interruption partial checkpoints, retry8 marker-preflight failure, retry9 shared-memory scheduling diagnostic, and old formal Oracle data remain diagnostic; the new formal collection and budget will be written under `open_drawer_grasp_timing_sweep_v1_direct_oracle_formal_retry1/`
- `auxiliary_ood20_probe`: `/data/zhaozhixuan/Ask4Help-open-drawer/results/open_drawer_grasp_timing_sweep_v1_retry10_adaptive/ood20_probe/`; 20 Grasp-OOD episodes after each adaptive checkpoint on the held audited GPU, diagnostic only and separate from formal 100-episode denominators
- `priority_ood20_gate_20260829`: 用户最新决定：每个新完成的 timing checkpoint 先完成对应 20 条 Grasp-OOD 诊断评测并通过独立产物审计，再启动下一个 checkpoint 训练；探针资源等待时训练停在边界，seed/anchor/budget/success predicate 与已登记 GPU 候选不变，正式100-ID/100-OOD顺序与分母不变
- `dynamic_gpu_policy_20260830`: 用户曾授权实时监测 GPU；该历史上限不再适用。
- `leadership_gpu_cap_20260831`: 5090 当前最多使用 2 张实际空闲卡。已冻结 4 卡总控，准备在完整周期 checkpoint 后停止两个低进度 anchor，保留 anchor 0/50 两张卡继续；不改变 seed、anchor、预算、阈值、success predicate 或训练规则。
- `adaptive_training_policy_20260830`: 用户新增规则：每个 checkpoint 至少 5000 步；20 条 Grasp-OOD 严格 SR `<=40%` 时每次追加 2500 步，直到首次 `>40%`；冻结该累计步数供后续所有 checkpoint 使用，并严格按“训练→20-OOD 审计→下一训练”交替。retry5 的 2500-step 结果仅 diagnostic。
- `fixed_5000_override_20260901`: 用户明确覆盖上述 adaptive continuation：正式比较固定使用每个 anchor 累计 `5000` steps；anchor 0 使用其 5000-step OOD20 `5/20=25%`，anchor 50 使用其 5000-step OOD20 `15/20=75%`，不再训练或纳入 7500/10000 continuation；剩余 anchor `80/120/160/220` 从可用 partial checkpoint 续训到累计 5000。所有 anchor 仍使用同一 direct-grasp expert、同一共同专家动作预算 `2413`，但轨迹内容随 takeover anchor 不同。该决定已先写入 manifest，再执行进程切换。
- `one_model_per_anchor_20260830`: 用户进一步确定每个 anchor 只训练一个模型，不再重复 seed；当前采用所有 anchor 共用冻结 seed `9301` 的映射，最终比较为 6 个 anchor 模型。每个模型至少5000步，首个 `>40%` 的累计步数冻结给后续 anchor；旧多seed结果不混入。
- `adaptive_retry6_failure_20260830`: retry6 在 Ray 初始化阶段因临时目录过长导致 AF_UNIX socket 超过 107 字节，未生成 checkpoint；失败日志和空 partial root 保留，不计入科学分母。
- `adaptive_retry7_interrupted_20260830`: retry7 已修复 Ray 路径并开始 anchor 0，但在无正式 checkpoint 前因用户要求改为四卡并行而停止；其 partial checkpoints 仅作工程 diagnostic。
- `adaptive_retry8_marker_preflight_20260830`: retry8 在训练前因将 `TIMING_COLLECTION_COMPLETE` 错认在 formal 子目录而退出；未产生训练/评测产物，保留为工程 diagnostic。
- `adaptive_retry9_shared_memory_20260830`: retry9 启动两个单卡 Ray 作业后发现默认每作业约 200 GiB object store，不适合扩展到四卡；在正式 checkpoint 前停止，产物仅作工程 diagnostic。
- `adaptive_active_root_20260830`: adaptive 单 seed 训练/交替 OOD20 结果原写入 retry10，但已因 Oracle repair 暂停；retry10 partial checkpoints 仅 diagnostic。Oracle 修复视频位于 `/data/zhaozhixuan/Ask4Help-open-drawer/results/open_drawer_grasp_timing_sweep_v1_direct_oracle_retry4/`，用户审核通过后才恢复 adaptive 训练。
- `direct_oracle_video_validation_20260830`: 新 `direct_grasp` continuation 已完成六个 anchor，各 3/3 accepted success（18 accepted、24 raw videos）；关闭 drawer 仅用必要 handle path，drawer 已开后直接 object grasp，使用 shortest-joint-path，等待用户视频审核。
- `direct_oracle_handle_grasp_branch_20260830`: 新增双指 `drawer_link` 接触且 TCP 到黄色把手距离不超过 `0.075 m` 的 live-state 判定；已抓住把手时直接从当前 TCP 继续拉动，现场仿真中 `direct_handle_pregrasp_steps=0`、抽屉 qpos 约 `0 -> -0.363`、TCP z 变化约 `0.006 m`。扩展视频验证已在 retry5 完成（15/15 条件、30 accepted、33 raw，含 `t=300`），用户已批准将该 Oracle 作为正式 expert。
- `direct_oracle_formal_recollection_20260831`: 已用 `direct_grasp` 重新收集正式六个 anchor `{0,50,80,120,160,220}`，每个 30 条 accepted；独立 audit 通过，新的最大共同可达整轨迹 budget 已冻结为 `2413` actions，旧 formal 根不参与训练。
- `direct_oracle_adaptive_retry1_20260831`: 已启动每 anchor 一个模型、共同 seed `9301` 的 adaptive training；当前将在 leadership cap 下保留 anchor 0/50 两张卡继续，scratch 使用短路径 `/sdd/r_od1`，其他占用 GPU 保持保护；anchor 80/120 的已落盘周期 checkpoint 保留为可恢复工程产物。原并行总控在资源上限调整后暂停；恢复总控 `tools/run_open_drawer_direct_oracle_adaptive_recovery_controller.py` 已在远端脱离会话启动（PID 由 `recovery_controller.pid` 记录），先接管现存 anchor 0/50，不重复训练，并以 900/1800 秒长间隔推进后续阶段。另有轻量 shell watchdog `tools/watch_open_drawer_adaptive_shell_sleep.sh`（PID 由 `shell_sleep_watch.pid` 记录）实际执行字面 `sleep 900`/`sleep 1800`，只记录健康状态，不重启或抢占资源。
- `recovery_retry2_20260831`: retry1 在两个 retained 进程退出且未写出最终 checkpoint 后已 fail-closed；完整周期 checkpoint `anchor_0=4000`、`anchor_50=3500` 与所有原日志保留。新的恢复总控已从这两个 checkpoint 启动同一累计目标 `5000`（远端 controller PID 由 `recovery_controller.pid` 记录，当前训练 PID 写入 `recovery_active_jobs.json`，GPU4/7），并通过命令行/environment audit 确认 resume source、seed `9301`、budget `2413` 和 canonical output root；不重复使用 base model、不改变科学变量。持久 shell recovery supervisor `tools/watch_open_drawer_adaptive_recovery_restart.sh` 已启动（PID 由 `recovery_supervisor.pid` 记录），仅在 controller fail 且无训练 PID 时以 `sleep 900` 延迟重试。
- `adaptive_formal_eval_20260830`: adaptive 完成后由 `tools/run_open_drawer_adaptive_formal_eval_controller.sh` 按冻结步数逐 anchor 运行 100-ID/100-Grasp-OOD；`tools/summarize_open_drawer_adaptive_timing.py` 独立核对分母、checkpoint、D-path/EAS/DCA 和最终 reconciliation。该控制器不改变训练 seed、anchor、预算或成功定义。
- `resource_preflight_20260830`: H20 两卡只读预检因缺少 OpenDrawer immutable 资产、rootfs 100% 满和其他进程占用而拒绝；5090 保持选中，adaptive 控制器等待无外部进程的实际空闲 GPU。
- `manifest`: `configs/pipelines/open_drawer_grasp_timing_sweep_v1.json`
- `plan`: `docs/experiment_management/plans/OpenDrawer_Grasp_Timing_Sweep.md`
- `source_commit`: `244a1ed`; fixed-timing collector and timing inputs remain synced from the local GitHub branch; adaptive controller `run_open_drawer_adaptive_timing_controller.sh` is recorded in the manifest and synced to the server source tree
- `adaptive_source_commit`: `3be1672`; six-anchor single-seed adaptive controller with 5000-step minimum, 2500-step continuation and audited OOD20 alternation; recovery entrypoint `tools/run_open_drawer_direct_oracle_adaptive_recovery_controller.py` preserves paused lower-step roots and adopts retained PIDs
- `base_checkpoint`: `/sdd/ask4help-open-drawer/results/open_drawer_pi05_v9_recovery_from5000_v4/training/v9_full_prompt/checkpoints/global_step_5000`; its independent ID policy rollout is `61/100` with complete `100/100` videos/actions/states/timelines/reset metadata. It remains labeled preliminary (`>50%` diagnostic base), not `ID_BASE_VALIDATED`.
- `task`: OpenDrawer `grasp_ood` (object yaw 80-100 degrees), max episode steps 400, execute horizon 5; current pure-policy audit is `96/100` drawer-opened, `11/100` ever-grasped, `0/100` strict success.
- `timing_anchors`: provisional frozen diagnostic set `{0,50,80,120,160,220}` representing immediate, pre-open, post-open, object approach, grasp boundary, and post-failure recovery. These are not changed after observing downstream SR; any revision requires a new manifest.
- `diagnostic`: `t=0` single-episode smoke produced a complete expert suffix (`257` actions) and full task-state timeline; planner first failed under pi0.5 2.7/NumPy2 runtime, then succeeded with the isolated `simplerenv_ms3` planner environment. The final smoke process emitted `free(): invalid pointer` only after artifacts were written; retained as cleanup diagnostic.
- `forbidden`: do not reuse old prompt-mismatch checkpoint or q80 gate collections; do not alter Grasp-OOD geometry, ID norm, success predicate, timing anchors after SR, expert budget, or protected GPU ownership; do not claim `PIPELINE_COMPLETE` from the smoke.

## OpenDrawer pi0.5 Continuation From global_step_10000

- `pipeline_id`: `open_drawer_pi05_resume_from10000_v9`
- `authorized`: `true`
- `owner_thread`: `019fdb64-2df4-7a53-9a66-7a5c9b9fe97a`
- `server`: `zhaozhixuan@111.198.58.150:12001`
- `current_stage`: `interrupted_server_reboot_data_mount_missing` (v1 Ray socket; v2/v3 multi-GPU illegal-memory; v4 DCP-load; v5 controller-path; v6 step-offset; v7 disk-capacity; v8 smoke migration; v9 dual-GPU FSDP diagnostics retained)
- `next_stage`: restore `/data` mount and native runtime, reconcile latest complete checkpoint, then resume/evaluate without treating the interruption as scientific failure
- `controller`: no live process after the 5090 reboot; last log step `6484/10000`, loss `0.0164--0.0205` in the final window, valid-action-ratio about `0.98`; complete checkpoints `global_step_2500` and `global_step_5000` remain on `/sdd`; interruption marker `/sdd/ask4help-open-drawer/results/open_drawer_pi05_resume_from10000_v9/ENGINEERING_SERVER_RESTART_DATA_MOUNT_MISSING`
- `run_root`: `/sdd/ask4help-open-drawer/results/open_drawer_pi05_resume_from10000_v9/`
- `retry_diagnostic`: `/data/zhaozhixuan/Ask4Help-open-drawer/results/open_drawer_pi05_resume_from10000_v1/ENGINEERING_RAY_SOCKET_PATH_DIAGNOSTIC`
- `ray_tmp_root`: `/sdd/od_pi05_10k_v9`
- `retry_diagnostics`: v1 Ray socket; v2 CUDA illegal memory; v3 multi-GPU NCCL; v4 single-GPU DCP load hang; v5 controller config path; v6 step-offset mismatch; v7 disk capacity; v8 smoke migration
- `gpu_candidates`: physical GPU2, world-size 1; v9 reuses v8's audited smoke from `/sdd/ask4help-open-drawer/diagnostics/pi05_weights_smoke_v8/`, then saves native 2500/5000/7500/10000 (cumulative 12500/15000/17500/20000), estimated ~76GiB with ~46GiB current margin
- `historical_topology_audit`: prior successful OpenDrawer pi0.5 continuation logs used `FlexiblePlacementStrategy` with `[[2]]` and `local_world_size=1`; current v9 deliberately matches that proven single-GPU topology. The earlier dual-GPU failures are not evidence that the previous successful run used two cards.
- `disk_audit`: last verified approximately `928GB` free on `/data` and `356GB` free on `/sdd`; formal output is on `/sdd`, so the four-checkpoint budget plus safety margin remains satisfied.
- `parallel_gpu_probe`: pure NCCL 2-card all-reduce on physical GPU4/5 passed, but isolated RLinf FSDP 2-card probes failed across FSDP1/FSDP2/no-shard/full-shard variants: initialization/forward illegal-memory or empty-shard errors, and no-shard step1 checkpoint failed in DCP reduce-scatter. Consolidated diagnostic: `/sdd/ask4help-open-drawer/diagnostics/pi05_2gpu_fsdpp_runtime_summary_v1/ENGINEERING_2GPU_RLINF_FSDP_CHECKPOINT_UNSAFE`. Earlier 4-card placement conflict remains at `ENGINEERING_GPU_MAPPING_CONFLICT_OOM`. Formal v9 remains world-size1 on GPU2; no dual-GPU run is promoted.
- `ckpt2500_id_diagnostic`: independent ID-only probe completed on GPU4 with seeds `86000..86019`; strict success `1/20` (`5%`), drawer-opened `16/20` (`80%`), grasp `8/20` (`40%`), lift `6/20` (`30%`), in-target `1/20` (`5%`). Evidence audit passed with `20/20` videos, actions, states, timelines, and reset metadata. Output `/sdd/ask4help-open-drawer/diagnostics/eval_ckpt2500_id20_v1/`; evaluator cleanup emitted `free(): invalid pointer` only after complete artifacts, retained as engineering diagnostic and not treated as evidence failure.
- `checkpoint_watcher`: `/sdd/ask4help-open-drawer/tools/watch_open_drawer_pi05_checkpoints.sh`; waits for native 2500/5000/7500/10000, checks full_weights/DCP sizes and CPU-side meta reload, then writes checkpoint audit and cumulative markers.
- `post_training`: `/sdd/ask4help-open-drawer/tools/run_open_drawer_pi05_v9_post_training_controller.sh`; waits for `TRAINING_COMPLETE`, then runs four 20-ID checkpoint probes, selects by highest strict success with earliest tie, and runs one audited 100-ID gate; never starts OOD automatically.
- `resume_checkpoint`: `/data/zhaozhixuan/Ask4Help-open-drawer/results/open_drawer_failure_detection_v1/id_base_continuation_from4000_v7/training/sft_from4000_to10000/checkpoints/global_step_10000/`
- `checkpoint_evidence`: `full_weights.pt` and `actor/dcp_checkpoint` present
- `contract`: full canonical prompt for the continuation, 8D action, action horizon 10, temporal mask, Flow-SDE, train-expert-only, AWBC false, global batch 128, micro batch 32, frozen ID norm
- `history_warning`: source checkpoint used the older simplified prompt; all prior gates remain diagnostic and are not relabeled
- `smoke`: v7 exact new-step smoke passed; native checkpoint `global_step_2` maps to cumulative `10002`, full_weights+DCP present, loss `0.0216/0.0250` finite, valid_action_ratio `0.967/0.974`, optimizer/scheduler fresh-reset
- `reload_forward`: `/data/zhaozhixuan/Ask4Help-open-drawer/results/open_drawer_pi05_resume_from10000_v7/smoke_weights_10000_to_10002/reload_forward_eval/`; artifacts complete, cleanup abort recorded as `SIMULATOR_EXIT_AFTER_ARTIFACTS`
- `ood`: locked; no PCA/DAgger/OOD before an independent ID gate reaches `>=80/100`

## OpenDrawer pi0.5 Short-Prompt Parallel Continuation

- `pipeline_id`: `open_drawer_pi05_shortprompt_parallel_v1`
- `authorized`: `true` by the latest user decision; current v9 remains running and untouched
- `owner_thread`: `019fdb64-2df4-7a53-9a66-7a5c9b9fe97a`
- `server`: `zhaozhixuan@111.198.58.150:12001`
- `current_stage`: `training_to_20000`; next stage native checkpoint `2500` then continued short-prompt training
- `run_root`: `/sdd/ask4help-open-drawer/results/open_drawer_pi05_shortprompt_from10000_v3/`; v1 Ray path and v2 LR-type failures retained as diagnostics
- `source_checkpoint`: immutable pi0.5 `global_step_10000`; weights-only with fresh optimizer/scheduler
- `prompt`: `open the drawer and place the object in the tray`; this exact prompt must be consumed by both training and future evaluation
- `training_contract`: single GPU4, world size1, global batch128/micro32, 8D action, action horizon10, temporal mask, Flow-SDE, train-expert-only, AWBC=false, conservative lr `5e-6`, warmup100, native checkpoints every2500
- `placement_preflight`: checkpoint/dataset/norm/native runtime co-located on 5090; GPU4 selected idle; GPU2 is protected for v9; output is persistent `/sdd`; short Ray root `/sdd/odsp4`
- `current_stage`: `interrupted_server_reboot_data_mount_missing`; no live process after the 5090 reboot. Last log step `3408/10000`, finite loss about `0.014--0.018`, valid-action-ratio about `0.97`; complete checkpoint `global_step_2500` remains on `/sdd`; interruption marker `/sdd/ask4help-open-drawer/results/open_drawer_pi05_shortprompt_from10000_v3/ENGINEERING_SERVER_RESTART_DATA_MOUNT_MISSING`
- `controller`: `/data/zhaozhixuan/Ask4Help-open-drawer/tools/run_open_drawer_pi05_shortprompt_parallel_controller.sh`; last controller PID `743424`, training PID `834181`; state file is stale after reboot; 2-step smoke passed with full_weights+DCP; GPU4 was correctly isolated before interruption
- `forbidden`: stop or modify v9, touch GPU2/other-user processes, start OOD/PCA/DAgger, alter task/success/norm/mask

## OpenDrawer pi0.5 Recovery After Runtime Restoration

- `pipeline_id`: `open_drawer_pi05_recovery_v4`; `authorized=true`; owner remains A4H15
- `current_stage`: `id_checkpoint_gate_v9_native_5000`; next stage short-prompt native `5000` then `7500` ID gates
- `runtime`: `/sdd/ask4help-open-drawer/runtime/pi05_rlinf_v4/.venv`; torch `2.7.1+cu128`, `sm_120` and CUDA tensor smoke passed; archive `/sdd/ask4help-open-drawer/runtime_archives/pi05_rlinf_v4_20260824.tar.zst` validated
- `v9_full_prompt`: complete native `5000/5000`, target cumulative `20000`, GPU4/CPU80-99, full canonical prompt, original LR `2.5e-5`; training exited cleanly. The fixed-ID gate is complete at `10/20` strict (`drawer_opened=19/20`, `grasp=14/20`, `lift=14/20`, `in_target=10/20`), below the `16/20` qualification threshold. Evidence is complete at `20/20` videos/actions/states/timelines/reset metadata under `/sdd/ask4help-open-drawer/results/open_drawer_pi05_recovery_v4_checkpoint_id_gates/v9_full_prompt/native_step_5000/`; the prior empty waiting directory remains preserved as `native_step_5000_waiting_gpu_diagnostic_20260825`.
- `short_prompt`: native `7500/7500` complete, target cumulative `10000`, GPU5/CPU100-119, prompt `open the drawer and place the object in the tray`, conservative LR `5e-6`; native checkpoints `2500`, `5000`, and `7500` are complete. The command and log prove this recovery starts at native2500 and runs 7500 new steps, so the stale target cumulative `20000` was reconciled to `10000` in `ENGINEERING_STATE_RECONCILIATION_FINAL`. The native5000 fixed-ID gate is `6/20` strict and native7500 is `5/20`, both below threshold; native7500 evidence is complete at `20/20` videos/actions/states/timelines/reset metadata. The watcher-written native5000 gate JSON formatting defect remains preserved and independently reconciled.
- `data/norm/task/mask`: same immutable 128-ID dataset, frozen norm, 8D action, horizon10 temporal mask, Flow-SDE and train-expert-only contract
- `checkpoint_gate_watcher`: watcher PID `2313468` uses the fixed JSON writer and hard free-memory/no-compute-app GPU pool `[4,5,6,7]`; it has verified v9 native5000 and shortprompt native5000 and waits for shortprompt native7500. Fixed ID seeds are `88000..88019`, `max_episode_steps=400`, `execute_horizon=5`; `>=16/20` writes `ID_BASE_VALIDATED_*`, otherwise diagnostic. `/sdd` has about `27GB` free while short training continues, so no checkpoint cleanup or new copies are allowed without a task-owned manifest decision.
- `disk_risk_repair`: after verifying v9 `training_complete` and no v9 process, the finished v9 `smoke_2step` (~19GB) and `runtime_tmp` (~5.8GB) were migrated to `/data/zhaozhixuan/Ask4Help-open-drawer/archive/open_drawer_pi05_v9_recovery_from5000_v4_20260825/`; original `/sdd` paths remain symlinks. Migration manifest records source/destination byte totals and completion-only guard. `/sdd` free space is now about `44GB`; active short-prompt runtime/checkpoints were not touched.
- `forbidden`: GPUs1/2/3 are protected; no multi-GPU FSDP, OOD, PCA or DAgger before ID gate

## OpenDrawer v9 Formal Failure Detection and Three-Stage OOD Override

- `pipeline_id`: `open_drawer_pi05_v9_formal_failure_ood_v1`; `authorized=true` by the explicit 2026-08-25 user decision
- `owner_thread`: `019fdb64-2df4-7a53-9a66-7a5c9b9fe97a`; `server`: `zhaozhixuan@111.198.58.150:12001`
- `base_checkpoint`: v9 full-prompt native5000 at `/sdd/ask4help-open-drawer/results/open_drawer_pi05_v9_recovery_from5000_v4/training/v9_full_prompt/checkpoints/global_step_5000`; independent ID probe `10/20`, so `base_policy_status=preliminary_id_checkpoint` and not `ID_BASE_VALIDATED`
- `user_override`: Oracle gate is `20` independent resets per split with strict `>=18/20`; formal failure-detection rows include FIDeL, CRSAIL, ACC and STAC in addition to internal PCA/LLMD/kNN and Diff-DAgger. The user explicitly permits these benchmark and OOD-training rows to be registered as formal results under the preliminary base; policy qualification remains reported separately.
- `splits`: `handle_ood` handle offset `0.085`, `grasp_ood` yaw `80--100` degrees, `goal_ood` goal center `y=+0.30`; each split remains independent and paired with ID factors.
- `legacy_q95_diagnostic`: `/data/zhaozhixuan/Ask4Help-open-drawer/results/open_drawer_pi05_v9_formal_failure_ood_v1_serial_retry3/` was stopped before completion and is marked `diagnostic_q95_stopped_user_switched_to_q90`; its accepted rows and partial Grasp/PCA parts remain unchanged and are not mixed into the new run.
- `legacy_q90_diagnostic`: `/data/zhaozhixuan/Ask4Help-open-drawer/results/open_drawer_pi05_v9_formal_failure_ood_v1_serial_q90_retry1/` was stopped on Grasp/PCA after `308` raw, `308` videos and `4` accepted (`2 OOD / 2 ID`) because the q=.90 trigger rate was too low; it is marked `Q90_GRASP_PCA_STOPPED_FOR_LOWER_THRESHOLD_DIAGNOSTIC` and is not mixed into q=.80.
- `q80_retry_diagnostics`: retry1 failed because its calibration omitted the required `detectors` mapping; retry2 failed because the controller launch expanded an undefined `CTRL`. Both roots are preserved with engineering diagnostic markers and are excluded from retry3.
- `current_stage`: `grasp_pca_ratio_retry6_collection`; retry4 completed all 12 method/split collections with 100 accepted trajectories each, complete raw videos and datasets, and independent raw-stream audits. The retry4 Grasp/PCA gate failure (`26 ID / 74 OOD`, pilot `12/20` OOD) remains preserved as diagnostic. Retry5 used the same q=`.80` threshold and disjoint seeds but omitted planner environment variables; all 79 trigger attempts were `planner exited before becoming ready`, so retry5 is preserved as an engineering diagnostic. Retry6 restores the known-good planner environment with the same fixed threshold, geometry and seeds `240000/340000`; controller PID `1701330`, collector PID `1701344`, and output root `/data/zhaozhixuan/Ask4Help-open-drawer/results/open_drawer_pi05_v9_formal_failure_ood_v1_serial_q80_retry6_grasp_pca_plannerfix/`. It is accepted only if it reaches 100 accepted with both formal and first-20 pilot OOD ratios `>=80%`; otherwise it writes a final diagnostic and training remains locked. No training or downstream stage is active.
- `disk_risk_update`: the completed task-owned archive `pi05_rlinf_v4_20260824.tar.zst` was byte-checked and migrated from `/sdd/ask4help-open-drawer/runtime_archives/` to `/data/zhaozhixuan/Ask4Help-open-drawer/archive/runtime_archives/pi05_rlinf_v4_20260824/`; the old `/sdd` filename remains a symlink, and active runtime/checkpoints were not touched.
- `disk_risk_update_v2`: the completed, unreferenced `runtime/pi05_rlinf_v3` (`INSTALL_COMPLETE`, about 13G) was regular-file-byte/file-count/rsync-dry-run verified and migrated to `/data/zhaozhixuan/Ask4Help-open-drawer/archive/runtime/pi05_rlinf_v3_20260824/`; the old runtime path remains a symlink. Active `runtime/pi05_rlinf_v4` and all checkpoints were preserved.
- `combined_metrics_update`: independently generated `combined_id_ood_v1` contains 72 rows for `id+handle_ood`, `id+grasp_ood`, and `id+goal_ood` (200 episodes each), using the existing trajectory-max score and frozen ID calibration; separate OOD splits were not merged together.
- `next_stage`: explicit protocol decision for the Grasp/PCA OOD-ratio failure, then (only if the decision preserves a valid pre-registered collection) rebuild the affected collection before matched-budget training and final evaluation. Oracle20 and detector assets are already complete; do not copy Airplane/StackCube metric values or q95/q90 rows into the q80 run.
- `manifest`: `configs/pipelines/open_drawer_pi05_v9_formal_failure_ood_v1.json`
- `retry6_live_update`: planner-fixed Grasp/PCA retry is active at `513 raw / 513 videos / 38 accepted` (`10 ID / 28 OOD`, 73.7% OOD; first-20 pilot 75% OOD), with zero planner errors and strict raw ID/OOD alternation; controller PID `1701330`, collector PID `1701344`. Retry5 remains an engineering diagnostic and retry4's 74% OOD-ratio gate failure remains scientific diagnostic.
- `retry7_live_update`: q=.75 threshold was rebuilt from the same 61 successful ID trajectories, reproducing q=.80=`0.2992814481` and yielding q=.75=`0.2827334106`; ID asset audit passes with zero OOD calibration rows. The active retry7 Grasp/PCA controller is PID `1886504`, collector PID `1886515`, using disjoint seeds `250000/350000`; training remains locked until the formal and first-20 pilot OOD ratios both reach `>=80%`.
- `retry8_live_update`: q=.65 threshold was rebuilt from the same 61 successful ID trajectories, q=.80 reproduction remains `0.2992814481`, and q=.65=`0.2680910826`; ID asset audit passes with zero OOD calibration rows. The retry8 Grasp/PCA controller PID `1968260` and collector PID `1968298` used disjoint seeds `260000/360000` on physical GPU2 and stopped at `211 raw / 211 videos / 48 accepted` (`17 ID / 31 OOD`, 64.6% OOD; first-20 pilot 65%), with planner errors=0 and strict raw alternation. Training remains locked.

## OpenDrawer X-VLA Foundation Adaptation

- `pipeline_id`: `open_drawer_xvla_foundation_v1`
- `authorized`: `true`
- `owner_thread`: `019fdb64-2df4-7a53-9a66-7a5c9b9fe97a`
- `server`: `zhaozhixuan@111.198.58.150:12001`
- `current_stage`: `diagnostic_stopped_user_switched_to_pi05` (latest complete checkpoint `ckpt-35000` retained)
- `next_stage`: none; X-VLA is retained as diagnostic while the pi0.5 continuation is the active mainline
- `controller`: stopped by explicit user model switch; no X-VLA process remains
- `pipeline_state`: `/sdd/ask4help-open-drawer/results/xvla_opendrawer_foundation_adaptation_v8/pipeline_state.json`
- `design_root`: `/sdd/ask4help-open-drawer/results/xvla_opendrawer_foundation_adaptation_v8/`
- `foundation`: `/data/zhaozhixuan/Ask4Help-airplane-5090/results/xvla_airplane_v1/model_cache/X-VLA-Pt-local`
- `domain_id`: `19` (audited unused; no domain table resize)
- `data_root`: `/sdd/ask4help-open-drawer/results/xvla_opendrawer_foundation_adaptation_v2/dataset/xvla_id_512_adapter_space/` (512 episodes, 91,533 frames; CPU adapter complete)
- `plan`: `docs/experiment_management/plans/OpenDrawer_XVLA_Foundation_Adaptation.md`
- `manifest`: `configs/pipelines/open_drawer_xvla_foundation_v1.json`
- `completion`: v8 stopped at the latest complete `ckpt-35000` by explicit user decision; X-VLA selection/formal gate was not promoted; all v7/v8/retry4/retry5 artifacts remain diagnostic
- `execution_audit`: formal gate 前必须独立核对实际 X-VLA entrypoint、foundation/checkpoint、完整 prompt、512-ID manifest、domain id、8D-to-20D adapter、norm、temporal mask、effective batch/gradient accumulation、freeze/warmup、checkpoint reload，以及每个 selection/formal episode 的 video/actions/states/timeline/reset metadata。v7 中断写入 `ENGINEERING_RESOURCE_OR_RUNTIME_DIAGNOSTIC`；v8 smoke/reload-forward 已通过。任一 mismatch 写 `ENGINEERING_PROTOCOL_DIAGNOSTIC`，不得进入正式 gate。

不可变条件：X-VLA foundation、完整 canonical prompt、原 ID 任务/成功定义、512-ID 数据 provenance、temporal/action mask。禁止恢复任何 pi0.5 resume 路线，禁止使用 Airplane/StackCube task checkpoint，ID `<80/100` 禁止 OOD/PCA/DAgger/two-way。

当前推进顺序：

1. CPU-only FK/IK、20D action、inactive-arm mask、30-step temporal-mask 测试（已通过）；
2. foundation domain row 审计并冻结 OpenDrawer `domain_id=19`/soft-prompt（已通过）；
3. 生成并审计 X-VLA 512-ID manifest、anchor/tail/norm；
4. 创建 restart-tolerant Controller；
5. 同拓扑 2-step/reload/checkpoint smoke；
6. 50k fresh adaptation，每5k checkpoint；
7. 固定20-ID selection和独立100-ID gate；ckpt-15000 diagnostic retry5 已完成并通过完整 evidence audit（strict `0/20`，不解锁）；
8. `>=80/100` 后按已批准下游计划继续，否则执行 manifest 中的 ID recovery/科学停止。

## OpenDrawer pi0.5 Representative Detector Calibration Recovery

- `pipeline_id`: `open_drawer_representative_detector_calibration_v3`
- `authorized`: `true`; active Goal owner `019fdb64-2df4-7a53-9a66-7a5c9b9fe97a`
- `server`: `zhaozhixuan@111.198.58.150:12001`
- `current_stage`: `completed_timing_diagnostic_clean_all_layers_not_met`
- `next_stage`: `user_decision_for_more_successful_policy_ID_calibration_or_new_temporal_detector`
- `run_root`: `/sdd/ask4help-open-drawer/results/open_drawer_calibration_recovery_v3/`
- `checkpoint`: `/sdd/ask4help-open-drawer/results/open_drawer_pi05_v9_recovery_from5000_v4/training/v9_full_prompt/checkpoints/global_step_5000`; fixed prior seed=`0`; canonical prompt and ID norm are recorded in the manifest
- `ID_provenance`: expert-ID is the 128-demo full observation bank; policy-success-ID is 20 independent strict-success episodes from `open_drawer_policy_id_calibration_v1_retry3`, matched at video/action/state level
- `representative_layers`: VLM visual input, VLM bridge, VLM block 08, Action Expert block 08, Action Expert final/pre-output
- `detectors`: pooled PCA residual versus source-aware tokenwise PCA+OT; q=`.80/.95`; visual valid masks are retained and action tokens remain time ordered
- `live_processes`: none; expert/policy asset builders and all rescore processes exited after their artifacts were audited
- `asset_audit`: expert-ID tokenwise=`22,973` observations; policy-success raw-replay tokenwise=`4,853` observations from 20 successes plus an expanded `7,303` observations from 31 successes; all raw replays have max qpos error=`0`; pooled, pre-open-safe and phase-aligned assets also audited finite
- `smoke`: tokenwise expert smoke=`1000` observations; policy-success video smoke=`2` episodes (`329` matched observations); remote unit tests=`5 passed`
- `timing_sources`: fixed saved ID source `/sdd/ask4help-open-drawer/results/open_drawer_representative_pca_v1_id_control_retry1` and Grasp-OOD source `/sdd/ask4help-open-drawer/results/open_drawer_representative_pca_v1_retry4/grasp_ood_smoke`; passive replay only, no new policy action sampling
- `forbidden`: no formal OOD collection, DAgger or training in this Goal; old q=.80 outputs remain Bridge-PCA diagnostics and are not relabeled as Input PCA
- `plan`: `docs/experiment_management/plans/OpenDrawer_Representative_Detector_Calibration_Recovery.md`
- `manifest`: `configs/pipelines/open_drawer_representative_detector_calibration_v3.json`
- `results`: `/sdd/ask4help-open-drawer/results/open_drawer_calibration_recovery_v3/metrics_final_v4/comparison.{json,csv,md}`; final audit=`/sdd/ask4help-open-drawer/results/open_drawer_calibration_recovery_v3/FINAL_AUDIT.json`; annotated videos=`/sdd/ask4help-open-drawer/results/open_drawer_calibration_recovery_v3/annotated_videos/` and `annotated_videos_31/`
- `completion`: `OPEN_DRAWER_REPRESENTATIVE_CALIBRATION_COMPLETE` is present. This closes the approved diagnostic plan; it does not claim all five layers meet clean timing, and formal OOD/DAgger/training remain locked.

## StackPyramid Grasp Recovery

- `pipeline_id`: `stackpyramid_grasp_recovery_v1`
- `authorized`: `true`
- `owner_thread`: `019ff58e-8e47-7ca3-a028-07a2705e2c28`
- `server`: `root@39.101.70.188:1012`
- `current_stage`: `passive_pca_failure_detection_complete`
- `next_stage`: `needs_user_decision`
- `controller`: passive PCA retry8 complete; protocol audit passed; no active process
- `remote_pause`: `USER_PAUSED_TRAINING_FOR_HORIZON600_AUDIT` at recorded `global_step=32280`; durable `NEEDS_USER_DECISION` marker is present, so no restart is authorized without a new user decision
- `pipeline_state`: `/mnt/data/ask4help/results/xvla_stackpyramid_oracle_repair_v3/grasp_recovery_v1/pipeline_state.json`
- `baseline_checkpoint`: `/mnt/data/ask4help/results/xvla_stackpyramid_oracle_repair_v3/continuation_50k_from_ckpt10000_lr1e-4_retry1/training/ckpt-40000`
- `baseline_formal`: `/mnt/data/ask4help/results/xvla_stackpyramid_oracle_repair_v3/final_checkpoint_formal_id_gate_100_retry3/`
- `plan`: `docs/experiment_management/plans/StackPyramid_ID_Grasp_Recovery_From_ckpt40000.md`
- `manifest`: `configs/pipelines/stackpyramid_grasp_recovery_v1.json`
- `completion`: recovery checkpoint达到独立100-ID `>=80/100` 并注册为 ID base，或写入预注册科学停止 marker
- `execution_audit`: passive PCA 只读评测在每个 split 收口前独立核对固定 `ckpt-40000`、v4 geometry、green/blue-only shift、paired seeds、600-step horizon、ID-calibrated threshold、原 policy actions、完整分母及 video/actions/states/timeline/reset metadata。PCA 不得改变动作；任一 mismatch 写 `ENGINEERING_PROTOCOL_DIAGNOSTIC`，不得注册最终指标。

Baseline：strict `45/100`、red grasp `56/100`、red place `52/100`、blue lift `51/100`，证据100/100完整。该 checkpoint 是 recovery baseline，不是已接受 ID base。

已完成 baseline failure audit、horizon-450 diagnostic `20/20`（strict `11/20`）和 adapter/gripper audit。ID-only recovery collection 与 audit 已通过：`128/128` accepted、`33,293` anchors、`1,152` tail anchors、`128/128` 视频可解码。用户已于 2026-08-21 批准从 `ckpt-40000` 继续训练额外 `50,000` optimizer steps。沿用已审计的640条ID-only数据、80/20 source-balanced batches、`lr=1e-4`、soft-prompt coefficient `0.1`、`bf16`、batch 8、冻结norm/adapter/formal contract；原proposal中的20k暴露长度由本次用户明确授权覆盖。

当前推进顺序：

1. 对100条 baseline timeline 做 first-bottleneck、hover、gripper、action-repeat、timeout 分类；
2. 同 checkpoint 做独立20条 `450`-step diagnostic，正式300-step协议不变；
3. 审计 8D/20D adapter、gripper sign、normalization、chunk execution；
4. 若确认数据覆盖不足，新增128--256条同ID pre-grasp/contact/close/lift demonstrations；
5. passive PCA failure-detection evaluation 已完成：retry8 固定 `ckpt-40000`、v4、600 steps、ID q=.95 threshold、score-only policy rollout；Stage2/Stage3 ID/OOD 四组均 `100/100` evidence，paired reset/runtime metadata audit PASS，最终 metrics 位于 `/mnt/data/ask4help/results/xvla_stackpyramid_oracle_repair_v3/grasp_recovery_v1/failure_detection_pca_v1_retry8/`。旧 root 因 runtime metadata 缺失保留为 diagnostic；Oracle失败分支仍完全隔离；
6. 固定20-ID selection和独立100-ID gate；
7. `>=80/100` 后才允许进入下游，失败则写科学停止 marker。

## PickSingleYCB Object Variation OOD

- `pipeline_id`: `pick_single_ycb_object_variation_pi05_v1`
- `authorized`: `true`
- `owner_thread`: `019ffbc4-f3a9-78f3-8684-e0b4cba3552a`
- `owner_label`: `codex-object-variation-pick-single-ycb`
- `server`: `zhaozhixuan@111.198.58.150:12001`
- `current_stage`: `diff_collection_scientific_gate_failed`
- `next_stage`: `needs_user_decision` after passive detection, Bridge-PCA, Failure-Recovery and Offline-Oracle completed but Diff-DAgger failed its collection denominator
- `run_root`: `/data/zhaozhixuan/Ask4Help-airplane-5090/results/object_variation_pick_single_ycb_v1/`
- `manifest`: `configs/pipelines/pick_single_ycb_object_variation_pi05_v1.json`
- `plan`: `docs/experiment_management/plans/PickSingleYCB_ObjectVariation_OOD.md`
- `baseline_protocol`: `docs/experiment_management/plans/PickSingleYCB_ObjectVariation_FailureDetection_Baseline_Protocol.md` (mandatory passive baselines, internal extensions, four downstream data/training rows, and evidence gate)
- `task_contract`: generic PickSingleYCB instruction; ID=`005_tomato_soup_can`; OOD=`008_pudding_box`; paired reset differs only in `object_model_id`
- `placement`: 5090 selected because native RLinf/OpenPI/ManiSkill, pretrained pi0.5 base and YCB assets are co-located on persistent `/data`; H20 rejected because an existing Ray owner occupies its resource pool and root filesystem is full
- `runtime_repair`: shared NumPy 2.4.4 remains untouched; pipeline-only NumPy 1.26.4 overlay restores the official MPlib Panda planner, which otherwise exits during `mplib.Planner` construction
- `gpu_plan`: current resume uses independently idle GPU1 with CPU `20-39`; GPU0 is occupied by another user's PID `1023482` (`/ws/bench_fine.py`) and GPU2 belongs to existing PID `1198612`; neither is touched
- `next_action`: user explicitly selected complete `global_step_4500` as the formal ID checkpoint despite its `15/20` ID probe; retain that probe as diagnostic, then run passive failure detection before any OOD data collection
- `evidence_so_far`: Oracle gate `20/20 ID + 20/20 OOD`; ID collection `128/128` with `128/128` videos; data audit `6634` anchors / `1152` tail anchors; ID norm and resume/reload smoke passed; `step4500` probe `15/20` with `20/20` videos/actions; passive detection ID/OOD `100/100`; Bridge-PCA `100/100`, Failure-Recovery `100/100`, Offline-Oracle `100/100`; Diff-DAgger `10/100` after `600` raw attempts; `NEEDS_USER_DECISION` written and matched training/final comparison blocked
- `user_diagnostic_override`: 2026-08-25 user authorized separate Diff threshold-sensitivity collection at override threshold `0.05` with `q=.95`, `patience=2`, preserving canonical threshold `0.6456781893968583`; diagnostic root is `collections_diagnostic_v1/diffdagger_low_threshold_005_retry1`; while it runs, provisional Bridge-PCA, Failure-Recovery, and Offline-BC training may run in the separate `provisional_training_while_diff_collection_v1` root. These outputs are not canonical until the Diff branch is audited and all four methods are retrained under matched expert-action budget.
- `runtime_recovery`: `/data` was re-mounted from the existing `/dev/sdb1`; the original run root was re-audited on `2026-08-24 11:11+08:00`. The selected formal ID checkpoint is the complete `global_step_4500`; all earlier resume retry diagnostics and the four-GPU NCCL diagnostic remain untouched. The old single-GPU training may finish independently, but downstream experiments use only the frozen user-selected step4500 checkpoint and ID norm.
- `completion`: only `PIPELINE_COMPLETE`, `NEEDS_USER_DECISION`, `ORACLE_NOT_ACCEPTED`, `ID_BASE_NOT_ACCEPTED`, or unrecoverable `PIPELINE_FAILED`

## X-VLA Put Vegetable in Basket Object Variation OOD

- `pipeline_id`: `xvla_put_vegetable_basket_object_ood_v1`
- `authorized`: `true`
- `owner_label`: `codex-xvla-vegetable-basket-object-ood`
- `owner_thread`: `current-thread`
- `server`: `root@39.101.70.188:1012`
- `current_stage`: `id_gate_scientific_stop_visible_retry1`
- `next_stage`: `needs_user_decision`
- `run_root`: `/mnt/data/ask4help/results/xvla_put_vegetable_basket_object_ood_v1/`
- `manifest`: `configs/pipelines/xvla_put_vegetable_basket_object_ood_v1.json`
- `plan`: `docs/experiment_management/plans/XVLA_PutVegetableBasket_ObjectVariation_OOD.md`
- `task_contract`: controlled `PutEggplantInBasketScene-v1`; ID=`eggplant`; OOD=`bridge_carrot_generated_modified`; same WidowX/sink/basket/camera/instruction/reset/success; only object asset changes
- `model_contract`: X-VLA `widowx-air` domain id `4`, real action 10D, model max action 20D, action chunk 30, fresh adaptation from `/mnt/data/ask4help/models/X-VLA-Pt_from5090_v4`
- `placement`: H20 X-VLA source/model/runtime are present; GPU0/1 remain registered to existing Ray/RLinf PID `276925`, but the user authorized a scoped shared-idle exception. Repeated pre-launch audits must show zero utilization and stable memory; do not signal or reconfigure the existing PID/Ray session. Root overlay remains 100% full; `/tmp` was cleaned to about 20 GiB free and new outputs/caches go to `/mnt/data` and `/tmp`
- `retry`: `rgb_visible_retry1`; new run root=`/mnt/data/ask4help/results/xvla_put_vegetable_basket_object_ood_v1_rgb_visible_retry1/`; fixed code now follows the official greenscreen exclusion for ID and OOD. The old hidden-RGB dataset/checkpoint remain diagnostic and are forbidden as retry inputs.
- `short_term_goal`: visible-object smoke passed; formal Oracle ID/OOD=`20/20` each; fresh ID collection=`128/128`; temporal-mask/norm audit and 2-step reload smoke passed; fresh ID SFT reached `10000` with every 500-step checkpoint; independent ID gate completed `20/20` episodes and `20/20` videos but achieved `0/20` success. OOD remains locked.
- `scientific_stop`: visible-RGB retry has a complete denominator but failed the ID gate; evidence=`/mnt/data/ask4help/results/xvla_put_vegetable_basket_object_ood_v1_rgb_visible_retry1/provenance/id_gate_evidence_visible_retry1.json`; action diagnostic=`/mnt/data/ask4help/results/xvla_put_vegetable_basket_object_ood_v1_rgb_visible_retry1/diagnostics/policy_action_trace_ckpt10000_seed94000.json`.
- `completion`: only `PIPELINE_COMPLETE`, `NEEDS_USER_DECISION`, `ORACLE_NOT_ACCEPTED`, `ID_BASE_NOT_ACCEPTED`, or unrecoverable `PIPELINE_FAILED`

## X-VLA Panda Put Vegetable in Basket Object Variation OOD

- `pipeline_id`: `xvla_panda_put_vegetable_basket_object_ood_v1`
- `authorized`: `true` by the user-set active goal; this is a new Panda line, not the existing WidowX line
- `owner_thread`: `019ffbc4-f3a9-78f3-8684-e0b4cba3552a`; `owner_label`: `codex-xvla-panda-vegetable-basket-object-ood`
- `server_preference`: 5090 selected after live preflight; H20 rejected because both cards belong to PID `276925` and its root filesystem is full
- `current_stage`: `id_sft_10000_v2`
- `next_stage`: `checkpoint selection -> independent ID gate -> passive detection`
- `run_root`: `/data/zhaozhixuan/Ask4Help-airplane-5090/results/xvla_panda_put_vegetable_basket_object_ood_v1/`
- `manifest`: `configs/pipelines/xvla_panda_put_vegetable_basket_object_ood_v1.json`
- `plan`: `docs/experiment_management/plans/XVLA_PandaPutVegetableBasket_ObjectVariation_OOD.md`
- `task_contract`: Panda BridgeData basket task; ID=`eggplant`, OOD=`eggplant` at fixed scale `1.25`; same sink, basket, prompt, paired reset and success predicate; only object size changes within the main comparison
- `model_contract`: fresh X-VLA-Pt adaptation with a newly audited Panda domain row; active 10D EE6D block padded to 20D; Panda adapter and temporal mask required; WidowX domain 4 and old WidowX outputs forbidden
- `execution_contract`: `preflight -> task/oracle smoke -> 128 ID demos -> ID-only SFT and gate -> passive detection -> four data branches -> matched-budget training -> final evaluation -> result registration`
- `placement_preflight`: passed on 5090; GPU5--7 were idle, `/data` had about 4.7TB free, native X-VLA runtime/foundation/Panda source/BridgeData assets were co-located; controller PID is recorded in remote `pipeline.pid`
- `oracle_gate`: `oracle_gate_v11` passed ID `20/20` and OOD `20/20` strict; profile is lift `0.35m`, release wait `60`, horizon `150`, object-local-y closing axis, three fresh-environment retries retained under `raw_attempts/`
- `controller`: `tools/run_xvla_panda_vegetable_basket_full_pipeline.py`; remote PID is in `pipeline.pid`; current child is ID SFT on physical GPU5/7 with gloo, batch8 and accumulation8 (effective global batch128)
- `forbidden`: do not modify or reuse `xvla_put_vegetable_basket_object_ood_v1`; do not mix WidowX data/norm/checkpoint/results; do not unlock OOD before the independent ID gate
- `completion`: only `PIPELINE_COMPLETE`, `ORACLE_NOT_ACCEPTED`, `ID_BASE_NOT_ACCEPTED`, `NEEDS_USER_DECISION`, or unrecoverable `PIPELINE_FAILED` after evidence audit

## Heartbeat Watchdog Rule

Heartbeat 每次只做：读取本文件、读取两个 manifest、核对实际 PID/controller/marker/progress。如果 `authorized=true` 且 `next_stage` 非空，但没有健康进程或完成 marker，必须唤醒 Owner 启动/修复，不得返回“无变化”。

对已授权的完整 pipeline，Heartbeat/Owner 必须持续监测并推进到 `PIPELINE_COMPLETE`、预注册科学停止、`NEEDS_USER_DECISION` 或用户明确暂停；不得因为设计、smoke、Oracle、collection、checkpoint、partial evaluation 或任一中间阶段完成就停止。用户要求“持续完成这个任务，直到完成目标”时，该要求属于本 pipeline 的持续监督契约。
