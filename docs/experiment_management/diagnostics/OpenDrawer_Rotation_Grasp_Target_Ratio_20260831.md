# OpenDrawer rotation-and-grasp target-anchor ratio

日期2026-08-31，owner=codex-root-open-drawer-target-ratio。用户指定Grasp-OOD目标为旋转调整后夹住，不含lift/transport/place。本轮仅离线计算，不改训练/gate。

产物：`artifacts/open_drawer_target_ratio_20260831/` 内README、analysis.json、audit.json、target_ratio.png、2个逐点着色视频。源数据为new direct-oracle formal retry1，主表采用同root的既有2413-action预算整轨迹选择清单。

## 定义与计算

Q=目标阶段内、与成功参考的物体相对TCP位姿/开合/接触相容的真实专家anchors数，除以全部选入训练的真实专家anchors数。分母保留开抽屉、lift、transport、place等非目标动作，不计policy前缀及padding。

目标开始=释放把手2个开夹命令后的object pregrasp/旋转运动；包含随后对准下探，结束于close阶段3步稳定夹持完成，半开区间不含第一条lift动作。180成功轨迹的Oracle阶段步数与总专家动作数全部核对一致。

使用30条完整OOD专家拆18参考/6校准/6检查。同OOD参考库中选择一个最匹配完整目标序列的专家，子阶段内单调DP，禁止逐点换专家。query属于anchor0时同时从reference/calibration排除并重算。

局部误差尺度=2cm/15deg/10mm，q=.925按子阶段校准；组合半径下限0.13017，源于1mm/1deg/1mm标准化量。最终rotation半径0.71335，reach/close为0.13017。此下限明确为诊断容差，不是SR最优参数。

世界系版本把物体摆放差异当偏移，已保留为diagnostic；当前使用物体相对框架，需要object pose与contact标签，不声称纯本体、无标注部署。检查6条专家的目标兼容点=232/251，其中close=18/18；这些专家曾用于工程诊断，不视为未触碰的正式测试集。

## 主结果

| takeover | selected episodes | target-compatible anchors / 2413 | Q |
|---:|---:|---:|---:|
| 0 | 15 | 626 | 25.94% |
| 50 | 21 | 884 | 36.63% |
| 80 | 25 | 899 | 37.26% |
| 120 | 28 | 569 | 23.58% |
| 160 | 28 | 493 | 20.43% |
| 220 | 28 | 398 | 16.49% |

q=.90–.95的本轮检查保留相近趋势，但没有证明t80最优，t50/t80仅相差0.62个百分点。

同t220视频案例：seed78829近似正常继续，42/82=51.2%；seed78817返回并纠正，9/88=10.2%。正常参考seed78324的目标区间[79,123)，后续lift不进入分子。

全部262 attempts/180 accepted/82 incomplete保留；主表只是已选入训练的专家语料占比，不是含失败采集成本的端到端效率。未证明与downstream SR正相关，没有梯度影响或跨任务普适性声明。
