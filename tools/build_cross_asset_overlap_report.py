#!/usr/bin/env python3
"""Complete readable inventory, including missing metrics and negative results."""
import argparse,json,re,subprocess
from pathlib import Path

NAMES={'fixedgrid_stackcube':'X-VLA StackCube：固定接管时刻','fixedgrid_airplane':'X-VLA Grab Plane：固定接管时刻',
 'opendrawer_probe':'OpenDrawer Grasp-OOD：5000步，20条probe','stack_stage2_timing':'X-VLA StackCube Stage 2：时刻实验',
 'stack_stage2_gates':'X-VLA StackCube Stage 2：gate比较','ycb_object':'π0.5 PickSingleYCB：Object Variation',
 'xvla_stack_gates':'X-VLA StackCube：旧四组gate','xvla_plane_gates':'X-VLA Grab Plane：旧四组gate',
 'pi05_plane_v1':'π0.5 Grab Plane：training_v1','pi05_plane_v2':'π0.5 Grab Plane：training_v2',
 'openvla_plane':'OpenVLA Grab Plane','pi05_stack_knn':'π0.5 StackCube：kNN v3','pi05_stack_offline':'π0.5 StackCube：Offline v3',
 'pi05_stack_diff':'π0.5 StackCube：Diff v5','pi05_stack_recovery':'π0.5 StackCube：Recovery v5'}

def main(root):
 root=root.resolve();comparison=json.loads((root/'comparison.json').read_text());inventory=json.loads((root/'inputs/source_schema_inventory.json').read_text())
 rows=list(comparison['rows']);utility=comparison['utility_inventory'];budget=inventory['5090']['ycb_budget']['common_expert_action_budget']
 skip={'fixedgrid','stack_stage2_timing','stack_stage2_gates','drawer_probe_live'}
 for cohort in dict.fromkeys(u['cohort'] for u in utility if u['cohort'] not in skip):
  for method in dict.fromkeys(u['method'] for u in utility if u['cohort']==cohort):
   us=[u for u in utility if u['cohort']==cohort and u['method']==method];ood=next(u for u in us if u['split']=='ood');idrow=next((u for u in us if u['split']=='id'),None)
   match=re.search(r'(?:global_step_|ckpt-|checkpoint_)(\d+)',ood['checkpoint'])
   if not match:match=re.search(r'eval_ood_step(\d+)',ood['summary_path'])
   rows.append(dict(cohort=cohort,task=NAMES[cohort],condition=method,budget=budget if cohort=='ycb_object' else None,
    training_steps=int(match.group(1)) if match else None,Q=None,ID_SR=idrow['SR'] if idrow else None,OOD_SR=ood['SR'],OOD_successes=ood['successes'],OOD_episodes=ood['episodes'],
    endpoint=ood['endpoint'],OOD_strict_successes=ood['strict_successes'],evaluation=f"1 training seed x {ood['episodes']} episodes",utility_evidence=[u['summary_path'] for u in us],
    metric_status='MISSING_DYNAMIC_OBJECT_AND_CONTACT_EVIDENCE',checkpoint=ood['checkpoint'],
    protocol_caveat='Diff threshold=0.05 diagnostic branch; not the frozen canonical threshold' if cohort=='ycb_object' else 'historical cohort; matched-budget equivalence not established here'))
 (root/'all_results.json').write_text(json.dumps({'rows':rows,'computed_conditions':sum(r['Q'] is not None for r in rows),'missing_conditions':sum(r['Q'] is None for r in rows)},indent=2,allow_nan=False))
 lines=['# 全资产轨迹重合度 × Downstream Utility','',
  f"本次找到并列入 **{len(rows)}组条件／方法**：**21组**可计算同一定义下的轨迹重合度，另外**22组**已核对SR，但缺少完整动态状态，Q保留为空。不是所有任务都完成了重合度验证。",'',
  '共读取111份逐episode评测summary（包括额外1份t0/7500步probe）；重合度主表不纳入该额外7500步点。完成15个新着色视频与4个合辑，OpenDrawer沿用已审阅的敏感性视频。','',
  '## 核心发现','',
  '- StackCube固定时刻：×3的Pearson=0.850，Spearman=1.000（5组）；×2也保持排序一致。',
  '- Grab Plane固定时刻：×3的Pearson=0.616，Spearman=0.800（4组）。t20重合度高于t30，但SR更低：不是完美排序。',
  '- OpenDrawer：×3的Pearson=0.726，Spearman=0.771（6组20条probe）。t220仍是反例；测试seed块不同，不是配对评测。',
  '- StackCube Stage2 gate：PCA的×3 Q=0.265、SR=0.74；Diff的Q=0.189、SR=0.45。仅两种方法，不能据此估计有意义的相关系数。',
  '- **Stage2时刻实验是重要反例**：×3时Pearson=-0.800。post-grasp/post-lift模型仅13/100、15/100曾抓到方块，失败大多早于被指标衡量的搬运片段。后段高重合度不能保证前段能力，也不能保证整任务SR。','',
  '这些结果是事后探索：不按SR选择倍数、排除反例或重定义成功。不将跨task/backbone/protocol数据汇总成一个相关系数。','',
  f"![分任务相关性]({root/'overlap_vs_utility.png'})",'',
  '## 完整数值表','',
  'Q是目标片段内兼容点数／全部新增专家轨迹点数，取值0—1。×1/×2/×3只改变匹配半径，参考、对齐、接触条件与分母固定。表中“—”是未知／缺少，**不是0**。','']
 for cohort in dict.fromkeys(r['cohort'] for r in rows):
  lines+=['### '+NAMES[cohort],'','| 条件／方法 | 实际专家预算 | 训练步 | Q×1 | Q×2 | Q×3 | ID SR | OOD SR | 评测规模 |','|---|---:|---:|---:|---:|---:|---:|---:|---|']
  for r in rows:
   if r['cohort']!=cohort:continue
   q=' | '.join(f"{r['Q'][str(f)]:.4f}" for f in [1,2,3]) if r['Q'] else '— | — | —'
   ids='—' if r['ID_SR'] is None else f"{r['ID_SR']:.4f}"
   lines.append(f"| {r['condition']} | {r['budget'] if r['budget'] is not None else '未核定同预算'} | {r['training_steps']} | {q} | {ids} | {r['OOD_SR']:.4f} | {r['evaluation']} |")
  if cohort in ['fixedgrid_airplane','xvla_plane_gates','pi05_plane_v1','pi05_plane_v2','openvla_plane']:lines+=['','此处SR保留原始 `ever_grasped` 口径，不是复核后的稳定机身抓取成功率。strict completion另存于原始summary/完整JSON。','']
  if cohort=='ycb_object':lines+=['','同为5954新增专家点、5000训练步、100-ID/100-OOD。Diff使用0.05低阈值诊断分支；不冒充原冻结阈值。没有动态物体和接触记录，暂不提供精确重合度。','']
  if cohort=='opendrawer_probe':lines+=['','SR为Grasp-OOD条件下端到端success，不是ever_grasped；已核对20条逐轨迹结果。正式100条在该输出根仍无summary。','']
 lines+=['','## Stage2反例的失败阶段核对','','| 接管方式 | 曾抓住 | 曾抬起 | 曾叠上 | 最终成功 |','|---|---:|---:|---:|---:|']
 evidence=json.loads((root/'inputs/utility_evidence_h20.json').read_text())
 for v in evidence['stack_stage2_timing']['files']:
  s=v['summary']
  if s.get('split')!='stage2_ood':continue
  method=Path(v['relative_path']).parts[-2];counts=[sum(bool(r[k]) for r in s['rows']) for k in ['grasped_once','lifted_once','on_cube_once','success']]
  lines.append('| '+method+' | '+' | '.join(f'{c}/100' for c in counts)+' |')
 lines+=['','这支持“下游前置技能缺失可能限制SR”的解释；没有证明具体训练因果。不能把它改写成“只要后段重合度高就足够”。','',
  '## 视频','',
  '每个新视频显示同一真实帧下的×1/×2/×3分类，区分整组Q、示例轨迹Q和整组下游SR。蓝色为policy前缀；红色只表示不匹配，不能直接称作Recovery。','']
 for name,label in [('stackcube_all_timings_x1_x2_x3.mp4','StackCube：五个接管时刻'),('airplane_all_timings_x1_x2_x3.mp4','Grab Plane：四个接管时刻'),('stack_stage2_timing_all_x1_x2_x3.mp4','StackCube Stage2：四种接管方式（含反例）'),('stack_stage2_gates_all_x1_x2_x3.mp4','StackCube Stage2：PCA与Diff')]:
  lines.append(f"- [{label}]({root/'videos'/name})")
 for file in ['video_audit.json','stage2_video_audit.json']:
  lines+=['','单个视频（'+file.removesuffix('_audit.json')+'）：','']
  for v in json.loads((root/file).read_text()):lines.append(f"- [{Path(v['output']).stem}]({v['output']})")
 drawer=Path('artifacts/open_drawer_tolerance_sweep_20260903/videos').resolve()
 lines+=['','OpenDrawer此前的×1/×2/×3同帧视频：','']
 for v in sorted(drawer.glob('*.mp4')):lines.append(f'- [{v.stem}]({v})')
 ycb=root/'ycb_unscored_video_manifest.json'
 if ycb.exists():
  lines+=['','PickSingleYCB Object的真实已选专家后缀示例（**未着色，Q未计算**）：','']
  for v in json.loads(ycb.read_text()):lines.append(f"- [{v['method']}，seed {v['seed']}，接管t={v['expert_start']}]({v['local_video']})")
 lines+=['','## 缺少Q的分支：需要补什么','',
  'PickSingleYCB Object、旧X-VLA/π0.5/ OpenVLA gate结果保留真实SR，但原始训练数据通常只有9D关节状态、动作和视频。能由正运动学恢复TCP位置、姿态与开合，却不能唯一恢复动态物体位置、接触事实或物体相对姿态。末帧pose／初始pose不能替代全程轨迹，policy被动轨迹也不能替代实际专家后缀。','',
  '下一步优先对**原有被训练选中的专家轨迹**进行少量确定性回放验证：逐步对齐原始qpos/动作；只有回放一致才批量补记录动态物体与接触状态。回放不一致则需要原始状态快照或在新记录协议下重采集，不能凭视频猜测后填。补这些记录本身不需要重训已存在的模型，但重新采集的不同轨迹不能冒充其原训练集。','',
  '之前Airplane回放出现状态漂移，因此本轮没有启动未经验证的批量仿真回放，也没有新增训练或占用GPU。','',
  'StackPyramid按用户此前“先不管”的要求未纳入；未获通过ID基线或没有闭合downstream结果的新任务不强行填入主表。此表是本轮已定位并核对的历史资产清单，并非整个服务器全部文件的无限穷举。','',
  '## 复核材料','',
  f"- [完整逐条件JSON]({root/'all_results.json'})",f"- [计算细节与相关性]({root/'comparison.json'})",f"- [独立数值审计]({root/'independent_audit.json'})",f"- [原始source schema盘点]({root/'inputs/source_schema_inventory.json'})",'']
 (root/'完整结果与视频.md').write_text('\n'.join(lines));print('FULL_REPORT',len(rows),'conditions',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
