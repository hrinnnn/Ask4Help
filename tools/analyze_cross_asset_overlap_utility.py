#!/usr/bin/env python3
"""Frozen-radius sensitivity and within-cohort utility joins; no fitting to SR."""
import argparse, copy, json
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FACTORS=[1.,2.,3.]
FIXED=Path('artifacts/stackcube_airplane_target_ratio_20260831')
DRAWER=Path('artifacts/open_drawer_tolerance_sweep_20260903')

def fixed_variants(output):
 original=json.loads((FIXED/'analysis.json').read_text());variants={}
 for factor in FACTORS:
  variant=copy.deepcopy(original)
  for task,analysis in variant.items():
   poses={(r['step'],r['seed']):dict(np.load(FIXED/task/f"pose_step_{r['step']}_seed_{r['seed']}.npz")) for r in analysis['rows']}
   for r in analysis['rows']:
    query=poses[r['step'],r['seed']];ref=poses[0,r['reference_seed']];labels=[]
    for i,d in enumerate(r['distance']):
     if d is None:labels.append('non_target');continue
     phase=next(p for p,(a,b) in r['blocks'].items() if a<=i<b)
     contact=bool(query['contact'][i])==bool(ref['contact'][r['reference_mapping'][i]])
     labels.append('target_compatible' if contact and d<=r['thresholds'][phase]*factor else 'target_mismatch')
    if factor==1.:assert labels==r['labels'],(task,r['step'],r['seed'])
    r['labels']=labels;r['compatible_anchors']=labels.count('target_compatible');r['Q']=r['compatible_anchors']/r['expert_anchors']
    r['baseline_thresholds']=dict(r['thresholds']);r['thresholds']={p:t*factor for p,t in r['thresholds'].items()}
    for p,(a,b) in r['blocks'].items():r['phase_stats'][p]['compatible']=labels[a:b].count('target_compatible')
   for step,s in analysis['summary'].items():
    chosen=[r for r in analysis['rows'] if r['selected'] and r['step']==int(step)]
    n=sum(r['expert_anchors'] for r in chosen);assert n==s['expert_anchors']
    s['compatible_anchors']=sum(r['compatible_anchors'] for r in chosen);s['Q']=s['compatible_anchors']/n
   analysis['protocol']['radius_multiplier']=factor
   analysis['protocol']['diagnostic_status']='POST_HOC; utility known before tolerance comparison'
  variants[factor]=variant
  p=output/f'fixedgrid_x{factor:g}';p.mkdir(parents=True,exist_ok=True);(p/'analysis.json').write_text(json.dumps(variant,indent=2,allow_nan=False))
 return variants

def utility_rows(root):
 all_rows=[]
 for filename in sorted((root/'inputs').glob('utility_evidence_*.json')):
  for cohort,group in json.loads(filename.read_text()).items():
   for entry in group['files']:
    a=entry['summary'];audit=entry['audit'];n=a.get('episodes',a.get('num_episodes'))
    if not isinstance(n,int):continue
    parts=entry['relative_path'].split('/');ckpt=a.get('checkpoint',a.get('checkpoint_dir',''))
    if cohort=='fixedgrid':task=parts[0];method=parts[1];seed=parts[2];split=parts[3]
    elif cohort=='drawer_probe_live':task='opendrawer';method=parts[0];seed=parts[1];split=a['split']
    else:
     task=cohort;seed='single';split=a.get('split',next((p for p in parts if p in ['id','ood','stage2_ood']),'ood'))
     method=cohort.removeprefix('pi05_stack_') if len(parts)==1 else parts[-2]
     if method in ['id','ood','stage2_ood','id100','ood100']:method=parts[-3]
    endpoint='ever_grasped' if 'ever_grasped_recount' in audit and cohort!='drawer_probe_live' else 'success'
    successes=a.get('ever_grasped_successes',a.get('ever_grasped',audit.get('ever_grasped_recount'))) if endpoint=='ever_grasped' else a.get('successes')
    if successes is None:continue
    recount=audit.get(endpoint+'_recount');verified=audit['row_denominator_matches'] is True and recount==successes
    all_rows.append(dict(cohort=cohort,task=task,method=method,training_seed=seed,split=split,episodes=n,
     successes=successes,SR=successes/n,endpoint=endpoint,strict_successes=a.get('strict_successes',a.get('strict_success',audit.get('success_recount') if endpoint=='ever_grasped' else None)),
     checkpoint=ckpt,summary_path=entry['path'],denominator_and_labels_verified=verified,audit=audit))
 return all_rows

def build(root):
 variants=fixed_variants(root);utilities=utility_rows(root);rows=[]
 for task in ['stackcube','airplane']:
  for step,s in variants[1.][task]['summary'].items():
   data=[u for u in utilities if u['cohort']=='fixedgrid' and u['task']==task and u['method']=='step_'+step]
   ood=[u for u in data if u['split']=='ood'];ids=[u for u in data if u['split']=='id']
   assert len(ood)==3 and len(ids)==3 and all(u['denominator_and_labels_verified'] for u in data)
   rows.append(dict(cohort='fixedgrid_'+task,task=task,condition='t='+step,takeover_step=int(step),budget=s['expert_anchors'],training_steps=2500,
    Q={str(int(f)):variants[f][task]['summary'][step]['Q'] for f in FACTORS},
    compatible_points={str(int(f)):variants[f][task]['summary'][step]['compatible_anchors'] for f in FACTORS},
    ID_SR=float(np.mean([u['SR'] for u in ids])),OOD_SR=float(np.mean([u['SR'] for u in ood])),
    OOD_seed_SR=[u['SR'] for u in ood],OOD_SR_std=float(np.std([u['SR'] for u in ood],ddof=1)),
    evaluation='3 training seeds x 100 episodes per split',endpoint=ood[0]['endpoint'],utility_status='LIVE_SUMMARY_ROWS_VERIFIED',
    utility_evidence=[u['summary_path'] for u in data],interpretation_caveat='ever_grasped is not verified stable neck grasp' if task=='airplane' else None))
 # Same user-provided probe values, now independently recounted from remote rows.
 for step,successes in [(0,5),(50,15),(80,16),(120,7),(160,8),(220,13)]:
  Q={};counts={}
  for f in FACTORS:
   a=json.loads((DRAWER/f'factor_{f:.2f}'.replace('.','p')/'analysis.json').read_text())
   s=a['summary'][str(step)]['training_budget'];Q[str(int(f))]=s['Q'];counts[str(int(f))]=s['compatible_anchors']
  evidence=next(u for u in utilities if u['cohort']=='drawer_probe_live' and u['method']==f'anchor_{step}' and '/steps_5000/' in u['summary_path'])
  assert evidence['successes']==successes and evidence['episodes']==20 and evidence['denominator_and_labels_verified']
  rows.append(dict(cohort='opendrawer_probe',task='opendrawer',condition='t='+str(step),takeover_step=step,budget=2413,training_steps=5000,
   Q=Q,compatible_points=counts,ID_SR=None,OOD_SR=successes/20,OOD_successes=successes,OOD_episodes=20,
   evaluation='1 training seed x 20 episode probe (different test seed blocks)',endpoint='end-to-end success under grasp OOD, not ever_grasped',utility_status='LIVE_PROBE_ROWS_VERIFIED',utility_evidence=[evidence['summary_path']]))
 stage=root/'stage2_overlap.json'
 if stage.exists():
  sr=json.loads(stage.read_text())
  for cohort,group in sr['summaries'].items():
   for method,s in group.items():
    data=[u for u in utilities if u['cohort']==cohort and u['method']==method]
    ood=[u for u in data if u['split']=='stage2_ood'];ids=[u for u in data if u['split']=='id'];assert len(ood)==len(ids)==1
    rows.append(dict(cohort=cohort,task='stackcube_stage2',condition=method,budget=s['expert_points'],training_steps=2000,
     Q=s['Q'],compatible_points=s['compatible_points'],ID_SR=ids[0]['SR'],OOD_SR=ood[0]['SR'],evaluation='1 training seed x 100 episodes per split',
     endpoint='success',utility_status='LIVE_SUMMARY_ROWS_VERIFIED' if all(x['denominator_and_labels_verified'] for x in data) else 'INCOMPLETE_AUDIT',
     utility_evidence=[u['summary_path'] for u in data]))
 correlations={}
 for cohort in dict.fromkeys(r['cohort'] for r in rows):
  group=[r for r in rows if r['cohort']==cohort];correlations[cohort]={}
  for f in FACTORS:
   x=[r['Q'][str(int(f))] for r in group];y=[r['OOD_SR'] for r in group]
   correlations[cohort][str(int(f))]=dict(n=len(group),pearson=float(pearsonr(x,y).statistic),spearman=float(spearmanr(x,y).statistic)) if len(group)>=3 else dict(n=len(group),note='two method values are not meaningful correlation evidence')
 result=dict(status='OFFLINE_EXPLORATORY_ASSOCIATION_NOT_INDEPENDENT_VALIDATION',rows=rows,correlations=correlations,utility_inventory=utilities)
 (root/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False))
 lines=['# 轨迹重合度与 Downstream Utility','',
  '性质：已知SR之后的离线敏感性分析，不是独立预测验证。统一展示原半径、2倍、3倍，不按结果选择任务或倍数。Q分母为实际新增专家轨迹点数；不是policy成功率。','',
  '绿色表示目标片段内几何及接触相容；红色只表示不相容，不能直接称作失败恢复；灰色非目标点仍保留在分母。','']
 for cohort in correlations:
  lines+=['## '+cohort,'','| 条件 | 专家点数 | Q×1 | Q×2 | Q×3 | ID SR | OOD SR | 评测 |','|---|---:|---:|---:|---:|---:|---:|---|']
  for r in rows:
   if r['cohort']!=cohort:continue
   idstr='—' if r['ID_SR'] is None else f"{r['ID_SR']:.4f}"
   lines.append(f"| {r['condition']} | {r['budget']} | "+' | '.join(f"{r['Q'][str(int(f))]:.4f}" for f in FACTORS)+f" | {idstr} | {r['OOD_SR']:.4f} | {r['evaluation']} |")
  lines+=['','相关性（n是条件数，不把同一Q的多个训练seed当作独立Q样本）：','']
  for f,c in correlations[cohort].items():lines.append(f"- ×{f}: n={c['n']}, Pearson={c['pearson']:.3f}, Spearman={c['spearman']:.3f}" if 'pearson' in c else '- 仅两组方法，不计算相关性。')
 lines+=['','## 比较边界','',
  '- StackCube/Grab Plane固定网格使用相同预算与2500训练步；每条件3训练seed。OpenDrawer为已逐条核对的单训练seed、每条件20条probe；不同条件使用不同测试seed块，不是配对评测，不与前两者合并。',
  '- OpenDrawer这6个数是Grasp-OOD条件下的端到端success，不是ever_grasped。额外t0的7500步probe不混入统一5000步比较。',
  '- Grab Plane此处SR严格保留旧口径ever_grasped；未补做稳定机身抓取复核，不能解释为该新成功定义。',
  '- 放宽阈值是在已知结果后进行；当前相关性只能用于生成假设。尤其保留晚接管和中间时刻的排序反例。',
  '- 固定网格坐标系是当前物体中心+本条轨迹稳定抓取时TCP轴，不是完整动态物体坐标系；不能称作实时部署指标。',
  '- 其他gate分支的SR已单独读取；若缺少相应训练轨迹的物体/接触状态，不用passive轨迹冒充专家后缀，不填造Q。','']
 (root/'README.md').write_text('\n'.join(lines))
 fig,axes=plt.subplots(2,3,figsize=(15,8.5),squeeze=False,constrained_layout=True)
 for ax,cohort in zip(axes.flat,correlations):
  group=[r for r in rows if r['cohort']==cohort]
  for f,color in [(1.,'#417ab5'),(2.,'#d99b32'),(3.,'#258667')]:
   ax.scatter([r['Q'][str(int(f))] for r in group],[r['OOD_SR'] for r in group],label=f'x{f:g}',c=color)
  for r in group:
   dy=-12 if cohort=='fixedgrid_stackcube' and r['condition']=='t=30' else 7 if cohort=='fixedgrid_stackcube' and r['condition']=='t=45' else 3
   ax.annotate(r['condition'],(r['Q']['3'],r['OOD_SR']),xytext=(4,dy),textcoords='offset points',fontsize=8)
  xs=[r['Q'][str(int(f))] for r in group for f in FACTORS];width=max(xs)-min(xs);ax.set_xlim(max(0,min(xs)-.08*width),max(xs)+.40*width)
  ax.set(title=cohort,xlabel='Trajectory overlap Q (not SR)',ylabel='OOD downstream utility',ylim=(0,1));ax.legend();ax.grid(alpha=.2)
 for ax in list(axes.flat)[len(correlations):]:ax.set_visible(False)
 fig.savefig(root/'overlap_vs_utility.png',dpi=160);plt.close(fig)
 print(json.dumps({'rows':len(rows),'utility_summaries':len(utilities),'correlations':correlations}),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);build(p.parse_args().root)
