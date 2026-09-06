"""OOD-only target numerator, all original selected expert points denominator."""
import copy,json,re
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr,spearmanr

SOURCE=Path('artifacts/pi05_table_tasr_20260903')
OUT=Path('artifacts/pi05_tasr_ood_gap_20260906')

def main():
 OUT.mkdir(exist_ok=True)
 utilities=json.loads((SOURCE/'pi05_tasr_table_values.json').read_text())['rows'];results={};flat=[]
 for task in ['stackcube','airplane','object']:
  original=json.loads((SOURCE/f'tasr_{task}.json').read_text());a=copy.deepcopy(original)
  a['protocol'].update(numerator_domain='OOD only; ID excluded regardless of geometric similarity',denominator_domain='All actual selected new expert points, ID plus OOD',
   target_semantic_status='Task-stage proxy: grasp approach/alignment and stable close; not independently identified policy skill gap',
   correction='Domain membership only; phase bounds, correspondence, radii and references unchanged')
  for r in a['rows']:
   assert r['split'] in ['id','ood']
   r['old_compatible_points']=dict(r['compatible_points']);r['old_Q']=dict(r['Q'])
   if r['split']=='id':
    r['labels']={f:['excluded_id']*r['expert_points'] for f in ['1','2','3']}
    r['compatible_points']={f:0 for f in ['1','2','3']};r['Q']={f:0. for f in ['1','2','3']}
  summaries={}
  for method,old in original['summary'].items():
   rows=[r for r in a['rows'] if r['selected'] and r['method']==method];n=sum(r['expert_points'] for r in rows);assert n==old['total_expert_points'] and old['missing_expert_points']==0
   id_n=sum(r['expert_points'] for r in rows if r['split']=='id');ood_n=n-id_n
   green={f:sum(r['compatible_points'][f] for r in rows) for f in ['1','2','3']}
   removed={f:sum(r['old_compatible_points'][f] for r in rows if r['split']=='id') for f in ['1','2','3']}
   for f in green:assert green[f]+removed[f]==old['compatible_points'][f]
   sr=next(u for u in utilities if u['task']==task and u['method']==method)
   s=dict(task=task,method=method,total_expert_points=n,id_points=id_n,ood_points=ood_n,ood_budget_share=ood_n/n,
    compatible_points=green,removed_id_compatible_points=removed,TASR={f:green[f]/n for f in green},old_TASR=old['TASR'],
    ood_conditional_alignment={f:green[f]/ood_n if ood_n else None for f in green},SR_percent=sr['SR_percent'],sr_evidence=sr['sr_evidence'])
   for f in green:assert np.isclose(s['TASR'][f],s['ood_budget_share']*s['ood_conditional_alignment'][f])
   summaries[method]=s;flat.append(s)
  a['summary']=summaries
  a['correlations']={f:{'n':len(summaries),'pearson':float(pearsonr([s['TASR'][f] for s in summaries.values()],[s['SR_percent'] for s in summaries.values()]).statistic),
   'spearman':float(spearmanr([s['TASR'][f] for s in summaries.values()],[s['SR_percent'] for s in summaries.values()]).statistic)} for f in ['1','2','3']}
  results[task]=a;(OUT/f'tasr_{task}.json').write_text(json.dumps(a,indent=2,allow_nan=False))
 (OUT/'summary.json').write_text(json.dumps(dict(status='DOMAIN_CORRECTION_COMPLETE_STAGE_PROXY_RETAINED',rows=flat,correlations={t:a['correlations'] for t,a in results.items()}),indent=2))
 # Independent tally from original point labels, not cached compatible counts.
 audit=[]
 for task,a in results.items():
  original=json.loads((SOURCE/f'tasr_{task}.json').read_text())
  for method,s in a['summary'].items():
   rows=[r for r in original['rows'] if r['selected'] and r['method']==method]
   for f in ['1','2','3']:
    count=sum(sum(label=='target_compatible' for label in r['labels'][f]) for r in rows if r['split']=='ood')
    assert count==s['compatible_points'][f]
   audit.append(dict(task=task,method=method,denominator=s['total_expert_points'],numerator_x3=s['compatible_points']['3'],status='PASS'))
 (OUT/'audit.json').write_text(json.dumps(dict(status='PASS',groups=audit,total_selected_points=sum(s['total_expert_points'] for s in flat),id_green_after_correction=0),indent=2))
 lines=['# OOD gap TASR：分子域条件纠正','','公式：OOD轨迹中的兼容目标点 / 全部新增专家轨迹点。ID贡献分子为0，分母保留。固定原有1/2/3倍半径，未再次拟合。','',
  '| Task | Method | SR | ID点/总点 | 旧TASR×3 | 移除ID绿色点 | 新TASR×1 | ×2 | ×3 |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
 for s in flat:lines.append(f"| {s['task']} | {s['method']} | {s['SR_percent']} | {s['id_points']}/{s['total_expert_points']} | {s['old_TASR']['3']:.4f} | {s['removed_id_compatible_points']['3']} | "+' | '.join(f"{s['TASR'][f]:.4f}" for f in ['1','2','3'])+' |')
 lines+=['','## 目标片段核对','','- 旧StackCube：接近红方块、对齐并抓住；不包括搬运至绿方块及释放。因此本结果仅是该旧配置的抓取监督指标，不能代表完整任务所有缺口。',
  '- Grab Plane：在OOD姿态下对齐飞机、闭合至4步稳定抓取；不含旧oracle后续闭合等待、抬起及搬运。',
  '- YCB Object：对OOD物体对齐、闭合至4步稳定抓取；不含抬起及搬运。BC训练子集全是OOD，域条件纠正不改变BC。',
  '- ID排除由原始split元数据决定。OOD归属与目标阶段匹配均满足才计绿；OOD的非目标段仍计分母。',
  '- 这些阶段是明确的任务相关候选gap，还没有通过基础policy独立验证“阶段内每个状态确实需要学习”。本次不利用下游SR选择边界；若需进一步缩小gap，需要独立标注或base-policy失败阶段证据。','',
  '## 与SR对照','']
 for t,a in results.items():lines.append(t+': '+', '.join(f"×{f} Spearman={v['spearman']:.3f}, Pearson={v['pearson']:.3f}" for f,v in a['correlations'].items()))
 lines+=['','每task仅4方法，事后探索，保留排序反例。原数值计算可复现，但旧版将ID几何相容点算进OOD有效分子，语义错误；论文应以本修正版替换旧TASR。','',
  '实现上的等式：TASR = 新专家预算中的OOD占比 × OOD内目标相容点占比。后者只作解释分解，不作为替代主指标。','']
 (OUT/'结果与目标片段核对.md').write_text('\n'.join(lines))
 # Keep the user's original two column conventions available.
 for filename in ['active_learning_results_filled.tex','active_learning_all_actual_gates.tex']:
  tex=(SOURCE/filename).read_text()
  replacements={f"{s['old_TASR']['3']:.4f}":f"{s['TASR']['3']:.4f}" for s in flat}
  tex=re.sub(r'\b0\.\d{4}\b',lambda m:replacements.get(m.group(),m.group()),tex)
  tex=tex.replace('TASR measures the target alignment of collected expert supervision.',
   'TASR is the fraction of all collected expert trajectory points that originate from OOD episodes and align with the designated OOD target segment.')
  tex=tex.replace('SR is in percent;', 'ID expert points contribute only to the TASR denominator. SR is in percent;')
  (OUT/filename).write_text(tex)
 print('\n'.join(lines))

if __name__=='__main__':main()
