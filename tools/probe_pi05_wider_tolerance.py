"""Post-hoc radius sensitivity, including the contact-compatible upper bound."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr,spearmanr

root=Path('artifacts/pi05_table_tasr_20260903')
out=Path('artifacts/pi05_wider_tolerance_20260906');out.mkdir(exist_ok=True)
utilities=json.loads((root/'pi05_tasr_table_values.json').read_text())['rows']
result={}
for task in ['stackcube','airplane','object']:
 a=json.loads((root/f'tasr_{task}.json').read_text());groups={}
 for method,summary in a['summary'].items():
  rows=[r for r in a['rows'] if r['method']==method and r['selected']]
  ratios=[]
  for r in rows:
   for phase,(lo,hi) in r['blocks'].items():
    ratios.extend(r['distance'][i]/r['thresholds'][phase] for i in range(lo,hi) if r['contact_compatible'][i])
  ratios=np.array(ratios);den=summary['total_expert_points']
  assert sum(r['expert_points'] for r in rows)==den
  values={str(f):int(np.sum(ratios<=f))/den for f in [1,2,3,4,6,10]}
  values['unlimited']=len(ratios)/den
  assert abs(values['3']-summary['TASR']['3'])<1e-12
  sr=next(r['SR_percent'] for r in utilities if r['task']==task and r['method']==method)/100
  groups[method]={'SR':sr,'TASR':values}
 correlations={}
 for f in ['1','2','3','4','6','10','unlimited']:
  x=[v['TASR'][f] for v in groups.values()];y=[v['SR'] for v in groups.values()]
  correlations[f]={'pearson':float(pearsonr(x,y).statistic),'spearman':float(spearmanr(x,y).statistic)}
 result[task]={'groups':groups,'correlations':correlations}
(out/'results.json').write_text(json.dumps(result,indent=2))
lines=['# 更宽匹配半径的事后敏感性分析','','固定原轨迹、参考、对应关系、接触约束、目标片段和分母，仅改变半径倍数。unlimited完全取消几何距离限制，仍保留目标阶段与接触相容。已知SR后的探索，不选择正式最优参数。','']
for task,r in result.items():
 lines+=['## '+task,'','| 方法 | SR | ×1 | ×3 | ×4 | ×6 | ×10 | unlimited |','|---|---:|---:|---:|---:|---:|---:|---:|']
 for method,v in r['groups'].items():lines.append('| '+method+f" | {v['SR']:.2f} | "+' | '.join(f"{v['TASR'][f]:.4f}" for f in ['1','3','4','6','10','unlimited'])+' |')
 lines+=['','Spearman: '+', '.join(f"{f}: {v['spearman']:.3f}" for f,v in r['correlations'].items()),'']
(out/'报告.md').write_text('\n'.join(lines))
print('\n'.join(lines))
