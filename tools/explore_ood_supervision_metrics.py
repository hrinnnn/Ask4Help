"""Post-hoc mechanistic ablations with transparent leave-one-cohort diagnostics."""
import itertools,json
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr,spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('artifacts/ood_supervision_exploration_20260906')
BASE=Path('artifacts/pi05_table_tasr_20260903')
FACTORS=[1,3,6]

def main():
 ROOT.mkdir(exist_ok=True);utility=json.loads((BASE/'pi05_tasr_table_values.json').read_text())['rows'];groups={};raw={}
 for task in ['stackcube','airplane','object']:
  a=json.loads((BASE/f'tasr_{task}.json').read_text());groups[task]=[]
  for method,s in a['summary'].items():
   rows=[r for r in a['rows'] if r['method']==method and r['selected']];ood=[r for r in rows if r['split']=='ood']
   n=sum(r['expert_points'] for r in rows);o=sum(r['expert_points'] for r in ood);ratios=[];contacts=[];target=0
   for r in ood:
    for phase,(lo,hi) in r['blocks'].items():
     ratios.extend(r['distance'][i]/r['thresholds'][phase] for i in range(lo,hi));contacts.extend(r['contact_compatible'][i] for i in range(lo,hi));target+=hi-lo
   sr=next(u['SR_percent']/100 for u in utility if u['task']==task and u['method']==method)
   row=dict(method=method,N=n,O=o,target_points=target,unscored_outside_target=o-target,P=o/n,SR=sr,episodes=len(rows),ood_episodes=len(ood))
   groups[task].append(row);raw[task,method]=(np.array(ratios),np.array(contacts,dtype=bool))
 # OpenDrawer fixed-timing group: every formal collection is from grasp OOD.
 drawer=json.loads(Path('artifacts/open_drawer_tolerance_sweep_20260903/factor_1p00/analysis.json').read_text());groups['opendrawer']=[]
 lookup={r['seed']:r for r in drawer['rows']};contact_cache={}
 def contact(seed):
  if seed not in contact_cache:
   rs=json.loads((Path(lookup[seed]['directory'])/'task_state_timeline.json').read_text())['rows'];contact_cache[seed]=np.array([r['object_grasped'] for r in rs],dtype=bool)
  return contact_cache[seed]
 srmap={0:.25,50:.75,80:.8,120:.35,160:.4,220:.65}
 for step,sr in srmap.items():
  indices=drawer['budget_manifest']['selected_source_episode_indices'][f'anchor_{step}'];rows=[r for r in drawer['rows'] if r['anchor']==step and r['accepted'] and r['accepted_index'] in indices]
  n=sum(r['expert_anchors'] for r in rows);assert n==2413;ratios=[];contacts=[]
  for r in rows:
   for phase,(lo,hi) in r['blocks'].items():
    for t in range(lo,hi):
     i=t-r['takeover'];ratios.append(r['distance'][i]/r['thresholds'][phase]);contacts.append(bool(contact(r['seed'])[t]==contact(r['reference_seed'])[r['reference_mapping'][i]]))
  method=f't{step}';groups['opendrawer'].append(dict(method=method,N=n,O=n,P=1.,target_points=len(ratios),unscored_outside_target=n-len(ratios),SR=sr,episodes=len(rows),ood_episodes=len(rows)))
  raw['opendrawer',method]=(np.array(ratios),np.array(contacts,dtype=bool))
 def score(row,ratio,contact,f,b,alpha,kernel):
  weights=(ratio<=f).astype(float) if kernel=='hard' else np.exp(-.5*(ratio/f)**2)
  g=float(np.sum(weights*contact));return row['P']**alpha*(g+b*row['unscored_outside_target'])/row['O']
 def corr(x,y):
  if np.ptp(x)<1e-12:return dict(pearson=None,spearman=None,ranking_score=0.,reason='constant metric; no ranking information')
  rho=float(spearmanr(x,y).statistic);return dict(pearson=float(pearsonr(x,y).statistic),spearman=rho,ranking_score=rho)
 trials=[]
 for f,b,alpha,kernel in itertools.product(FACTORS,[0.,.5,1.],[1.,2.,4.],['hard','soft']):
  row=dict(factor=f,outside_credit=b,selection_exponent=alpha,kernel=kernel,cohorts={},values={})
  for task,items in groups.items():
   values=[score(item,*raw[task,item['method']],f,b,alpha,kernel) for item in items]
   row['values'][task]={item['method']:v for item,v in zip(items,values)};row['cohorts'][task]=corr(values,[r['SR'] for r in items])
  row['mean_rank_score']=float(np.mean([v['ranking_score'] for v in row['cohorts'].values()]));trials.append(row)
 # Ordering above resolves ties without consulting the held-out cohort.
 loto={}
 for held in groups:
  best=max(trials,key=lambda t:np.mean([t['cohorts'][g]['ranking_score'] for g in groups if g!=held]))
  loto[held]=dict(selected={k:best[k] for k in ['factor','outside_credit','selection_exponent','kernel']},training_mean_rank_score=float(np.mean([best['cohorts'][g]['ranking_score'] for g in groups if g!=held])),held_out_result=best['cohorts'][held],values=best['values'][held])
 named={}
 for name,config in [('ood_stage_x3',(3,0.,1.,'hard')),('broad_ood_minus_mismatch_x3',(3,1.,1.,'hard')),('broad_ood_minus_large_mismatch_x6',(6,1.,1.,'hard')),('selection_weighted_stage',(3,0.,2.,'hard')),('soft_stage',(3,0.,1.,'soft'))]:
  named[name]=next(t for t in trials if (t['factor'],t['outside_credit'],t['selection_exponent'],t['kernel'])==config)
 purity={g:dict(values={r['method']:r['P'] for r in rows},correlation=corr([r['P'] for r in rows],[r['SR'] for r in rows])) for g,rows in groups.items()}
 best=max(trials,key=lambda t:t['mean_rank_score'])
 result=dict(status='POST_HOC_EXPLORATION_NOT_PROSPECTIVE_VALIDATION',formula='P^alpha * (weighted_compatible_target_points + beta * unscored_outside_target_points) / OOD_points',
  caution='beta>0 credits unmeasured outside-target states optimistically; mismatch is not a validated recovery label. Selection exponent changes the interpretation away from a literal point fraction.',
  groups=groups,named=named,purity=purity,leave_one_cohort_out=loto,best_in_sample=best,trials=trials)
 (ROOT/'exploration.json').write_text(json.dumps(result,indent=2,allow_nan=False))
 lines=['# OOD监督选择与轨迹质量探索','','保留OOD-only分子规则与全部新增专家点分母。下面所有参数探索均在已知SR后进行；留一任务仅为交叉检验，任务曾用于设计讨论，非全新盲测。','',
  'P=OOD新增专家点/全部新增专家点；G=目标片段相容点；U=未被旧指标评分的OOD后续点。探索式 Q=P^alpha (G+beta U)/O。beta=1是假定后续点可用的乐观消融；不能称这些点已被验证不是Recovery。soft版本把硬阈值改为连续高斯权重，接触约束保留。','',
  '## 数据构成','', '| task | method | SR | OOD预算占比P | 目标片段/OOD | 后续未评分/OOD |','|---|---|---:|---:|---:|---:|']
 for task,rs in groups.items():
  for r in rs:lines.append(f"| {task} | {r['method']} | {r['SR']:.2f} | {r['P']:.4f} | {r['target_points']/r['O']:.4f} | {r['unscored_outside_target']/r['O']:.4f} |")
 lines+=['','## 命名候选的排序相关性','','| variant | SC | Plane | YCB | Drawer |','|---|---:|---:|---:|---:|']
 for name,t in named.items():lines.append('| '+name+' | '+' | '.join(f"{t['cohorts'][g]['ranking_score']:.3f}" for g in groups)+' |')
 lines+=['| OOD预算占比P | '+' | '.join('无排序信息' if purity[g]['correlation']['spearman'] is None else f"{purity[g]['correlation']['spearman']:.3f}" for g in groups)+' |','',
  '## 54个组合与留一任务检验','',f"全数据最优（事后描述）：{ {k:best[k] for k in ['factor','outside_credit','selection_exponent','kernel','mean_rank_score']} }",'']
 for held,v in loto.items():lines.append(f"- 留出{held}: 使用{v['selected']}，训练均值={v['training_mean_rank_score']:.3f}，留出Spearman={v['held_out_result']['spearman']}。")
 lines+=['','## 相机与坐标','',
  '- 现有TASR读取qpos正运动学、物体位置、接触及稳定抓取TCP轴，没有读取RGB或camera extrinsics；直接改变相机位置不会改变本分数。相机可能影响训练SR，要单独核对模型实际观察。',
  '- 物体中心与本条稳定抓取TCP轴归一化会去掉一部分绝对位置和朝向差异；对Object/Pose OOD，“相同夹爪形状”不等于覆盖相同观察状态。几何指标对视觉与接触动力学差异并不充分。',
  '- YCB任务合同只有object_model_id变化，camera与初始化分布固定；本轮不能从SR反推相机设计错误。BC只收OOD，Diff含大量ID，Recovery几乎全OOD；这三者需要分别解释选择与恢复开销。',
  '- OpenDrawer原分数是同一OOD条件内的timing比较，OOD占比为1；另三组同时混有ID/OOD选择和gate方法区别。用选择项改善混合流任务，不应要求它解释Drawer固定时刻差别。','']
 (ROOT/'调研与拟合结果.md').write_text('\n'.join(lines));print('\n'.join(lines))
 fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
 for ax,(task,rs) in zip(axes.flat,groups.items()):
  methods=[r['method'] for r in rs];x=np.arange(len(rs));ax.bar(x,[r['P'] for r in rs],color='#b1cfe7',label='OOD budget share')
  ax.bar(x,[named['ood_stage_x3']['values'][task][m] for m in methods],color='#26916c',label='Current target-compatible share')
  ax.plot(x,[r['SR'] for r in rs],'o-',color='#bc563b',label='Downstream SR');ax.set_xticks(x,methods,rotation=25,ha='right');ax.set(title=task,ylim=(0,1.08));ax.legend(fontsize=8)
 fig.savefig(ROOT/'selection_target_sr.png',dpi=150);plt.close(fig)

if __name__=='__main__':main()
