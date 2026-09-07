"""Aggregate separate experiment stages and common preference diagnostics."""
import argparse,json
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
stages=['development','directional','validation','bounded_development','bounded_validation','longstream'];rows=[];total=0
for stage in stages:
 by=defaultdict(list);folder=args.root/stage
 for path in sorted(folder.glob('seed_*/*/summary.json')):
  r=json.loads(path.read_text())['rows'];total+=len(r)
  for split in ['id','stage2_ood']:by[(path.parent.name,split)].extend(x for x in r if x['split']==split)
 for (variant,split),rs in by.items():
  q=[x for r in rs for x in r['queries']];cost=[r['expert_actions'] for r in rs];times=[r['takeover'] for r in rs if r['takeover'] is not None]
  rows.append(dict(stage=stage,variant=variant,split=split,n=len(rs),assisted_or_unassisted_complete=sum(r['success'] for r in rs),expert_actions=sum(cost),expert_actions_median=float(np.median(cost)),takeovers=len(times),reset_takeovers=sum(t==0 for t in times),median_takeover=float(np.median(times)) if times else None,wait_flips=sum(not x['stop'] and x['baseline_stop'] for x in q),request_flips=sum(x['stop'] and not x['baseline_stop'] for x in q),undo_flags=sum(r['motion_undo'] is not None and r['motion_undo']>1.9004588285680322 for r in rs),feedback=dict(Counter(r['feedback']['reason'] for r in rs if r['feedback']))))
pref=[]
for stage in stages:
 path=args.root/stage/'common_preference_windows.json'
 if not path.exists():continue
 d=json.loads(path.read_text());c=defaultdict(Counter)
 for r in d['rows']:c[r['variant'],r['split']][r['status']]+=1
 for (v,s),counts in c.items():pref.append(dict(stage=stage,variant=v,split=s,counts=dict(counts),n=sum(counts.values())))
report=dict(new_collection_episodes=total,rows=rows,common_preference=pref)
(args.out/'all_experiment_results.json').write_text(json.dumps(report,indent=2))
md='# 夜间实验完整结果\n\n所有表中的完成数包含专家帮助，不能作为SFT后的自主成功率。各阶段使用不同的采集序列；不得只合并有利结果。成本包含失败专家动作；专家动作中位数与总成本同时给出，避免少数150步失败左右均值。\n\n'
for stage in stages:
 md+=f'## {stage}\n\n| Variant | Split | Episodes | Completed | Expert actions | Median cost | Takeovers | Reset takeovers | Wait flips | Request flips | Undo flags |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
 for r in rows:
  if r['stage']==stage:md+='| '+' | '.join(str(r[k]) for k in ['variant','split','n','assisted_or_unassisted_complete','expert_actions','expert_actions_median','takeovers','reset_takeovers','wait_flips','request_flips','undo_flags'])+' |\n'
md+='\n## 同一标准的初始专家行为诊断\n\npreferred_proxy指前5个专家anchor内出现超过共同q_e的自由动作误差，且初始反向纠正未越界；并非独立标注最佳时间。early_cue为至少5个完整anchor均低误差；late_cue为反向纠正高；短段保留删失。\n\n| Stage | Variant | Split | Scored | Preferred proxy | Early cue | Late cue | Short censored |\n|---|---|---|---:|---:|---:|---:|---:|\n'
for r in pref:md+='| '+' | '.join(map(str,[r['stage'],r['variant'],r['split'],r['n'],*[r['counts'].get(k,0) for k in ['preferred_proxy','early_cue','late_cue','censored_short']]]))+' |\n'
(args.out/'完整实验表.md').write_text(md)
print(json.dumps(dict(total=total,aggregate_rows=len(rows),preference_rows=len(pref)),indent=2))
