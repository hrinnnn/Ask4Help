#!/usr/bin/env python3
"""Post-hoc sensitivity only: fixed matching, phases, contacts and data budgets."""
import argparse,copy,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FACTORS=[1.,1.25,1.5,2.,3.,4.]
ANCHORS=[0,50,80,120,160,220]


def name(factor):return f'factor_{factor:.2f}'.replace('.','p')


def classify(row,contact,factor):
    labels=[]
    for i,d in enumerate(row['distance']):
        if d is None:labels.append('non_target');continue
        t=row['takeover']+i
        phase=next(p for p,(lo,hi) in row['blocks'].items() if lo<=t<hi)
        labels.append('target_compatible' if contact[i] and d<=row['thresholds'][phase]*factor else 'target_mismatch')
    return labels


def aggregate(rows):
    n=sum(r['expert_anchors'] for r in rows);green=sum(r['compatible_anchors'] for r in rows)
    return dict(episodes=len(rows),expert_anchors=n,compatible_anchors=green,Q=green/n if n else None,
                target_stage_anchors=sum(r['target_stage_anchors'] for r in rows))


def main(source,output):
    output.mkdir(parents=True,exist_ok=True)
    original=json.loads((source/'analysis.json').read_text())
    snapshot=output/'baseline_analysis.json'
    if snapshot.exists():assert json.loads(snapshot.read_text())==original,'Source changed; use a new diagnostic root.'
    else:snapshot.write_text(json.dumps(original,indent=2,allow_nan=False))
    lookup={r['seed']:r for r in original['rows']}
    contacts={}
    def raw(seed):
        if seed not in contacts:
            timeline=json.loads((Path(lookup[seed]['directory'])/'task_state_timeline.json').read_text())['rows']
            contacts[seed]=np.array([r['object_grasped'] for r in timeline],dtype=bool)
        return contacts[seed]
    compatibility={}
    for row in original['rows']:
        q=raw(row['seed']);r=raw(row['reference_seed'])
        compatibility[row['seed']]=[None if j<0 else bool(q[row['takeover']+i]==r[j]) for i,j in enumerate(row['reference_mapping'])]
        assert classify(row,compatibility[row['seed']],1.)==row['labels']

    selected={a:[r for r in original['rows'] if r['anchor']==a and r['accepted'] and r['accepted_index'] in original['budget_manifest']['selected_source_episode_indices'][f'anchor_{a}']] for a in ANCHORS}
    all_eligible={}
    upper={}
    for a,rows in selected.items():
        assert sum(r['expert_anchors'] for r in rows)==2413
        ratios=[];blocked=0
        for r in rows:
            for i,d in enumerate(r['distance']):
                if d is None:continue
                if not compatibility[r['seed']][i]:blocked+=1;continue
                t=r['takeover']+i;phase=next(p for p,(lo,hi) in r['blocks'].items() if lo<=t<hi)
                ratios.append(d/r['thresholds'][phase])
        all_eligible[a]=np.sort(ratios)
        upper[str(a)]=dict(Q=len(ratios)/2413,eligible=len(ratios),contact_blocked=blocked,target_stage=sum(r['target_stage_anchors'] for r in rows))

    summaries=[]
    for factor in FACTORS:
        variant=copy.deepcopy(original)
        variant['protocol']['baseline_analysis']=str((source/'analysis.json').resolve())
        variant['protocol']['sensitivity_status']='POST_HOC_SENSITIVITY_AFTER_20_EPISODE_SR; no formal threshold selected'
        variant['protocol']['radius_multiplier']=factor
        variant['protocol']['baseline_thresholds']=dict(original['protocol']['thresholds'])
        variant['protocol']['effective_minimum_radius']=original['protocol']['minimum_radius']*factor
        variant['protocol']['thresholds']={p:t*factor for p,t in original['protocol']['thresholds'].items()}
        for r,base in zip(variant['rows'],original['rows']):
            r['labels']=classify(base,compatibility[r['seed']],factor)
            r['compatible_anchors']=r['labels'].count('target_compatible');n=r['expert_anchors'];r['Q']=r['compatible_anchors']/n if n else None
            r['baseline_thresholds']=dict(base['thresholds']);r['thresholds']={p:t*factor for p,t in base['thresholds'].items()}
            for phase,(lo,hi) in r['blocks'].items():
                subset=r['labels'][lo-r['takeover']:hi-r['takeover']]
                r['phase_stats'][phase]=dict(target=len(subset),compatible=subset.count('target_compatible'))
        for a in ANCHORS:
            rows=[r for r in variant['rows'] if r['anchor']==a]
            chosen=[r for r in rows if r['accepted'] and r['accepted_index'] in original['budget_manifest']['selected_source_episode_indices'][f'anchor_{a}']]
            variant['summary'][str(a)]=dict(training_budget=aggregate(chosen),all_accepted=aggregate([r for r in rows if r['accepted']]),incomplete=aggregate([r for r in rows if not r['accepted']]))
        checks=[r for r in variant['rows'] if r['seed'] in original['protocol']['check_seeds']]
        check={p:dict(total=sum(r['phase_stats'][p]['target'] for r in checks),compatible=sum(r['phase_stats'][p]['compatible'] for r in checks)) for p in original['reference_checks']}
        variant['reference_checks']=check
        folder=output/name(factor);folder.mkdir(exist_ok=True)
        (folder/'analysis.json').write_text(json.dumps(variant,indent=2,allow_nan=False))
        summary=dict(factor=factor,thresholds=variant['protocol']['thresholds'],groups={str(a):variant['summary'][str(a)]['training_budget'] for a in ANCHORS},reference_checks=check,
                     normal_target_retention=sum(v['compatible'] for v in check.values())/sum(v['total'] for v in check.values()))
        summaries.append(summary);print(json.dumps(summary),flush=True)

    crossings={}
    for a in [120,160,220]:
        points=np.unique(np.r_[1.,all_eligible[0],all_eligible[a]])
        points=np.nextafter(points[points>=1.],np.inf)
        found=next((f for f in points if np.searchsorted(all_eligible[a],f,side='right')>np.searchsorted(all_eligible[0],f,side='right')),None)
        if found is not None:
            crossings[str(a)]=dict(multiplier=float(found),Q0=int(np.searchsorted(all_eligible[0],found,side='right'))/2413,Qlate=int(np.searchsorted(all_eligible[a],found,side='right'))/2413,
                                   interpretation='post-hoc discrete crossover, not a recommended operating threshold')
        else:crossings[str(a)]=None
    report=dict(status='POST_HOC_SENSITIVITY_ONLY',source_analysis=str((source/'analysis.json').resolve()),factors=FACTORS,summary=summaries,
                fixed=['reference selection','per-point matching','target phases','contact compatibility','datasets and budgets'],
                changed='one common multiplicative tolerance radius across all phases and all conditions',upper_bound_without_distance_limit=upper,crossover_vs_t0=crossings,
                caution='Known probe SR and desired ordering were available before this sweep; this is not independent predictive validation.')
    (output/'sensitivity.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for a in ANCHORS:
        axes[0].plot(FACTORS,[s['groups'][str(a)]['Q'] for s in summaries],marker='o',label=f't={a}')
    axes[0].set(xlabel='Common radius multiplier',ylabel='Trajectory overlap Q',title='Fixed data, matching and contacts');axes[0].legend(ncol=2)
    for s in summaries:
        axes[1].plot(ANCHORS,[s['groups'][str(a)]['Q'] for a in ANCHORS],marker='o',label=f"x{s['factor']:g}")
    axes[1].set(xlabel='Takeover step',ylabel='Trajectory overlap Q',title='Post-hoc sensitivity, not SR validation');axes[1].legend(ncol=2)
    fig.savefig(output/'threshold_sensitivity.png',dpi=170);plt.close(fig)
    lines=['# OpenDrawer 重合容忍半径敏感性','',
           '**看到20条probe结果后的探索；不选择或覆盖正式阈值。** 这里只改匹配容忍度，不是独立Recovery分类或PCA报警阈值。','',
           '| 统一倍数 | t0 | t50 | t80 | t120 | t160 | t220 | 正常目标点保留 |','|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in summaries:lines.append('| '+f"{s['factor']:g}"+' | '+' | '.join(f"{s['groups'][str(a)]['Q']:.4f}" for a in ANCHORS)+f" | {s['normal_target_retention']:.4f} |")
    lines+=['','每格分母保持2413。目标片段、参考、对应关系和接触条件不变，非目标灰色不可能被本次放宽变绿。',
            '原来不匹配的接触状态仍不匹配，不能靠距离放宽改变夹持事实。','',
            '## 后期条件首次超过t0的位置（事后诊断，不用于推荐）','']
    for a,v in crossings.items():lines.append(f"- t{a}: multiplier={v['multiplier']:.6f}; Q0={v['Q0']:.6f}, Qlate={v['Qlate']:.6f}" if v else f'- t{a}: 无交叉')
    lines+=['','## 完全取消距离限制的上界','',
            '这不是推荐设置。此时指标退化为目标阶段及接触相容的时间占比，无法再区分该阶段中的大幅恢复路径。','']
    for a,v in upper.items():lines.append(f"- t{a}: {v['eligible']}/2413 = {v['Q']:.4f}; contact-blocked={v['contact_blocked']}")
    lines+=['','## 统计边界','',
            '正常检查来自原6条成功专家（已在此前调试中查看），不是新的独立测试集。','看到已知SR后挑一个能排出预期顺序的倍数不能作为指标有效性证明；如选择新版本，应另存并用未用于选择的结果评估。',
            '本分析没有计算或优化与probe SR的相关系数，也没有修改任何训练或评测控制器。','']
    (output/'README.md').write_text('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.source,a.output)
