"""Actual-model feedback feasibility; expert-state replay is not policy evaluation."""
import collections,json,time
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from expert_feedback_pca_core import ExpertFeedbackPCA,first_boundary_feedback

ROOT=Path('artifacts/expert_feedback_pca_20260907')

def main():
 cache=np.load(ROOT/'forward_v1/head_probe_cache.npz');samples=json.loads((ROOT/'forward_v1/samples.json').read_text());x=cache['bridge_features'];loss=cache['native_loss_mc'].mean(1);valid=cache['valid_mask'].sum(1)
 ep=np.array([s['episode_id'] for s in samples]);offset=np.array([s['offset'] for s in samples]);groups=np.array([s['group'] for s in samples]);idcal=groups=='id_calibration';idcheck=groups=='id_check'
 for e in dict.fromkeys(ep):
  ids=np.flatnonzero(ep==e);assert np.array_equal(offset[ids],np.arange(len(ids)));assert np.array_equal(valid[ids],np.minimum(10,len(ids)-np.arange(len(ids))))
 asset=np.load(ROOT/'original_pca_bridge.npz');meta=json.loads((ROOT/'original_pca_bridge.json').read_text());basis=asset['eigenvectors'][:,:len(asset['mean'])-meta['pca_dim']]
 score=np.linalg.norm((x.astype(np.float32)-asset['mean'])@basis,axis=1);tau=meta['threshold']
 q_action=float(np.quantile(cache['native_loss_per_action_mc'][idcal,:,0].mean(1),.95,method='higher'))
 variants={};calibrations={};per_episode=[]
 for calibration_mode in ['point','episode_max']:
  initial=ExpertFeedbackPCA.calibrate(x[idcal],ep[idcal],loss[idcal],valid[idcal],tau,calibration_mode=calibration_mode)
  calibrations[calibration_mode]=dict(q_loss=initial.q_loss,radius=initial.radius,scale=initial.scale,calibration_episodes=len(set(ep[idcal])),calibration_points=int(idcal.sum()))
  for mode in ['document','censored_wait','no_temporal_lookahead']:
   label=calibration_mode+'_'+mode;episodes=[];decisions=[];id_decisions=[]
   for group in dict.fromkeys(groups[~(idcal|idcheck)]):
    gate=ExpertFeedbackPCA(tau,initial.center,initial.scale,initial.radius,initial.q_loss)
    for eid in dict.fromkeys(ep[groups==group]):
     idx=np.flatnonzero(ep==eid);assert eid not in gate.memory_episodes;observed=[]
     for index in idx[offset[idx]%5==0]:
      query=gate.query(x[index],score[index]);query.update(episode=eid,group=group,offset=int(offset[index]),score=float(score[index]),memory_episodes=len(gate.memory_episodes));decisions.append(query);observed.append(query)
     feedback=gate.add_episode(eid,x[idx],score[idx],loss[idx],valid[idx],allow_censored_wait=mode=='censored_wait',lookahead=mode!='no_temporal_lookahead')
     first_old=next((d['offset'] for d in observed if d['baseline_stop']),None);first_new=next((d['offset'] for d in observed if d['stop']),None)
     episodes.append(dict(episode=eid,group=group,length=len(idx),actual_takeover=samples[idx[0]]['actual_takeover'],**feedback,old_first_on_expert_states=first_old,new_first_on_expert_states=first_new))
    for eid in dict.fromkeys(ep[idcheck]):
     idx=np.flatnonzero((ep==eid)&(offset%5==0));qs=[gate.query(x[i],score[i]) for i in idx]
     id_decisions.append(dict(memory_from_group=group,id_episode=eid,baseline_any=any(q['baseline_stop'] for q in qs),new_any=any(q['stop'] for q in qs),changed=sum(q['stop']!=q['baseline_stop'] for q in qs)))
   summary=dict(expert_episodes=len(episodes),queries=len(decisions),event_at_zero=sum(e['event']==0 for e in episodes),event_later=sum(e['event'] is not None and e['event']>0 for e in episodes),no_event=sum(e['event'] is None for e in episodes),
    reason_counts=dict(collections.Counter(d['reason'] for d in decisions)),threshold_increased=sum(d['threshold']>tau for d in decisions),threshold_decreased=sum(d['threshold']<tau for d in decisions),
    newly_requested_queries=sum(d['stop'] and not d['baseline_stop'] for d in decisions),newly_wait_queries=sum(not d['stop'] and d['baseline_stop'] for d in decisions),
    changed_expert_path_first_crossings=sum(e['old_first_on_expert_states']!=e['new_first_on_expert_states'] for e in episodes),
    id_contexts_tested=len(id_decisions),id_baseline_any=sum(d['baseline_any'] for d in id_decisions),id_new_any=sum(d['new_any'] for d in id_decisions))
   variants[label]=dict(summary=summary,episodes=episodes,decisions=decisions,id_checks=id_decisions);print(label,summary,flush=True)
  # Per-episode first supervised boundary, using independent action-zero and MC views.
  for eid in dict.fromkeys(ep):
   idx=np.flatnonzero(ep==eid);f=first_boundary_feedback(loss[idx],valid[idx],q=initial.q_loss);mc=[]
   for k in [0,1]:
    qmc=float(np.quantile(cache['native_loss_mc'][idcal&(valid==10),k],.95,method='higher'))
    mc.append(first_boundary_feedback(cache['native_loss_mc'][idx,k],valid[idx],q=qmc)['event'])
   action_boundary=first_boundary_feedback(cache['native_loss_per_action_mc'][idx,:,0].mean(1),valid[idx],q=q_action)['event']
   if calibration_mode=='point':per_episode.append(dict(episode=eid,group=samples[idx[0]]['group'],length=len(idx),first_chunk_loss_boundary=f['event'],first_action_loss_boundary=action_boundary,
    separate_MC_boundaries=mc,loss=loss[idx].tolist(),pca_score=score[idx].tolist(),valid=valid[idx].tolist()))
 result=dict(status='REAL_MODEL_EXPERT_STATE_FEEDBACK_PROBE_ONLY',source_contract=json.loads((ROOT/'forward_v1/contract.json').read_text()),
  dataset_size=len(samples),calibrations=calibrations,original_pca=meta,variants=variants,episode_signals=per_episode,
  limits=['No autonomous rollout was generated by the new gate.','Only successful archived expert suffixes available; failure feedback not tested.','Expert-state local decisions are a transfer/support diagnostic, not evidence of changed policy timing or SR.','ID references were part of original model training; held-out IDs here are held out only from feedback calibration.'])
 (ROOT/'feedback_probe_results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
 rows=variants['point_document']['episodes'];fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
 for ax,group in zip(axes.flat,dict.fromkeys(e['group'] for e in rows)):
  for e in [x for x in per_episode if x['group']==group]:
   values=np.array(e['loss'])/calibrations['point']['q_loss'];values[np.array(e['valid'])<10]=np.nan
   ax.plot(np.arange(e['length']),values,alpha=.55)
  ax.axhline(1,color='black',linestyle='--');ax.set(title=group,xlabel='Expert action offset',ylabel='Native chunk loss / ID q95')
 fig.savefig(ROOT/'expert_feedback_loss_curves.png',dpi=140);plt.close(fig)
 lines=['# 真实模型专家反馈诊断','','原X-VLA StackCube ckpt7500冻结前向，原Bridge PCA资产；只用ID数据校准，不用OOD参考示范或最佳时刻标签。48段归档专家后缀按组逐episode重放记忆更新，当前episode先查询后写记忆。','',
  '| 版本 | k*=0 | k*>0 | 无事件 | 支持 | 冲突 | 无支持 | 新增request | 新增wait | 首报改变（专家路径） |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for name,v in variants.items():
  s=v['summary'];lines.append('| '+name+' | '+' | '.join(str(z) for z in [s['event_at_zero'],s['event_later'],s['no_event'],s['reason_counts'].get('supported',0),s['reason_counts'].get('conflict',0),s['reason_counts'].get('no_support',0),s['newly_requested_queries'],s['newly_wait_queries'],s['changed_expert_path_first_crossings']])+' |')
 lines+=['','校准：'+str(calibrations),'','这不是动态gate采集后的SR：专家已改变后续状态，表中改变只能证明同一组专家观测上的判定不同，不能用它推算自主继续时会发生什么。','']
 (ROOT/'真实模型反馈诊断.md').write_text('\n'.join(lines))

if __name__=='__main__':main()
