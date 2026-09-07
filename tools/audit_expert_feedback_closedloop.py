"""Independent artifact/chronology reconciliation for the small feedback pilot.

Does not import the gate or collector implementation. Never reports post-SFT SR.
"""
import json
import argparse
from pathlib import Path
import subprocess
from collections import Counter
import numpy as np

ROOT=Path('artifacts/expert_feedback_pca_20260907')

def main(args):
    root=args.root
    cal=json.loads((ROOT/'free_action_v1/gate_calibration.json').read_text())
    center=np.array(cal['center']);pca=np.load(ROOT/'original_pca_bridge.npz')
    summary={};allrows={};checks=Counter();details=[]
    for arm in args.arms:
        source=json.loads((root/arm/'summary.json').read_text());assert source['episodes']==20
        assert (root/arm/'PILOT_COMPLETE.json').exists()
        memory=[];committed=0;cost=0;rows=[]
        assert len(list((root/arm).glob('episode_*/result.json')))==20
        for e in range(20):
            directory=root/arm/f'episode_{e:03d}';r=json.loads((directory/'result.json').read_text())
            a=np.load(directory/'trace.npz');n=len(a['actions']);x=a['policy_query_features'];scores=a['policy_query_scores']
            assert r==source['rows'][e] and r['episode']==e and r['arm']==arm
            assert r['seed']==(930700 if e%2==0 else 940700)+e//2+args.seed_offset
            assert r['split']==('id' if e%2==0 else 'stage2_ood')
            assert a['qpos'].shape[0]==a['tcp'].shape[0]==n+1
            assert a['actions'].shape==(n,8) and np.isfinite(a['actions']).all()
            assert len(x)==len(scores)==len(r['queries'])
            assert np.allclose(np.linalg.norm((x-pca['mean'])@pca['eigenvectors'][:,:512],axis=1),scores,rtol=1e-5,atol=1e-5)
            cost+=r['expert_actions'];assert cost==r['cumulative_expert_actions']
            assert n==r['total_actions']==r['policy_steps']+r['expert_actions']
            z=(x-center)/cal['scale'];stops=[]
            for i,q in enumerate(r['queries']):
                assert q['step']==i*5 and q['score']==scores[i] and q['memory_episodes']==committed
                neighbors=[m for m in memory if np.linalg.norm(z[i]-m['z'])<=cal['radius']]
                wait=[m['score'] for m in neighbors if not m['request']]
                request=[m['score'] for m in neighbors if m['request']]
                lo=max(wait,default=-np.inf);hi=np.nextafter(min(request),-np.inf) if request else np.inf
                reason='no_support' if not neighbors else ('supported' if lo<=hi else 'conflict')
                threshold=float(np.clip(cal['tau0'],lo,hi)) if reason=='supported' else cal['tau0']
                assert q['reason']==reason and q['neighbors']==len(neighbors)
                assert q['threshold']==threshold and q['stop']==(scores[i]>threshold)
                assert q['baseline_stop']==(scores[i]>cal['tau0'])
                stops.extend([q['step']] if q['stop'] else [])
                checks['actual_policy_queries']+=1
            assert stops==([] if r['takeover'] is None else [r['takeover']])
            tg=r['takeover'];u=None
            if tg is not None and tg>=5 and tg+5<n+1:
                p=a['tcp'];w=a['qpos'][:,-2:].sum(1);before=p[tg]-p[tg-5];after=p[tg+5]-p[tg];norm=np.linalg.norm(before)
                back=max(0.,-float(after@before)/norm) if norm>1e-8 else 0.
                wb=w[tg]-w[tg-5];wa=w[tg+5]-w[tg];bg=max(0.,-np.sign(wb)*wa) if abs(wb)>1e-8 else 0.
                u=float(np.hypot(back/.02,bg/.01));assert np.isclose(u,r['motion_undo'])
            else:assert r['motion_undo'] is None
            if tg is not None and r['expert_actions']>=10:
                loss=a['expert_feedback_loss'];assert len(loss)==r['expert_actions']
                full=len(loss)-9;finite=np.flatnonzero(np.isfinite(loss))
                assert np.array_equal(finite,np.arange(len(finite))) and len(finite)<=full
                high=np.flatnonzero(loss[finite]>cal['q_loss']);event=int(high[0]) if len(high) else None
                assert len(finite)==(event+1 if event is not None else full)
                if arm!='fixed':
                    f=r['feedback'];assert f['event']==event
                    corrective=arm=='adaptive_prefix' and u is not None and u>1.9004588285680322
                    index=None;request=None
                    if corrective:index=max(0,len(x)-2);request=True
                    elif event is not None:index=len(x)-1;request=event<5
                    if index is not None:
                        assert f['credit_step']==r['queries'][index]['step'] and f['request']==request
                        memory.append(dict(z=z[index],score=scores[index],request=request,episode=e))
                    else:assert f['credit_step'] is None
                    committed+=1
                else:assert r['feedback']['first_observed_high']==event
                checks['feedback_episodes']+=1
            assert r['memory_size']==len(memory)
            video=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=width,height,nb_read_frames','-of','json',str(directory/'video.mp4')]))['streams'][0]
            assert (video['width'],video['height'],int(video['nb_read_frames']))==(768,384,n)
            checks['episode_artifacts_and_videos']+=1;rows.append(r)
        assert cost==source['expert_actions'];allrows[arm]=rows
        summary[arm]={}
        for split in ['id','stage2_ood']:
            sub=[r for r in rows if r['split']==split];q=[q for r in sub for q in r['queries']];times=[r['takeover'] for r in sub if r['takeover'] is not None]
            summary[arm][split]=dict(episodes=len(sub),completed_with_optional_expert=sum(r['success'] for r in sub),
                unassisted_completions=sum(r['success'] and r['takeover'] is None for r in sub),takeovers=len(times),
                takeover_times=[r['takeover'] for r in sub],median_takeover=float(np.median(times)) if times else None,
                expert_actions=sum(r['expert_actions'] for r in sub),policy_queries=len(q),supported_queries=sum(v['reason']=='supported' for v in q),
                adjusted_threshold_queries=sum(v['threshold']!=cal['tau0'] for v in q),decision_flips=sum(v['stop']!=v['baseline_stop'] for v in q),
                undo_flags=sum(r['motion_undo'] is not None and r['motion_undo']>1.9004588285680322 for r in sub),
                feedback_reasons=dict(Counter(r['feedback']['reason'] for r in sub if r['feedback'])))
        summary[arm]['total_expert_actions']=cost
    for e in range(20):
        baseline=allrows['fixed'][e]
        for arm in [v for v in args.arms if v!='fixed']:
            r=allrows[arm][e];assert r['seed']==baseline['seed'] and r['reset_metadata']==baseline['reset_metadata']
            if e<2:assert r['queries']==baseline['queries'] and r['takeover']==baseline['takeover']
            details.append(dict(arm=arm,episode=e,split=r['split'],fixed_takeover=baseline['takeover'],adaptive_takeover=r['takeover'],
                                fixed_cost=baseline['expert_actions'],adaptive_cost=r['expert_actions']))
            checks['paired_seed_geometry']+=1
    report=dict(status='INDEPENDENT_COLLECTION_AUDIT_PASS',checks=dict(checks),summary=summary,paired=details,
                limitations=['One ordered stream,10 ID+10 OOD scenes per arm; no independent stream seeds.',
                             'Successful completion after an expert intervenes is assisted, not downstream policy SR.',
                             'Fixed episode count; expert costs are measured, not matched.',
                             'ID false requests cannot all be called false alarms without an unassisted reference.',
                             'NaNs after the first high feedback window denote intentionally uncomputed windows.'])
    (root/'independent_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report['summary'],indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT/'closedloop/pilot384');p.add_argument('--seed-offset',type=int,default=0)
    p.add_argument('--arms',nargs='+',default=['fixed','adaptive_current','adaptive_prefix']);main(p.parse_args())
