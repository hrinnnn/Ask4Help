#!/usr/bin/env python3
"""Offline target-anchor ratio: object-oriented rotation/reach/close, no lift.

Task phases come from the existing Oracle stage logs; pose/contact compatibility
is calibrated on nominal experts only. No training, new rollout or SR tuning.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analyze_open_drawer_piecewise_ed import read_jsonl,load_episode
from analyze_open_drawer_suffix_recovery import pose_channels,suffix_map

PHASES=['rotation_pregrasp','reach','close']
UNITS=np.array([.02,np.deg2rad(15),.01])
MIN_RADIUS=float(np.linalg.norm(np.array([.001,np.deg2rad(1),.001])/UNITS))


def relative_pose(pose,rows):
    object_p=np.array([r['object_position'] for r in rows])
    object_q=np.array([r['object_quaternion'] for r in rows])
    object_R=Rotation.from_quat(object_q[:,[1,2,3,0]])
    tcp_R=Rotation.from_quat(pose['quaternion'][:,[1,2,3,0]])
    return dict(position=object_R.inv().apply(pose['position']-object_p),
                quaternion=(object_R.inv()*tcp_R).as_quat()[:,[3,0,1,2]],width=pose['width'])


def stage_blocks(meta):
    e=meta.get('expert_result') or {}
    take=int(meta['actual_takeover_step']);n=int(meta['expert_action_steps'])
    if not e.get('drawer_opened_after_takeover',False):return {},None
    handle=0
    if not e['drawer_opened_before_takeover']:
        handle=sum(e[k] for k in ['direct_handle_pregrasp_steps','direct_handle_reach_steps','direct_pull_steps'])
        handle+=4 if e['direct_pull_from_existing_handle_grasp'] else 14
    if e.get('object_grasped_before_takeover',False):return {},take+handle
    # The 2 release/open commands precede the object orientation/pregrasp move.
    cursor=take+handle+2;blocks={}
    for phase,key in zip(PHASES,['direct_object_pregrasp_steps','direct_object_reach_steps','direct_object_close_steps']):
        end=cursor+int(e.get(key,0));blocks[phase]=(min(cursor,take+n),min(end,take+n));cursor=end
    if meta['accepted']:
        assert cursor+e['direct_lift_steps']+e['direct_transport_steps']+e['direct_place_steps']+4==take+n
    return blocks,cursor


def compare_block(query,ref,phase):
    a,b=query['blocks'].get(phase,(0,0));c,d=ref['blocks'][phase]
    if a>=b:return dict(indices=[],reference_indices=[],distance=[],contact=[])
    qp={k:v[a:b] for k,v in query['comparison_pose'].items()};rp={k:v[c:d] for k,v in ref['comparison_pose'].items()}
    residual=pose_channels(qp,rp)
    score=np.linalg.norm(residual/UNITS,axis=2)
    qc=np.array([r['object_grasped'] for r in query['rows'][a:b]])
    rc=np.array([r['object_grasped'] for r in ref['rows'][c:d]])
    mapping=suffix_map(score+1000*(qc[:,None]!=rc[None,:]),(0,len(rc)-1),False)
    return dict(indices=list(range(a,b)),reference_indices=(mapping+c).tolist(),
                distance=score[np.arange(len(mapping)),mapping].tolist(),
                contact=(qc==rc[mapping]).tolist())


def main(root,previous,frame):
    formal=previous/'inputs/formal';meta_root=root/'inputs/formal'
    episodes=[]
    for anchor in [0,50,80,120,160,220]:
        accepted={r['seed']:r['episode_index'] for r in read_jsonl(formal/f'anchor_{anchor}/accepted_experts.jsonl')}
        for meta_file in sorted((meta_root/f'anchor_{anchor}/raw_attempts').glob('*/attempt.json')):
            meta=json.loads(meta_file.read_text());directory=formal/f'anchor_{anchor}/raw_attempts'/meta_file.parent.name
            ep=load_episode(directory,meta);blocks,end=stage_blocks(meta)
            ep['comparison_pose']=ep['pose']
            if frame=='object_relative':
                ep['comparison_pose']=relative_pose(ep['pose'],ep['rows'])
            assert len(ep['rows'])==meta['actual_takeover_step']+meta['expert_action_steps']+1
            ep.update(anchor=anchor,blocks=blocks,close_end=end,accepted_index=accepted.get(ep['seed']))
            if meta['accepted'] and blocks:
                assert end is not None
                assert all(ep['rows'][k]['object_grasped'] for k in range(end-2,end+1))
                assert not ep['rows'][end]['ever_lifted'], (ep['seed'],end)
            episodes.append(ep)
    experts=[e for e in episodes if e['anchor']==0 and e['meta']['accepted']]
    refs=[e for e in experts if e['seed']%5<3]
    cal=[e for e in experts if e['seed']%5==3]
    held=[e for e in experts if e['seed']%5==4]

    matches={}
    def match(e,r,p):
        key=(e['seed'],r['seed'],p)
        if key not in matches:matches[key]=compare_block(e,r,p)
        return matches[key]

    def nearest(e,bank):
        # One coherent reference for the whole target sequence, not switching
        # to a different expert at every point. All candidates share the OOD.
        def cost(r):
            parts=[match(e,r,p) for p in PHASES]
            values=[np.mean(v['distance'])+1000*(1-np.mean(v['contact'])) for v in parts if v['indices']]
            return float(np.mean(values)) if values else 0.
        return min((r for r in bank if r['seed']!=e['seed']),key=cost)

    cache={}
    def calibration(skip=None):
        if skip in cache:return cache[skip]
        bank=[r for r in refs if r['seed']!=skip]
        samples={p:[] for p in PHASES};pairs=[]
        for e in cal:
            if e['seed']==skip:continue
            r=nearest(e,bank);pairs.append([e['seed'],r['seed']])
            for p in PHASES:
                part=match(e,r,p)
                samples[p].extend(v for v,ok in zip(part['distance'],part['contact']) if ok)
        # No OOD test trajectories or utility labels enter this calibration.
        result=dict(bank=bank,samples=samples,pairs=pairs)
        cache[skip]=result;return result

    def score(e,q=.925):
        # When scoring anchor0, exclude that episode from references AND calibration.
        skip=e['seed'] if e['anchor']==0 else None
        calfit=calibration(skip);ref=nearest(e,calfit['bank'])
        thresholds={p:max(MIN_RADIUS,float(np.quantile(calfit['samples'][p],q))) for p in PHASES}
        take=int(e['meta']['actual_takeover_step']);n=int(e['meta']['expert_action_steps'])
        labels=np.full(n,'non_target',dtype=object);mappings=np.full(n,-1,dtype=int);dist=np.full(n,np.nan)
        target_count=0;phase_stats={}
        for p in PHASES:
            part=match(e,ref,p);n_phase=len(part['indices']);target_count+=n_phase
            keep=0
            for t,j,v,contact in zip(part['indices'],part['reference_indices'],part['distance'],part['contact']):
                index=t-take;mappings[index]=j;dist[index]=v
                ok=contact and v<=thresholds[p]
                labels[index]='target_compatible' if ok else 'target_mismatch'
                keep+=int(ok)
            phase_stats[p]=dict(target=n_phase,compatible=keep)
        compatible=int(sum(labels=='target_compatible'))
        return dict(anchor=e['anchor'],seed=e['seed'],accepted=e['meta']['accepted'],success=e['meta']['success'],
            accepted_index=e['accepted_index'],takeover=take,expert_anchors=n,target_stage_anchors=target_count,
            compatible_anchors=compatible,Q=compatible/n if n else None,reference_seed=ref['seed'],
            blocks=e['blocks'],close_end=e['close_end'],thresholds=thresholds,phase_stats=phase_stats,
            labels=labels.tolist(),reference_mapping=mappings.tolist(),
            distance=[float(v) if np.isfinite(v) else None for v in dist],video=e['meta']['video'],directory=e['directory'])

    scored=[score(e) for e in episodes]
    budget=json.loads((root/'inputs/formal_budget/budget_manifest.json').read_text())
    assert budget['common_expert_action_budget']==2413
    def aggregate(rows):
        n=sum(r['expert_anchors'] for r in rows);valid=sum(r['compatible_anchors'] for r in rows)
        return dict(episodes=len(rows),expert_anchors=n,compatible_anchors=valid,
                    Q=valid/n if n else None,target_stage_anchors=sum(r['target_stage_anchors'] for r in rows))
    summary={};sensitivity=[]
    for anchor in [0,50,80,120,160,220]:
        selected=set(budget['selected_source_episode_indices'][f'anchor_{anchor}'])
        rows=[r for r in scored if r['anchor']==anchor]
        training=[r for r in rows if r['accepted'] and r['accepted_index'] in selected]
        main_result=aggregate(training);assert main_result['expert_anchors']==2413
        summary[str(anchor)]=dict(training_budget=main_result,all_accepted=aggregate([r for r in rows if r['accepted']]),
                                 incomplete=aggregate([r for r in rows if not r['accepted']]))
        print('ANCHOR',anchor,main_result,flush=True)
    for q in [.90,.925,.95]:
        rows=scored if q==.925 else [score(e,q) for e in episodes if e['meta']['accepted']]
        sensitivity.append(dict(q=q,values={str(anchor):aggregate([r for r in rows if r['anchor']==anchor and r['accepted'] and r['accepted_index'] in budget['selected_source_episode_indices'][f'anchor_{anchor}']])['Q'] for anchor in [0,50,80,120,160,220]}))
    heldrows=[r for r in scored if r['seed'] in {e['seed'] for e in held}]
    inspection={p:dict(total=sum(r['phase_stats'][p]['target'] for r in heldrows),
                       compatible=sum(r['phase_stats'][p]['compatible'] for r in heldrows)) for p in PHASES}
    fit=calibration()
    result=dict(diagnostic_only=True,protocol=dict(
        target='object-oriented rotation/pregrasp + reach/downstroke + close to 3-step confirmed grasp; EXCLUDES lift/transport/place',
        stage_source='recorded Oracle action counts; 2 handle-release/open commands excluded before object pregrasp',
        denominator='all real expert observation-action anchors in exact-budget selected episodes, including excluded later stages',
        numerator='target-stage anchor whose monotone phase-compatible match is within expert-calibrated pose/gripper tolerance and has same object-contact flag',
        units=dict(position_m=.02,rotation_deg=15,gripper_m=.01),q=.925,comparison_frame=frame,
        reference_selection='single expert minimizing equal-phase mean pose/contact matching cost within same OOD reference bank',
        minimum_radius=MIN_RADIUS,minimum_radius_components=dict(position_m=.001,orientation_deg=1,gripper_m=.001),
        reference_seeds=[e['seed'] for e in refs],calibration_seeds=[e['seed'] for e in cal],check_seeds=[e['seed'] for e in held],
        thresholds={p:max(MIN_RADIUS,float(np.quantile(fit['samples'][p],.925))) for p in PHASES},calibration_pairs=fit['pairs'],
        anchor0_leave_one_out='query excluded from both reference bank and threshold calibration; recompute thresholds',
        caveats=['stage logs, object contact and reset geometry are used; not a proprioception-only deployment metric',
        'current target membership is at observation-action anchor level, not gradient usefulness or action-chunk influence',
        '6 checking experts have been inspected in earlier diagnostics; this is not a pristine final test split',
        'same budget and different reset pools; no downstream SR claim',
        'q is local pointwise compatibility, NOT trajectory false-alarm control']),
        reference_blocks={str(e['seed']):e['blocks'] for e in experts},reference_checks=inspection,
        budget_manifest=budget,summary=summary,sensitivity=sensitivity,rows=scored)
    old=root/'analysis.json'
    if old.exists() and not (root/'analysis_world_frame_v1.json').exists():
        (root/'analysis_world_frame_v1.json').write_text(old.read_text())
    if old.exists() and 'reference_selection' not in json.loads(old.read_text())['protocol']:
        (root/'analysis_object_relative_single_reference_v2.json').write_text(old.read_text())
    old.write_text(json.dumps(result,indent=2,allow_nan=False))
    fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
    x=[0,50,80,120,160,220];y=[summary[str(a)]['training_budget']['Q']*100 for a in x]
    ax.bar(range(6),y,color='#30866c');ax.set_xticks(range(6),[str(a) for a in x])
    for i,value in enumerate(y):ax.text(i,value+.5,f'{value:.1f}%',ha='center')
    ax.set(xlabel='Takeover environment step',ylabel='Target-compatible expert anchors (%)',ylim=(0,max(y)*1.25),
        title='OpenDrawer Grasp-OOD: rotation + grasp only\nEach condition: 2,413 actual expert training anchors; no lift/place in numerator')
    fig.savefig(root/'target_ratio.png',dpi=170);plt.close(fig)
    print('REFERENCE_CHECKS',inspection,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--frame',choices=['world','object_relative'],default='object_relative')
    a=p.parse_args();main(a.root,a.previous,a.frame)
