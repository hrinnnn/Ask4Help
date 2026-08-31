#!/usr/bin/env python3
"""Exploratory post-takeover deformation and motion-debt diagnostics.

Never changes gate thresholds. FK is independently checked against saved TCP.
No learned policy or simulator is run. A score requires the expert continuation.
"""
import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import spearmanr, rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analyze_open_drawer_piecewise_ed import read_jsonl, load_episode

POSITION_SCALE=.02
ROTATION_SCALE=np.deg2rad(15)
WIDTH_SCALE=.01
CUTS=[0,20,40,60,80,100,120,140]


def rotations(q):
    return Rotation.from_quat(np.asarray(q)[:,[1,2,3,0]])


class PandaFK:
    def __init__(self,urdf):
        tree=ET.parse(urdf).getroot()
        parents={j.find('child').attrib['link']:j for j in tree.findall('joint')}
        link='panda_hand_tcp';chain=[]
        while link in parents:
            joint=parents[link];chain.append(joint);link=joint.find('parent').attrib['link']
        self.chain=list(reversed(chain))

    def pose(self,states):
        n=len(states);transform=np.broadcast_to(np.eye(4),(n,4,4)).copy()
        for joint in self.chain:
            origin=joint.find('origin');T=np.eye(4)
            if origin is not None:
                T[:3,3]=np.fromstring(origin.attrib.get('xyz','0 0 0'),sep=' ')
                T[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.attrib.get('rpy','0 0 0'),sep=' ')).as_matrix()
            transform=transform@T
            if joint.attrib['type']=='revolute':
                axis=np.fromstring(joint.find('axis').attrib['xyz'],sep=' ')
                index=int(joint.attrib['name'].removeprefix('panda_joint'))-1
                dynamic=np.broadcast_to(np.eye(4),(n,4,4)).copy()
                dynamic[:,:3,:3]=Rotation.from_rotvec(states[:,index,None]*axis).as_matrix()
                transform=transform@dynamic
        return transform[:,:3,3],transform[:,:3,:3]


def pose_channels(query,reference):
    position=np.linalg.norm(query['position'][:,None]-reference['position'][None,:],axis=2)
    q=np.asarray(query['quaternion']);q=q/np.linalg.norm(q,axis=1)[:,None]
    r=np.asarray(reference['quaternion']);r=r/np.linalg.norm(r,axis=1)[:,None]
    angle=2*np.arccos(np.clip(abs(q@r.T),0,1))
    width=abs(query['width'].reshape(-1,1)-reference['width'].reshape(1,-1))
    return np.stack([position,angle,width],axis=-1)


def motion(pose):
    p=float(np.linalg.norm(np.diff(pose['position'],axis=0),axis=1).sum())
    R=rotations(pose['quaternion'])
    a=float((R[1:]*R[:-1].inv()).magnitude().sum())
    g=float(abs(np.diff(pose['width'].ravel())).sum())
    return np.array([p,a,g])


def suffix_map(cost,start_range,force_end):
    """Every query point retained. Free start in allowed milestone; variable end.

    Per query observation reference can advance 0..5 samples. DP retains all
    hypotheses; full offline backtracking, not an online decision rule.
    """
    n,m=cost.shape;parent=np.zeros((n,m),dtype=np.int32)
    prev=np.full(m,np.inf);lo,hi=start_range
    prev[lo:hi+1]=cost[0,lo:hi+1]
    for i in range(1,n):
        choices=np.full((6,m),np.inf);choices[0]=prev
        for k in range(1,min(6,m)):choices[k,k:]=prev[:-k]
        parent[i]=np.arange(m)-np.argmin(choices,axis=0)
        prev=cost[i]+np.min(choices,axis=0)
    mapping=np.zeros(n,dtype=int)
    end=m-1 if force_end and np.isfinite(prev[-1]) else int(np.argmin(prev))
    mapping[-1]=end
    for i in range(n-1,0,-1):mapping[i-1]=parent[i,mapping[i]]
    return mapping


def first_event(rows,key):
    return next((r['step'] for r in rows if r.get(key,False)),None)


def trim_success(item):
    t=first_event(item['rows'],'success')
    if t is not None:
        item=dict(item,pose={k:v[:t+1] for k,v in item['pose'].items()},rows=item['rows'][:t+1])
    return item


def reference_range(query,ref):
    # Benchmark-only milestone context is explicit, not claimed as proprioceptive.
    state=query['rows'][0];rows=ref['rows'];end=len(rows)-1
    opened=first_event(rows,'ever_drawer_opened')
    lifted=first_event(rows,'ever_lifted')
    if state.get('ever_lifted',False):return (lifted or 0,end)
    if state.get('ever_drawer_opened',False):return (opened or 0,lifted or end)
    return (0,opened or end)


def slice_episode(item,start):
    return dict(item,pose={k:v[start:] for k,v in item['pose'].items()},rows=item['rows'][start:])


def summarize_pair(query,reference,force_end=True):
    residual=pose_channels(query['pose'],reference['pose'])
    units=np.array([POSITION_SCALE,ROTATION_SCALE,WIDTH_SCALE])
    costs=np.linalg.norm(residual/units,axis=2)
    start_range=reference_range(query,reference)
    mapping=suffix_map(costs,start_range,force_end)
    matched=residual[np.arange(len(mapping)),mapping]
    normalized=np.linalg.norm(matched/units,axis=1)
    persistent=float(max(np.min(normalized[i:i+3]) for i in range(max(1,len(normalized)-2))))
    ref_pose={k:v[mapping[0]:mapping[-1]+1] for k,v in reference['pose'].items()}
    query_motion=motion(query['pose']);ref_motion=motion(ref_pose)
    debt=np.maximum(query_motion-ref_motion,0)
    # Event-independent comparison of joint gripper transitions; no success label.
    gap=query['pose']['width'].ravel()
    openings=sum(gap[i-1]<.06<=gap[i] for i in range(1,len(gap)))
    refgap=ref_pose['width'].ravel()
    refopenings=sum(refgap[i-1]<.06<=refgap[i] for i in range(1,len(refgap)))
    return dict(reference_seed=reference['seed'],reference_start=int(mapping[0]),
        reference_end=int(mapping[-1]),start_range=list(start_range),
        shape_mean=float(normalized.mean()),shape_persistent_peak=persistent,
        mean_position_cm=float(matched[:,0].mean()*100),mean_orientation_deg=float(np.rad2deg(matched[:,1].mean())),
        mean_width_mm=float(matched[:,2].mean()*1000),
        extra_path_cm=float(debt[0]*100),extra_rotation_deg=float(np.rad2deg(debt[1])),
        extra_gripper_travel_mm=float(debt[2]*1000),extra_open_events=int(max(0,openings-refopenings)),
        total_path_cm=float(query_motion[0]*100),total_rotation_deg=float(np.rad2deg(query_motion[1])),
        total_gripper_travel_mm=float(query_motion[2]*1000),
        normalized_timeline=normalized.tolist(),position_cm_timeline=(matched[:,0]*100).tolist(),
        orientation_deg_timeline=np.rad2deg(matched[:,1]).tolist(),width_mm_timeline=(matched[:,2]*1000).tolist(),
        reference_mapping=mapping.tolist(),query_steps=len(mapping)-1,
        reference_remaining_actions=int(mapping[-1]-mapping[0]))


def main(root,previous):
    formal=previous/'inputs/formal'
    experts=[trim_success(load_episode(formal/'anchor_0/accepted'/f"episode_{r['episode_index']:06d}",r))
             for r in read_jsonl(formal/'anchor_0/accepted_experts.jsonl')]
    refs=[r for r in experts if r['seed']%5<3]
    calibration=[r for r in experts if r['seed']%5==3]
    holdout=[r for r in experts if r['seed']%5==4]
    contexts=np.array([r['context'] for r in refs])
    scales=np.maximum(1.4826*np.median(abs(contexts-np.median(contexts,axis=0)),axis=0),.001)

    def nearest(item,bank):
        return min((r for r in bank if r['seed']!=item['seed']),
                   key=lambda r:np.linalg.norm((item['context']-r['context'])/scales))

    # Fit world transform once, then verify independent trajectories / all frames.
    fk=PandaFK(root/'inputs/panda_v2.urdf')
    e=experts[0];q=np.load(Path(e['directory'])/'states.npy')[:len(e['rows'])]
    pos,R=fk.pose(q);Rworld=rotations(e['pose']['quaternion']).as_matrix()[0]@R[0].T
    translation=e['pose']['position'][0]-Rworld@pos[0]
    position_errors=[];angle_errors=[]
    for e in experts:
        q=np.load(Path(e['directory'])/'states.npy')[:len(e['rows'])]
        pos,R=fk.pose(q);pred=pos@Rworld.T+translation
        position_errors.extend(np.linalg.norm(pred-e['pose']['position'],axis=1))
        angle_errors.extend((Rotation.from_matrix(Rworld@R)*rotations(e['pose']['quaternion']).inv()).magnitude())
    fk_audit=dict(max_position_error_m=float(max(position_errors)),max_orientation_error_deg=float(np.rad2deg(max(angle_errors))),
                  episodes=30,frames=len(position_errors),translation=translation.tolist(),rotation=Rworld.tolist())
    assert max(position_errors)<1e-4 and max(angle_errors)<1e-3,fk_audit
    print('FK_AUDIT',fk_audit,flush=True)

    idroot=root/'inputs/policy_id';ids=[]
    for meta in json.loads((idroot/'summary.json').read_text())['rows']:
        directory=idroot/'episodes'/f"episode_{meta['episode_index']:06d}"
        states=np.load(directory/'states.npy')
        timeline=json.loads((directory/'timeline.json').read_text())['timeline']
        rows=[dict(step=r['step'],**r['events']) for r in timeline]
        first=first_event(rows,'success')
        assert (first is not None)==bool(meta['success'])
        if first is not None:states=states[:first+1];rows=rows[:first+1]
        p,R=fk.pose(states);R=Rworld@R;p=p@Rworld.T+translation
        q=Rotation.from_matrix(R).as_quat()[:,[3,0,1,2]]
        reset=json.loads((directory/'reset_metadata.json').read_text())
        ids.append(dict(seed=meta['seed'],meta=meta,rows=rows,context=np.array(reset['object_pose']['p']+reset['target_pose']['p']),
                        pose=dict(position=p,quaternion=q,width=states[:,-2:].sum(axis=1)[:,None])))
    success_ids=[r for r in ids if r['meta']['success']]
    idrefs=[r for i,r in enumerate(success_ids) if i%3==0]
    idcal=[r for i,r in enumerate(success_ids) if i%3==1]
    idtest=[r for i,r in enumerate(success_ids) if i%3==2]
    id_rows=[]
    # Use a single success-ID reference bank; no failure/OOD used to fit thresholds.
    for group,items in [('id_calibration',idcal),('id_heldout_success',idtest),('id_failure',[r for r in ids if not r['meta']['success']])]:
        for item in items:
            # Same free-end rule for successes and failures: outcome labels must
            # not change the policy deviation scoring algorithm.
            scored=summarize_pair(item,nearest(item,idrefs),force_end=False)
            id_rows.append(dict(group=group,seed=item['seed'],**scored))
    policies=[]
    for meta in json.loads((previous/'inputs/policy20/summary.json').read_text())['rows']:
        item=load_episode(previous/'inputs/policy20/episodes'/f"episode_{meta['episode_index']:06d}",meta)
        policies.append(item)
        id_rows.append(dict(group='ood_failure',seed=item['seed'],**summarize_pair(item,nearest(item,idrefs),False)))
    id_tau=float(np.quantile([r['shape_persistent_peak'] for r in id_rows if r['group']=='id_calibration'],.925))
    id_summary={g:dict(n=len([r for r in id_rows if r['group']==g]),
        above=sum(r['shape_persistent_peak']>id_tau for r in id_rows if r['group']==g),
        median_peak=float(np.median([r['shape_persistent_peak'] for r in id_rows if r['group']==g])))
        for g in ['id_calibration','id_heldout_success','id_failure','ood_failure']}
    print('ID_REFERENCE',id_tau,id_summary,flush=True)

    nominal=[]
    for group,bank in [('calibration',calibration),('heldout',holdout)]:
        for item in bank:
            reference=nearest(item,refs)
            for cut in CUTS:
                if cut>=len(item['rows'])-3:continue
                query=slice_episode(item,cut)
                scored=summarize_pair(query,reference)
                nominal.append(dict(group=group,seed=item['seed'],cut=cut,**scored))
    keys=['shape_mean','shape_persistent_peak','extra_path_cm','extra_rotation_deg','extra_gripper_travel_mm']
    floors=[1,1,2,15,10]
    norms={k:max(f,float(np.quantile([r[k] for r in nominal if r['group']=='calibration'],.925))) for k,f in zip(keys,floors)}

    def candidate(row):
        row['D_shape']=row['shape_persistent_peak']/norms['shape_persistent_peak']
        row['D_motion']=max(row[k]/norms[k] for k in keys[2:])
        row['D_combined']=max(row['D_shape'],row['D_motion'])
        return row
    nominal=[candidate(r) for r in nominal]
    attempts=[]
    for anchor in [0,50,80,120,160,220]:
        folder=formal/f'anchor_{anchor}'
        for meta in read_jsonl(folder/'raw_attempts.jsonl'):
            item=load_episode(folder/'raw_attempts'/Path(meta['evidence_dir']).name,meta)
            item=trim_success(item);take=int(meta['actual_takeover_step'])
            assert take<len(item['rows'])
            query=slice_episode(item,take)
            score=summarize_pair(query,nearest(item,refs),force_end=meta['success'])
            row=dict(anchor=anchor,seed=item['seed'],accepted=meta['accepted'],success=meta['success'],takeover=take,
                     expert_action_steps=meta['expert_action_steps'],video=meta['video'],directory=item['directory'],**score)
            candidate(row)
            # History tags are independent of score, NOT manually validated recovery labels.
            opening=item['pose']['width'].ravel()
            opened=first_event(item['rows'],'ever_drawer_opened')
            closed_then_reopened=False
            if opened is not None and opened<take:
                segment=opening[opened:take+1]
                closures=np.flatnonzero(segment<.012)
                if len(closures):closed_then_reopened=bool(np.any(segment[closures[0]:]>.06))
            row['pre_take_empty_close_reopen_proxy']=closed_then_reopened
            row['pre_take_drawer_opened']=opened is not None and opened<=take
            attempts.append(row)
        print('ANCHOR_COMPLETE',anchor,flush=True)

    def summary(rows):
        return dict(n=len(rows),**{k:float(np.median([r[k] for r in rows])) if rows else None
               for k in ['D_shape','D_motion','D_combined']+keys[2:]})
    anchors={str(a):dict(accepted=summary([r for r in attempts if r['anchor']==a and r['accepted']]),
                          failed_partial=summary([r for r in attempts if r['anchor']==a and not r['accepted']])) for a in [0,50,80,120,160,220]}
    correlations={}
    for a in [50,80,120,160,220]:
        rows=[r for r in attempts if r['anchor']==a and r['accepted']]
        correlations[str(a)]={k:float(spearmanr([r[k] for r in rows],[r['expert_action_steps'] for r in rows]).statistic)
                              for k in ['D_shape','D_motion','D_combined']}
    out=dict(diagnostic_only=True,fk_audit=fk_audit,protocol=dict(
        sources=dict(ood_expert=str(formal),id_policy=str(idroot)),reference_seeds=[r['seed'] for r in refs],
        calibration_seeds=[r['seed'] for r in calibration],heldout_seeds=[r['seed'] for r in holdout],
        id_reference_seeds=[r['seed'] for r in idrefs],id_calibration_seeds=[r['seed'] for r in idcal],id_heldout_seeds=[r['seed'] for r in idtest],
        physical_scales=dict(position_m=POSITION_SCALE,orientation_deg=15,gripper_m=WIDTH_SCALE),
        norms=norms,nominal_cuts=CUTS,threshold_q=.925,
        caveats=['post-takeover evaluator requires expert rollout; not an online gate',
        'nearest reset object+target context and coarse task milestones are used for offline reference matching',
        'failed continuations are partial; their low scores cannot be interpreted as good takeovers',
        'candidate max-combination is exploratory; not a false-alarm probability',
        '61 ID successes truncated at first success; 20 independent OOD failures; ID vs OOD geometry differs',
        '8 cuts per calibration expert are correlated; only 6 independent calibration experts',
        'no manual Tref supplied; E not calculated; no downstream SR used']),
        successful_id_reference=dict(tau=id_tau,summary=id_summary,rows=id_rows),
        nominal=nominal,attempts=attempts,anchor_summary=anchors,cost_correlations=correlations)
    (root/'analysis.json').write_text(json.dumps(out,indent=2,allow_nan=False))
    plots(root,out)
    print('SUFFIX_ANALYSIS_COMPLETE',json.dumps(anchors),flush=True)


def plots(root,out):
    fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    keys=['D_shape','D_motion','D_combined','extra_path_cm','extra_rotation_deg','extra_gripper_travel_mm']
    for ax,key in zip(axes.flat,keys):
        groups=[[r[key] for r in out['nominal'] if r['group']=='heldout']]+[[r[key] for r in out['attempts'] if r['anchor']==a and r['accepted']] for a in [0,50,80,120,160,220]]
        ax.boxplot(groups,tick_labels=['normal','t0','t50','t80','t120','t160','t220'],showfliers=False)
        rng=np.random.default_rng(0)
        for i,values in enumerate(groups):ax.scatter(i+1+rng.uniform(-.13,.13,len(values)),values,s=8,alpha=.35)
        ax.set(title=key,ylabel='Diagnostic score / physical amount',yscale='symlog')
        ax.set_ylim(bottom=0)
    fig.suptitle('Post-takeover expert vs nominal OOD expert | accepted continuations only | different reset pools')
    fig.savefig(root/'suffix_candidates.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
    groups=['id_calibration','id_heldout_success','id_failure','ood_failure']
    rows=out['successful_id_reference']['rows'];tau=out['successful_id_reference']['tau']
    ax.boxplot([[r['shape_persistent_peak']/tau for r in rows if r['group']==g] for g in groups],tick_labels=groups)
    ax.axhline(1,color='black',ls='--');ax.set(ylabel='Persistent pose score / successful-ID q92.5',title='Successful policy as reference (position + orientation + gripper)')
    fig.savefig(root/'successful_id_reference.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--previous',type=Path,required=True)
    args=p.parse_args();main(args.root,args.previous)
