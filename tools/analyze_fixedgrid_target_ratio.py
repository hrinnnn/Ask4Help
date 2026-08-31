#!/usr/bin/env python3
"""Target-supervision ratio on existing StackCube / Airplane timing corpora.

Axes are from the observed stable grasp, NOT unrecorded object quaternions.
Translation uses the actually recorded current object centre. The paired-nominal
axis mode is retained explicitly as a diagnostic alternative.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_open_drawer_suffix_recovery import PandaFK,pose_channels,suffix_map

PHASES=['approach_alignment','close']
UNITS=np.array([.02,np.deg2rad(15),.01])
MIN_RADIUS=float(np.linalg.norm(np.array([.001,np.deg2rad(1),.001])/UNITS))


def define_blocks(task,entry,arrays):
    start=int(entry['train']['expert_start_step']);n=int(entry['train']['expert_action_steps'])
    actions=arrays['actions'][start:start+n]
    negative=np.flatnonzero(actions[:,-1]<0)
    if not len(negative):raise ValueError(f"no closing command for {task}/{entry['meta']['seed']}")
    closing=int(negative[0])
    if task=='stackcube':
        count=5 # Frozen StackCubePrivilegedChunkOracle close chunk length.
        stable=3
    else:
        oracle=entry['meta']['oracle']
        selected=next(r for r in oracle['attempts'] if r['candidate']==oracle['selected_candidate'] and r['accepted'])
        assert selected['candidate'].startswith('neck_center')
        assert selected['accepted_lift'] and selected['still_grasped_after_lift']
        count=int(selected['close_executed_steps']);stable=int(selected['stable_grasp_steps'])
    end=closing+count
    if end>n:raise ValueError(f"closing exceeds suffix for {task}/{entry['meta']['seed']}")
    ts=arrays['task_states'];grasp=ts[:,-2]>.5
    assert np.all(grasp[start+end-stable+1:start+end+1]),(task,entry['meta']['seed'],closing,end)
    # Lift is used only to verify retained successful grasp, not to extend target.
    lift=float(np.max(ts[start+end:,2])-ts[start+end,2])
    if task=='airplane':assert lift>.02,(entry['meta']['seed'],lift)
    return {'approach_alignment':(0,closing),'close':(closing,end)},dict(close_steps=count,stable_steps=stable,verified_lift_after_close_m=lift)


def compare(query,ref,phase):
    a,b=query['blocks'][phase];c,d=ref['blocks'][phase]
    if a>=b:return dict(indices=[],mapping=[],distance=[],contact=[])
    q={k:v[a:b] for k,v in query['pose'].items()};r={k:v[c:d] for k,v in ref['pose'].items()}
    residual=pose_channels(q,r);cost=np.linalg.norm(residual/UNITS,axis=2)
    qc=query['contact'][a:b];rc=ref['contact'][c:d]
    mapping=suffix_map(cost+1000*(qc[:,None]!=rc[None,:]),(0,len(rc)-1),False)
    return dict(indices=list(range(a,b)),mapping=(mapping+c).tolist(),distance=cost[np.arange(len(mapping)),mapping].tolist(),contact=(qc==rc[mapping]).tolist())


def analyze_task(task,manifest,root,fk,frame_mode):
    episodes=[];fk_errors=[]
    for step,condition in manifest['conditions'].items():
        for entry in json.loads(Path(condition['episodes_file']).read_text()):
            arrays=dict(np.load(entry['arrays']));qpos=arrays['qpos'];n=len(qpos);start=int(entry['train']['expert_start_step'])
            ts=arrays['task_states'][start:start+n];p,R=fk.pose(qpos);p+=np.array([-.615,0,0])
            err=np.linalg.norm(p-ts[:,6:9],axis=1);fk_errors.extend(err)
            assert np.max(err)<1e-4,(task,entry['meta']['seed'],np.max(err))
            if task=='airplane':assert np.max(abs(qpos-ts[:,15:24]))<1e-6
            blocks,evidence=define_blocks(task,entry,arrays)
            episodes.append(dict(task=task,step=int(step),seed=int(entry['meta']['seed']),entry=entry,n=n,start=start,
                blocks=blocks,evidence=evidence,world_position=p,world_rotation=R,object_position=ts[:,:3],
                width=qpos[:,-2:].sum(axis=1)[:,None],contact=ts[:,-2]>.5,
                raw_task_states=arrays['task_states'],actions=arrays['actions']))
    nominal={e['seed']:e for e in episodes if e['step']==0}
    for e in episodes:
        base=nominal[e['seed']] if frame_mode=='paired_nominal' else e
        j=base['blocks']['close'][1]-1
        axes=base['world_rotation'][j]
        rotation=Rotation.from_matrix(axes.T@e['world_rotation'])
        e['pose']=dict(position=(e['world_position']-e['object_position'])@axes,
                       quaternion=rotation.as_quat()[:,[3,0,1,2]],width=e['width'])
        e['frame_seed']=e['seed'];e['frame_step']=j;e['frame_source_step']=base['step']
    refs=[e for e in nominal.values() if e['seed']%5<3]
    cal=[e for e in nominal.values() if e['seed']%5==3]
    checks=[e for e in nominal.values() if e['seed']%5==4]
    comparison_cache={}
    def match(e,r,p):
        key=(e['step'],e['seed'],r['seed'],p)
        if key not in comparison_cache:comparison_cache[key]=compare(e,r,p)
        return comparison_cache[key]
    def choose(e,bank):
        def cost(r):
            parts=[match(e,r,p) for p in PHASES]
            return np.mean([np.mean(v['distance'])+1000*(1-np.mean(v['contact'])) for v in parts if v['indices']])
        # Exclude the same reset across ALL timing groups, not only t0.
        return min((r for r in bank if r['seed']!=e['seed']),key=cost)
    calibration_cache={}
    def thresholds(skip=None,q=.925):
        key=(skip,q)
        if key in calibration_cache:return calibration_cache[key]
        bank=[r for r in refs if r['seed']!=skip];values={p:[] for p in PHASES}
        for e in cal:
            if e['seed']==skip:continue
            r=choose(e,bank)
            for p in PHASES:
                m=match(e,r,p);values[p].extend(v for v,ok in zip(m['distance'],m['contact']) if ok)
        result={p:max(MIN_RADIUS,float(np.quantile(values[p],q))) for p in PHASES}
        calibration_cache[key]=result;return result
    def score(e,q=.925):
        # Leave-one-reset-out at every timing group for reference/calibration symmetry.
        skip=e['seed'];bank=[r for r in refs if r['seed']!=skip]
        r=choose(e,bank);limits=thresholds(skip,q);labels=['non_target']*e['n'];mapping=[-1]*e['n'];distance=[None]*e['n'];stats={}
        for p in PHASES:
            m=match(e,r,p);count=0
            for i,j,d,ok in zip(m['indices'],m['mapping'],m['distance'],m['contact']):
                green=ok and d<=limits[p];labels[i]='target_compatible' if green else 'target_mismatch';mapping[i]=j;distance[i]=d;count+=int(green)
            stats[p]=dict(target=len(m['indices']),compatible=count)
        valid=labels.count('target_compatible')
        return dict(task=task,step=e['step'],seed=e['seed'],selected=e['entry']['selected'],expert_start=e['start'],expert_anchors=e['n'],
            compatible_anchors=valid,Q=valid/e['n'],blocks=e['blocks'],phase_stats=stats,thresholds=limits,
            reference_seed=r['seed'],reference_mapping=mapping,distance=distance,labels=labels,
            arrays=e['entry']['arrays'],video=e['entry']['meta']['video'],grasp_evidence=e['evidence'],
            grasp_frame_seed=e['frame_seed'],grasp_frame_step=e['frame_step'],grasp_frame_source_step=e['frame_source_step'])
    rows=[score(e) for e in episodes]
    summary={}
    for step in manifest['anchors']:
        selected=[r for r in rows if r['step']==step and r['selected']]
        denominator=sum(r['expert_anchors'] for r in selected);numerator=sum(r['compatible_anchors'] for r in selected)
        assert denominator==manifest['budget']
        summary[str(step)]=dict(episodes=len(selected),expert_anchors=denominator,compatible_anchors=numerator,Q=numerator/denominator)
    checkrows=[r for r in rows if r['step']==0 and r['seed'] in {e['seed'] for e in checks}]
    check={p:dict(target=sum(r['phase_stats'][p]['target'] for r in checkrows),compatible=sum(r['phase_stats'][p]['compatible'] for r in checkrows)) for p in PHASES}
    sensitivity=[]
    for q in [.90,.925,.95]:
        current=rows if q==.925 else [score(e,q) for e in episodes if e['entry']['selected']]
        sensitivity.append(dict(q=q,values={str(step):sum(r['compatible_anchors'] for r in current if r['step']==step and r['selected'])/manifest['budget'] for step in manifest['anchors']}))
    result=dict(protocol=dict(task=task,target='approach/alignment and closing to stable object grasp; lift/transport/release excluded',
        frame_mode=frame_mode,frame='current object-centred translation; axes from observed stable TCP grasp; NOT measured object-body quaternion',
        frame_reason='source task-state logs do not contain per-frame object rotations; do not substitute terminal metadata',
        source=manifest['root'],budget=manifest['budget'],q=.925,units=dict(position_m=.02,angle_deg=15,gripper_mm=10),minimum_radius=MIN_RADIUS,
        reference_seeds=[e['seed'] for e in refs],calibration_seeds=[e['seed'] for e in cal],check_seeds=[e['seed'] for e in checks],
        reference_rule='one same-task nominal expert selected by equal-phase mean distance/contact; same reset excluded from reference and calibration',
        bounds_rule='close begins at first negative gripper command; StackCube closes for frozen 5-step chunk; Airplane uses accepted neck-candidate close_executed_steps',
        no_new_training=True),nominal_check=check,default_thresholds=thresholds(),max_fk_position_error_m=float(max(fk_errors)),fk_frames=len(fk_errors),
        summary=summary,sensitivity=sensitivity,rows=rows)
    print(task,'SUMMARY',json.dumps(summary),flush=True);print(task,'CHECK',check,flush=True)
    directory=root/task;directory.mkdir(exist_ok=True)
    # Lightweight derived poses facilitate independent audit and video overlays.
    for e in episodes:
        np.savez_compressed(directory/f"pose_step_{e['step']}_seed_{e['seed']}.npz",**e['pose'],contact=e['contact'],
                            world_position=e['world_position'],world_rotation=e['world_rotation'],object_position=e['object_position'])
    return result


def main(root,frame_mode):
    manifest=json.loads((root/'inputs/manifest.json').read_text())
    urdf=Path('artifacts/open_drawer_suffix_d_20260831/inputs/panda_v2.urdf');fk=PandaFK(urdf)
    result={task:analyze_task(task,m,root,fk,frame_mode) for task,m in manifest.items()}
    if (root/'analysis.json').exists() and not (root/'analysis_paired_nominal_v1.json').exists():
        (root/'analysis_paired_nominal_v1.json').write_text((root/'analysis.json').read_text())
    (root/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for ax,(task,r) in zip(axes,result.items()):
        steps=list(r['summary']);values=[r['summary'][s]['Q']*100 for s in steps]
        ax.bar(steps,values,color='#30866c')
        for i,v in enumerate(values):ax.text(i,v+.8,f'{v:.1f}%',ha='center')
        ax.set(title=f"{task}: budget {r['protocol']['budget']}",xlabel='Takeover step',ylabel='Target-compatible expert anchors (%)',ylim=(0,max(values)*1.25))
    fig.savefig(root/'target_ratios.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--frame-mode',choices=['paired_nominal','own_grasp'],default='own_grasp')
    args=p.parse_args();main(args.root,args.frame_mode)
