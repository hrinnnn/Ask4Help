"""Frozen historical TASR on new expert suffixes; no threshold tuning."""
import argparse
import io
import json
import tempfile
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from scipy.spatial.transform import Rotation
from compute_pi05_tasr import PandaFK, PHASES, pair, target_blocks
from pi05_feedback_artifacts import episode_path


def compatible_pose(tcp, rotations, object_p, widths, end):
    axes=rotations[end-1]
    return {'position':(tcp-object_p)@axes,
            'quaternion':Rotation.from_matrix(axes.T@rotations).as_quat()[:,[3,0,1,2]],
            'width':widths[:,None]}


def reference_bank(bundle, task, fk):
    meta=json.loads(bundle.read('manifest.json'))[task];refs=[]
    for r in meta['normal']:
        if r['seed'] not in meta['protocol']['references']:continue
        assert r['target_replay_max_error']<1e-4
        original=np.load(io.BytesIO(bundle.read(r['arrays'])))
        replay=np.load(io.BytesIO(bundle.read(r['replay_array'])))
        start,n=r['expert_start'],r['expert_points']
        positions,rotations=fk.pose(original['qpos']);positions+=np.array([-.615,0,0])
        pose=compatible_pose(positions,rotations,replay['object_p'][start:start+n],original['qpos'][:,-2:].sum(1),r['blocks']['close'][1])
        refs.append({'row':{'seed':r['seed']},'pose':pose,'contact':replay['grasped'][start:start+n],'blocks':r['blocks']})
    return refs,meta


def budget_selection(arms):
    reachable={}
    for arm,rows in arms.items():
        sums={0:[]}
        for r in rows:
            for total,chosen in list(sums.items()):
                value=total+r['expert_points']
                if value not in sums:sums[value]=chosen+[r['episode']]
        reachable[arm]=sums
    common=set.intersection(*(set(v) for v in reachable.values()))
    budget=max(common)
    return budget,{arm:values[budget] for arm,values in reachable.items()}


def run(args):
    with ZipFile(args.reference) as bundle, tempfile.TemporaryDirectory(prefix='tasr_reference_') as tmp:
        urdf=Path(tmp)/'panda_v2.urdf';urdf.write_bytes(bundle.read('panda_v2.urdf'));fk=PandaFK(urdf)
        refs,meta=reference_bank(bundle,args.task,fk)
    all_rows={};gaps=[]
    for arm in ['fixed','feedback']:
        source=json.loads((args.root/arm/'summary.json').read_text());rows=[]
        for r in source['rows']:
            if not r['accepted']:continue
            n=r['expert_suffix_actions'];start=r['takeover'];episode=r['episode']
            row={'arm':arm,'episode':episode,'seed':r['seed'],'split':r['split'],'expert_points':n}
            if r['split']=='id':
                row.update(compatible_points={str(f):0 for f in [1,2,3]},coverage='ID_excluded_from_numerator');rows.append(row);continue
            data=np.load(episode_path(args.root/arm,r)/'trace.npz')
            try:
                blocks=target_blocks(args.task,data['actions'],data['grasped'],start,n)
                qpos=data['qpos'][start:start+n];tcp=data['tcp'][start:start+n]
                rotations=Rotation.from_quat(data['tcp_q'][start:start+n][:,[1,2,3,0]]).as_matrix()
                checked,checked_rot=fk.pose(qpos);checked+=np.array([-.615,0,0])
                if np.max(abs(checked-tcp))>1e-4:raise ValueError('recorded TCP and historical reference FK differ')
                e={'pose':compatible_pose(tcp,rotations,data['object_p'][start:start+n],qpos[:,-2:].sum(1),blocks['close'][1]),
                   'contact':data['grasped'][start:start+n],'blocks':blocks}
                candidates=[]
                for ref in refs:
                    if ref['row']['seed']==r['seed']:continue
                    parts={p:pair(e,ref,p) for p in PHASES}
                    if any(x is None for x in parts.values()):continue
                    score=np.mean([np.mean(x['distance'])+1000*(1-np.mean(x['contact'])) for x in parts.values() if x['indices']])
                    candidates.append((score,ref['row']['seed'],parts))
                _,ref_seed,parts=min(candidates,key=lambda x:x[0])
                counts={str(f):sum(sum(ok and distance<=meta['limits'][p]*f for distance,ok in zip(part['distance'],part['contact'])) for p,part in parts.items()) for f in [1,2,3]}
                row.update(compatible_points=counts,coverage='COMPLETE',reference_seed=ref_seed,blocks=blocks,parts=parts)
            except (ValueError,IndexError) as error:
                row.update(compatible_points=None,coverage='METRIC_GAP',reason=str(error));gaps.append(row.copy())
            rows.append(row)
        all_rows[arm]=rows
    budget,selected=budget_selection(all_rows)
    results={}
    for arm,rows in all_rows.items():
        group=[r for r in rows if r['episode'] in selected[arm]]
        known=[r for r in group if r['compatible_points'] is not None]
        unknown=sum(r['expert_points'] for r in group if r['compatible_points'] is None)
        numerators={str(f):sum(r['compatible_points'][str(f)] for r in known) for f in [1,2,3]}
        results[arm]={'expert_action_budget':budget,'selected_episodes':selected[arm],'unscored_points':unknown,
                      'OOD_expert_points':sum(r['expert_points'] for r in group if r['split']=='ood'),
                      'compatible_points':numerators,'TASR':{f:n/budget if budget and not unknown else None for f,n in numerators.items()},
                      'bounds':{f:[n/budget,(n+unknown)/budget] if budget else None for f,n in numerators.items()}}
    out={'status':'PILOT_TASR_COMPLETE' if budget and not any(v['unscored_points'] for v in results.values()) else 'PILOT_TASR_HAS_GAPS',
         'task':args.task,'metric_reference':str(args.reference),'frozen_limits':meta['limits'],'results':results,'rows':all_rows,'gaps':gaps,
         'interpretation':'Pretraining pilot TASR, exact whole-suffix budget and frozen historical proximity radii. No downstream SR claim.'}
    (args.root/'pilot_TASR.json').write_text(json.dumps(out,indent=2));print(json.dumps({'status':out['status'],'results':results}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['stackcube','airplane'],required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--root',type=Path,required=True);run(p.parse_args())
