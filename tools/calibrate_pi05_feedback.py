"""ID-only calibration for a frozen pi0.5 PCA/timing-feedback gate."""
import argparse
import io
import json
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from pi05_feedback_runtime import Pi05FeedbackRuntime
from pi05_timing_feedback import action_block_error, corrective_motion


ASSETS = {
 'stackcube_legacy_ood': {
  'pca':'/mnt/data/ask4help/results/stackcube_vla_fail/internal_detector_matrix_v1/assets/internal_detector_assets.pt',
  'cache':'/mnt/data/ask4help/results/stackcube_vla_fail/multilayer_llmd_step7000_v1/assets/multilayer_feature_cache.pt'},
 'airplane_yaw_ood': {
  'pca':'/mnt/data/ask4help/results/pick_single_ycb_airplane/yaw_swap_v1/detector_assets_step5000_all_id98_v1/detector_assets.pt',
  'cache':'/mnt/data/ask4help/results/pick_single_ycb_airplane/yaw_swap_v1/detector_assets_step5000_all_id98_v1/feature_cache.pt'},
}


def panda_tcp(states, urdf):
    joints=ET.parse(urdf).getroot().findall('joint')
    parents={j.find('child').get('link'):j for j in joints}
    link='panda_hand_tcp';chain=[]
    while link in parents:
        joint=parents[link];chain.append(joint);link=joint.find('parent').get('link')
    transform=np.broadcast_to(np.eye(4),(len(states),4,4)).copy()
    for joint in reversed(chain):
        origin=joint.find('origin');fixed=np.eye(4)
        if origin is not None:
            fixed[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
            fixed[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
        transform=transform@fixed
        if joint.get('type')=='revolute':
            idx=int(joint.get('name').removeprefix('panda_joint'))-1
            axis=np.fromstring(joint.find('axis').get('xyz'),sep=' ')
            moving=np.broadcast_to(np.eye(4),(len(states),4,4)).copy()
            moving[:,:3,:3]=Rotation.from_rotvec(states[:,idx,None]*axis).as_matrix()
            transform=transform@moving
    return transform[:,:3,3]


def main(args):
    args.output.mkdir(parents=True,exist_ok=False)
    start=time.time()
    def progress(stage, **values):
        row={'pid':os.getpid(),'stage':stage,'elapsed':time.time()-start,**values}
        (args.output/'progress.json').write_text(json.dumps(row,indent=2));print(json.dumps(row),flush=True)
    manifest=json.loads(args.manifest.read_text())
    assets=manifest['task_assets'][args.task]
    runtime=Pi05FeedbackRuntime(args.task,assets)
    torch=runtime.torch;torch.set_num_threads(4)
    import pyarrow.parquet as pq
    from mani_skill import PACKAGE_ASSET_DIR
    urdf=Path(PACKAGE_ASSET_DIR)/'robots/panda/panda_v2.urdf'
    dataset=Path(assets.get('ID_dataset',assets.get('ID_dataset_candidate')))
    if not (dataset/'meta/info.json').exists():dataset=dataset/'lerobot'
    episodes=[json.loads(s) for s in (dataset/'meta/episodes.jsonl').read_text().splitlines()]
    progress('load_ID_reference')
    prior_asset=torch.load(ASSETS[args.task]['pca'],map_location='cpu',weights_only=False)
    cache=torch.load(ASSETS[args.task]['cache'],map_location='cpu',weights_only=False)
    if args.task=='stackcube_legacy_ood':
        pca=prior_asset['detectors']['vlm_bridge_final_mean__pca_residual']['statistics']
        features=cache['layers']['vlm_bridge_final_mean'].float().reshape(-1,2048)
    else:
        pca=prior_asset['statistics']['bridge_pca_residual']
        features=cache['bridge'].float().reshape(-1,2048)
    assert prior_asset['checkpoint']==assets['checkpoint']
    episode_ids=np.concatenate([np.repeat(e['episode_index'],e['length']) for e in episodes])
    assert len(episode_ids)==len(features)
    center=features.mean(0);scale=float(torch.sqrt(torch.mean(torch.sum((features-center)**2,dim=1))))
    z=(features-center)/scale;nearest=[]
    for first in range(0,len(z),128):
        distance=torch.cdist(z[first:first+128],z)
        distance[torch.as_tensor(episode_ids[first:first+128,None]==episode_ids[None,:])]=torch.inf
        nearest.extend(distance.min(1).values.numpy().tolist())
    radius=float(np.quantile(nearest,.95,method='higher'))
    mean=pca['mean'].float().reshape(-1).numpy()
    basis=pca['principal_components'].float().squeeze(0).numpy()
    def score(feature):
        v=feature-mean;return float(np.linalg.norm(v-(v@basis)@basis.T))
    progress('model_load',reference_observations=len(features),radius=radius,principal_dim=pca['principal_dim'])
    runtime.load_model()
    def raw_row(row):
        main=np.array(Image.open(io.BytesIO(row['image']['bytes'])).convert('RGB'))
        wrist=np.array(Image.open(io.BytesIO(row['wrist_image']['bytes'])).convert('RGB'))
        return {'agent':{'qpos':torch.as_tensor(row['state']).reshape(1,-1)},
                'sensor_data':{'base_camera':{'rgb':torch.from_numpy(main).unsqueeze(0)},
                               'hand_camera':{'rgb':torch.from_numpy(wrist).unsqueeze(0)}}}
    selected=np.linspace(0,len(episodes)-1,min(args.demo_episodes,len(episodes)),dtype=int)
    demo_rows=[];reversals=[];error_maxima=[]
    offsets=np.cumsum([0]+[e['length'] for e in episodes])
    for j,episode in enumerate(selected):
        rows=pq.read_table(dataset/f'data/chunk-000/episode_{episode:06d}.parquet').to_pylist()
        actions=np.asarray([r['actions'] for r in rows]);states=np.asarray([r['state'] for r in rows])
        positions=panda_tcp(states,urdf);width=states[:,-2:].sum(1)
        errors=[]
        for k in range(len(rows)-4):
            raw=raw_row(rows[k]);samples=[]
            for m in range(2):
                prediction,model_action=runtime.predict(raw,271009+int(episode)*10000+k*3+m)
                samples.append(np.clip(prediction,-1,1))
            errors.append(action_block_error(np.asarray(samples),actions[k:k+5]))
            if k==0:
                actual=runtime.bridge(raw,model_action)
                reference=features[int(offsets[episode])].numpy()
                error=float(np.max(np.abs(actual-reference)))
                if error>.02:raise ValueError(f'ID cached bridge mismatch: {error}')
        for k in range(5,len(rows)-5):
            reversals.append(corrective_motion(positions[k-5],positions[k],positions[k+5],width[k-5],width[k],width[k+5]))
        error_maxima.append(max(errors))
        result={'episode':int(episode),'anchors':len(rows),'observed_windows':len(errors),'max_error':max(errors),'errors':errors}
        demo_rows.append(result)
        (args.output/'demo_progress.json').write_text(json.dumps(demo_rows))
        progress('ID_expert_calibration',done=j+1,total=len(selected),last_max_error=max(errors))
    q_e=float(np.quantile(error_maxima,.95,method='higher'))
    q_r=float(np.quantile(reversals,.95,method='higher'))
    policy_rows=[];success_maxima=[]
    for episode in range(args.policy_episodes):
        env=runtime.build_env('id');seed=args.seed+episode
        raw,info=env.reset(seed=seed);scores=[];steps=0;strict=False;ever_grasped=False
        while steps<runtime.horizon:
            action,model_action=runtime.predict(raw,seed*1000+steps)
            scores.append(score(runtime.bridge(raw,model_action)))
            for a in runtime.clip(action[:5],env):
                raw,_,terminated,truncated,info=env.step(torch.as_tensor(a,device=env.unwrapped.device).reshape(1,-1))
                steps+=1;strict=bool(info['success'])
                if args.task=='airplane_yaw_ood':ever_grasped|=bool(env.unwrapped.agent.is_grasping(env.unwrapped.obj))
                if strict or bool(terminated) or bool(truncated):break
            if strict or bool(terminated) or bool(truncated):break
        qualifies=strict if args.task=='stackcube_legacy_ood' else ever_grasped
        row={'episode':episode,'seed':seed,'strict_success':strict,'ever_grasped':ever_grasped,
             'calibration_success':qualifies,'steps':steps,'scores':scores,'maximum':max(scores)}
        if qualifies:success_maxima.append(max(scores))
        policy_rows.append(row);env.close()
        (args.output/'ID_policy_progress.json').write_text(json.dumps(policy_rows))
        progress('ID_policy_calibration',done=episode+1,total=args.policy_episodes,qualifying_successes=len(success_maxima))
    if len(success_maxima)<20:raise ValueError(f'Only {len(success_maxima)} qualifying ID calibration successes; threshold not frozen')
    gamma=float(np.quantile(success_maxima,.95,method='higher'))
    buf=io.BytesIO();np.savez_compressed(buf,mean=mean,basis=basis,center=center.numpy())
    (args.output/'gate_arrays.npz').write_bytes(buf.getvalue())
    result={'task':args.task,'baseline_threshold':gamma,'error_reference':q_e,'reversal_reference':q_r,
            'scale':scale,'radius':radius,'principal_dim':int(pca['principal_dim']),
            'ID_demonstrations':len(selected),'qualifying_ID_policy_episodes':len(success_maxima),
            'policy_episodes':args.policy_episodes,'fixed_prior':'bridge only; action prior does not affect VLM prefix',
            'calibration_success_rule':'strict' if args.task=='stackcube_legacy_ood' else 'ever_grasped',
            'provenance':runtime.provenance(),'reference_assets':ASSETS[args.task],
            'source_dataset':str(dataset),'array_file':'gate_arrays.npz'}
    (args.output/'calibration.json').write_text(json.dumps(result,indent=2))
    (args.output/'CALIBRATION_COMPLETE.json').write_text(json.dumps({'task':args.task,'status':'ID_ONLY_CALIBRATION_COMPLETE'}))
    progress('complete',calibration=result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',choices=list(ASSETS),required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--demo-episodes',type=int,default=32);p.add_argument('--policy-episodes',type=int,default=50)
    p.add_argument('--seed',type=int,default=1740000);args=p.parse_args()
    try:main(args)
    except Exception as error:
        if args.output.exists():(args.output/'CALIBRATION_FAILED.json').write_text(json.dumps({'type':type(error).__name__,'message':str(error)}))
        raise
