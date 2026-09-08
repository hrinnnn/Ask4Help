"""Full ID calibration in two parallel shards, using original OpenDrawer eval mode."""
import argparse,io,json,os,time
from pathlib import Path
import numpy as np
from PIL import Image
from pi05_feedback_runtime import Pi05FeedbackRuntime
from pi05_timing_feedback import action_block_error,corrective_motion
from calibrate_pi05_feedback import panda_tcp


def run(args):
    args.output.mkdir(parents=True,exist_ok=False);start=time.time()
    manifest=json.loads(args.manifest.read_text());assets=manifest['task_assets']['open_drawer_grasp_ood']
    runtime=Pi05FeedbackRuntime('open_drawer_grasp_ood',assets);torch=runtime.torch;torch.set_num_threads(4)
    assert runtime.inference_mode=='eval'
    reference=torch.load(args.reference,map_location='cpu',weights_only=False)
    pca=reference['detectors']['vlm_bridge_final_mean__pca_residual']['statistics']
    features=reference['layers']['vlm_bridge_final_mean'].float().reshape(-1,2048)
    mean=pca['mean'].float().reshape(-1).numpy();basis=pca['principal_components'].float().squeeze(0).numpy()
    dataset=Path(assets['ID_dataset']);episodes=[json.loads(x) for x in (dataset/'meta/episodes.jsonl').read_text().splitlines()]
    offsets=np.cumsum([0]+[r['length'] for r in episodes]);runtime.load_model()
    provenance={'runtime':runtime.provenance(),'component':args.component,'shard':args.shard,'reference_asset':str(args.reference),
                'ID_dataset':str(dataset),'shard_rule':'contiguous halves of the original fixed32 demonstrations or fixed50 ID-policy seeds'}
    (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2));rows=[]
    def progress(**kw):
        r={'pid':os.getpid(),'component':args.component,'shard':args.shard,'done':len(rows),'elapsed':time.time()-start,**kw}
        (args.output/'progress.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
    if args.component=='demos':
        import pyarrow.parquet as pq
        from mani_skill import PACKAGE_ASSET_DIR
        urdf=Path(PACKAGE_ASSET_DIR)/'robots/panda/panda_v2.urdf'
        selected=np.linspace(0,len(episodes)-1,32,dtype=int)[args.shard*16:(args.shard+1)*16]
        for episode in selected:
            data=pq.read_table(dataset/f'data/chunk-000/episode_{episode:06d}.parquet').to_pylist()
            states=np.asarray([r['state'] for r in data]);actions=np.asarray([r['actions'] for r in data])
            positions=panda_tcp(states,urdf);width=states[:,-2:].sum(1);errors=[]
            for k in range(len(data)-4):
                r=data[k]
                raw={'agent':{'qpos':torch.as_tensor(r['state']).reshape(1,-1)},'sensor_data':{
                  'base_camera':{'rgb':torch.from_numpy(np.array(Image.open(io.BytesIO(r['image']['bytes'])).convert('RGB'))).unsqueeze(0)},
                  'hand_camera':{'rgb':torch.from_numpy(np.array(Image.open(io.BytesIO(r['wrist_image']['bytes'])).convert('RGB'))).unsqueeze(0)}}}
                samples=[]
                for m in range(2):
                    prediction,prior=runtime.predict(raw,271009+int(episode)*10000+k*3+m);samples.append(np.clip(prediction,-1,1))
                errors.append(action_block_error(np.asarray(samples),actions[k:k+5]))
                if k==0:
                    difference=float(np.max(abs(runtime.bridge(raw,prior)-features[int(offsets[episode])].numpy())))
                    assert difference<1e-6,'Mode-invariant native Bridge reference must remain identical'
            reversals=[corrective_motion(positions[k-5],positions[k],positions[k+5],width[k-5],width[k],width[k+5]) for k in range(5,len(data)-5)]
            rows.append({'episode':int(episode),'anchors':len(data),'observed_windows':len(errors),'max_error':max(errors),'errors':errors,'reversals':reversals})
            (args.output/'rows.json').write_text(json.dumps(rows));progress(total=16,last_episode=int(episode),last_max_error=max(errors))
    else:
        for i in range(args.shard*25,(args.shard+1)*25):
            seed=1781000+i;env=runtime.build_env('id');raw,_=env.reset(seed=seed);scores=[];steps=0;strict=False
            if i==args.shard*25:Image.fromarray(runtime.snapshot(env,raw)['main']).save(args.output/'reset_main.png')
            while steps<400:
                action,prior=runtime.predict(raw,seed*1000+steps)
                feature=runtime.bridge(raw,prior);v=feature-mean;scores.append(float(np.linalg.norm(v-(v@basis)@basis.T)))
                for a in runtime.clip(action[:5],env):
                    raw,_,terminated,truncated,info=env.step(torch.as_tensor(a,device=env.unwrapped.device).reshape(1,-1))
                    steps+=1;strict=bool(info['success'])
                    if strict or bool(terminated) or bool(truncated):break
                if strict or bool(terminated) or bool(truncated):break
            if i==args.shard*25:Image.fromarray(runtime.snapshot(env,raw)['main']).save(args.output/'terminal_main.png')
            rows.append({'episode':i,'seed':seed,'strict_success':strict,'ever_grasped':bool(env.unwrapped.evaluate()['ever_grasped']),
                         'calibration_success':strict,'steps':steps,'scores':scores,'maximum':max(scores)})
            env.close();(args.output/'rows.json').write_text(json.dumps(rows));progress(total=25,qualifying_successes=sum(r['strict_success'] for r in rows))
    (args.output/'COMPONENT_COMPLETE.json').write_text(json.dumps({'rows':len(rows),'component':args.component,'shard':args.shard}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--component',choices=['demos','policies'],required=True)
    p.add_argument('--shard',type=int,choices=[0,1],required=True);a=p.parse_args()
    try:run(a)
    except Exception as e:
        if a.output.exists():(a.output/'COMPONENT_FAILED.json').write_text(json.dumps({'error':repr(e)}))
        raise
