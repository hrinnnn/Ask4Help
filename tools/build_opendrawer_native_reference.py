"""Re-extract the same 128-ID reference in the actual H20 scoring runtime."""
import argparse,io,json,os,time
from pathlib import Path
import numpy as np
from PIL import Image
from pi05_feedback_runtime import Pi05FeedbackRuntime


def run(args):
    args.output.mkdir(parents=True,exist_ok=True);shards=args.output/'feature_shards';shards.mkdir(exist_ok=True)
    manifest=json.loads(args.manifest.read_text());assets=manifest['task_assets']['open_drawer_grasp_ood'];dataset=Path(assets['ID_dataset'])
    episodes=[json.loads(x) for x in (dataset/'meta/episodes.jsonl').read_text().splitlines()]
    selected=[e['episode_index'] for e in episodes]
    if args.episodes is not None:selected=[int(x) for x in args.episodes.split(',')]
    runtime=Pi05FeedbackRuntime('open_drawer_grasp_ood',assets);torch=runtime.torch;torch.set_num_threads(4)
    import pyarrow.parquet as pq
    from rlinf.algorithms.vla_fail import fit_pca_residual_statistics
    start=time.time();runtime.load_model()
    def raw_row(r):
        return {'agent':{'qpos':torch.as_tensor(r['state']).reshape(1,-1)},'sensor_data':{
          'base_camera':{'rgb':torch.from_numpy(np.array(Image.open(io.BytesIO(r['image']['bytes'])).convert('RGB'))).unsqueeze(0)},
          'hand_camera':{'rgb':torch.from_numpy(np.array(Image.open(io.BytesIO(r['wrist_image']['bytes'])).convert('RGB'))).unsqueeze(0)}}}
    equivalence=[]
    for i in selected:
        shard=shards/f'episode_{i:06d}.npy'
        expected=next(e['length'] for e in episodes if e['episode_index']==i)
        if shard.exists():
            assert np.load(shard,mmap_mode='r').shape==(expected,2048);continue
        rows=pq.read_table(dataset/f'data/chunk-000/episode_{i:06d}.parquet').to_pylist();values=[]
        assert len(rows)==expected
        for k,row in enumerate(rows):
            raw=raw_row(row);value=runtime.reference_bridge(raw)
            if k==0:
                shape=(1,runtime.model.config.action_horizon,runtime.model.config.action_dim)
                check=runtime.bridge(raw,torch.zeros(shape,device='cuda'))
                error=float(np.max(abs(value-check)));assert error<1e-6,error
                equivalence.append({'episode':i,'maximum_difference':error})
            values.append(value)
        buf=io.BytesIO();np.save(buf,np.stack(values));shard.write_bytes(buf.getvalue())
        progress={'pid':os.getpid(),'stage':'native_ID_reference','completed_shards':len(list(shards.glob('episode_*.npy'))),'total':len(episodes),'last_episode':i,'elapsed':time.time()-start}
        (args.output/'progress.json').write_text(json.dumps(progress,indent=2));print(json.dumps(progress),flush=True)
    (args.output/f'equivalence_{os.getpid()}.json').write_text(json.dumps(equivalence,indent=2))
    if not all((shards/f'episode_{e["episode_index"]:06d}.npy').exists() for e in episodes):
        print('SHARD_PASS_COMPLETE_FULL_REFERENCE_PENDING',flush=True);return
    features=torch.from_numpy(np.concatenate([np.load(shards/f'episode_{e["episode_index"]:06d}.npy') for e in episodes])).unsqueeze(1)
    assert len(features)==22973
    pca=fit_pca_residual_statistics(features,principal_dim=1000)
    result={'format':'opendrawer_runtime_native_ID128_reference_v1','checkpoint':assets['reference_checkpoint'],'runtime_checkpoint':assets['checkpoint'],
            'dataset_root':str(dataset),'indices':list(range(len(features))),'layers':{'vlm_bridge_final_mean':features},
            'detectors':{'vlm_bridge_final_mean__pca_residual':{'statistics':pca.state_dict()}},'runtime':runtime.provenance()}
    buf=io.BytesIO();torch.save(result,buf);(args.output/'bridge_reference.pt').write_bytes(buf.getvalue())
    (args.output/'NATIVE_REFERENCE_COMPLETE.json').write_text(json.dumps({'episodes':len(episodes),'anchors':len(features),'principal_dim':1000,'full_original_ID_corpus':True},indent=2))
    print('NATIVE_REFERENCE_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--episodes');args=p.parse_args()
    try:run(args)
    except Exception as e:
        if args.output.exists():(args.output/f'BUILD_FAILED_{os.getpid()}.json').write_text(json.dumps({'type':type(e).__name__,'message':str(e)}))
        raise
