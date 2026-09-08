"""Export exact selected complete expert suffixes with the native LeRobot writer."""
import argparse,io,json,os,shutil,tempfile,time
from pathlib import Path
import numpy as np
from PIL import Image


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    selection=json.loads((args.paired_root/'pilot_TASR.json').read_text())['results']
    budgets={r['expert_action_budget'] for r in selection.values()};assert len(budgets)==1 and min(budgets)>0
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    import pyarrow.parquet as pq
    rows=[];started=time.time()
    for arm in ['fixed','feedback']:
        summary=json.loads((args.paired_root/arm/'summary.json').read_text())
        provenance=json.loads((args.paired_root/arm/'provenance.json').read_text())
        assert (args.paired_root/arm/'INDEPENDENT_COLLECTION_AUDIT.json').exists()
        chosen=selection[arm]['selected_episodes'];assert len(chosen)==len(set(chosen))
        sources=[summary['rows'][i] for i in chosen]
        assert all(r['accepted'] for r in sources)
        assert sum(r['expert_suffix_actions'] for r in sources)==selection[arm]['expert_action_budget']
        destination=args.output/arm;episode_map=[]
        with tempfile.TemporaryDirectory(prefix='feedback_lerobot_',dir=os.environ.get('TMPDIR','/tmp')) as temporary:
            local=Path(temporary)/'dataset'
            dataset=LeRobotDataset.create(repo_id=f'local/pi05_feedback_{summary["task"]}_{arm}',root=local,
                fps=10,robot_type='panda',use_videos=False,image_writer_threads=2,image_writer_processes=0,
                features={'image':{'dtype':'image','shape':(384,384,3),'names':['height','width','channel']},
                          'wrist_image':{'dtype':'image','shape':(384,384,3),'names':['height','width','channel']},
                          'state':{'dtype':'float32','shape':(9,),'names':['state']},
                          'actions':{'dtype':'float32','shape':(8,),'names':['actions']}})
            for output_index,r in enumerate(sources):
                source=args.paired_root/arm/f'episode_{r["episode"]:04d}'/'trace.npz'
                d=dict(np.load(source));take=r['takeover'];n=r['expert_suffix_actions']
                assert len(d['actions'])==take+n and len(d['qpos'])==take+n+1
                for k in range(take,take+n):
                    dataset.add_frame({'image':d['main'][k],'wrist_image':d['wrist'][k],
                         'state':d['qpos'][k].astype(np.float32),'actions':d['actions'][k].astype(np.float32),
                         'task':provenance['runtime']['instruction']})
                dataset.save_episode()
                table=pq.read_table(local/f'data/chunk-000/episode_{output_index:06d}.parquet')
                assert len(table)==n
                for j in [0,n-1]:
                    row=table.slice(j,1).to_pylist()[0]
                    assert np.array_equal(np.asarray(row['actions'],dtype=np.float32),d['actions'][take+j])
                    assert np.array_equal(np.asarray(row['state'],dtype=np.float32),d['qpos'][take+j])
                    for feature,key in [('image','main'),('wrist_image','wrist')]:
                        rgb=np.asarray(Image.open(io.BytesIO(row[feature]['bytes'])).convert('RGB'))
                        assert np.array_equal(rgb,d[key][take+j])
                episode_map.append({'output_episode':output_index,'source_episode':r['episode'],'seed':r['seed'],
                                    'split':r['split'],'source_trace':str(source),'source_start':take,'anchors':n,
                                    'tail_anchors':min(9,n),'last_anchor_valid_targets':1})
                print(json.dumps({'arm':arm,'exported_episodes':len(episode_map),'elapsed':time.time()-started}),flush=True)
            assert dataset.meta.total_frames==selection[arm]['expert_action_budget']
            dataset.stop_image_writer()
            shutil.copytree(local,destination)
        histogram={str(k):sum(max(0,r['anchors']-9) if k==10 else int(r['anchors']>=k) for r in episode_map) for k in range(1,11)}
        row={'arm':arm,'dataset':str(destination),'anchors':sum(r['anchors'] for r in episode_map),
             'episodes':len(episode_map),'episode_map':episode_map,'horizon':10,'valid_target_histogram':histogram,
             'source_collection':str(args.paired_root/arm),'selection_source':str(args.paired_root/'pilot_TASR.json'),
             'status':'EXPORTED_PILOT_DATA_NOT_SFT_RESULT','padding_rule':'native LeRobot repeats final action; native mask adapter excludes temporal padding during SFT',
             'terminal_observation':'Trace terminal RGB remains in source; every observation with a real expert action is an SFT anchor.'}
        assert sum(histogram.values())==row['anchors']
        (args.output/f'{arm}_export_manifest.json').write_text(json.dumps(row,indent=2));rows.append(row)
    (args.output/'EXPORT_COMPLETE.json').write_text(json.dumps({'arms':rows,'same_budget':min(budgets),'SFT_started':False},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--paired-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    try:run(a)
    except Exception as e:
        if a.output.exists():(a.output/'EXPORT_FAILED.json').write_text(json.dumps({'error':repr(e)}))
        raise
