"""Fixed fresh nominal Goal-OOD expert bank; never used by the online gate."""
import argparse,io,json,os,time
from pathlib import Path
import numpy as np
from pi05_feedback_runtime import Pi05FeedbackRuntime
from collect_pi05_feedback import write_json,write_trace,planner_diagnostic


def goal_blocks(report,n):
    handle=0
    if not report['drawer_opened_before_takeover']:
        handle=sum(report[k] for k in ['direct_handle_pregrasp_steps','direct_handle_reach_steps','direct_pull_steps'])
        handle+=4 if report['direct_pull_from_existing_handle_grasp'] else 14
    grasp=0 if report['object_grasped_before_takeover'] else 2+sum(report[k] for k in ['direct_object_pregrasp_steps','direct_object_reach_steps','direct_object_close_steps'])
    start=handle+grasp+report['direct_lift_steps']
    transport_end=start+report['direct_transport_steps'];place_end=transport_end+report['direct_place_steps']
    assert 0<=start<=transport_end<=place_end<n and 1<=n-place_end<=4
    return {'transport':[start,transport_end],'place':[transport_end,place_end],'release':[place_end,n]}


def run(args):
    args.root.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(args.manifest.read_text());task='open_drawer_goal_ood'
    runtime=Pi05FeedbackRuntime(task,manifest['task_assets'][task]);runtime.torch.set_num_threads(4)
    write_json(args.root/'provenance.json',{'runtime':runtime.provenance(),'VLA_loaded':False,
               'seed_start':1786000,'raw_attempts':30,'purpose':'evaluation-only nominal Goal reference, not gate calibration'})
    rows=[];start=time.time()
    for i,seed in enumerate(range(1786000,1786030)):
        env=runtime.build_env('ood');raw,_=env.reset(seed=seed)
        reset=runtime.reset_metadata(env,split='ood');initial=runtime.snapshot(env,raw)
        result=runtime.opendrawer_expert(env,raw,seed,400)
        snapshots=[initial]+result['snapshots'];actions=result['actions'];n=len(actions)
        assert n<=400 and len(snapshots)==n+1
        directory=args.root/f'episode_{i:04d}';directory.mkdir()
        write_trace(directory,snapshots,actions,[],result)
        accepted=bool(result['report']['accepted'])
        partition='reference' if seed%5<3 else ('calibration' if seed%5==3 else 'checking')
        row={'episode':i,'seed':seed,'partition':partition,'accepted':accepted,'expert_actions':n,
             'reset':reset,'expert_report':planner_diagnostic(result['report'])}
        if accepted:
            blocks=goal_blocks(result['report'],n);row['blocks']=blocks
            a=blocks['transport'][0]
            assert snapshots[a]['grasped'] and snapshots[a]['ever_lifted']
            values={'position':np.asarray([s['tcp']-s['target_p'] for s in snapshots]),
                    'quaternion':np.asarray([s['tcp_q'] for s in snapshots]),
                    'width':np.asarray([s['qpos'][-2:].sum() for s in snapshots])[:,None],
                    'contact':np.asarray([s['grasped'] for s in snapshots])}
            buf=io.BytesIO();np.savez_compressed(buf,**values);(directory/'comparison_pose.npz').write_bytes(buf.getvalue())
        write_json(directory/'result.json',row);rows.append(row);env.close()
        progress={'pid':os.getpid(),'raw_attempts':len(rows),'accepted':sum(r['accepted'] for r in rows),
                  'elapsed':time.time()-start,'last_seed':seed,'last_success':accepted}
        write_json(args.root/'progress.json',progress);print(json.dumps(progress),flush=True)
    counts={p:sum(r['accepted'] and r['partition']==p for r in rows) for p in ['reference','calibration','checking']}
    write_json(args.root/'summary.json',{'rows':rows,'accepted_partition_counts':counts,'used_for_online_gate':False})
    qualified=counts['reference']>=8 and counts['calibration']>=3 and counts['checking']>=3
    write_json(args.root/'NOMINAL_BANK_RECORDED.json',{'counts':counts,'enough_for_scoring_calibration':qualified,
               'next_stage':'independent_raw_visual_audit_then_metric_calibration','whole_pipeline_complete':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    args=p.parse_args()
    try:run(args)
    except Exception as e:
        if args.root.exists():write_json(args.root/'NOMINAL_BANK_FAILED.json',{'error':repr(e)})
        raise
