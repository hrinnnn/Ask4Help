"""Visual/current-state oracle audit before any OpenDrawer paired collection."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
from pi05_feedback_runtime import Pi05FeedbackRuntime
from collect_pi05_feedback import write_json, write_trace, planner_diagnostic, jsonable


def main(args):
    args.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(args.manifest.read_text())
    runtime=Pi05FeedbackRuntime(args.task,manifest['task_assets'][args.task])
    runtime.torch.set_num_threads(4);runtime.load_model()
    write_json(args.output/'provenance.json',runtime.provenance())
    rows=[];start=time.time()
    for requested in [0,120]:
        for split in ['id','ood']:
            seed=args.seed+requested
            env=runtime.build_env(split);raw,_=env.reset(seed=seed)
            snapshots=[runtime.snapshot(env,raw)];actions=[];queries=[];ended=False
            while len(actions)<requested and not ended:
                step=len(actions);pred,_=runtime.predict(raw,seed*1000+step)
                for action in runtime.clip(pred[:min(5,requested-step)],env):
                    raw,_,term,trunc,info=env.step(runtime.torch.as_tensor(action,device=env.unwrapped.device).reshape(1,-1))
                    actions.append(action);snapshots.append(runtime.snapshot(env,raw))
                    ended=bool(term) or bool(trunc) or bool(info['success'])
                    if ended:break
            takeover=len(actions);initial=snapshots[-1]
            metadata=runtime.reset_metadata(env,split=split)
            result={'actions':[],'snapshots':[],'all_actions':[],'all_snapshots':[],
                    'report':{'accepted':False,'reason':'policy_ended_before_takeover'}}
            if not ended:
                result=runtime.opendrawer_expert(env,raw,seed,runtime.horizon-takeover)
                actions.extend(result['actions']);snapshots.extend(result['snapshots'])
            assert len(actions)<=400 and len(snapshots)==len(actions)+1
            assert len(result['actions'])==len(result['all_actions'])
            assert np.isfinite(np.asarray(actions)).all()
            directory=args.output/f'episode_{len(rows):04d}';directory.mkdir()
            write_trace(directory,snapshots,actions,queries,result)
            row={'split':split,'seed':seed,'scheduled_takeover':requested,'takeover':takeover,
                 'policy_ended_before_takeover':ended,'expert_actions':len(result['actions']),
                 'strict_success':bool(snapshots[-1]['success']),
                 'drawer_opened_at_takeover':initial['ever_drawer_opened'],
                 'object_grasped_at_takeover':initial['grasped'],
                 'final_ever_grasped':snapshots[-1]['ever_grasped'],
                 'reset':metadata,'expert_report':planner_diagnostic(result['report'])}
            write_json(directory/'result.json',row);rows.append(row);env.close()
            progress={'pid':os.getpid(),'episodes':len(rows),'elapsed':time.time()-start,'last':row}
            write_json(args.output/'progress.json',progress);print(json.dumps(jsonable(progress)),flush=True)
    write_json(args.output/'summary.json',{'rows':rows,'scope':'oracle smoke, not gate efficacy or SFT'})
    write_json(args.output/'ORACLE_SMOKE_RECORDED.json',{'episodes':len(rows),'visual_audit_pending':True})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--task',choices=['open_drawer_grasp_ood','open_drawer_goal_ood'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=1781100)
    args=p.parse_args()
    try:main(args)
    except Exception as e:
        if args.output.exists():write_json(args.output/'ORACLE_SMOKE_FAILED.json',{'error':repr(e)})
        raise
