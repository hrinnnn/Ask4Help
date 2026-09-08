"""Reserved-seed unassisted evaluation; no gate, expert, adaptation, or training."""
import argparse,gc,json,os,time
from pathlib import Path
import numpy as np
from pi05_feedback_runtime import Pi05FeedbackRuntime
from pi05_feedback_artifacts import completed_rows
from collect_pi05_feedback import write_json,write_trace


def run(a):
    a.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(a.manifest.read_text());protocol=json.loads(a.protocol.read_text())
    specification=protocol['tasks'][a.task]
    seeds=list(range(specification[a.split+'_start'],specification[a.split+'_start']+protocol['episodes_per_split']))
    assets={**manifest['task_assets'][a.task],'checkpoint':str(a.checkpoint)}
    runtime=Pi05FeedbackRuntime(a.task,assets);runtime.inference_mode='eval';runtime.torch.set_num_threads(4)
    provenance={'purpose':'reserved_final_evaluation','runtime':runtime.provenance(),'task':a.task,'split':a.split,
                'protocol':str(a.protocol),'seeds':seeds,'primary_endpoint':specification['primary_endpoint'],
                'base_checkpoint':manifest['task_assets'][a.task]['checkpoint'],'expert_or_gate_used':False}
    rows=[]
    if a.resume_from:
        old=json.loads((a.resume_from/'provenance.json').read_text())
        for key in ['task','split','seeds','primary_endpoint','expert_or_gate_used']:assert old[key]==provenance[key]
        assert old['runtime']['checkpoint']==str(a.checkpoint) and old['runtime']['inference_mode']=='eval'
        rows=completed_rows(a.resume_from);provenance['resume_from']=str(a.resume_from)
    write_json(a.output/'provenance.json',provenance)
    runtime.load_model();started=time.time()
    for episode in range(len(rows),min(len(seeds),len(rows)+a.max_new_episodes)):
        seed=seeds[episode];env=runtime.build_env(a.split);raw,_=env.reset(seed=seed)
        reset=runtime.reset_metadata(env,split=a.split)
        strict=bool(env.unwrapped.evaluate()['success'])
        first=runtime.snapshot(env,raw);first['eval_strict_success']=strict
        snapshots=[first];actions=[];queries=[];ended=strict
        while len(actions)<runtime.horizon and not ended:
            step=len(actions);prediction,_=runtime.predict(raw,seed*1000+step);queries.append(step)
            for action in runtime.clip(prediction[:5],env):
                raw,_,terminated,truncated,info=env.step(runtime.torch.as_tensor(action,device=env.unwrapped.device).reshape(1,-1))
                strict=bool(info['success']);snapshot=runtime.snapshot(env,raw);snapshot['eval_strict_success']=strict
                actions.append(action);snapshots.append(snapshot)
                ended=strict or bool(terminated) or bool(truncated)
                if ended:break
        grasped=any(s['grasped'] for s in snapshots)
        directory=a.output/('episode_%04d'%episode);directory.mkdir()
        write_trace(directory,snapshots,actions,[],{})
        row={'episode':episode,'seed':seed,'split':a.split,'strict_success':strict,'ever_grasped':grasped,
             'success':grasped if specification['primary_endpoint']=='ever_grasped' else strict,
             'actions':len(actions),'policy_query_steps':queries,'reset':reset,'artifact_directory':str(directory),
             'expert_actions':0,'takeover':None}
        write_json(directory/'result.json',row);rows.append(row);env.close()
        del snapshots,actions,env,raw
        gc.collect()
        progress={'pid':os.getpid(),'episodes':len(rows),'successes':sum(r['success'] for r in rows),
                  'strict_successes':sum(r['strict_success'] for r in rows),'elapsed':time.time()-started}
        write_json(a.output/'progress.json',progress);print(json.dumps(progress),flush=True)
    summary={'task':a.task,'split':a.split,'episodes':len(rows),'successes':sum(r['success'] for r in rows),
             'strict_successes':sum(r['strict_success'] for r in rows),'ever_grasped_successes':sum(r['ever_grasped'] for r in rows),
             'primary_endpoint':specification['primary_endpoint'],'checkpoint':str(a.checkpoint),'rows':rows,
             'expert_or_gate_used':False,'complete':len(rows)==len(seeds)}
    write_json(a.output/'summary.json',summary)
    marker='EVALUATION_COMPLETE.json' if summary['complete'] else 'EVALUATION_CHUNK_COMPLETE.json'
    write_json(a.output/marker,{'episodes':len(rows),'complete':summary['complete'],'checkpoint':str(a.checkpoint)})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['manifest','protocol','checkpoint','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--task',required=True);p.add_argument('--split',choices=['id','ood'],required=True)
    p.add_argument('--resume-from',type=Path);p.add_argument('--max-new-episodes',type=int,default=20);a=p.parse_args()
    try:run(a)
    except Exception as error:
        if a.output.exists():write_json(a.output/'EVALUATION_FAILED.json',{'error':repr(error)})
        raise
