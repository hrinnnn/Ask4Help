"""Real single-takeover pi0.5 fixed-vs-feedback expert data collection."""
import argparse
import io
import json
import os
import time
import gc
from pathlib import Path

import numpy as np

from pi05_feedback_runtime import Pi05FeedbackRuntime
from pi05_timing_feedback import TimingFeedbackGate, action_block_error, corrective_motion, timing_cue, isolated_python_numpy_rng
from pi05_timing_feedback import gripper_commitment_opening, commitment_timing_cue
from pi05_timing_feedback import observed_agreement_steps
from pi05_feedback_artifacts import completed_rows,restore_memory


def jsonable(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:jsonable(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [jsonable(x) for x in v]
    return v


def write_json(path, value):
    path.write_text(json.dumps(jsonable(value),indent=2,allow_nan=False))


def planner_diagnostic(value):
    """Preserve a planner's explicit unreachable-distance sentinel as text.

    This conversion is limited to planner diagnostics, never policy signals,
    actions, feedback errors or outcome measurements.
    """
    if isinstance(value,(float,np.floating)) and not np.isfinite(value):return f'nonfinite:{value}'
    if isinstance(value,dict):return {k:planner_diagnostic(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [planner_diagnostic(v) for v in value]
    return value


def write_trace(directory, snapshots, actions, prefix, expert_all):
    arrays={k:np.asarray([s[k] for s in snapshots]) for k in snapshots[0]}
    arrays.update(actions=np.asarray(actions,dtype=np.float32).reshape(-1,8),
                  query_features=np.asarray([q['feature'] for q in prefix]),
                  query_steps=np.asarray([q['step'] for q in prefix]),
                  all_expert_actions=np.asarray(expert_all.get('all_actions',[]),dtype=np.float32).reshape(-1,8))
    for key in ['qpos','tcp','tcp_q','object_p','object_q','grasped']:
        arrays['all_expert_'+key]=np.asarray([s[key] for s in expert_all.get('all_snapshots',[])])
    buffer=io.BytesIO();np.savez_compressed(buffer,**arrays)
    (directory/'trace.npz').write_bytes(buffer.getvalue())
    import cv2
    scratch=Path(os.environ.get('TMPDIR','/tmp'))/f'feedback_{os.getpid()}_{directory.name}.mp4'
    h,w=snapshots[0]['main'].shape[:2]
    writer=cv2.VideoWriter(str(scratch),cv2.VideoWriter_fourcc(*'mp4v'),10,(2*w,h))
    if not writer.isOpened():raise RuntimeError('video writer unavailable')
    for s in (snapshots[:-1] or snapshots[:1]):writer.write(np.concatenate([s['main'],s['wrist']],axis=1)[:,:,::-1])
    writer.release();(directory/'trajectory.mp4').write_bytes(scratch.read_bytes());scratch.unlink()


def collect_episode(runtime, gate, calibration, mean, basis, seed, split, episode, force_step=None,
                    feedback_rule='displacement_v1',opening_reference=None):
    env=runtime.build_env(split);raw,info=env.reset(seed=seed)
    snapshots=[runtime.snapshot(env,raw)];actions=[];prefix=[];queries=[]
    metadata=runtime.reset_metadata(env,split=split);gate.begin_episode(episode)
    takeover=None;success=False;ended=False;expert_all={};expert_report=None
    while len(actions)<runtime.horizon and not ended:
        step=len(actions);prediction,latent=runtime.predict(raw,seed*1000+step)
        feature=runtime.bridge(raw,latent);v=feature-mean
        score=float(np.linalg.norm(v-(v@basis)@basis.T))
        decision=gate.query(feature,score,step)
        trigger=decision['stop'] if force_step is None else step>=force_step
        queries.append({'step':step,'score':score,**decision,'actual_trigger':bool(trigger),'memory_episodes':len(gate.memory)})
        prefix.append({'step':step,'feature':feature})
        if trigger:takeover=step;break
        for action in runtime.clip(prediction[:5],env):
            raw,_,terminated,truncated,info=env.step(runtime.torch.as_tensor(action,device=env.unwrapped.device).reshape(1,-1))
            actions.append(action);snapshots.append(runtime.snapshot(env,raw))
            success=bool(info['success']);ended=success or bool(terminated) or bool(truncated)
            if ended:break
    if takeover is not None:
        if runtime.task=='stackcube_legacy_ood':
            from rlinf.envs.maniskill.stack_cube_privileged_oracle import StackCubePrivilegedChunkOracle
            oracle=StackCubePrivilegedChunkOracle(chunk_size=5);phases=[]
            with isolated_python_numpy_rng(seed+600000):
                while len(actions)<runtime.horizon and not ended:
                    plan=oracle.plan(env)
                    phases.append({'step':len(actions),'phase':plan.phase,'planning_succeeded':plan.planning_succeeded})
                    for j in range(len(plan.actions)):
                        action=runtime.clip(np.asarray(plan.action_at(raw['agent']['qpos'],j))[None],env)[0]
                        raw,_,terminated,truncated,info=env.step(runtime.torch.as_tensor(action,device=env.unwrapped.device).reshape(1,-1))
                        actions.append(action);snapshots.append(runtime.snapshot(env,raw))
                        success=bool(info['success']);ended=success or bool(terminated) or bool(truncated)
                        if ended:break
            expert_all={'all_actions':actions[takeover:],'all_snapshots':snapshots[takeover+1:]}
            expert_report={'phases':phases,'accepted':success,'attempt_lengths':[len(actions)-takeover]}
        else:
            expert=runtime.opendrawer_expert if runtime.task.startswith('open_drawer_') else runtime.airplane_expert
            result=expert(env,raw,seed,runtime.horizon-takeover)
            actions.extend(result['actions']);snapshots.extend(result['snapshots'])
            expert_all=result;expert_report=result['report'];expert_report['attempt_lengths']=result['attempt_lengths']
            success=bool(result['report']['accepted'])
    errors=[];reversal=None;cue=None;feedback_seconds=0.;opening_score=0.;later_blocks=[]
    expert_n=0 if takeover is None else len(actions)-takeover
    if takeover is not None:
        begin=time.time()
        if takeover>=5 and expert_n>=5:
            before,current,after=snapshots[takeover-5],snapshots[takeover],snapshots[takeover+5]
            reversal=corrective_motion(before['tcp'],current['tcp'],after['tcp'],before['qpos'][-2:].sum(),current['qpos'][-2:].sum(),after['qpos'][-2:].sum())
            opening_score=gripper_commitment_opening(np.asarray(actions[takeover-5:takeover])[:,-1],
                np.asarray(actions[takeover:takeover+5])[:,-1],current['qpos'][-2:].sum(),after['qpos'][-2:].sum())
        # A directional cue only needs to distinguish crossing within one
        # block versus a completely observed initial low-error block.
        for k in range(min(5,max(0,expert_n-4))):
            target=np.asarray(actions[takeover+k:takeover+k+5])
            raw_expert=runtime.raw_snapshot(snapshots[takeover+k]);samples=[]
            for m in range(2):
                pred,_=runtime.predict(raw_expert,280009+seed*10+k*2+m)
                samples.append(runtime.clip(pred,env))
            errors.append(action_block_error(np.asarray(samples),target))
            if errors[-1]>calibration['error_reference']:break
        arguments=dict(reversal=reversal,reversal_reference=calibration['reversal_reference'],has_previous_query=len(prefix)>=2)
        if feedback_rule=='commitment_v2':
            cue=commitment_timing_cue(errors,calibration['error_reference'],opening_score=opening_score,
                                      opening_reference=opening_reference,**arguments)
        else:cue=timing_cue(errors,calibration['error_reference'],**arguments)
        if cue is not None and cue['direction']==-1 and gate.use_later_duration:
            for offset in range(0,expert_n-4,5):
                target=np.asarray(actions[takeover+offset:takeover+offset+5])
                samples=[]
                for m in range(2):
                    pred,_=runtime.predict(runtime.raw_snapshot(snapshots[takeover+offset]),280009+seed*10+offset*2+m)
                    samples.append(runtime.clip(pred,env)[:5])
                pred=np.asarray(samples)
                record={'offset':offset,'overall_MSE':float(((pred-target[None])**2).mean()),
                        'gripper_sign_disagreement':float((np.sign(pred[:,:,7])!=np.sign(target[None,:,7])).mean()),
                        'predictions':pred.tolist(),'expert_actions':target.tolist()}
                later_blocks.append(record)
                if record['overall_MSE']>calibration['error_reference'] or record['gripper_sign_disagreement']>0:break
            cue={**cue,'later_valid_steps':observed_agreement_steps(later_blocks,calibration['error_reference'])}
        if cue is not None:
            credit=prefix[-2] if cue['attribution']=='previous_query' else prefix[-1]
            cue={**cue,'credit_step':credit['step']}
            gate.commit(cue,credit['feature'],completed_episode=episode)
        feedback_seconds=time.time()-begin
    ever_grasped=any(s['grasped'] for s in snapshots)
    env.close()
    summary={'episode':episode,'seed':seed,'split':split,'takeover':takeover,
             'strict_success':success,'ever_grasped':ever_grasped,'accepted':bool(success and expert_n>0),
             'expert_suffix_actions':expert_n,'all_executed_expert_actions':len(expert_all.get('all_actions',[])),
             'total_retained_path_actions':len(actions),'query_count':len(queries),'queries':queries,
             'reset':metadata,'feedback_errors':errors,'motion_reversal':reversal,'cue':cue,
             'feedback_seconds':feedback_seconds,'expert_report':planner_diagnostic(expert_report),
             'feedback_rule':feedback_rule,'opening_score_m':opening_score,
             'later_blocks':later_blocks,
             'forced_takeover_diagnostic':force_step is not None,
             'video_scope':'policy prefix plus final expert candidate; abandoned planner candidates counted separately'}
    return summary,snapshots,actions,prefix,expert_all


def main(args):
    args.output.mkdir(parents=True,exist_ok=False);start=time.time()
    manifest=json.loads(args.manifest.read_text());cal=json.loads((args.calibration/'calibration.json').read_text())
    if args.task.startswith('open_drawer_'):
        assert cal['provenance'].get('inference_mode')=='eval','OpenDrawer requires original eval-mode calibration, not earlier generic SDE diagnostics'
    assert cal['task']==args.task or (
        cal['task'].startswith('open_drawer_') and args.task.startswith('open_drawer_')
        and manifest['task_assets'][cal['task']]['checkpoint']==manifest['task_assets'][args.task]['checkpoint']
        and manifest['task_assets'][cal['task']]['norm']==manifest['task_assets'][args.task]['norm'])
    arrays=np.load(args.calibration/'gate_arrays.npz');mean=arrays['mean'];basis=arrays['basis']
    cfg={**manifest['feedback'],'radius_multiplier':manifest['local_radius_multiplier'],'rule':args.feedback_rule,
         'support_mode':args.support_mode}
    opening_reference=None
    if args.feedback_rule=='commitment_v2':
        if args.opening_calibration is None:raise ValueError('commitment_v2 requires its frozen ID opening calibration')
        opening=json.loads(args.opening_calibration.read_text())
        assert opening['ID_calibration']==str(args.calibration)
        opening_reference=opening['opening_reference_m'];cfg['opening_reference_m']=opening_reference
        cfg['opening_calibration']=str(args.opening_calibration)
    gate=TimingFeedbackGate(cal['baseline_threshold'],arrays['center'],cal['scale'],cal['radius']*cfg['radius_multiplier'],
                            regularization=cfg['lambda'],strength=cfg['beta'],min_support=cfg['minimum_interventions'],
                            min_vote=cfg['minimum_absolute_vote'],block=cfg['execution_block'],enabled=args.arm=='feedback',
                            support_mode=args.support_mode,max_wait_blocks=cfg.get('max_wait_blocks',1),
                            remember_deferred_alarm=cfg.get('remember_deferred_alarm',False),
                            use_later_duration=cfg.get('use_later_duration',False))
    runtime=Pi05FeedbackRuntime(args.task,manifest['task_assets'][args.task]);runtime.torch.set_num_threads(4)
    provenance={'runtime':runtime.provenance(),'calibration':str(args.calibration),'arm':args.arm,'seed':args.seed,'episodes':args.episodes,'accepted_target':args.accepted_target,'feedback':cfg}
    results=[]
    if args.resume_from is not None:
        old=json.loads((args.resume_from/'provenance.json').read_text())
        for key in ['calibration','arm','seed','episodes','accepted_target','feedback']:
            assert old[key]==provenance[key],('resume contract changed',key)
        for key in ['task','checkpoint','norm','inference_mode']:
            assert old['runtime'].get(key)==provenance['runtime'].get(key)
        results=completed_rows(args.resume_from)
        restore_memory(gate,results)
        provenance['resume_from']=str(args.resume_from)
        provenance['resumed_completed_episodes']=len(results)
    write_json(args.output/'provenance.json',provenance)
    runtime.load_model();accepted=sum(r['accepted'] for r in results)
    stop=min(args.episodes,len(results)+args.max_new_episodes)
    for episode in range(len(results),stop):
        split='id' if episode%2==0 else 'ood';seed=args.seed+episode//2
        result,snapshots,actions,prefix,expert_all=collect_episode(runtime,gate,cal,mean,basis,seed,split,episode,args.force_takeover_step,
                                                                args.feedback_rule,opening_reference)
        directory=args.output/f'episode_{episode:04d}';directory.mkdir()
        result['artifact_directory']=str(directory)
        write_trace(directory,snapshots,actions,prefix,expert_all);write_json(directory/'result.json',result)
        accepted+=int(result['accepted']);results.append(result)
        row={'pid':os.getpid(),'arm':args.arm,'task':args.task,'episodes':len(results),'accepted':accepted,
             'elapsed':time.time()-start,'last_takeover':result['takeover'],'last_cue':result['cue'],
             'all_expert_actions':sum(r['all_executed_expert_actions'] for r in results),
             'open_file_descriptors':len(os.listdir('/proc/self/fd'))}
        write_json(args.output/'progress.json',row);print(json.dumps(row),flush=True)
        del snapshots,actions,prefix,expert_all
        gc.collect()
        if args.accepted_target is not None and accepted>=args.accepted_target:break
    summary={'task':args.task,'arm':args.arm,'episodes':len(results),'accepted':accepted,
             'accepted_ID':sum(r['accepted'] and r['split']=='id' for r in results),
             'accepted_OOD':sum(r['accepted'] and r['split']=='ood' for r in results),
             'all_expert_actions':sum(r['all_executed_expert_actions'] for r in results),
             'successful_expert_actions':sum(r['expert_suffix_actions'] for r in results if r['accepted']),
             'rows':results,'whole_pipeline_complete':False}
    write_json(args.output/'summary.json',summary)
    complete=len(results)>=args.episodes or (args.accepted_target is not None and accepted>=args.accepted_target)
    marker='COLLECTION_COMPLETE.json' if complete else 'COLLECTION_CHUNK_COMPLETE.json'
    write_json(args.output/marker,{'episodes':len(results),'accepted':accepted,'target_reached':complete})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['stackcube_legacy_ood','airplane_yaw_ood','open_drawer_grasp_ood','open_drawer_goal_ood'],required=True)
    p.add_argument('--arm',choices=['fixed','feedback'],required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--episodes',type=int,default=20)
    p.add_argument('--accepted-target',type=int);p.add_argument('--force-takeover-step',type=int)
    p.add_argument('--feedback-rule',choices=['displacement_v1','commitment_v2'],default='displacement_v1')
    p.add_argument('--opening-calibration',type=Path)
    p.add_argument('--support-mode',choices=['hard_radius','soft_mass','continuous'],default='hard_radius')
    p.add_argument('--resume-from',type=Path)
    p.add_argument('--max-new-episodes',type=int,default=20)
    args=p.parse_args()
    try:main(args)
    except Exception as error:
        if args.output.exists():write_json(args.output/'COLLECTION_FAILED.json',{'type':type(error).__name__,'message':str(error)})
        raise
