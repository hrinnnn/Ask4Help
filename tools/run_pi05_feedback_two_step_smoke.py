"""Bounded engineering-only two-step SFT after the reserved GPU is free."""
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path


def run(a):
    a.output.mkdir(parents=True,exist_ok=False)
    plan=json.loads(a.plan.read_text())
    assert plan['status']=='PREPARED_NOT_LAUNCHED' and plan['updates']==2
    state={'controller_pid':os.getpid(),'stage':'waiting_stackcube_pilot','scope':'engineering_only_not_formal_SFT',
           'plan':str(a.plan),'whole_pipeline_complete':False}
    def save():(a.output/'controller_state.json').write_text(json.dumps(state,indent=2))
    save()
    while not a.wait_for.exists():
        if (a.wait_for.parent/'pipeline_state.json').exists():
            p=json.loads((a.wait_for.parent/'pipeline_state.json').read_text())
            if p.get('stage') in ['collector_requires_repair','audit_failed']:
                state.update(stage='upstream_requires_review');save();return 1
        time.sleep(60)
    env={**os.environ,**plan['environment']}
    gpu=env['FEEDBACK_ASSIGNED_GPU']
    uuid=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
    assert not [line for line in active.splitlines() if uuid in line and line.split(',')[0].strip()!='276925'],'Assigned GPU is occupied'
    scratch=Path(env['FEEDBACK_SFT_OUTPUT'])
    assert scratch.parent==Path('/dev/shm') and scratch.name.startswith('pi05_feedback_sft_smoke_')
    assert not scratch.exists(),'Use a new engineering retry directory'
    assert shutil.disk_usage('/dev/shm').free>64*1024**3
    for name in ['TMPDIR','RAY_TMPDIR']:Path(env[name]).mkdir(parents=True,exist_ok=True)
    log=a.output/'training.log'
    with log.open('w') as stream:p=subprocess.Popen(plan['command'],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env=env)
    state.update(stage='two_step_training',training_pid=p.pid,command=plan['command'],log=str(log));save()
    code=p.wait();state['returncode']=code;save()
    if code:state.update(stage='training_engineering_failure');save();return code
    checkpoint=Path(plan['checkpoint_expected'])
    assert (checkpoint/'actor/model_state_dict/full_weights.pt').exists(),'Expected two-step checkpoint missing'
    state['stage']='reload_forward';save()
    cmd=[sys.executable,str(a.code/'tools/reload_pi05_feedback_sft_smoke.py'),'--manifest',str(a.code/'configs/pipelines/pi05_timing_feedback_ablation_v1.json'),
         '--task',plan['task'],'--checkpoint',str(checkpoint),'--output',str(a.output/'reload')]
    with (a.output/'reload.log').open('w') as stream:
        p=subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    state['reload_pid']=p.pid;save()
    code=p.wait();state['reload_returncode']=code;save()
    if code:state.update(stage='reload_engineering_failure');save();return code
    state['stage']='persist_checkpoint';save()
    shutil.copytree(scratch,a.output/'training_artifacts')
    state.update(stage='two_step_smoke_finished_audit_pending',scratch_preserved=str(scratch));save()
    (a.output/'TWO_STEP_PROCESS_AND_RELOAD_COMPLETE.json').write_text(json.dumps(state,indent=2))
    return 0


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['code','plan','output','wait-for']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    try:raise SystemExit(run(a))
    except Exception as error:
        if a.output.exists():(a.output/'SMOKE_FAILED.json').write_text(json.dumps({'error':repr(error)}))
        raise
