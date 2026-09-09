"""Execute the frozen two-task replication after the current autonomy diagnostic."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path


def run(a):
    a.root.mkdir(parents=True,exist_ok=True);a.logs.mkdir(parents=True,exist_ok=True)
    plan=json.loads((a.code/'configs/pipelines/pi05_commitment_replication_v1.json').read_text())
    manifest=json.loads((a.code/'configs/pipelines/pi05_timing_feedback_ablation_v1.json').read_text())
    state={'controller_pid':os.getpid(),'stage':'waiting_autonomy_probe','jobs':[],'whole_pipeline_complete':False}
    def save():(a.root/'controller_state.json').write_text(json.dumps(state,indent=2))
    save()
    while not (a.upstream/'AUTONOMY_PROBE_COMPLETE.json').exists():
        p=a.upstream/'controller_state.json'
        if p.exists() and json.loads(p.read_text()).get('stage') in ['probe_requires_review','upstream_requires_review','reserved_GPUs_not_free']:
            state.update(stage='upstream_requires_review');save();return 1
        time.sleep(60)
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
    for t in plan['task_settings']:
        assets=manifest['task_assets'][t['task']]
        for p in [assets['checkpoint'],assets['norm'],t['opening_calibration'],str(Path(t['calibration'])/'INDEPENDENT_CALIBRATION_AUDIT.json')]:
            assert Path(p).exists(),p
    state['stage']='waiting_reserved_GPUs';save()
    while True:
        output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
        if {int(x) for x in output.splitlines() if x.strip().isdigit()}<={276925}:break
        time.sleep(60)
    jobs=[];state['stage']='paired_commitment_replication';save()
    for t in plan['task_settings']:
        cmd=[sys.executable,str(a.code/'tools/run_pi05_feedback_pilot_stage.py'),'--code',str(a.code),
             '--root',str(a.root/t['short_name']),'--logs',str(a.logs/t['short_name']),'--calibration',t['calibration'],
             '--task',t['task'],'--seed',str(t['seed']),'--gpu',t['gpu'],'--cpu',t['cpu'],
             '--episodes',str(plan['raw_episodes_per_arm']),'--feedback-rule',plan['feedback_rule'],
             '--opening-calibration',t['opening_calibration'],'--support-mode',plan['support_mode']]
        log=a.logs/(t['short_name']+'_controller.log')
        with log.open('w') as f:p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,
                              env={**env,'TMPDIR':f"/tmp/pi05_timing_feedback_ablation_v1/tmp{t['gpu']}"})
        row={'task':t['task'],'pid':p.pid,'command':cmd,'log':str(log)};state['jobs'].append(row);save();jobs.append((p,row))
    for p,row in jobs:
        row['returncode']=p.wait();save()
        if row['returncode']:state.update(stage='replication_requires_review');save();return 1
    state['stage']='TASR';save()
    for t in plan['task_settings']:
        subprocess.run([sys.executable,str(a.code/'tools/score_pi05_feedback_tasr.py'),'--task',t['short_name'],
                        '--reference',str(a.reference),'--root',str(a.root/t['short_name'])],check=True,env=env)
    state.update(stage='replication_timing_TASR_complete',next_stage='training_admission_and_policy_update');save()
    (a.root/'COMMITMENT_REPLICATION_COMPLETE.json').write_text(json.dumps(state,indent=2));return 0


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['code','root','logs','upstream','reference']:p.add_argument('--'+name,type=Path,required=True)
    raise SystemExit(run(p.parse_args()))
