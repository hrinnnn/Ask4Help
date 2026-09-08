"""Stage-aware controller: nominal metric audit -> ID gate -> two stage pilots."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path


def run(args):
    args.root.mkdir(parents=True,exist_ok=True);args.logs.mkdir(parents=True,exist_ok=True)
    state={'controller_pid':os.getpid(),'stage':'nominal_reference','jobs':[],
           'next_stage':'nominal_audit_ID_gate_audit_two_stage_pairs_TASR','whole_pipeline_complete':False}
    def save():(args.root/'controller_state.json').write_text(json.dumps(state,indent=2))
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
             PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR='/tmp/pi05_timing_feedback_ablation_v1/mpl',
             NUMBA_CACHE_DIR='/tmp/pi05_timing_feedback_ablation_v1/numba')
    def wait_marker(marker,failures,stage):
        state['stage']=stage;save()
        while not marker.exists():
            failed=[str(p) for p in failures if p.exists()]
            if failed:raise RuntimeError('upstream failed: '+', '.join(failed))
            time.sleep(60)
    def start(stage,command,extra_env=None):
        logfile=args.logs/(stage+'.log')
        with logfile.open('w') as f:child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,
                                      env={**env,**(extra_env or {})},start_new_session=True)
        row={'stage':stage,'pid':child.pid,'command':command,'log':str(logfile)};state['jobs'].append(row);save()
        return child,row
    def finish(child,row):
        row['returncode']=child.wait();save()
        if row['returncode']:raise RuntimeError(f"{row['stage']} failed with {row['returncode']}")
    def command(script,*argv):return [sys.executable,str(args.code/'tools'/script),*map(str,argv)]
    try:
        wait_marker(args.bank/'NOMINAL_BANK_RECORDED.json',[args.bank/'NOMINAL_BANK_FAILED.json'],'waiting_nominal_bank')
        if not (args.bank/'NOMINAL_REFERENCE_AUDITED.json').exists():
            finish(*start('Goal_metric_calibration',command('score_opendrawer_goal_feedback_tasr.py','--bank',args.bank,'--calibrate')))
        for split in ['grasp','goal']:
            smoke=args.smokes/split
            finish(*start(split+'_oracle_audit',command('audit_opendrawer_feedback_smoke.py','--root',smoke)))
        cal=args.calibration or args.gate_root/'ID_calibration_native_v2'
        completion=cal/'CALIBRATION_COMPLETE.json' if args.calibration else args.gate_root/'REFERENCE_CALIBRATION_COMPLETE.json'
        failures=[cal/'CALIBRATION_FAILED.json']
        if args.calibration:failures.append(cal.parent/'ID_CALIBRATION_CONTROLLER_FAILED.json')
        wait_marker(completion,failures,'waiting_successful_ID_gate_calibration')
        finish(*start('ID_gate_audit',command('audit_pi05_feedback_calibration.py','--root',cal)))
        # Both H20 slots are reserved by this Goal; never occupy a new external job.
        state['stage']='waiting_reserved_GPUs';save()
        while True:
            output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
            pids={int(x.strip()) for x in output.splitlines() if x.strip().isdigit()}
            if pids<={276925}:break
            time.sleep(60)
        running=[];state['stage']='two_stage_paired_collection';save()
        for task,split,gpu,cpu,seed in [('open_drawer_grasp_ood','grasp','0','0-3',args.grasp_seed),
                                        ('open_drawer_goal_ood','goal','1','4-7',args.goal_seed)]:
            output=args.root/split
            if (output/'PAIRED_PILOT_COMPLETE.json').exists():continue
            cmd=command('run_pi05_feedback_pilot_stage.py','--code',args.code,'--root',output,
                        '--logs',args.logs/split,'--calibration',cal,'--task',task,'--seed',seed,
                        '--gpu',gpu,'--cpu',cpu,'--episodes',args.episodes,'--feedback-rule','displacement_v1',
                        '--support-mode',args.support_mode)
            running.append(start(split+'_paired',cmd,{'TMPDIR':f'/tmp/pi05_timing_feedback_ablation_v1/tmp{gpu}'}))
        for child,row in running:finish(child,row)
        state['stage']='two_stage_TASR';save()
        finish(*start('Grasp_TASR',command('score_opendrawer_feedback_tasr.py','--root',args.root/'grasp','--reference',args.grasp_reference)))
        finish(*start('Goal_TASR',command('score_opendrawer_goal_feedback_tasr.py','--root',args.root/'goal','--bank',args.bank)))
        state.update(stage='two_stage_timing_TASR_complete',next_stage='review_formal_collection_and_training_admission',finished=time.time());save()
        (args.root/'TIMING_TASR_PILOTS_COMPLETE.json').write_text(json.dumps(state,indent=2));return 0
    except Exception as error:
        state.update(stage='requires_review',error=repr(error));save()
        (args.root/'STAGE_CONTROLLER_FAILED.json').write_text(json.dumps(state,indent=2));raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['code','root','logs','bank','smokes','gate-root','grasp-reference']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--calibration',type=Path,help='Explicit successful-ID calibration extension root, preserving previous failed50 root')
    p.add_argument('--support-mode',choices=['hard_radius','soft_mass'],default='hard_radius')
    p.add_argument('--episodes',type=int,default=20)
    p.add_argument('--grasp-seed',type=int,default=1782000)
    p.add_argument('--goal-seed',type=int,default=1783000)
    raise SystemExit(run(p.parse_args()))
