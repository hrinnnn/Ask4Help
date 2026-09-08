"""Run the preregistered same-length hard-feedback control after soft pilots."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path


def run(a):
    a.root.mkdir(parents=True,exist_ok=True)
    state={'controller_pid':os.getpid(),'stage':'waiting_soft_completion','soft_root':str(a.soft_root),
           'next_stage':'same_seed_hard40_then_common_three_arm_TASR','whole_pipeline_complete':False}
    def save():(a.root/'hard_control_state.json').write_text(json.dumps(state,indent=2))
    save()
    while not (a.soft_root/'TIMING_TASR_PILOTS_COMPLETE.json').exists():
        if (a.soft_root/'STAGE_CONTROLLER_FAILED.json').exists():
            state.update(stage='upstream_requires_repair');save();return 1
        time.sleep(60)
    cmd=[sys.executable,str(a.code/'tools/run_opendrawer_stage_feedback_pilots.py'),
         '--code',str(a.code),'--root',str(a.root),'--logs',str(a.logs),
         '--bank',str(a.bank),'--smokes',str(a.smokes),'--gate-root',str(a.gate_root),
         '--calibration',str(a.calibration),'--grasp-reference',str(a.grasp_reference),
         '--grasp-seed','1784000','--goal-seed','1785000','--episodes','40',
         '--support-mode','hard_radius','--shared-fixed-root',str(a.soft_root)]
    state.update(stage='hard40_collection',command=cmd);save()
    p=subprocess.Popen(cmd,start_new_session=True);state['child_pid']=p.pid;save()
    rc=p.wait();state['returncode']=rc;save()
    if rc:state.update(stage='hard40_requires_repair');save();return rc
    cmd=[sys.executable,str(a.code/'tools/compare_opendrawer_feedback_three_arms.py'),
         '--soft-root',str(a.soft_root),'--hard-root',str(a.root)]
    subprocess.run(cmd,check=True)
    state.update(stage='three_arm_timing_TASR_complete',next_stage='training_admission_and_frozen_version_validation');save()
    (a.root/'THREE_ARM_PILOTS_COMPLETE.json').write_text(json.dumps(state,indent=2));return 0


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['code','root','soft-root','logs','bank','smokes','gate-root','calibration','grasp-reference']:
        p.add_argument('--'+name,type=Path,required=True)
    raise SystemExit(run(p.parse_args()))
