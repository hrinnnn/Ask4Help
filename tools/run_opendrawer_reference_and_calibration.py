"""Resume-safe original-ID reference rebuilding followed by ID-only calibration."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path


def run(args):
    args.root.mkdir(parents=True,exist_ok=True);args.logs.mkdir(parents=True,exist_ok=True)
    state={'controller_pid':os.getpid(),'stage':'initial_restored_ID_shards','next_stage':'full_original_ID_reference_then_ID_calibration','jobs':[],'whole_pipeline_complete':False}
    def save():(args.root/'controller_state.json').write_text(json.dumps(state,indent=2))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    reference=args.root/'native_reference_v1';manifest=args.code/'configs/pipelines/pi05_timing_feedback_ablation_v1.json'
    base=['taskset','-c','0-3',sys.executable,'-u',str(args.code/'tools/build_opendrawer_native_reference.py'),'--manifest',str(manifest),'--output',str(reference)]
    def child(stage,command):
        state['stage']=stage;log=args.logs/(stage+'.log')
        with log.open('w') as stream:p=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env=env)
        row={'stage':stage,'pid':p.pid,'command':command,'log':str(log)};state['jobs'].append(row);save()
        row['returncode']=p.wait();save()
        if row['returncode']!=0:state['stage']='engineering_review_required';save();return False
        return True
    initial='0,4,8,12,16,20,24,28,32,36,40,45,49,53,57,61,65,69,73,77,81,86,90,94,98,102,106,110,114,118,122,127'
    if not (reference/'NATIVE_REFERENCE_COMPLETE.json').exists():
        if not child('initial_ID_shards',base+['--episodes',initial]):return 1
        marker=args.inputs/'FULL_ID_RESTORE_COMPLETE.json';state['stage']='waiting_original_ID_remainder';save()
        while not marker.exists():time.sleep(30)
        if not child('full_ID_reference',base):return 1
    calibration=args.root/'ID_calibration_native_v2'
    if not (calibration/'CALIBRATION_COMPLETE.json').exists():
        command=['taskset','-c','0-3',sys.executable,'-u',str(args.code/'tools/calibrate_pi05_feedback.py'),'--task','open_drawer_grasp_ood','--manifest',str(manifest),'--reference-asset',str(reference/'bridge_reference.pt'),'--output',str(calibration),'--seed','1781000']
        if not child('native_ID_calibration',command):return 1
    if not child('calibration_audit',[sys.executable,str(args.code/'tools/audit_pi05_feedback_calibration.py'),'--root',str(calibration)]):return 1
    state.update(stage='native_reference_and_calibration_audited',next_stage='OpenDrawer_current_state_oracle_smoke_then_stage_pairs',finished=time.time());save()
    (args.root/'REFERENCE_CALIBRATION_COMPLETE.json').write_text(json.dumps(state,indent=2));return 0


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--code',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--logs',type=Path,required=True);raise SystemExit(run(p.parse_args()))
