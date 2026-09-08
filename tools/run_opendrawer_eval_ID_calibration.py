"""Two-GPU full calibration, same observations/windows, original task eval mode."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path


def run(a):
    a.root.mkdir(parents=True,exist_ok=True);a.logs.mkdir(parents=True,exist_ok=True)
    state={'controller_pid':os.getpid(),'stage':'dense_ID_demos','jobs':[],'inference_mode':'eval','whole_pipeline_complete':False}
    def save():(a.root/'controller_state.json').write_text(json.dumps(state,indent=2))
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',
             VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    try:
        for component in ['demos','policies']:
            state['stage']=component;save();jobs=[]
            for shard in [0,1]:
                output=a.root/'components'/f'{component}_{shard}'
                if (output/'COMPONENT_COMPLETE.json').exists():continue
                if output.exists():raise RuntimeError(f'Preserve partial component and use named retry: {output}')
                cpu='0-3' if shard==0 else '4-7'
                cmd=['taskset','-c',cpu,sys.executable,'-u',str(a.code/'tools/collect_opendrawer_ID_calibration_component.py'),
                     '--manifest',str(a.code/'configs/pipelines/pi05_timing_feedback_ablation_v1.json'),
                     '--reference',str(a.reference),'--output',str(output),'--component',component,'--shard',str(shard)]
                log=a.logs/f'{component}_{shard}.log'
                with log.open('w') as stream:p=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,
                    env={**env,'CUDA_VISIBLE_DEVICES':str(shard),'TMPDIR':f'/tmp/pi05_timing_feedback_ablation_v1/tmp{shard}'})
                row={'component':component,'shard':shard,'pid':p.pid,'command':cmd,'log':str(log)}
                state['jobs'].append(row);save();jobs.append((p,row))
            for p,row in jobs:
                row['returncode']=p.wait();save()
                if row['returncode']:raise RuntimeError(f'component failed: {row}')
        state['stage']='assemble_and_audit';save()
        cal=a.root/'calibration'
        if not (cal/'CALIBRATION_COMPLETE.json').exists():
            subprocess.run([sys.executable,str(a.code/'tools/assemble_opendrawer_ID_calibration.py'),'--components',str(a.root/'components'),
                            '--reference',str(a.reference),'--output',str(cal)],check=True,env=env)
        subprocess.run([sys.executable,str(a.code/'tools/audit_pi05_feedback_calibration.py'),'--root',str(cal)],check=True,env=env)
        state.update(stage='eval_ID_calibration_audited',finished=time.time(),next_stage='two_stage_feedback_pairs');save()
        (a.root/'ID_CALIBRATION_EVAL_COMPLETE.json').write_text(json.dumps(state,indent=2))
    except Exception as error:
        state.update(stage='requires_review',error=repr(error));save()
        (a.root/'ID_CALIBRATION_CONTROLLER_FAILED.json').write_text(json.dumps(state,indent=2));raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['code','root','reference','logs']:p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
