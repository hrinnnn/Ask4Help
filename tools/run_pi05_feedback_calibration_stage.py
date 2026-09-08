"""Durable calibration phase controller; never marks the whole study complete."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def run(args):
    args.root.mkdir(parents=True,exist_ok=True);args.logs.mkdir(parents=True,exist_ok=True)
    state={'controller_pid':os.getpid(),'stage':'ID_calibration','next_stage':'expert_collector_smoke_then_paired_collection',
           'started':time.time(),'jobs':[],'whole_pipeline_complete':False}
    def save():
        (args.root/'pipeline_state.json').write_text(json.dumps(state,indent=2))
    jobs=[]
    for i,task in enumerate(['stackcube_legacy_ood','airplane_yaw_ood']):
        output=args.root/'calibration_v1'/task
        if (output/'CALIBRATION_COMPLETE.json').exists():continue
        if output.exists():raise RuntimeError(f'Existing partial output requires a named retry: {output}')
        command=['taskset','-c',f'{4*i}-{4*i+3}',sys.executable,str(args.code/'tools/calibrate_pi05_feedback.py'),
                 '--task',task,'--manifest',str(args.code/'configs/pipelines/pi05_timing_feedback_ablation_v1.json'),
                 '--output',str(output),'--seed',str(1740000+10000*i)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(i),VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json',
                 OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',
                 TMPDIR=f'/tmp/pi05_timing_feedback_ablation_v1/tmp{i}',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        log=args.logs/f'{task}_calibration_v1.log'
        with log.open('w') as stream:
            child=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        row={'task':task,'pid':child.pid,'command':command,'log':str(log),'output':str(output),'returncode':None}
        state['jobs'].append(row);jobs.append((child,row));save()
    for child,row in jobs:
        row['returncode']=child.wait()
        row['complete']=(Path(row['output'])/'CALIBRATION_COMPLETE.json').exists()
        save()
    passed=all(r['returncode']==0 and r['complete'] for r in state['jobs'])
    state.update(stage='ID_calibration_ready_for_audit' if passed else 'ID_calibration_requires_repair',finished=time.time())
    save()
    if passed:(args.root/'ID_CALIBRATION_STAGE_COMPLETE.json').write_text(json.dumps(state,indent=2))
    return 0 if passed else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--code',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--logs',type=Path,required=True)
    args=p.parse_args();sys.exit(run(args))
