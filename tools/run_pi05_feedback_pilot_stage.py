"""Persistent two-arm pilot controller with raw-array auditing and timing table."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np


def paired_report(root):
    fixed=json.loads((root/'fixed/summary.json').read_text())
    feedback=json.loads((root/'feedback/summary.json').read_text())
    assert len(fixed['rows'])==len(feedback['rows'])
    pairs=[]
    for a,b in zip(fixed['rows'],feedback['rows']):
        assert (a['episode'],a['seed'],a['split'])==(b['episode'],b['seed'],b['split'])
        i=a['episode']
        left=np.load(root/f'fixed/episode_{i:04d}/trace.npz')
        right=np.load(root/f'feedback/episode_{i:04d}/trace.npz')
        common=min(a['takeover'] if a['takeover'] is not None else len(left['actions']),
                   b['takeover'] if b['takeover'] is not None else len(right['actions']))
        difference=float(np.max(abs(left['actions'][:common]-right['actions'][:common]))) if common else 0.
        assert difference<1e-5,(i,difference)
        pairs.append({'episode':i,'split':a['split'],'seed':a['seed'],'fixed_takeover':a['takeover'],
                      'feedback_takeover':b['takeover'],'shift':b['takeover']-a['takeover'] if a['takeover'] is not None and b['takeover'] is not None else None,
                      'common_prefix_actions':common,'common_prefix_max_difference':difference,
                      'fixed_expert_cost':a['all_executed_expert_actions'],'feedback_expert_cost':b['all_executed_expert_actions'],
                      'fixed_assisted_success':a['strict_success'],'feedback_assisted_success':b['strict_success'],
                      'feedback_cue':b['cue']})
    summary={}
    for split in ['id','ood']:
        rows=[r for r in pairs if r['split']==split];both=[r['shift'] for r in rows if r['shift'] is not None]
        summary[split]={'episodes':len(rows),'paired_alarm_count':len(both),
                        'earlier':sum(x<0 for x in both),'same':sum(x==0 for x in both),'later':sum(x>0 for x in both),
                        'median_shift_among_both_detected':float(np.median(both)) if both else None,
                        'fixed_no_alarm':sum(r['fixed_takeover'] is None for r in rows),
                        'feedback_no_alarm':sum(r['feedback_takeover'] is None for r in rows),
                        'fixed_expert_cost':sum(r['fixed_expert_cost'] for r in rows),
                        'feedback_expert_cost':sum(r['feedback_expert_cost'] for r in rows)}
    return {'status':'PAIRED_PILOT_AUDITED','summary':summary,'pairs':pairs,
            'interpretation':'Development collection timing/cost only. Not TASR, matched-budget learning, or post-SFT success.'}


def main(args):
    args.root.mkdir(parents=True,exist_ok=True);args.logs.mkdir(parents=True,exist_ok=True)
    state={'controller_pid':os.getpid(),'task':args.task,'stage':'paired_pilot_collection',
           'next_stage':'TASR_and_formal_collection_preflight','jobs':[],'whole_pipeline_complete':False}
    def save():(args.root/'pipeline_state.json').write_text(json.dumps(state,indent=2))
    for arm in ['fixed','feedback']:
        output=args.root/arm
        if (output/'INDEPENDENT_COLLECTION_AUDIT.json').exists():continue
        if output.exists():raise RuntimeError(f'Partial arm needs named retry: {output}')
        cmd=['taskset','-c',args.cpu,sys.executable,str(args.code/'tools/collect_pi05_feedback.py'),
             '--task',args.task,'--arm',arm,'--manifest',str(args.code/'configs/pipelines/pi05_timing_feedback_ablation_v1.json'),
             '--calibration',str(args.calibration),'--output',str(output),'--seed',str(args.seed),'--episodes',str(args.episodes)]
        cmd+=['--feedback-rule',args.feedback_rule]
        if args.opening_calibration is not None:cmd+=['--opening-calibration',str(args.opening_calibration)]
        log=args.logs/f'{args.task}_{arm}_pilot.log'
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu,VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json',
                 OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        with log.open('w') as stream:child=subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        row={'arm':arm,'pid':child.pid,'command':cmd,'log':str(log),'output':str(output)}
        state['jobs'].append(row);save();row['returncode']=child.wait();save()
        if row['returncode']!=0:state['stage']='collector_requires_repair';save();return 1
        rc=subprocess.call([sys.executable,str(args.code/'tools/audit_pi05_feedback_collection.py'),'--root',str(output)])
        if rc:state['stage']='audit_failed';save();return 1
    result=paired_report(args.root)
    (args.root/'paired_timing_report.json').write_text(json.dumps(result,indent=2))
    state.update(stage='paired_pilot_audited',finished=time.time());save()
    (args.root/'PAIRED_PILOT_COMPLETE.json').write_text(json.dumps(state,indent=2));return 0


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--code',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--logs',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--task',required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--gpu',required=True);p.add_argument('--cpu',required=True);p.add_argument('--episodes',type=int,default=20)
    p.add_argument('--feedback-rule',choices=['displacement_v1','commitment_v2'],default='displacement_v1')
    p.add_argument('--opening-calibration',type=Path)
    sys.exit(main(p.parse_args()))
