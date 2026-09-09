"""Restart-tolerant diagnostic controller. No training stage or training entrypoint."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from pi05_feedback_artifacts import completed_rows

def main(a):
    planpath=a.plan or a.code/'configs/pipelines/pi05_feedback_small30_v1.json'
    plan=json.loads(planpath.read_text())
    assert plan['authorized'] and not plan['training_authorized']
    task=next(t for t in plan['task_settings'] if t['name']==a.task)
    parent=json.loads((a.code/plan['parent_manifest']).read_text())
    a.root.mkdir(parents=True,exist_ok=True);a.logs.mkdir(parents=True,exist_ok=True)
    statepath=a.root/'controller_state.json'
    state=json.loads(statepath.read_text()) if statepath.exists() else dict(completed_roots={},jobs=[],stage='preflight')
    state.update(controller_pid=os.getpid(),training_authorized=False,whole_pipeline_complete=False)
    def save():statepath.write_text(json.dumps(state,indent=2))
    def run(cmd,label):
        log=a.logs/(label+'.log')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=task['gpu'],OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
                 PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                 VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json',
                 TMPDIR=f"/tmp/pi05_timing_feedback_ablation_v1/tmp{task['gpu']}")
        with log.open('a') as f:child=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        row=dict(pid=child.pid,command=cmd,log=str(log),label=label);state['jobs'].append(row);save()
        row['returncode']=child.wait();save()
        if row['returncode']:raise RuntimeError(f'{label}: see {log}')
    assets=parent['task_assets'][task['task']]
    for p in [assets['checkpoint'],assets['norm'],task['calibration']+'/gate_arrays.npz',task['calibration']+'/INDEPENDENT_CALIBRATION_AUDIT.json']:
        assert Path(p).exists(),p
    if task['opening_calibration']:assert Path(task['opening_calibration']).exists()
    save()
    try:
        for name,settings in [*plan['arms'].items(),('passive',plan['arms']['fixed'])]:
            if name in state['completed_roots']:continue
            state.update(stage='collect_'+name,next_stage='independent_detection' if name=='sensitive' else 'continue_diagnostics');save()
            armroot=a.root/name;armroot.mkdir(exist_ok=True)
            manifest=json.loads(json.dumps(parent));manifest['feedback']['max_wait_blocks']=settings['max_wait_blocks']
            for key in ['remember_deferred_alarm','use_later_duration']:
                manifest['feedback'][key]=settings.get(key,False)
            manifestpath=armroot/'manifest.json'
            if manifestpath.exists():assert json.loads(manifestpath.read_text())==manifest
            else:manifestpath.write_text(json.dumps(manifest,indent=2))
            cap=plan['passive_raw_episodes'] if name=='passive' else plan['max_raw_attempts']
            target=None if name=='passive' else plan['accepted_target']
            seed=task['evaluation_seed'] if name=='passive' else task['collection_seed']
            resume=None;rows=[]
            # Named chunks preserve completed prefixes and allow immutable recovery.
            chunks=sorted(armroot.glob('chunk_*'))
            if chunks:
                resume=chunks[-1];rows=completed_rows(resume)
                if not (resume/'INDEPENDENT_COLLECTION_AUDIT.json').exists():
                    if not (resume/'summary.json').exists():raise RuntimeError(f'Partial chunk requires engineering review: {resume}')
                    run([sys.executable,str(a.code/'tools/audit_pi05_feedback_collection.py'),'--root',str(resume)],name+'_resume_audit')
            while len(rows)<cap and (target is None or sum(r['accepted'] for r in rows)<target):
                output=armroot/f'chunk_{len(rows):04d}'
                if output.exists():raise RuntimeError(f'Preserve partial chunk and investigate: {output}')
                cmd=['taskset','-c',task['cpu'],sys.executable,str(a.code/'tools/collect_pi05_feedback.py'),
                     '--task',task['task'],'--arm','feedback' if settings['enabled'] else 'fixed',
                     '--manifest',str(manifestpath),'--calibration',task['calibration'],'--output',str(output),
                     '--seed',str(seed),'--episodes',str(cap),'--max-new-episodes',str(plan['chunk_size']),
                     '--feedback-rule',task['feedback_rule'],'--support-mode',settings['support_mode']]
                if target is not None:cmd+=['--accepted-target',str(target)]
                if resume is not None:cmd+=['--resume-from',str(resume)]
                if task['opening_calibration']:cmd+=['--opening-calibration',task['opening_calibration']]
                if name=='passive':cmd+=['--force-takeover-step','1000000']
                run(cmd,name+f'_from{len(rows):04d}')
                run([sys.executable,str(a.code/'tools/audit_pi05_feedback_collection.py'),'--root',str(output)],name+f'_audit{len(rows):04d}')
                resume=output;rows=completed_rows(output)
            state['completed_roots'][name]=str(resume);save()
        state.update(stage='independent_timing_BA_analysis',next_stage='video_review_no_training');save()
        run([sys.executable,str(a.code/'tools/analyze_pi05_feedback_small30.py'),'--root',str(a.root),
             '--plan',str(planpath),'--task',a.task],'analysis')
        state.update(stage='timing_BA_complete_pending_visual_review',next_stage='qualitative_video_review',finished=time.time());save()
        (a.root/'TIMING_BA_COMPLETE.json').write_text(json.dumps(state,indent=2))
    except Exception as e:
        state.update(stage='engineering_review_required',error=repr(e));save();raise

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['code','root','logs']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--plan',type=Path)
    p.add_argument('--task',required=True);main(p.parse_args())
