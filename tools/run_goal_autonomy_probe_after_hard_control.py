"""Bounded development replay to resolve censoring, after the preregistered controls."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path


def run(a):
    a.root.mkdir(parents=True,exist_ok=True);a.logs.mkdir(parents=True,exist_ok=True)
    state={'controller_pid':os.getpid(),'stage':'waiting_hard_control','whole_pipeline_complete':False}
    def save():(a.root/'controller_state.json').write_text(json.dumps(state,indent=2))
    save()
    while not (a.hard_root/'THREE_ARM_PILOTS_COMPLETE.json').exists():
        if (a.hard_root/'STAGE_CONTROLLER_FAILED.json').exists():
            state.update(stage='upstream_requires_review');save();return 1
        time.sleep(60)
    manifest=a.code/'configs/pipelines/pi05_timing_feedback_ablation_v1.json'
    checkpoint=json.loads(manifest.read_text())['task_assets']['open_drawer_goal_ood']['checkpoint']
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',
             HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json')
    output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
    if {int(x) for x in output.splitlines() if x.strip().isdigit()}-{276925}:
        state.update(stage='reserved_GPUs_not_free');save();return 1
    jobs=[];state['stage']='autonomous_ID_OOD_development';state['jobs']=[]
    for split,gpu,cpu in [('id','0','0-3'),('ood','1','4-7')]:
        cmd=['taskset','-c',cpu,sys.executable,'-u',str(a.code/'tools/evaluate_pi05_feedback_policy.py'),
             '--manifest',str(manifest),'--protocol',str(a.code/'configs/pipelines/opendrawer_goal_autonomous_diagnostic_v1.json'),
             '--task','open_drawer_goal_ood','--split',split,'--checkpoint',checkpoint,'--output',str(a.root/split)]
        log=a.logs/(split+'.log')
        with log.open('w') as f:p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,
            env={**env,'CUDA_VISIBLE_DEVICES':gpu,'TMPDIR':f'/tmp/pi05_timing_feedback_ablation_v1/tmp{gpu}'})
        row={'split':split,'pid':p.pid,'command':cmd,'log':str(log)};state['jobs'].append(row);save();jobs.append((p,row))
    for p,row in jobs:
        row['returncode']=p.wait();save()
        if row['returncode']:state.update(stage='probe_requires_review');save();return 1
    subprocess.run([sys.executable,str(a.code/'tools/audit_goal_autonomy_censoring.py'),
                    '--root',str(a.root),'--gated-root',str(a.gated_root)],check=True)
    state.update(stage='autonomous_development_probe_complete');save()
    (a.root/'AUTONOMY_PROBE_COMPLETE.json').write_text(json.dumps(state,indent=2));return 0


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['code','root','logs','hard-root','gated-root']:p.add_argument('--'+name,type=Path,required=True)
    raise SystemExit(run(p.parse_args()))
