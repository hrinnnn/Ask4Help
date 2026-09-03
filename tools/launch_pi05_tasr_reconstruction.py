#!/usr/bin/env python3
"""Launch two restartable H20 replay workers, leaving old workloads untouched."""
import argparse,json,subprocess
from pathlib import Path

REMOTE=r'''
import json,os,subprocess,datetime
from pathlib import Path
scratch=Path('/tmp/pi05_tasr_reconstruction_v1');out=Path('/mnt/data/ask4help/results/pi05_tasr_reconstruction_v1')/run
out.mkdir(parents=True,exist_ok=True);logs=scratch/'logs'/run;logs.mkdir(parents=True,exist_ok=True)
jobs=[]
for shard in [0,1]:
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(shard),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',
  TMPDIR=str(scratch/'tmp'),XDG_CACHE_HOME=str(scratch/'cache'),VK_ICD_FILENAMES='/opt/conda/envs/robo-dopamine/lib/python3.10/site-packages/sapien/vulkan_library/nvidia_icd.json')
 cmd=['taskset','-c',f'{2*shard},{2*shard+1}','/root/.venvs/xvla-h20/bin/python',str(scratch/'code/reconstruct_pi05_tasr_worker.py'),
  '--input',str(scratch/'replay_bundle.json.gz'),'--output',str(out),'--shard',str(shard),'--shards','2']
 log=logs/f'worker_{shard}.log';handle=log.open('ab');p=subprocess.Popen(cmd,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True);handle.close()
 jobs.append(dict(shard=shard,gpu=shard,pid=p.pid,log=str(log),command=cmd))
 state=dict(authorized=True,owner='codex-root-pi05-table-tasr',current_stage='bulk_original_action_replay',next_stage='owner_download_audit_tasr_table',jobs=jobs,
  output=str(out),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),completion='reconciliation_shard_0.json and reconciliation_shard_1.json; every row independently audited before TASR')
(out/'launch_state.json').write_text(json.dumps(state,indent=2));print(json.dumps(state))
'''

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--local-root',type=Path,required=True);a=p.parse_args()
 r=subprocess.run(['ssh','-o','ConnectTimeout=12','-o','BatchMode=yes','-p','1012','root@39.101.70.188','python3 -'],input=('run='+repr(a.run)+'\n'+REMOTE).encode(),stdout=subprocess.PIPE,check=True)
 state=json.loads(r.stdout);(a.local_root/f'{a.run}_launch_state.json').write_text(json.dumps(state,indent=2));print(json.dumps(state),flush=True)
