#!/usr/bin/env python3
"""CPU-only recovery probe for original pi05 training suffix states. No policy."""
import argparse,gzip,json,subprocess
from pathlib import Path

NUMERIC=r'''
import json,gzip,sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
rows=[]
for item in items:
 p=Path(item['collection']);ds=Path(item['dataset'])
 if item['task']=='object':
  raw=[json.loads(x) for x in (p/'attempts.jsonl').read_text().splitlines() if x.strip()];accepted=[r for r in raw if r['accepted']]
  index,meta=next((i,r) for i,r in enumerate(accepted) if r['split']=='ood');start=meta['expert_start_step'];rawindex=meta['attempt']
 else:
  train=[json.loads(x) for x in (p/'training_episodes.jsonl').read_text().splitlines() if x.strip()]
  tr=next(r for r in train if r['split']=='ood');index=tr['dataset_episode_index'];start=tr.get('expert_start_step',tr.get('start_step'))
  raw=[json.loads(x) for x in (p/'episodes.jsonl').read_text().splitlines() if x.strip()]
  meta=next(r for r in raw if r['seed']==tr['seed'] and r['split']=='ood');rawindex=tr.get('raw_attempt_index',tr.get('raw_episode_index'))
 f=ds/f'data/chunk-000/episode_{index:06d}.parquet';table=pq.read_table(f,columns=['state','actions'])
 qpos=np.asarray(table['state'].to_pylist(),dtype=np.float32);targets=np.asarray(table['actions'].to_pylist(),dtype=np.float32)
 source=p/'raw_archive/actions'/f'episode_{rawindex:06d}_seed_{meta["seed"]:06d}.npy'
 if not source.exists():
  found=list((p/'raw_archive/actions').glob(f'*seed_{meta["seed"]:06d}*.npy'));assert len(found)==1;source=found[0]
 actions=np.load(source);assert np.allclose(targets,actions[start:start+len(targets)],atol=1e-6)
 rows.append(dict(**item,index=index,seed=meta['seed'],start=start,metadata=meta,actions=actions.tolist(),qpos=qpos.tolist(),source_actions=str(source),source_parquet=str(f)))
sys.stdout.buffer.write(gzip.compress(json.dumps(rows).encode()))
'''

REPLAY=r'''
import json,sys,types,traceback
import numpy as np,torch,gymnasium as gym
import mani_skill.envs
torch.set_num_threads(2)
out=[]
for row in rows:
 task=row['task'];mod=types.ModuleType('tasr_'+task);sys.modules[mod.__name__]=mod;exec(sources[task],mod.__dict__)
 if task=='object':mod.register_controlled_pick_single_ycb_object_variants();env_id=mod.PICK_SINGLE_YCB_OBJECT_OOD_ENV_ID
 elif task=='airplane':mod.register_controlled_pick_single_ycb_airplane_variants();env_id=mod.PICK_SINGLE_YCB_AIRPLANE_OOD_ENV_ID
 else:mod.register_controlled_stack_cube_variants();env_id=mod.STACK_CUBE_OOD_ENV_ID
 kw={'robot_uids':'panda_wristcam'} if task in ['object','stackcube'] else {}
 print('CPU_REPLAY_START',task,row['method'],row['seed'],file=sys.stderr,flush=True)
 try:
  env=gym.make(env_id,num_envs=1,obs_mode='none',control_mode='pd_joint_delta_pos',reward_mode='sparse',render_mode=None,
   sim_backend='physx_cpu',render_backend=render_backend,sim_config={'sim_freq':100,'control_freq':10},max_episode_steps=1000,**kw)
  print('CPU_ENV_CREATED',task,file=sys.stderr,flush=True)
  base=env.unwrapped;env.reset(seed=row['seed']);recorded=[];obj=base.cubeA if task=='stackcube' else base.obj
  print('CPU_RESET_COMPLETE',task,file=sys.stderr,flush=True)
  def snapshot():
   return dict(qpos=base.agent.robot.get_qpos().reshape(-1).cpu().numpy().copy().tolist(),object_p=obj.pose.p.reshape(-1).cpu().numpy().copy().tolist(),
    object_q=obj.pose.q.reshape(-1).cpu().numpy().copy().tolist(),grasped=bool(base.agent.is_grasping(obj)))
  recorded.append(snapshot())
  for t,a in enumerate(row['actions']):
   if task=='airplane' and t==row['start']:
    # Original _plan_and_execute_expert restores its snapshot even for the
    # first candidate. Preserve that physics/contact-cache reset operation.
    base.set_state_dict(base.get_state_dict())
   env.step(torch.as_tensor(a,dtype=torch.float32).reshape(1,-1));recorded.append(snapshot())
  print('CPU_REPLAY_STEPS_COMPLETE',task,file=sys.stderr,flush=True)
  q=np.array([r['qpos'] for r in recorded]);expected=np.array(row['qpos']);start=row['start'];error=np.max(np.abs(q[start:start+len(expected)]-expected),axis=1)
  result=dict(task=task,method=row['method'],seed=row['seed'],status='PASS_QPOS_REPLAY_ONLY' if error.max()<1e-4 else 'REPLAY_DRIFT',
   frames=len(recorded),max_qpos_error=float(error.max()),first_bad_offset=int(np.flatnonzero(error>=1e-4)[0]) if np.any(error>=1e-4) else None,
   actual_robot=base.agent.uid,initial=recorded[0],final=recorded[-1],states=recorded if error.max()<1e-4 else None)
  out.append(result);print('CPU_REPLAY_RESULT',json.dumps({k:v for k,v in result.items() if k not in ['states','initial','final']}),file=sys.stderr,flush=True);env.close()
 except Exception as e:
  out.append(dict(task=task,method=row['method'],status='REPLAY_RUNTIME_FAILURE',error=repr(e)));traceback.print_exc(file=sys.stderr)
print(json.dumps(out))
'''

def main(root,gpu,backend):
 root.mkdir(parents=True,exist_ok=True)
 h='/mnt/data/ask4help/results/';y='/data/zhaozhixuan/Ask4Help-airplane-5090/results/object_variation_pick_single_ycb_v1/'
 requests=[('5090',[dict(task='object',method='offline_oracle',collection=y+'collections_v1/offline_oracle',dataset=y+'datasets/offline_oracle_v1'),
  dict(task='object',method='failure_recovery',collection=y+'collections_v1/failure_recovery',dataset=y+'datasets/failure_recovery_v1')]),
  ('h20',[dict(task='airplane',method=m,collection=h+'pick_single_ycb_airplane/four_group_dagger_v1/formal_v3/'+m+'_collection',dataset=h+'pick_single_ycb_airplane/four_group_dagger_v1/formal_v3/'+m+'_collection/lerobot') for m in ['offline_oracle','failure_recovery']])]
 input_file=root/'replay_probe_inputs.json'
 rows=json.loads(input_file.read_text()) if input_file.exists() else []
 for host,items in ([] if rows else requests):
  port,target,python=('12001','zhaozhixuan@111.198.58.150','/data/zhaozhixuan/Ask4Help-airplane-5090/RLinf/.venv/bin/python') if host=='5090' else ('1012','root@39.101.70.188','/root/.venvs/xvla-h20/bin/python')
  r=subprocess.run(['ssh','-o','ConnectTimeout=12','-o','BatchMode=yes','-p',port,target,python+' -'],input=('items='+repr(items)+'\n'+NUMERIC).encode(),stdout=subprocess.PIPE,check=True)
  rows.extend(json.loads(gzip.decompress(r.stdout)))
 (root/'replay_probe_inputs.json').write_text(json.dumps(rows,indent=2))
 sources={task:Path('RLinf/rlinf/envs/maniskill/'+file).read_text() for task,file in [('object','pick_single_ycb_object_variation.py'),('airplane','pick_single_ycb_airplane_variants.py')]}
 script='import json\nrows=json.loads('+repr(json.dumps(rows))+')\nsources='+repr(sources)+'\nrender_backend='+repr(backend)+'\n'+REPLAY
 scratch='/tmp/pi05_tasr_reconstruction_v1'
 subprocess.run(['ssh','-p','1012','root@39.101.70.188',f'mkdir -p {scratch}/cache {scratch}/tmp'],check=True)
 env=f"CUDA_VISIBLE_DEVICES='{gpu}' OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 TMPDIR={scratch}/tmp XDG_CACHE_HOME={scratch}/cache VK_ICD_FILENAMES=/opt/conda/envs/robo-dopamine/lib/python3.10/site-packages/sapien/vulkan_library/nvidia_icd.json"
 r=subprocess.run(['ssh','-o','ConnectTimeout=12','-o','BatchMode=yes','-p','1012','root@39.101.70.188',
  f"env {env} /root/.venvs/xvla-h20/bin/python -"],input=script.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 (root/'replay_probe.stdout').write_bytes(r.stdout);(root/'replay_probe.stderr').write_bytes(r.stderr)
 print(r.stderr.decode(errors='replace')[-12000:]);print('REMOTE_EXIT',r.returncode)
 if r.returncode==0:
  # Preserve simulator stdout noise separately; final line is structured result.
  result=json.loads(r.stdout.decode().splitlines()[-1]);(root/'replay_probe_results.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--gpu',default='');p.add_argument('--render-backend',default='none');a=p.parse_args();main(a.root,a.gpu,a.render_backend)
