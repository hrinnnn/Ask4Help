#!/usr/bin/env python3
"""Deterministic action replay; reconstruct missing states only after auditing qpos."""
import argparse,gzip,io,json,os,time,types,sys
from pathlib import Path
import numpy as np

def main(args):
 import torch,gymnasium as gym,mani_skill.envs
 torch.set_num_threads(2)
 payload=json.loads(gzip.decompress(args.input.read_bytes()));rows=payload['rows'];sources=payload['sources']
 modules={}
 for task,source in sources.items():
  module=types.ModuleType('tasr_'+task);sys.modules[module.__name__]=module;exec(source,module.__dict__);modules[task]=module
  getattr(module,{'object':'register_controlled_pick_single_ycb_object_variants','airplane':'register_controlled_pick_single_ycb_airplane_variants','stackcube':'register_controlled_stack_cube_variants'}[task])()
 if args.limit_per_group:
  counts={};chosen=[]
  for r in rows:
   key=r['task'],r['method']
   if r['split']=='ood' and r['selected'] and counts.get(key,0)<args.limit_per_group:chosen.append(r);counts[key]=counts.get(key,0)+1
  rows=chosen
 rows=[r for i,r in enumerate(rows) if i%args.shards==args.shard]
 args.output.mkdir(parents=True,exist_ok=True);started=time.time();reports=[]
 state_path=args.output/f'pipeline_state_shard_{args.shard}.json'
 def save_state(stage,last=None):
  state=dict(stage=stage,pid=os.getpid(),shard=args.shard,shards=args.shards,total=len(rows),done=len(reports),passed=sum(r['status']=='PASS' for r in reports),
   failed=sum(r['status']!='PASS' for r in reports),elapsed_seconds=time.time()-started,last=last)
  tmp=state_path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(state_path)
 save_state('RUNNING')
 for row in rows:
  task=row['task'];method=row['method'];index=row['source_episode_index'];folder=args.output/task/method;folder.mkdir(parents=True,exist_ok=True)
  jsonpath=folder/f'episode_{index:06d}.json';npzpath=folder/f'episode_{index:06d}.npz'
  if jsonpath.exists() and npzpath.exists():
   old=json.loads(jsonpath.read_text())
   if old['status']=='PASS':reports.append(old);continue
  module=modules[task];split=row['split'].upper()
  prefix={'object':'PICK_SINGLE_YCB_OBJECT','airplane':'PICK_SINGLE_YCB_AIRPLANE','stackcube':'STACK_CUBE'}[task]
  env_id=getattr(module,f'{prefix}_{split}_ENV_ID')
  env=gym.make(env_id,num_envs=1,robot_uids='panda_wristcam',obs_mode='none',control_mode='pd_joint_delta_pos',reward_mode='sparse',
   render_mode=None,sim_backend='physx_cpu',render_backend='gpu',sim_config={'sim_freq':100,'control_freq':10},max_episode_steps=1000)
  base=env.unwrapped
  # The historical StackCube full-suffix materializer retried identical actions
  # up to eight times because contact outcomes were not bitwise deterministic.
  # Warmups reconstruct that reset history, never alter actions or seeds.
  for warmup in range(args.warmup_replays):
   env.reset(seed=row['seed'])
   for t,a in enumerate(row['actions']):
    if task=='airplane' and t==row['expert_start']:base.set_state_dict(base.get_state_dict())
    env.step(torch.tensor(a,dtype=torch.float32).reshape(1,-1))
  env.reset(seed=row['seed']);obj=base.cubeA if task=='stackcube' else base.obj
  snapshots=[]
  def snap():
   def vec(x):return x.reshape(-1).detach().cpu().numpy().copy()
   info=base.evaluate()
   snapshots.append(dict(qpos=vec(base.agent.robot.get_qpos()),tcp_p=vec(base.agent.tcp.pose.p),tcp_q=vec(base.agent.tcp.pose.q),
    object_p=vec(obj.pose.p),object_q=vec(obj.pose.q),grasped=bool(base.agent.is_grasping(obj)),success=bool(info['success'])))
  snap();initial=base.get_state_dict() if task=='airplane' and row['expert_start']==0 else None
  for t,a in enumerate(row['actions']):
   if task=='airplane' and t==row['expert_start']:base.set_state_dict(initial if initial is not None else base.get_state_dict())
   env.step(torch.tensor(a,dtype=torch.float32).reshape(1,-1));snap()
  data={k:np.asarray([s[k] for s in snapshots]) for k in snapshots[0]};expected=np.asarray(row['qpos']);start=row['expert_start']
  error=np.max(np.abs(data['qpos'][start:start+len(expected)]-expected),axis=1)
  report={k:row[k] for k in ['task','method','seed','split','source_episode_index','selected','expert_start','expert_points']}
  report.update(status='PASS' if float(error.max())<1e-4 else 'REPLAY_DRIFT',max_qpos_error=float(error.max()),
   first_bad_offset=int(np.flatnonzero(error>=1e-4)[0]) if np.any(error>=1e-4) else None,
   final_success=bool(data['success'][-1]),array_file=str(npzpath),state_rows=len(snapshots),original_action_count=len(row['actions']),
   per_joint_max_error=np.max(np.abs(data['qpos'][start:start+len(expected)]-expected),axis=0).tolist(),warmup_replays=args.warmup_replays)
  pose_key='cube_a_pose' if task=='stackcube' else 'object_pose';pose_index=-1 if task=='airplane' else 0
  if pose_key in row['metadata']:
   original=row['metadata'][pose_key];pe=float(np.max(np.abs(data['object_p'][pose_index]-original['p'])))
   qe=min(float(np.max(np.abs(data['object_q'][pose_index]-original['q']))),float(np.max(np.abs(data['object_q'][pose_index]+original['q']))))
   report.update(object_checkpoint_position_error=pe,object_checkpoint_quaternion_error=qe,object_checkpoint='terminal' if pose_index==-1 else 'reset')
   if pe>=1e-4 or qe>=1e-4:report['status']='OBJECT_CHECKPOINT_MISMATCH'
  if not report['final_success']:report['status']='REPLAY_SUCCESS_MISMATCH'
  # The saved trace of a mismatch is diagnostic only, never an eligible score input.
  buffer=io.BytesIO();np.savez_compressed(buffer,**data);npzpath.write_bytes(buffer.getvalue())
  jsonpath.write_text(json.dumps(report,indent=2));reports.append(report);env.close()
  save_state('RUNNING',report);print(json.dumps({k:v for k,v in report.items() if k not in ['per_joint_max_error','array_file']}),flush=True)
 (args.output/f'reconciliation_shard_{args.shard}.json').write_text(json.dumps(reports,indent=2));save_state('REPLAY_FINISHED')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--limit-per-group',type=int);p.add_argument('--warmup-replays',type=int,default=0)
 main(p.parse_args())
