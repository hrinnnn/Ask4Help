"""Small real closed-loop collection pilot. Frozen VLA; no SFT."""
import argparse,io,json,os,sys,time
from pathlib import Path
import numpy as np

def main(args):
 import torch,gymnasium as gym,mani_skill.envs,cv2
 torch.set_num_threads(4)
 sys.path[:0]=['/root/Ask4Help-stage2-ood','/root/Ask4Help-stage2-ood/RLinf',str(Path(__file__).resolve().parent)]
 from tools.xvla_airplane_runtime import XVLAAirplanePolicy
 from tools.stackcube_stage2_ood import register_stack_cube_splits,stack_cube_env_id,stack_cube_reset_metadata
 from rlinf.envs.maniskill.stack_cube_privileged_oracle import StackCubePrivilegedChunkOracle
 from expert_feedback_pca_core import ExpertFeedbackPCA
 register_stack_cube_splits();args.output.mkdir(parents=True,exist_ok=True);started=time.time()
 cal=json.loads(args.calibration.read_text());asset=np.load(args.pca);mean=torch.tensor(asset['mean'],device='cuda');basis=torch.tensor(asset['eigenvectors'][:,:512],device='cuda')
 gate=ExpertFeedbackPCA(cal['tau0'],np.array(cal['center']),cal['scale'],cal['radius'],cal['q_loss']);policy=XVLAAirplanePolicy(Path('/root/xvla_stage2_inputs_priority/ckpt-7500'),Path('/root/X-VLA-stackpyramid-clean'))
 rows=[];cost=0;instruction='stack the red cube on the green cube'
 def snapshot(env,obs):
  def cpu(v):return v.detach().cpu().numpy().copy()
  return dict(qpos=cpu(obs['agent']['qpos']).reshape(-1),image=cpu(obs['sensor_data']['base_camera']['rgb'])[0],
   wrist=cpu(obs['sensor_data']['hand_camera']['rgb'])[0],tcp=cpu(env.unwrapped.agent.tcp.pose.p).reshape(3))
 def raw(r):return dict(agent={'qpos':torch.from_numpy(r['qpos']).reshape(1,-1)},sensor_data={'base_camera':{'rgb':torch.from_numpy(r['image']).unsqueeze(0)},'hand_camera':{'rgb':torch.from_numpy(r['wrist']).unsqueeze(0)}})
 def observed_undo(records,tg):
  if tg<5 or tg+5>=len(records):return None
  p=np.array([r['tcp'] for r in records]);w=np.array([r['qpos'][-2:].sum() for r in records]);previous=p[tg]-p[tg-5];after=p[tg+5]-p[tg];norm=np.linalg.norm(previous)
  back=max(0.,-float(after@previous)/norm) if norm>1e-8 else 0.;wp=w[tg]-w[tg-5];wa=w[tg+5]-w[tg];bg=max(0.,-np.sign(wp)*wa) if abs(wp)>1e-8 else 0.
  return float(np.hypot(back/.02,bg/.01))
 for episode in range(args.episodes):
  if cost>=3000:break
  split='id' if episode%2==0 else 'stage2_ood';seed=(930700 if split=='id' else 940700)+episode//2
  env=gym.make(stack_cube_env_id(split),num_envs=1,robot_uids='panda_wristcam',obs_mode='rgb',control_mode='pd_joint_delta_pos',reward_mode='sparse',render_mode='rgb_array',sim_backend='physx_cpu',render_backend='gpu',
   sim_config={'sim_freq':100,'control_freq':10},sensor_configs={'width':384,'height':384},max_episode_steps=400)
  obs,_=env.reset(seed=seed);reset=stack_cube_reset_metadata(env,split=split);records=[snapshot(env,obs)];actions=[];queries=[];prefix=[];takeover=None;success=False;oracle=None;expert_steps=0
  while not success and len(actions)<(150 if takeover is None else takeover+150):
   if takeover is None:
    prediction,feature,inputs,encoding=policy.predict(obs,instruction,seed=seed*1000+len(actions),steps=10)
    score=float(((feature.cuda().float()-mean)@basis).norm().item());z=feature.numpy().reshape(-1)
    proposed=gate.query(z,score);decision=score>cal['tau0'] if args.arm=='fixed' else proposed['stop']
    query=dict(step=len(actions),score=score,threshold=cal['tau0'] if args.arm=='fixed' else proposed['threshold'],stop=decision,baseline_stop=score>cal['tau0'],reason=proposed['reason'],memory_episodes=len(gate.memory_episodes),neighbors=proposed['neighbors'])
    queries.append(query);prefix.append(dict(feature=z,score=score,step=len(actions)))
    if decision:
     takeover=len(actions);oracle=StackCubePrivilegedChunkOracle(chunk_size=5);oracle.initialize_from_state(env)
    else:candidate=np.asarray(prediction).reshape(-1,8)[:5]
   if takeover is not None:
    if cost>=3000:break
    plan=oracle.plan(env);candidate=plan.actions
   for j,action in enumerate(candidate):
    if takeover is not None:
     if cost>=3000:break
     action=plan.action_at(obs['agent']['qpos'],j)
    action=np.asarray(action,dtype=np.float32).reshape(-1);assert action.shape==(8,) and np.isfinite(action).all();action=np.clip(action,-1,1)
    obs,_,terminated,truncated,info=env.step(torch.tensor(action,device=env.unwrapped.device).reshape(1,-1));actions.append(action);records.append(snapshot(env,obs))
    if takeover is not None:cost+=1;expert_steps+=1
    success=bool(info['success'])
    if success or bool(terminated) or bool(truncated):break
   if bool(terminated) or bool(truncated):break
  feedback=None;u=observed_undo(records,takeover) if takeover is not None else None
  if takeover is not None and expert_steps>=10:
   n=expert_steps;losses=np.full(n,np.nan);valid=np.minimum(10,n-np.arange(n));teacher=np.asarray(actions[takeover:]);feedback_started=time.time()
   for i in range(n-9):
    inputs=policy.prepare(raw(records[takeover+i]),instruction)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
     encoding=policy.model.forward_vlm(inputs['input_ids'],inputs['image_input'],inputs['image_mask']);errs=[]
     for mc in [0,1]:
      torch.manual_seed(270907+episode*10000+i*17+mc);pred=policy._generate_from_encoding(inputs,encoding,steps=10);pred=policy.model.action_space.postprocess(pred).float().clamp(-1,1)
      errs.append(float(((pred[0,:5,:8]-torch.tensor(teacher[i:i+5],device='cuda'))**2).mean().cpu())*100)
     losses[i]=np.mean(errs)
    if losses[i]>cal['q_loss']:break
   if args.arm.startswith('adaptive_'):
    feedback=gate.add_takeover_prefix_credit(f'{args.arm}:{episode}',np.array([p['feature'] for p in prefix]),np.array([p['score'] for p in prefix]),np.array([p['step'] for p in prefix]),losses,valid,
     regression=u,regression_bound=1.9004588285680322,progress_supported=args.arm=='adaptive_prefix' and u is not None)
   else:feedback=dict(reason='fixed_gate_no_update',first_observed_high=next((i for i,v in enumerate(losses) if np.isfinite(v) and v>cal['q_loss']),None))
   feedback['forward_seconds']=time.time()-feedback_started
  directory=args.output/f'episode_{episode:03d}';directory.mkdir(exist_ok=True)
  arrays=dict(qpos=np.array([r['qpos'] for r in records]),tcp=np.array([r['tcp'] for r in records]),actions=np.asarray(actions),policy_query_features=np.array([p['feature'] for p in prefix]),policy_query_scores=np.array([p['score'] for p in prefix]))
  if takeover is not None and expert_steps>=10:arrays['expert_feedback_loss']=losses
  buf=io.BytesIO();np.savez_compressed(buf,**arrays);(directory/'trace.npz').write_bytes(buf.getvalue())
  # Encode on local disk, then stream to OSS; frames are before each action.
  height,width=records[0]['image'].shape[:2]
  temporary=Path('/tmp/expert_feedback_pca_probe_v1')/f'{args.arm}_{episode:03d}.mp4';writer=cv2.VideoWriter(str(temporary),cv2.VideoWriter_fourcc(*'mp4v'),10,(2*width,height))
  assert writer.isOpened()
  for r in records[:-1]:writer.write(np.concatenate([r['image'],r['wrist']],axis=1)[:,:,::-1])
  writer.release();(directory/'video.mp4').write_bytes(temporary.read_bytes())
  row=dict(arm=args.arm,episode=episode,seed=seed,split=split,success=success,takeover=takeover,policy_steps=takeover if takeover is not None else len(actions),expert_actions=expert_steps,total_actions=len(actions),
   cumulative_expert_actions=cost,feedback=feedback,motion_undo=u,queries=queries,reset_metadata=reset,memory_size=len(gate.memory),video=str(directory/'video.mp4'))
  (directory/'result.json').write_text(json.dumps(row,indent=2));rows.append(row);(args.output/'progress.json').write_text(json.dumps(dict(pid=os.getpid(),episodes=len(rows),total=args.episodes,expert_actions=cost,elapsed=time.time()-started,last={k:v for k,v in row.items() if k not in ['queries','reset_metadata']}),indent=2))
  print('EPISODE',args.arm,episode,split,'success',success,'takeover',takeover,'expert',expert_steps,'feedback',feedback,flush=True);env.close()
 summary=dict(status='CLOSED_LOOP_COLLECTION_PILOT_COMPLETE',arm=args.arm,episodes=len(rows),requested=args.episodes,expert_actions=cost,elapsed=time.time()-started,rows=rows,
  interpretation='Assisted collection outcomes only; no SFT or downstream unassisted SR. Fixed episode counts, actual expert costs reported, not assumed matched.')
 (args.output/'summary.json').write_text(json.dumps(summary,indent=2));(args.output/'PILOT_COMPLETE.json').write_text(json.dumps(dict(arm=args.arm,episodes=len(rows),expert_actions=cost)));print('PILOT_COMPLETE',args.arm,len(rows),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['fixed','adaptive_current','adaptive_prefix'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True);p.add_argument('--pca',type=Path,required=True);p.add_argument('--episodes',type=int,default=20);main(p.parse_args())
