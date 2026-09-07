"""Fresh-seed evaluation-only takeover branches; no feedback enters a gate."""
import argparse,io,json,sys,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args();args.out.mkdir(exist_ok=True,parents=True)
import torch,gymnasium as gym,mani_skill.envs
torch.set_num_threads(4);sys.path[:0]=['/root/Ask4Help-stage2-ood','/root/Ask4Help-stage2-ood/RLinf']
from tools.xvla_airplane_runtime import XVLAAirplanePolicy
from tools.stackcube_stage2_ood import register_stack_cube_splits,stack_cube_env_id,stack_cube_reset_metadata
from rlinf.envs.maniskill.stack_cube_privileged_oracle import StackCubePrivilegedChunkOracle
register_stack_cube_splits();policy=XVLAAirplanePolicy(Path('/root/xvla_stage2_inputs_priority/ckpt-7500'),Path('/root/X-VLA-stackpyramid-clean'));rows=[]
def build():return gym.make(stack_cube_env_id('stage2_ood'),num_envs=1,robot_uids='panda_wristcam',obs_mode='rgb',control_mode='pd_joint_delta_pos',reward_mode='sparse',render_mode='rgb_array',sim_backend='physx_cpu',render_backend='gpu',sim_config={'sim_freq':100,'control_freq':10},sensor_configs={'width':384,'height':384},max_episode_steps=400)
def snap(env,obs):
 return dict(qpos=obs['agent']['qpos'].cpu().numpy().reshape(-1).copy(),tcp=env.unwrapped.agent.tcp.pose.p.cpu().numpy().reshape(-1).copy(),image=obs['sensor_data']['base_camera']['rgb'].cpu().numpy().copy(),wrist=obs['sensor_data']['hand_camera']['rgb'].cpu().numpy().copy())
def raw(s):return dict(agent={'qpos':torch.tensor(s['qpos']).reshape(1,-1)},sensor_data={'base_camera':{'rgb':torch.tensor(s['image'])},'hand_camera':{'rgb':torch.tensor(s['wrist'])}})
def save_npz(path,**arrays):
    stream=io.BytesIO();np.savez_compressed(stream,**arrays);path.write_bytes(stream.getvalue())
for seed in range(944000,944005):
 env=build();obs,_=env.reset(seed=seed);meta=stack_cube_reset_metadata(env,split='stage2_ood');actions=[];states=[snap(env,obs)];done=False
 while len(actions)<100 and not done:
  pred,_,_,_=policy.predict(obs,'stack the red cube on the green cube',seed=seed*1000+len(actions),steps=10)
  for action in np.asarray(pred).reshape(-1,8)[:5]:
   action=np.clip(action,-1,1).astype(np.float32);obs,_,term,trunc,info=env.step(torch.tensor(action,device=env.unwrapped.device).reshape(1,-1));actions.append(action);states.append(snap(env,obs));done=bool(term) or bool(trunc) or bool(info['success'])
   if done:break
 env.close();base=args.out/f'seed_{seed}';base.mkdir(exist_ok=True)
 save_npz(base/'policy_prefix.npz',actions=np.array(actions),qpos=np.array([s['qpos'] for s in states]),tcp=np.array([s['tcp'] for s in states]))
 for t in [0,10,20,30,40,60,80]:
  if t>len(actions):continue
  env=build();obs,_=env.reset(seed=seed)
  for action in actions[:t]:obs,_,_,_,_=env.step(torch.tensor(action,device=env.unwrapped.device).reshape(1,-1))
  np.testing.assert_allclose(obs['agent']['qpos'].cpu().numpy().reshape(-1),states[t]['qpos'],atol=1e-4)
  oracle=StackCubePrivilegedChunkOracle(chunk_size=5);oracle.initialize_from_state(env);expert=[];es=[snap(env,obs)];success=False
  while len(expert)<150 and not success:
   plan=oracle.plan(env)
   for j in range(len(plan.actions)):
    action=np.clip(np.asarray(plan.action_at(obs['agent']['qpos'],j),dtype=np.float32),-1,1);obs,_,term,trunc,info=env.step(torch.tensor(action,device=env.unwrapped.device).reshape(1,-1));expert.append(action);es.append(snap(env,obs));success=bool(info['success'])
    if success or bool(term) or bool(trunc) or len(expert)>=150:break
   if bool(term) or bool(trunc):break
  e=[]
  for k in range(min(5,max(0,len(expert)-9))):
   inp=policy.prepare(raw(es[k]),'stack the red cube on the green cube');errs=[]
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
    enc=policy.model.forward_vlm(inp['input_ids'],inp['image_input'],inp['image_mask'])
    for mc in [0,1]:
     torch.manual_seed(290907+seed*100+k*17+mc);pred=policy._generate_from_encoding(inp,enc,steps=10);pred=policy.model.action_space.postprocess(pred).float().clamp(-1,1)
     errs.append(float(((pred[0,:5,:8]-torch.tensor(np.array(expert[k:k+5]),device='cuda'))**2).mean().cpu())*100)
   e.append(float(np.mean(errs)))
  u=None
  if t>=5 and len(expert)>=5:
   before=states[t]['tcp']-states[t-5]['tcp'];after=es[5]['tcp']-es[0]['tcp'];norm=np.linalg.norm(before);pundo=max(0.,-float(after@before)/norm) if norm>1e-8 else 0.
   wbefore=states[t]['qpos'][-2:].sum()-states[t-5]['qpos'][-2:].sum();wafter=es[5]['qpos'][-2:].sum()-es[0]['qpos'][-2:].sum();wundo=max(0.,-np.sign(wbefore)*wafter) if abs(wbefore)>1e-8 else 0.;u=float(np.hypot(pundo/.02,wundo/.01))
  from PIL import Image
  folder=base/f'takeover_{t}';folder.mkdir(exist_ok=True)
  for label,index in [('takeover',0),('expert5',min(5,len(expert))),('terminal',len(expert))]:Image.fromarray(np.concatenate([es[index]['image'][0],es[index]['wrist'][0]],axis=1)).save(folder/(label+'.png'))
  save_npz(folder/'expert.npz',actions=np.array(expert),qpos=np.array([s['qpos'] for s in es]),tcp=np.array([s['tcp'] for s in es]))
  r=dict(seed=seed,takeover=t,success=success,expert_actions=len(expert),initial_free5_error=e,motion_undo=u,reset=meta);rows.append(r);(folder/'result.json').write_text(json.dumps(r,indent=2));(args.out/'summary.json').write_text(json.dumps(dict(rows=rows,scope='Evaluation-only35branches over5fresh seeds; not SFT, no expert warmstart into adaptive gates'),indent=2));print(seed,t,success,len(expert),flush=True);env.close()
(args.out/'COUNTERFACTUAL_COMPLETE.json').write_text(json.dumps(dict(rows=len(rows),seeds=5)))
