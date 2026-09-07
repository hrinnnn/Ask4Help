"""Post-collection common-feedback-window scoring; never feeds evaluated gates."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True);args=p.parse_args()
import torch,gymnasium as gym,mani_skill.envs
torch.set_num_threads(4);sys.path[:0]=['/root/Ask4Help-stage2-ood','/root/Ask4Help-stage2-ood/RLinf']
from tools.xvla_airplane_runtime import XVLAAirplanePolicy
from tools.stackcube_stage2_ood import register_stack_cube_splits,stack_cube_env_id
register_stack_cube_splits()
policy=XVLAAirplanePolicy(Path('/root/xvla_stage2_inputs_priority/ckpt-7500'),Path('/root/X-VLA-stackpyramid-clean'))
q=json.loads(args.calibration.read_text())['q_loss'];out=args.root/'common_preference_windows.json';done=json.loads(out.read_text()) if out.exists() else dict(rows=[],q=q,protocol='same free5 M2 over initial5 expert anchors; completeH10; posthoc only, not independent ground-truth timing')
completed={r['key'] for r in done['rows']}
for path in sorted(args.root.glob('seed_*/*/episode_*/result.json')):
 key=str(path.relative_to(args.root))
 if key in completed:continue
 r=json.loads(path.read_text());tg=r['takeover'];n=r['expert_actions']
 if tg is None or n<10:continue
 a=np.load(path.parent/'trace.npz');errors=[]
 env=gym.make(stack_cube_env_id(r['split']),num_envs=1,robot_uids='panda_wristcam',obs_mode='rgb',control_mode='pd_joint_delta_pos',reward_mode='sparse',render_mode='rgb_array',sim_backend='physx_cpu',render_backend='gpu',sim_config={'sim_freq':100,'control_freq':10},sensor_configs={'width':384,'height':384},max_episode_steps=400)
 obs,_=env.reset(seed=r['seed']);max_state_error=0.
 for j in range(tg):
  max_state_error=max(max_state_error,float(np.max(np.abs(obs['agent']['qpos'].cpu().numpy().reshape(-1)-a['qpos'][j]))))
  obs,_,_,_,_=env.step(torch.tensor(a['actions'][j],device=env.unwrapped.device).reshape(1,-1))
 for k in range(min(5,n-9)):
  max_state_error=max(max_state_error,float(np.max(np.abs(obs['agent']['qpos'].cpu().numpy().reshape(-1)-a['qpos'][tg+k]))))
  assert max_state_error<1e-4,(key,k,max_state_error)
  inp=policy.prepare(obs,'stack the red cube on the green cube');err=[]
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
   enc=policy.model.forward_vlm(inp['input_ids'],inp['image_input'],inp['image_mask'])
   for mc in [0,1]:
    torch.manual_seed(270907+r['episode']*10000+k*17+mc);pred=policy._generate_from_encoding(inp,enc,steps=10);pred=policy.model.action_space.postprocess(pred).float().clamp(-1,1)
    err.append(float(((pred[0,:5,:8]-torch.tensor(a['actions'][tg+k:tg+k+5],device='cuda'))**2).mean().cpu())*100)
  errors.append(float(np.mean(err)))
  if 'expert_feedback_loss' in a and np.isfinite(a['expert_feedback_loss'][k]):assert np.isclose(errors[-1],a['expert_feedback_loss'][k],rtol=2e-3,atol=2e-3),(key,k,errors[-1],a['expert_feedback_loss'][k])
  obs,_,_,_,_=env.step(torch.tensor(a['actions'][tg+k],device=env.unwrapped.device).reshape(1,-1))
 env.close();first=next((i for i,e in enumerate(errors) if e>q),None);u=r['motion_undo'];reversal=u is not None and u>1.9004588285680322
 status='late_cue' if reversal else ('preferred_proxy' if first is not None else ('early_cue' if len(errors)>=5 else 'censored_short'))
 row=dict(key=key,stream=path.parts[-4],variant=path.parts[-3],split=r['split'],seed=r['seed'],takeover=tg,errors=errors,first_high_initial5=first,motion_undo=u,status=status,max_state_error=max_state_error)
 done['rows'].append(row);out.write_text(json.dumps(done,indent=2));print(key,status,flush=True)
(args.root/'COMMON_PREFERENCE_COMPLETE.json').write_text(json.dumps(dict(rows=len(done['rows']),source='original action replay, state check, same384RGB; shared diagnostic protocol')))
