"""Replay only the historical discarded candidates before saved expert actions.

No candidate, action, or seed is selected by downstream utility. The accepted
suffix is always the original saved action sequence, validated against qpos.
"""
import json,types,sys
from pathlib import Path
import numpy as np

def restore_candidate_history(env,row):
 path=Path('/root/Ask4Help-pick-airplane-four-group/RLinf/toolkits/lerobot/diagnose_pick_single_ycb_airplane_oracle.py')
 mod=types.ModuleType('tasr_historical_airplane_oracle');mod.__file__=str(path);sys.modules[mod.__name__]=mod;exec(path.read_text(),mod.__dict__)
 base=env.unwrapped;state=base.get_state_dict();history=[]
 for attempt in row['metadata']['oracle']['attempts'][:-1]:
  base.set_state_dict(state);original_step=env.step;count=[0]
  def step_hook(solver_action,*args,**kwargs):
   result=None
   for _ in range(4):
    q=base.agent.robot.get_qpos()[0].detach().cpu().numpy()
    if result is not None and np.max(np.abs(q[:7]-solver_action[:7]))<=.012:break
    target=np.asarray(solver_action,dtype=np.float32).reshape(-1)
    lower=np.full(7,-.1,dtype=np.float32);upper=np.full(7,.1,dtype=np.float32)
    clipped=np.clip(target[:7]-q[:7],lower,upper);arm=2*(clipped-lower)/(upper-lower)-1
    action=np.r_[arm,target[7:8]].astype(np.float32)
    result=original_step(action,*args,**kwargs);count[0]+=1
   return result
  env.step=step_hook
  try:
   result=mod.try_candidate(env,seed=row['seed'],name=attempt['candidate'],local_point=np.asarray(attempt['local_point']),
    close_steps=attempt.get('close_steps',60),complete_task=True,reset_before_attempt=False,force_planner_pd_joint_pos=True,
    closing_sign=-1.0 if attempt['candidate'].endswith('_flip') else 1.0)
  finally:env.step=original_step
  history.append(dict(candidate=attempt['candidate'],steps=count[0],recorded_steps=attempt.get('delta_servo_substeps'),accepted=bool(result['accepted']),recorded_accepted=bool(attempt['accepted'])))
 base.set_state_dict(state)
 print('HISTORICAL_DISCARDED_CANDIDATES',json.dumps(history),flush=True)
 return history
