#!/usr/bin/env python3
"""CPU/no-render replay to recover missing object quaternions; fails on drift."""
import argparse,json,time,sys
from pathlib import Path
import numpy as np


def main(args):
    sys.path[:0]=[str(args.repo_root),str(args.repo_root/'RLinf')]
    import torch
    torch.set_num_threads(2)
    import gymnasium as gym
    import mani_skill.envs
    from tools.stackcube_stage2_ood import register_stack_cube_splits,stack_cube_env_id
    from rlinf.envs.maniskill.pick_single_ycb_airplane_variants import register_controlled_pick_single_ycb_airplane_variants,PICK_SINGLE_YCB_AIRPLANE_OOD_ENV_ID
    register_stack_cube_splits();register_controlled_pick_single_ycb_airplane_variants()
    rows=json.loads(args.manifest.read_text());rows=rows[:args.limit] if args.limit else rows
    args.output.mkdir(parents=True,exist_ok=True);started=time.time();completed=[]
    for task in ['stackcube','airplane']:
        subset=[r for r in rows if r['task']==task]
        if not subset:continue
        env=gym.make(stack_cube_env_id('ood') if task=='stackcube' else PICK_SINGLE_YCB_AIRPLANE_OOD_ENV_ID,
            robot_uids='panda_wristcam',num_envs=1,obs_mode='none',control_mode='pd_joint_delta_pos',
            reward_mode='sparse',render_mode=None,sim_backend='physx_cpu',
            sim_config={'sim_freq':100,'control_freq':10},max_episode_steps=400)
        base=env.unwrapped
        def snapshot():
            obj=base.cubeA if task=='stackcube' else base.obj
            return (base.agent.tcp.pose.p.reshape(-1).detach().cpu().numpy().copy(),
                    obj.pose.p.reshape(-1).detach().cpu().numpy().copy(),
                    obj.pose.q.reshape(-1).detach().cpu().numpy().copy(),
                    bool(base.agent.is_grasping(obj)))
        try:
            for row in subset:
                key=f"{task}_step_{row['step']}_seed_{row['seed']}";file=args.output/f'{key}.npz'
                if file.exists():
                    completed.append(dict(key=key,cached=True));continue
                actions=np.load(row['actions']);expected=np.load(row['task_states'])
                env.reset(seed=row['seed']);observed=[snapshot()]
                for action in actions:
                    env.step(torch.as_tensor(action,dtype=torch.float32,device=base.device).reshape(1,-1));observed.append(snapshot())
                tcp=np.array([s[0] for s in observed]);obj=np.array([s[1] for s in observed]);quat=np.array([s[2] for s in observed]);grasp=np.array([s[3] for s in observed])
                pe=float(np.max(abs(tcp-expected[:,6:9])));oe=float(np.max(abs(obj-expected[:,:3])));ce=int(np.sum(grasp!=(expected[:,-2]>.5)))
                result=dict(key=key,steps=len(actions),tcp_max_error=pe,object_max_error=oe,contact_mismatches=ce)
                if pe>1e-4 or oe>1e-4 or ce:
                    (args.output/'REPLAY_MISMATCH.json').write_text(json.dumps(result,indent=2))
                    raise RuntimeError(result)
                np.savez_compressed(file,object_quaternion=quat,object_position=obj,tcp_position=tcp,grasped=grasp)
                completed.append(result)
                print(json.dumps(result),flush=True)
                (args.output/'progress.json').write_text(json.dumps(dict(completed=len(completed),total=len(rows),elapsed=time.time()-started,last=result)))
        finally:env.close()
    (args.output/'REPLAY_COMPLETE.json').write_text(json.dumps(dict(rows=completed,count=len(completed),elapsed=time.time()-started),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo-root',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int)
    main(p.parse_args())
