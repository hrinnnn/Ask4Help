"""Version-pinned task adapters for the pi0.5 feedback ablation.

The historical model/task source roots are explicit immutable dependencies,
separate from this versioned experiment code and from the Python environment.
"""
import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
from pi05_timing_feedback import isolated_python_numpy_rng

SOURCE_ROOTS = {
    'stackcube_legacy_ood': Path('/root/Ask4Help-online-awbc-code'),
    'airplane_yaw_ood': Path('/root/Ask4Help-pick-airplane-four-group'),
    'open_drawer_grasp_ood': Path('/root/Ask4Help-pick-airplane-four-group'),
    'open_drawer_goal_ood': Path('/root/Ask4Help-pick-airplane-four-group'),
}


class Pi05FeedbackRuntime:
    def __init__(self, task, task_assets, source_root=None):
        self.task = task
        self.assets = task_assets
        self.source = Path(source_root or SOURCE_ROOTS[task])
        sys.path[:0] = [str(self.source), str(self.source/'RLinf')]
        helper_path = self.source/'tools/maniskill_pi05_vfd_online_awbc.py'
        spec = importlib.util.spec_from_file_location('_feedback_legacy_model', helper_path)
        self.helper = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.helper
        spec.loader.exec_module(self.helper)
        import torch
        self.torch = torch
        self.model = None
        if task == 'stackcube_legacy_ood':
            from rlinf.envs.maniskill.stack_cube_variants import STACK_CUBE_TASK, reset_metadata
            self.instruction = STACK_CUBE_TASK
            self.reset_metadata = reset_metadata
            self.horizon = 100
        elif task.startswith('open_drawer_'):
            import rlinf.envs.maniskill as package
            snapshot_path=Path(__file__).parent/'runtime_snapshots/opendrawer'
            package.__path__=[str(snapshot_path),*package.__path__]
            from rlinf.envs.maniskill.open_drawer_retrieve_place_spec import TASK_INSTRUCTION, ENV_IDS, reset_metadata
            import rlinf.envs.maniskill.open_drawer_retrieve_place
            self.instruction=TASK_INSTRUCTION;self.open_drawer_ids=ENV_IDS
            self.ood_split='grasp_ood' if task.endswith('grasp_ood') else 'goal_ood'
            self.reset_metadata=lambda env,split:reset_metadata(env,split=split if split=='id' else self.ood_split)
            self.horizon=400
        else:
            from rlinf.envs.maniskill.pick_single_ycb_airplane_variants import PICK_SINGLE_YCB_AIRPLANE_TASK, reset_metadata
            from toolkits.lerobot.collect_maniskill_pick_single_ycb_airplane_lerobot import _build_env
            self.instruction = PICK_SINGLE_YCB_AIRPLANE_TASK
            self.reset_metadata = reset_metadata
            self.airplane_build_env = _build_env
            self.horizon = 250

    def provenance(self):
        def revision(path):
            return subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
        return {'task':self.task,'instruction':self.instruction,
                'source_root':str(self.source),'source_commit':revision(self.source),
                'rlinf_commit':revision(self.source/'RLinf'),
                'python':sys.executable,'torch':self.torch.__version__,
                'checkpoint':self.assets['checkpoint'],'norm':self.assets['norm'],
                'action_dimension':8,'execute_horizon':5,'sensor_resolution':[384,384]}

    def load_model(self):
        self.model = self.helper._load_model(Path(self.assets['checkpoint']),
                                            Path(self.assets['norm']),
                                            Path('/mnt/data/ask4help/models/pi05_base_torch'))

    def build_env(self, split):
        if self.task == 'stackcube_legacy_ood':
            return self.helper._build_env(self.horizon,task='stack',split=split,sim_backend='physx_cpu')
        if self.task.startswith('open_drawer_'):
            import gymnasium as gym
            actual=split if split=='id' else self.ood_split
            return gym.make(self.open_drawer_ids[actual],robot_uids='panda_wristcam',num_envs=1,obs_mode='rgb',
                            control_mode='pd_joint_delta_pos',sim_backend='physx_cpu',reward_mode='sparse',
                            sim_config={'sim_freq':100,'control_freq':10},sensor_configs={'width':384,'height':384},
                            max_episode_steps=400,render_mode='rgb_array')
        args = argparse.Namespace(split=split,image_size=384,control_freq=10,
                                  max_episode_steps=self.horizon,sim_backend='physx_cpu')
        return self.airplane_build_env(args,control_mode='pd_joint_delta_pos')

    def observation(self, raw):
        state=raw['agent']['qpos']
        return {'main_images':raw['sensor_data']['base_camera']['rgb'],
                'wrist_images':raw['sensor_data']['hand_camera']['rgb'],
                'extra_view_images':None,'states':state,
                'task_descriptions':[self.instruction],
                'task_ids':self.torch.zeros(1,dtype=self.torch.long,device=state.device)}

    def predict(self, raw, rng_seed):
        torch=self.torch
        with isolated_python_numpy_rng(rng_seed), torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(int(rng_seed));torch.cuda.manual_seed_all(int(rng_seed))
            with torch.inference_mode():
                actions, result=self.model.predict_action_batch(env_obs=self.observation(raw),mode='train')
        return actions.detach().float().cpu().numpy()[0],result['forward_inputs']['model_action']

    def bridge(self, raw, model_actions):
        torch=self.torch
        with torch.inference_mode():
            prior=torch.zeros_like(model_actions).reshape(-1,self.model.config.action_horizon,self.model.config.action_dim)
            layers=self.model.extract_multilayer_llmd_features(self.observation(raw),prior)
        return layers['vlm_bridge_final_mean'].detach().float().cpu().numpy().reshape(-1)

    def reference_bridge(self, raw):
        """VLM-only reference extraction; checked against bridge() by builder."""
        from openpi.models import model as observation_model
        from rlinf.algorithms.vla_fail import pool_valid_prefix_tokens
        torch=self.torch
        with torch.inference_mode():
            processed=self.model.precision_processor(self.model.input_transform(self.model.obs_processor(self.observation(raw)),transpose=False))
            observation=observation_model.Observation.from_dict(processed)
            images,masks,tokens,token_masks,state=self.model._preprocess_observation(observation,train=False)
            device=state.device
            images=[x.to(device) for x in images];masks=[x.to(device) for x in masks]
            if tokens is not None:tokens=tokens.to(device)
            if token_masks is not None:token_masks=token_masks.to(device)
            prefix,valid,_cache=self.model._build_prefix_cache(images,masks,tokens,token_masks)
            return pool_valid_prefix_tokens(prefix,valid).detach().float().cpu().numpy().reshape(-1)

    @staticmethod
    def snapshot(env, raw):
        def array(v): return v.detach().cpu().numpy().copy()
        base=env.unwrapped
        obj=base.cubeA if hasattr(base,'cubeA') else base.obj
        snapshot={'main':array(raw['sensor_data']['base_camera']['rgb'])[0],
                'wrist':array(raw['sensor_data']['hand_camera']['rgb'])[0],
                'qpos':array(raw['agent']['qpos'])[0],
                'tcp':array(base.agent.tcp.pose.p)[0],
                'tcp_q':array(base.agent.tcp.pose.q)[0],
                'object_p':array(obj.pose.p)[0],
                'object_q':array(obj.pose.q)[0],
                'grasped':bool(base.agent.is_grasping(obj))}
        if hasattr(base,'drawer'):
            evaluation=base.evaluate()
            snapshot.update(drawer_qpos=array(base.drawer.get_qpos())[0],
                            target_p=array(base.target_tray.pose.p)[0])
            for key in ['ever_drawer_opened','ever_grasped','ever_lifted',
                        'object_in_target','object_released','is_robot_static','success']:
                snapshot[key]=bool(evaluation[key])
        return snapshot

    def opendrawer_expert(self, env, raw, seed, remaining_actions):
        """Original direct-grasp oracle, from live state, with a hard endpoint."""
        import toolkits.lerobot as package
        snapshot_path=Path(__file__).parent/'runtime_snapshots/opendrawer'
        if str(snapshot_path) not in package.__path__:
            package.__path__.insert(0,str(snapshot_path))
        from toolkits.lerobot.validate_open_drawer_retrieve_place_oracle import PandaPosePlannerClient
        from toolkits.lerobot.collect_maniskill_peg_lerobot_joint import (
            _joint_delta_arm_bounds, _convert_solver_action_to_joint_delta)
        spec=importlib.util.spec_from_file_location('_feedback_direct_drawer_oracle',
                                                   Path(__file__).parent/'open_drawer_direct_takeover_oracle.py')
        oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
        lower,upper=_joint_delta_arm_bounds(env)
        actions=[];snapshots=[];runtime=self;stages={}
        class Endpoint(Exception):
            def __init__(self,success,reason):self.success=success;self.reason=reason
        class Proxy:
            def record_expert_stages(self,value):
                nonlocal stages
                stages=value
            @property
            def unwrapped(self):return env.unwrapped
            def __getattr__(self,name):return getattr(env,name)
            def step(self,solver_action,*args,**kwargs):
                qpos=env.unwrapped.agent.robot.get_qpos().detach().cpu().numpy().reshape(-1)
                action=_convert_solver_action_to_joint_delta(qpos,solver_action,lower,upper)
                result=env.step(action,*args,**kwargs)
                actions.append(np.asarray(action,dtype=np.float32).copy())
                snapshots.append(runtime.snapshot(env,result[0]))
                if bool(result[4]['success']):raise Endpoint(True,'task_success')
                if bool(result[2]) or bool(result[3]) or len(actions)>=remaining_actions:
                    raise Endpoint(False,'episode_horizon_or_termination')
                return result
        if remaining_actions<=0:raise ValueError('no expert action budget remains')
        planner=PandaPosePlannerClient()
        try:
            with isolated_python_numpy_rng(seed+600000):
                report=oracle.continue_episode(Proxy(),planner,seed=seed)
            report['accepted']=bool(report['success'])
        except Endpoint as endpoint:
            report={**stages,'accepted':endpoint.success,'success':endpoint.success,
                    'termination_reason':endpoint.reason,'takeover_from_current_state':True,
                    'oracle_mode':'direct_grasp_from_current_state'}
        finally:
            planner.close()
        return {'actions':actions,'snapshots':snapshots,'all_actions':actions,
                'all_snapshots':snapshots,'attempt_lengths':[len(actions)],'report':report}

    def raw_snapshot(self, snapshot):
        torch=self.torch
        return {'agent':{'qpos':torch.as_tensor(snapshot['qpos']).reshape(1,-1)},
                'sensor_data':{'base_camera':{'rgb':torch.as_tensor(snapshot['main']).unsqueeze(0)},
                               'hand_camera':{'rgb':torch.as_tensor(snapshot['wrist']).unsqueeze(0)}}}

    def airplane_expert(self, env, raw, seed, remaining_actions):
        """Use the original validated expert, recording every executed retry.

        Only the final candidate is a continuous retained suffix. Failed
        candidate attempts remain separate and contribute to total expert cost.
        """
        path=self.source/'tools/collect_pick_single_ycb_airplane_gated_dagger.py'
        spec=importlib.util.spec_from_file_location('_feedback_legacy_airplane_collector',path)
        collector=importlib.util.module_from_spec(spec);sys.modules[spec.name]=collector;spec.loader.exec_module(collector)
        lower,upper=collector._joint_delta_arm_bounds(env)
        actual_actions=[];actual_snapshots=[];original_step=env.step
        original_restore=env.unwrapped.set_state_dict;boundaries=[]
        class EpisodeEndpoint(Exception):
            def __init__(self,success,reason):self.success=success;self.reason=reason
        def recorded_restore(state,*args,**kwargs):
            boundaries.append(len(actual_actions))
            return original_restore(state,*args,**kwargs)
        def recorded_step(action,*args,**kwargs):
            result=original_step(action,*args,**kwargs)
            a=action.detach().cpu().numpy() if hasattr(action,'detach') else np.asarray(action)
            actual_actions.append(a.astype(np.float32).reshape(-1).copy())
            actual_snapshots.append(self.snapshot(env,result[0]))
            if bool(result[4]['success']):raise EpisodeEndpoint(True,'task_success')
            if bool(result[2]) or bool(result[3]) or len(actual_actions)>=remaining_actions:
                raise EpisodeEndpoint(False,'episode_horizon_or_termination')
            return result
        env.step=recorded_step;env.unwrapped.set_state_dict=recorded_restore
        try:
            with isolated_python_numpy_rng(seed):
                _records,selected_actions,report=collector._plan_and_execute_expert(env,None,seed=seed,raw_obs=raw,lower=lower,upper=upper)
        except EpisodeEndpoint as endpoint:
            lengths=np.diff(boundaries+[len(actual_actions)]).tolist()
            report={'accepted':endpoint.success,'termination_reason':endpoint.reason,
                    'attempts':[{'candidate':collector.ORACLE_NECK_CANDIDATES[i][0],'delta_servo_substeps':n,
                                 'outcome':'endpoint' if i==len(lengths)-1 else 'rejected_candidate'} for i,n in enumerate(lengths)]}
            selected_actions=actual_actions[boundaries[-1]:] if endpoint.success else []
        finally:
            env.step=original_step;env.unwrapped.set_state_dict=original_restore
        lengths=[int(a.get('delta_servo_substeps',0)) for a in report['attempts']]
        assert sum(lengths)==len(actual_actions)
        n=len(selected_actions) if report['accepted'] else (lengths[-1] if lengths else 0)
        return {'actions':actual_actions[-n:] if n else [],'snapshots':actual_snapshots[-n:] if n else [],
                'all_actions':actual_actions,'all_snapshots':actual_snapshots,
                'attempt_lengths':lengths,'report':report}

    @staticmethod
    def clip(actions, env):
        low=np.asarray(env.action_space.low).reshape(-1)
        high=np.asarray(env.action_space.high).reshape(-1)
        return np.clip(actions[:,:8],low,high).astype(np.float32)
