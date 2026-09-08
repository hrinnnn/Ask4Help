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

    @staticmethod
    def snapshot(env, raw):
        def array(v): return v.detach().cpu().numpy().copy()
        return {'main':array(raw['sensor_data']['base_camera']['rgb'])[0],
                'wrist':array(raw['sensor_data']['hand_camera']['rgb'])[0],
                'qpos':array(raw['agent']['qpos'])[0],
                'tcp':array(env.unwrapped.agent.tcp.pose.p)[0]}

    @staticmethod
    def clip(actions, env):
        low=np.asarray(env.action_space.low).reshape(-1)
        high=np.asarray(env.action_space.high).reshape(-1)
        return np.clip(actions[:,:8],low,high).astype(np.float32)
