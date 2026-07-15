#!/usr/bin/env bash
set -euo pipefail

RLINF_ROOT=${RLINF_ROOT:-/root/Ask4Help/RLinf}
PYTHON=${PYTHON:-"${RLINF_ROOT}/.venv/bin/python"}

cd "${RLINF_ROOT}"
"${PYTHON}" - <<'PY'
import gymnasium as gym
import torch

import mani_skill  # noqa: F401
import openpi  # noqa: F401
import ray  # noqa: F401
import rlinf  # noqa: F401
import transformers  # noqa: F401
from rlinf.envs.maniskill.peg_insertion_side_variants import (
    PEG_INSERTION_SIDE_WIDE_ENV_ID,
    register_rlinf_peg_insertion_side_variants,
)

print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
assert torch.cuda.is_available(), "CUDA is required for the ManiSkill RLT smoke"

register_rlinf_peg_insertion_side_variants()
env = gym.make(
    PEG_INSERTION_SIDE_WIDE_ENV_ID,
    num_envs=1,
    obs_mode="rgb",
    control_mode="pd_joint_delta_pos",
    sim_backend="gpu",
    reward_mode="sparse",
    sim_config={"sim_freq": 100, "control_freq": 10},
    sensor_configs={"width": 384, "height": 384},
)
obs, info = env.reset(seed=0)
sensor_data = obs["sensor_data"]
qpos = obs["agent"]["qpos"]
print(f"camera_keys={sorted(sensor_data)}")
print(f"qpos_shape={tuple(qpos.shape)}")
assert qpos.shape[-1] >= 9

action = torch.zeros((1, 8), dtype=torch.float32, device=qpos.device)
_, _, _, _, info = env.step(action)
print(f"success_shape={tuple(info['success'].shape)}")
env.close()
print("RLT Peg GPU environment smoke passed")
PY
