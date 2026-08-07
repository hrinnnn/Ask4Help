#!/usr/bin/env bash
set -euo pipefail

CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
ENV_NAME="${ENV_NAME:-openvla-airplane}"
OPENVLA_ROOT="${OPENVLA_ROOT:-/root/openvla-official}"
RESULT_ROOT="${RESULT_ROOT:-/mnt/data/ask4help/results/pick_single_ycb_airplane/openvla_original_lora_r32_v1}"

if [[ ! -x "$CONDA_ROOT/bin/conda" ]]; then
  installer=/tmp/Miniconda3-py310_24.11.1-0-Linux-x86_64.sh
  curl -fL --retry 5 -o "$installer" \
    https://repo.anaconda.com/miniconda/Miniconda3-py310_24.11.1-0-Linux-x86_64.sh
  bash "$installer" -b -p "$CONDA_ROOT"
fi

source "$CONDA_ROOT/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -n "$ENV_NAME" python=3.10 -y
fi
conda activate "$ENV_NAME"

python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install \
  accelerate==0.30.1 draccus==0.8.0 einops huggingface_hub json-numpy jsonlines \
  matplotlib peft==0.11.1 protobuf rich sentencepiece==0.1.99 \
  timm==0.9.10 tokenizers==0.19.1 transformers==4.40.1 wandb \
  packaging ninja numpy==1.26.4 pyarrow pandas scikit-learn pytest pillow opencv-python-headless==4.9.0.80
python -m pip install \
  tensorflow==2.15.0 tensorflow-datasets==4.9.3 tensorflow-graphics==2021.12.3 \
  "dlimp @ git+https://github.com/moojink/dlimp_openvla"
python -m pip install -e "$OPENVLA_ROOT" --no-deps
python -m pip install \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.5/flash_attn-2.5.5%2Bcu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

mkdir -p "$RESULT_ROOT/environment"
python - <<'PY'
import json
import platform
from pathlib import Path

import peft
import timm
import tokenizers
import torch
import torchvision
import transformers

out = Path("/mnt/data/ask4help/results/pick_single_ycb_airplane/openvla_original_lora_r32_v1/environment")
payload = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "transformers": transformers.__version__,
    "tokenizers": tokenizers.__version__,
    "timm": timm.__version__,
    "peft": peft.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "cuda_runtime": torch.version.cuda,
}
(out / "runtime.json").write_text(json.dumps(payload, indent=2))
print(json.dumps(payload, indent=2))
PY
python -m pip freeze > "$RESULT_ROOT/environment/pip-freeze.txt"
touch "$RESULT_ROOT/environment/READY"
