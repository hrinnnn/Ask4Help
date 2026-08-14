#!/usr/bin/env bash
set -euo pipefail

# Wait for a genuinely unused 5090 GPU, then run the validated smoke/reload
# checks before the long StackPyramid ID SFT stage.
ROOT=/data/zhaozhixuan/xvla_stackcube_data
PY=/data/zhaozhixuan/envs/xvla_official_5090/bin/python
XVLA=/data/zhaozhixuan/X-VLA
COLLECTION=$ROOT/stackpyramid_formal_collection_v2
RUN_ROOT=$ROOT/stackpyramid_id_sft_10000_v1
FORMAL=$RUN_ROOT/formal_id_sft
LOG=$RUN_ROOT/pipeline.log
STATE=$RUN_ROOT/pipeline_state.txt

mkdir -p "$RUN_ROOT"
exec >>"$LOG" 2>&1

write_state() {
  printf '%s\n' "$1" >"$STATE"
  printf '[%s] %s\n' "$(date -Is)" "$1"
}

write_state waiting_for_idle_gpu
while true; do
  for gpu in 0 1 2 3 4 5 6 7; do
    if ! nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; then
      export CUDA_VISIBLE_DEVICES="$gpu"
      write_state "gpu_${gpu}_selected"
      break 2
    fi
  done
  sleep 300
done

if [ ! -f "$RUN_ROOT/smoke/TRAINING_COMPLETE" ]; then
  write_state smoke
  "$PY" "$ROOT/run_stackpyramid_xvla_training.py" \
    --xvla-root "$XVLA" \
    --model /data/zhaozhixuan/Ask4Help-airplane-5090/results/xvla_airplane_v1/model_cache/X-VLA-Pt-local \
    --collection-root "$COLLECTION" --split id --target-episodes 2 \
    --output "$RUN_ROOT/smoke" --steps 2 --save-interval 2 \
    --batch-size 1 --freeze-steps 0 --dtype bf16 --log-interval 1
fi

if [ ! -f "$RUN_ROOT/smoke_reload_ok.json" ]; then
  write_state reload_smoke
  "$PY" - <<'PY'
import json, math, os, sys
from pathlib import Path
import torch
sys.path.insert(0, "/data/zhaozhixuan/X-VLA")
sys.path.insert(0, "/data/zhaozhixuan/xvla_stackcube_data")
from run_stackpyramid_xvla_training import StackPyramidH5Dataset, collate_fn, load_model, masked_flow_loss
model, processor = load_model(Path("/data/zhaozhixuan/xvla_stackcube_data/stackpyramid_id_sft_10000_v1/smoke/ckpt-2"), Path("/data/zhaozhixuan/X-VLA"), dtype="bf16")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = StackPyramidH5Dataset(Path("/data/zhaozhixuan/xvla_stackcube_data/stackpyramid_formal_collection_v2"), "id", target_episodes=2, horizon=10)
batch = collate_fn([dataset[len(dataset)-1]], processor)
model.to(device).eval()
batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
with torch.no_grad():
    value = float(masked_flow_loss(model, batch).float().item())
assert math.isfinite(value)
Path("/data/zhaozhixuan/xvla_stackcube_data/stackpyramid_id_sft_10000_v1/smoke_reload_ok.json").write_text(json.dumps({"finite_loss": value, "device": str(device), "tail_valid_count": int(batch["action_valid_mask"].sum().item())}) + "\n")
PY
fi

if [ ! -f "$FORMAL/TRAINING_COMPLETE" ]; then
  write_state formal_id_sft
  "$PY" "$ROOT/run_stackpyramid_xvla_training.py" \
    --xvla-root "$XVLA" \
    --model /data/zhaozhixuan/Ask4Help-airplane-5090/results/xvla_airplane_v1/model_cache/X-VLA-Pt-local \
    --collection-root "$COLLECTION" --split id --target-episodes 128 \
    --output "$FORMAL" --steps 10000 --save-interval 500 \
    --batch-size 8 --freeze-steps 1000 --warmup-steps 2000 \
    --dtype bf16 --log-interval 20
fi

write_state training_complete
