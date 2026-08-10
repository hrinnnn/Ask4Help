#!/usr/bin/env python3
"""Train four X-VLA StackCube adaptation groups in parallel on four GPUs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


METHODS = ("vlm_bridge_pca", "offline_oracle", "failure_recovery", "diffdagger")
ROOT = Path(os.environ.get("ASK4HELP_ROOT", Path(__file__).resolve().parents[1]))
RESULT = Path(os.environ["XVLA_STACKCUBE_FOUR_GROUP_RESULT"])
RUN = Path(os.environ.get("XVLA_STACKCUBE_FOUR_GROUP_TRAIN_RUN", RESULT / "training_v1_5000"))
XVLA = Path("/data/zhaozhixuan/X-VLA")
ENV = Path("/data/zhaozhixuan/envs/xvla_official_5090")
START = Path(
    "/data/zhaozhixuan/Ask4Help-airplane-5090/results/xvla_stackcube_v1/"
    "temporal_mask_v2/id_sft_from3500_to10000_official_2gpu_retry1/ckpt-7500"
)
ID_META = Path(
    "/data/zhaozhixuan/Ask4Help-airplane-5090/results/xvla_stackcube_v1/"
    "manifests/panda_stackcube_id_128.json"
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_training_meta(method: str) -> Path:
    output = RUN / "metas" / method
    output.mkdir(parents=True, exist_ok=False)
    original = read_json(ID_META)
    original["dataset_name"] = f"panda_stackcube_id_replay_{method}"
    write_json(output / "00_id_replay.json", original)

    dataset = RESULT / "datasets" / method
    info = read_json(dataset / "meta/info.json")
    episodes = [
        json.loads(line)
        for line in (dataset / "meta/episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(episodes) != 100 or any(int(row["length"]) < 1 for row in episodes):
        raise RuntimeError(f"invalid {method} expert dataset")
    write_json(
        output / "01_new_expert.json",
        {
            "dataset_name": f"panda_stackcube_new_expert_{method}",
            "robot_type": "panda_airplane",
            "root_path": str(dataset.resolve()),
            "chunks_size": int(info["chunks_size"]),
            "data_path": info["data_path"],
            "datalist": episodes,
        },
    )
    return output


def training_command(meta: Path, output: Path, *, steps: int, save_interval: int) -> list[str]:
    return [
        str(ENV / "bin/python"),
        "train.py",
        "--models", str(START),
        "--output_dir", str(output),
        "--train_metas_path", str(meta),
        "--batch_size", "8",
        "--num_actions", "10",
        "--num_views", "2",
        "--action_mode", "auto",
        "--real_action_dim", "8",
        "--max_action_dim", "20",
        "--distributed_backend", "gloo",
        "--gradient_checkpointing",
        "--gradient_accumulation_steps", "4",
        "--learning_rate", "1e-4",
        "--learning_coef", "0.1",
        "--weight_decay", "0",
        "--betas", "0.9", "0.95",
        "--max_grad_norm", "1.0",
        "--freeze_steps", "1000",
        "--warmup_steps", "2000",
        "--iters", str(steps),
        "--save_interval", str(save_interval),
        "--log_interval", "20",
        "--seed", "6200",
    ]


def child_env(gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "OMP_NUM_THREADS": "20",
            "MKL_NUM_THREADS": "20",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "ACCELERATE_MIXED_PRECISION": "bf16",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(29720 + gpu),
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
            "LOCAL_WORLD_SIZE": "1",
        }
    )
    return env


def launch_wave(metas: dict[str, Path], phase: str, steps: int, save_interval: int) -> dict[str, Path]:
    processes: dict[str, subprocess.Popen] = {}
    outputs: dict[str, Path] = {}
    handles = []
    for gpu, method in enumerate(METHODS):
        output = RUN / phase / method
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(output)
        log_path = RUN / "logs" / f"{phase}_{method}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("w", encoding="utf-8")
        handles.append(handle)
        processes[method] = subprocess.Popen(
            [
                "taskset", "-c", f"{gpu * 20}-{gpu * 20 + 19}",
                *training_command(metas[method], output, steps=steps, save_interval=save_interval),
            ],
            cwd=XVLA,
            env=child_env(gpu),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        outputs[method] = output
        write_json(RUN / "pids" / f"{phase}_{method}.json", {"pid": processes[method].pid})
        print(f"[stackcube-four-train] phase={phase} method={method} gpu={gpu} pid={processes[method].pid}", flush=True)
    failures = {method: process.wait() for method, process in processes.items()}
    for handle in handles:
        handle.close()
    if any(code != 0 for code in failures.values()):
        raise RuntimeError(f"{phase} failed: {failures}")
    return outputs


def main() -> None:
    if RUN.exists():
        raise FileExistsError(RUN)
    RUN.mkdir(parents=True)
    metas = {method: build_training_meta(method) for method in METHODS}
    write_json(
        RUN / "training_contract.json",
        {
            "initial_checkpoint": str(START),
            "optimizer": "reset independently for every group",
            "methods": list(METHODS),
            "source_sampling": "1:1 original ID replay and group new expert",
            "original_id_trajectories": 128,
            "new_expert_trajectories": 100,
            "per_device_batch": 8,
            "world_size_per_group": 1,
            "gradient_accumulation_steps": 4,
            "effective_global_batch": 32,
            "action_chunk": 10,
            "tail_handling": (
                "every observation is an anchor; repeat the final real action only as storage padding; "
                "action_valid_mask excludes unavailable future targets and padded action dimensions"
            ),
            "formal_steps": 5000,
            "save_interval": 500,
        },
    )
    smoke = launch_wave(metas, "smoke_2", 2, 2)
    for method, output in smoke.items():
        if not (output / "ckpt-2/model.safetensors").exists():
            raise RuntimeError(f"missing {method} smoke checkpoint")
    print("[stackcube-four-train] four smoke checkpoints passed", flush=True)
    formal = launch_wave(metas, "formal_5000", 5000, 500)
    for method, output in formal.items():
        for step in range(500, 5001, 500):
            if not (output / f"ckpt-{step}/model.safetensors").exists():
                raise RuntimeError(f"missing {method} checkpoint {step}")
    (RUN / "TRAINING_COMPLETE").write_text("complete\n", encoding="utf-8")
    print("[stackcube-four-train] complete", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
