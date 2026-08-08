#!/usr/bin/env python3
"""Run four fair single-GPU X-VLA airplane adaptation jobs in parallel."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


METHODS = ("vlm_pool_pca", "offline_oracle", "failure_recovery", "diffdagger")
ROOT = Path("/data/zhaozhixuan/Ask4Help-airplane-event-close-v2")
RESULT = Path(
    "/data/zhaozhixuan/Ask4Help-airplane-5090/results/xvla_airplane_v1/"
    "ood_dagger_id_ood_alternating_event_close_v2"
)
XVLA = Path("/data/zhaozhixuan/X-VLA")
ENV = Path("/data/zhaozhixuan/envs/xvla_official_5090")
START = Path(
    "/data/zhaozhixuan/Ask4Help-airplane-5090/results/xvla_airplane_v1/"
    "id_sft_10000_official_2gpu/ckpt-2500"
)
ID_META = Path(
    "/data/zhaozhixuan/Ask4Help-airplane-5090/results/xvla_airplane_v1/"
    "manifests/panda_airplane_id.json"
)
RUN = RESULT / "training_v1_5000"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_training_meta(method: str) -> Path:
    output = RUN / "metas" / method
    output.mkdir(parents=True, exist_ok=False)
    original = read_json(ID_META)
    original["dataset_name"] = f"panda_airplane_id_replay_{method}"
    write_json(output / "00_id_replay.json", original)

    dataset = RESULT / "datasets" / method
    info = read_json(dataset / "meta/info.json")
    episodes = [
        json.loads(line)
        for line in (dataset / "meta/episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(episodes) != 100 or any(int(row["length"]) < 10 for row in episodes):
        raise RuntimeError(f"invalid {method} expert dataset")
    write_json(
        output / "01_new_expert.json",
        {
            "dataset_name": f"panda_airplane_new_expert_{method}",
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
        "-m",
        "accelerate.commands.launch",
        "--num_processes",
        "1",
        "--mixed_precision",
        "bf16",
        "--gpu_ids",
        "0",
        "train.py",
        "--models",
        str(START),
        "--output_dir",
        str(output),
        "--train_metas_path",
        str(meta),
        "--batch_size",
        "8",
        "--num_actions",
        "10",
        "--num_views",
        "2",
        "--action_mode",
        "auto",
        "--real_action_dim",
        "8",
        "--max_action_dim",
        "20",
        "--distributed_backend",
        "gloo",
        "--gradient_checkpointing",
        "--gradient_accumulation_steps",
        "4",
        "--learning_rate",
        "1e-4",
        "--learning_coef",
        "0.1",
        "--weight_decay",
        "0",
        "--betas",
        "0.9",
        "0.95",
        "--max_grad_norm",
        "1.0",
        "--freeze_steps",
        "1000",
        "--warmup_steps",
        "2000",
        "--iters",
        str(steps),
        "--save_interval",
        str(save_interval),
        "--log_interval",
        "20",
        "--seed",
        "4200",
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
        }
    )
    return env


def launch_training_wave(
    metas: dict[str, Path], *, phase: str, steps: int, save_interval: int
) -> dict[str, Path]:
    processes: dict[str, subprocess.Popen] = {}
    outputs: dict[str, Path] = {}
    for gpu, method in enumerate(METHODS):
        output = RUN / phase / method
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(output)
        log_path = RUN / "logs" / f"{phase}_{method}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("w", encoding="utf-8")
        command = [
            "taskset",
            "-c",
            f"{gpu * 20}-{gpu * 20 + 19}",
            *training_command(metas[method], output, steps=steps, save_interval=save_interval),
        ]
        processes[method] = subprocess.Popen(
            command,
            cwd=XVLA,
            env=child_env(gpu),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        outputs[method] = output
        write_json(RUN / "pids" / f"{phase}_{method}.json", {"pid": processes[method].pid})
        print(f"[event-close-training] {phase} {method} pid={processes[method].pid} gpu={gpu}", flush=True)
    failures = {method: process.wait() for method, process in processes.items()}
    if any(code != 0 for code in failures.values()):
        raise RuntimeError(f"{phase} failed: {failures}")
    return outputs


def forward_smoke(outputs: dict[str, Path]) -> None:
    processes: dict[str, subprocess.Popen] = {}
    for gpu, method in enumerate(METHODS):
        checkpoint = outputs[method] / "ckpt-2"
        if not (checkpoint / "model.safetensors").exists():
            raise RuntimeError(f"missing smoke checkpoint for {method}")
        output = RUN / "smoke_forward" / method
        log_path = RUN / "logs" / f"smoke_forward_{method}.log"
        log = log_path.open("w", encoding="utf-8")
        command = [
            "taskset",
            "-c",
            f"{gpu * 20}-{gpu * 20 + 19}",
            str(ENV / "bin/python"),
            str(ROOT / "tools/evaluate_pick_single_ycb_airplane_xvla.py"),
            "--checkpoint",
            str(checkpoint),
            "--xvla-root",
            str(XVLA),
            "--output-dir",
            str(output),
            "--split",
            "id",
            "--episodes",
            "1",
            "--seed",
            "59000",
            "--execute-horizon",
            "5",
            "--max-episode-steps",
            "5",
            "--flow-steps",
            "10",
        ]
        processes[method] = subprocess.Popen(
            command,
            cwd=ROOT,
            env=child_env(gpu),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    failures = {method: process.wait() for method, process in processes.items()}
    if any(code != 0 for code in failures.values()):
        raise RuntimeError(f"forward smoke failed: {failures}")


def main() -> None:
    if RUN.exists():
        raise FileExistsError(RUN)
    RUN.mkdir(parents=True)
    metas = {method: build_training_meta(method) for method in METHODS}
    write_json(
        RUN / "training_contract.json",
        {
            "initial_checkpoint": str(START),
            "methods": list(METHODS),
            "source_sampling": "equal probability per sample: original ID replay vs group new expert",
            "per_device_batch": 8,
            "world_size_per_group": 1,
            "gradient_accumulation_steps": 4,
            "effective_global_batch": 32,
            "action_chunk": 10,
            "tail_handling": (
                "complete 10-step windows only; no temporal padding; the final real action is "
                "the final target in the last legal window"
            ),
            "action_channel_padding": "real 8D action is embedded in the X-VLA 20D interface",
            "formal_steps": 5000,
            "save_interval": 500,
            "optimizer": "reset independently for every group",
        },
    )
    smoke = launch_training_wave(metas, phase="smoke_2", steps=2, save_interval=2)
    forward_smoke(smoke)
    print("[event-close-training] all train/reload/forward smoke checks passed", flush=True)
    formal = launch_training_wave(metas, phase="formal_5000", steps=5000, save_interval=500)
    for method, output in formal.items():
        for step in range(500, 5001, 500):
            if not (output / f"ckpt-{step}/model.safetensors").exists():
                raise RuntimeError(f"missing {method} checkpoint {step}")
    (RUN / "TRAINING_COMPLETE").write_text("complete\n", encoding="utf-8")
    print("[event-close-training] complete", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
