#!/usr/bin/env python3
"""Finish X-VLA airplane OOD collection, equal-step training, and evaluation."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path("/data/zhaozhixuan/Ask4Help-airplane-5090")
RESULT = ROOT / "results/xvla_airplane_v1"
PIPELINE = RESULT / "ood_dagger_v1"
ENV = Path("/data/zhaozhixuan/envs/xvla_official_5090")
XVLA = Path("/data/zhaozhixuan/X-VLA")
START = RESULT / "id_sft_10000_official_2gpu/ckpt-2500"
ID_META = RESULT / "manifests/panda_airplane_id.json"
METHODS = ("vlm_pool_pca", "offline_oracle", "failure_recovery", "diffdagger")
COLLECTIONS = {
    "vlm_pool_pca": PIPELINE / "collections/vlm_pool_pca_retry1",
    "offline_oracle": PIPELINE / "collections/offline_oracle",
    "failure_recovery": PIPELINE / "collections/failure_recovery_retry1",
    "diffdagger": PIPELINE / "collections/diffdagger_retry1",
}
DATASETS = {
    "vlm_pool_pca": PIPELINE / "datasets/vlm_pool_pca_retry1",
    "offline_oracle": PIPELINE / "datasets/offline_oracle",
    "failure_recovery": PIPELINE / "datasets/failure_recovery_retry1",
    "diffdagger": PIPELINE / "datasets/diffdagger_retry1",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def wait_for_collections() -> None:
    while True:
        ready = []
        for method in METHODS:
            summary = COLLECTIONS[method] / "summary.json"
            info = DATASETS[method] / "meta/info.json"
            if summary.exists() and info.exists():
                ready.append(
                    read_json(summary).get("accepted_total") == 100
                    and read_json(info).get("total_episodes") == 100
                )
            else:
                ready.append(False)
        print(f"[pipeline] collection_ready={dict(zip(METHODS, ready))}", flush=True)
        if all(ready):
            return
        time.sleep(300)


def build_training_meta(method: str) -> Path:
    output = PIPELINE / "training_metas" / method
    if output.exists():
        return output
    output.mkdir(parents=True)
    id_meta = read_json(ID_META)
    id_meta["dataset_name"] = f"panda_airplane_id_replay_{method}"
    (output / "00_id_replay.json").write_text(json.dumps(id_meta, indent=2) + "\n")

    dataset = DATASETS[method]
    info = read_json(dataset / "meta/info.json")
    episodes = [json.loads(line) for line in (dataset / "meta/episodes.jsonl").read_text().splitlines()]
    if len(episodes) != 100 or any(int(row["length"]) < 1 for row in episodes):
        raise RuntimeError(f"invalid {method} episode metadata")
    new_meta = {
        "dataset_name": f"panda_airplane_ood_{method}",
        "robot_type": "panda_airplane",
        "root_path": str(dataset.resolve()),
        "chunks_size": int(info["chunks_size"]),
        "data_path": info["data_path"],
        "datalist": episodes,
    }
    (output / "01_ood_expert.json").write_text(json.dumps(new_meta, indent=2) + "\n")
    return output


def training_command(cpu_start: int, meta: Path, output: Path) -> list[str]:
    return [
        "taskset", "-c", f"{cpu_start}-{cpu_start + 39}",
        str(ENV / "bin/python"), "-m", "accelerate.commands.launch",
        "--num_processes", "2", "--multi_gpu", "--mixed_precision", "bf16",
        "--gpu_ids", "0,1",
        "train.py", "--models", str(START), "--output_dir", str(output),
        "--train_metas_path", str(meta), "--batch_size", "8", "--num_actions", "10",
        "--num_views", "2", "--action_mode", "auto", "--real_action_dim", "8",
        "--max_action_dim", "20", "--distributed_backend", "gloo",
        "--gradient_checkpointing", "--gradient_accumulation_steps", "2",
        "--learning_rate", "1e-4", "--learning_coef", "0.1", "--weight_decay", "0",
        "--betas", "0.9", "0.95", "--max_grad_norm", "1.0", "--freeze_steps", "1000",
        "--warmup_steps", "2000", "--iters", "2500", "--save_interval", "500",
        "--log_interval", "20", "--seed", "4200",
    ]


def launch_all_training(metas: dict[str, Path]) -> None:
    for wave in (METHODS[:2], METHODS[2:]):
        processes: dict[str, subprocess.Popen] = {}
        for slot, method in enumerate(wave):
            output = PIPELINE / "training" / f"{method}_sft_2500_retry1"
            if output.exists():
                raise FileExistsError(output)
            log_path = PIPELINE / "logs" / f"train_{method}_2500_retry1.log"
            log = log_path.open("w")
            gpu_start = slot * 2
            env = os.environ.copy()
            env.update({
                "CUDA_VISIBLE_DEVICES": f"{gpu_start},{gpu_start + 1}",
                "OMP_NUM_THREADS": "20", "MKL_NUM_THREADS": "20",
                "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1", "NCCL_P2P_DISABLE": "1",
                "NCCL_IB_DISABLE": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            })
            processes[method] = subprocess.Popen(
                training_command(gpu_start * 20, metas[method], output),
                cwd=XVLA, env=env, stdout=log, stderr=subprocess.STDOUT,
            )
            print(f"[pipeline] train_start method={method} pid={processes[method].pid}", flush=True)
        failures = {method: proc.wait() for method, proc in processes.items()}
        if any(code != 0 for code in failures.values()):
            raise RuntimeError(f"training failed: {failures}")
    for method in METHODS:
        output = PIPELINE / "training" / f"{method}_sft_2500_retry1"
        for step in range(500, 2501, 500):
            if not (output / f"ckpt-{step}/model.safetensors").exists():
                raise RuntimeError(f"missing {method} ckpt-{step}")


def launch_all_evaluation() -> None:
    processes: dict[str, subprocess.Popen] = {}
    for gpu, method in enumerate(METHODS):
        checkpoint = PIPELINE / "training" / f"{method}_sft_2500_retry1/ckpt-2500"
        output = PIPELINE / "evaluation" / f"{method}_ood100_h150"
        log = (PIPELINE / "logs" / f"eval_{method}_ood100_h150.log").open("w")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        command = [
            "taskset", "-c", f"{gpu * 20}-{gpu * 20 + 19}", str(ENV / "bin/python"),
            str(ROOT / "tools/evaluate_pick_single_ycb_airplane_xvla.py"),
            "--checkpoint", str(checkpoint), "--xvla-root", str(XVLA),
            "--output-dir", str(output), "--split", "ood", "--episodes", "100",
            "--seed", "60000", "--execute-horizon", "5", "--max-episode-steps", "150",
            "--flow-steps", "10",
        ]
        processes[method] = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        print(f"[pipeline] eval_start method={method} pid={processes[method].pid}", flush=True)
    failures = {method: proc.wait() for method, proc in processes.items()}
    if any(code != 0 for code in failures.values()):
        raise RuntimeError(f"evaluation failed: {failures}")
    rows = []
    for method in METHODS:
        summary = read_json(PIPELINE / "evaluation" / f"{method}_ood100_h150/summary.json")
        rows.append({
            "method": method,
            "episodes": summary["episodes"],
            "ever_grasped_successes": summary["ever_grasped_successes"],
            "ever_grasped_rate": summary["ever_grasped_rate"],
            "strict_successes": summary["strict_successes"],
            "strict_success_rate": summary["strict_success_rate"],
        })
    (PIPELINE / "final_ood100_comparison.json").write_text(json.dumps(rows, indent=2) + "\n")


def main() -> None:
    wait_for_collections()
    metas = {method: build_training_meta(method) for method in METHODS}
    launch_all_training(metas)
    launch_all_evaluation()
    (PIPELINE / "PIPELINE_COMPLETE").write_text("complete\n")
    print("[pipeline] complete", flush=True)


if __name__ == "__main__":
    main()
