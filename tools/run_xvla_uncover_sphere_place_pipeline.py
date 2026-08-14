#!/usr/bin/env python3
"""Run the post-training UncoverSpherePlace evaluation pipeline.

The controller waits for the already-running ID base-policy job, then advances
through checkpoint selection, fixed policy evaluation, ID-only detector fitting,
passive detector evaluation, matched-budget gated collection, and one updated
policy evaluation. Each stage has its own output and completion marker; an
existing partial stage is never overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def next_output(path: Path) -> Path:
    if not path.exists():
        return path
    index = 1
    while True:
        candidate = path.with_name(f"{path.name}_retry{index}")
        if not candidate.exists():
            return candidate
        index += 1


def completed(path: Path, marker: str) -> bool:
    return (path / marker).is_file()


def write_marker(path: Path, marker: str, text: str = "complete\n") -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / marker).write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--id-manifest", type=Path, required=True)
    parser.add_argument("--gpu-ids", default="4,5")
    parser.add_argument("--eval-gpu", default="4")
    parser.add_argument("--train-port", type=int, default=29531)
    parser.add_argument("--id-selection-episodes", type=int, default=20)
    parser.add_argument("--id-confirm-episodes", type=int, default=32)
    parser.add_argument("--final-episodes", type=int, default=100)
    parser.add_argument("--gate-target-id", type=int, default=64)
    parser.add_argument("--gate-target-ood", type=int, default=64)
    parser.add_argument("--gate-max-attempts", type=int, default=512)
    return parser.parse_args()


def stage_state(state_path: Path, stage: str, **extra: Any) -> None:
    payload = {"stage": stage, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **extra}
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[uncover-pipeline] stage={stage} {extra}", flush=True)


def run_command(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    env: dict[str, str],
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[uncover-pipeline] run={' '.join(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"command failed with exit code {result.returncode}: {command}")


def base_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": args.eval_gpu,
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    return env


def evaluator_command(
    args: argparse.Namespace,
    checkpoint: Path,
    output: Path,
    split: str,
    episodes: int,
    seed: int,
) -> list[str]:
    return [
        str(args.python),
        "tools/evaluate_uncover_sphere_place_xvla.py",
        "--checkpoint",
        str(checkpoint),
        "--xvla-root",
        str(args.xvla_root),
        "--output-dir",
        str(output),
        "--split",
        split,
        "--episodes",
        str(episodes),
        "--seed",
        str(seed),
        "--execute-horizon",
        "5",
        "--max-episode-steps",
        "2500",
        "--flow-steps",
        "10",
    ]


def wait_for_base_training(args: argparse.Namespace) -> None:
    marker = args.train_dir / "TRAINING_COMPLETE"
    while not marker.is_file():
        command_running = subprocess.run(
            ["pgrep", "-af", "uncover_id_sft_10000_retry4|accelerate launch.*29524"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode == 0
        if not command_running:
            time.sleep(30)
            if not marker.is_file():
                raise RuntimeError(f"base training stopped without {marker}")
        print("[uncover-pipeline] waiting for ID base training", flush=True)
        time.sleep(300)
    expected = [args.train_dir / f"ckpt-{step}" / "model.safetensors" for step in range(500, 10001, 500)]
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise RuntimeError(f"base training marker exists but checkpoints are missing: {missing[:5]}")


def checkpoint_selection(args: argparse.Namespace, root: Path) -> Path:
    output = root / "checkpoint_selection"
    if completed(output, "CHECKPOINT_SELECTION_COMPLETE"):
        return Path(read_json(output / "selection.json")["checkpoint"])
    output = next_output(output)
    output.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    selected: Path | None = None
    for step in (2000, 4000, 6000, 8000, 10000):
        checkpoint = args.train_dir / f"ckpt-{step}"
        eval_dir = output / f"id_validation_{step}"
        run_command(
            evaluator_command(
                args,
                checkpoint,
                eval_dir,
                "id",
                args.id_selection_episodes,
                30000 + step,
            ),
            cwd=args.repo_root,
            log_path=root / "logs" / f"checkpoint_selection_id_{step}.log",
            env=base_env(args),
        )
        summary = read_json(eval_dir / "summary.json")
        row = {
            "step": step,
            "checkpoint": str(checkpoint),
            "successes": summary["successes"],
            "episodes": summary["episodes"],
            "success_rate": summary["success_rate"],
            "validation_summary": str(eval_dir / "summary.json"),
        }
        records.append(row)
        print(f"[uncover-pipeline] selection step={step} sr={row['success_rate']:.3f}", flush=True)
        if row["success_rate"] >= 0.8:
            confirm_dir = output / f"id_confirmation_{step}"
            run_command(
                evaluator_command(
                    args,
                    checkpoint,
                    confirm_dir,
                    "id",
                    args.id_confirm_episodes,
                    40000 + step,
                ),
                cwd=args.repo_root,
                log_path=root / "logs" / f"checkpoint_confirmation_id_{step}.log",
                env=base_env(args),
            )
            confirmation = read_json(confirm_dir / "summary.json")
            row["confirmation_success_rate"] = confirmation["success_rate"]
            row["confirmation_summary"] = str(confirm_dir / "summary.json")
            if confirmation["success_rate"] >= 0.8:
                selected = checkpoint
                break
    if selected is None:
        (output / "BASE_POLICY_NOT_ACCEPTED").write_text(
            "No checkpoint reached the frozen ID validation criterion of 80%.\n",
            encoding="utf-8",
        )
        (output / "selection.json").write_text(json.dumps({"records": records}, indent=2) + "\n")
        raise RuntimeError("ID base policy did not pass the 80% validation criterion")
    payload = {
        "checkpoint": str(selected),
        "selected_step": int(selected.name.split("-")[-1]),
        "criterion": "earliest checkpoint with selection and independent confirmation ID success >= 0.8",
        "records": records,
    }
    (output / "selection.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_marker(output, "CHECKPOINT_SELECTION_COMPLETE")
    return selected


def final_policy_evaluation(args: argparse.Namespace, root: Path, checkpoint: Path) -> None:
    output = root / "final_evaluation"
    if completed(output, "FINAL_POLICY_EVALUATION_COMPLETE"):
        return
    output = next_output(output)
    splits = ("id", "handle_ood", "goal_ood")
    for offset, split in enumerate(splits):
        run_command(
            evaluator_command(args, checkpoint, output / split, split, args.final_episodes, 50000 + offset * 1000),
            cwd=args.repo_root,
            log_path=root / "logs" / f"final_policy_{split}.log",
            env=base_env(args),
        )
    rows = []
    for split in splits:
        summary = read_json(output / split / "summary.json")
        rows.append(
            {
                "split": split,
                "episodes": summary["episodes"],
                "successes": summary["successes"],
                "success_rate": summary["success_rate"],
                "sphere_grasp_rate": summary["sphere_grasp_rate"],
                "summary": str(output / split / "summary.json"),
            }
        )
    (output / "comparison.json").write_text(json.dumps({"checkpoint": str(checkpoint), "rows": rows}, indent=2) + "\n")
    write_marker(output, "FINAL_POLICY_EVALUATION_COMPLETE")


def build_detector_assets(args: argparse.Namespace, root: Path, checkpoint: Path) -> Path:
    output = root / "detector_assets"
    if (output / "multilayer_detector_assets.pt").is_file() and completed(output, "DETECTOR_ASSETS_COMPLETE"):
        return output / "multilayer_detector_assets.pt"
    output = next_output(output)
    asset = output / "multilayer_detector_assets.pt"
    manifest = output / "assets_manifest.json"
    run_command(
        [
            str(args.python),
            "tools/build_xvla_airplane_multilayer_assets.py",
            "--checkpoint",
            str(checkpoint),
            "--xvla-root",
            str(args.xvla_root),
            "--metadata",
            str(args.id_manifest),
            "--output-dir",
            str(output),
            "--batch-size",
            "8",
            "--probe-seed",
            "0",
            "--probe-steps",
            "5",
            "--pca-dim",
            "512",
            "--ridge",
            "1e-6",
        ],
        cwd=args.repo_root,
        log_path=root / "logs" / "build_detector_assets.log",
        env=base_env(args),
    )
    if not asset.is_file():
        raise RuntimeError(f"missing detector asset {asset}")
    write_marker(output, "DETECTOR_ASSETS_COMPLETE")
    return asset


def passive_detector_evaluation(args: argparse.Namespace, root: Path, checkpoint: Path, assets: Path) -> Path:
    output = root / "passive_detector_rollouts"
    if completed(output, "PASSIVE_DETECTOR_EVALUATION_COMPLETE"):
        return output
    output = next_output(output)
    for offset, split in enumerate(("id", "handle_ood", "goal_ood")):
        run_command(
            [
                str(args.python),
                "tools/evaluate_uncover_sphere_place_failure_detectors.py",
                "--checkpoint",
                str(checkpoint),
                "--xvla-root",
                str(args.xvla_root),
                "--multilayer-assets",
                str(assets),
                "--output-dir",
                str(output / split),
                "--split",
                split,
                "--episodes",
                str(args.final_episodes),
                "--seed",
                str(60000 + offset * 1000),
                "--execute-horizon",
                "5",
                "--max-episode-steps",
                "2500",
                "--flow-steps",
                "10",
                "--probe-steps",
                "5",
                "--probe-seed",
                "0",
                "--device",
                "cuda",
            ],
            cwd=args.repo_root,
            log_path=root / "logs" / f"passive_detector_{split}.log",
            env=base_env(args),
        )
    write_marker(output, "PASSIVE_DETECTOR_EVALUATION_COMPLETE")
    return output


def calibrate_threshold(args: argparse.Namespace, root: Path, passive_root: Path) -> dict[str, Any]:
    output = root / "calibration"
    target = output / "thresholds.json"
    if target.is_file() and completed(output, "CALIBRATION_COMPLETE"):
        return read_json(target)
    output = next_output(output)
    output.mkdir(parents=True)
    summary = read_json(passive_root / "id" / "summary.json")
    successful = [row for row in summary["rows"] if bool(row["success"])]
    if len(successful) < 20:
        raise RuntimeError(f"ID calibration needs at least 20 successful rollouts, got {len(successful)}")
    methods = sorted(
        {
            name
            for row in successful
            for point in row.get("timeline", [])
            for name, value in point.get("scores", {}).items()
            if value is not None and np.isfinite(value)
        }
    )
    thresholds: dict[str, Any] = {}
    for method in methods:
        maxima = []
        for row in successful:
            values = [
                float(point["scores"][method])
                for point in row.get("timeline", [])
                if point.get("scores", {}).get(method) is not None
                and np.isfinite(point["scores"][method])
            ]
            if values:
                maxima.append(max(values))
        if len(maxima) < 20:
            continue
        thresholds[method] = {
            "threshold": float(np.quantile(np.asarray(maxima, dtype=np.float64), 0.95)),
            "q": 0.95,
            "successful_trajectory_count": len(maxima),
            "source": str(passive_root / "id" / "summary.json"),
        }
    selected = "vlm_action_bridge_pca"
    if selected not in thresholds:
        raise RuntimeError(f"required internal detector threshold missing: {selected}")
    payload = {
        "format": "uncover_sphere_place_internal_detector_calibration_v1",
        "checkpoint": str(read_json(passive_root / "id" / "summary.json")["checkpoint"]),
        "calibration_split": "successful ID policy rollouts only",
        "methods": thresholds,
        "selected_detector": selected,
    }
    target = output / "thresholds.json"
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_marker(output, "CALIBRATION_COMPLETE")
    return payload


def gated_collection(args: argparse.Namespace, root: Path, checkpoint: Path, assets: Path, calibration: dict[str, Any]) -> Path:
    output = root / "gated_collection_internal_pca"
    if completed(output, "COLLECTION_COMPLETE"):
        return output
    output = next_output(output)
    selected = calibration["selected_detector"]
    threshold = calibration["methods"][selected]["threshold"]
    run_command(
        [
            str(args.python),
            "tools/collect_uncover_sphere_place_gated_dagger.py",
            "--checkpoint",
            str(checkpoint),
            "--xvla-root",
            str(args.xvla_root),
            "--multilayer-assets",
            str(assets),
            "--output-dir",
            str(output),
            "--detector",
            selected,
            "--threshold",
            str(threshold),
            "--split",
            "mixed",
            "--episodes",
            str(args.gate_max_attempts),
            "--target-id",
            str(args.gate_target_id),
            "--target-ood",
            str(args.gate_target_ood),
            "--execute-horizon",
            "5",
            "--max-episode-steps",
            "2500",
            "--flow-steps",
            "10",
            "--probe-steps",
            "5",
            "--probe-seed",
            "0",
            "--device",
            "cuda",
        ],
        cwd=args.repo_root,
        log_path=root / "logs" / "gated_collection_internal_pca.log",
        env=base_env(args),
    )
    write_marker(output, "COLLECTION_COMPLETE")
    return output


def build_mixed_metadata(args: argparse.Namespace, root: Path, gated: Path) -> Path:
    output = root / "gated_training_metas"
    if (output / "00_id_replay.json").is_file() and (output / "01_expert_suffix.json").is_file():
        return output
    if output.exists() and any(output.iterdir()):
        output = next_output(output)
    output.mkdir(parents=True, exist_ok=True)
    id_meta = read_json(args.id_manifest)
    id_meta["dataset_name"] = "panda_uncover_id_replay_internal_pca"
    (output / "00_id_replay.json").write_text(json.dumps(id_meta, indent=2) + "\n", encoding="utf-8")
    dataset = gated / "lerobot_dataset"
    info = read_json(dataset / "meta" / "info.json")
    episodes = read_jsonl(dataset / "meta" / "episodes.jsonl")
    if not episodes or any(int(row["length"]) < 1 for row in episodes):
        raise RuntimeError("gated dataset has no valid episodes")
    new_meta = {
        "dataset_name": "panda_uncover_expert_internal_pca",
        "robot_type": "panda_airplane",
        "root_path": str(dataset.resolve()),
        "chunks_size": int(info["chunks_size"]),
        "data_path": info["data_path"],
        "datalist": episodes,
    }
    (output / "01_expert_suffix.json").write_text(json.dumps(new_meta, indent=2) + "\n", encoding="utf-8")
    return output


def updated_training(args: argparse.Namespace, root: Path, checkpoint: Path, metas: Path) -> Path:
    output = root / "gated_sft_2500"
    if completed(output, "TRAINING_COMPLETE"):
        return output
    output = next_output(output)
    command = [
        str(args.python.with_name("accelerate")),
        "launch",
        "--num_processes",
        "2",
        "--multi_gpu",
        "--main_process_port",
        str(args.train_port),
        "--mixed_precision",
        "bf16",
        "--gpu_ids",
        args.gpu_ids,
        "train.py",
        "--models",
        str(checkpoint),
        "--output_dir",
        str(output),
        "--train_metas_path",
        str(metas),
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
        "2",
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
        "2500",
        "--save_interval",
        "500",
        "--log_interval",
        "20",
        "--seed",
        "5600",
    ]
    env = base_env(args)
    env["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    run_command(
        command,
        cwd=args.xvla_root,
        log_path=root / "logs" / "gated_sft_2500.log",
        env=env,
    )
    for step in range(500, 2501, 500):
        if not (output / f"ckpt-{step}" / "model.safetensors").is_file():
            raise RuntimeError(f"missing updated checkpoint {output / f'ckpt-{step}'}")
    write_marker(output, "TRAINING_COMPLETE")
    return output


def updated_evaluation(args: argparse.Namespace, root: Path, training: Path) -> None:
    output = root / "gated_final_evaluation"
    if completed(output, "FINAL_EVALUATION_COMPLETE"):
        return
    output = next_output(output)
    checkpoint = training / "ckpt-2500"
    for offset, split in enumerate(("id", "handle_ood", "goal_ood")):
        run_command(
            evaluator_command(args, checkpoint, output / split, split, args.final_episodes, 80000 + offset * 1000),
            cwd=args.repo_root,
            log_path=root / "logs" / f"gated_final_{split}.log",
            env=base_env(args),
        )
    rows = []
    for split in ("id", "handle_ood", "goal_ood"):
        summary = read_json(output / split / "summary.json")
        rows.append({"split": split, "summary": str(output / split / "summary.json"), **{
            key: summary[key]
            for key in ("episodes", "successes", "success_rate", "sphere_grasp_rate", "sphere_in_bowl_rate")
        }})
    (output / "comparison.json").write_text(json.dumps({"checkpoint": str(checkpoint), "rows": rows}, indent=2) + "\n")
    write_marker(output, "FINAL_EVALUATION_COMPLETE")


def main() -> None:
    args = parse_args()
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    state = root / "pipeline_state.json"
    stage_state(state, "waiting_for_id_training", train_dir=str(args.train_dir))
    wait_for_base_training(args)
    stage_state(state, "checkpoint_selection")
    checkpoint = checkpoint_selection(args, root)
    stage_state(state, "final_policy_evaluation", checkpoint=str(checkpoint))
    final_policy_evaluation(args, root, checkpoint)
    stage_state(state, "detector_assets", checkpoint=str(checkpoint))
    assets = build_detector_assets(args, root, checkpoint)
    stage_state(state, "passive_detector_evaluation", assets=str(assets))
    passive = passive_detector_evaluation(args, root, checkpoint, assets)
    stage_state(state, "calibration", passive=str(passive))
    calibration = calibrate_threshold(args, root, passive)
    stage_state(state, "gated_collection", detector=calibration["selected_detector"])
    gated = gated_collection(args, root, checkpoint, assets, calibration)
    stage_state(state, "gated_training_metadata", gated=str(gated))
    metas = build_mixed_metadata(args, root, gated)
    stage_state(state, "gated_training", metas=str(metas))
    updated = updated_training(args, root, checkpoint, metas)
    stage_state(state, "gated_final_evaluation", training=str(updated))
    updated_evaluation(args, root, updated)
    write_marker(root, "PIPELINE_COMPLETE")
    stage_state(state, "PIPELINE_COMPLETE", checkpoint=str(checkpoint), updated_training=str(updated))
    print("[uncover-pipeline] PIPELINE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
