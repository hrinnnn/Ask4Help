#!/usr/bin/env python3
"""Durable StackPyramid four-method gated-DAgger comparison controller."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


METHODS = ("bridge_pca", "offline_oracle", "failure_recovery", "diffdagger")
STAGES = ("stage1_ood", "stage2_ood", "stage3_ood")
MARKERS = {
    "collection": "COLLECTION_COMPLETE",
    "smoke": "RELOAD_SMOKE_COMPLETE",
    "training": "TRAINING_COMPLETE",
    "evaluation": "EVAL_COMPLETE",
}


class Controller:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root = args.output_root
        self.logs = self.root / "logs"
        self.state_path = self.root / "pipeline_state.json"
        self.root.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        self.gpus = [int(value) for value in args.gpus.split(",")]
        if len(self.gpus) != 2:
            raise ValueError("this controller expects exactly two available GPUs")
        self.cpu_sets = args.cpu_sets.split(",")
        if len(self.cpu_sets) != 2:
            raise ValueError("provide two disjoint CPU sets")
        self.state: dict[str, Any] = {"format": "stackpyramid_four_method_comparison_v1"}
        self.diff_calibration = self.root / "calibration" / "diffdagger.json"
        self.pca_calibration = args.pca_calibration
        self.protocol_audit = args.protocol_audit
        if self.protocol_audit is None:
            raise ValueError("--protocol-audit is required for the corrected StackPyramid protocol")
        if not (self.protocol_audit / "PROTOCOL_AUDIT_COMPLETE").is_file():
            raise RuntimeError(f"protocol audit gate missing: {self.protocol_audit / 'PROTOCOL_AUDIT_COMPLETE'}")
        audit = json.loads((self.protocol_audit / "audit.json").read_text())
        gates = audit.get("gates", {})
        if not gates.get("oracle_pass") or not gates.get("base_policy_pass"):
            raise RuntimeError(f"protocol audit gates are not satisfied: {gates}")
        if self.pca_calibration is None:
            raise ValueError("--pca-calibration is required for independent ID calibration")

    def write_state(self, **updates: Any) -> None:
        self.state.update(updates)
        self.state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.state_path.write_text(json.dumps(self.state, indent=2, default=str) + "\n", encoding="utf-8")

    def log_path(self, label: str) -> Path:
        path = self.logs / f"{label}.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def fresh_path(self, base: Path, marker: str) -> Path:
        if (base / marker).is_file():
            return base
        if not base.exists():
            return base
        for index in range(1, 20):
            candidate = base.with_name(f"{base.name}_retry{index}")
            if not candidate.exists():
                return candidate
        raise RuntimeError(f"too many partial output directories for {base}")

    def resolve_path(self, base: Path, marker: str) -> Path:
        """Find an existing completed retry before allocating a new directory."""
        if (base / marker).is_file():
            return base
        candidates = sorted(
            candidate
            for candidate in base.parent.glob(f"{base.name}_retry*")
            if (candidate / marker).is_file()
        )
        if candidates:
            return candidates[-1]
        return self.fresh_path(base, marker)

    def record_paths(self, stage: str, **paths: dict[str, Path]) -> None:
        manifest_path = self.root / "stage_paths.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        stage_record = manifest.setdefault(stage, {})
        for group, values in paths.items():
            stage_record[group] = {key: str(value) for key, value in values.items()}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    def command_env(self, gpu: int) -> dict[str, str]:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["STACKPYRAMID_OOD_GEOMETRY"] = self.args.geometry
        env["PYTHONPATH"] = os.pathsep.join([str(self.args.repo_root), str(self.args.xvla_root)])
        return env

    def run_process(self, label: str, command: list[str], gpu: int, cpu_set: str) -> None:
        log = self.log_path(label)
        with log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"command": command, "gpu": gpu, "cpu_set": cpu_set}) + "\n")
            stream.flush()
            process = subprocess.Popen(
                ["taskset", "-c", cpu_set, *command],
                cwd=self.args.repo_root,
                env=self.command_env(gpu),
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
            self.write_state(running={"label": label, "pid": process.pid, "gpu": gpu, "log": str(log)})
            return_code = process.wait()
            stream.write(json.dumps({"return_code": return_code}) + "\n")
        # ManiSkill can abort while releasing its renderer after writing all
        # requested artifacts. Callers verify the stage marker and payload
        # immediately after this function, so preserve that documented case.
        if return_code not in (0, -6):
            raise RuntimeError(f"{label} exited with {return_code}; see {log}")

    def run_pair(self, jobs: list[tuple[str, list[str], int]]) -> None:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(self.run_process, label, command, gpu, self.cpu_sets[index]) for index, (label, command, gpu) in enumerate(jobs)]
            errors = []
            for future in futures:
                try:
                    future.result()
                except Exception as exc:
                    errors.append(exc)
        if errors:
            raise RuntimeError("; ".join(str(error) for error in errors))

    def python_command(self, script: str, *extra: str) -> list[str]:
        return [str(self.args.python), str(self.args.repo_root / "tools" / script), *extra]

    def collect_one(self, stage: str, method: str, gpu: int, cpu_set: str, output: Path, stage_index: int) -> None:
        threshold = json.loads(self.diff_calibration.read_text())["threshold"]
        command = self.python_command(
            "collect_stackpyramid_xvla_dagger.py",
            "--method", method,
            "--checkpoint", str(self.args.base_model),
            "--xvla-root", str(self.args.xvla_root),
            "--output-dir", str(output),
            "--split", stage,
            "--target", "100",
            "--id-seed", str(70000 + stage_index * 2000),
            "--ood-seed", str(80000 + stage_index * 2000),
            "--max-attempts", "500",
            "--flow-steps", "5",
            "--diff-timesteps", "16",
            "--diff-patience", "2",
            "--sim-backend", "cpu",
            "--render-backend", "cpu",
        )
        if method == "bridge_pca":
            command += ["--asset", str(self.args.pca_asset), "--pca-threshold", str(self.pca_threshold), "--min-ood-fraction", str(self.args.pca_min_ood_fraction)]
        elif method == "diffdagger":
            command += ["--diff-threshold", str(threshold)]
        self.run_process(f"collect_{stage}_{method}", command, gpu, cpu_set)
        if not (output / MARKERS["collection"]).is_file():
            raise RuntimeError(f"collection marker missing: {output}")

    def collect_stage(self, stage: str, stage_index: int) -> dict[str, Path]:
        self.write_state(stage=stage, phase="collection")
        outputs: dict[str, Path] = {}
        for method in METHODS:
            outputs[method] = self.resolve_path(self.root / "collections" / stage / method, MARKERS["collection"])
        # Two workers are available. The pairings keep the maximum GPU use at two.
        first = [("bridge_pca", outputs["bridge_pca"], self.gpus[0]), ("offline_oracle", outputs["offline_oracle"], self.gpus[1])]
        second = [("failure_recovery", outputs["failure_recovery"], self.gpus[0]), ("diffdagger", outputs["diffdagger"], self.gpus[1])]
        for pair in (first, second):
            jobs = [
                (f"collect_{stage}_{method}", self._collection_command(stage, method, output, stage_index), gpu)
                for method, output, gpu in pair
                if not (output / MARKERS["collection"]).is_file()
            ]
            if jobs:
                self.run_pair(jobs)
            for method, output, _gpu in pair:
                if not (output / MARKERS["collection"]).is_file():
                    raise RuntimeError(f"collection marker missing: {output}")
        self.record_paths(stage, collections=outputs)
        self.validate_collection_selection(stage, outputs)
        return outputs

    def _collection_command(self, stage: str, method: str, output: Path, stage_index: int) -> list[str]:
        threshold = json.loads(self.diff_calibration.read_text())["threshold"]
        command = self.python_command(
            "collect_stackpyramid_xvla_dagger.py",
            "--method", method,
            "--checkpoint", str(self.args.base_model),
            "--xvla-root", str(self.args.xvla_root),
            "--output-dir", str(output),
            "--split", stage,
            "--target", "100",
            "--id-seed", str(70000 + stage_index * 2000),
            "--ood-seed", str(80000 + stage_index * 2000),
            "--max-attempts", "500",
            "--flow-steps", "5",
            "--diff-timesteps", "16",
            "--diff-patience", "2",
            "--sim-backend", "cpu",
            "--render-backend", "cpu",
        )
        if method == "bridge_pca":
            command += ["--asset", str(self.args.pca_asset), "--pca-threshold", str(self.pca_threshold), "--min-ood-fraction", str(self.args.pca_min_ood_fraction)]
        elif method == "diffdagger":
            command += ["--diff-threshold", str(threshold)]
        return command

    def validate_collection_selection(self, stage: str, paths: dict[str, Path]) -> None:
        """Stop before budget matching if a gate is not selecting OOD-dominant data."""
        for method, path in paths.items():
            summary = json.loads((path / "summary.json").read_text())
            accepted = summary.get("accepted_by_split", {})
            total = int(summary.get("accepted_total", 0))
            ood = int(accepted.get(stage, 0))
            required = self.args.pca_min_ood_fraction if method == "bridge_pca" else None
            if total != 100 or (required is not None and ood / max(1, total) < required):
                raise RuntimeError(
                    f"{stage}/{method} failed OOD-dominant selection gate: "
                    f"accepted={total}, accepted_by_split={accepted}, "
                    f"required_ood_fraction={required}"
                )
            metrics = summary.get("selection_metrics", {})
            if not metrics.get("alternating_stream") or not metrics.get("selection_is_not_forced_50_50"):
                raise RuntimeError(f"{stage}/{method} missing corrected selection protocol metrics")
            gate = summary.get("selection_gate", {})
            if required is not None and gate.get("pass") is not True:
                raise RuntimeError(f"{stage}/{method} missing passing method-specific selection gate: {gate}")

    def prepare_stage(self, stage: str, source_paths: dict[str, Path]) -> Path:
        self.write_state(stage=stage, phase="budget_selection")
        output = self.root / "selected" / stage
        if not (output / "BUDGET_SELECTION_COMPLETE").is_file():
            output = self.fresh_path(output, "BUDGET_SELECTION_COMPLETE")
            command = self.python_command(
                "prepare_stackpyramid_method_comparison.py",
                "--output", str(output),
                "--stage", stage,
                "--id-h5", str(self.args.id_h5),
            )
            for method in METHODS:
                command += ["--source", f"{method}={source_paths[method] / 'accepted_suffixes.h5'}"]
            self.run_process(f"budget_{stage}", command, self.gpus[0], self.cpu_sets[0])
        self.record_paths(stage, selected={"root": output})
        return output

    def train_one(self, stage: str, method: str, gpu: int, cpu_set: str, selected: Path, output: Path, smoke: bool) -> None:
        command = self.python_command(
            "run_stackpyramid_gated_training.py",
            "--xvla-root", str(self.args.xvla_root),
            "--model", str(self.args.base_model),
            "--id-h5", str(self.args.id_h5),
            "--expert-h5", str(selected / method / "accepted_suffixes.h5"),
            "--output", str(output),
            "--steps", "2" if smoke else str(self.args.training_steps),
            "--save-interval", "2" if smoke else "500",
            "--batch-size", str(self.args.batch_size),
            "--seed", str(self.args.seed + hash((stage, method, smoke)) % 10000),
        )
        if smoke:
            command.append("--smoke-only")
        self.run_process(f"{'smoke' if smoke else 'train'}_{stage}_{method}", command, gpu, cpu_set)
        marker = MARKERS["smoke" if smoke else "training"]
        if not (output / marker).is_file():
            raise RuntimeError(f"{marker} missing: {output}")

    def train_stage(self, stage: str, selected: Path) -> dict[str, Path]:
        outputs = {method: self.resolve_path(self.root / "training" / stage / method, MARKERS["training"]) for method in METHODS}
        smoke_outputs = {method: self.fresh_path(self.root / "smoke" / stage / method, MARKERS["smoke"]) for method in METHODS}
        self.write_state(stage=stage, phase="smoke", training_outputs=outputs, smoke_outputs=smoke_outputs)
        for pair_methods in (("bridge_pca", "offline_oracle"), ("failure_recovery", "diffdagger")):
            jobs = []
            for index, method in enumerate(pair_methods):
                if not (smoke_outputs[method] / MARKERS["smoke"]).is_file():
                    jobs.append((f"smoke_{stage}_{method}", self._training_command(stage, method, selected, smoke_outputs[method], smoke=True), self.gpus[index]))
            if jobs:
                self.run_pair(jobs)
            for method in pair_methods:
                if not (smoke_outputs[method] / MARKERS["smoke"]).is_file():
                    raise RuntimeError(f"smoke marker missing: {smoke_outputs[method]}")
        self.write_state(stage=stage, phase="formal_training", training_outputs=outputs)
        for pair_methods in (("bridge_pca", "offline_oracle"), ("failure_recovery", "diffdagger")):
            jobs = []
            for index, method in enumerate(pair_methods):
                if not (outputs[method] / MARKERS["training"]).is_file():
                    jobs.append((f"train_{stage}_{method}", self._training_command(stage, method, selected, outputs[method], smoke=False), self.gpus[index]))
            if jobs:
                self.run_pair(jobs)
            for method in pair_methods:
                if not (outputs[method] / MARKERS["training"]).is_file():
                    raise RuntimeError(f"training marker missing: {outputs[method]}")
        self.record_paths(stage, training=outputs, smoke=smoke_outputs)
        return outputs

    def _training_command(self, stage: str, method: str, selected: Path, output: Path, smoke: bool) -> list[str]:
        command = self.python_command(
            "run_stackpyramid_gated_training.py",
            "--xvla-root", str(self.args.xvla_root),
            "--model", str(self.args.base_model),
            "--id-h5", str(self.args.id_h5),
            "--expert-h5", str(selected / method / "accepted_suffixes.h5"),
            "--output", str(output),
            "--steps", "2" if smoke else str(self.args.training_steps),
            "--save-interval", "2" if smoke else "500",
            "--batch-size", str(self.args.batch_size),
            "--seed", str(self.args.seed + (sum(ord(char) for char in stage + method) % 10000)),
        )
        if smoke:
            command.append("--smoke-only")
        return command

    def eval_method(self, stage: str, method: str, gpu: int, cpu_set: str, checkpoint: Path) -> None:
        base = self.root / "evaluation" / stage / method
        for split, start_seed in (("id", 70000), (stage, 80000)):
            output = self.fresh_path(base / split, MARKERS["evaluation"])
            if (output / MARKERS["evaluation"]).is_file():
                continue
            command = self.python_command(
                "evaluate_stackpyramid_xvla.py",
                "--checkpoint", str(checkpoint),
                "--xvla-root", str(self.args.xvla_root),
                "--output", str(output),
                "--split", split,
                "--episodes", "100",
                "--start-seed", str(start_seed),
                "--max-episode-steps", "250",
                "--execute-horizon", "5",
                "--flow-steps", "5",
                "--sim-backend", "gpu",
                "--render-backend", "gpu",
            )
            self.run_process(f"eval_{stage}_{method}_{split}", command, gpu, cpu_set)
            if not (output / MARKERS["evaluation"]).is_file():
                raise RuntimeError(f"evaluation marker missing: {output}")

    def evaluate_stage(self, stage: str, training: dict[str, Path]) -> None:
        self.write_state(stage=stage, phase="evaluation")
        jobs = []
        for index, method in enumerate(METHODS):
            jobs.append((method, self.gpus[index % 2]))
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            for offset in range(0, len(jobs), 2):
                current = jobs[offset : offset + 2]
                futures = [executor.submit(self.eval_method, stage, method, gpu, self.cpu_sets[index], training[method] / "ckpt-2000") for index, (method, gpu) in enumerate(current)]
                for future in futures:
                    future.result()

    def summarize(self) -> None:
        rows: list[dict[str, Any]] = []
        for stage in STAGES:
            manifest_path = self.root / "selected" / stage / "budget_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            for method in METHODS:
                paths = json.loads((self.root / "stage_paths.json").read_text()).get(stage, {})
                collection_root = Path(paths.get("collections", {}).get(method, str(self.root / "collections" / stage / method)))
                training_root = Path(paths.get("training", {}).get(method, str(self.root / "training" / stage / method)))
                collection = json.loads((collection_root / "summary.json").read_text())
                budget = manifest["methods"][method]
                losses = [json.loads(line)["loss"] for line in (training_root / "train.jsonl").read_text().splitlines() if line.strip()]
                metrics = {}
                for split in ("id", stage):
                    metrics[split] = json.loads((self.root / "evaluation" / stage / method / split / "summary.json").read_text())
                rows.append({
                    "stage": stage,
                    "method": method,
                    "collection_raw_attempts": collection["raw_attempts"],
                    "collection_accepted": collection["accepted_total"],
                    "collection_accepted_by_split": collection["accepted_by_split"],
                    "collection_expert_actions": collection["expert_action_steps"],
                    "selected_episodes": budget["selected_episodes"],
                    "selected_expert_actions": budget["selected_expert_action_steps"],
                    "common_budget": manifest["common_expert_action_budget"],
                    "training_final_loss": losses[-1] if losses else None,
                    "id": {key: metrics["id"][key] for key in ("ever_grasped", "ever_base_completed", "strict_success", "video_count")},
                    "stage_ood": {key: metrics[stage][key] for key in ("ever_grasped", "ever_base_completed", "strict_success", "video_count")},
                })
        comparison = {
            "format": "stackpyramid_four_method_comparison_v1",
            "base_model": str(self.args.base_model),
            "id_h5": str(self.args.id_h5),
            "training_steps": self.args.training_steps,
            "batch_size": self.args.batch_size,
            "methods": list(METHODS),
            "stages": list(STAGES),
            "rows": rows,
            "limitations": [
                "Diff-DAgger is implemented as the current model's sampled flow-score gate, not a claim of reproducing an external diffusion-policy codebase.",
                "Failure-Recovery uses a fixed 50 low-level policy-step takeover for this comparison.",
                "Offline BC uses full oracle trajectories collected under the same alternating seed manifest.",
            ],
        }
        (self.root / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
        lines = [
            "# StackPyramid 四方法 gated-DAgger 对照",
            "",
            f"共同基座：`{self.args.base_model}`；训练步数：`{self.args.training_steps}`；batch：`{self.args.batch_size}`。",
            "所有方法使用相同 stage、交替 ID/OOD seed、ID replay、成功定义和 100 ID/100 OOD policy evaluation；expert action 总量先按四方法共同可达预算匹配。",
            "",
            "| Stage | Method | Raw/accepted | Accepted ID/OOD | Selected episodes | Common expert actions | ID ever grasped | OOD ever grasped | ID strict | OOD strict |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            id_metrics = row["id"]
            ood_metrics = row["stage_ood"]
            lines.append(
                f"| {row['stage']} | {row['method']} | {row['collection_raw_attempts']}/{row['collection_accepted']} | {row['collection_accepted_by_split']} | {row['selected_episodes']} | {row['common_budget']} | {id_metrics['ever_grasped']}/100 | {ood_metrics['ever_grasped']}/100 | {id_metrics['strict_success']}/100 | {ood_metrics['strict_success']}/100 |"
            )
        lines += [
            "",
            "限制：Diff-DAgger 是当前 X-VLA Flow-SDE 模型内部 sampled-flow score 的实现；Failure-Recovery 固定在 50 个低层 policy steps 后接管；四方法的 raw videos 与逐 episode timeline 均保留在各自 collection/evaluation 目录。",
        ]
        (self.root / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def run(self) -> None:
        self.write_state(phase="pca_calibration", protocol_audit=str(self.protocol_audit))
        pca_marker = self.pca_calibration.parent / "PCA_CALIBRATION_COMPLETE"
        if not (pca_marker.is_file() and self.pca_calibration.is_file()):
            output = self.fresh_path(self.pca_calibration.parent, "PCA_CALIBRATION_COMPLETE")
            command = self.python_command(
                "calibrate_stackpyramid_bridge_pca.py",
                "--checkpoint", str(self.args.base_model),
                "--xvla-root", str(self.args.xvla_root),
                "--asset", str(self.args.pca_asset),
                "--output", str(output / "calibration.json"),
                "--successful-rollouts", "25",
                "--max-attempts", "60",
                "--start-seed", "45000",
                "--flow-steps", "5",
                "--sim-backend", "cpu",
                "--render-backend", "cpu",
            )
            self.run_process("calibrate_pca", command, self.gpus[0], self.cpu_sets[0])
            (output / "PCA_CALIBRATION_COMPLETE").write_text("complete\n", encoding="utf-8")
            self.pca_calibration = output / "calibration.json"
        self.pca_threshold = float(json.loads(self.pca_calibration.read_text())["threshold"])
        self.write_state(phase="diff_calibration", pca_calibration=str(self.pca_calibration), pca_threshold=self.pca_threshold)
        calibration_marker = self.diff_calibration.parent / "DIFF_CALIBRATION_COMPLETE"
        if calibration_marker.is_file() and (self.diff_calibration.parent / "calibration.json").is_file():
            self.diff_calibration = self.diff_calibration.parent / "calibration.json"
        else:
            output = self.fresh_path(self.diff_calibration.parent, "DIFF_CALIBRATION_COMPLETE")
            command = self.python_command(
                "calibrate_stackpyramid_diffdagger.py",
                "--checkpoint", str(self.args.base_model),
                "--xvla-root", str(self.args.xvla_root),
                "--output", str(output / "calibration.json"),
                "--successful-rollouts", "25",
                "--max-attempts", "60",
                "--start-seed", "46000",
                "--flow-steps", "5",
                "--diff-timesteps", "16",
                "--max-episode-steps", "150",
                "--sim-backend", "cpu",
                "--render-backend", "cpu",
            )
            self.run_process("calibrate_diffdagger", command, self.gpus[0], self.cpu_sets[0])
            if not (output / "DIFF_CALIBRATION_COMPLETE").is_file():
                raise RuntimeError("Diff calibration did not complete")
            self.diff_calibration = output / "calibration.json"
        for index, stage in enumerate(STAGES):
            # Every stage is a separate experiment from the immutable ID base;
            # no stage may resume another stage's optimizer or checkpoint.
            self.write_state(stage=stage, base_model=str(self.args.base_model), optimizer_reset=True)
            source_paths = self.collect_stage(stage, index)
            selected = self.prepare_stage(stage, source_paths)
            training = self.train_stage(stage, selected)
            self.evaluate_stage(stage, training)
        self.write_state(phase="summary")
        self.summarize()
        (self.root / "PIPELINE_COMPLETE").write_text("complete\n", encoding="utf-8")
        self.write_state(phase="complete")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--id-h5", type=Path, required=True)
    parser.add_argument("--pca-asset", type=Path, required=True)
    parser.add_argument("--pca-threshold", type=float)
    parser.add_argument("--pca-calibration", type=Path)
    parser.add_argument("--protocol-audit", type=Path, required=True)
    parser.add_argument("--pca-min-ood-fraction", type=float, default=0.80)
    parser.add_argument("--gpus", default="4,5")
    parser.add_argument("--cpu-sets", default="80-99,100-119")
    parser.add_argument("--training-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=8200)
    parser.add_argument("--geometry", choices=("v1", "v2"), default="v2")
    args = parser.parse_args()
    controller = Controller(args)
    try:
        controller.run()
    except Exception as exc:
        controller.write_state(phase="failed", error=repr(exc))
        (controller.root / "PIPELINE_FAILED").write_text(repr(exc) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
