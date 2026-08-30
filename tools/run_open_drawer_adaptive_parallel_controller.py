#!/usr/bin/env python3
"""Run the OpenDrawer adaptive timing sweep with a bounded worker pool.

The pilot rule and the one-model-per-anchor rule stay unchanged.  The only
relaxed constraint is scheduling: up to four independently audited idle GPUs
may train different anchors concurrently.  All initial models reach 5000
steps; the pilot's first strict Grasp-OOD-20 rate above 0.40 freezes the
cumulative step count, and any shorter model is continued from its own
checkpoint to that same count.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ANCHORS = (0, 50, 80, 120, 160, 220)
MAX_WORKERS = 4
MIN_STEPS = 5000
INCREMENT = 2500
OOD_THRESHOLD = 0.4
TRAIN_SEED = 9301
OOD_SEED_START = 79000
POLL_SECONDS = 10


def env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


ROOT = env_path("OPEN_DRAWER_ROOT", "/data/zhaozhixuan/Ask4Help-open-drawer")
RL = env_path("OPEN_DRAWER_RLINF_ROOT", str(ROOT / "RLinf"))
PYTHON = env_path("OPEN_DRAWER_PYTHON", "/data/zhaozhixuan/Ask4Help-airplane-5090/RLinf/.venv/bin/python")
RUN = env_path(
    "OPEN_DRAWER_TIMING_ROOT",
    str(ROOT / "results/open_drawer_grasp_timing_sweep_v1_retry8_adaptive"),
)
MODEL = env_path("OPEN_DRAWER_TIMING_CHECKPOINT", "")
PI05_BASE = env_path("OPEN_DRAWER_TIMING_PI05_BASE", str(ROOT / "results/model_cache/pi05_base_pytorch_v1"))
ID_DATASET = env_path(
    "OPEN_DRAWER_TIMING_ID_DATASET",
    str(ROOT / "results/id_oracle_collection_v1/lerobot_datasets/open_drawer_retrieve_place/id_oracle_128_retry1_v1"),
)
NORM = env_path(
    "OPEN_DRAWER_TIMING_NORM",
    str(ROOT / "results/id_policy_training_v1/norm_stats_open_drawer_id_raw_v1"),
)
FORMAL_ROOT = env_path(
    "OPEN_DRAWER_TIMING_FORMAL_ROOT",
    str(ROOT / "results/open_drawer_grasp_timing_sweep_v1_formal/formal"),
)
BUDGET_ROOT = env_path(
    "OPEN_DRAWER_TIMING_BUDGET_ROOT",
    str(ROOT / "results/open_drawer_grasp_timing_sweep_v1_retry3/formal_budget"),
)
GPU_POOL = tuple(int(value) for value in os.environ.get("OPEN_DRAWER_TIMING_GPU_POOL", "0 1 2 3 4 5 6 7").split())
RAY_TMP_ROOT = env_path("OPEN_DRAWER_TIMING_RAY_TMP_ROOT", "/sdd/od_adaptive_ray_retry8")
TMP_ROOT = env_path("OPEN_DRAWER_TIMING_TMP_ROOT", "/sdd/od_adaptive_tmp_retry8")
LOCK_ROOT = env_path("OPEN_DRAWER_TIMING_GPU_LOCK_ROOT", str(RUN / ".gpu_locks"))
STATE = RUN / "adaptive_timing_pipeline_state.json"
LOG = RUN / "adaptive_timing_parallel_controller.log"
FROZEN_STEPS = RUN / "adaptive_steps.json"


def write_state(stage: str, status: str, detail: str = "") -> None:
    payload = {
        "format": "open_drawer_adaptive_timing_parallel_v1",
        "stage": stage,
        "status": status,
        "detail": detail,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "anchors": list(ANCHORS),
        "training_seed": TRAIN_SEED,
        "max_workers": MAX_WORKERS,
    }
    STATE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def fail(stage: str, detail: str) -> "NoReturn":
    write_state(stage, "failed", detail)
    (RUN / "ADAPTIVE_TIMING_FAILED").write_text(f"stage={stage} detail={detail}\n", encoding="utf-8")
    raise SystemExit(1)


def cpu_set(gpu: int) -> str:
    if gpu < 0 or gpu > 7:
        raise ValueError(f"unsupported GPU {gpu}")
    return f"{20 * gpu}-{20 * gpu + 19}"


def gpu_is_idle(gpu: int) -> bool:
    try:
        query = subprocess.check_output(
            ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used,utilization.gpu,uuid", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        used_s, util_s, uuid = [value.strip() for value in query.split(",", 2)]
        apps = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError, ValueError):
        return False
    return int(float(used_s)) <= 100 and int(float(util_s)) <= 5 and uuid not in apps


def acquire_gpu(held: set[int]) -> tuple[int, Path] | None:
    LOCK_ROOT.mkdir(parents=True, exist_ok=True)
    for gpu in GPU_POOL:
        if gpu in held or not gpu_is_idle(gpu):
            continue
        lock = LOCK_ROOT / f"gpu_{gpu}"
        try:
            lock.mkdir()
        except FileExistsError:
            owner_path = lock / "owner"
            try:
                owner = int(owner_path.read_text().strip())
            except (OSError, ValueError):
                owner = 0
            if owner and not _pid_alive(owner):
                try:
                    owner_path.unlink()
                    lock.rmdir()
                except OSError:
                    pass
            continue
        if not gpu_is_idle(gpu):
            shutil.rmtree(lock, ignore_errors=True)
            continue
        (lock / "owner").write_text(f"{os.getpid()}\n", encoding="utf-8")
        held.add(gpu)
        return gpu, lock
    return None


def release_gpu(gpu: int, held: set[int]) -> None:
    lock = LOCK_ROOT / f"gpu_{gpu}"
    try:
        (lock / "owner").unlink()
    except OSError:
        pass
    try:
        lock.rmdir()
    except OSError:
        pass
    held.discard(gpu)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def segment_root(anchor: int, cumulative: int) -> Path:
    return RUN / "training" / f"anchor_{anchor}" / f"seed_{TRAIN_SEED}" / f"steps_{cumulative}"


def segment_checkpoint(anchor: int, cumulative: int, segment_steps: int) -> Path:
    return segment_root(anchor, cumulative) / "run" / "checkpoints" / f"global_step_{segment_steps}"


def marker_checkpoint(marker: Path) -> Path:
    match = re.search(r"(?:^|\s)checkpoint=([^\s]+)", marker.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"checkpoint missing in {marker}")
    path = Path(match.group(1))
    return path if path.is_absolute() else marker.parent / path


def complete_checkpoint(path: Path) -> bool:
    return (path / "actor/model_state_dict/full_weights.pt").is_file() and (path / "actor/dcp_checkpoint").is_dir()


def checkpoint_segment_steps(path: Path) -> int:
    match = re.fullmatch(r"global_step_(\d+)", path.name)
    if match is None:
        raise ValueError(f"invalid checkpoint directory name: {path}")
    return int(match.group(1))


def find_existing(anchor: int) -> tuple[int, Path] | None:
    root = RUN / "training" / f"anchor_{anchor}" / f"seed_{TRAIN_SEED}"
    if not root.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for path in root.glob("steps_*/SEGMENT_COMPLETE"):
        try:
            cumulative = int(path.parent.name.split("_", 1)[1])
            checkpoint = marker_checkpoint(path)
        except (OSError, ValueError):
            continue
        if complete_checkpoint(checkpoint):
            candidates.append((cumulative, checkpoint))
    return max(candidates, key=lambda item: item[0]) if candidates else None


def audit_probe(output: Path) -> float:
    summary_path = output / "summary.json"
    if not summary_path.is_file():
        raise ValueError(f"missing {summary_path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if payload.get("split") != "grasp_ood" or payload.get("episodes") != 20 or len(payload.get("rows", [])) != 20:
        raise ValueError(f"probe denominator mismatch: {summary_path}")
    if len(list((output / "videos").glob("*.mp4"))) != 20:
        raise ValueError(f"probe video denominator mismatch: {output}")
    for row in payload["rows"]:
        for key in ("actions", "states", "timeline", "reset_metadata", "video"):
            path = Path(str(row.get(key, "")))
            if not path.is_file():
                raise ValueError(f"missing {key}: {path}")
        actions = np.load(Path(str(row["actions"])))
        states = np.load(Path(str(row["states"])))
        if actions.ndim != 2 or actions.shape[1] != 8 or states.shape != (len(actions) + 1, 9):
            raise ValueError(f"shape mismatch in {row}")
        timeline = json.loads(Path(str(row["timeline"])).read_text(encoding="utf-8"))
        values = timeline.get("timeline", timeline.get("rows", []))
        if len(values) != len(actions) + 1:
            raise ValueError(f"timeline mismatch in {row}")
        json.loads(Path(str(row["reset_metadata"])).read_text(encoding="utf-8"))
    return float(payload["successes"]) / float(payload["episodes"])


def run_probe(anchor: int, cumulative: int, checkpoint: Path, gpu: int, cpuset: str, probe_index: int) -> float:
    output = RUN / "ood20_probe" / f"anchor_{anchor}" / f"seed_{TRAIN_SEED}" / f"steps_{cumulative}" / "grasp_ood"
    marker = output.parent / "OOD20_COMPLETE"
    if marker.is_file():
        return audit_probe(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing partial probe output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    write_state(f"ood20_anchor_{anchor}_steps_{cumulative}", "running", f"gpu={gpu}")
    command = [
        str(PYTHON), "-u", str(ROOT / "tools/evaluate_open_drawer_id_pi05.py"),
        "--checkpoint", str(checkpoint), "--pi05-base", str(PI05_BASE), "--norm-stats", str(NORM),
        "--output-dir", str(output), "--episodes", "20", "--seed", str(OOD_SEED_START + probe_index * 100),
        "--split", "grasp_ood", "--execute-horizon", "5", "--max-episode-steps", "400",
    ]
    log_path = RUN / "logs" / f"ood20_anchor_{anchor}_steps_{cumulative}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "ASK4HELP_RLINF_ROOT": str(RL),
        "PYTHONPATH": f"{ROOT}:{RL}",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "20",
        "MKL_NUM_THREADS": "20",
        "PYTHONUNBUFFERED": "1",
    })
    with log_path.open("w", encoding="utf-8") as stream:
        result = subprocess.run(["taskset", "-c", cpuset, *command], env=env, stdout=stream, stderr=subprocess.STDOUT, check=False)
    rate = audit_probe(output)
    marker.write_text(f"20 Grasp-OOD episodes audited; returncode={result.returncode}\n", encoding="utf-8")
    return rate


@dataclass
class Job:
    anchor: int
    target_steps: int
    segment_steps: int
    source: Path
    output: Path
    checkpoint: Path
    gpu: int
    lock: Path
    process: subprocess.Popen[str]


def launch_training(anchor: int, target_steps: int, segment_steps: int, source: Path, gpu: int, lock: Path) -> Job:
    output = segment_root(anchor, target_steps)
    checkpoint = segment_checkpoint(anchor, target_steps, segment_steps)
    if output.exists() and any(output.iterdir()):
        if complete_checkpoint(checkpoint):
            marker = output / "SEGMENT_COMPLETE"
            if not marker.exists():
                marker.write_text(f"checkpoint={checkpoint} source={source} segment_steps={segment_steps} cumulative_steps={target_steps}\n", encoding="utf-8")
            return Job(anchor, target_steps, segment_steps, source, output, checkpoint, gpu, lock, _completed_process())
        raise ValueError(f"refusing partial training output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    cpuset = cpu_set(gpu)
    config_path = RL / "examples/sft/config/open_drawer_retrieve_place_dagger_sft_openpi_pi05.yaml"
    if "default_prompt: open the drawer, retrieve the blue object, and place it in the green tray" not in config_path.read_text(encoding="utf-8"):
        raise ValueError(f"canonical prompt missing in {config_path}")
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "ASK4HELP_RLINF_PLACEMENT": f"{gpu}-{gpu}",
        "RLINF_RAY_ADDRESS": "local",
        "EMBODIED_PATH": str(RL / "examples/sft"),
        "PYTHONPATH": f"{ROOT}:{RL}",
        "OPEN_DRAWER_ID_DATASET": str(ID_DATASET),
        "OPEN_DRAWER_EXPERT_DATASET": str(BUDGET_ROOT / f"anchor_{anchor}"),
        "OPEN_DRAWER_ID_NORM_STATS": str(NORM),
        "OPEN_DRAWER_PI05_MODEL_PATH": str(source),
        "OPEN_DRAWER_RUN_ROOT": str(output),
        "OPEN_DRAWER_EXPERIMENT_NAME": "run",
        "OPEN_DRAWER_TRAIN_SEED": str(TRAIN_SEED),
        "OPEN_DRAWER_RESUME_DIR": "",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_DATASETS_CACHE": str(RUN / "runtime/hf_datasets"),
        "HF_HOME": str(RUN / "runtime/hf_home"),
        "RAY_TMPDIR": str(RAY_TMP_ROOT / f"a{anchor}s{target_steps}"),
        "TMPDIR": str(TMP_ROOT / f"a{anchor}s{target_steps}"),
        "PYTHONUNBUFFERED": "1",
    })
    (RAY_TMP_ROOT / f"a{anchor}s{target_steps}").mkdir(parents=True, exist_ok=True)
    (TMP_ROOT / f"a{anchor}s{target_steps}").mkdir(parents=True, exist_ok=True)
    log_path = RUN / "logs" / f"train_anchor_{anchor}_steps_{target_steps}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(PYTHON), str(RL / "examples/sft/train_vla_sft.py"),
        "--config-path", str(RL / "examples/sft/config"),
        "--config-name", "open_drawer_retrieve_place_dagger_sft_openpi_pi05",
        f"runner.max_steps={segment_steps}", "runner.save_interval=500",
        f"actor.optim.total_training_steps={segment_steps}", "runner.resume_dir=null",
    ]
    stream = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(["taskset", "-c", cpuset, *command], cwd=str(RL / "examples/sft"), env=env, stdout=stream, stderr=subprocess.STDOUT, text=True)
    return Job(anchor, target_steps, segment_steps, source, output, checkpoint, gpu, lock, process)


def _completed_process() -> subprocess.Popen[str]:
    # Existing checkpoints are handled by the event loop without a child.
    return subprocess.Popen(["true"], text=True)


def finish_job(job: Job) -> None:
    code = job.process.wait()
    if code != 0 or not complete_checkpoint(job.checkpoint):
        raise RuntimeError(f"training failed anchor={job.anchor} target={job.target_steps} returncode={code}")
    marker = job.output / "SEGMENT_COMPLETE"
    marker.write_text(
        f"checkpoint={job.checkpoint} source={job.source} segment_steps={job.segment_steps} cumulative_steps={job.target_steps}\n",
        encoding="utf-8",
    )


def freeze(rate: float, steps: int) -> None:
    FROZEN_STEPS.write_text(json.dumps({
        "frozen_steps": steps,
        "pilot_anchor": 0,
        "training_seed": TRAIN_SEED,
        "ood_success_rate": rate,
        "threshold": OOD_THRESHOLD,
        "rule": "first strict OOD20 rate above threshold",
        "parallel_initial_training": True,
    }, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "training").mkdir(exist_ok=True)
    (RUN / "ood20_probe").mkdir(exist_ok=True)
    (RUN / "logs").mkdir(exist_ok=True)
    (RUN / "runtime").mkdir(exist_ok=True)
    if not MODEL:
        fail("preflight", "OPEN_DRAWER_TIMING_CHECKPOINT is required")
    for path in (MODEL / "actor/model_state_dict/full_weights.pt", FORMAL_ROOT / "TIMING_COLLECTION_COMPLETE", FORMAL_ROOT / "AUDIT_PASS", BUDGET_ROOT / "BUDGET_AUDIT_PASS"):
        if not path.exists():
            fail("preflight", f"missing {path}")
    (RUN / "formal").symlink_to(FORMAL_ROOT, target_is_directory=True) if not (RUN / "formal").exists() else None
    (RUN / "formal_budget").symlink_to(BUDGET_ROOT, target_is_directory=True) if not (RUN / "formal_budget").exists() else None
    RAY_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    write_state("preflight", "running", f"parallel_workers={MAX_WORKERS} gpu_pool={GPU_POOL}")
    held: set[int] = set()
    active: dict[int, Job] = {}
    pending: deque[tuple[int, int, int, Path]] = deque()
    completed_steps: dict[int, int] = {}
    frozen_steps: int | None = None
    if FROZEN_STEPS.is_file():
        frozen_steps = int(json.loads(FROZEN_STEPS.read_text(encoding="utf-8"))["frozen_steps"])
    for anchor in ANCHORS:
        existing = find_existing(anchor)
        if existing is not None:
            completed_steps[anchor] = existing[0]
            probe_marker = RUN / "ood20_probe" / f"anchor_{anchor}" / f"seed_{TRAIN_SEED}" / f"steps_{existing[0]}" / "OOD20_COMPLETE"
            if not probe_marker.is_file():
                pending.append((anchor, existing[0], checkpoint_segment_steps(existing[1]), MODEL))
            elif frozen_steps is not None and existing[0] < frozen_steps:
                pending.append((anchor, frozen_steps, frozen_steps - existing[0], existing[1]))
        elif frozen_steps is None:
            pending.append((anchor, MIN_STEPS, MIN_STEPS, MODEL))
        else:
            pending.append((anchor, frozen_steps, frozen_steps, MODEL))
    probe_counter = len(list((RUN / "ood20_probe").glob("anchor_*/seed_*/steps_*/OOD20_COMPLETE")))
    pilot_rate: float | None = None
    try:
        while pending or active:
            # Fill the bounded pool from the queue, using only audited idle GPUs.
            while pending and len(active) < MAX_WORKERS:
                acquired = acquire_gpu(held)
                if acquired is None:
                    break
                gpu, lock = acquired
                anchor, target, segment_steps, source = pending.popleft()
                try:
                    job = launch_training(anchor, target, segment_steps, source, gpu, lock)
                except Exception:
                    release_gpu(gpu, held)
                    raise
                active[gpu] = job
                write_state(f"parallel_training_{len(active)}_jobs", "running", f"anchor={anchor} target_steps={target} gpu={gpu}")
            progressed = False
            for gpu, job in list(active.items()):
                if job.process.poll() is None:
                    continue
                finish_job(job)
                completed_steps[job.anchor] = job.target_steps
                probe_index = probe_counter
                probe_counter += 1
                try:
                    # Keep the just-finished job's audited GPU for its paired
                    # OOD20 probe; no new allocation or race is needed.
                    rate = run_probe(job.anchor, job.target_steps, job.checkpoint, gpu, cpu_set(gpu), probe_index)
                finally:
                    release_gpu(gpu, held)
                del active[gpu]
                write_state(f"ood20_anchor_{job.anchor}_complete", "running", f"strict_rate={rate}")
                if job.anchor == 0 and (frozen_steps is None or job.target_steps <= frozen_steps):
                    pilot_rate = rate
                    if rate > OOD_THRESHOLD:
                        frozen_steps = job.target_steps
                        freeze(rate, frozen_steps)
                    else:
                        next_steps = job.target_steps + INCREMENT
                        pending.append((0, next_steps, INCREMENT, job.checkpoint))
                if frozen_steps is not None and completed_steps.get(job.anchor, 0) < frozen_steps:
                    current = completed_steps[job.anchor]
                    pending.append((job.anchor, frozen_steps, frozen_steps - current, job.checkpoint))
                # When the pilot freezes, upgrade any anchors that finished early.
                if frozen_steps is not None:
                    for anchor in ANCHORS:
                        current = completed_steps.get(anchor, 0)
                        if current and current < frozen_steps and not any(task[0] == anchor and task[1] == frozen_steps for task in pending) and not any(item.anchor == anchor for item in active.values()):
                            source_existing = find_existing(anchor)
                            source_path = source_existing[1] if source_existing is not None else MODEL
                            pending.append((anchor, frozen_steps, frozen_steps - current, source_path))
                progressed = True
            if not progressed:
                time.sleep(POLL_SECONDS)
        if frozen_steps is None:
            fail("pilot", "pilot did not produce a passing or continuation result")
        missing = [anchor for anchor in ANCHORS if completed_steps.get(anchor) != frozen_steps]
        if missing:
            fail("parallel_training", f"missing frozen-step anchors: {missing}")
        (RUN / "ADAPTIVE_TIMING_TRAINING_COMPLETE").write_text(
            f"all six anchors trained with seed={TRAIN_SEED}, frozen_steps={frozen_steps}, parallel_workers={MAX_WORKERS}\n",
            encoding="utf-8",
        )
        write_state("all_adaptive_training", "complete", f"frozen_steps={frozen_steps} pilot_rate={pilot_rate}")
        print("OPEN_DRAWER_ADAPTIVE_TIMING_PARALLEL_COMPLETE", flush=True)
    except KeyboardInterrupt:
        for job in active.values():
            job.process.send_signal(signal.SIGTERM)
        raise
    except Exception as exc:
        for job in active.values():
            job.process.send_signal(signal.SIGTERM)
        fail("parallel_controller", repr(exc))
    finally:
        for gpu in list(held):
            release_gpu(gpu, held)


if __name__ == "__main__":
    main()
