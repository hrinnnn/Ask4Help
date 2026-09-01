#!/usr/bin/env python3
"""Restart-tolerant recovery controller for the direct-grasp adaptive sweep.

The first two timing jobs were deliberately left running when the 5090
resource cap was reduced to two cards.  Their original parent controllers were
stopped, so this controller adopts those already-running jobs, waits at long
intervals, audits their final checkpoints, and then resumes the pre-registered
train -> Grasp-OOD20 -> next-train protocol.  It never changes the checkpoint,
seed, anchor, budget, threshold, or success predicate.

This is an operational recovery controller, not a new scientific policy.  It
fails closed on an unexpected partial output, process mismatch, or artifact
audit failure.  The two currently retained jobs can be supplied through
OPEN_DRAWER_RETAINED_JOBS_JSON; the default is the audited PID/GPU mapping from
the leadership-cap pause.
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


ANCHORS = (0, 50, 80, 120, 160, 220)
TRAIN_SEED = 9301
MIN_STEPS = 5000
INCREMENT = 2500
OOD_THRESHOLD = 0.4
MAX_WORKERS = 2
FIXED_5000_OVERRIDE = os.environ.get("OPEN_DRAWER_FIXED_5000", "0") == "1"

# The long sleeps are intentional.  This process is the durable supervisor;
# the actual training/evaluation subprocesses are independent of it.
CHECK_SECONDS = int(os.environ.get("OPEN_DRAWER_RECOVERY_CHECK_SECONDS", "900"))
STABLE_CHECK_SECONDS = int(os.environ.get("OPEN_DRAWER_RECOVERY_STABLE_SECONDS", "1800"))
GPU_WAIT_SECONDS = int(os.environ.get("OPEN_DRAWER_RECOVERY_GPU_WAIT_SECONDS", "900"))


def env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


ROOT = env_path("OPEN_DRAWER_ROOT", "/data/zhaozhixuan/Ask4Help-open-drawer")
RL = env_path("OPEN_DRAWER_RLINF_ROOT", str(ROOT / "RLinf"))
PYTHON = env_path(
    "OPEN_DRAWER_PYTHON",
    "/data/zhaozhixuan/Ask4Help-airplane-5090/RLinf/.venv/bin/python",
)
MODEL = env_path("OPEN_DRAWER_TIMING_CHECKPOINT", "")
PI05_BASE = env_path(
    "OPEN_DRAWER_TIMING_PI05_BASE",
    str(ROOT / "results/model_cache/pi05_base_pytorch_v1"),
)
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
    str(ROOT / "results/open_drawer_grasp_timing_sweep_v1_direct_oracle_formal_retry1/formal"),
)
BUDGET_ROOT = env_path(
    "OPEN_DRAWER_TIMING_BUDGET_ROOT",
    str(ROOT / "results/open_drawer_grasp_timing_sweep_v1_direct_oracle_formal_retry1/formal_budget"),
)
RUN = env_path(
    "OPEN_DRAWER_TIMING_ROOT",
    str(ROOT / "results/open_drawer_grasp_timing_sweep_v1_direct_oracle_adaptive_retry1"),
)
GPU_POOL = tuple(
    int(value)
    for value in os.environ.get("OPEN_DRAWER_TIMING_GPU_POOL", "0 1 3 4 6 7").split()
)
RAY_TMP_ROOT = env_path("OPEN_DRAWER_TIMING_RAY_TMP_ROOT", "/sdd/r_od1")
TMP_ROOT = env_path("OPEN_DRAWER_TIMING_TMP_ROOT", "/sdd/t_od1")
OOD_SEED_START = int(os.environ.get("OPEN_DRAWER_OOD20_SEED_START", "79000"))

RECOVERY_STATE = RUN / "direct_oracle_adaptive_recovery_state.json"
MAIN_STATE = RUN / "adaptive_timing_pipeline_state.json"
TOTAL_STATE = RUN / "direct_oracle_adaptive_total_state.json"
RECOVERY_LOG = RUN / "logs/direct_oracle_adaptive_recovery.log"
LOCK_ROOT = RUN / ".recovery_controller_lock"
JOB_STATE = RUN / "recovery_active_jobs.json"
FROZEN_STEPS = RUN / "adaptive_steps.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def write_recovery_state(stage: str, status: str, detail: str = "", **extra: Any) -> None:
    payload: dict[str, Any] = {
        "format": "open_drawer_direct_oracle_adaptive_recovery_v1",
        "stage": stage,
        "status": status,
        "detail": detail,
        "updated_at": now(),
        "anchors": list(ANCHORS),
        "training_seed": TRAIN_SEED,
        "max_workers": MAX_WORKERS,
        "check_seconds": CHECK_SECONDS,
        "stable_check_seconds": STABLE_CHECK_SECONDS,
    }
    payload.update(extra)
    atomic_json(RECOVERY_STATE, payload)


def write_main_state(stage: str, status: str, detail: str = "", **extra: Any) -> None:
    """Keep the canonical adaptive state readable by existing auditors."""

    payload: dict[str, Any] = {
        "format": "open_drawer_adaptive_timing_parallel_v1",
        "stage": stage,
        "status": status,
        "detail": detail,
        "updated_at": now(),
        "anchors": list(ANCHORS),
        "training_seed": TRAIN_SEED,
        "max_workers": MAX_WORKERS,
    }
    payload.update(extra)
    atomic_json(MAIN_STATE, payload)


def write_total_state(stage: str, status: str, detail: str = "", **extra: Any) -> None:
    payload: dict[str, Any] = {
        "format": "open_drawer_direct_oracle_adaptive_total_v1",
        "stage": stage,
        "status": status,
        "detail": detail,
        "updated_at": now(),
        "max_cards": MAX_WORKERS,
        "retained_anchors": [0, 50],
        "paused_anchors": [80, 120],
    }
    payload.update(extra)
    atomic_json(TOTAL_STATE, payload)


def append_log(message: str) -> None:
    RECOVERY_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RECOVERY_LOG.open("a", encoding="utf-8") as stream:
        stream.write(f"[{now()}] {message}\n")
        stream.flush()


def fail(stage: str, detail: str) -> "NoReturn":
    append_log(f"FAILED stage={stage} detail={detail}")
    write_recovery_state(stage, "failed", detail)
    write_main_state(stage, "failed", detail)
    write_total_state(stage, "failed", detail)
    (RUN / "DIRECT_ORACLE_ADAPTIVE_RECOVERY_FAILED").write_text(
        f"stage={stage} detail={detail}\n", encoding="utf-8"
    )
    raise SystemExit(1)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return False
    fields = stat.rsplit(")", 1)[-1].strip().split()
    return bool(fields) and fields[0] != "Z"


def process_cmd(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\0", b" ").decode(errors="replace").strip()


def process_env(pid: int) -> dict[str, str]:
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return {}
    result: dict[str, str] = {}
    for item in raw.split(b"\0"):
        if b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        result[key.decode(errors="replace")] = value.decode(errors="replace")
    return result


def cpu_set(gpu: int) -> str:
    if gpu < 0 or gpu > 7:
        raise ValueError(f"unsupported GPU {gpu}")
    return f"{20 * gpu}-{20 * gpu + 19}"


def complete_checkpoint(path: Path) -> bool:
    weights = path / "actor/model_state_dict/full_weights.pt"
    dcp = path / "actor/dcp_checkpoint"
    return weights.is_file() and weights.stat().st_size > 0 and dcp.is_dir() and any(dcp.iterdir())


def checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"global_step_(\d+)", path.name)
    if match is None:
        raise ValueError(f"invalid checkpoint path: {path}")
    return int(match.group(1))


def canonical_root(anchor: int, target: int) -> Path:
    return RUN / "training" / f"anchor_{anchor}" / f"seed_{TRAIN_SEED}" / f"steps_{target}"


def checkpoints_under(root: Path) -> list[tuple[int, Path]]:
    checkpoint_root = root / "run/checkpoints"
    if not checkpoint_root.is_dir():
        return []
    result: list[tuple[int, Path]] = []
    for path in checkpoint_root.glob("global_step_*"):
        try:
            step = checkpoint_step(path)
        except ValueError:
            continue
        if complete_checkpoint(path):
            result.append((step, path))
    return sorted(result)


def latest_checkpoint(root: Path) -> tuple[int, Path] | None:
    values = checkpoints_under(root)
    return values[-1] if values else None


def marker_checkpoint(marker: Path) -> Path:
    text = marker.read_text(encoding="utf-8")
    match = re.search(r"(?:^|\s)checkpoint=([^\s]+)", text)
    if match is None:
        raise ValueError(f"checkpoint missing from {marker}")
    path = Path(match.group(1))
    return path if path.is_absolute() else marker.parent / path


def segment_marker(anchor: int, target: int) -> Path:
    return canonical_root(anchor, target) / "SEGMENT_COMPLETE"


def mark_segment(anchor: int, target: int, checkpoint: Path, source: Path) -> None:
    marker = segment_marker(anchor, target)
    if marker.exists():
        try:
            existing = marker_checkpoint(marker)
        except ValueError as exc:
            fail(f"audit_anchor_{anchor}", str(exc))
        if existing != checkpoint:
            fail(f"audit_anchor_{anchor}", f"marker points to {existing}, expected {checkpoint}")
        return
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        f"checkpoint={checkpoint} source={source} segment_steps={target} cumulative_steps={target}\n",
        encoding="utf-8",
    )


def audited_marker(anchor: int, target: int) -> Path | None:
    marker = segment_marker(anchor, target)
    if not marker.is_file():
        return None
    try:
        checkpoint = marker_checkpoint(marker)
    except ValueError as exc:
        fail(f"audit_anchor_{anchor}", str(exc))
    if not complete_checkpoint(checkpoint):
        fail(f"audit_anchor_{anchor}", f"incomplete checkpoint referenced by {marker}: {checkpoint}")
    return marker


def existing_marked_completions() -> dict[int, int]:
    """Return only canonical, fully audited training markers already present."""

    result: dict[int, int] = {}
    for anchor in ANCHORS:
        parent = RUN / "training" / f"anchor_{anchor}" / f"seed_{TRAIN_SEED}"
        if not parent.is_dir():
            continue
        candidates: list[tuple[int, Path]] = []
        for marker in parent.glob("steps_*/SEGMENT_COMPLETE"):
            match = re.fullmatch(r"steps_(\d+)", marker.parent.name)
            if match is None:
                continue
            target = int(match.group(1))
            try:
                checkpoint = marker_checkpoint(marker)
            except ValueError as exc:
                fail(f"preflight_anchor_{anchor}", str(exc))
            if not complete_checkpoint(checkpoint):
                fail(f"preflight_anchor_{anchor}", f"marker references incomplete checkpoint: {checkpoint}")
            candidates.append((target, marker))
        if candidates:
            target = max(candidates, key=lambda item: item[0])[0]
            probe_marker = RUN / "ood20_probe" / f"anchor_{anchor}" / f"seed_{TRAIN_SEED}" / f"steps_{target}" / "OOD20_COMPLETE"
            if not probe_marker.is_file():
                fail(
                    f"preflight_anchor_{anchor}",
                    f"existing training marker lacks paired audited OOD20 marker: {probe_marker}",
                )
            result[anchor] = target
    return result


def existing_fixed_5000_completions() -> dict[int, int]:
    """Adopt only audited cumulative-5000 artifacts under the fixed override."""

    result: dict[int, int] = {}
    for anchor in ANCHORS:
        root = canonical_root(anchor, MIN_STEPS)
        marker = root / "SEGMENT_COMPLETE"
        checkpoint = root / "run/checkpoints" / f"global_step_{MIN_STEPS}"
        if not marker.is_file():
            continue
        if not complete_checkpoint(checkpoint):
            fail(
                f"fixed5000_anchor_{anchor}",
                f"5000-step marker lacks complete checkpoint: {checkpoint}",
            )
        probe_output = RUN / "ood20_probe" / f"anchor_{anchor}" / f"seed_{TRAIN_SEED}" / f"steps_{MIN_STEPS}" / "grasp_ood"
        probe_marker = probe_output.parent / "OOD20_COMPLETE"
        if not probe_marker.is_file():
            continue
        result[anchor] = MIN_STEPS
    return result


def audit_training_root(anchor: int, target: int) -> Path:
    root = canonical_root(anchor, target)
    checkpoint = root / "run/checkpoints" / f"global_step_{target}"
    if not complete_checkpoint(checkpoint):
        fail(f"audit_anchor_{anchor}", f"missing complete final checkpoint: {checkpoint}")
    return checkpoint


def load_retained_jobs() -> dict[int, dict[str, int]]:
    default = {
        0: {"pid": 4190202, "gpu": 4},
        50: {"pid": 2650153, "gpu": 7},
    }
    raw = os.environ.get("OPEN_DRAWER_RETAINED_JOBS_JSON", "")
    if not raw:
        path = RUN / "retained_training_jobs.json"
        if path.is_file():
            raw = path.read_text(encoding="utf-8")
    if not raw:
        return default
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        fail("preflight", f"invalid retained job mapping: {exc}")
    result: dict[int, dict[str, int]] = {}
    for key, value in payload.items():
        anchor = int(key)
        if anchor not in (0, 50):
            continue
        result[anchor] = {"pid": int(value["pid"]), "gpu": int(value["gpu"])}
    if set(result) != {0, 50}:
        fail("preflight", f"retained job mapping must contain anchors 0 and 50: {result}")
    atomic_json(RUN / "retained_training_jobs.json", {str(k): v for k, v in result.items()})
    return result


def verify_retained_process(anchor: int, pid: int) -> None:
    if not pid_alive(pid):
        return
    cmd = process_cmd(pid)
    env = process_env(pid)
    expected_root = str(canonical_root(anchor, MIN_STEPS))
    if "train_vla_sft.py" not in cmd or env.get("OPEN_DRAWER_RUN_ROOT") != expected_root:
        fail(
            "preflight",
            f"retained pid mismatch anchor={anchor} pid={pid} root={env.get('OPEN_DRAWER_RUN_ROOT')} cmd={cmd[:240]}",
        )


def gpu_is_idle(gpu: int) -> bool:
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                str(gpu),
                "--query-gpu=memory.used,utilization.gpu,uuid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        used_s, util_s, uuid = [item.strip() for item in raw.split(",", 2)]
        apps = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return int(float(used_s)) <= 100 and int(float(util_s)) <= 5 and uuid not in apps
    except (OSError, subprocess.CalledProcessError, ValueError):
        return False


def pid_for_lock(lock: Path) -> int:
    try:
        return int((lock / "owner").read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return 0


def acquire_gpu(held: set[int]) -> tuple[int, Path] | None:
    LOCK_ROOT.mkdir(parents=True, exist_ok=True)
    for gpu in GPU_POOL:
        if gpu in held or not gpu_is_idle(gpu):
            continue
        lock = LOCK_ROOT / f"gpu_{gpu}"
        try:
            lock.mkdir()
        except FileExistsError:
            owner = pid_for_lock(lock)
            if owner and not pid_alive(owner):
                try:
                    (lock / "owner").unlink()
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


def audit_probe(output: Path) -> float:
    summary_path = output / "summary.json"
    if not summary_path.is_file():
        raise ValueError(f"missing probe summary: {summary_path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if payload.get("split") != "grasp_ood" or payload.get("episodes") != 20:
        raise ValueError(f"probe split/denominator mismatch: {summary_path}")
    rows = payload.get("rows", [])
    if len(rows) != 20 or len(list((output / "videos").glob("*.mp4"))) != 20:
        raise ValueError(f"probe rows/video denominator mismatch: {output}")
    for row in rows:
        for key in ("actions", "states", "timeline", "reset_metadata", "video"):
            path = Path(str(row.get(key, "")))
            if not path.is_file():
                raise ValueError(f"missing {key}: {path}")
        import numpy as np

        actions = np.load(Path(str(row["actions"])))
        states = np.load(Path(str(row["states"])))
        if actions.ndim != 2 or actions.shape[1] != 8:
            raise ValueError(f"bad action shape: {actions.shape}")
        if states.shape != (len(actions) + 1, 9):
            raise ValueError(f"bad state shape: {states.shape}")
        timeline = json.loads(Path(str(row["timeline"])).read_text(encoding="utf-8"))
        values = timeline.get("timeline", timeline.get("rows", []))
        if len(values) != len(actions) + 1:
            raise ValueError("timeline/action alignment mismatch")
        json.loads(Path(str(row["reset_metadata"])).read_text(encoding="utf-8"))
    return float(payload["successes"]) / float(payload["episodes"])


def run_probe(anchor: int, target: int, checkpoint: Path, gpu: int, probe_index: int) -> float:
    output_root = RUN / "ood20_probe" / f"anchor_{anchor}" / f"seed_{TRAIN_SEED}" / f"steps_{target}"
    output = output_root / "grasp_ood"
    marker = output_root / "OOD20_COMPLETE"
    if marker.is_file():
        try:
            return audit_probe(output)
        except Exception as exc:
            fail(f"ood20_anchor_{anchor}_steps_{target}", f"existing probe audit failed: {exc}")
    if output_root.exists() and any(output_root.iterdir()):
        fail(f"ood20_anchor_{anchor}_steps_{target}", f"refusing partial probe output: {output_root}")
    output.mkdir(parents=True, exist_ok=True)
    log_path = RUN / "logs" / f"recovery_ood20_anchor_{anchor}_steps_{target}.log"
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "ASK4HELP_RLINF_ROOT": str(RL),
            "PYTHONPATH": f"{ROOT / 'tools' / 'adaptive_ray_shim'}:{ROOT}:{RL}",
            "ASK4HELP_RAY_OBJECT_STORE_MEMORY": str(100 * 1024**3),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "20",
            "MKL_NUM_THREADS": "20",
            "PYTHONUNBUFFERED": "1",
        }
    )
    command = [
        str(PYTHON),
        "-u",
        str(ROOT / "tools/evaluate_open_drawer_id_pi05.py"),
        "--checkpoint",
        str(checkpoint),
        "--pi05-base",
        str(PI05_BASE),
        "--norm-stats",
        str(NORM),
        "--output-dir",
        str(output),
        "--episodes",
        "20",
        "--seed",
        str(OOD_SEED_START + probe_index * 100),
        "--split",
        "grasp_ood",
        "--execute-horizon",
        "5",
        "--max-episode-steps",
        "400",
    ]
    write_recovery_state(
        f"ood20_anchor_{anchor}_steps_{target}",
        "running",
        f"gpu={gpu} seed={OOD_SEED_START + probe_index * 100}",
    )
    with log_path.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            ["taskset", "-c", cpu_set(gpu), *command],
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    try:
        rate = audit_probe(output)
    except Exception as exc:
        fail(f"ood20_anchor_{anchor}_steps_{target}", f"probe artifact audit failed: {exc}")
    marker.write_text(
        f"20 Grasp-OOD episodes audited; returncode={result.returncode}; strict_rate={rate:.6f}\n",
        encoding="utf-8",
    )
    append_log(f"OOD20 anchor={anchor} target={target} rate={rate:.4f} returncode={result.returncode}")
    return rate


@dataclass
class Job:
    anchor: int
    target: int
    source: Path
    output: Path
    checkpoint: Path
    gpu: int
    lock: Path
    process: subprocess.Popen[Any]
    stream: Any


def prepare_resume_source(anchor: int, target: int) -> Path | None:
    """Preserve the leadership-pause partial root and return a valid source."""

    root = canonical_root(anchor, target)
    marker = segment_marker(anchor, target)
    if marker.is_file():
        checkpoint = marker_checkpoint(marker)
        if not complete_checkpoint(checkpoint):
            fail(f"preflight_anchor_{anchor}", f"existing marker references incomplete checkpoint: {checkpoint}")
        return checkpoint
    if root.exists() and not any(root.iterdir()):
        root.rmdir()
        return None

    # A partial root may be the exact target (the two jobs paused at 500), or a
    # lower-step root when the pilot needs one or more +2500 continuations.  We
    # never overwrite it: only the exact target root is renamed, while a lower
    # root remains in place as the immutable resume source.
    candidates: list[tuple[int, Path, Path]] = []
    parent = root.parent
    if parent.is_dir():
        for candidate_root in parent.glob("steps_*"):
            if not candidate_root.is_dir():
                continue
            match = re.fullmatch(r"steps_(\d+)(?:_partial_from_\d+)?", candidate_root.name)
            if match is None:
                continue
            candidate_target = int(match.group(1))
            latest = latest_checkpoint(candidate_root)
            if latest is None:
                continue
            step, checkpoint = latest
            if step < target:
                candidates.append((step, candidate_root, checkpoint))
    if not candidates:
        if root.exists() and any(root.iterdir()):
            fail(f"preflight_anchor_{anchor}", f"partial root has no complete checkpoint below target={target}: {root}")
        return None
    step, source_root, source = max(candidates, key=lambda item: item[0])
    if source_root == root:
        preserved = root.with_name(root.name + f"_partial_from_{step}")
        if preserved.exists():
            fail(
                f"preflight_anchor_{anchor}",
                f"both target and preserved partial roots exist; refusing overwrite: {root}, {preserved}",
            )
        shutil.move(str(root), str(preserved))
        source_root = preserved
        source = preserved / "run/checkpoints" / f"global_step_{step}"
        atomic_json(
            preserved / "PARTIAL_TRAINING_PRESERVED.json",
            {
                "anchor": anchor,
                "seed": TRAIN_SEED,
                "target_steps": target,
                "source_checkpoint": str(source),
                "reason": "leadership two-GPU cap pause; preserved before recovery continuation",
                "updated_at": now(),
            },
        )
        append_log(f"preserved partial anchor={anchor} step={step} root={preserved}")
    return source


def launch_training(anchor: int, target: int, source: Path | None, gpu: int, lock: Path) -> Job:
    output = canonical_root(anchor, target)
    if output.exists() and any(output.iterdir()):
        fail(f"launch_anchor_{anchor}", f"refusing nonempty training output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "run/checkpoints" / f"global_step_{target}"
    config_path = RL / "examples/sft/config/open_drawer_retrieve_place_dagger_sft_openpi_pi05.yaml"
    if not config_path.is_file():
        fail(f"launch_anchor_{anchor}", f"missing canonical config: {config_path}")
    prompt = "default_prompt: open the drawer, retrieve the blue object, and place it in the green tray"
    if prompt not in config_path.read_text(encoding="utf-8"):
        fail(f"launch_anchor_{anchor}", f"canonical prompt missing in {config_path}")
    scratch_name = f"a{anchor}s{target}"
    ray_tmp = RAY_TMP_ROOT / scratch_name
    tmp_root = TMP_ROOT / scratch_name
    ray_tmp.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)
    log_path = RUN / "logs" / f"recovery_train_anchor_{anchor}_steps_{target}.log"
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "ASK4HELP_RLINF_PLACEMENT": f"{gpu}-{gpu}",
            "RLINF_RAY_ADDRESS": "local",
            "EMBODIED_PATH": str(RL / "examples/sft"),
            "PYTHONPATH": f"{ROOT / 'tools' / 'adaptive_ray_shim'}:{ROOT}:{RL}",
            "ASK4HELP_RAY_OBJECT_STORE_MEMORY": str(100 * 1024**3),
            "OPEN_DRAWER_ID_DATASET": str(ID_DATASET),
            "OPEN_DRAWER_EXPERT_DATASET": str(BUDGET_ROOT / f"anchor_{anchor}"),
            "OPEN_DRAWER_ID_NORM_STATS": str(NORM),
            "OPEN_DRAWER_PI05_MODEL_PATH": str(MODEL),
            "OPEN_DRAWER_RUN_ROOT": str(output),
            "OPEN_DRAWER_EXPERIMENT_NAME": "run",
            "OPEN_DRAWER_TRAIN_SEED": str(TRAIN_SEED),
            "OPEN_DRAWER_RESUME_DIR": str(source) if source is not None else "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_DATASETS_CACHE": str(RUN / "runtime/hf_datasets"),
            "HF_HOME": str(RUN / "runtime/hf_home"),
            "RAY_TMPDIR": str(ray_tmp),
            "TMPDIR": str(tmp_root),
            "PYTHONUNBUFFERED": "1",
        }
    )
    resume_arg = str(source) if source is not None else "null"
    command = [
        str(PYTHON),
        str(RL / "examples/sft/train_vla_sft.py"),
        "--config-path",
        str(RL / "examples/sft/config"),
        "--config-name",
        "open_drawer_retrieve_place_dagger_sft_openpi_pi05",
        f"runner.max_steps={target}",
        "runner.save_interval=500",
        f"actor.optim.total_training_steps={target}",
        f"runner.resume_dir={resume_arg}",
    ]
    stream = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        ["taskset", "-c", cpu_set(gpu), *command],
        cwd=str(RL / "examples/sft"),
        env=env,
        stdout=stream,
        stderr=subprocess.STDOUT,
        text=True,
    )
    append_log(
        f"launched anchor={anchor} target={target} pid={process.pid} gpu={gpu} source={source or 'base'}"
    )
    return Job(anchor, target, source or MODEL, output, checkpoint, gpu, lock, process, stream)


def process_job(job: Job, held: set[int], probe_index: int) -> tuple[float, int]:
    code = job.process.wait()
    job.stream.close()
    if code != 0:
        fail(f"training_anchor_{job.anchor}_steps_{job.target}", f"returncode={code}")
    checkpoint = audit_training_root(job.anchor, job.target)
    mark_segment(job.anchor, job.target, checkpoint, job.source)
    try:
        rate = run_probe(job.anchor, job.target, checkpoint, job.gpu, probe_index)
    finally:
        release_gpu(job.gpu, held)
    write_recovery_state(
        f"ood20_anchor_{job.anchor}_complete",
        "running",
        f"strict_rate={rate:.6f}",
    )
    return rate, probe_index + 1


def process_adopted(
    anchor: int,
    target: int,
    gpu: int,
    pid: int,
    held: set[int],
    probe_index: int,
) -> tuple[float, int]:
    while pid_alive(pid):
        write_recovery_state(
            f"waiting_adopted_anchor_{anchor}",
            "running",
            f"pid={pid} gpu={gpu}; next check in {CHECK_SECONDS}s",
        )
        write_main_state(
            "two_gpu_cap_recovery_wait",
            "running",
            f"adopted anchor={anchor} pid={pid}; long-interval check",
            retained_anchors=[0, 50],
            paused_anchors=[80, 120],
        )
        time.sleep(CHECK_SECONDS)
    append_log(f"adopted process exited anchor={anchor} pid={pid}; waiting for GPU cleanup")
    held.discard(gpu)
    while not gpu_is_idle(gpu):
        write_recovery_state(
            f"waiting_adopted_anchor_{anchor}_gpu_cleanup",
            "waiting",
            f"gpu={gpu}; next check in {GPU_WAIT_SECONDS}s",
        )
        time.sleep(GPU_WAIT_SECONDS)
    checkpoint = audit_training_root(anchor, target)
    marker_source = canonical_root(anchor, target)
    mark_segment(anchor, target, checkpoint, marker_source)
    probe_gpu_lock = acquire_gpu(held)
    while probe_gpu_lock is None:
        write_recovery_state(
            f"waiting_ood20_anchor_{anchor}",
            "waiting",
            f"no audited idle GPU; next check in {GPU_WAIT_SECONDS}s",
        )
        time.sleep(GPU_WAIT_SECONDS)
        probe_gpu_lock = acquire_gpu(held)
    probe_gpu, _ = probe_gpu_lock
    try:
        rate = run_probe(anchor, target, checkpoint, probe_gpu, probe_index)
    finally:
        release_gpu(probe_gpu, held)
    append_log(f"adopted anchor={anchor} target={target} audited rate={rate:.4f}")
    return rate, probe_index + 1


def finish_adopted_job(
    anchor: int,
    target: int,
    gpu: int,
    held: set[int],
    probe_index: int,
) -> tuple[float, int]:
    """Audit one retained job after its PID has exited, without waiting on the other."""

    held.discard(gpu)
    while not gpu_is_idle(gpu):
        write_recovery_state(
            f"waiting_adopted_anchor_{anchor}_gpu_cleanup",
            "waiting",
            f"gpu={gpu}; next check in {GPU_WAIT_SECONDS}s",
        )
        time.sleep(GPU_WAIT_SECONDS)
    checkpoint = audit_training_root(anchor, target)
    marker_source = canonical_root(anchor, target)
    mark_segment(anchor, target, checkpoint, marker_source)
    probe_gpu_lock = acquire_gpu(held)
    while probe_gpu_lock is None:
        write_recovery_state(
            f"waiting_ood20_anchor_{anchor}",
            "waiting",
            f"no audited idle GPU; next check in {GPU_WAIT_SECONDS}s",
        )
        time.sleep(GPU_WAIT_SECONDS)
        probe_gpu_lock = acquire_gpu(held)
    probe_gpu, _ = probe_gpu_lock
    try:
        rate = run_probe(anchor, target, checkpoint, probe_gpu, probe_index)
    finally:
        release_gpu(probe_gpu, held)
    append_log(f"adopted anchor={anchor} target={target} audited rate={rate:.4f}")
    return rate, probe_index + 1


def freeze_steps(rate: float, steps: int) -> None:
    atomic_json(
        FROZEN_STEPS,
        {
            "frozen_steps": steps,
            "pilot_anchor": 0,
            "training_seed": TRAIN_SEED,
            "ood_success_rate": rate,
            "threshold": OOD_THRESHOLD,
            "rule": "first strict OOD20 rate above threshold",
            "parallel_initial_training": True,
            "recovery_controller": "direct_oracle_adaptive_recovery_v1",
        },
    )


def enqueue_pending(
    pending: deque[tuple[int, int, Path | None]],
    active: dict[int, Job],
    completed: dict[int, int],
    anchor: int,
    target: int,
    source: Path | None,
) -> None:
    if completed.get(anchor, 0) >= target or anchor in active:
        return
    if any(item[0] == anchor for item in pending):
        return
    pending.append((anchor, target, source))


def refresh_pending(
    pending: deque[tuple[int, int, Path | None]],
    active: dict[int, Job],
    adopted: dict[int, dict[str, int]],
    completed: dict[int, int],
    rates: dict[int, float],
    frozen: int | None,
) -> None:
    """Queue only protocol-approved work that is not already busy."""

    busy = set(active) | set(adopted) | {item[0] for item in pending}
    if frozen is None:
        # The pilot is the only model allowed to determine the global frozen
        # step count.  A below-threshold result queues only its own +2500
        # continuation; other anchors wait for the decision.
        pilot_steps = completed.get(0, 0)
        if (
            pilot_steps >= MIN_STEPS
            and rates.get(0, 0.0) <= OOD_THRESHOLD
            and 0 not in busy
        ):
            source = canonical_root(0, pilot_steps) / "run/checkpoints" / f"global_step_{pilot_steps}"
            enqueue_pending(pending, active, completed, 0, pilot_steps + INCREMENT, source)
        return
    for anchor in ANCHORS:
        if anchor in busy or completed.get(anchor, 0) >= frozen:
            continue
        source = prepare_resume_source(anchor, frozen)
        enqueue_pending(pending, active, completed, anchor, frozen, source)


def start_pending_jobs(
    pending: deque[tuple[int, int, Path | None]],
    active: dict[int, Job],
    adopted: dict[int, dict[str, int]],
    held: set[int],
) -> bool:
    """Fill the two-card cap without counting the adopted jobs twice."""

    started = False
    while pending and len(active) + len(adopted) < MAX_WORKERS:
        acquired = acquire_gpu(held)
        if acquired is None:
            write_recovery_state(
                "waiting_training_gpu",
                "waiting",
                f"pending={[(a, t) for a, t, _ in pending]}; next check in {GPU_WAIT_SECONDS}s",
            )
            break
        gpu, lock = acquired
        anchor, target, source = pending.popleft()
        try:
            if source is None:
                source = prepare_resume_source(anchor, target)
            job = launch_training(anchor, target, source, gpu, lock)
        except Exception:
            release_gpu(gpu, held)
            raise
        active[anchor] = job
        persist_jobs(active, adopted)
        write_recovery_state(
            "adaptive_training",
            "running",
            f"anchor={anchor} target={target} gpu={gpu}; adopted={list(adopted)}",
        )
        write_main_state(
            "adaptive_training",
            "running",
            f"anchor={anchor} target={target} gpu={gpu}; recovery retry remains protocol-preserving",
            retained_anchors=[0, 50],
            paused_anchors=[80, 120],
            recovery_retry=2,
            scientific_variables_unchanged=True,
        )
        started = True
    return started


def persist_jobs(active: dict[int, Job], adopted: dict[int, dict[str, int]]) -> None:
    payload: dict[str, Any] = {
        "updated_at": now(),
        "active": {
            str(anchor): {
                "pid": job.process.pid,
                "target": job.target,
                "gpu": job.gpu,
                "output": str(job.output),
                "checkpoint": str(job.checkpoint),
            }
            for anchor, job in active.items()
        },
        "adopted": {str(anchor): value for anchor, value in adopted.items()},
    }
    atomic_json(JOB_STATE, payload)


def run_formal_evaluation(frozen: int) -> None:
    training_marker = RUN / "ADAPTIVE_TIMING_TRAINING_COMPLETE"
    if not training_marker.is_file():
        training_marker.write_text(
            f"all six anchors trained with seed={TRAIN_SEED}, frozen_steps={frozen}, max_workers={MAX_WORKERS}\n",
            encoding="utf-8",
        )
    eval_script = ROOT / "tools/run_open_drawer_adaptive_formal_eval_controller.sh"
    if not eval_script.is_file():
        fail("formal_evaluation", f"missing evaluator controller: {eval_script}")
    write_total_state("formal_evaluation", "running", f"frozen_steps={frozen}")
    write_recovery_state("formal_evaluation", "running", f"frozen_steps={frozen}")
    env = os.environ.copy()
    env.update(
        {
            "OPEN_DRAWER_ROOT": str(ROOT),
            "OPEN_DRAWER_RLINF_ROOT": str(RL),
            "OPEN_DRAWER_PYTHON": str(PYTHON),
            "OPEN_DRAWER_TIMING_ROOT": str(RUN),
            "OPEN_DRAWER_TIMING_CHECKPOINT": str(MODEL),
            "OPEN_DRAWER_TIMING_PI05_BASE": str(PI05_BASE),
            "OPEN_DRAWER_TIMING_NORM": str(NORM),
            "OPEN_DRAWER_TIMING_FORMAL_ROOT": str(FORMAL_ROOT),
            "OPEN_DRAWER_TIMING_BUDGET_ROOT": str(BUDGET_ROOT),
            "OPEN_DRAWER_TIMING_GPU_POOL": " ".join(str(gpu) for gpu in GPU_POOL),
            "OPEN_DRAWER_TIMING_FORMAL_POLL_SECONDS": str(GPU_WAIT_SECONDS),
            "OPEN_DRAWER_TIMING_TRAIN_SEED": str(TRAIN_SEED),
            "OPEN_DRAWER_TIMING_EVALUATOR": str(ROOT / "tools/evaluate_open_drawer_id_pi05.py"),
            "OPEN_DRAWER_TIMING_EVAL_WRAPPER": str(ROOT / "tools/run_open_drawer_timing_eval.py"),
            "OPEN_DRAWER_TIMING_RECONCILER": str(ROOT / "tools/summarize_open_drawer_adaptive_timing.py"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    log_path = RUN / "logs/direct_oracle_adaptive_formal_eval_recovery.log"
    with log_path.open("a", encoding="utf-8") as stream:
        result = subprocess.run(["bash", str(eval_script)], env=env, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        fail("formal_evaluation", f"controller returncode={result.returncode}")
    if not (RUN / "INDEPENDENT_RECONCILIATION_COMPLETE").is_file():
        fail("independent_reconciliation", "missing INDEPENDENT_RECONCILIATION_COMPLETE")
    if not (RUN / "final_report.json").is_file() or not (RUN / "final_report.md").is_file():
        fail("independent_reconciliation", "missing final_report.json or final_report.md")
    (RUN / "PIPELINE_COMPLETE").write_text(
        "direct Oracle adaptive timing pipeline complete after independent reconciliation\n",
        encoding="utf-8",
    )
    write_recovery_state("pipeline_complete", "complete", f"frozen_steps={frozen}")
    write_main_state("pipeline_complete", "complete", f"frozen_steps={frozen}")
    write_total_state("pipeline_complete", "complete", f"frozen_steps={frozen}")
    append_log(f"PIPELINE_COMPLETE frozen_steps={frozen}")


def acquire_controller_lock() -> None:
    try:
        LOCK_ROOT.mkdir(parents=True)
    except FileExistsError:
        owner = pid_for_lock(LOCK_ROOT)
        if owner and pid_alive(owner):
            fail("preflight", f"another recovery controller is live: pid={owner}")
        shutil.rmtree(LOCK_ROOT, ignore_errors=True)
        LOCK_ROOT.mkdir(parents=True)
    (LOCK_ROOT / "owner").write_text(f"{os.getpid()}\n", encoding="utf-8")


def main() -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "logs").mkdir(parents=True, exist_ok=True)
    acquire_controller_lock()
    retained = load_retained_jobs()
    if not MODEL:
        fail("preflight", "OPEN_DRAWER_TIMING_CHECKPOINT is required")
    for path in (
        MODEL / "actor/model_state_dict/full_weights.pt",
        FORMAL_ROOT / "AUDIT_PASS",
        BUDGET_ROOT / "BUDGET_AUDIT_PASS",
    ):
        if not path.exists():
            fail("preflight", f"missing required immutable artifact: {path}")
    if not (FORMAL_ROOT / "TIMING_COLLECTION_COMPLETE").exists() and not (FORMAL_ROOT.parent / "TIMING_COLLECTION_COMPLETE").exists():
        fail("preflight", "missing direct-grasp formal collection marker")
    RAY_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    write_total_state(
        "two_gpu_cap_recovery",
        "running",
        "adopting retained anchor 0/50 processes; no duplicate launch",
        retained_anchors=[0, 50],
        paused_anchors=[80, 120],
        scientific_variables_unchanged=True,
    )
    write_main_state(
        "direct_oracle_recovery_retry_training",
        "running",
        "retrying from preserved periodic checkpoints; no scientific variable changes",
        retained_anchors=[0, 50],
        paused_anchors=[80, 120],
        recovery_retry=2,
        scientific_variables_unchanged=True,
    )
    write_recovery_state(
        "preflight",
        "running",
        f"retained={retained}; check={CHECK_SECONDS}s stable={STABLE_CHECK_SECONDS}s",
    )
    append_log(f"started recovery controller pid={os.getpid()} retained={retained}")

    held: set[int] = {item["gpu"] for item in retained.values() if pid_alive(item["pid"])}
    adopted: dict[int, dict[str, int]] = {}
    for anchor, item in retained.items():
        verify_retained_process(anchor, item["pid"])
        if pid_alive(item["pid"]):
            adopted[anchor] = item
    persist_jobs({}, adopted)
    probe_counter = len(list((RUN / "ood20_probe").glob("anchor_*/seed_*/steps_*/OOD20_COMPLETE")))
    completed: dict[int, int] = {}
    rates: dict[int, float] = {}
    frozen: int | None = None
    if FROZEN_STEPS.is_file():
        try:
            frozen = int(json.loads(FROZEN_STEPS.read_text(encoding="utf-8"))["frozen_steps"])
        except (OSError, ValueError, KeyError) as exc:
            fail("preflight", f"invalid adaptive_steps.json: {exc}")
    # A previous recovery controller may already have written canonical
    # segment markers.  Adopt those artifacts instead of scheduling duplicate
    # training; the paired OOD20 marker is checked below before they can count
    # toward the formal stage.
    if FIXED_5000_OVERRIDE:
        if frozen is not None and frozen != MIN_STEPS:
            fail("fixed5000_preflight", f"existing frozen step count is {frozen}, expected {MIN_STEPS}")
        frozen = MIN_STEPS
        precompleted = existing_fixed_5000_completions()
        fixed_rates: dict[int, float] = {}
        for anchor in precompleted:
            probe_output = RUN / "ood20_probe" / f"anchor_{anchor}" / f"seed_{TRAIN_SEED}" / f"steps_{MIN_STEPS}" / "grasp_ood"
            try:
                fixed_rates[anchor] = audit_probe(probe_output)
            except Exception as exc:
                fail(f"fixed5000_anchor_{anchor}", f"existing OOD20 audit failed: {exc}")
        atomic_json(
            FROZEN_STEPS,
            {
                "frozen_steps": MIN_STEPS,
                "pilot_anchor": 0,
                "training_seed": TRAIN_SEED,
                "threshold": OOD_THRESHOLD,
                "rule": "fixed cumulative 5000 per user override",
                "override": "fixed_5000_override_20260901",
                "observed_ood20_rates": fixed_rates,
            },
        )
    else:
        precompleted = existing_marked_completions()
    for anchor, target in precompleted.items():
        completed[anchor] = target
        probe_output = RUN / "ood20_probe" / f"anchor_{anchor}" / f"seed_{TRAIN_SEED}" / f"steps_{target}" / "grasp_ood"
        try:
            rates[anchor] = audit_probe(probe_output)
        except Exception as exc:
            fail(f"preflight_anchor_{anchor}", f"existing OOD20 audit failed: {exc}")
    pending: deque[tuple[int, int, Path | None]] = deque()
    active: dict[int, Job] = {}

    # The leadership-cap pause left anchor 0/50 processes orphaned.  If a
    # retained PID has since disappeared before its final checkpoint, resume
    # from its newest complete periodic checkpoint instead of looping forever
    # or starting from the base model.  The exact target remains the frozen
    # protocol target (5000 until the pilot freezes a later count), and the
    # old partial root is preserved by prepare_resume_source().
    retained_target = frozen if frozen is not None else MIN_STEPS
    for anchor in (0, 50):
        if anchor in adopted or anchor in completed:
            continue
        if pid_alive(retained[anchor]["pid"]):
            continue
        source = prepare_resume_source(anchor, retained_target)
        if source is None:
            fail(
                "recover_retained_partial",
                f"retained anchor={anchor} pid={retained[anchor]['pid']} is gone and no complete checkpoint exists below target={retained_target}",
            )
        enqueue_pending(
            pending,
            active,
            completed,
            anchor,
            retained_target,
            source,
        )
        append_log(
            f"queued retained recovery anchor={anchor} target={retained_target} source={source}"
        )

    # Unified event loop: retained jobs are monitored alongside newly launched
    # jobs.  Thus a freed retained GPU can immediately run the next approved
    # training segment while the other retained job is still in flight.
    while True:
        progressed = start_pending_jobs(pending, active, adopted, held)

        for anchor, item in list(adopted.items()):
            if pid_alive(item["pid"]):
                continue
            rate, probe_counter = finish_adopted_job(
                anchor, MIN_STEPS, item["gpu"], held, probe_counter
            )
            completed[anchor] = MIN_STEPS
            rates[anchor] = rate
            adopted.pop(anchor, None)
            progressed = True
            append_log(f"adopted completion anchor={anchor} target={MIN_STEPS} rate={rate:.4f}")
            if anchor == 0 and frozen is None and rate > OOD_THRESHOLD:
                frozen = MIN_STEPS
                freeze_steps(rate, frozen)

        for anchor, job in list(active.items()):
            if job.process.poll() is None:
                continue
            rate, probe_counter = process_job(job, held, probe_counter)
            completed[anchor] = job.target
            rates[anchor] = rate
            del active[anchor]
            progressed = True
            if anchor == 0 and frozen is None and rate > OOD_THRESHOLD:
                frozen = job.target
                freeze_steps(rate, frozen)
                append_log(f"frozen_steps={frozen} pilot_rate={rate:.4f}")

        refresh_pending(pending, active, adopted, completed, rates, frozen)
        if start_pending_jobs(pending, active, adopted, held):
            progressed = True
        persist_jobs(active, adopted)

        if (
            frozen is not None
            and all(completed.get(anchor, 0) >= frozen for anchor in ANCHORS)
            and not pending
            and not active
            and not adopted
        ):
            break
        if not progressed:
            detail = (
                f"pending={[(a, t) for a, t, _ in pending]} "
                f"active={list(active)} adopted={list(adopted)} "
                f"completed={completed} frozen={frozen}"
            )
            delay = CHECK_SECONDS if (active or adopted or pending) else STABLE_CHECK_SECONDS
            write_recovery_state("adaptive_training_wait", "running", f"{detail}; next check in {delay}s")
            time.sleep(delay)

    if frozen is None:
        fail("pilot", "no first strict OOD20 rate above threshold")
    missing = [anchor for anchor in ANCHORS if completed.get(anchor) != frozen]
    if missing:
        fail("adaptive_training", f"missing frozen-step anchors: {missing}")
    write_recovery_state("all_adaptive_training", "complete", f"frozen_steps={frozen}", completed_steps=completed, rates=rates)
    write_main_state("all_adaptive_training", "complete", f"frozen_steps={frozen}", completed_steps=completed)
    run_formal_evaluation(frozen)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        append_log("received SIGINT; leaving independent training processes untouched")
        raise
    except SystemExit:
        raise
    except Exception as exc:
        fail("uncaught_exception", repr(exc))
