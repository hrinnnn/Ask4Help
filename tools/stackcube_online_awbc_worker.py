#!/usr/bin/env python3
"""Resident StackCube online AWBC worker.

The process deliberately owns both pi0.5 members for its complete lifetime.
Commands are served over a local Unix socket so a shell launcher can drive the
round without ever reloading a policy between calibration, collection and
in-place updates.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
import socketserver
import sys
import time
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RLINF_ROOT = ROOT / "RLinf"
sys.path[:0] = [str(ROOT), str(RLINF_ROOT)]

from rlinf.algorithms.online_awbc import (  # noqa: E402
    FixedThresholdChunkController,
    FixedVFDThreshold,
    first_vfd_action_candidate,
    uniformly_spaced_chunk_indices,
)
from rlinf.algorithms.online_awbc_buffer import (  # noqa: E402
    OnlineChunk,
    adaptive_update_steps,
    persistent_flux_weight,
    quality_online_chunks,
    sample_active_batch,
)
from rlinf.data.online_awbc import build_online_awbc_manifest  # noqa: E402

from tools.maniskill_pi05_vfd_online_awbc import (  # noqa: E402
    _action_chunk,
    _build_env,
    _create_dataset,
    _extract_record,
    _load_model,
    _select_camera,
    _task_label,
    _wrap_obs,
    _write_jsonl,
    _build_frames,
    MAIN_CAMERA_CANDIDATES,
    WRIST_CAMERA_CANDIDATES,
)
from rlinf.data.online_awbc import OnlineAWBCChunk, OnlineAWBCFrame  # noqa: E402
from rlinf.data.maniskill_stack_cube_progress import StackCubeProgressState  # noqa: E402
from rlinf.envs.maniskill.stack_cube_privileged_oracle import (  # noqa: E402
    StackCubePrivilegedChunkOracle,
)
from rlinf.envs.maniskill.stack_cube_variants import STACK_CUBE_TASK  # noqa: E402
from toolkits.lerobot.collect_maniskill_plug_lerobot_joint import (  # noqa: E402
    write_episode_video_durably,
)


HORIZON = 10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _bool(value: Any) -> bool:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return bool(np.asarray(value, dtype=bool).reshape(-1).any())


@dataclass
class ResidentMember:
    name: str
    model: Any
    optimizer: torch.optim.Optimizer
    seed: int
    version: int = 7000


class _AnchorStore:
    """Lazy LeRobot anchor reader that never joins observations across episodes."""

    def __init__(self, dataset_path: Path):
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        self.dataset = LeRobotDataset(str(dataset_path))
        self.positions_by_episode: dict[int, list[int]] = {}
        for position, episode_id in enumerate(self.dataset.hf_dataset["episode_index"]):
            self.positions_by_episode.setdefault(int(episode_id), []).append(position)
        self.lengths = {episode: len(rows) for episode, rows in self.positions_by_episode.items()}

    @staticmethod
    def _action(item: dict[str, Any]) -> torch.Tensor:
        value = item.get("action", item.get("actions"))
        if value is None:
            raise KeyError("LeRobot item needs action or actions")
        return torch.as_tensor(value, dtype=torch.float32).reshape(-1)

    def batch_for_anchors(self, anchors: Iterable[tuple[int, int]]) -> dict[str, Any]:
        images: list[torch.Tensor] = []
        wrists: list[torch.Tensor] = []
        states: list[torch.Tensor] = []
        actions: list[torch.Tensor] = []
        tasks: list[str] = []
        for episode_id, local_start in anchors:
            positions = self.positions_by_episode[int(episode_id)]
            if local_start < 0 or local_start + HORIZON > len(positions):
                raise ValueError("anchor crosses an episode boundary")
            first = self.dataset[positions[local_start]]
            target = torch.cat(
                [self._action(self.dataset[positions[local_start + offset]]) for offset in range(HORIZON)]
            )
            images.append(torch.as_tensor(first["image"]))
            wrists.append(torch.as_tensor(first.get("wrist_image", first["image"])))
            states.append(torch.as_tensor(first["state"]))
            actions.append(target)
            tasks.append(str(first.get("task", STACK_CUBE_TASK)))
        return {
            "image": torch.stack(images),
            "wrist_image": torch.stack(wrists),
            "state": torch.stack(states),
            "actions": torch.stack(actions),
            "task": tasks,
        }


class StackCubeOnlineWorker:
    """Own two resident policies and execute exactly one online AWBC round."""

    def __init__(
        self,
        *,
        member0_checkpoint: Path,
        member1_checkpoint: Path,
        pi05_base: Path,
        norm_stats: Path,
        expert_dataset: Path,
        output_dir: Path,
        seed0: int = 1000,
        seed1: int = 1001,
        device: str = "cuda:0",
        microbatch_size: int = 1,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("resident pi0.5 worker requires CUDA")
        self.device = torch.device(device)
        if self.device.index not in {None, 0}:
            raise ValueError("worker must see its dedicated GPU as cuda:0")
        self.output_dir = output_dir.expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pi05_base = pi05_base.expanduser().resolve()
        self.norm_stats = norm_stats.expanduser().resolve()
        self.expert_store = _AnchorStore(expert_dataset.expanduser().resolve())
        self.online_store: _AnchorStore | None = None
        self.microbatch_size = int(microbatch_size)
        if self.microbatch_size != 1:
            raise ValueError("first resident smoke fixes microbatch_size=1 for exact accounting")
        self.round_id = "round_0001"
        self.threshold: FixedVFDThreshold | None = None
        self.quality_chunks: tuple[OnlineChunk, ...] = ()
        self.raw_online_dataset: Path | None = None
        self._last_collection: dict[str, Any] | None = None
        self._closed = False

        torch.cuda.set_device(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        member0 = _load_model(member0_checkpoint, self.norm_stats, self.pi05_base)
        member1 = _load_model(member1_checkpoint, self.norm_stats, self.pi05_base)
        for model in (member0, member1):
            model.to(self.device)
            model.eval()
            model.requires_grad_(True)
        self.members = {
            "member0": ResidentMember("member0", member0, torch.optim.AdamW(member0.parameters(), lr=1e-5), seed0),
            "member1": ResidentMember("member1", member1, torch.optim.AdamW(member1.parameters(), lr=1e-5), seed1),
        }
        # A residency preflight deliberately exercises a real optimizer step,
        # but that probe must not turn the policy used for calibration into an
        # unrecorded post-update policy. Keep CPU snapshots only for restoring
        # the just-probed parameters; normal online updates remain in-place.
        self._preflight_states = {
            name: {key: value.detach().cpu().clone() for key, value in member.model.state_dict().items()}
            for name, member in self.members.items()
        }
        self._write_status("initialized")

    def _gpu_stats(self) -> dict[str, int]:
        return {
            "allocated_bytes": int(torch.cuda.memory_allocated(self.device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(self.device)),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(self.device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(self.device)),
        }

    def _write_status(self, phase: str) -> dict[str, Any]:
        payload = {
            "round_id": self.round_id,
            "phase": phase,
            "members": {name: {"version": item.version, "seed": item.seed} for name, item in self.members.items()},
            "quality_online_count": len(self.quality_chunks),
            "raw_online_dataset": None if self.raw_online_dataset is None else str(self.raw_online_dataset),
            "threshold": None if self.threshold is None else self.threshold.__dict__,
            "gpu0": self._gpu_stats(),
            "timestamp": time.time(),
        }
        _json_dump(self.output_dir / "worker_status.json", payload)
        return payload

    def status(self) -> dict[str, Any]:
        return self._write_status("status")

    def _per_sample_flow_loss(self, member: ResidentMember, batch: dict[str, Any]) -> torch.Tensor:
        prepared = member.model.prepare_lerobot_sft_batch(batch)
        # OpenPi0ForRLActionPrediction.forward is an RL dispatcher. Call its
        # PI0Pytorch parent directly to retain the unreduced flow-loss tensor.
        loss = super(type(member.model), member.model).forward(
            prepared["observation"], prepared["actions"]
        )
        loss = loss[:, : member.model.config.action_chunk, : member.model.config.action_env_dim]
        return loss.reshape(loss.shape[0], -1).mean(dim=1)

    def preflight(self) -> dict[str, Any]:
        """Hard residency gate: two eval forwards then one backward per member."""
        first_episode = min(self.expert_store.lengths)
        if self.expert_store.lengths[first_episode] < HORIZON:
            raise RuntimeError("expert dataset has no full 10-step anchor for preflight")
        batch = self.expert_store.batch_for_anchors([(first_episode, 0)])
        results: dict[str, Any] = {"before": self._gpu_stats()}
        try:
            with torch.inference_mode():
                for name, member in self.members.items():
                    member.model.eval()
                    self._per_sample_flow_loss(member, batch)
                    results[f"{name}_eval"] = self._gpu_stats()
            for name, member in self.members.items():
                member.optimizer.zero_grad(set_to_none=True)
                member.model.train()
                self._per_sample_flow_loss(member, batch).mean().backward()
                member.optimizer.step()
                member.optimizer.zero_grad(set_to_none=True)
                # The optimizer step is a capacity check, not part of online
                # learning. Restore the original resident parameters and
                # discard the temporary optimizer state before proceeding.
                member.model.load_state_dict(self._preflight_states[name])
                member.optimizer.state.clear()
                member.model.eval()
                results[f"{name}_backward"] = self._gpu_stats()
        except torch.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            results["oom"] = str(error)
            _json_dump(self.output_dir / "resident_preflight.json", results)
            raise RuntimeError("resident dual-model preflight OOM; no reload fallback was used") from error
        results["passed"] = True
        _json_dump(self.output_dir / "resident_preflight.json", results)
        self._write_status("preflight_passed")
        return results

    def calibrate(
        self,
        *,
        seeds: list[int],
        successes: int = 5,
        samples_per_episode: int | None = None,
        min_scores: int = 100,
        quantile: float = 0.95,
    ) -> dict[str, Any]:
        if successes <= 0:
            raise ValueError("successes must be positive")
        env = _build_env(100, task="stack", split="id")
        scores: list[float] = []
        accepted: list[int] = []
        traces: list[dict[str, Any]] = []
        try:
            for seed in seeds:
                raw_obs, info = env.reset(seed=seed)
                progress = StackCubeProgressState()
                episode_scores: list[float] = []
                timeline: list[dict[str, Any]] = []
                records = [_extract_record(raw_obs)]
                actions: list[np.ndarray] = []
                done = False
                step = 0
                while not done and step < 100:
                    start = len(actions)
                    obs = _wrap_obs(raw_obs, info, task="stack")
                    generator = torch.Generator(device=self.device).manual_seed(seed + step)
                    with torch.inference_mode():
                        candidates, vfd = self.members["member0"].model.compute_vfd_uncertainty(
                            obs, self.members["member1"].model, num_action_samples=5, generator=generator
                        )
                    score = float(vfd.reshape(-1)[0].cpu())
                    chunk_actions = _action_chunk(first_vfd_action_candidate(candidates), HORIZON)
                    for action in chunk_actions:
                        raw_obs, _reward, terminated, truncated, info = env.step(
                            torch.as_tensor(action, device=env.unwrapped.device).unsqueeze(0)
                        )
                        actions.append(np.asarray(action, dtype=np.float32))
                        step += 1
                        progress.update(env, info)
                        records.append(_extract_record(raw_obs))
                        done = _bool(terminated) or _bool(truncated) or _bool(info.get("success", False))
                        if done:
                            break
                    executed = step - start
                    if executed == HORIZON:
                        episode_scores.append(score)
                        timeline.append(
                            {
                                "chunk_index": len(timeline),
                                "frame_index": start,
                                "next_frame_index": step,
                                "vfd": score,
                                "controller": "policy",
                            }
                        )
                if _bool(info.get("success", False)):
                    video_frames = _build_frames(
                        records=records,
                        actions=actions,
                        task=_task_label("stack"),
                        main_camera=_select_camera(records[0].obs, "", MAIN_CAMERA_CANDIDATES, "main"),
                        wrist_camera=_select_camera(records[0].obs, "", WRIST_CAMERA_CANDIDATES, "wrist"),
                    )
                    video_path = write_episode_video_durably(
                        video_frames,
                        video_dir=self.output_dir / "calibration" / "id_vfd_videos",
                        episode_index=len(traces),
                        seed=seed,
                        fps=10,
                    )
                    traces.append({"seed": seed, "success": True, "timeline": timeline, "video": str(video_path)})
                    if samples_per_episode is None:
                        scores.extend(episode_scores)
                    else:
                        scores.extend(
                            episode_scores[index]
                            for index in uniformly_spaced_chunk_indices(
                                len(episode_scores), samples_per_episode=samples_per_episode
                            )
                        )
                    accepted.append(seed)
                    if len(accepted) >= successes and len(scores) >= min_scores:
                        break
        finally:
            env.close()
        if len(accepted) < successes or len(scores) < min_scores:
            raise RuntimeError(
                f"VFD calibration found {len(accepted)} successful ID trajectories and {len(scores)}/{min_scores} scores"
            )
        self.threshold = FixedVFDThreshold.calibrate(scores, quantile=quantile)
        payload = {**self.threshold.__dict__, "seeds": accepted, "scores": scores, "sha256": hashlib.sha256(json.dumps(scores).encode()).hexdigest()}
        _json_dump(self.output_dir / "calibration" / "fixed_vfd_threshold.json", payload)
        _json_dump(self.output_dir / "calibration" / "id_vfd_trajectories.json", traces)
        self._write_status("calibrated")
        return payload

    def _collect_episode(self, env: Any, *, seed: int, episode_index: int, offset: int) -> tuple[list[dict[str, Any]], list[OnlineAWBCFrame], list[OnlineAWBCChunk], dict[str, Any]]:
        if self.threshold is None:
            raise RuntimeError("collect requires a calibrated VFD threshold")
        raw_obs, info = env.reset(seed=seed)
        progress = StackCubeProgressState()
        oracle = StackCubePrivilegedChunkOracle(chunk_size=HORIZON)
        controller = FixedThresholdChunkController(self.threshold)
        records = [_extract_record(raw_obs)]
        actions: list[np.ndarray] = []
        frames: list[OnlineAWBCFrame] = []
        chunks: list[OnlineAWBCChunk] = []
        timeline: list[dict[str, Any]] = []
        main_camera = _select_camera(records[0].obs, "", MAIN_CAMERA_CANDIDATES, "main")
        wrist_camera = _select_camera(records[0].obs, "", WRIST_CAMERA_CANDIDATES, "wrist")
        terminated = truncated = False
        while not (terminated or truncated) and len(actions) < 100:
            start = len(actions)
            phi = progress.update(env, info)
            obs = _wrap_obs(raw_obs, info, task="stack")
            generator = torch.Generator(device=self.device).manual_seed(seed + start)
            with torch.inference_mode():
                candidates, score = self.members["member0"].model.compute_vfd_uncertainty(
                    obs, self.members["member1"].model, num_action_samples=5, generator=generator
                )
            vfd = float(score.reshape(-1)[0].cpu())
            source = "expert" if controller.decide([vfd]).expert_mask.item() else "policy"
            plan = oracle.plan(env) if source == "expert" else None
            action_sequence = plan.actions if plan is not None else _action_chunk(first_vfd_action_candidate(candidates), HORIZON)
            for local_step, action in enumerate(action_sequence):
                if plan is not None:
                    action = plan.action_at(raw_obs["agent"]["qpos"], local_step)
                frames.append(OnlineAWBCFrame(offset + len(actions), episode_index, len(actions), source, phi))
                raw_obs, _reward, terminated, truncated, info = env.step(
                    torch.as_tensor(action, device=env.unwrapped.device).unsqueeze(0)
                )
                actions.append(np.asarray(action, dtype=np.float32))
                records.append(_extract_record(raw_obs))
                if _bool(terminated) or _bool(truncated) or _bool(info.get("success", False)):
                    break
            next_frame = len(actions)
            phi_next = progress.update(env, info)
            chunks.append(OnlineAWBCChunk(offset + start, episode_index, start, next_frame, source, phi, phi_next, vfd, self.threshold.threshold, _bool(info.get("success", False))))
            timeline.append({"episode_index": episode_index, "chunk_index": len(chunks) - 1, "frame_index": start, "next_frame_index": next_frame, "controller": source, "vfd": vfd, "threshold": self.threshold.threshold})
        video_frames = _build_frames(records=records, actions=actions, task=_task_label("stack"), main_camera=main_camera, wrist_camera=wrist_camera)
        return video_frames, frames, chunks, {"seed": seed, "success": _bool(info.get("success", False)), "timeline": timeline}

    def collect(self, *, seeds: list[int], trajectories: int = 2) -> dict[str, Any]:
        if trajectories != 2:
            raise ValueError("the first smoke intentionally collects exactly two trajectories")
        # A failed trigger gate is still valuable online evidence. Keep each
        # collection attempt immutable instead of overwriting its raw archive.
        collection_id = f"collection_{time.time_ns()}"
        raw_dir = self.output_dir / "raw_online_archive" / collection_id
        # LeRobot materializes local datasets below its repo id.  The worker may
        # be restarted for the same logical round, so the collection id must be
        # unique as well; reusing round_id would collide with an earlier raw
        # archive and prevents a restart from collecting anything.
        repo_id = f"stackcube_online_awbc_{self.round_id}_{collection_id}"
        env = _build_env(100, task="stack", split="ood")
        dataset = None
        all_frames: list[OnlineAWBCFrame] = []
        all_chunks: list[OnlineAWBCChunk] = []
        episodes: list[dict[str, Any]] = []
        assisted_episode_indices: list[int] = []
        try:
            for attempt, seed in enumerate(seeds):
                if len(assisted_episode_indices) == trajectories:
                    break
                video_frames, frames, chunks, episode = self._collect_episode(env, seed=seed, episode_index=len(episodes), offset=len(all_frames))
                if dataset is None:
                    dataset = _create_dataset(repo_id=repo_id, image_shape=tuple(video_frames[0]["image"].shape), wrist_image_shape=tuple(video_frames[0]["wrist_image"].shape), fps=10, image_writer_threads=4, image_writer_processes=4)
                for frame in video_frames:
                    dataset.add_frame(frame)
                dataset.save_episode()
                write_episode_video_durably(video_frames, video_dir=raw_dir / "videos", episode_index=len(episodes), seed=seed, fps=10)
                all_frames.extend(frames)
                all_chunks.extend(chunks)
                episodes.append(episode)
                if any(chunk.source == "expert" for chunk in chunks):
                    assisted_episode_indices.append(len(episodes) - 1)
        finally:
            if dataset is not None and getattr(dataset, "image_writer", None) is not None:
                dataset.image_writer.wait_until_done()
            env.close()
        source_dataset = Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id
        if not source_dataset.is_dir():
            raise FileNotFoundError(f"LeRobot did not create online dataset: {source_dataset}")
        durable_dataset = raw_dir / "dataset"
        if durable_dataset.exists():
            raise FileExistsError(f"refusing to overwrite raw online archive: {durable_dataset}")
        import shutil
        shutil.copytree(source_dataset, durable_dataset)
        manifest = build_online_awbc_manifest(all_frames, all_chunks)
        _write_jsonl(raw_dir / "progress_privileged.jsonl", manifest)
        _write_jsonl(raw_dir / "online_chunks.jsonl", [chunk.diagnostic_dict() for chunk in all_chunks])
        _json_dump(raw_dir / "controller_timeline.json", [entry for episode in episodes for entry in episode["timeline"]])
        _json_dump(raw_dir / "episodes.json", episodes)
        _json_dump(
            raw_dir / "successful_episodes.json",
            {"successful_episodes": [index for index, episode in enumerate(episodes) if episode["success"]]},
        )
        self.raw_online_dataset = durable_dataset
        self.online_store = _AnchorStore(durable_dataset)
        self._last_collection = {"episodes": episodes, "chunks": [chunk.diagnostic_dict() for chunk in all_chunks], "assisted_episode_indices": assisted_episode_indices}
        self._write_status("collected")
        result = {"dataset": str(durable_dataset), "episodes": episodes, "expert_chunks": sum(chunk.source == "expert" for chunk in all_chunks), "assisted_episode_indices": assisted_episode_indices}
        if len(assisted_episode_indices) != trajectories:
            raise RuntimeError(f"collected {len(assisted_episode_indices)}/{trajectories} assisted trajectories after {len(episodes)} OOD attempts; raw archive={durable_dataset}")
        return result

    def admit_quality_buffer(
        self,
        *,
        progress_rows: list[dict[str, Any]] | None = None,
        progress_path: str | None = None,
        minimum_weight: float = 0.1,
    ) -> dict[str, Any]:
        if self._last_collection is None or self.online_store is None:
            raise RuntimeError("progress annotation requires collected online trajectories")
        if progress_rows is None:
            if not progress_path:
                raise ValueError("admit_quality_buffer requires progress_rows or progress_path")
            progress_rows = [
                json.loads(line)
                for line in Path(progress_path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        by_dataset_index = {int(row["dataset_index"]): row for row in progress_rows}
        raw_chunks = self._last_collection["chunks"]
        episode_lengths: dict[int, int] = {}
        for chunk in raw_chunks:
            episode_lengths[int(chunk["episode_index"])] = episode_lengths.get(int(chunk["episode_index"]), 0) + 1
        mean_length = float(np.mean(list(episode_lengths.values())))
        chunks: list[OnlineChunk] = []
        for raw in raw_chunks:
            row = by_dataset_index.get(int(raw["dataset_index"]))
            if row is None:
                raise ValueError(f"GRM progress is missing online dataset_index={raw['dataset_index']}")
            next_frame = int(raw["next_frame_index"])
            full = next_frame - int(raw["frame_index"]) == HORIZON
            valid = bool(row.get("valid", False)) and full
            length = episode_lengths[int(raw["episode_index"])]
            weight = persistent_flux_weight(phi=row.get("phi"), phi_next=row.get("phi_next"), valid=valid, episode_length_chunks=length, mean_episode_length_chunks=mean_length)
            chunks.append(OnlineChunk(episode_id=int(raw["episode_index"]), frame_index=int(raw["frame_index"]), next_frame_index=next_frame, phi=row.get("phi"), phi_next=row.get("phi_next"), valid=valid, episode_length_chunks=length, controller=str(raw["source"]), weight=weight))
        self.quality_chunks = quality_online_chunks(chunks, minimum_weight=minimum_weight)
        payload = [{**chunk.__dict__, "admitted": chunk in self.quality_chunks} for chunk in chunks]
        _write_jsonl(self.output_dir / "quality_online_buffer" / "chunks.jsonl", payload)
        summary = {"raw_chunks": len(chunks), "quality_chunks": len(self.quality_chunks), "minimum_weight": minimum_weight, "mean_episode_length_chunks": mean_length}
        _json_dump(self.output_dir / "quality_online_buffer" / "summary.json", summary)
        self._write_status("quality_admitted")
        return summary

    def _member_update(self, member: ResidentMember, active) -> dict[str, Any]:
        if self.online_store is None:
            raise RuntimeError("online store is unavailable")
        total_denominator = float(len(active.expert) + sum(item.weight for item in active.online))
        if total_denominator <= len(active.expert):
            raise RuntimeError("quality buffer has no positive online loss mass")
        expert_anchors = [(item.episode_id, item.frame_index) for item in active.expert]
        online_anchors = [(item.episode_id, item.frame_index) for item in active.online]
        member.model.train()
        member.optimizer.zero_grad(set_to_none=True)
        numerator_value = 0.0
        for anchor in expert_anchors:
            per_sample = self._per_sample_flow_loss(member, self.expert_store.batch_for_anchors([anchor]))
            numerator_value += float(per_sample.detach().sum())
            (per_sample.sum() / total_denominator).backward()
        for anchor, online_chunk in zip(online_anchors, active.online, strict=True):
            per_sample = self._per_sample_flow_loss(member, self.online_store.batch_for_anchors([anchor]))
            numerator_value += float((per_sample.detach().sum() * online_chunk.weight))
            (per_sample.sum() * online_chunk.weight / total_denominator).backward()
        torch.nn.utils.clip_grad_norm_(member.model.parameters(), 1.0)
        member.optimizer.step()
        member.optimizer.zero_grad(set_to_none=True)
        member.model.eval()
        return {"loss": numerator_value / total_denominator, "numerator": numerator_value, "denominator": total_denominator}

    def snapshot(self, *, member_name: str) -> Path:
        member = self.members[member_name]
        member.version += 1
        directory = self.output_dir / "checkpoints" / f"{member_name}_v{member.version:05d}"
        directory.mkdir(parents=True, exist_ok=False)
        target = directory / "full_weights.pt"
        torch.save(member.model.state_dict(), target)
        (directory / "full_weights.pt.sha256").write_text(_sha256(target) + "\n", encoding="utf-8")
        _json_dump(directory / "metadata.json", {"member": member_name, "version": member.version, "round_id": self.round_id, "gpu0": self._gpu_stats()})
        return directory

    def update(self) -> dict[str, Any]:
        if not self.quality_chunks:
            raise RuntimeError("all online chunks failed quality admission; refusing SFT fallback")
        generator = torch.Generator().manual_seed(20260723)
        active = sample_active_batch(self.quality_chunks, self.expert_store.lengths, generator=generator, max_online_chunks=32, horizon=HORIZON)
        steps = adaptive_update_steps(online_count=len(self.quality_chunks), active_online_count=len(active.online))
        history: dict[str, list[dict[str, Any]]] = {"member0": [], "member1": []}
        for name, member in self.members.items():
            for _ in range(steps):
                history[name].append(self._member_update(member, active))
            checkpoint = self.snapshot(member_name=name)
            history[name][-1]["checkpoint"] = str(checkpoint)
        _json_dump(self.output_dir / "updates" / "update_summary.json", {"steps": steps, "active_batch": active.size, "online_count": len(active.online), "history": history})
        self._write_status("updated")
        return {"steps": steps, "active_batch": active.size, "history": history}

    def forward_smoke(self) -> dict[str, Any]:
        """Compare resident parameters with saved weights and run resident forwards.

        A third GPU model would invalidate the residency/OOM gate, so reload
        equivalence is checked against CPU checkpoint tensors while the actual
        forward continues to execute on the two resident GPU models.
        """
        first_episode = min(self.expert_store.lengths)
        batch = self.expert_store.batch_for_anchors([(first_episode, 0)])
        result: dict[str, Any] = {}
        for name, member in self.members.items():
            checkpoint = self.output_dir / "checkpoints" / f"{name}_v{member.version:05d}" / "full_weights.pt"
            disk = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
            resident = member.model.state_dict()
            mismatched = [key for key, value in disk.items() if key not in resident or not torch.equal(value, resident[key].detach().cpu())]
            if mismatched:
                raise RuntimeError(f"checkpoint/resident mismatch for {name}: {mismatched[:3]}")
            with torch.inference_mode():
                per_sample = self._per_sample_flow_loss(member, batch)
            result[name] = {"checkpoint": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint), "forward_loss": float(per_sample.mean().cpu()), "matched_tensors": len(disk)}
        _json_dump(self.output_dir / "forward_reload_smoke.json", result)
        self._write_status("forward_smoke_passed")
        return result

    def shutdown(self) -> dict[str, Any]:
        self._closed = True
        self._write_status("shutdown")
        return {"shutdown": True}

    def dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        command = str(request.get("command", ""))
        if command == "status":
            return self.status()
        if command == "preflight":
            return self.preflight()
        if command == "calibrate":
            return self.calibrate(**request.get("args", {}))
        if command == "collect":
            return self.collect(**request.get("args", {}))
        if command == "admit_quality_buffer":
            return self.admit_quality_buffer(**request.get("args", {}))
        if command == "update":
            return self.update()
        if command == "snapshot":
            return {"checkpoint": str(self.snapshot(**request.get("args", {})))}
        if command == "forward_smoke":
            return self.forward_smoke()
        if command == "shutdown":
            return self.shutdown()
        raise ValueError(f"unknown worker command: {command}")


class _SocketServer(socketserver.UnixStreamServer):
    allow_reuse_address = True


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        for line in self.rfile:
            try:
                request = json.loads(line)
                result = self.server.worker.dispatch(request)  # type: ignore[attr-defined]
                payload = {"ok": True, "result": result}
            except Exception as error:  # pragma: no cover - service boundary
                payload = {"ok": False, "error": type(error).__name__, "message": str(error)}
            self.wfile.write((json.dumps(payload) + "\n").encode())
            self.wfile.flush()
            if getattr(self.server.worker, "_closed", False):  # type: ignore[attr-defined]
                self.server.shutdown()
                return


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--member0", type=Path, required=True)
    parser.add_argument("--member1", type=Path, required=True)
    parser.add_argument("--pi05-base", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--expert-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.socket.exists():
        raise FileExistsError(f"refusing to reuse worker socket: {args.socket}")
    worker = StackCubeOnlineWorker(member0_checkpoint=args.member0, member1_checkpoint=args.member1, pi05_base=args.pi05_base, norm_stats=args.norm_stats, expert_dataset=args.expert_dataset, output_dir=args.output_dir)
    args.socket.parent.mkdir(parents=True, exist_ok=True)
    with _SocketServer(str(args.socket), _Handler) as server:
        server.worker = worker  # type: ignore[attr-defined]
        server.serve_forever()


if __name__ == "__main__":
    main()
