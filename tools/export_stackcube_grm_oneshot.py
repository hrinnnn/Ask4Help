#!/usr/bin/env python3
"""Export one successful StackCube demonstration to Robo-Dopamine's raw format."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rgb(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    image = np.asarray(value)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.dtype.kind == "f":
        image = image * 255 if image.max() <= 1 else image
    return np.clip(image[..., :3], 0, 255).astype(np.uint8)


def _load_events(path: Path, *, episode_id: int) -> dict[int, float]:
    events: dict[int, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if int(row.get("episode_index", -1)) != episode_id:
            continue
        frame = int(row["frame_index"])
        phi = row.get("phi_next", row.get("phi"))
        if phi is not None:
            events[frame] = max(events.get(frame, 0.0), float(phi))
    return events


def _keyframes(events: dict[int, float], length: int) -> list[dict[str, int | str]]:
    if not events:
        raise ValueError("privileged event sidecar has no rows for the selected episode")
    milestones = [("grasp", 0.25), ("lift", 0.5), ("near_target", 0.75), ("success", 1.0)]
    boundaries = [0]
    labels: list[str] = ["start"]
    for label, threshold in milestones:
        matched = [frame for frame, phi in sorted(events.items()) if phi >= threshold]
        if matched:
            boundaries.append(matched[0])
            labels.append(label)
    if boundaries[-1] != length - 1:
        boundaries.append(length - 1)
        labels.append("terminal")
    result = []
    for index in range(len(boundaries) - 1):
        start, end = boundaries[index], boundaries[index + 1]
        if end <= start:
            continue
        result.append({"anotation": labels[min(index + 1, len(labels) - 1)], "start_frame_id": start, "end_frame_id": end})
    if not any(item["anotation"] == "success" for item in result):
        raise ValueError("selected one-shot source is not privileged-successful")
    return result


def main() -> None:
    try:
        import imageio.v3 as iio
    except ImportError as error:
        raise RuntimeError("StackCube one-shot video export requires imageio[ffmpeg]") from error
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--privileged-events", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--local-staging-dir",
        type=Path,
        default=Path("/tmp"),
        help="Local filesystem used to finalize mp4 trailers before OSSFS copy.",
    )
    parser.add_argument("--task", default="stack the red cube on the green cube")
    args = parser.parse_args()

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(str(args.dataset))
    positions = [index for index, episode in enumerate(dataset.hf_dataset["episode_index"]) if int(episode) == args.episode_index]
    if not positions:
        raise ValueError(f"dataset has no episode {args.episode_index}")
    output = args.output_dir / "episode_001"
    output.mkdir(parents=True, exist_ok=False)
    main = np.stack([_rgb(dataset[position]["image"]) for position in positions])
    wrist = np.stack([_rgb(dataset[position].get("wrist_image", dataset[position]["image"])) for position in positions])
    args.local_staging_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="stackcube_grm_", dir=args.local_staging_dir) as temporary:
        staging = Path(temporary)
        for name, frames in (("cam_high.mp4", main), ("cam_left_wrist.mp4", wrist), ("cam_right_wrist.mp4", wrist)):
            local_video = staging / name
            iio.imwrite(local_video, frames, fps=10)
            decoded = list(iio.imiter(local_video))
            if len(decoded) != len(frames):
                raise RuntimeError(f"video verification failed for {name}: {len(decoded)} != {len(frames)} frames")
            shutil.copy2(local_video, output / name)
    events = _load_events(args.privileged_events, episode_id=args.episode_index)
    keyframes = _keyframes(events, len(positions))
    (output / "annotated_keyframes.json").write_text(json.dumps(keyframes, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "task_instruction.json").write_text(json.dumps([args.task], indent=2) + "\n", encoding="utf-8")
    manifest = {
        "episode_id": args.episode_index,
        "seed": args.seed,
        "source_dataset": str(args.dataset.resolve()),
        "source_privileged_events": str(args.privileged_events.resolve()),
        "task": args.task,
        "views": {"cam_high": "base_camera", "cam_left_wrist": "hand_camera", "cam_right_wrist": "hand_camera duplicated"},
        "keyframe_source": "privileged StackCube event sidecar",
        "files": {name: _sha256(output / name) for name in ("cam_high.mp4", "cam_left_wrist.mp4", "cam_right_wrist.mp4", "annotated_keyframes.json")},
    }
    (args.output_dir / "oneshot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
