import json
from pathlib import Path

import pytest

from tools.run_xvla_stackcube_id_training import TASK, build_manifest, training_command


def make_dataset(root: Path, *, short_episode: int | None = None) -> Path:
    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps({"chunks_size": 1000, "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"}),
        encoding="utf-8",
    )
    rows = []
    for index in range(128):
        rows.append(
            json.dumps(
                {
                    "episode_index": index,
                    "length": 9 if index == short_episode else 20,
                    "tasks": ["stale task"],
                }
            )
        )
    (meta / "episodes.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return root


def test_manifest_uses_all_128_id_episodes_and_canonical_task(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path / "dataset")
    output = tmp_path / "manifest.json"
    manifest = build_manifest(dataset, output)

    assert manifest["dataset_name"] == "panda_stackcube_id_128"
    assert len(manifest["datalist"]) == 128
    assert {tuple(row["tasks"]) for row in manifest["datalist"]} == {(TASK,)}
    assert json.loads(output.read_text(encoding="utf-8")) == manifest


def test_manifest_rejects_episode_without_full_action_chunk(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path / "dataset", short_episode=17)
    with pytest.raises(ValueError, match="episode 17 has only 9 actions"):
        build_manifest(dataset, tmp_path / "manifest.json")


def test_training_command_matches_official_two_gpu_airplane_recipe() -> None:
    command = training_command(Path("manifest.json"), Path("out"), steps=10_000, save_interval=500)
    assert command[1:8] == [
        "launch",
        "--num_processes",
        "2",
        "--multi_gpu",
        "--mixed_precision",
        "bf16",
        "--gpu_ids",
    ]
    assert command[8] == "0,1"
    assert command[command.index("--batch_size") + 1] == "8"
    assert command[command.index("--gradient_accumulation_steps") + 1] == "2"
    assert command[command.index("--num_actions") + 1] == "10"
    assert command[command.index("--iters") + 1] == "10000"
    assert command[command.index("--save_interval") + 1] == "500"
