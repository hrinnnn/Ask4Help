from pathlib import Path

import json

from tools.run_xvla_airplane_event_close_training import (
    child_env,
    forward_smoke_succeeded,
    training_command,
)


def test_single_gpu_command_preserves_effective_batch_32() -> None:
    command = training_command(Path("meta"), Path("out"), steps=5000, save_interval=500)
    assert command[:2] == [str(Path("/data/zhaozhixuan/envs/xvla_official_5090/bin/python")), "train.py"]
    assert command[command.index("--batch_size") + 1] == "8"
    assert command[command.index("--gradient_accumulation_steps") + 1] == "4"
    assert command[command.index("--iters") + 1] == "5000"
    assert command[command.index("--save_interval") + 1] == "500"


def test_single_gpu_environment_initializes_one_process_group_per_job() -> None:
    environment = child_env(3)
    assert environment["CUDA_VISIBLE_DEVICES"] == "3"
    assert environment["WORLD_SIZE"] == "1"
    assert environment["RANK"] == "0"
    assert environment["MASTER_PORT"] == "29623"


def test_forward_smoke_accepts_teardown_abort_only_with_complete_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "forward"
    output.mkdir()
    (output / "summary.json").write_text(json.dumps({"episodes": 1}), encoding="utf-8")
    (output / "episode.npy").write_bytes(b"actions")
    (output / "episode.mp4").write_bytes(b"video")

    assert forward_smoke_succeeded(output, 0)
    assert forward_smoke_succeeded(output, -6)
    assert not forward_smoke_succeeded(output, 1)

    (output / "episode.mp4").unlink()
    assert not forward_smoke_succeeded(output, -6)
