from pathlib import Path

from tools.run_xvla_airplane_event_close_training import training_command


def test_single_gpu_command_preserves_effective_batch_32() -> None:
    command = training_command(Path("meta"), Path("out"), steps=5000, save_interval=500)
    assert command[:2] == [str(Path("/data/zhaozhixuan/envs/xvla_official_5090/bin/python")), "train.py"]
    assert command[command.index("--batch_size") + 1] == "8"
    assert command[command.index("--gradient_accumulation_steps") + 1] == "4"
    assert command[command.index("--iters") + 1] == "5000"
    assert command[command.index("--save_interval") + 1] == "500"
