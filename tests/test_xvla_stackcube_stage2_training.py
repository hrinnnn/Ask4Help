from pathlib import Path

from tools.run_xvla_stackcube_stage2_training import (
    temporal_mask_report,
    training_command,
)


def test_stage2_training_uses_comparable_2500_step_protocol() -> None:
    command = training_command(
        Path("python"),
        Path("start"),
        Path("meta"),
        Path("output"),
        steps=2500,
        save_interval=500,
        seed=7300,
    )
    assert command[command.index("--iters") + 1] == "2500"
    assert command[command.index("--save_interval") + 1] == "500"
    assert command[command.index("--gradient_accumulation_steps") + 1] == "4"
    assert command[command.index("--batch_size") + 1] == "8"


def test_temporal_mask_report_retains_every_tail_anchor() -> None:
    report = temporal_mask_report([{"length": 12}, {"length": 4}], horizon=10)
    assert report["total_anchors"] == 16
    assert report["tail_anchors"] == 13
    assert report["valid_target_count_distribution"]["1"] == 2
    assert report["final_observation_valid_targets"] == 1
