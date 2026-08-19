from pathlib import Path
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "RLinf"))

from rlinf.envs.maniskill.open_drawer_retrieve_place_spec import (  # noqa: E402
    ENV_IDS,
    TASK_INSTRUCTION,
    validate_reset_metadata,
    _to_numpy,
)


def valid_id_metadata() -> dict:
    return {
        "env_id": ENV_IDS["id"],
        "split": "id",
        "instruction": TASK_INSTRUCTION,
        "control_mode": "pd_joint_delta_pos",
        "camera": {
            "main": "base_camera",
            "wrist": "hand_camera",
            "main_shape": [384, 384, 3],
            "wrist_shape": [384, 384, 3],
            "requested_size": [384, 384],
        },
        "handle_lateral_offset": 0.0,
        "drawer_qpos": [0.0],
        "object_pose": {"p": [0.305, 0.0, 0.058], "q": [1.0, 0.0, 0.0, 0.0], "yaw_deg": 0.0},
        "target_pose": {"p": [0.03, -0.30, 0.0], "q": [1.0, 0.0, 0.0, 0.0]},
        "lifecycle": {
            "drawer_opened": False,
            "object_grasped": False,
            "object_lifted": False,
            "object_in_target": False,
            "object_released_now": True,
            "release_after_grasp_event": False,
            "ever_drawer_opened": False,
            "ever_grasped": False,
            "ever_lifted": False,
            "is_robot_static": True,
            "success": False,
        },
    }


def test_valid_id_reset_metadata_passes() -> None:
    assert validate_reset_metadata(valid_id_metadata(), split="id") == []


def test_reset_state_tensor_conversion_handles_cpu_and_cuda() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    value = torch.tensor([1.0, 2.0], device=device)
    converted = _to_numpy(value)
    assert isinstance(converted, np.ndarray)
    assert converted.tolist() == [1.0, 2.0]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("instruction", "open the drawer and place the object in the tray"),
        ("env_id", "wrong-env-v1"),
        ("drawer_qpos", [0.02]),
    ],
)
def test_reset_metadata_rejects_prompt_env_and_range(field: str, value) -> None:
    metadata = valid_id_metadata()
    metadata[field] = value
    assert validate_reset_metadata(metadata, split="id")


def test_reset_metadata_rejects_real_lifecycle_and_camera_shape() -> None:
    lifecycle_metadata = valid_id_metadata()
    lifecycle_metadata["lifecycle"]["object_grasped"] = True
    assert validate_reset_metadata(lifecycle_metadata, split="id")

    camera_metadata = valid_id_metadata()
    camera_metadata["camera"]["wrist_shape"] = [224, 224, 3]
    assert validate_reset_metadata(camera_metadata, split="id")


def test_all_open_drawer_pi05_yaml_prompts_match_task_constant() -> None:
    config_paths = (
        ROOT / "RLinf/examples/sft/config/open_drawer_retrieve_place_id_sft_openpi_pi05.yaml",
        ROOT / "RLinf/examples/sft/config/open_drawer_retrieve_place_id_continuation_openpi_pi05.yaml",
        ROOT / "configs/open_drawer_retrieve_place_dagger_sft_openpi_pi05.yaml",
    )
    expected = f"default_prompt: {TASK_INSTRUCTION}"
    for path in config_paths:
        assert expected in path.read_text(encoding="utf-8")
