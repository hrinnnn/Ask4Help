from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).parents[1]
MODULE_PATH = ROOT / "tools" / "libero_plus_failure" / "full_reference_bank.py"
SPEC = importlib.util.spec_from_file_location("libero_plus_full_reference_bank", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
BANK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BANK)

FIT_PATH = ROOT / "tools" / "libero_plus_failure" / "fit_all_observation_reference_assets.py"
FIT_SPEC = importlib.util.spec_from_file_location("libero_plus_full_reference_fit", FIT_PATH)
assert FIT_SPEC is not None and FIT_SPEC.loader is not None
FIT = importlib.util.module_from_spec(FIT_SPEC)
sys.modules[FIT_SPEC.name] = FIT
FIT_SPEC.loader.exec_module(FIT)


def _record(*, task: int, episode: int, frame: int, length: int) -> dict:
    raw = [frame + offset for offset in range(BANK.ACTION_HORIZON)]
    padded = [value >= length for value in raw]
    return {
        "task_index": task, "episode_index": episode, "frame_id": frame,
        "action_indices": [min(length - 1, value) for value in raw],
        "action_is_pad": padded, "tail_padding_count": sum(padded),
    }


def test_all_observation_records_keep_terminal_tail_and_validate() -> None:
    records = []
    for task in range(10):
        records.append(_record(task=task, episode=task, frame=0, length=2))
        records.append(_record(task=task, episode=task, frame=1, length=2))
    result = BANK.validate_record_sequence(records, expected_frames=20)
    assert result["terminal_tail_records"] == 20
    tail = records[1]
    assert tail["action_indices"] == [1] * BANK.ACTION_HORIZON
    assert tail["action_is_pad"] == [False] + [True] * (BANK.ACTION_HORIZON - 1)


def test_all_observation_validation_rejects_duplicate_and_bad_padding() -> None:
    records = [_record(task=task, episode=task, frame=0, length=1) for task in range(10)]
    records.append(dict(records[0]))
    with pytest.raises(ValueError, match="duplicate"):
        BANK.validate_record_sequence(records)
    records = [_record(task=task, episode=task, frame=0, length=1) for task in range(10)]
    records[0]["tail_padding_count"] = 0
    with pytest.raises(ValueError, match="padding"):
        BANK.validate_record_sequence(records)


def test_completed_episode_shard_requires_matching_features_and_metadata(tmp_path: Path) -> None:
    feature = tmp_path / "episode_000000.npz"
    metadata = tmp_path / "episode_000000.json"
    np.savez(feature, bridge=np.zeros((2, 1, 2048), dtype=np.float32), action_expert_final=np.zeros((2, 10, 1024), dtype=np.float32))
    metadata.write_text(json.dumps({"format": BANK.FORMAT, "frame_count": 2, "records": [{}, {}]}), encoding="utf-8")
    assert BANK.complete_episode_shard(feature, metadata)
    metadata.write_text(json.dumps({"format": BANK.FORMAT, "frame_count": 1, "records": [{}]}), encoding="utf-8")
    assert not BANK.complete_episode_shard(feature, metadata)


def test_streaming_moments_match_monolithic_population_statistics() -> None:
    values = np.arange(5 * 2 * 3, dtype=np.float32).reshape(5, 2, 3)
    moments = FIT.StreamingMoments.empty((2, 3))
    moments.update(values[:2])
    moments.update(values[2:])
    direct = torch.as_tensor(values, dtype=torch.float64)
    expected_mean = direct.mean(dim=0)
    centered = direct - expected_mean.unsqueeze(0)
    expected_m2 = torch.einsum("nth,ntk->thk", centered, centered)
    assert moments.count == 5
    torch.testing.assert_close(moments.mean, expected_mean)
    torch.testing.assert_close(moments.m2, expected_m2)
