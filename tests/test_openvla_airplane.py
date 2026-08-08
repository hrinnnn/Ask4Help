import json
from pathlib import Path

import numpy as np
import pytest

from openvla_airplane.dataset import action_token_round_trip, normalize_action
from openvla_airplane.layers import SELECTED_LLAMA_BLOCKS, validate_selected_blocks
from openvla_airplane.metrics import summarize


class FakeTokenizer:
    vocab_size = 1000

    def __call__(self, text, add_special_tokens=False):
        return type("Tokenized", (), {"input_ids": [int(token[1:]) for token in text.split(",")]})


class FakeActionTokenizer:
    def __call__(self, action):
        return ",".join(f"t{index + 100}" for index in range(len(action)))

    def decode_token_ids_to_actions(self, token_ids):
        return np.linspace(-1.0, 1.0, len(token_ids), dtype=np.float32)


def _summary(path: Path, split: str, scores: list[float], success: list[bool]):
    path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "split": split,
                        "ever_grasped": label,
                        "timeline": [{"scores": {"detector": score}} for score in [value]],
                    }
                    for value, label in zip(scores, success)
                ]
            }
        )
    )


def test_action_token_round_trip_is_eight_dimensional():
    result = action_token_round_trip(np.linspace(-1, 1, 8), FakeTokenizer(), FakeActionTokenizer())
    assert result["action_dim"] == 8
    assert len(result["token_ids"]) == 8


def test_action_normalization_uses_fixed_id_bounds():
    stats = {"q01": [-1.0, 0.0], "q99": [1.0, 2.0]}
    np.testing.assert_allclose(normalize_action(np.asarray([0.0, 1.0]), stats), [0.0, 0.0])


def test_metrics_uses_trajectory_max_and_separates_id_false_alarm(tmp_path):
    id_path, ood_path, thresholds = tmp_path / "id.json", tmp_path / "ood.json", tmp_path / "thresholds.json"
    _summary(id_path, "id", [0.1, 0.2], [True, True])
    _summary(ood_path, "ood", [0.3, 0.9], [False, False])
    thresholds.write_text(json.dumps({"methods": {"detector": {"threshold": 0.25}}}))
    payload = summarize(id_path, ood_path, thresholds)
    row = payload["methods"][0]
    assert row["auroc"] == 1.0
    assert row["false_alarm_rate_id"] == 0.0
    assert row["oracle_best_balanced_accuracy"] == 1.0


def test_metrics_without_calibration_reports_only_threshold_free_and_oracle_metrics(tmp_path):
    id_path, ood_path = tmp_path / "id.json", tmp_path / "ood.json"
    _summary(id_path, "id", [0.1], [True])
    _summary(ood_path, "ood", [0.9], [False])
    row = summarize(id_path, ood_path)["methods"][0]
    assert row["auprc"] == 1.0
    assert row["auroc"] == 1.0
    assert row["oracle_best_balanced_accuracy"] == 1.0
    assert row["threshold"] is None
    assert row["balanced_accuracy"] is None


def test_representative_llama_blocks_are_uniformly_spaced():
    assert SELECTED_LLAMA_BLOCKS == (8, 16, 24, 32)
    assert validate_selected_blocks(32) == SELECTED_LLAMA_BLOCKS
    with pytest.raises(ValueError, match="block 32"):
        validate_selected_blocks(31)
