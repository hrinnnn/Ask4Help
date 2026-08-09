import numpy as np
import pytest

from tools.evaluate_stackcube_xvla import clip_action_chunk


def test_clip_action_chunk_executes_only_real_panda_dimensions() -> None:
    predicted = np.arange(200, dtype=np.float32).reshape(1, 10, 20)
    low = np.full(8, -1, dtype=np.float32)
    high = np.full(8, 1, dtype=np.float32)
    chunk = clip_action_chunk(predicted, low, high, 5)
    assert chunk.shape == (5, 8)
    np.testing.assert_array_equal(chunk, np.clip(predicted[0, :5, :8], low, high))


def test_clip_action_chunk_rejects_multi_environment_batch() -> None:
    with pytest.raises(ValueError, match="one rollout environment"):
        clip_action_chunk(
            np.zeros((2, 10, 20), dtype=np.float32),
            np.full(8, -1, dtype=np.float32),
            np.full(8, 1, dtype=np.float32),
            5,
        )
