import numpy as np
import pytest
import torch

from tools.xvla_stackcube_temporal_mask import masked_action_mse, padded_action_chunk


def test_final_observation_is_retained_with_one_valid_target() -> None:
    actions = np.arange(40 * 8, dtype=np.float32).reshape(40, 8)
    chunk, mask = padded_action_chunk(actions, anchor=39, horizon=10)
    assert mask.tolist() == [True] + [False] * 9
    np.testing.assert_array_equal(chunk[0], actions[-1])
    np.testing.assert_array_equal(chunk[1:], np.repeat(actions[-1][None], 9, axis=0))


def test_every_real_observation_can_be_an_anchor() -> None:
    actions = np.zeros((39, 8), dtype=np.float32)
    chunks = [padded_action_chunk(actions, anchor, 10) for anchor in range(len(actions))]
    assert len(chunks) == 39
    assert [int(mask.sum()) for _, mask in chunks[-10:]] == list(range(10, 0, -1))


def test_masked_loss_ignores_padded_timesteps_and_dummy_dimensions() -> None:
    pred = torch.zeros(1, 3, 20)
    target = torch.zeros_like(pred)
    pred[:, 0, :8] = 2
    pred[:, 1:, :8] = 100
    pred[:, :, 8:] = 1000
    loss = masked_action_mse(
        pred, target, torch.tensor([[True, False, False]]), real_dim=8
    )
    assert torch.allclose(loss, torch.tensor(4.0))


def test_masked_loss_rejects_anchor_without_real_target() -> None:
    with pytest.raises(ValueError, match="at least one real action"):
        masked_action_mse(
            torch.zeros(1, 3, 20),
            torch.zeros(1, 3, 20),
            torch.zeros(1, 3, dtype=torch.bool),
            real_dim=8,
        )
