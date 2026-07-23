import torch

from rlinf.algorithms.online_awbc import first_vfd_action_candidate


def test_candidate_zero_has_action_chunk_shape():
    candidates = torch.arange(1 * 5 * 10 * 8).reshape(1, 5, 10, 8)
    selected = first_vfd_action_candidate(candidates)
    assert selected.shape == (1, 10, 8)
    torch.testing.assert_close(selected, candidates[:, 0])
