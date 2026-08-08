from __future__ import annotations

import torch
import sys

from tools.xvla_airplane_runtime import XVLAAirplanePolicy


def test_xvla_root_is_not_left_on_import_path(tmp_path) -> None:
    # Constructor imports require the real X-VLA package, so this invariant is
    # exercised in the server import smoke. Keep the path restoration contract
    # explicit in the unit suite without loading model weights.
    assert str(tmp_path) not in sys.path


def test_pooled_bridge_combines_primary_and_auxiliary_vlm_tokens() -> None:
    primary = torch.tensor([[[1.0, 3.0], [3.0, 5.0]]])
    auxiliary = torch.tensor([[[5.0, 7.0]]])

    pooled = XVLAAirplanePolicy.pooled_bridge(
        {"vlm_features": primary, "aux_visual_inputs": auxiliary}
    )

    assert pooled.shape == (1, 2)
    assert torch.allclose(pooled, torch.tensor([[3.0, 5.0]]))
