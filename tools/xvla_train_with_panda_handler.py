#!/usr/bin/env python3
"""Register the existing PandaAirplane handler before invoking X-VLA train.py."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: xvla_train_with_panda_handler.py /path/to/train.py [args...]")
    train_path = Path(sys.argv[1]).resolve()
    sys.argv = [str(train_path), *sys.argv[2:]]
    # X-VLA is the working directory for the subprocess; make its package
    # import explicit because this shim itself lives in Ask4Help/tools.
    sys.path.insert(0, str(train_path.parent))
    from datasets.domain_handler import registry
    from datasets.domain_handler.panda_airplane import PandaAirplaneHandler
    from models.action_hub import AutoActionSpace
    import torch

    registry._REGISTRY["panda_airplane"] = PandaAirplaneHandler

    def masked_auto_loss(self, pred, target, valid_mask=None):
        pred = self._pad_to_model_dim(pred)
        target = self._pad_to_model_dim(target)
        if valid_mask is None:
            value = (pred[..., : self.real_dim] - target[..., : self.real_dim]) ** 2
            return {"joints_loss": value.mean() * self.JOINTS_SCALE}
        mask = valid_mask
        if mask.ndim == pred.ndim - 1:
            mask = mask.unsqueeze(-1)
        mask = mask.to(dtype=pred.dtype)[..., : self.real_dim]
        value = (pred[..., : self.real_dim] - target[..., : self.real_dim]) ** 2
        return {
            "joints_loss": (value * mask).sum()
            / mask.sum().clamp_min(1.0)
            * self.JOINTS_SCALE
        }

    AutoActionSpace.compute_loss = masked_auto_loss
    runpy.run_path(str(train_path), run_name="__main__")


if __name__ == "__main__":
    main()
