from __future__ import annotations

from collections.abc import Mapping

import torch


def move_pixel_values(value, device: int, dtype: torch.dtype = torch.bfloat16):
    if isinstance(value, Mapping):
        return {key: move_pixel_values(item, device, dtype) for key, item in value.items()}
    return value.to(device=device, dtype=dtype)
