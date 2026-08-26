"""Register the Panda-airplane parquet handler in an X-VLA runtime.

The upstream X-VLA checkout used by this project may contain the handler as a
local runtime patch, while a fresh checkout does not.  Keeping the exact
registration here makes dataset readers used by asset builders and training
shims share one auditable implementation without modifying the upstream tree.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def install_panda_airplane_handler() -> None:
    """Install the explicit ``panda_airplane`` domain handler once."""

    from datasets.domain_handler import registry

    if "panda_airplane" in registry._REGISTRY:
        return
    from datasets.domain_handler.base import DomainHandler
    from datasets.utils import decode_image_from_bytes, read_parquet

    class PandaAirplaneHandler(DomainHandler):
        """Read the existing 8D Panda airplane LeRobot parquet dataset."""

        def iter_episode(
            self,
            traj_idx: int,
            *,
            num_actions: int,
            training: bool,
            image_aug,
            lang_aug_map: dict | None,
            **kwargs,
        ):
            item = self.meta["datalist"][traj_idx]
            episode_index = int(item["episode_index"])
            episode_chunk = episode_index // int(self.meta["chunks_size"])
            data_path = self.meta["data_path"].format(
                episode_chunk=episode_chunk,
                episode_index=episode_index,
            )
            data = read_parquet(f"{self.meta['root_path']}/{data_path}")
            total = len(data["actions"])
            indices = list(range(total))
            if training:
                random.shuffle(indices)
            instruction = item["tasks"][0]

            for index in indices:
                main_image = decode_image_from_bytes(data["image"][index]["bytes"])
                wrist_image = decode_image_from_bytes(data["wrist_image"][index]["bytes"])
                image_input = torch.stack(
                    [image_aug(main_image), image_aug(wrist_image)], dim=0
                )
                proprio = np.zeros(20, dtype=np.float32)
                proprio[:9] = np.asarray(data["state"][index], dtype=np.float32)
                real_actions = np.asarray(
                    data["actions"][index : index + num_actions], dtype=np.float32
                )
                valid_steps = len(real_actions)
                action = np.zeros((num_actions, 20), dtype=np.float32)
                action[:valid_steps, :8] = real_actions
                if valid_steps < num_actions:
                    action[valid_steps:, :8] = np.asarray(
                        data["actions"][-1], dtype=np.float32
                    )
                yield {
                    "language_instruction": instruction,
                    "image_input": image_input,
                    "image_mask": torch.ones(2, dtype=torch.bool),
                    "abs_trajectory": torch.from_numpy(
                        np.concatenate([proprio[None], action], axis=0)
                    ),
                    "action_valid_mask": torch.arange(num_actions) < valid_steps,
                    "idx_for_delta": [],
                    "idx_for_mask_proprio": [],
                }

    registry._REGISTRY["panda_airplane"] = PandaAirplaneHandler
