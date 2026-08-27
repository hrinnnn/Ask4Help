"""Panda-controlled BridgeData vegetable-basket task variants.

The stock ``BaseBridgeEnv`` selects a WidowX robot internally.  This module
keeps its scene/evaluation implementation but duplicates only its constructor
so the same BridgeData assets are paired with the existing Panda agent.
"""

from __future__ import annotations

import numpy as np
import torch
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks.digital_twins.bridge_dataset_eval.base_env import (
    BRIDGE_DATASET_ASSET_PATH,
    BaseBridgeEnv,
)
from mani_skill.envs.tasks.digital_twins.base_env import BaseDigitalTwinEnv
from mani_skill.utils import io_utils
from mani_skill.utils.registration import register_env


try:
    from rlinf.envs.maniskill.tasks.panda_table_agent import PandaBridgeDatasetFlatTable
except ModuleNotFoundError as exc:  # pragma: no cover - exercised on runtime host
    raise ModuleNotFoundError(
        "PandaBridgeDatasetFlatTable must be importable from the RLinf checkout"
    ) from exc


def _reset_configs() -> tuple[torch.Tensor, torch.Tensor]:
    """Return the paired central reset used by both object variants."""

    target = np.array([-0.125, 0.025, 1.0], dtype=np.float32)
    center = np.array([-0.143, 0.269], dtype=np.float32)
    xyz = torch.tensor(
        np.stack([np.stack([np.array([center[0], center[1], 0.937]), target])]),
        dtype=torch.float32,
    )
    source_quat = euler2quat(0, 0, 0, "sxyz")
    target_quat = np.array([1, 0, 0, 0], dtype=np.float32)
    quats = torch.tensor(
        np.stack([[source_quat, target_quat]]), dtype=torch.float32
    )
    return xyz, quats


class _PandaPutVegetableInBasket(BaseBridgeEnv):
    """Shared Panda version of the BridgeData sink/basket task."""

    scene_setting = "sink"
    source_asset = "eggplant"
    task_instruction = "put the vegetable into the yellow basket"
    objects_excluded_from_greenscreening = ["eggplant"]

    def __init__(self, **kwargs):
        xyz_configs, quat_configs = _reset_configs()
        self.objs = {}
        self.obj_names = [self.source_asset, "dummy_sink_target_plane"]
        self.source_obj_name = self.obj_names[0]
        self.target_obj_name = self.obj_names[1]
        self.xyz_configs = xyz_configs
        self.quat_configs = quat_configs
        self.rgb_overlay_paths = {
            "3rd_view_camera": str(
                BRIDGE_DATASET_ASSET_PATH / "real_inpainting/bridge_sink.png"
            )
        }
        self.model_db = io_utils.load_json(
            BRIDGE_DATASET_ASSET_PATH / "custom/info_bridge_custom_v0.json"
        )
        self.consecutive_grasp = None
        self.episode_stats = None
        BaseDigitalTwinEnv.__init__(
            self,
            robot_uids=PandaBridgeDatasetFlatTable,
            **kwargs,
        )

    def evaluate(self, *args, **kwargs):
        return self._evaluate(
            success_require_src_completely_on_target=False,
            z_flag_required_offset=0.06,
            *args,
            **kwargs,
        )

    def get_language_instruction(self, **kwargs):
        return [self.task_instruction] * self.num_envs

    def _load_lighting(self, options):
        self.scene.set_ambient_light([0.3, 0.3, 0.3])
        self.scene.add_directional_light(
            [0, 0, -1],
            [0.3, 0.3, 0.3],
            position=[0, 0, 1],
            shadow=False,
            shadow_scale=5,
            shadow_map_size=2048,
        )


@register_env(
    "XVLAPandaPutVegetableInBasketID-v1",
    max_episode_steps=120,
    asset_download_ids=["bridge_v2_real2sim"],
)
class PandaPutVegetableInBasketID(_PandaPutVegetableInBasket):
    """Panda ID split using the eggplant asset."""

    source_asset = "eggplant"
    objects_excluded_from_greenscreening = ["eggplant"]


@register_env(
    "XVLAPandaPutVegetableInBasketOOD-v1",
    max_episode_steps=120,
    asset_download_ids=["bridge_v2_real2sim"],
)
class PandaPutVegetableInBasketOOD(_PandaPutVegetableInBasket):
    """Panda object-OOD split using the modified carrot asset."""

    source_asset = "bridge_carrot_generated_modified"
    objects_excluded_from_greenscreening = ["bridge_carrot_generated_modified"]
