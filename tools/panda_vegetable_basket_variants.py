"""Panda-controlled BridgeData vegetable-basket task variants.

The stock ``BaseBridgeEnv`` selects a WidowX robot internally.  This module
keeps its scene/evaluation implementation but duplicates only its constructor
so the same BridgeData assets are paired with the existing Panda agent.
"""

from __future__ import annotations

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks.digital_twins.bridge_dataset_eval.base_env import (
    BRIDGE_DATASET_ASSET_PATH,
    BaseBridgeEnv,
)
from mani_skill.envs.tasks.digital_twins.base_env import BaseDigitalTwinEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import io_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.geometry import rotation_conversions
from mani_skill.utils.structs.pose import Pose


try:
    from rlinf.envs.maniskill.tasks.panda_table_agent import PandaBridgeDatasetFlatTable
except ModuleNotFoundError as exc:  # pragma: no cover - exercised on runtime host
    raise ModuleNotFoundError(
        "PandaBridgeDatasetFlatTable must be importable from the RLinf checkout"
    ) from exc


def _reset_configs() -> tuple[torch.Tensor, torch.Tensor]:
    """Return the paired central reset used by both object variants."""

    target = np.array([-0.125, 0.025, 1.0], dtype=np.float32)
    # The WidowX sink camera's historical y=0.269 reset is at the edge of the
    # Panda Bridge camera.  Keep one fixed Panda-specific central point.
    center = np.array([-0.143, 0.140], dtype=np.float32)
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

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Reset Bridge objects and the Panda using its 9-DoF initial state."""

        with torch.device(self.device):
            batch_size = len(env_idx)
            if "episode_id" in options:
                episode_id = options["episode_id"]
                if isinstance(episode_id, int):
                    episode_id = torch.tensor([episode_id])
                position_ids = (episode_id % (len(self.xyz_configs) * len(self.quat_configs))) // len(self.quat_configs)
                quaternion_ids = episode_id % len(self.quat_configs)
            else:
                position_ids = torch.randint(0, len(self.xyz_configs), size=(batch_size,))
                quaternion_ids = torch.randint(0, len(self.quat_configs), size=(batch_size,))

            for index, actor in enumerate(self.objs.values()):
                actor.set_pose(
                    Pose.create_from_pq(
                        p=self.xyz_configs[position_ids, index],
                        q=self.quat_configs[quaternion_ids, index],
                    )
                )
            if self.scene.gpu_sim_enabled:
                self.scene._gpu_apply_all()
            self._settle(0.5)
            if self.scene.gpu_sim_enabled:
                self.scene._gpu_fetch_all()

            linear_velocity = 0.0
            angular_velocity = 0.0
            for obj in self.objs.values():
                linear_velocity += torch.linalg.norm(obj.linear_velocity)
                angular_velocity += torch.linalg.norm(obj.angular_velocity)
            if linear_velocity > 1e-3 or angular_velocity > 1e-2:
                if self.scene.gpu_sim_enabled:
                    self.scene._gpu_apply_all()
                self._settle(6)
                if self.scene.gpu_sim_enabled:
                    self.scene._gpu_fetch_all()

            self.agent.robot.set_pose(
                sapien.Pose([0.3, 0.028, 0.870], q=[0, 0, 0, 1])
            )
            panda_qpos = np.array(
                [0.0, 0.259, 0.0, -2.289, 0.0, 2.515, np.pi / 4, 0.04, 0.015],
                dtype=np.float32,
            )
            self.agent.reset(init_qpos=panda_qpos)

            self.episode_source_obj_xyz_after_settle = self.objs[self.source_obj_name].pose.p
            self.episode_target_obj_xyz_after_settle = self.objs[self.target_obj_name].pose.p
            self.episode_obj_xyzs_after_settle = {
                name: obj.pose.p for name, obj in self.objs.items()
            }
            self.episode_source_obj_bbox_world = self.episode_model_bbox_sizes[self.source_obj_name].float()
            self.episode_target_obj_bbox_world = self.episode_model_bbox_sizes[self.target_obj_name].float()
            self.episode_source_obj_bbox_world = (
                rotation_conversions.quaternion_to_matrix(self.objs[self.source_obj_name].pose.q)
                @ self.episode_source_obj_bbox_world[..., None]
            )[0, :, 0]
            self.episode_target_obj_bbox_world = (
                rotation_conversions.quaternion_to_matrix(self.objs[self.target_obj_name].pose.q)
                @ self.episode_target_obj_bbox_world[..., None]
            )[0, :, 0]

            if self.consecutive_grasp is None:
                self.consecutive_grasp = torch.zeros(self.num_envs, dtype=torch.int32).to(self.device)
            if self.episode_stats is None:
                self.episode_stats = {
                    "moved_correct_obj": torch.zeros((self.num_envs,), dtype=torch.bool).to(self.device),
                    "moved_wrong_obj": torch.zeros((self.num_envs,), dtype=torch.bool).to(self.device),
                    "is_src_obj_grasped": torch.zeros((self.num_envs,), dtype=torch.bool).to(self.device),
                    "consecutive_grasp": torch.zeros((self.num_envs,), dtype=torch.bool).to(self.device),
                }
            self.consecutive_grasp[env_idx] = 0
            for key in self.episode_stats:
                self.episode_stats[key][env_idx] = 0

    @property
    def _default_human_render_camera_configs(self):
        """Use a Panda-compatible debug camera instead of the WidowX mount."""

        return CameraConfig(
            "render_camera",
            pose=sapien.Pose(
                [0.00, -0.16, 0.336],
                [0.909182, -0.0819809, 0.347277, 0.214629],
            ),
            width=512,
            height=512,
            intrinsic=np.array(
                [[623.588, 0, 319.501], [0, 623.588, 239.545], [0, 0, 1]],
                dtype=np.float32,
            ),
            near=0.01,
            far=100,
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
