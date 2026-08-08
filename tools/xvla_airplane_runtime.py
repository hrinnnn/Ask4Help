"""Shared X-VLA preprocessing, inference, and detector helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def _rgb_image(value: torch.Tensor) -> Image.Image:
    array = value.detach().cpu().numpy()
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected one HWC RGB image, got {array.shape}")
    # Match the OpenCV byte decoder used by the training handler.
    return Image.fromarray(np.ascontiguousarray(array[..., ::-1].astype(np.uint8)))


class XVLAAirplanePolicy:
    def __init__(self, checkpoint: Path, xvla_root: Path, *, device: str = "cuda"):
        root = str(xvla_root.resolve())
        inserted = root not in sys.path
        if inserted:
            sys.path.insert(0, root)
        try:
            from models.modeling_xvla import XVLA
            from models.processing_xvla import XVLAProcessor
        finally:
            # X-VLA has a top-level ``datasets`` package. Leaving its root at
            # sys.path[0] shadows Hugging Face datasets, which LeRobot needs
            # when a rollout is admitted and written to disk.
            if inserted:
                sys.path.remove(root)

        self.device = torch.device(device)
        self.model = XVLA.from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16
        ).to(self.device).eval()
        self.processor = XVLAProcessor.from_pretrained(checkpoint)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def prepare(self, raw_obs: dict[str, Any], instruction: str) -> dict[str, torch.Tensor]:
        sensors = raw_obs["sensor_data"]
        images = torch.stack(
            [
                self.transform(_rgb_image(sensors["base_camera"]["rgb"])),
                self.transform(_rgb_image(sensors["hand_camera"]["rgb"])),
            ]
        ).unsqueeze(0).to(device=self.device, dtype=torch.bfloat16)
        proprio = torch.zeros((1, 20), dtype=torch.bfloat16, device=self.device)
        qpos = raw_obs["agent"]["qpos"].reshape(-1).to(self.device, torch.bfloat16)
        proprio[0, : min(9, qpos.numel())] = qpos[:9]
        input_ids = self.processor.encode_language([instruction])["input_ids"].to(self.device)
        return {
            "input_ids": input_ids,
            "image_input": images,
            "image_mask": torch.ones((1, 2), dtype=torch.bool, device=self.device),
            "domain_id": torch.zeros(1, dtype=torch.long, device=self.device),
            "proprio": proprio,
        }

    @staticmethod
    def pooled_bridge(encoding: dict[str, torch.Tensor]) -> torch.Tensor:
        tokens = torch.cat(
            [encoding["vlm_features"], encoding["aux_visual_inputs"]], dim=1
        )
        return tokens.float().mean(dim=1)

    def _generate_from_encoding(
        self,
        inputs: dict[str, torch.Tensor],
        encoding: dict[str, torch.Tensor],
        *,
        steps: int,
    ) -> torch.Tensor:
        batch = inputs["input_ids"].shape[0]
        dim = self.model.action_space.dim_action
        prior = torch.randn(
            batch,
            self.model.num_actions,
            dim,
            device=self.device,
            dtype=inputs["proprio"].dtype,
        )
        action = torch.zeros_like(prior)
        for index in range(max(1, int(steps)), 0, -1):
            time = torch.full(
                (batch,), index / steps, device=self.device, dtype=prior.dtype
            )
            noisy = prior * time[:, None, None] + action * (1 - time[:, None, None])
            proprio, noisy = self.model.action_space.preprocess(inputs["proprio"], noisy)
            action = self.model.transformer(
                domain_id=inputs["domain_id"],
                action_with_noise=noisy,
                proprio=proprio,
                t=time,
                **encoding,
            )
        return self.model.action_space.postprocess(action)

    @torch.inference_mode()
    def predict(
        self,
        raw_obs: dict[str, Any],
        instruction: str,
        *,
        seed: int,
        steps: int = 10,
    ) -> tuple[
        np.ndarray,
        torch.Tensor,
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        torch.manual_seed(seed)
        inputs = self.prepare(raw_obs, instruction)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            encoding = self.model.forward_vlm(
                inputs["input_ids"], inputs["image_input"], inputs["image_mask"]
            )
            actions = self._generate_from_encoding(inputs, encoding, steps=steps)
        return (
            actions.float().cpu().numpy(),
            self.pooled_bridge(encoding).cpu(),
            inputs,
            encoding,
        )

    @torch.inference_mode()
    def diffdagger_score(
        self,
        inputs: dict[str, torch.Tensor],
        encoding: dict[str, torch.Tensor],
        generated_actions: np.ndarray,
        *,
        num_timesteps: int = 16,
        num_noise_samples: int = 1,
    ) -> float:
        action = torch.as_tensor(generated_actions, device=self.device, dtype=torch.bfloat16)
        if action.ndim == 2:
            action = action.unsqueeze(0)
        action = self.model.action_space._pad_to_model_dim(action)
        scores = torch.zeros(action.shape[0], device=self.device, dtype=torch.float32)
        times = (torch.arange(num_timesteps, device=self.device) + 0.5) / num_timesteps
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(num_noise_samples):
                for value in times:
                    noise = torch.randn_like(action)
                    time = value.to(action.dtype).expand(action.shape[0])
                    noisy = noise * time[:, None, None] + action * (1 - time[:, None, None])
                    proprio, noisy = self.model.action_space.preprocess(inputs["proprio"], noisy)
                    prediction = self.model.transformer(
                        domain_id=inputs["domain_id"],
                        action_with_noise=noisy,
                        proprio=proprio,
                        t=time,
                        **encoding,
                    )
                    error = prediction[..., :8].float() - action[..., :8].float()
                    scores += error.square().flatten(1).mean(1)
        return float((scores / (num_timesteps * num_noise_samples))[0].cpu())
