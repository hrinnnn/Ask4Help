"""Shared multi-layer feature and detector utilities for X-VLA airplane tests."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch


def first_tensor(value: Any) -> torch.Tensor:
    """Return the hidden-state tensor emitted by a transformer block hook."""

    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            if isinstance(item, torch.Tensor):
                return item
    if isinstance(value, Mapping):
        for item in value.values():
            if isinstance(item, torch.Tensor):
                return item
    raise TypeError(f"Hook output does not contain a tensor: {type(value)!r}")


def masked_token_mean(tokens: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Pool valid sequence tokens without letting padding affect the feature."""

    if mask is None:
        return tokens.float().mean(dim=1)
    valid = mask.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(-1)
    denominator = valid.sum(dim=1).clamp_min(1)
    return (tokens * valid).sum(dim=1).float() / denominator.float()


class XVLAMultilayerProbe:
    """Capture all Florence encoder and action-transformer block outputs.

    Action features are evaluated with a fixed prior and a fixed integration
    schedule. This keeps the detector probe deterministic without changing the
    stochastic action chunk executed by the policy.
    """

    def __init__(self, model: torch.nn.Module, *, probe_seed: int = 0, probe_steps: int = 5):
        if probe_steps < 1:
            raise ValueError("probe_steps must be positive")
        self.model = model
        self.probe_steps = int(probe_steps)
        self.captured: dict[str, torch.Tensor] = {}
        encoder = model.vlm.language_model.model.encoder
        vlm_layers = getattr(encoder, "layers", None)
        if vlm_layers is None:
            raise AttributeError("Florence encoder does not expose .layers")
        action_layers = model.transformer.blocks
        self.vlm_layer_count = len(vlm_layers)
        self.action_layer_count = len(action_layers)
        self.encoder_inputs: torch.Tensor | None = None
        self.encoder_attention_mask: torch.Tensor | None = None
        self.handles = []
        self.handles.append(
            encoder.register_forward_pre_hook(self._encoder_input_hook, with_kwargs=True)
        )
        for index, layer in enumerate(vlm_layers, start=1):
            self.handles.append(layer.register_forward_hook(self._hook(f"vlm_encoder_{index:02d}")))
        for index, layer in enumerate(action_layers, start=1):
            self.handles.append(layer.register_forward_hook(self._hook(f"action_block_{index:02d}")))
        generator = torch.Generator(device="cpu").manual_seed(int(probe_seed))
        prior = torch.randn(
            1,
            model.num_actions,
            model.action_space.dim_action,
            generator=generator,
            dtype=torch.float32,
        )
        self.fixed_prior = prior

    def _hook(self, name: str):
        def capture(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            self.captured[name] = first_tensor(output)

        return capture

    def _encoder_input_hook(
        self,
        _module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        inputs = kwargs.get("inputs_embeds")
        if inputs is None and args:
            inputs = first_tensor(args)
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("Florence encoder pre-hook did not receive inputs_embeds")
        self.encoder_inputs = inputs
        attention_mask = kwargs.get("attention_mask")
        self.encoder_attention_mask = (
            attention_mask if isinstance(attention_mask, torch.Tensor) else None
        )

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def extract(
        self, inputs: Mapping[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Return one pooled feature per registered model location."""

        self.captured.clear()
        self.encoder_inputs = None
        self.encoder_attention_mask = None
        encoding = self.model.forward_vlm(
            inputs["input_ids"], inputs["image_input"], inputs["image_mask"]
        )
        if self.encoder_inputs is None:
            raise RuntimeError("Florence encoder input hook did not run")
        features: dict[str, torch.Tensor] = {}
        features["vlm_input_pool"] = masked_token_mean(
            self.encoder_inputs, self.encoder_attention_mask
        )
        for index in range(1, self.vlm_layer_count + 1):
            name = f"vlm_encoder_{index:02d}"
            features[name] = self.captured[name].float().mean(dim=1)
        bridge_tokens = torch.cat(
            [encoding["vlm_features"], encoding["aux_visual_inputs"]], dim=1
        )
        features["vlm_action_bridge"] = bridge_tokens.float().mean(dim=1)

        batch = inputs["input_ids"].shape[0]
        prior = self.fixed_prior.to(
            device=inputs["proprio"].device, dtype=inputs["proprio"].dtype
        ).expand(batch, -1, -1)
        action = torch.zeros_like(prior)
        for index in range(self.probe_steps, 0, -1):
            time = torch.full(
                (batch,),
                index / self.probe_steps,
                device=prior.device,
                dtype=prior.dtype,
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
        for index in range(1, self.action_layer_count + 1):
            name = f"action_block_{index:02d}"
            # The first num_actions tokens are the action segment.
            features[name] = self.captured[name][:, : self.model.num_actions].float().mean(dim=1)
        return features, encoding


def layer_names(vlm_layers: int, action_layers: int) -> list[str]:
    return (
        ["vlm_input_pool"]
        + [f"vlm_encoder_{index:02d}" for index in range(1, vlm_layers + 1)]
        + ["vlm_action_bridge"]
        + [f"action_block_{index:02d}" for index in range(1, action_layers + 1)]
    )


def fit_layer_asset(
    values: torch.Tensor, *, pca_dim: int = 512, ridge: float = 1e-6
) -> dict[str, torch.Tensor | int | float]:
    """Fit PCA residual, full Gaussian LLMD, and the kNN reference bank."""

    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError("features must have shape [observations, hidden] with both dimensions >= 2")
    values = values.detach().float()
    if not torch.isfinite(values).all():
        raise ValueError("features must be finite")
    mean = values.mean(dim=0)
    centered = values - mean
    covariance = centered.T @ centered / values.shape[0]
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    dimension = min(int(pca_dim), values.shape[0] - 1, values.shape[1] - 1)
    if dimension < 1:
        raise ValueError("PCA dimension is empty")
    return {
        "mean": mean.cpu(),
        "eigenvalues": eigenvalues.cpu(),
        "eigenvectors": eigenvectors.cpu(),
        "pca_dim": dimension,
        "ridge": float(ridge),
        "reference": values.to(dtype=torch.float16).cpu(),
        "num_observations": int(values.shape[0]),
    }


class XVLAMultilayerScorer:
    """Score all cached layers with PCA residual, LLMD, and Deep kNN."""

    def __init__(self, asset_path: Path, *, device: str = "cuda", knn_k: int = 10):
        payload = torch.load(asset_path, map_location="cpu", weights_only=False)
        if payload.get("format") != "xvla_airplane_multilayer_detector_assets_v1":
            raise ValueError("unexpected X-VLA detector asset format")
        self.device = torch.device(device)
        self.knn_k = int(knn_k)
        self.layers: dict[str, dict[str, Any]] = {}
        for name, source in payload["layers"].items():
            self.layers[name] = {
                key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                for key, value in source.items()
            }

    @torch.inference_mode()
    def score(self, features: Mapping[str, torch.Tensor]) -> dict[str, float]:
        result: dict[str, float] = {}
        for name, asset in self.layers.items():
            query = features[name].to(self.device, dtype=torch.float32)
            centered = query - asset["mean"].float()
            basis = asset["eigenvectors"].float()
            coordinates = centered @ basis
            pca_dim = int(asset["pca_dim"])
            residual = coordinates[:, : basis.shape[1] - pca_dim]
            llmd = (
                coordinates.square()
                / (asset["eigenvalues"].float() + float(asset["ridge"]))
            ).sum(dim=-1).sqrt()
            reference = asset["reference"]
            distances = (reference.float() - query[:, None, :]).square().sum(dim=-1).sqrt()
            k = min(self.knn_k, distances.shape[1])
            knn = distances.topk(k, largest=False, dim=-1).values.mean(dim=-1)
            result[f"{name}_pca"] = float(residual.norm(dim=-1)[0].item())
            result[f"{name}_llmd"] = float(llmd[0].item())
            result[f"{name}_knn"] = float(knn[0].item())
        return result


def trajectory_score_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert evaluator timelines to the shared airplane metric protocol."""

    converted = []
    for row in rows:
        methods = sorted(
            {
                name
                for point in row.get("timeline", [])
                for name, value in point.get("scores", {}).items()
                if value is not None and np.isfinite(value)
            }
        )
        converted.append(
            {
                **row,
                "scores": {
                    method: [
                        float(point["scores"][method])
                        for point in row["timeline"]
                        if point.get("scores", {}).get(method) is not None
                        and np.isfinite(point["scores"][method])
                    ]
                    for method in methods
                },
            }
        )
    return converted
