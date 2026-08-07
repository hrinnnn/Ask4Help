"""Online scoring of detector assets during an OpenVLA rollout."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from .detectors import mahalanobis_score, residual_score


class DetectorBank:
    def __init__(self, assets_dir: Path, device: int = 0):
        manifest = json.loads((assets_dir / "manifest.json").read_text())
        self.assets_dir = assets_dir
        self.methods = {}
        self.knn = {}
        for name in manifest["methods"]:
            method_dir = assets_dir / name
            meta = json.loads((method_dir / "asset.json").read_text())
            asset = {key: np.load(method_dir / value) if isinstance(value, str) and value.endswith(".npy") else value for key, value in meta.items()}
            self.methods[name] = (meta["kind"], asset)
            if meta["kind"] == "knn":
                self.knn[name] = NearestNeighbors(n_neighbors=int(asset["k"])).fit(asset["reference"])

    def _score(self, name: str, values: np.ndarray) -> float:
        kind, asset = self.methods[name]
        if kind == "residual_pca":
            return float(residual_score(values[None], asset)[0])
        if kind == "mahalanobis":
            return float(mahalanobis_score(values[None], asset)[0])
        if kind == "knn":
            return float(self.knn[name].kneighbors(values[None], return_distance=True)[0].mean())
        if kind == "pca_kmeans":
            centered = values.astype(np.float32) - asset["pca_mean"]
            embedding = centered @ asset["pca_components"].T
            return float(np.linalg.norm(embedding[None] - asset["centers"], axis=1).min())
        raise ValueError(f"Online scoring is not implemented for {name}: {kind}")

    @torch.inference_mode()
    def score(self, model, inputs: dict) -> dict[str, float]:
        output = model(
            input_ids=inputs["input_ids"],
            attention_mask=torch.ones_like(inputs["input_ids"]),
            pixel_values=inputs["pixel_values"],
            output_hidden_states=True,
            output_projector_features=True,
            return_dict=True,
        )
        projector = output.projector_features[0].float().mean(dim=0).cpu().numpy()
        hidden = output.hidden_states
        patch_count = int(output.projector_features.shape[1])
        core = model.get_base_model() if hasattr(model, "get_base_model") else model
        if core.vision_backbone.use_fused_vision_backbone:
            if isinstance(inputs["pixel_values"], dict):
                values = list(inputs["pixel_values"].values())
                dino_input, siglip_input = values[0], values[-1]
            else:
                dino_input, siglip_input = torch.split(inputs["pixel_values"], [3, 3], dim=1)
            dino_values = core.vision_backbone.featurizer(dino_input)[0].float().mean(dim=0).cpu().numpy()
            siglip_values = core.vision_backbone.fused_featurizer(siglip_input)[0].float().mean(dim=0).cpu().numpy()
        else:
            vision_values = core.vision_backbone(inputs["pixel_values"])[0].float().mean(dim=0).cpu().numpy()
            dino_values = siglip_values = vision_values
        result = {}
        for name in self.methods:
            if name.startswith("vlm_input_pooled_"):
                result[name] = self._score(name, projector)
            elif name == "dino_pooled_residual_pca":
                result[name] = self._score(name, dino_values)
            elif name == "siglip_pooled_residual_pca":
                result[name] = self._score(name, siglip_values)
            elif name.startswith("llama_layer_") and name.endswith("_residual_pca"):
                layer = int(name.split("_")[2])
                values = hidden[layer][0, 1 : 1 + patch_count].float().mean(dim=0).cpu().numpy()
                result[name] = self._score(name, values)
            elif name.startswith("llama_layer_") and name.endswith("_llmd"):
                layer = int(name.split("_")[2])
                values = hidden[layer][0, 1 : 1 + patch_count].float().mean(dim=0).cpu().numpy()
                result[name] = self._score(name, values)
        return result
