"""Online scoring of detector assets during an OpenVLA rollout."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from .dataset import AIRPLANE_INSTRUCTION
from .detectors import mahalanobis_score, residual_score


class DetectorBank:
    def __init__(self, assets_dir: Path, device: int = 0, external_assets: Path | None = None, action_samples: int = 1):
        manifest = json.loads((assets_dir / "manifest.json").read_text())
        self.assets_dir = assets_dir
        self.methods = {}
        self.knn = {}
        self.external = None
        self.device = device
        self.action_samples = int(action_samples)
        if self.action_samples < 1:
            raise ValueError("action_samples must be positive")
        for name in manifest["methods"]:
            method_dir = assets_dir / name
            meta = json.loads((method_dir / "asset.json").read_text())
            asset = {key: np.load(method_dir / value) if isinstance(value, str) and value.endswith(".npy") else value for key, value in meta.items()}
            self.methods[name] = (meta["kind"], asset)
            if meta["kind"] == "knn":
                self.knn[name] = NearestNeighbors(n_neighbors=int(asset["k"])).fit(asset["reference"])
        if external_assets is not None:
            self.external = ExternalVisualDetectors(external_assets, device)

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
        if kind == "scalar":
            return float(values.reshape(-1)[0] * float(asset["direction"]))
        raise ValueError(f"Online scoring is not implemented for {name}: {kind}")

    @torch.inference_mode()
    def _sample_consistency(self, model, image, seed: int) -> tuple[dict[str, float], float]:
        if self.action_samples == 1:
            return {}, 0.0
        started = perf_counter()
        image_tensor = torch.as_tensor(np.asarray(image), device=self.device).unsqueeze(0)
        image_tensor = image_tensor.repeat(self.action_samples, 1, 1, 1)
        device_index = torch.device(f"cuda:{self.device}").index
        with torch.random.fork_rng(devices=[device_index]):
            torch.manual_seed(seed)
            actions, _ = model.predict_action_batch(
                env_obs={
                    "main_images": image_tensor,
                    "task_descriptions": [AIRPLANE_INSTRUCTION] * self.action_samples,
                },
                calculate_logprobs=False,
                calculate_values=False,
                do_sample=True,
                max_new_tokens=8,
                use_cache=True,
            )
        values = actions[:, 0].detach().cpu().numpy().astype(np.float32)
        variance = np.var(values, axis=0)
        return {
            "c10_action_total_variance": float(variance.sum()),
            "c10_arm_joint_variance": float(variance[:7].sum()),
            "c10_gripper_variance": float(variance[7]),
        }, (perf_counter() - started) * 1000.0

    def score(self, model, inputs: dict, image=None, sample_seed: int = 0) -> tuple[dict[str, float], dict[str, float]]:
        started = perf_counter()
        forward_inputs = {key: value.clone() if torch.is_tensor(value) else value for key, value in inputs.items()}
        output = model(
            forward_inputs=forward_inputs,
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
        action_tokens = inputs["action_tokens"].reshape(inputs["action_tokens"].shape[0], -1)
        action_logits = output.logits[:, -action_tokens.shape[1] - 1 : -1].float()
        core = model.get_base_model() if hasattr(model, "get_base_model") else model
        action_logits[..., : core.vocab_size - core.config.n_action_bins] = -torch.inf
        action_logits[..., core.vocab_size :] = -torch.inf
        log_probs = action_logits.log_softmax(dim=-1)
        generated_logprob = log_probs.gather(-1, action_tokens.unsqueeze(-1)).mean().item()
        generated_entropy = (-(log_probs.exp() * log_probs).nan_to_num().sum(dim=-1)).mean().item()
        for name in self.methods:
            if name.startswith("vlm_input_pooled_"):
                result[name] = self._score(name, projector)
            elif name.startswith("dino_pooled_"):
                result[name] = self._score(name, dino_values)
            elif name.startswith("siglip_pooled_"):
                result[name] = self._score(name, siglip_values)
            elif name.startswith("llama_visual_layer_"):
                block = int(name.split("_")[3])
                values = hidden[block][0, 1 : 1 + patch_count].float().mean(dim=0).cpu().numpy()
                result[name] = self._score(name, values)
            elif name.startswith("action_layer_"):
                block = int(name.split("_")[2])
                values = hidden[block][0, -action_tokens.shape[1] :].float().mean(dim=0).cpu().numpy()
                result[name] = self._score(name, values)
            elif name.startswith("prompt_layer_"):
                block = int(name.split("_")[2])
                values = hidden[block][0, -action_tokens.shape[1] - 1].float().cpu().numpy()
                result[name] = self._score(name, values)
            elif name == "action_logprob":
                result[name] = self._score(name, np.asarray([generated_logprob], dtype=np.float32))
            elif name == "action_entropy":
                result[name] = self._score(name, np.asarray([generated_entropy], dtype=np.float32))
        latency = {"openvla_internal_total_ms": (perf_counter() - started) * 1000.0}
        if self.external is not None:
            if image is None:
                raise ValueError("external visual detectors require the policy image")
            external_scores, external_latency = self.external.score(image)
            result.update(external_scores)
            latency.update(external_latency)
        sampled_scores, sampled_latency = self._sample_consistency(model, image, sample_seed)
        result.update(sampled_scores)
        if sampled_scores:
            latency["c10_sampling_total_ms"] = sampled_latency
        return result, latency


class ExternalVisualDetectors:
    """Official FIDeL representation score and deployable CRSAIL vision score."""

    def __init__(self, asset_path: Path, device: int):
        from torchvision.models import ResNet18_Weights, resnet18
        from tools.pick_single_ycb_airplane_external_detectors import CRSAILBank, OfficialFIDeLMemory

        payload = torch.load(asset_path, map_location="cpu", weights_only=False)
        self.fidel = OfficialFIDeLMemory.from_state_dict(payload["fidel"]["memory"])
        self.crsail = CRSAILBank.from_state_dict(payload["crsail"]["vision_resnet18"])
        self.device = torch.device(f"cuda:{device}")
        weights = ResNet18_Weights.DEFAULT
        self.preprocess = weights.transforms()
        self.encoder = resnet18(weights=weights)
        self.encoder.fc = torch.nn.Identity()
        self.encoder = self.encoder.eval().to(self.device)
        self.fidel = OfficialFIDeLMemory(self.fidel.mean.to(self.device))
        self.crsail = CRSAILBank(
            values=self.crsail.values.to(self.device),
            k=self.crsail.k,
            center=None if self.crsail.center is None else self.crsail.center.to(self.device),
            scale=None if self.crsail.scale is None else self.crsail.scale.to(self.device),
        )

    @torch.inference_mode()
    def score(self, image) -> tuple[dict[str, float], dict[str, float]]:
        from tools.pick_single_ycb_airplane_external_detectors import crsail_score, official_fidel_euclidean_score

        started = perf_counter()
        values = torch.as_tensor(np.asarray(image), device=self.device)
        values = values.permute(2, 0, 1).float().div(255.0).unsqueeze(0)
        features = self.encoder(self.preprocess(values))
        encoded_ms = (perf_counter() - started) * 1000.0
        fidel_started = perf_counter()
        fidel = official_fidel_euclidean_score(features.unsqueeze(1), self.fidel)[0].item()
        fidel_ms = (perf_counter() - fidel_started) * 1000.0
        crsail_started = perf_counter()
        crsail = crsail_score(features, self.crsail, cosine=True).item()
        crsail_ms = (perf_counter() - crsail_started) * 1000.0
        return (
            {"fidel_official": float(fidel), "crsail_vision_k5": float(crsail)},
            {
                "external_resnet18_shared_ms": encoded_ms,
                "fidel_official_score_ms": fidel_ms,
                "crsail_vision_k5_score_ms": crsail_ms,
            },
        )
