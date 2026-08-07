"""Cache OpenVLA visual, language, and action-token representations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.models.backbones.llm.prompting import PurePromptBuilder

from .dataset import AirplaneDataset, build_example, compute_action_stats
from .model import load_openvla
from .utils import move_pixel_values


class FeatureCollator:
    def __init__(self, processor, stats):
        self.processor = processor
        self.stats = stats
        self.action_tokenizer = ActionTokenizer(processor.tokenizer)
        self.base = PaddedCollatorForActionPrediction(
            processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
        )

    def __call__(self, samples):
        examples = [
            build_example(
                sample,
                self.processor.tokenizer,
                self.action_tokenizer,
                self.processor.image_processor.apply_transform,
                self.stats,
                PurePromptBuilder,
            )
            for sample in samples
        ]
        batch = self.base(examples)
        batch["episode_index"] = [sample["episode_index"] for sample in samples]
        batch["frame_index"] = [sample["frame_index"] for sample in samples]
        return batch


def _allocate(path: Path, shape: tuple[int, ...]) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)


def _core(model):
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def _pool(value: torch.Tensor) -> torch.Tensor:
    return value.float().mean(dim=1)


def extract(data_dir: Path, checkpoint: Path, base_path: str, output: Path, batch_size: int, device: int) -> dict:
    dataset = AirplaneDataset(data_dir)
    stats = compute_action_stats(data_dir)
    model, processor = load_openvla(base_path, checkpoint, device)
    collator = FeatureCollator(processor, stats)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator, num_workers=0)
    core = _core(model)
    hidden_dim = int(core.config.text_config.hidden_size)
    layer_count = int(core.config.text_config.num_hidden_layers)
    output.mkdir(parents=True, exist_ok=True)
    n = len(dataset)
    arrays = {
        "dino_pooled": _allocate(output / "dino_pooled.npy", (n, int(core.vision_backbone.featurizer.embed_dim))),
        "siglip_pooled": _allocate(output / "siglip_pooled.npy", (n, int(core.vision_backbone.fused_featurizer.embed_dim) if core.vision_backbone.use_fused_vision_backbone else int(core.vision_backbone.featurizer.embed_dim))),
        "projector_pooled": _allocate(output / "projector_pooled.npy", (n, hidden_dim)),
        "llama_visual_pooled": _allocate(output / "llama_visual_pooled.npy", (n, layer_count, hidden_dim)),
        "llama_action_pooled": _allocate(output / "llama_action_pooled.npy", (n, layer_count, hidden_dim)),
        "prompt_decision": _allocate(output / "prompt_decision.npy", (n, layer_count, hidden_dim)),
        "action_logprob": _allocate(output / "action_logprob.npy", (n,)),
        "action_entropy": _allocate(output / "action_entropy.npy", (n,)),
    }
    episode_index = np.empty(n, dtype=np.int32)
    frame_index = np.empty(n, dtype=np.int32)
    patch_count = None
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    with torch.inference_mode():
        cursor = 0
        for batch in loader:
            pixel_values = move_pixel_values(batch["pixel_values"], device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            output_model = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                output_hidden_states=True,
                output_projector_features=True,
                return_dict=True,
            )
            projector = output_model.projector_features
            if patch_count is None:
                patch_count = int(projector.shape[1])
            core_features = core.vision_backbone(pixel_values)
            if core.vision_backbone.use_fused_vision_backbone:
                if isinstance(pixel_values, dict):
                    dino_input = pixel_values.get("dino", pixel_values.get("image"))
                    siglip_input = pixel_values.get("siglip", pixel_values.get("image"))
                else:
                    dino_input, siglip_input = torch.split(pixel_values, [3, 3], dim=1)
                dino = core.vision_backbone.featurizer(dino_input)
                siglip = core.vision_backbone.fused_featurizer(siglip_input)
            else:
                dino = core_features
                siglip = core_features
            hidden_states = output_model.hidden_states
            batch_size_actual = input_ids.shape[0]
            for local in range(batch_size_actual):
                length = int(attention_mask[local].sum().item())
                action_start = patch_count + length - 8
                visual_slice = slice(1, 1 + patch_count)
                action_slice = slice(action_start, action_start + 8)
                prompt_index = min(patch_count + 1, hidden_states[0].shape[1] - 1)
                for layer in range(layer_count):
                    state = hidden_states[layer + 1][local]
                    arrays["llama_visual_pooled"][cursor + local, layer] = _pool(state[visual_slice].unsqueeze(0))[0].cpu().numpy()
                    arrays["llama_action_pooled"][cursor + local, layer] = _pool(state[action_slice].unsqueeze(0))[0].cpu().numpy()
                    arrays["prompt_decision"][cursor + local, layer] = state[prompt_index].float().cpu().numpy()
                arrays["dino_pooled"][cursor + local] = _pool(dino[local].unsqueeze(0))[0].cpu().numpy()
                arrays["siglip_pooled"][cursor + local] = _pool(siglip[local].unsqueeze(0))[0].cpu().numpy()
                arrays["projector_pooled"][cursor + local] = _pool(projector[local].unsqueeze(0))[0].cpu().numpy()
                target = batch["labels"][local, -8:].to(device)
                logits = output_model.logits[local, patch_count + length - 9 : patch_count + length - 1]
                log_probs = logits.log_softmax(-1)
                arrays["action_logprob"][cursor + local] = float(log_probs.gather(-1, target.unsqueeze(-1)).mean().item())
                probs = log_probs.exp()
                arrays["action_entropy"][cursor + local] = float((-probs * log_probs).sum(-1).mean().item())
                episode_index[cursor + local] = batch["episode_index"][local]
                frame_index[cursor + local] = batch["frame_index"][local]
            cursor += batch_size_actual
    for array in arrays.values():
        array.flush()
    np.save(output / "episode_index.npy", episode_index)
    np.save(output / "frame_index.npy", frame_index)
    manifest = {
        "source": str(data_dir),
        "checkpoint": str(checkpoint),
        "observations": n,
        "layers": layer_count,
        "hidden_dim": hidden_dim,
        "projector_tokens": patch_count,
        "action_dim": 8,
        "camera": "base camera only",
        "feature_files": {key: str(path.name) for key, path in ((key, output / f"{key}.npy") for key in arrays)},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-path", default="openvla/openvla-7b")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(extract(args.data_dir, args.checkpoint, args.base_path, args.output, args.batch_size, args.device), indent=2))


if __name__ == "__main__":
    main()
