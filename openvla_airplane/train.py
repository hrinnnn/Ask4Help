"""Official OpenVLA LoRA training loop backed by the airplane LeRobot data."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import ColorJitter, Compose, RandomResizedCrop
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModelForVision2Seq, AutoProcessor

from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer

from .dataset import AirplaneDataset, build_example, compute_action_stats, validate_lerobot_dataset
from .utils import move_pixel_values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vla-path", default="openvla/openvla-7b")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--new-data-dir", type=Path)
    parser.add_argument("--action-stats", type=Path)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--image-aug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size > 1


class AirplaneCollator:
    def __init__(self, processor, action_tokenizer, stats: dict, image_aug: bool):
        self.processor = processor
        self.action_tokenizer = action_tokenizer
        self.stats = stats
        self.image_aug = image_aug
        self.augmentation = (
            Compose(
                [
                    RandomResizedCrop(
                        384,
                        scale=(0.9, 0.9),
                        ratio=(1.0, 1.0),
                        interpolation=InterpolationMode.BILINEAR,
                    ),
                    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                ]
            )
            if image_aug
            else None
        )
        self.base_collator = PaddedCollatorForActionPrediction(
            processor.tokenizer.model_max_length,
            processor.tokenizer.pad_token_id,
            padding_side="right",
        )
        self.prompt_builder_fn = (
            PurePromptBuilder if "v01" not in getattr(processor, "name_or_path", "") else VicunaV15ChatPromptBuilder
        )

    def __call__(self, batch: list[dict]) -> dict:
        examples = []
        for sample in batch:
            transformed_sample = sample
            if self.augmentation is not None:
                transformed_sample = dict(sample)
                transformed_sample["image"] = self.augmentation(sample["image"])
            examples.append(
                build_example(
                    transformed_sample,
                    self.processor.tokenizer,
                    self.action_tokenizer,
                    self.processor.image_processor.apply_transform,
                    self.stats,
                    self.prompt_builder_fn,
                )
            )
        return self.base_collator(examples)


class SourceBalancedBatchSampler(Sampler[list[int]]):
    """Draw exactly half of every micro-batch from ID replay and new expert data."""

    def __init__(self, id_size: int, new_size: int, batch_size: int, seed: int):
        if batch_size < 2 or batch_size % 2:
            raise ValueError("source-balanced batch size must be a positive even number")
        if id_size < 1 or new_size < 1:
            raise ValueError("both source datasets must be non-empty")
        self.id_size = id_size
        self.new_size = new_size
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        half = self.batch_size // 2
        return max(1, (max(self.id_size, self.new_size) + half - 1) // half)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        half = self.batch_size // 2
        for _ in range(len(self)):
            id_indices = torch.randint(self.id_size, (half,), generator=generator).tolist()
            new_indices = (torch.randint(self.new_size, (half,), generator=generator) + self.id_size).tolist()
            order = torch.randperm(self.batch_size, generator=generator).tolist()
            combined = id_indices + new_indices
            yield [combined[index] for index in order]


def save_checkpoint(
    model,
    processor,
    optimizer,
    run_dir: Path,
    stats: dict,
    config: dict,
    step: int,
    metrics: dict,
) -> None:
    checkpoint_dir = run_dir / f"checkpoint_{step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    adapter = checkpoint_dir / "adapter"
    unwrapped = model.module if hasattr(model, "module") else model
    # OSSFS does not support the mmap-based safetensors writer. PEFT's PyTorch
    # serialization preserves the same adapter state without requiring mmap.
    unwrapped.save_pretrained(adapter, safe_serialization=False)
    processor.save_pretrained(checkpoint_dir / "processor")
    (checkpoint_dir / "action_stats.json").write_text(json.dumps(stats, indent=2))
    (checkpoint_dir / "train_config.json").write_text(json.dumps(config, indent=2, default=str))
    (checkpoint_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    torch.save({"step": step, "optimizer": optimizer.state_dict()}, checkpoint_dir / "optimizer.pt")


def main() -> None:
    args = parse_args()
    rank, local_rank, distributed = setup_distributed()
    is_main = rank == 0
    torch.manual_seed(args.seed + rank)
    if is_main:
        validate_lerobot_dataset(args.data_dir, expected_episodes=98, expected_frames=9109)
        if args.new_data_dir is not None:
            validate_lerobot_dataset(args.new_data_dir)
        args.run_dir.mkdir(parents=True, exist_ok=True)
        stats = (
            json.loads(args.action_stats.read_text(encoding="utf-8"))
            if args.action_stats is not None
            else compute_action_stats(args.data_dir)
        )
        (args.run_dir / "action_stats.json").write_text(json.dumps(stats, indent=2))
    else:
        stats = None
    if distributed:
        dist.barrier()
        stats = json.loads((args.run_dir / "action_stats.json").read_text())

    processor_source = (
        args.init_checkpoint / "processor"
        if args.init_checkpoint is not None and (args.init_checkpoint / "processor").exists()
        else args.vla_path
    )
    processor = AutoProcessor.from_pretrained(processor_source, trust_remote_code=True)
    base_model = AutoModelForVision2Seq.from_pretrained(
        args.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if args.init_checkpoint is None:
        lora = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        model = get_peft_model(base_model, lora)
    else:
        model = PeftModel.from_pretrained(base_model, args.init_checkpoint / "adapter", is_trainable=True)
    model = model.to(local_rank)
    if is_main:
        model.print_trainable_parameters()
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True) if distributed else model

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable, lr=args.learning_rate)
    id_dataset = AirplaneDataset(args.data_dir)
    if args.new_data_dir is None:
        dataset = id_dataset
        sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed) if distributed else None
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            collate_fn=AirplaneCollator(processor, ActionTokenizer(processor.tokenizer), stats, args.image_aug),
            num_workers=0,
            drop_last=True,
        )
    else:
        if distributed:
            raise ValueError("source-balanced continuation uses one process per model")
        new_dataset = AirplaneDataset(args.new_data_dir)
        dataset = ConcatDataset([id_dataset, new_dataset])
        sampler = None
        loader = DataLoader(
            dataset,
            batch_sampler=SourceBalancedBatchSampler(len(id_dataset), len(new_dataset), args.batch_size, args.seed),
            collate_fn=AirplaneCollator(processor, ActionTokenizer(processor.tokenizer), stats, args.image_aug),
            num_workers=0,
        )
    config = vars(args).copy()
    config["global_batch_size"] = (
        args.batch_size
        * (dist.get_world_size() if distributed else 1)
        * args.gradient_accumulation_steps
    )
    config["source_mix"] = "1:1 ID replay:new expert" if args.new_data_dir is not None else "ID only"
    config["optimizer_state"] = "reset"
    config["trainable_scope"] = "LoRA all-linear; OpenVLA vision backbone and base weights frozen"
    config["action_dim"] = 8
    config["camera_input"] = "base camera only"
    config["quantization"] = False
    log_path = args.run_dir / "train_metrics.jsonl"
    step = 0
    micro_step = 0
    epoch = 0
    patch_count = None
    model.train()
    optimizer.zero_grad(set_to_none=True)
    while step < args.max_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch += 1
        for batch in loader:
            if step >= args.max_steps:
                break
            pixel_values = move_pixel_values(batch["pixel_values"], local_rank)
            input_ids = batch["input_ids"].to(local_rank)
            attention_mask = batch["attention_mask"].to(local_rank)
            labels = batch["labels"].to(local_rank)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, labels=labels)
                loss = output.loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step + 1}: {loss.item()}")
            (loss / args.gradient_accumulation_steps).backward()
            micro_step += 1
            if micro_step % args.gradient_accumulation_steps:
                continue
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if patch_count is None:
                with torch.no_grad():
                    patch_count = int(
                        model.module.vision_backbone(pixel_values).shape[1]
                        if distributed
                        else model.vision_backbone(pixel_values).shape[1]
                    )
            action_logits = output.logits[:, patch_count:-1]
            action_gt = labels[:, 1:]
            action_tokenizer = ActionTokenizer(processor.tokenizer)
            mask = action_gt > action_tokenizer.action_token_begin_idx
            token_accuracy = float(((action_logits.argmax(-1) == action_gt) & mask).sum() / mask.sum().clamp_min(1))
            action_preds = action_logits.argmax(-1)
            predicted_continuous = torch.as_tensor(
                action_tokenizer.decode_token_ids_to_actions(action_preds[mask].detach().cpu().numpy())
            )
            target_continuous = torch.as_tensor(
                action_tokenizer.decode_token_ids_to_actions(action_gt[mask].detach().cpu().numpy())
            )
            metrics = {
                "step": step,
                "loss": float(loss.item()),
                "token_accuracy": token_accuracy,
                "action_l1": float(torch.nn.functional.l1_loss(predicted_continuous, target_continuous).item()),
                "epoch": epoch,
                "time": time.time(),
            }
            if is_main:
                with log_path.open("a") as handle:
                    handle.write(json.dumps(metrics) + "\n")
                if step == 1 or step % 10 == 0:
                    print(json.dumps(metrics), flush=True)
            if step % args.save_steps == 0 or step == args.max_steps:
                if distributed:
                    dist.barrier()
                if is_main:
                    save_checkpoint(model, processor, optimizer, args.run_dir, stats, config, step, metrics)
                if distributed:
                    dist.barrier()
            if args.smoke_steps and step >= args.smoke_steps:
                break
        if args.smoke_steps and step >= args.smoke_steps:
            break
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
