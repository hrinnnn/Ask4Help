"""Official OpenVLA LoRA training loop backed by the airplane LeRobot data."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
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
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size > 1


class AirplaneCollator:
    def __init__(self, processor, action_tokenizer, stats: dict, image_aug: bool):
        self.processor = processor
        self.action_tokenizer = action_tokenizer
        self.stats = stats
        self.image_aug = image_aug
        self.base_collator = PaddedCollatorForActionPrediction(
            processor.tokenizer.model_max_length,
            processor.tokenizer.pad_token_id,
            padding_side="right",
        )
        self.prompt_builder_fn = (
            PurePromptBuilder if "v01" not in getattr(processor, "name_or_path", "") else VicunaV15ChatPromptBuilder
        )

    def __call__(self, batch: list[dict]) -> dict:
        examples = [
            build_example(
                sample,
                self.processor.tokenizer,
                self.action_tokenizer,
                self.processor.image_processor.apply_transform,
                self.stats,
                self.prompt_builder_fn,
            )
            for sample in batch
        ]
        return self.base_collator(examples)


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
    model.module.save_pretrained(adapter)
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
        validate_lerobot_dataset(args.data_dir)
        args.run_dir.mkdir(parents=True, exist_ok=True)
        stats = compute_action_stats(args.data_dir)
        (args.run_dir / "action_stats.json").write_text(json.dumps(stats, indent=2))
    else:
        stats = None
    if distributed:
        dist.barrier()
        stats = json.loads((args.run_dir / "action_stats.json").read_text())

    processor = AutoProcessor.from_pretrained(args.vla_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(local_rank)
    lora = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        init_lora_weights="gaussian",
    )
    model = get_peft_model(model, lora)
    if is_main:
        model.print_trainable_parameters()
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True) if distributed else model

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable, lr=args.learning_rate)
    dataset = AirplaneDataset(args.data_dir)
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
    config = vars(args).copy()
    config["global_batch_size"] = args.batch_size * (dist.get_world_size() if distributed else 1)
    config["trainable_scope"] = "LoRA all-linear; OpenVLA vision backbone and base weights frozen"
    config["action_dim"] = 8
    config["camera_input"] = "base camera only"
    config["quantization"] = False
    log_path = args.run_dir / "train_metrics.jsonl"
    step = 0
    epoch = 0
    patch_count = None
    iterator = iter(loader)
    model.train()
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
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, labels=labels)
                loss = output.loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step + 1}: {loss.item()}")
            loss.backward()
            optimizer.step()
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
            metrics = {
                "step": step,
                "loss": float(loss.item()),
                "token_accuracy": token_accuracy,
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
