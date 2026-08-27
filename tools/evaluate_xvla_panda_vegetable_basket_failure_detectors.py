#!/usr/bin/env python3
"""Run passive Panda basket rollouts and record every detector score."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.libero_plus_failure_protocol import single_sample_overlap_score  # noqa: E402
from tools.pick_single_ycb_airplane_external_detectors import (  # noqa: E402
    CRSAILBank,
    OfficialFIDeLMemory,
    crsail_score,
    official_fidel_euclidean_score,
)
from tools.panda_vegetable_basket_adapter import (  # noqa: E402
    encode_base_ee6d,
    model_target_to_panda_action,
    tcp_pose_world,
    world_pose_to_base,
)
from tools.xvla_airplane_failure_detection import (  # noqa: E402
    XVLAMultilayerProbe,
    XVLAMultilayerScorer,
)


TASK = "put the vegetable into the yellow basket"
ENV_IDS = {
    "id": "XVLAPandaPutVegetableInBasketID-v1",
    "ood": "XVLAPandaPutVegetableInBasketOOD-v1",
}


def scalar(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0].item())
    return float(np.asarray(value).reshape(-1)[0])


def bool_scalar(value) -> bool:
    return bool(scalar(value))


def array(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def pose(actor) -> np.ndarray:
    return array(actor.pose.raw_pose).reshape(-1, 7)[0].astype(np.float32)


def rgb(obs) -> np.ndarray:
    value = array(obs["sensor_data"]["3rd_view_camera"]["rgb"])
    return value[0].astype(np.uint8) if value.ndim == 4 else value.astype(np.uint8)


def current_gripper(env) -> float:
    qpos = array(env.unwrapped.agent.robot.get_qpos()).reshape(-1)
    return float(np.clip((float(np.mean(qpos[-2:])) + 0.01) / 0.05, 0.0, 1.0))


def load_task_module(path: Path) -> None:
    spec = importlib.util.spec_from_file_location("panda_vegetable_basket_variants", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import task module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


class PandaXVLAPolicy:
    def __init__(self, checkpoint: Path, xvla_root: Path, device: torch.device, domain_id: int):
        sys.path.insert(0, str(xvla_root.resolve()))
        from models.modeling_xvla import XVLA
        from models.processing_xvla import XVLAProcessor

        self.device = device
        self.domain_id = domain_id
        self.model = XVLA.from_pretrained(str(checkpoint), torch_dtype=torch.bfloat16).to(device).eval()
        self.processor = XVLAProcessor.from_pretrained(str(checkpoint))

    def inputs(self, env, obs) -> dict[str, torch.Tensor]:
        encoded = self.processor.encode_image([[Image.fromarray(rgb(obs), mode="RGB")]])
        language = self.processor.encode_language([TASK])
        proprio = torch.from_numpy(
            encode_base_ee6d(world_pose_to_base(env, tcp_pose_world(env)), current_gripper(env))[None]
        )
        result = {
            **encoded,
            "input_ids": language["input_ids"],
            "domain_id": torch.tensor([self.domain_id], dtype=torch.long),
            "proprio": proprio,
        }
        return {
            key: value.to(self.device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in result.items()
        }

    def generate(self, inputs: dict[str, torch.Tensor], encoding: dict[str, torch.Tensor], steps: int) -> torch.Tensor:
        batch = inputs["input_ids"].shape[0]
        prior = torch.randn(
            batch,
            self.model.num_actions,
            self.model.action_space.dim_action,
            device=self.device,
            dtype=inputs["proprio"].dtype,
        )
        action = torch.zeros_like(prior)
        for index in range(max(1, int(steps)), 0, -1):
            time = torch.full((batch,), index / steps, device=self.device, dtype=prior.dtype)
            noisy = prior * time[:, None, None] + action * (1 - time[:, None, None])
            proprio, noisy = self.model.action_space.preprocess(inputs["proprio"], noisy)
            action = self.model.transformer(
                domain_id=inputs["domain_id"], action_with_noise=noisy, proprio=proprio, t=time, **encoding
            )
        return self.model.action_space.postprocess(action)

    def diff_score(
        self,
        inputs: dict[str, torch.Tensor],
        encoding: dict[str, torch.Tensor],
        generated: torch.Tensor,
        timesteps: int,
        noise_samples: int,
    ) -> float:
        action = self.model.action_space._pad_to_model_dim(generated)
        scores = torch.zeros(action.shape[0], device=self.device, dtype=torch.float32)
        times = (torch.arange(timesteps, device=self.device) + 0.5) / timesteps
        for _ in range(noise_samples):
            for value in times:
                noise = torch.randn_like(action)
                time = value.to(action.dtype).expand(action.shape[0])
                noisy = noise * time[:, None, None] + action * (1 - time[:, None, None])
                proprio, noisy = self.model.action_space.preprocess(inputs["proprio"], noisy)
                prediction = self.model.transformer(
                    domain_id=inputs["domain_id"], action_with_noise=noisy, proprio=proprio, t=time, **encoding
                )
                scores += (prediction[..., :8].float() - action[..., :8].float()).square().flatten(1).mean(1)
        return float((scores / (timesteps * noise_samples))[0].item())


def save_video(path: Path, frames: list[np.ndarray]) -> None:
    if not frames:
        raise RuntimeError("empty video")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video {path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def external_features(encoder, preprocess, image: np.ndarray, device: torch.device) -> torch.Tensor:
    with torch.inference_mode():
        return encoder(preprocess(Image.fromarray(image, mode="RGB")).unsqueeze(0).to(device))[0]


def run_episode(
    *,
    policy: PandaXVLAPolicy,
    probe: XVLAMultilayerProbe,
    scorer: XVLAMultilayerScorer,
    external_encoder,
    external_preprocess,
    fidel: OfficialFIDeLMemory | None,
    crsail_state: CRSAILBank | None,
    crsail_vision: CRSAILBank | None,
    task_module: Path,
    split: str,
    seed: int,
    index: int,
    output: Path,
    flow_steps: int,
    execute_horizon: int,
    max_episode_steps: int,
    diff_timesteps: int,
    diff_noise_samples: int,
) -> dict:
    del task_module
    env = gym.make(
        ENV_IDS[split],
        obs_mode="rgb+segmentation",
        render_mode="rgb_array",
        sim_backend="physx_cpu",
        control_mode="pd_ee_body_target_delta_pose_real",
        max_episode_steps=max_episode_steps,
    )
    try:
        obs, _ = env.reset(seed=int(seed))
        base = env.unwrapped
        frames = [rgb(obs)]
        commands: list[np.ndarray] = []
        states: list[np.ndarray] = []
        timeline: list[dict] = []
        previous_points = None
        ever_grasped = False
        ever_on_target = False
        terminated = False
        while len(commands) < max_episode_steps and not terminated:
            torch.manual_seed(int(seed * 1000 + len(commands)))
            inputs = policy.inputs(env, obs)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                encoding = policy.model.forward_vlm(inputs["input_ids"], inputs["image_input"], inputs["image_mask"])
                features, _ = probe.extract(inputs)
                generated = policy.generate(inputs, encoding, flow_steps)
                scores = scorer.score(features)
                image_features = external_features(external_encoder, external_preprocess, rgb(obs), policy.device)
                if fidel is not None:
                    fidel_value, fidel_time = official_fidel_euclidean_score(image_features.unsqueeze(1), fidel)
                    scores["fidel_official"] = float(fidel_value[0].item())
                    matched_time = int(fidel_time[0].item())
                else:
                    matched_time = None
                if crsail_state is not None:
                    scores["crsail_observable_state_k5"] = float(
                        crsail_score(inputs["proprio"][0, :10], crsail_state)[0].item()
                    )
                if crsail_vision is not None:
                    scores["crsail_vision_k5"] = float(
                        crsail_score(image_features.unsqueeze(0), crsail_vision, cosine=True)[0].item()
                    )
                generated_np = generated.float().cpu().numpy()[0]
                diff_value = policy.diff_score(inputs, encoding, generated, diff_timesteps, diff_noise_samples)
            points = generated_np[:, :3]
            acc = None
            stac = None
            if previous_points is not None:
                if len(points) >= 3:
                    acc = float(np.linalg.norm(np.diff(points, n=2, axis=0), axis=1).mean())
                if len(points) > execute_horizon:
                    stac = single_sample_overlap_score(previous_points, points, execute_horizon=execute_horizon)
            scores["diffdagger"] = diff_value
            scores["acc"] = acc
            scores["stac_single"] = stac
            timeline.append(
                {
                    "decision_index": len(timeline),
                    "env_step": len(commands),
                    "scores": scores,
                    "action_chunk": generated_np[:, :10].tolist(),
                    "acc_definition": "mean second difference of predicted absolute xyz chunk",
                    "stac_definition": "single-sample overlap of predicted absolute xyz chunks",
                    "fidel_matched_time": matched_time,
                }
            )
            previous_points = points
            for model_action in generated_np[:execute_horizon, :10]:
                command = model_target_to_panda_action(env, model_action)
                obs, _reward, term, trunc, _info = env.step(command)
                command = np.asarray(command, dtype=np.float32).reshape(-1)
                commands.append(command)
                frames.append(rgb(obs))
                source_pose = pose(base.objs[base.source_obj_name])
                target_pose = pose(base.objs[base.target_obj_name])
                eval_result = base.evaluate()
                grasped = bool_scalar(base.agent.is_grasping(base.objs[base.source_obj_name]))
                on_target = bool_scalar(eval_result["src_on_target"])
                ever_grasped |= grasped
                ever_on_target |= on_target
                states.append(
                    np.concatenate(
                        [source_pose, target_pose, tcp_pose_world(env), [grasped, on_target]]
                    ).astype(np.float32)
                )
                terminated = bool_scalar(term) or bool_scalar(trunc) or bool_scalar(eval_result["success"])
                if terminated or len(commands) >= max_episode_steps:
                    break
        final_eval = base.evaluate()
        strict_success = bool_scalar(final_eval["success"])
        output.joinpath("videos").mkdir(parents=True, exist_ok=True)
        output.joinpath("actions").mkdir(parents=True, exist_ok=True)
        output.joinpath("states").mkdir(parents=True, exist_ok=True)
        output.joinpath("timelines").mkdir(parents=True, exist_ok=True)
        output.joinpath("reset_metadata").mkdir(parents=True, exist_ok=True)
        stem = f"episode_{index:06d}"
        video_path = output / "videos" / f"{stem}.mp4"
        action_path = output / "actions" / f"{stem}.npy"
        state_path = output / "states" / f"{stem}.npy"
        timeline_path = output / "timelines" / f"{stem}.json"
        reset_path = output / "reset_metadata" / f"{stem}.json"
        save_video(video_path, frames)
        np.save(action_path, np.asarray(commands, dtype=np.float32))
        np.save(state_path, np.asarray(states, dtype=np.float32))
        source = base.objs[base.source_obj_name]
        target = base.objs[base.target_obj_name]
        reset_metadata = {
            "seed": int(seed),
            "split": split,
            "robot": type(base.agent).__name__,
            "source_object": base.source_obj_name,
            "target_object": base.target_obj_name,
            "source_model_scale": float(base.episode_model_scales[base.source_obj_name]),
            "target_model_scale": float(base.episode_model_scales[base.target_obj_name]),
            "configured_source_pose": array(base.xyz_configs[0, 0]).tolist(),
            "configured_target_pose": array(base.xyz_configs[0, 1]).tolist(),
            "start_source_pose": pose(source).tolist(),
            "start_target_pose": pose(target).tolist(),
        }
        reset_path.write_text(json.dumps(reset_metadata, indent=2) + "\n", encoding="utf-8")
        timeline_path.write_text(json.dumps(timeline, indent=2) + "\n", encoding="utf-8")
        return {
            "episode_index": index,
            "seed": int(seed),
            "split": split,
            "strict_success": strict_success,
            "success": strict_success,
            "ever_grasped": ever_grasped,
            "ever_on_target": ever_on_target,
            "num_actions": len(commands),
            "num_frames": len(frames),
            "execute_horizon": execute_horizon,
            "max_episode_steps": max_episode_steps,
            "video": str(video_path),
            "actions": str(action_path),
            "states": str(state_path),
            "timeline": timeline,
            "reset_metadata": str(reset_path),
            "final_eval": {
                key: bool_scalar(value) if array(value).dtype == bool else scalar(value)
                for key, value in final_eval.items()
            },
        }
    finally:
        env.close()


def move_bank(payload: dict, device: torch.device) -> CRSAILBank:
    bank = CRSAILBank.from_state_dict(payload)
    return CRSAILBank(
        values=bank.values.to(device),
        k=bank.k,
        center=None if bank.center is None else bank.center.to(device),
        scale=None if bank.scale is None else bank.scale.to(device),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--rlinf-root", type=Path, required=True)
    parser.add_argument("--task-module", type=Path, required=True)
    parser.add_argument("--multilayer-assets", type=Path, required=True)
    parser.add_argument("--external-assets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("id", "ood"), required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--domain-id", type=int, default=20)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--execute-horizon", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=150)
    parser.add_argument("--probe-steps", type=int, default=5)
    parser.add_argument("--diff-timesteps", type=int, default=16)
    parser.add_argument("--diff-noise-samples", type=int, default=1)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    sys.path.insert(0, str(args.rlinf_root.resolve()))
    load_task_module(args.task_module)
    from torchvision.models import ResNet18_Weights, resnet18

    device = torch.device("cuda")
    policy = PandaXVLAPolicy(args.checkpoint, args.xvla_root, device, args.domain_id)
    payload = torch.load(args.multilayer_assets / "multilayer_detector_assets.pt", map_location="cpu", weights_only=False)
    scorer = XVLAMultilayerScorer(args.multilayer_assets / "multilayer_detector_assets.pt", device="cuda", knn_k=10)
    probe = XVLAMultilayerProbe(policy.model, probe_seed=0, probe_steps=args.probe_steps)
    external_payload = torch.load(args.external_assets / "external_detector_assets.pt", map_location="cpu", weights_only=False)
    weights = ResNet18_Weights.DEFAULT
    encoder = resnet18(weights=weights)
    encoder.fc = torch.nn.Identity()
    encoder.eval().to(device)
    fidel = OfficialFIDeLMemory.from_state_dict(external_payload["fidel"]["memory"])
    fidel = OfficialFIDeLMemory(mean=fidel.mean.to(device))
    crsail_state = move_bank(external_payload["crsail"]["observable_state"], device)
    crsail_vision = move_bank(external_payload["crsail"]["vision_resnet18"], device)
    rows = []
    try:
        for index in range(args.episodes):
            row = run_episode(
                policy=policy,
                probe=probe,
                scorer=scorer,
                external_encoder=encoder,
                external_preprocess=weights.transforms(),
                fidel=fidel,
                crsail_state=crsail_state,
                crsail_vision=crsail_vision,
                task_module=args.task_module,
                split=args.split,
                seed=args.seed_start + index,
                index=index,
                output=args.output_dir,
                flow_steps=args.flow_steps,
                execute_horizon=args.execute_horizon,
                max_episode_steps=args.max_episode_steps,
                diff_timesteps=args.diff_timesteps,
                diff_noise_samples=args.diff_noise_samples,
            )
            rows.append(row)
            print(json.dumps({key: row[key] for key in ("episode_index", "seed", "strict_success", "ever_grasped", "num_actions")}), flush=True)
    finally:
        probe.close()
        del encoder
        torch.cuda.empty_cache()
    summary = {
        "format": "xvla_panda_vegetable_basket_failure_rollouts_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "split": args.split,
        "episodes": len(rows),
        "strict_successes": sum(int(row["strict_success"]) for row in rows),
        "ever_grasped_successes": sum(int(row["ever_grasped"]) for row in rows),
        "videos": len(list((args.output_dir / "videos").glob("*.mp4"))),
        "actions": len(list((args.output_dir / "actions").glob("*.npy"))),
        "timelines": len(list((args.output_dir / "timelines").glob("*.json"))),
        "protocol": {
            "flow_steps": args.flow_steps,
            "execute_horizon": args.execute_horizon,
            "max_episode_steps": args.max_episode_steps,
            "failure_definition": "not strict_success",
            "detector_assets_fit_split": "ID expert anchors only",
        },
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "EVALUATION_COMPLETE").write_text("complete\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
