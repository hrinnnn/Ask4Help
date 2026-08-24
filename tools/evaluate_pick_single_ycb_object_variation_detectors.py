#!/usr/bin/env python3
"""Passive failure-detection evaluation for the object-variation task."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RLINF_ROOT = ROOT / "RLinf"
sys.path[:0] = [str(ROOT), str(RLINF_ROOT)]

from rlinf.algorithms.vla_fail import (  # noqa: E402
    KNNStatistics,
    LLMDStatistics,
    PCAResidualStatistics,
    fixed_gaussian_prior,
    knn_score,
    llmd_score,
    pca_residual_score,
    velocity_normalized_acc,
)
from rlinf.envs.maniskill.pick_single_ycb_object_variation import (  # noqa: E402
    PICK_SINGLE_YCB_OBJECT_ID_ENV_ID,
    PICK_SINGLE_YCB_OBJECT_OOD_ENV_ID,
    PICK_SINGLE_YCB_OBJECT_TASK,
    register_controlled_pick_single_ycb_object_variants,
    reset_metadata,
)
from tools.evaluate_pick_single_ycb_object_variation_pi05 import _load_model  # noqa: E402
from tools.pick_single_ycb_airplane_eval_common import clip_action_chunk  # noqa: E402
from tools.pick_single_ycb_airplane_detector_protocol import summary_for_method, threshold_free_summary  # noqa: E402
from tools.pick_single_ycb_airplane_external_detectors import (  # noqa: E402
    CRSAILBank,
    OfficialFIDeLMemory,
    crsail_score,
    official_fidel_euclidean_score,
)
from tools.evaluate_stackcube_vla_fail import PandaEndEffectorProjector  # noqa: E402
from tools.libero_plus_failure.rollout_records import single_sample_overlap  # noqa: E402
from toolkits.lerobot.collect_maniskill_peg_lerobot_joint import _build_frames, _extract_record, _select_camera  # noqa: E402


def model_observation(raw_obs):
    states = raw_obs["agent"]["qpos"]
    return {"main_images": raw_obs["sensor_data"]["base_camera"]["rgb"], "wrist_images": raw_obs["sensor_data"]["hand_camera"]["rgb"], "extra_view_images": None, "states": states, "task_descriptions": [PICK_SINGLE_YCB_OBJECT_TASK], "task_ids": torch.zeros(1, dtype=torch.long, device=states.device)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pi05-base", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--detector-assets", type=Path, required=True)
    parser.add_argument("--external-detector-assets", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("id", "ood"), required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--execute-horizon", type=int, default=5)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "videos").mkdir()
    (args.output_dir / "actions").mkdir()

    import gymnasium as gym
    import imageio.v2 as imageio
    import mani_skill.envs  # noqa: F401

    register_controlled_pick_single_ycb_object_variants()
    model = _load_model(args.checkpoint, args.norm_stats, args.pi05_base)
    asset = torch.load(args.detector_assets, map_location="cpu", weights_only=False)
    stats = asset["statistics"]
    bridge_pca = PCAResidualStatistics.from_state_dict(stats["bridge_pca_residual"])
    bridge_knn = KNNStatistics.from_state_dict(stats["bridge_deep_knn"])
    bridge_llmd = LLMDStatistics.from_state_dict(stats["bridge_llmd"])
    final_llmd = LLMDStatistics.from_state_dict(stats["final_llmd"])
    action_final_pca = PCAResidualStatistics.from_state_dict(stats["action_expert_final_pca"])
    prior = asset["fixed_prior"].to("cuda")
    external = torch.load(args.external_detector_assets, map_location="cpu", weights_only=False)
    if external.get("format") != "pick_single_ycb_object_variation_external_detector_assets_v1":
        raise ValueError("wrong external detector asset format")
    fidel_memory = OfficialFIDeLMemory.from_state_dict(external["fidel"]["memory"])
    fidel_memory = OfficialFIDeLMemory(mean=fidel_memory.mean.to("cuda"))
    def move_bank(payload):
        bank = CRSAILBank.from_state_dict(payload)
        return CRSAILBank(
            values=bank.values.to("cuda"),
            k=bank.k,
            center=None if bank.center is None else bank.center.to("cuda"),
            scale=None if bank.scale is None else bank.scale.to("cuda"),
        )
    crsail_state = move_bank(external["crsail"]["observable_state"])
    crsail_vision = move_bank(external["crsail"]["vision_resnet18"])
    from torchvision.models import ResNet18_Weights, resnet18
    resnet_weights = ResNet18_Weights.DEFAULT
    external_encoder = resnet18(weights=resnet_weights)
    external_encoder.fc = torch.nn.Identity()
    external_encoder.eval().to("cuda")
    external_preprocess = resnet_weights.transforms()
    calibration = json.loads(args.calibration.read_text())
    fixed_thresholds = {"bridge_pca": float(calibration["pca_threshold"]), "diffdagger": float(calibration["diff_threshold"])}
    env_id = PICK_SINGLE_YCB_OBJECT_ID_ENV_ID if args.split == "id" else PICK_SINGLE_YCB_OBJECT_OOD_ENV_ID
    env = gym.make(env_id, num_envs=1, robot_uids="panda_wristcam", obs_mode="rgb", control_mode="pd_joint_delta_pos", reward_mode="sparse", sim_backend="physx_cpu", sim_config={"sim_freq": 100, "control_freq": 10}, render_mode="rgb_array", max_episode_steps=args.max_episode_steps)
    projector = PandaEndEffectorProjector(env, requested_link=None)
    low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    rows = []
    try:
        for ep in range(args.episodes):
            seed = args.seed + ep
            raw_obs, _ = env.reset(seed=seed)
            metadata = reset_metadata(env, split=args.split)
            records = [_extract_record(raw_obs)]
            executed = []
            traces = {"bridge_pca": [], "bridge_knn": [], "bridge_llmd": [], "final_llmd": [], "action_final_pca": [], "diffdagger": [], "fidel_official": [], "crsail_observable_state_k5": [], "crsail_vision_k5": [], "acc": [], "stac_single": []}
            previous_points = None
            previous_acc_ema = None
            success = False
            while len(executed) < args.max_episode_steps and not success:
                with torch.inference_mode():
                    env_obs = model_observation(raw_obs)
                    features = model.extract_multilayer_llmd_features(env_obs, prior, action_expert_fractions=(0.5,), capture_vlm=True, include_action_expert_final=True)
                    predicted, result = model.predict_action_batch(env_obs=env_obs, mode="eval", compute_values=False)
                    model_actions = result["forward_inputs"]["model_action"]
                    traces["bridge_pca"].append(float(pca_residual_score(features["vlm_bridge_final_mean"], bridge_pca)[0].item()))
                    traces["bridge_knn"].append(float(knn_score(features["vlm_bridge_final_mean"], bridge_knn)[0].item()))
                    traces["bridge_llmd"].append(float(llmd_score(features["vlm_bridge_final_mean"], bridge_llmd)[0].item()))
                    traces["final_llmd"].append(float(llmd_score(features["action_expert_final"], final_llmd)[0].item()))
                    traces["action_final_pca"].append(float(pca_residual_score(features["action_expert_final"], action_final_pca)[0].item()))
                    traces["diffdagger"].append(float(model.compute_diffdagger_uncertainty(env_obs, model_actions, num_timesteps=16, num_noise_samples=1).reshape(-1)[0].detach().cpu()))
                    image = torch.as_tensor(env_obs["main_images"])
                    if image.ndim == 4 and image.shape[0] == 1:
                        image = image[0]
                    if image.shape[-1] in (1, 3, 4):
                        image = image.permute(2, 0, 1)
                    image = image[:3].float()
                    if image.max() > 1.0:
                        image = image / 255.0
                    external_feature = external_encoder(external_preprocess(image.unsqueeze(0)).to("cuda"))
                    fidel_score, _ = official_fidel_euclidean_score(external_feature.unsqueeze(1), fidel_memory)
                    state_feature = torch.as_tensor(env_obs["states"], device="cuda", dtype=torch.float32)
                    traces["fidel_official"].append(float(fidel_score[0].item()))
                    traces["crsail_observable_state_k5"].append(float(crsail_score(state_feature, crsail_state)[0].item()))
                    traces["crsail_vision_k5"].append(float(crsail_score(external_feature, crsail_vision, cosine=True)[0].item()))
                full_chunk = clip_action_chunk(predicted.detach().float().cpu().numpy(), low, high, int(model.config.action_horizon))
                chunk = full_chunk[: args.execute_horizon]
                qpos = raw_obs["agent"]["qpos"].detach().cpu().numpy()[0]
                points = projector.project(qpos, full_chunk)
                acc_ema = None
                stac = None
                if previous_points is not None:
                    _acc_raw, acc_ema = velocity_normalized_acc(previous_points, points, execute_horizon=args.execute_horizon, min_velocity=1e-3, previous_ema=previous_acc_ema, ema_alpha=0.9)
                    stac = single_sample_overlap(previous_points.numpy(), points.numpy(), execute_horizon=args.execute_horizon)
                traces["acc"].append(0.0 if acc_ema is None else float(acc_ema))
                traces["stac_single"].append(0.0 if stac is None else float(stac))
                previous_points = points
                previous_acc_ema = acc_ema
                for action in chunk:
                    raw_obs, _reward, terminated, truncated, info = env.step(torch.as_tensor(action, device=env.unwrapped.device).reshape(1, -1))
                    executed.append(np.asarray(action, dtype=np.float32))
                    records.append(_extract_record(raw_obs))
                    success = bool(np.asarray(info.get("success", False)).reshape(-1)[0])
                    if success or bool(np.asarray(terminated).reshape(-1)[0]) or bool(np.asarray(truncated).reshape(-1)[0]): break
            frames = _build_frames(records=records, actions=executed, task=PICK_SINGLE_YCB_OBJECT_TASK, main_camera=_select_camera(records[0].obs, "base_camera", ("base_camera",), "main"), wrist_camera=_select_camera(records[0].obs, "hand_camera", ("hand_camera",), "wrist"))
            video = args.output_dir / "videos" / f"episode_{ep:06d}_seed_{seed:06d}.mp4"
            writer = imageio.get_writer(video, format="FFMPEG", fps=10, codec="libx264", pixelformat="yuv420p")
            try:
                for frame in frames:
                    writer.append_data(np.concatenate([frame["image"], np.full((frame["image"].shape[0], 4, 3), 32, dtype=np.uint8), frame["wrist_image"]], axis=1))
            finally: writer.close()
            action_path=args.output_dir/"actions"/f"episode_{ep:06d}_seed_{seed:06d}.npy";np.save(action_path,np.asarray(executed,dtype=np.float32))
            rows.append({"episode_index":ep,"seed":seed,"success":success,"steps":len(executed),"execute_horizon":args.execute_horizon,"scores":traces,"video":str(video),"actions":str(action_path),**metadata})
    finally:
        env.close()
        del external_encoder
        torch.cuda.empty_cache()
    metrics={"split":args.split,"episodes":len(rows),"successes":sum(int(r["success"]) for r in rows),"fixed_thresholds":fixed_thresholds,"threshold_free":{m:threshold_free_summary(rows,m) for m in traces},"fixed":{m:summary_for_method(rows,m,fixed_thresholds[m]) for m in ("bridge_pca","diffdagger")}}
    (args.output_dir/"episodes.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows),encoding="utf-8")
    (args.output_dir/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n",encoding="utf-8")
    (args.output_dir/"summary.json").write_text(json.dumps({"format":"pick_single_ycb_object_variation_detector_rollouts_v1","checkpoint":str(args.checkpoint),"split":args.split,"episodes":len(rows),"videos":len(list((args.output_dir/"videos").glob("*.mp4")))},indent=2)+"\n",encoding="utf-8")
    (args.output_dir/"EVAL_COMPLETE").write_text(json.dumps(metrics,indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__": main()
