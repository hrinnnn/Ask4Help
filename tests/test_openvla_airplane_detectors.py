import json

import numpy as np

from openvla_airplane.detectors import fit_detector_assets


def test_detector_assets_use_manifest_block_numbers(tmp_path):
    feature_dir = tmp_path / "features"
    output = tmp_path / "assets"
    feature_dir.mkdir()
    rng = np.random.default_rng(4)
    observations, dimensions = 12, 6
    np.save(feature_dir / "projector_pooled.npy", rng.normal(size=(observations, dimensions)).astype(np.float32))
    np.save(feature_dir / "dino_pooled.npy", rng.normal(size=(observations, dimensions)).astype(np.float32))
    np.save(feature_dir / "siglip_pooled.npy", rng.normal(size=(observations, dimensions)).astype(np.float32))
    for name in ("llama_visual_pooled", "llama_action_pooled", "prompt_decision"):
        np.save(feature_dir / f"{name}.npy", rng.normal(size=(observations, 4, dimensions)).astype(np.float32))
    np.save(feature_dir / "action_logprob.npy", rng.normal(size=observations).astype(np.float32))
    np.save(feature_dir / "action_entropy.npy", rng.normal(size=observations).astype(np.float32))
    (feature_dir / "manifest.json").write_text(json.dumps({"selected_blocks": [8, 16, 24, 32]}))

    manifest = fit_detector_assets(feature_dir, output, pca_rank=3, knn_k=2)

    assert manifest["selected_blocks"] == [8, 16, 24, 32]
    assert "llama_visual_layer_08_residual_pca" in manifest["methods"]
    assert "action_layer_32_llmd" in manifest["methods"]
    assert "prompt_layer_16_residual_pca" in manifest["methods"]
    assert "llama_visual_layer_01_residual_pca" not in manifest["methods"]
