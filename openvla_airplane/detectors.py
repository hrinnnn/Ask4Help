"""Training-data-only detector assets for OpenVLA representations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def _fit_pca(values: np.ndarray, rank: int) -> dict:
    mean = values.mean(axis=0).astype(np.float32)
    pca = PCA(n_components=min(rank, values.shape[0] - 1, values.shape[1]), svd_solver="randomized", random_state=0)
    pca.fit(values.astype(np.float32))
    return {"mean": mean, "components": pca.components_.astype(np.float32), "singular_values": pca.singular_values_.astype(np.float32)}


def residual_score(values: np.ndarray, asset: dict) -> np.ndarray:
    centered = values.astype(np.float32) - asset["mean"]
    projection = centered @ asset["components"].T
    reconstruction = projection @ asset["components"]
    return np.linalg.norm(centered - reconstruction, axis=1)


def _fit_mahalanobis(values: np.ndarray) -> dict:
    from sklearn.covariance import LedoitWolf

    scaler = StandardScaler().fit(values)
    whitened = scaler.transform(values).astype(np.float32)
    estimator = LedoitWolf(assume_centered=False).fit(whitened)
    return {
        "mean": estimator.location_.astype(np.float32),
        "precision": estimator.precision_.astype(np.float32),
        "scale_mean": scaler.mean_.astype(np.float32),
        "scale_scale": scaler.scale_.astype(np.float32),
    }


def mahalanobis_score(values: np.ndarray, asset: dict) -> np.ndarray:
    whitened = (values - asset["scale_mean"]) / np.where(asset["scale_scale"] > 0, asset["scale_scale"], 1.0)
    centered = whitened - asset["mean"]
    return np.sqrt(np.maximum(0.0, np.einsum("ni,ij,nj->n", centered, asset["precision"], centered)))


def fit_detector_assets(feature_dir: Path, output: Path, pca_rank: int = 1000, knn_k: int = 10) -> dict:
    feature_dir = Path(feature_dir)
    output.mkdir(parents=True, exist_ok=True)
    projector = np.load(feature_dir / "projector_pooled.npy", mmap_mode="r").astype(np.float32)
    visual = np.load(feature_dir / "llama_visual_pooled.npy", mmap_mode="r").astype(np.float32)
    action = np.load(feature_dir / "llama_action_pooled.npy", mmap_mode="r").astype(np.float32)
    manifest = json.loads((feature_dir / "manifest.json").read_text())
    trained = {}

    def save_method(name: str, kind: str, asset: dict) -> None:
        method_dir = output / name
        method_dir.mkdir(parents=True, exist_ok=True)
        json_asset = {key: value for key, value in asset.items() if not isinstance(value, np.ndarray)}
        for key, value in asset.items():
            if isinstance(value, np.ndarray):
                np.save(method_dir / f"{key}.npy", value)
                json_asset[key] = f"{key}.npy"
        json_asset["kind"] = kind
        (method_dir / "asset.json").write_text(json.dumps(json_asset, indent=2))
        trained[name] = {"kind": kind, "path": str(method_dir)}

    save_method("vlm_input_pooled_residual_pca", "residual_pca", _fit_pca(projector, pca_rank))
    save_method("vlm_input_pooled_llmd", "mahalanobis", _fit_mahalanobis(projector))
    knn = NearestNeighbors(n_neighbors=knn_k, metric="euclidean", algorithm="auto").fit(projector)
    save_method("vlm_input_pooled_deep_knn", "knn", {"reference": projector, "k": knn_k})
    reduced_pca = PCA(n_components=min(64, projector.shape[0] - 1, projector.shape[1]), random_state=0).fit(projector)
    reduced = reduced_pca.transform(projector)
    kmeans = MiniBatchKMeans(n_clusters=min(32, max(2, reduced.shape[0] // 64)), random_state=0, batch_size=256, n_init=3).fit(reduced)
    save_method(
        "vlm_input_pooled_pca_kmeans",
        "pca_kmeans",
        {"pca_mean": reduced_pca.mean_.astype(np.float32), "centers": kmeans.cluster_centers_.astype(np.float32), "pca_components": reduced_pca.components_.astype(np.float32)},
    )
    save_method("dino_pooled_residual_pca", "residual_pca", _fit_pca(np.load(feature_dir / "dino_pooled.npy", mmap_mode="r"), pca_rank))
    save_method("siglip_pooled_residual_pca", "residual_pca", _fit_pca(np.load(feature_dir / "siglip_pooled.npy", mmap_mode="r"), pca_rank))
    for layer in range(visual.shape[1]):
        save_method(f"llama_layer_{layer + 1:02d}_residual_pca", "residual_pca", _fit_pca(visual[:, layer], pca_rank))
        save_method(f"llama_layer_{layer + 1:02d}_llmd", "mahalanobis", _fit_mahalanobis(visual[:, layer]))
    for layer in range(action.shape[1]):
        save_method(f"action_layer_{layer + 1:02d}_residual_pca", "residual_pca", _fit_pca(action[:, layer], pca_rank))
    for name in ("action_logprob", "action_entropy"):
        values = np.load(feature_dir / f"{name}.npy").astype(np.float32)
        save_method(name, "scalar", {"mean": np.asarray([values.mean()], dtype=np.float32), "std": np.asarray([values.std() + 1e-6], dtype=np.float32)})
    asset_manifest = {
        "source_feature_manifest": manifest,
        "id_observations": int(projector.shape[0]),
        "pca_rank_requested": pca_rank,
        "knn_k": knn_k,
        "methods": trained,
        "fit_split": "ID expert only",
        "threshold_policy": "calibration scores are stored separately and never used to fit representations",
    }
    (output / "manifest.json").write_text(json.dumps(asset_manifest, indent=2))
    return asset_manifest


def score_method(values: np.ndarray, method_dir: Path) -> np.ndarray:
    meta = json.loads((method_dir / "asset.json").read_text())
    asset = {}
    for key, value in meta.items():
        if isinstance(value, str) and value.endswith(".npy"):
            asset[key] = np.load(method_dir / value)
        else:
            asset[key] = value
    kind = meta["kind"]
    if kind == "residual_pca":
        return residual_score(values, asset)
    if kind == "mahalanobis":
        return mahalanobis_score(values, asset)
    if kind == "knn":
        distances = NearestNeighbors(n_neighbors=int(asset["k"])).fit(asset["reference"]).kneighbors(values, return_distance=True)[0]
        return distances.mean(axis=1)
    if kind == "scalar":
        return np.abs(values.reshape(-1) - float(asset["mean"][0])) / float(asset["std"][0])
    if kind == "pca_kmeans":
        centered = values.astype(np.float32) - asset["pca_mean"]
        embedding = centered @ asset["pca_components"].T
        distances = np.linalg.norm(embedding[:, None, :] - asset["centers"][None, :, :], axis=-1)
        return distances.min(axis=1)
    raise ValueError(f"Unsupported detector kind: {kind}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pca-rank", type=int, default=1000)
    args = parser.parse_args()
    print(json.dumps(fit_detector_assets(args.feature_dir, args.output, args.pca_rank), indent=2))


if __name__ == "__main__":
    main()
