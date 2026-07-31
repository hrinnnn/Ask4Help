#!/usr/bin/env python3
"""Serve the official JAX ``pi05_libero`` policy with frozen internal probes.

The action path is a line-for-line equivalent of OpenPI's ``Policy.infer``.
The only addition is one *separate*, deterministic feature forward at a fixed
Gaussian action prior and flow time ``t=1``.  It returns two no-training
failure-detection representations alongside the unmodified action chunk:

* ``bridge``: valid-token mean of the final VLM prefix representation;
* ``action_expert_final``: final hidden state for all ten action tokens.

This file intentionally imports OpenPI from the active environment rather
than patching its source tree.  It must be run with the same OpenPI revision
that owns the checkpoint under evaluation.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import socket
import time
from pathlib import Path
from typing import Any

import einops
import jax
import jax.numpy as jnp
import numpy as np
import tyro

from openpi.models import model as _model
from openpi.models.pi0 import make_attn_mask
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.shared import nnx_utils
from openpi.training import config as _config


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclasses.dataclass(frozen=True)
class ProbeSpec:
    """Immutable probe contract saved with every rollout."""

    action_horizon: int = 10
    action_dim: int = 7
    feature_seed: int = 20260731
    flow_timestep: float = 1.0


class InternalFeaturePolicy(_policy.Policy):
    """Official policy with one deterministic, input-only feature forward."""

    def __init__(self, *args: Any, probe_spec: ProbeSpec, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self._is_pytorch_model:
            raise RuntimeError("this strict LIBERO protocol currently requires the official JAX pi05_libero checkpoint")
        model = self._model
        if int(model.action_horizon) != probe_spec.action_horizon or int(model.action_dim) != probe_spec.action_dim:
            raise ValueError(
                f"probe expects action shape [{probe_spec.action_horizon}, {probe_spec.action_dim}], "
                f"got [{model.action_horizon}, {model.action_dim}]"
            )
        self._probe_spec = probe_spec
        key = jax.random.key(probe_spec.feature_seed)
        self._fixed_prior = jax.random.normal(
            key, (1, probe_spec.action_horizon, probe_spec.action_dim), dtype=jnp.float32
        )
        self._probe_features = nnx_utils.module_jit(self._probe_features_impl)

    def _probe_features_impl(self, observation: _model.Observation) -> dict[str, jax.Array]:
        """Return bridge and final expert features in one PaliGemma forward."""
        model = self._model
        observation = _model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]
        noisy_actions = jnp.broadcast_to(self._fixed_prior, (batch_size,) + self._fixed_prior.shape[1:])
        timestep = jnp.full((batch_size,), self._probe_spec.flow_timestep, dtype=jnp.float32)

        prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
            observation, noisy_actions, timestep
        )
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attention_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = model.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attention_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )
        if prefix_out is None or suffix_out is None:
            raise RuntimeError("PaliGemma probe unexpectedly returned an empty segment")
        valid = prefix_mask.astype(prefix_out.dtype)[..., None]
        bridge = (prefix_out * valid).sum(axis=1, keepdims=True) / valid.sum(axis=1, keepdims=True)
        action_final = suffix_out[:, -model.action_horizon :]
        return {"bridge": bridge, "action_expert_final": action_final}

    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[override]
        # This is intentionally kept equivalent to upstream Policy.infer.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        self._rng, sample_rng = jax.random.split(self._rng)
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise_array = jnp.asarray(noise)
            sample_kwargs["noise"] = noise_array[None, ...] if noise_array.ndim == 2 else noise_array
        observation = _model.Observation.from_dict(inputs)

        start = time.monotonic()
        actions = self._sample_actions(sample_rng, observation, **sample_kwargs)
        action_ms = (time.monotonic() - start) * 1000.0
        probe_start = time.monotonic()
        features = self._probe_features(observation)
        probe_ms = (time.monotonic() - probe_start) * 1000.0

        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), {"state": inputs["state"], "actions": actions})
        outputs = self._output_transform(outputs)
        outputs["failure_features"] = jax.tree.map(lambda x: np.asarray(x[0, ...], dtype=np.float32), features)
        outputs["failure_probe"] = {
            "format": "pi05_libero_internal_feature_probe_v1",
            "feature_seed": self._probe_spec.feature_seed,
            "flow_timestep": self._probe_spec.flow_timestep,
            "action_horizon": self._probe_spec.action_horizon,
            "action_dim": self._probe_spec.action_dim,
            "bridge_pooling": "mean_valid_final_vlm_prefix_tokens",
            "action_expert_layer": "final_hidden_before_action_out_proj",
            "feature_ms": probe_ms,
        }
        outputs["policy_timing"] = {"infer_ms": action_ms}
        return outputs


@dataclasses.dataclass
class Args:
    checkpoint: Path
    port: int = 8001
    config_name: str = "pi05_libero"
    feature_seed: int = 20260731


def main(args: Args) -> None:
    # create_trained_policy exposes transformed model through an ordinary
    # policy.  Rebuild with the same exact composed transforms below.
    config = _config.get_config(args.config_name)
    ordinary = _policy_config.create_trained_policy(config, args.checkpoint)
    policy = InternalFeaturePolicy(
        ordinary._model,
        rng=ordinary._rng,
        transforms=(),
        output_transforms=(),
        sample_kwargs=ordinary._sample_kwargs,
        metadata={
            **ordinary.metadata,
            "failure_probe_format": "pi05_libero_internal_feature_probe_v1",
            "checkpoint": str(args.checkpoint),
            "checkpoint_params_is_directory": (args.checkpoint / "params").is_dir(),
        },
        probe_spec=ProbeSpec(feature_seed=args.feature_seed),
    )
    # ``Policy`` stores only composed transforms. Keep them exactly, rather
    # than recreating training transforms from configs a second time.
    policy._input_transform = ordinary._input_transform
    policy._output_transform = ordinary._output_transform
    hostname = socket.gethostname()
    logging.info("Serving pi05 internal probes on %s:%d", hostname, args.port)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host="0.0.0.0", port=args.port, metadata=policy.metadata
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
