"""Checkpoint loading helpers shared by causal training and inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


_SCALE_SUFFIXES = (".weight_scale", ".bias_scale", ".input_scale")
_DIFFUSION_NON_TRANSFORMER_PREFIXES = (
    "audio_embeddings_connector.",
    "video_embeddings_connector.",
    "embeddings_connector.",
)
_NON_TRANSFORMER_PREFIXES = (
    "vae.",
    "audio_vae.",
    "vocoder.",
    "model.vae.",
    "model.audio_vae.",
    "model.vocoder.",
)


def load_checkpoint_config(checkpoint_path: str | Path) -> dict[str, Any]:
    """Read the JSON model config embedded in a safetensors checkpoint."""
    path = Path(checkpoint_path)
    if path.suffix != ".safetensors":
        return {}

    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    raw_config = metadata.get("config")
    if not raw_config:
        return {}
    config = json.loads(raw_config)
    if not isinstance(config, dict):
        raise ValueError(f"Checkpoint config must be a JSON object: {path}")
    return config


def load_checkpoint_state_dict(
    checkpoint_path: str | Path,
    *,
    prefer_ema: bool = False,
) -> dict[str, torch.Tensor]:
    """Load a safetensors, sharded safetensors, or PyTorch state dict."""
    path = Path(checkpoint_path)
    path_text = str(path)

    if path_text.endswith(".safetensors.index.json"):
        from safetensors.torch import load_file

        with path.open("r", encoding="utf-8") as handle:
            index = json.load(handle)
        shard_names = sorted(set(index.get("weight_map", {}).values()))
        if not shard_names:
            raise ValueError(f"No weight_map entries found in {path}")
        state_dict: dict[str, torch.Tensor] = {}
        for shard_name in shard_names:
            shard = load_file(str(path.parent / shard_name), device="cpu")
            duplicates = sorted(set(state_dict) & set(shard))
            if duplicates:
                raise ValueError(f"Duplicate tensors across checkpoint shards: {duplicates[:20]}")
            state_dict.update(shard)
        return state_dict

    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(path_text, device="cpu")

    loaded = torch.load(path_text, map_location="cpu", weights_only=False)
    if prefer_ema and isinstance(loaded, dict) and "generator_ema" in loaded:
        loaded = loaded["generator_ema"]
    elif isinstance(loaded, dict) and "generator" in loaded:
        loaded = loaded["generator"]
    elif isinstance(loaded, dict) and "model" in loaded:
        loaded = loaded["model"]
    elif isinstance(loaded, dict) and "state_dict" in loaded:
        loaded = loaded["state_dict"]

    if not isinstance(loaded, dict) or not all(torch.is_tensor(value) for value in loaded.values()):
        raise TypeError(f"Checkpoint does not contain a tensor state dict: {path}")
    return loaded


def fold_prequantized_fp8_scales(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Reconstruct trainable optimizer master weights from static FP8 tensors.

    LTX-2.3 FP8 checkpoints are intended for scaled matrix multiplication and
    keep scalar scales beside FP8 parameters. AdamW updates a BF16 master while
    torchao casts Linear operands to FP8 for forward/backward GEMMs, so the
    master must be reconstructed exactly once before torchao conversion.
    """
    fp8_dtypes = {
        dtype
        for dtype in (
            getattr(torch, "float8_e4m3fn", None),
            getattr(torch, "float8_e5m2", None),
        )
        if dtype is not None
    }
    if not fp8_dtypes or not any(value.dtype in fp8_dtypes for value in state_dict.values()):
        return state_dict

    scale_keys = {key for key in state_dict if key.endswith(_SCALE_SUFFIXES)}
    if not scale_keys:
        raise ValueError("FP8 tensors were found without companion scale tensors")

    folded: dict[str, torch.Tensor] = {}
    folded_count = 0
    for key, value in state_dict.items():
        if key in scale_keys:
            continue

        scale = state_dict.get(f"{key}_scale")
        if value.dtype in fp8_dtypes:
            if scale is None:
                raise ValueError(f"FP8 tensor has no companion scale: {key}")
            if scale.numel() != 1:
                raise ValueError(
                    f"Unsupported FP8 scale shape {tuple(scale.shape)} for {key}"
                )
            value = (
                value.to(torch.float32)
                * scale.to(device=value.device, dtype=torch.float32)
            ).to(torch.bfloat16)
            folded_count += 1
        folded[key] = value

    weight_scale_count = sum(
        key.endswith((".weight_scale", ".bias_scale")) for key in scale_keys
    )
    if weight_scale_count and folded_count != weight_scale_count:
        raise ValueError(
            "Not every FP8 parameter scale was folded: "
            f"folded={folded_count}, parameter_scales={weight_scale_count}"
        )

    print(
        f"[checkpoint] Reconstructed {folded_count} BF16 optimizer master weights and "
        f"dropped {len(scale_keys)} scale tensors",
        flush=True,
    )
    return folded


def remap_transformer_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    target_prefix: str = "model.",
    preserve_fp8: bool = False,
) -> dict[str, torch.Tensor]:
    """Map common LTX checkpoint layouts to one transformer key layout."""
    if not state_dict:
        return state_dict

    if not preserve_fp8:
        state_dict = fold_prequantized_fp8_scales(state_dict)

    if any(key.startswith("model.diffusion_model.") for key in state_dict):
        remapped = {}
        source_prefix = "model.diffusion_model."
        for key, value in state_dict.items():
            if not key.startswith(source_prefix):
                continue
            stripped = key[len(source_prefix) :]
            if stripped.startswith(_DIFFUSION_NON_TRANSFORMER_PREFIXES):
                continue
            remapped[f"{target_prefix}{stripped}"] = value
        return remapped

    if any(key.startswith("model.velocity_model.") for key in state_dict):
        source_prefix = "model.velocity_model."
        return {
            f"{target_prefix}{key[len(source_prefix):]}": value
            for key, value in state_dict.items()
            if key.startswith(source_prefix)
        }

    if all(key.startswith("model.") for key in state_dict):
        source_prefix = "model."
        return {
            f"{target_prefix}{key[len(source_prefix):]}": value
            for key, value in state_dict.items()
            if not any(key.startswith(prefix) for prefix in _NON_TRANSFORMER_PREFIXES)
        }

    return {
        f"{target_prefix}{key}": value
        for key, value in state_dict.items()
        if not any(key.startswith(prefix) for prefix in _NON_TRANSFORMER_PREFIXES)
    }
