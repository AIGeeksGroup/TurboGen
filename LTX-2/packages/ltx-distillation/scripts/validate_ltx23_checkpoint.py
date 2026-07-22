#!/usr/bin/env python
"""Validate LTX-2.3 base architecture and optional Step-2 compatibility."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open


DIFFUSION_PREFIX = "model.diffusion_model."
CONNECTOR_PREFIXES = (
    "audio_embeddings_connector.",
    "video_embeddings_connector.",
    "embeddings_connector.",
)
SCALE_SUFFIXES = (".weight_scale", ".bias_scale", ".input_scale")
ATTENTION_NAMES = (
    "attn1",
    "attn2",
    "audio_attn1",
    "audio_attn2",
    "audio_to_video_attn",
    "video_to_audio_attn",
)


def _load_base_description(path: Path) -> tuple[dict, dict[str, tuple[int, ...]], dict]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        config = json.loads(metadata.get("config", "{}"))
        transformer = config.get("transformer", {})
        keys = set(handle.keys())

        required_config = {
            "rope_type": "split",
            "cross_attention_adaln": True,
            "apply_gated_attention": True,
        }
        mismatched_config = {
            key: {"actual": transformer.get(key), "expected": expected}
            for key, expected in required_config.items()
            if transformer.get(key) != expected
        }
        if mismatched_config:
            raise RuntimeError(f"Not the expected LTX-2.3 architecture: {mismatched_config}")

        vocoder = config.get("vocoder", {})
        vocoder_config = vocoder.get("vocoder", {})
        bwe_config = vocoder.get("bwe", {})
        required_audio_config = {
            "vocoder.resblock": (vocoder_config.get("resblock"), "AMP1"),
            "vocoder.activation": (vocoder_config.get("activation"), "snakebeta"),
            "vocoder.stereo": (vocoder_config.get("stereo"), True),
            "bwe.resblock": (bwe_config.get("resblock"), "AMP1"),
            "bwe.activation": (bwe_config.get("activation"), "snakebeta"),
            "bwe.stereo": (bwe_config.get("stereo"), True),
            "bwe.input_sampling_rate": (
                bwe_config.get("input_sampling_rate"),
                16000,
            ),
            "bwe.output_sampling_rate": (
                bwe_config.get("output_sampling_rate"),
                48000,
            ),
        }
        bad_audio_config = {
            key: {"actual": actual, "expected": expected}
            for key, (actual, expected) in required_audio_config.items()
            if actual != expected
        }
        if bad_audio_config:
            raise RuntimeError(
                f"LTX-2.3 vocoder/BWE configuration mismatch: {bad_audio_config}"
            )

        vocoder_keys = {key for key in keys if key.startswith("vocoder.")}
        vocoder_key_counts = {
            "vocoder": sum(key.startswith("vocoder.vocoder.") for key in vocoder_keys),
            "bwe_generator": sum(
                key.startswith("vocoder.bwe_generator.") for key in vocoder_keys
            ),
            "mel_stft": sum(key.startswith("vocoder.mel_stft.") for key in vocoder_keys),
        }
        expected_vocoder_key_counts = {
            "vocoder": 667,
            "bwe_generator": 557,
            "mel_stft": 3,
        }
        if vocoder_key_counts != expected_vocoder_key_counts:
            raise RuntimeError(
                "LTX-2.3 vocoder checkpoint is incomplete: "
                f"{vocoder_key_counts} != {expected_vocoder_key_counts}"
            )

        num_layers = int(transformer.get("num_layers", 48))
        video_dim = int(transformer.get("num_attention_heads", 32)) * int(
            transformer.get("attention_head_dim", 128)
        )
        audio_dim = int(transformer.get("audio_num_attention_heads", 32)) * int(
            transformer.get("audio_attention_head_dim", 64)
        )

        missing_architecture_keys = []
        bad_shapes = []
        for layer in range(num_layers):
            layer_prefix = f"{DIFFUSION_PREFIX}transformer_blocks.{layer}."
            expected_shapes = {
                f"{layer_prefix}scale_shift_table": (9, video_dim),
                f"{layer_prefix}audio_scale_shift_table": (9, audio_dim),
                f"{layer_prefix}prompt_scale_shift_table": (2, video_dim),
                f"{layer_prefix}audio_prompt_scale_shift_table": (2, audio_dim),
            }
            for key, expected_shape in expected_shapes.items():
                if key not in keys:
                    missing_architecture_keys.append(key)
                    continue
                actual_shape = tuple(handle.get_slice(key).get_shape())
                if actual_shape != expected_shape:
                    bad_shapes.append((key, actual_shape, expected_shape))

            for attention_name in ATTENTION_NAMES:
                for suffix in ("weight", "bias"):
                    key = f"{layer_prefix}{attention_name}.to_gate_logits.{suffix}"
                    if key not in keys:
                        missing_architecture_keys.append(key)

        for key, expected_shape in {
            f"{DIFFUSION_PREFIX}adaln_single.linear.weight": (9 * video_dim, video_dim),
            f"{DIFFUSION_PREFIX}adaln_single.linear.bias": (9 * video_dim,),
            f"{DIFFUSION_PREFIX}audio_adaln_single.linear.weight": (9 * audio_dim, audio_dim),
            f"{DIFFUSION_PREFIX}audio_adaln_single.linear.bias": (9 * audio_dim,),
        }.items():
            if key not in keys:
                missing_architecture_keys.append(key)
                continue
            actual_shape = tuple(handle.get_slice(key).get_shape())
            if actual_shape != expected_shape:
                bad_shapes.append((key, actual_shape, expected_shape))

        for prefix in ("prompt_adaln_single.", "audio_prompt_adaln_single."):
            if not any(key.startswith(f"{DIFFUSION_PREFIX}{prefix}") for key in keys):
                missing_architecture_keys.append(f"{DIFFUSION_PREFIX}{prefix}*")

        if missing_architecture_keys or bad_shapes:
            raise RuntimeError(
                "LTX-2.3 structural validation failed: "
                f"missing={missing_architecture_keys[:10]}, bad_shapes={bad_shapes[:10]}"
            )

        weight_scale_keys = sorted(key for key in keys if key.endswith(".weight_scale"))
        input_scale_keys = sorted(key for key in keys if key.endswith(".input_scale"))
        fp8_weights = sorted(
            key
            for key in keys
            if key.endswith(".weight") and handle.get_slice(key).get_dtype() == "F8_E4M3"
        )
        quantization = json.loads(metadata.get("_quantization_metadata", "{}"))
        declared_layers = {
            name
            for name, spec in quantization.get("layers", {}).items()
            if isinstance(spec, dict) and spec.get("format") == "float8_e4m3fn"
        }
        actual_layers = {key.removesuffix(".weight") for key in fp8_weights}
        if declared_layers != actual_layers:
            raise RuntimeError(
                "FP8 metadata/tensor mismatch: "
                f"metadata_only={sorted(declared_layers - actual_layers)[:10]}, "
                f"tensor_only={sorted(actual_layers - declared_layers)[:10]}"
            )
        missing_scales = {
            key: [
                scale_key
                for scale_key in (
                    f"{key.removesuffix('.weight')}.weight_scale",
                    f"{key.removesuffix('.weight')}.input_scale",
                )
                if scale_key not in keys
            ]
            for key in fp8_weights
        }
        missing_scales = {key: value for key, value in missing_scales.items() if value}
        if missing_scales:
            raise RuntimeError(f"FP8 weights have incomplete scales: {list(missing_scales.items())[:10]}")
        if not (
            len(fp8_weights) == len(weight_scale_keys) == len(input_scale_keys) == 1496
        ):
            raise RuntimeError(
                "Expected 1496 FP8 weights and scale pairs, got "
                f"weights={len(fp8_weights)}, weight_scale={len(weight_scale_keys)}, "
                f"input_scale={len(input_scale_keys)}"
            )

        expected_shapes = {}
        for key in keys:
            if not key.startswith(DIFFUSION_PREFIX) or key.endswith(SCALE_SUFFIXES):
                continue
            stripped = key[len(DIFFUSION_PREFIX) :]
            if stripped.startswith(CONNECTOR_PREFIXES):
                continue
            expected_shapes[stripped] = tuple(handle.get_slice(key).get_shape())

    summary = {
        "num_layers": num_layers,
        "video_dim": video_dim,
        "audio_dim": audio_dim,
        "transformer_tensors": len(expected_shapes),
        "fp8_weights": len(fp8_weights),
        "weight_scales": len(weight_scale_keys),
        "input_scales": len(input_scale_keys),
        "rope_type": transformer["rope_type"],
        "cross_attention_adaln": transformer["cross_attention_adaln"],
        "apply_gated_attention": transformer["apply_gated_attention"],
        "vocoder_tensors": len(vocoder_keys),
        "vocoder_key_counts": vocoder_key_counts,
        "audio_output_sample_rate": bwe_config["output_sampling_rate"],
    }
    return config, expected_shapes, summary


def _unwrap_student(path: Path) -> tuple[dict[str, torch.Tensor], dict]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
        return load_file(str(path), device="cpu"), metadata

    checkpoint = torch.load(
        str(path),
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    metadata = checkpoint if isinstance(checkpoint, dict) else {}
    if isinstance(checkpoint, dict) and "generator" in checkpoint:
        state_dict = checkpoint["generator"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError(f"No state dict found in {path}")
    return state_dict, metadata


def _student_shapes(state_dict: dict[str, torch.Tensor]) -> dict[str, tuple[int, ...]]:
    shapes = {}
    for key, value in state_dict.items():
        if key.endswith(SCALE_SUFFIXES):
            continue
        if key.startswith("model.velocity_model."):
            key = key[len("model.velocity_model.") :]
        elif key.startswith("model."):
            key = key[len("model.") :]
        shapes[key] = tuple(value.shape)
    return shapes


def _static_fp8_student_summary(
    state_dict: dict[str, torch.Tensor],
    metadata: dict,
) -> dict:
    fp8_weights = {
        key: value
        for key, value in state_dict.items()
        if key.endswith(".weight") and value.dtype == torch.float8_e4m3fn
    }
    if not fp8_weights:
        return {"static_fp8": False, "fp8_weights": 0}

    errors = []
    for key in fp8_weights:
        module_name = key.removesuffix(".weight")
        for suffix in (".weight_scale", ".input_scale"):
            scale_key = module_name + suffix
            scale = state_dict.get(scale_key)
            if scale is None:
                errors.append((key, f"missing {scale_key}"))
            elif scale.dtype != torch.float32 or scale.numel() != 1:
                errors.append((scale_key, f"dtype={scale.dtype}, shape={tuple(scale.shape)}"))

    quantization = json.loads(metadata.get("_quantization_metadata", "{}"))
    declared = {
        name
        for name, spec in quantization.get("layers", {}).items()
        if isinstance(spec, dict) and spec.get("format") == "float8_e4m3fn"
    }
    actual = {key.removesuffix(".weight") for key in fp8_weights}
    if declared != actual:
        errors.append(
            (
                "_quantization_metadata",
                {
                    "metadata_only": sorted(declared - actual)[:10],
                    "tensor_only": sorted(actual - declared)[:10],
                },
            )
        )
    if len(fp8_weights) != 1496:
        errors.append(("fp8_weight_count", (len(fp8_weights), 1496)))
    if errors:
        raise RuntimeError(f"Static FP8 Student validation failed: {errors[:10]}")
    return {
        "static_fp8": True,
        "fp8_weights": len(fp8_weights),
        "weight_scales": sum(key.endswith(".weight_scale") for key in state_dict),
        "input_scales": sum(key.endswith(".input_scale") for key in state_dict),
    }


def _validate_student(
    path: Path,
    expected_shapes: dict[str, tuple[int, ...]],
) -> dict:
    state_dict, metadata = _unwrap_student(path)
    actual_shapes = _student_shapes(state_dict)
    missing = sorted(set(expected_shapes) - set(actual_shapes))
    unexpected = sorted(set(actual_shapes) - set(expected_shapes))
    shape_mismatches = sorted(
        (key, actual_shapes[key], expected_shapes[key])
        for key in set(expected_shapes) & set(actual_shapes)
        if actual_shapes[key] != expected_shapes[key]
    )

    summary = {
        "student_tensors": len(actual_shapes),
        "missing": len(missing),
        "unexpected": len(unexpected),
        "shape_mismatches": len(shape_mismatches),
        "format_version": metadata.get("format_version"),
        "step": metadata.get("step"),
        "missing_examples": missing[:10],
        "unexpected_examples": unexpected[:10],
        "shape_mismatch_examples": shape_mismatches[:10],
        **_static_fp8_student_summary(state_dict, metadata),
    }
    if missing or unexpected or shape_mismatches:
        raise RuntimeError(
            "Student checkpoint is not compatible with the complete LTX-2.3 architecture: "
            + json.dumps(summary, indent=2)
        )
    return summary


def _read_lmdb_shape(transaction, key: str) -> tuple[int, ...]:
    raw = transaction.get(f"{key}_shape".encode())
    if raw is None:
        raise RuntimeError(f"LMDB is missing {key}_shape")
    return tuple(int(value) for value in raw.decode("utf-8").split())


def _validate_ode_lmdb(
    path: Path,
    base_checkpoint: Path,
    denoising_step_list: list[int],
) -> dict:
    import lmdb
    import numpy as np

    if not path.is_dir():
        raise FileNotFoundError(f"ODE LMDB directory not found: {path}")

    environment = lmdb.open(
        str(path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    try:
        with environment.begin() as transaction:
            raw_manifest = transaction.get(b"manifest_json")
            if raw_manifest is None:
                raise RuntimeError(f"ODE LMDB has no provenance manifest: {path}")
            manifest = json.loads(raw_manifest.decode("utf-8"))
            video_shape = _read_lmdb_shape(transaction, "video_latents")
            audio_shape = _read_lmdb_shape(transaction, "audio_latents")
            sigmas_shape = _read_lmdb_shape(transaction, "sigmas")
            prompts_shape = _read_lmdb_shape(transaction, "prompts")
            first_sigmas = transaction.get(b"sigmas_0_data")

        if manifest.get("format_version") != 4:
            raise RuntimeError(
                f"Unsupported ODE manifest version: {manifest.get('format_version')}"
            )
        if manifest.get("producer") != "omniforcing-ltx23-full-architecture-v2":
            raise RuntimeError(
                f"Unsupported ODE manifest producer: {manifest.get('producer')}"
            )
        actual_teacher = os.path.realpath(manifest.get("teacher_checkpoint", ""))
        expected_teacher = os.path.realpath(str(base_checkpoint))
        if actual_teacher != expected_teacher:
            raise RuntimeError(
                f"ODE teacher checkpoint mismatch: {actual_teacher} != {expected_teacher}"
            )

        generation_config = manifest.get("generation_config")
        if not isinstance(generation_config, dict):
            raise RuntimeError("ODE manifest has no generation_config object")
        actual_steps = list(generation_config.get("denoising_step_list") or [])
        if actual_steps != denoising_step_list:
            raise RuntimeError(
                f"ODE denoising schedule mismatch: {actual_steps} != {denoising_step_list}"
            )

        if len(video_shape) != 6 or len(audio_shape) != 4 or len(sigmas_shape) != 2:
            raise RuntimeError(
                "Unexpected LMDB shapes: "
                f"video={video_shape}, audio={audio_shape}, sigmas={sigmas_shape}"
            )
        sample_count = video_shape[0]
        if sample_count <= 0 or audio_shape[0] != sample_count or sigmas_shape[0] != sample_count:
            raise RuntimeError("ODE LMDB modality sample counts do not match")
        if prompts_shape != (sample_count,):
            raise RuntimeError(
                f"ODE LMDB prompt count mismatch: {prompts_shape} != {(sample_count,)}"
            )
        trajectory_length = len(denoising_step_list)
        if video_shape[1] != trajectory_length or audio_shape[1] != trajectory_length:
            raise RuntimeError("ODE LMDB trajectory length does not match denoising schedule")
        if sigmas_shape != (sample_count, trajectory_length):
            raise RuntimeError(
                f"ODE sigma shape mismatch: {sigmas_shape} != {(sample_count, trajectory_length)}"
            )

        expected_video_frames = 1 + (int(generation_config["num_frames"]) - 1) // 8
        expected_height = int(generation_config["video_height"]) // 32
        expected_width = int(generation_config["video_width"]) // 32
        if video_shape[2:] != (expected_video_frames, 128, expected_height, expected_width):
            raise RuntimeError(
                "ODE video latent geometry mismatch: "
                f"{video_shape[2:]} != "
                f"{(expected_video_frames, 128, expected_height, expected_width)}"
            )

        if first_sigmas is None:
            raise RuntimeError("ODE LMDB is missing sigmas_0_data")
        sigma_values = np.frombuffer(first_sigmas, dtype=np.float32)
        if sigma_values.shape != (trajectory_length,):
            raise RuntimeError(f"Stored sigma row has shape {sigma_values.shape}")
        if not np.isfinite(sigma_values).all():
            raise RuntimeError("Stored sigma row contains NaN or Inf")
        if sigma_values[-1] != 0.0 or not np.all(sigma_values[:-1] > sigma_values[1:]):
            raise RuntimeError(
                f"Stored sigmas must decrease to zero, got {sigma_values.tolist()}"
            )

        return {
            "samples": sample_count,
            "video_shape": video_shape,
            "audio_shape": audio_shape,
            "sigmas_shape": sigmas_shape,
            "sigmas": sigma_values.tolist(),
            "teacher_checkpoint": actual_teacher,
            "denoising_step_list": actual_steps,
        }
    finally:
        environment.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path)
    parser.add_argument("--ode-lmdb", type=Path)
    parser.add_argument(
        "--denoising-step-list",
        default="1000,909,725,421,0",
        help="Comma-separated Step-2 schedule expected in the ODE LMDB.",
    )
    args = parser.parse_args()

    _, expected_shapes, base_summary = _load_base_description(args.base_checkpoint)
    print("BASE_VALID=1")
    print(json.dumps(base_summary, indent=2, sort_keys=True))

    if args.student_checkpoint is not None:
        student_summary = _validate_student(args.student_checkpoint, expected_shapes)
        print("STUDENT_VALID=1")
        print(json.dumps(student_summary, indent=2, sort_keys=True))

    if args.ode_lmdb is not None:
        expected_steps = [
            int(item.strip())
            for item in args.denoising_step_list.split(",")
            if item.strip()
        ]
        ode_summary = _validate_ode_lmdb(
            args.ode_lmdb,
            args.base_checkpoint,
            expected_steps,
        )
        print("ODE_LMDB_VALID=1")
        print(json.dumps(ode_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
