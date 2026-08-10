#!/usr/bin/env python3
"""Standalone OmniForcing causal AV inference.

This is the GitHub/Hugging Face release copy of the single-file inference
entry point. The checkpoint path below is intentionally a placeholder: upload
the OmniForcing checkpoint to Hugging Face, download it locally, then pass that
local file path to ``--generator-ckpt``.

Recommended checkpoint
----------------------
For the current 5-second model, use the OmniForcing causal checkpoint:

    <HF_ORG>/<HF_REPO>/omniforcing_ltx2_5s_causal.safetensors.index.json

When running this script, ``--generator-ckpt`` should point to the downloaded
checkpoint file, for example:

    /path/to/downloaded/omniforcing_ltx2_5s_causal.safetensors.index.json

For a Step-2 ODE checkpoint, the script defaults to prefix-rerun causal
inference with Euler updates. This matches the full-sequence causal path used
during Step-2 training and the ODE trajectory sampler. The KV-cache path is
kept as an explicit diagnostic option for later self-forcing checkpoints.

The loader accepts both the released sharded ``.safetensors.index.json`` form
and the training checkpoint ``model.pt`` form.  A ``model.pt`` file is a
container whose ``generator`` entry is the causal model state dict; it does not
need to be converted before inference.  For an LTX-2.3-trained model, use the
same LTX-2.3 FP8 base checkpoint used during training.  Do not combine it with
the older LTX-2 19B base shown in the historical example below.

The released 5-second settings are:

    - 5-second output: 121 frames at 24 FPS
    - resolution: 512 x 768
    - causal blocks: 4, 3, 3, 3, 3 latent video frames
    - denoising timesteps: 1000, 909, 725, 421, 0
    - prompt seeds: base seed + prompt index

Example
-------
Run one prompt:

    python scripts/omniforcing_causal_inference.py \
        --base-checkpoint /path/to/ltx-2-19b-dev.safetensors \
        --vae-checkpoint /path/to/ltx-2-19b-dev.safetensors \
        --gemma-path /path/to/gemma-3-12b-it-qat-q4_0-unquantized \
        --generator-ckpt /path/to/downloaded/omniforcing_ltx2_5s_causal.safetensors.index.json \
        --prompt "Realistic. Rain falls on a quiet street at night." \
        --output-dir outputs/demo

Run a prompt file, one prompt per line:

    python scripts/omniforcing_causal_inference.py \
        --base-checkpoint /path/to/ltx-2-19b-dev.safetensors \
        --vae-checkpoint /path/to/ltx-2-19b-dev.safetensors \
        --gemma-path /path/to/gemma-3-12b-it-qat-q4_0-unquantized \
        --generator-ckpt /path/to/downloaded/omniforcing_ltx2_5s_causal.safetensors.index.json \
        --prompt-file prompts/demo.txt \
        --output-dir outputs/demo

Outputs
-------
Each sample is saved as:

    sample_000.mp4
    sample_000.txt

The mp4 writer tries to mux generated audio into the video. If audio muxing is
not available in the local torchvision/ffmpeg build, a silent mp4 and a sidecar
sample_000.wav are written instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _add_repo_paths() -> None:
    """Make the script runnable from a source checkout without pip install -e."""
    script_path = Path(__file__).resolve()
    # LTX-2/packages/ltx-distillation/scripts/this_file.py
    packages_dir = script_path.parents[2]
    for package in ("ltx-distillation", "ltx-causal", "ltx-core", "ltx-pipelines"):
        src = packages_dir / package / "src"
        if src.exists():
            sys.path.insert(0, str(src))


_add_repo_paths()

import torch

from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.loader.registry import StateDictRegistry
from ltx_causal.transformer.causal_model import CausalLTXModel, CausalLTXModelConfig
from ltx_causal.wrapper import CausalLTX2DiffusionWrapper
from ltx_distillation.inference.causal_pipeline import CausalAVInferencePipeline
from ltx_distillation.inference.ode_benchmark_pipeline import ODEAutoregressiveBenchmarkPipeline
from ltx_distillation.models.text_encoder_wrapper import create_text_encoder_wrapper
from ltx_distillation.models.vae_wrapper import create_vae_wrappers


def load_checkpoint_state_dict(path: str, prefer_ema: bool = False) -> dict[str, torch.Tensor]:
    """Load a checkpoint and return the generator state dict when present."""
    checkpoint_path = Path(path)
    if checkpoint_path.is_dir():
        index_files = sorted(checkpoint_path.glob("*.safetensors.index.json"))
        if index_files:
            return load_checkpoint_state_dict(str(index_files[0]), prefer_ema=prefer_ema)
        shard_files = sorted(checkpoint_path.glob("*.safetensors"))
        if shard_files:
            return load_safetensors_shards(shard_files)
        raise FileNotFoundError(f"No safetensors checkpoint shards found in {checkpoint_path}")

    if path.endswith(".safetensors.index.json"):
        with open(path, "r", encoding="utf-8") as handle:
            index = json.load(handle)
        weight_map = index.get("weight_map", {})
        shard_names = sorted(set(weight_map.values()))
        shard_files = [checkpoint_path.parent / name for name in shard_names]
        return load_safetensors_shards(shard_files)

    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path)

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if prefer_ema and isinstance(checkpoint, dict) and "generator_ema" in checkpoint:
        return checkpoint["generator_ema"]
    if isinstance(checkpoint, dict) and "generator" in checkpoint:
        return checkpoint["generator"]
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def load_safetensors_shards(paths: list[Path]) -> dict[str, torch.Tensor]:
    """Load and merge a sharded safetensors checkpoint."""
    from safetensors.torch import load_file

    state_dict: dict[str, torch.Tensor] = {}
    for path in paths:
        shard = load_file(str(path))
        duplicates = sorted(set(state_dict) & set(shard))
        if duplicates:
            raise ValueError(f"Duplicate tensors across checkpoint shards: {duplicates[:20]}")
        state_dict.update(shard)
    return state_dict


def remap_state_dict_keys(
    state_dict: dict[str, torch.Tensor],
    *,
    preserve_fp8: bool = False,
) -> dict[str, torch.Tensor]:
    """Map common LTX checkpoint key layouts to CausalLTX2DiffusionWrapper keys."""
    from ltx_distillation.ode.ode_regression import LTX2ODERegression

    return LTX2ODERegression._remap_state_dict_keys(state_dict, preserve_fp8=preserve_fp8)


def legacy_remap_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Previous standalone remapper kept for reference/fallback debugging."""
    if not state_dict:
        return state_dict

    non_transformer_prefixes = (
        "vae.",
        "audio_vae.",
        "vocoder.",
        "model.vae.",
        "model.audio_vae.",
        "model.vocoder.",
    )
    remapped_non_transformer_prefixes = (
        "model.audio_embeddings_connector.",
        "model.video_embeddings_connector.",
    )

    if any(key.startswith("model.diffusion_model.") for key in state_dict):
        remapped = {}
        for key, value in state_dict.items():
            if not key.startswith("model.diffusion_model."):
                continue
            new_key = "model." + key[len("model.diffusion_model.") :]
            if any(new_key.startswith(prefix) for prefix in remapped_non_transformer_prefixes):
                continue
            remapped[new_key] = value
        return remapped

    first_key = next(iter(state_dict))
    if first_key.startswith("model.velocity_model."):
        return {
            "model." + key[len("model.velocity_model.") :]: value
            for key, value in state_dict.items()
            if key.startswith("model.velocity_model.")
        }
    if first_key.startswith("model."):
        return {
            key: value
            for key, value in state_dict.items()
            if not any(key.startswith(prefix) for prefix in non_transformer_prefixes)
        }
    return {
        "model." + key: value
        for key, value in state_dict.items()
        if not any(key.startswith(prefix) for prefix in non_transformer_prefixes)
    }


def add_noise(original: torch.Tensor, noise: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Flow-matching interpolation: x_t = (1 - sigma) * x_0 + sigma * noise."""
    if sigma.dim() == 1:
        sigma = sigma.reshape(-1, *[1] * (original.dim() - 1))
    elif sigma.dim() == 2:
        sigma = sigma.reshape(*sigma.shape, *[1] * (original.dim() - 2))
    sigma = sigma.to(dtype=original.dtype)
    return ((1 - sigma) * original + sigma * noise).to(dtype=original.dtype)


def print_tensor_stats(name: str, tensor: torch.Tensor) -> None:
    values = tensor.detach().float()
    finite = torch.isfinite(values)
    finite_values = values[finite]
    if finite_values.numel() == 0:
        print(f"[stats] {name}: shape={list(tensor.shape)} no finite values", flush=True)
        return
    print(
        f"[stats] {name}: shape={list(tensor.shape)} "
        f"min={finite_values.min().item():.4f} max={finite_values.max().item():.4f} "
        f"mean={finite_values.mean().item():.4f} std={finite_values.std().item():.4f} "
        f"nonfinite={(~finite).sum().item()}",
        flush=True,
    )


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def denoising_sigmas(
    denoising_step_list: list[int],
    num_inference_steps: int,
    device: torch.device,
) -> torch.Tensor:
    full_sigmas = LTX2Scheduler().execute(steps=num_inference_steps)
    selected = []
    for timestep in denoising_step_list:
        target_sigma = float(timestep) / 1000.0
        idx = (full_sigmas - target_sigma).abs().argmin().item()
        selected.append(full_sigmas[idx])
    return torch.stack(selected).to(device)


def compute_latent_shapes(
    num_frames: int,
    height: int,
    width: int,
    fps: float = 24.0,
    batch_size: int = 1,
) -> tuple[list[int], list[int]]:
    if (num_frames - 1) % 8 != 0:
        raise ValueError(f"num_frames must be 1 + 8*k, got {num_frames}")
    latent_frames = 1 + (num_frames - 1) // 8
    latent_h = height // 32
    latent_w = width // 32

    # Matches the training benchmark shape calculation. LTX-2 audio latents are
    # aligned to 16 kHz / 160 hop / 4 downsample = 25 latent frames per second.
    video_duration = float(num_frames) / float(fps)
    audio_frames = round(video_duration * 25.0)
    return (
        [batch_size, latent_frames, 128, latent_h, latent_w],
        [batch_size, audio_frames, 128],
    )


def read_prompts(prompt: list[str] | None, prompt_file: str | None, num_prompts: int | None) -> list[str]:
    prompts: list[str] = []
    if prompt:
        prompts.extend(prompt)
    if prompt_file:
        with open(prompt_file, "r", encoding="utf-8") as handle:
            prompts.extend(line.strip() for line in handle if line.strip())
    if num_prompts is not None:
        prompts = prompts[:num_prompts]
    if not prompts:
        raise ValueError("No prompts provided. Use --prompt or --prompt-file.")
    return prompts


def build_generator(args: argparse.Namespace, device: torch.device, dtype: torch.dtype):
    from ltx_distillation.ode.ode_regression import LTX2ODERegression

    checkpoint_config = LTX2ODERegression._load_checkpoint_config(args.base_checkpoint)
    causal_config = CausalLTXModelConfig.from_checkpoint_config(
        checkpoint_config,
        num_frame_per_block=args.num_frame_per_block,
        num_frame_per_block_first=args.num_frame_per_block_first,
        enable_causal_log_rescale=args.enable_causal_log_rescale,
    )
    print(
        "[init] architecture "
        f"rope={causal_config.rope_type.value} "
        f"cross_attention_adaln={causal_config.cross_attention_adaln} "
        f"gated_attention={causal_config.apply_gated_attention}",
        flush=True,
    )
    from ltx_core.model.transformer import (
        checkpoint_fp8_module_names,
        convert_to_fp8_training,
        convert_to_static_fp8,
    )

    quantized_names = checkpoint_fp8_module_names(args.base_checkpoint)
    model = CausalLTXModel(causal_config).to(dtype=dtype)
    if args.fp8_mode == "static":
        convert_to_static_fp8(model, quantized_names)
    model = model.to(device=device)
    generator = CausalLTX2DiffusionWrapper(
        model=model,
        video_height=args.height,
        video_width=args.width,
        num_frame_per_block=args.num_frame_per_block,
        num_frame_per_block_first=args.num_frame_per_block_first,
        disable_causal_mask=args.disable_causal_mask,
    )

    print(f"[init] Loading base checkpoint: {args.base_checkpoint}", flush=True)
    base_sd = remap_state_dict_keys(
        load_checkpoint_state_dict(args.base_checkpoint),
        preserve_fp8=args.fp8_mode == "static",
    )
    try:
        generator.load_state_dict(base_sd, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Base checkpoint is not architecture-compatible with the metadata-built "
            "LTX-2.3 causal model."
        ) from exc
    print("[init] base load missing=0 unexpected=0", flush=True)

    print(f"[init] Loading distilled generator: {args.generator_ckpt}", flush=True)
    gen_sd = remap_state_dict_keys(
        load_checkpoint_state_dict(args.generator_ckpt, prefer_ema=args.use_ema),
        preserve_fp8=args.fp8_mode == "static",
    )
    if args.fp8_mode == "static":
        fp8_weights = {
            key.removesuffix(".weight")
            for key, value in gen_sd.items()
            if key.endswith(".weight") and value.dtype == torch.float8_e4m3fn
        }
        weight_scales = {key.removesuffix(".weight_scale") for key in gen_sd if key.endswith(".weight_scale")}
        input_scales = {key.removesuffix(".input_scale") for key in gen_sd if key.endswith(".input_scale")}
        if not (
            len(fp8_weights) == len(quantized_names)
            and fp8_weights == weight_scales == input_scales
        ):
            raise RuntimeError(
                "--fp8-mode static requires an exported static FP8 Student with "
                f"{len(quantized_names)} FP8 weights and complete weight/input scales; "
                f"got weights={len(fp8_weights)}, weight_scales={len(weight_scales)}, "
                f"input_scales={len(input_scales)}. Use --fp8-mode training together "
                "with --export-fp8-student for a BF16-master training checkpoint."
            )
    try:
        generator.load_state_dict(gen_sd, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "The student checkpoint was trained with a different causal architecture. "
            "A legacy 6-way Step-2 checkpoint cannot be loaded as an LTX-2.3 9-way model; "
            "restart Step-2 from the corrected LTX-2.3 base."
        ) from exc
    print(
        f"[init] generator load tensors={len(gen_sd)} "
        "missing=0 unexpected=0",
        flush=True,
    )
    if args.fp8_mode == "training":
        convert_to_fp8_training(model, quantized_names)

    generator.fp8_mode = args.fp8_mode
    generator.requires_grad_(False)
    return generator.eval()


def save_sample(
    video_latent: torch.Tensor,
    audio_latent: torch.Tensor | None,
    video_vae,
    audio_vae,
    output_path: Path,
    fps: int,
    audio_sample_rate: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
    from ltx_core.model.video_vae import decode_video as vae_decode_video
    from ltx_pipelines.utils.media_io import encode_video

    latent_for_vae = video_latent.permute(0, 2, 1, 3, 4)
    print_tensor_stats("video_latent", video_latent)
    decoded_frames = list(vae_decode_video(latent_for_vae, video_vae.decoder))
    video = torch.cat(decoded_frames, dim=0)
    print_tensor_stats("decoded_video_uint8", video)

    audio_waveform = None
    if audio_latent is not None:
        try:
            audio_waveform = vae_decode_audio(
                audio_latent.unflatten(-1, (8, 16)).permute(0, 2, 1, 3),
                audio_vae.decoder,
                audio_vae.vocoder,
            )
        except Exception as exc:
            print(f"[warn] audio decode failed for {output_path.name}: {exc}", flush=True)

    audio_for_video = audio_waveform.cpu().float() if audio_waveform is not None else None
    try:
        encode_video(
            video=video,
            fps=fps,
            audio=audio_for_video,
            audio_sample_rate=audio_sample_rate if audio_for_video is not None else None,
            output_path=str(output_path),
            video_chunks_number=1,
        )
    except Exception as exc:
        if audio_for_video is None:
            raise
        print(f"[warn] encode_video with audio failed for {output_path.name}: {exc}", flush=True)
        encode_video(
            video=video,
            fps=fps,
            audio=None,
            audio_sample_rate=None,
            output_path=str(output_path),
            video_chunks_number=1,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="OmniForcing causal AV inference")
    parser.add_argument(
        "--base-checkpoint",
        required=True,
        help="LTX-2 base .safetensors checkpoint used for model config, text encoder, and VAE.",
    )
    parser.add_argument(
        "--vae-checkpoint",
        default=None,
        help="Optional original LTX-2 checkpoint used to load video/audio VAEs. Defaults to --base-checkpoint.",
    )
    parser.add_argument("--gemma-path", required=True, help="Gemma text encoder directory.")
    parser.add_argument(
        "--generator-ckpt",
        required=True,
        help=(
            "OmniForcing causal checkpoint: .pt, .safetensors, or .safetensors.index.json. "
            "The .pt training checkpoint is read from its generator entry."
        ),
    )
    parser.add_argument("--prompt", action="append", help="Prompt text. Can be passed multiple times.")
    parser.add_argument("--prompt-file", default=None, help="Text file with one prompt per line.")
    parser.add_argument("--num-prompts", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/omniforcing_causal")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--start-index", type=int, default=0)

    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--audio-sample-rate", type=int, default=48000)

    parser.add_argument("--denoising-step-list", default="1000,909,725,421,0")
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--num-frame-per-block", type=int, default=3)
    parser.add_argument("--num-frame-per-block-first", type=int, default=4)
    parser.add_argument("--context-noise", type=int, default=0)
    parser.add_argument("--num-train-timestep", type=int, default=1000)
    parser.add_argument("--disable-causal-mask", action="store_true")
    parser.add_argument("--enable-causal-log-rescale", action="store_true")
    parser.add_argument(
        "--pipeline",
        choices=("ode-prefix", "kv-cache"),
        default="ode-prefix",
        help="Step-2 uses ode-prefix; kv-cache is retained for controlled diagnostics.",
    )
    parser.add_argument(
        "--transition-mode",
        choices=("euler", "renoise"),
        default="euler",
        help="ODE checkpoints should use Euler; renoise reproduces the legacy sampler.",
    )
    parser.add_argument("--use-bidirectional-bootstrap", action="store_true")
    parser.add_argument("--log-step-stats", action="store_true")

    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument(
        "--fp8-mode",
        choices=("static", "training"),
        default="static",
        help="Use an exported static FP8 checkpoint, or a training checkpoint for calibration/export.",
    )
    parser.add_argument(
        "--export-fp8-student",
        default=None,
        help="With --fp8-mode training, calibrate on the requested prompts and export a static FP8 student.",
    )
    parser.add_argument("--save-latents", action="store_true")
    args = parser.parse_args()

    prompts = read_prompts(args.prompt, args.prompt_file, args.num_prompts)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    dtype = torch.float32 if args.fp32 else torch.bfloat16
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"[init] device={device} dtype={dtype}", flush=True)
    generator = build_generator(args, device=device, dtype=dtype)

    registry = StateDictRegistry()
    text_encoder = create_text_encoder_wrapper(
        checkpoint_path=args.base_checkpoint,
        gemma_path=args.gemma_path,
        device=device,
        dtype=dtype,
        registry=registry,
    ).eval()
    vae_checkpoint = args.vae_checkpoint or args.base_checkpoint
    video_vae, audio_vae = create_vae_wrappers(
        checkpoint_path=vae_checkpoint,
        device=device,
        dtype=dtype,
        registry=registry,
    )
    checkpoint_audio_sample_rate = int(audio_vae.vocoder.output_sample_rate)
    if args.audio_sample_rate != checkpoint_audio_sample_rate:
        raise RuntimeError(
            "--audio-sample-rate does not match the checkpoint vocoder: "
            f"{args.audio_sample_rate} != {checkpoint_audio_sample_rate}"
        )

    sigmas = denoising_sigmas(
        denoising_step_list=parse_int_list(args.denoising_step_list),
        num_inference_steps=args.num_inference_steps,
        device=device,
    )
    print(f"[init] denoising_sigmas={sigmas.detach().cpu().tolist()}", flush=True)

    print(
        f"[init] pipeline={args.pipeline} transition_mode={args.transition_mode} "
        f"bidirectional_bootstrap={args.use_bidirectional_bootstrap}",
        flush=True,
    )
    if args.pipeline == "ode-prefix":
        pipeline = ODEAutoregressiveBenchmarkPipeline(
            generator=generator,
            add_noise_fn=add_noise,
            denoising_sigmas=sigmas,
            num_frame_per_block=args.num_frame_per_block,
            num_frame_per_block_first=args.num_frame_per_block_first,
            clear_cuda_cache_per_round=True,
            transition_mode=args.transition_mode,
            use_bidirectional_bootstrap=args.use_bidirectional_bootstrap,
            log_step_stats=args.log_step_stats,
        )
    else:
        if args.transition_mode != "renoise":
            print(
                "[warn] kv-cache pipeline currently uses its legacy renoise transition; "
                "--transition-mode is ignored.",
                flush=True,
            )
        pipeline = CausalAVInferencePipeline(
            generator=generator,
            add_noise_fn=add_noise,
            denoising_sigmas=sigmas,
            num_frame_per_block=args.num_frame_per_block,
            num_frame_per_block_first=args.num_frame_per_block_first,
            context_noise=args.context_noise,
            num_train_timestep=args.num_train_timestep,
            clear_cuda_cache_per_round=True,
        )

    video_shape, audio_shape = compute_latent_shapes(
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        fps=args.fps,
        batch_size=1,
    )
    print(f"[init] video_shape={video_shape} audio_shape={audio_shape}", flush=True)

    metadata = {
        "base_checkpoint": args.base_checkpoint,
        "vae_checkpoint": vae_checkpoint,
        "gemma_path": args.gemma_path,
        "generator_ckpt": args.generator_ckpt,
        "seed": args.seed,
        "prompts": prompts,
        "video_shape": video_shape,
        "audio_shape": audio_shape,
        "denoising_sigmas": sigmas.detach().cpu().tolist(),
        "pipeline": args.pipeline,
        "transition_mode": args.transition_mode,
        "use_bidirectional_bootstrap": args.use_bidirectional_bootstrap,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    calibrator = None
    if args.export_fp8_student:
        if args.fp8_mode != "training":
            raise ValueError("--export-fp8-student requires --fp8-mode training")
        from ltx_core.model.transformer import FP8InputScaleCalibrator
        calibrator = FP8InputScaleCalibrator(generator)
        calibrator.__enter__()

    for local_idx, prompt in enumerate(prompts):
        prompt_idx = args.start_index + local_idx
        prompt_seed = args.seed + prompt_idx
        print(f"[infer] sample_{prompt_idx:03d} seed={prompt_seed}: {prompt}", flush=True)

        with torch.no_grad():
            conditional_dict = text_encoder(text_prompts=[prompt])
            fork_devices = [torch.cuda.current_device()] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=fork_devices):
                torch.manual_seed(prompt_seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed(prompt_seed)
                video_latent, audio_latent = pipeline.generate(
                    video_shape=tuple(video_shape),
                    audio_shape=tuple(audio_shape),
                    conditional_dict=conditional_dict,
                )

        sample_path = output_dir / f"sample_{prompt_idx:03d}.mp4"
        save_sample(
            video_latent=video_latent,
            audio_latent=audio_latent,
            video_vae=video_vae,
            audio_vae=audio_vae,
            output_path=sample_path,
            fps=args.fps,
            audio_sample_rate=args.audio_sample_rate,
        )
        (output_dir / f"sample_{prompt_idx:03d}.txt").write_text(prompt + "\n", encoding="utf-8")

        if args.save_latents:
            torch.save(
                {
                    "prompt": prompt,
                    "seed": prompt_seed,
                    "video_latent": video_latent.detach().cpu(),
                    "audio_latent": audio_latent.detach().cpu() if audio_latent is not None else None,
                },
                output_dir / f"sample_{prompt_idx:03d}_latents.pt",
            )

        del video_latent, audio_latent, conditional_dict
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if calibrator is not None:
        calibrator.__exit__(None, None, None)
        from ltx_core.model.transformer import fp8_inference_state_dict
        from safetensors.torch import save_file
        fp8_state = fp8_inference_state_dict(generator, calibrator.input_scales())
        quantized_layers = {
            key.removesuffix(".weight"): {"format": "float8_e4m3fn"}
            for key, value in fp8_state.items()
            if key.endswith(".weight") and value.dtype == torch.float8_e4m3fn
        }
        if len(quantized_layers) != 1496:
            raise RuntimeError(
                f"Refusing incomplete FP8 Student export: quantized_layers={len(quantized_layers)}, expected=1496"
            )
        from ltx_distillation.ode.ode_regression import LTX2ODERegression
        export_config = LTX2ODERegression._load_checkpoint_config(args.base_checkpoint)
        from safetensors import safe_open
        with safe_open(args.base_checkpoint, framework="pt", device="cpu") as handle:
            base_metadata = handle.metadata() or {}
        export_metadata = {
            "config": json.dumps(export_config),
            "model_version": base_metadata.get("model_version", "2.3.rc1"),
            "checkpoint_kind": "omniforcing_causal_student_static_fp8",
            "student_format_version": "1",
            "_quantization_metadata": json.dumps({"format_version": "1.0", "layers": quantized_layers}),
        }
        export_path = Path(args.export_fp8_student)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(fp8_state, str(export_path), metadata=export_metadata)
        print(f"[export] static FP8 student saved to {args.export_fp8_student}", flush=True)

    print(f"[done] saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
