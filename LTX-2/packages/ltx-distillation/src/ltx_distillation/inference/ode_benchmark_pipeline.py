"""Autoregressive benchmark pipeline for ODE-init causal training.

This pipeline implements blockwise autoregressive generation without relying on
the unfinished KV-cache runtime path in the current tracked causal wrapper.
Each block is generated with a prefix rerun strategy:
- previous blocks stay fixed at sigma=0 (clean causal context)
- the current block follows the multi-step denoising schedule
- only the current block is updated after each denoising step

The default Euler transition matches both ODE-pair generation and the official
LTX sampling pipelines. The legacy fresh-noise transition remains available for
controlled comparisons.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ltx_causal.attention.mask_builder import (
    compute_aligned_audio_frames,
    compute_av_blocks,
)
from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_distillation.inference.bidirectional_pipeline import BidirectionalAVInferencePipeline


class ODEAutoregressiveBenchmarkPipeline:
    """Prefix-rerun autoregressive pipeline for ODE benchmark inference."""

    def __init__(
        self,
        generator: nn.Module,
        add_noise_fn,
        denoising_sigmas: torch.Tensor,
        num_frame_per_block: int = 3,
        num_frame_per_block_first: int = 4,
        clear_cuda_cache_per_round: bool = True,
        transition_mode: str = "euler",
        use_bidirectional_bootstrap: bool = False,
        log_step_stats: bool = False,
    ):
        if denoising_sigmas.ndim != 1 or denoising_sigmas.numel() < 2:
            raise ValueError(
                "denoising_sigmas must be a 1D tensor with at least 2 entries"
            )

        if transition_mode not in {"euler", "renoise"}:
            raise ValueError(
                f"transition_mode must be 'euler' or 'renoise', got {transition_mode!r}"
            )

        self.generator = generator
        self.add_noise_fn = add_noise_fn
        self.denoising_sigmas = denoising_sigmas
        self.num_frame_per_block = max(1, int(num_frame_per_block))
        self.num_frame_per_block_first = max(1, int(num_frame_per_block_first))
        self.clear_cuda_cache_per_round = bool(clear_cuda_cache_per_round)
        self.transition_mode = transition_mode
        self.use_bidirectional_bootstrap = bool(use_bidirectional_bootstrap)
        self.log_step_stats = bool(log_step_stats)
        self.euler_step = EulerDiffusionStep()

    def _get_bootstrap_generator(self) -> nn.Module:
        get_delegate = getattr(self.generator, "_get_bidirectional_delegate", None)
        if callable(get_delegate):
            delegate = get_delegate()
            device, dtype = self._module_device_dtype(self.generator)
            return delegate.to(device=device, dtype=dtype)
        return self.generator

    def _release_bootstrap_generator(self, bootstrap_generator: nn.Module) -> None:
        if bootstrap_generator is self.generator:
            return
        bootstrap_generator.to(device="cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _module_device_dtype(module: nn.Module) -> Tuple[torch.device, torch.dtype]:
        param = next(module.parameters())
        return param.device, param.dtype

    @staticmethod
    def _zeros_sigma(
        batch_size: int,
        frames: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.zeros((batch_size, frames), device=device, dtype=dtype)

    @staticmethod
    def _full_sigma(
        sigma: torch.Tensor,
        batch_size: int,
        frames: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        sigma_value = sigma.to(device=device, dtype=dtype)
        return sigma_value.expand(batch_size, frames)

    def _renoise_block(self, clean_block: torch.Tensor, next_sigma: torch.Tensor) -> torch.Tensor:
        if clean_block is None:
            return None

        batch_size = clean_block.shape[0]
        num_frames = clean_block.shape[1]
        sigma = self._full_sigma(
            next_sigma,
            batch_size=batch_size,
            frames=num_frames,
            device=clean_block.device,
            dtype=clean_block.dtype,
        )
        return self.add_noise_fn(
            clean_block,
            torch.randn_like(clean_block),
            sigma,
        )

    def _advance_block(
        self,
        sample: torch.Tensor,
        denoised_sample: torch.Tensor,
        step_index: int,
    ) -> torch.Tensor:
        if self.transition_mode == "euler":
            return self.euler_step.step(
                sample=sample,
                denoised_sample=denoised_sample,
                sigmas=self.denoising_sigmas,
                step_index=step_index,
            )

        next_sigma = self.denoising_sigmas[step_index + 1]
        if float(next_sigma.item()) > 0.0:
            return self._renoise_block(denoised_sample, next_sigma)
        return denoised_sample

    def _log_step(
        self,
        block_index: int,
        step_index: int,
        sigma: torch.Tensor,
        sample: torch.Tensor,
        denoised_sample: torch.Tensor,
    ) -> None:
        if not self.log_step_stats:
            return
        sample_float = sample.detach().float()
        denoised_float = denoised_sample.detach().float()
        print(
            f"[step2] block={block_index} step={step_index} "
            f"sigma={float(sigma.item()):.6f} "
            f"input_mean={sample_float.mean().item():.4f} "
            f"input_std={sample_float.std().item():.4f} "
            f"x0_mean={denoised_float.mean().item():.4f} "
            f"x0_std={denoised_float.std().item():.4f}",
            flush=True,
        )

    @staticmethod
    def _merge_bootstrap_blocks(blocks):
        if len(blocks) < 2 or blocks[0].video_frames != 1:
            return blocks

        bootstrap = type(blocks[0])(
            block_idx=0,
            video_start=blocks[0].video_start,
            video_end=blocks[1].video_end,
            audio_start=blocks[0].audio_start,
            audio_end=blocks[1].audio_end,
        )
        return [bootstrap, *blocks[2:]]

    @torch.no_grad()
    def generate(
        self,
        video_shape: Tuple[int, ...],
        audio_shape: Optional[Tuple[int, ...]],
        conditional_dict: Dict[str, Any],
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if len(video_shape) != 5:
            raise ValueError(f"Expected video_shape=[B,F,C,H,W], got {video_shape}")

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        device, dtype = self._module_device_dtype(self.generator)

        batch_size = video_shape[0]
        total_video_frames = video_shape[1]
        blocks = compute_av_blocks(
            total_video_latent_frames=total_video_frames,
            num_frame_per_block=self.num_frame_per_block,
            num_frame_per_block_first=self.num_frame_per_block_first,
        )

        video = torch.zeros(video_shape, device=device, dtype=dtype)
        audio = None

        if audio_shape is not None:
            if len(audio_shape) != 3:
                raise ValueError(f"Expected audio_shape=[B,F,C], got {audio_shape}")
            expected_audio_frames = compute_aligned_audio_frames(
                total_video_latent_frames=total_video_frames,
                num_frame_per_block=self.num_frame_per_block,
                num_frame_per_block_first=self.num_frame_per_block_first,
            )
            if audio_shape[1] != expected_audio_frames:
                raise ValueError(
                    "audio_shape does not match causal block alignment: "
                    f"got F_a={audio_shape[1]}, expected {expected_audio_frames}"
                )
            audio = torch.zeros(audio_shape, device=device, dtype=dtype)

        for block in blocks:
            if block.block_idx == 0 and self.use_bidirectional_bootstrap:
                bootstrap_generator = self._get_bootstrap_generator()
                try:
                    bootstrap_pipeline = BidirectionalAVInferencePipeline(
                        generator=bootstrap_generator,
                        add_noise_fn=self.add_noise_fn,
                        denoising_sigmas=self.denoising_sigmas,
                    )
                    bootstrap_video_shape = (batch_size, block.video_frames, *video_shape[2:])
                    bootstrap_audio_shape = None
                    if audio is not None:
                        bootstrap_audio_shape = (batch_size, block.audio_frames, audio_shape[2])
                    current_video, current_audio = bootstrap_pipeline.generate(
                        video_shape=bootstrap_video_shape,
                        audio_shape=bootstrap_audio_shape,
                        conditional_dict=conditional_dict,
                        seed=seed,
                    )
                finally:
                    self._release_bootstrap_generator(bootstrap_generator)
                video[:, block.video_start:block.video_end] = current_video
                if audio is not None and current_audio is not None:
                    audio[:, block.audio_start:block.audio_end] = current_audio
                continue

            current_video = torch.randn(
                (batch_size, block.video_frames, *video_shape[2:]),
                device=device,
                dtype=dtype,
            )
            current_audio = None
            if audio is not None:
                current_audio = torch.randn(
                    (batch_size, block.audio_frames, audio_shape[2]),
                    device=device,
                    dtype=dtype,
                )

            prev_video = video[:, :block.video_start]
            prev_audio = audio[:, :block.audio_start] if audio is not None else None

            for sigma_idx, sigma in enumerate(self.denoising_sigmas[:-1]):
                model_input_video = current_video
                model_input_audio = current_audio
                prefix_video = torch.cat([prev_video, current_video], dim=1)
                video_sigma = torch.cat(
                    [
                        self._zeros_sigma(batch_size, prev_video.shape[1], device, dtype),
                        self._full_sigma(sigma, batch_size, current_video.shape[1], device, dtype),
                    ],
                    dim=1,
                )

                prefix_audio = None
                audio_sigma = None
                if current_audio is not None:
                    prefix_audio = torch.cat([prev_audio, current_audio], dim=1)
                    audio_sigma = torch.cat(
                        [
                            self._zeros_sigma(batch_size, prev_audio.shape[1], device, dtype),
                            self._full_sigma(sigma, batch_size, current_audio.shape[1], device, dtype),
                        ],
                        dim=1,
                    )

                pred_video_prefix, pred_audio_prefix = self.generator(
                    noisy_image_or_video=prefix_video,
                    conditional_dict=conditional_dict,
                    timestep=video_sigma,
                    noisy_audio=prefix_audio,
                    audio_timestep=audio_sigma,
                    use_causal_timestep=False,
                    force_bidirectional=False,
                )

                current_video = pred_video_prefix[:, block.video_start:block.video_end]
                if current_audio is not None:
                    if pred_audio_prefix is None:
                        raise RuntimeError(
                            "Generator returned no audio prediction for audio benchmark inference"
                        )
                    current_audio = pred_audio_prefix[:, block.audio_start:block.audio_end]

                self._log_step(
                    block_index=block.block_idx,
                    step_index=sigma_idx,
                    sigma=sigma,
                    sample=model_input_video,
                    denoised_sample=current_video,
                )
                current_video = self._advance_block(
                    sample=model_input_video,
                    denoised_sample=current_video,
                    step_index=sigma_idx,
                )
                if current_audio is not None:
                    current_audio = self._advance_block(
                        sample=model_input_audio,
                        denoised_sample=current_audio,
                        step_index=sigma_idx,
                    )

                if self.clear_cuda_cache_per_round:
                    torch.cuda.empty_cache()

            video[:, block.video_start:block.video_end] = current_video
            if audio is not None and current_audio is not None:
                audio[:, block.audio_start:block.audio_end] = current_audio

        return video, audio
