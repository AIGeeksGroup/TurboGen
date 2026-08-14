"""
DMD Distillation Training Script for LTX-2.

Usage:
    torchrun --nproc_per_node=8 -m ltx_distillation.train_distillation \
        --config_path configs/ltx2_bidirectional_dmd.yaml
"""

import argparse
import json
import math
import os
import shutil
import time
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import wandb
from omegaconf import OmegaConf

from ltx_distillation.dmd import LTX2DMD
from ltx_distillation.data import TextDataset, ODERegressionLMDBDataset
from ltx_distillation.util import (
    launch_distributed_job,
    set_seed,
    init_logging_folder,
    fsdp_wrap,
    fsdp_state_dict,
    load_fsdp_state_dict,
    barrier,
    ResumableDataIterator,
    ResumableDistributedSampler,
    capture_rng_state,
    restore_rng_state,
    persist_checkpoint_to_split_storage,
    upload_checkpoint_to_hf,
    validate_artifact_upload_config,
    wandb_video_from_path,
    artifact_paths_match,
)


def compute_latent_shapes(
    num_frames: int,
    video_height: int,
    video_width: int,
    batch_size: int = 1,
    latent_channels: int = 128,
    vae_temporal_compression: int = 8,
    vae_spatial_compression: int = 32,
    video_fps: float = 24.0,
    audio_sample_rate: int = 16000,
    audio_hop_length: int = 160,
    audio_latent_downsample: int = 4,
) -> Tuple[list, list]:
    """
    Compute latent shapes from video frames and resolution.

    Calculation logic matches LTX-2 native implementation (see ltx_core/types.py):
    - Video: frames = (num_frames - 1) // 8 + 1
    - Audio: frames = round(video_duration * audio_latent_fps)
             where audio_latent_fps = sample_rate / hop_length / downsample = 25

    Args:
        num_frames: Number of raw video frames (must satisfy 1 + 8*k constraint)
        video_height: Video height in pixels
        video_width: Video width in pixels
        batch_size: Batch size
        latent_channels: Number of latent channels
        vae_temporal_compression: VAE temporal compression ratio (default 8)
        vae_spatial_compression: VAE spatial compression ratio (default 32)
        video_fps: Video frame rate (default 24.0)
        audio_sample_rate: Audio sample rate (default 16000)
        audio_hop_length: Audio hop length (default 160)
        audio_latent_downsample: Audio latent downsampling factor (default 4)

    Returns:
        (video_shape, audio_shape)
        - video_shape: [B, latent_frames, C, H, W]
        - audio_shape: [B, audio_frames, C]
    """
    # Check frame count constraint
    if (num_frames - 1) % vae_temporal_compression != 0:
        raise ValueError(
            f"num_frames must be 1 + 8*k, got {num_frames}. "
            f"Valid values: 1, 9, 17, 25, ..., 121, ..., 241, ..."
        )

    # Compute video latent frames (matches LTX types.py:73)
    latent_frames = 1 + (num_frames - 1) // vae_temporal_compression

    # Compute latent spatial dimensions
    latent_h = video_height // vae_spatial_compression
    latent_w = video_width // vae_spatial_compression

    # Compute audio frames (matches LTX types.py:140-156)
    # video_duration = num_frames / video_fps
    # audio_latent_fps = sample_rate / hop_length / downsample = 16000/160/4 = 25
    # audio_frames = round(video_duration * audio_latent_fps)
    video_duration = float(num_frames) / float(video_fps)
    audio_latent_fps = float(audio_sample_rate) / float(audio_hop_length) / float(audio_latent_downsample)
    audio_frames = round(video_duration * audio_latent_fps)

    video_shape = [batch_size, latent_frames, latent_channels, latent_h, latent_w]
    audio_shape = [batch_size, audio_frames, latent_channels]

    return video_shape, audio_shape


class Trainer:
    """
    DMD Distillation Trainer for LTX-2.

    Handles:
    - Distributed training with FSDP
    - Alternating generator and critic training
    - Checkpointing and logging
    """

    def __init__(self, config):
        self.config = config
        self.training_stage = str(getattr(config, "training_stage", "dmd"))
        validate_artifact_upload_config(config)

        # Initialize distributed environment
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        rank, world_size, local_rank = launch_distributed_job()
        self.global_rank = rank
        self.world_size = world_size
        self.local_rank = local_rank

        expected_world_size = int(getattr(config, "expected_world_size", 0))
        if expected_world_size > 0 and self.world_size != expected_world_size:
            raise RuntimeError(
                f"Expected world_size={expected_world_size}, got {self.world_size}. "
                "Launch this configuration with the required number of GPUs."
            )

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = self.global_rank == 0

        # Set seed
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            if world_size > 1:
                dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + self.global_rank)

        # Initialize logging (main process only) then broadcast output_path
        # to all ranks so every rank can save benchmark files to shared FS.
        # Avoid NCCL object broadcast here: on this cluster it can fail during
        # early initialization with socket connection errors. Use the shared
        # filesystem plus a barrier instead.
        sync_token = f"{os.environ.get('MASTER_ADDR', 'localhost')}_{os.environ.get('MASTER_PORT', '29500')}"
        sync_token = sync_token.replace("/", "_").replace(":", "_")
        shared_run_path_file = os.path.join(config.output_path, f".run_path_{sync_token}.txt")
        if self.is_main_process:
            self.output_path, self.wandb_folder = init_logging_folder(config)
            os.makedirs(config.output_path, exist_ok=True)
            with open(shared_run_path_file, "w", encoding="utf-8") as f:
                f.write(self.output_path)
        else:
            self.output_path = None
            self.wandb_folder = None

        barrier()

        if not self.is_main_process:
            with open(shared_run_path_file, "r", encoding="utf-8") as f:
                self.output_path = f.read().strip()

        self.wandb_folder = os.path.join(self.output_path, "wandb")

        barrier()
        if self.is_main_process:
            try:
                os.remove(shared_run_path_file)
            except FileNotFoundError:
                pass

        # Initialize DMD module
        self.dmd = LTX2DMD(config, device=self.device)

        # Initialize models from checkpoints BEFORE FSDP wrapping
        # Models must exist before we can wrap them with FSDP
        self.dmd.init_models()

        self._validate_preinstalled_bidirectional_delegate()

        # FSDP wrapping
        self._wrap_with_fsdp()

        # Optimizers
        weight_decay = getattr(config, "weight_decay", 0.0)
        generator_lr = getattr(config, "generator_lr", config.lr)
        critic_lr = getattr(config, "critic_lr", config.lr)

        self.generator_optimizer = torch.optim.AdamW(
            [p for p in self.dmd.generator.parameters() if p.requires_grad],
            lr=generator_lr,
            betas=(config.beta1, config.beta2),
            weight_decay=weight_decay,
        )

        beta1_critic = getattr(config, "beta1_critic", config.beta1)
        beta2_critic = getattr(config, "beta2_critic", config.beta2)
        self.critic_optimizer = torch.optim.AdamW(
            [p for p in self.dmd.fake_score.parameters() if p.requires_grad],
            lr=critic_lr,
            betas=(beta1_critic, beta2_critic),
            weight_decay=weight_decay,
        )

        # Learning rate schedulers
        self.generator_scheduler = self._create_lr_scheduler(self.generator_optimizer)
        self.critic_scheduler = self._create_lr_scheduler(self.critic_optimizer)

        # EMA (initialized lazily at ema_start_step to save memory for early steps)
        self.ema_weight = getattr(config, "ema_weight", 0.0)
        self.ema_start_step = getattr(config, "ema_start_step", 200)
        self.generator_ema = None

        # Dataloader
        self._init_dataloader()

        # Benchmark prompts (for periodic inference visualization)
        self._init_benchmark_prompts()

        self.step = 0
        self.gradient_accumulation_steps = int(
            getattr(config, "gradient_accumulation_steps", 1)
        )
        self.max_grad_norm = getattr(config, "max_grad_norm", 10.0)
        self.log_iters = int(getattr(config, "log_iters", 0))
        self.save_iters = int(getattr(config, "save_iters", self.log_iters))
        self.last_saved_step = None
        self.layerwise_grad_log_interval = max(
            1, int(getattr(config, "layerwise_grad_log_interval", config.log_iters))
        )
        self.previous_time = None

        # Exact resume is restored only after models, optimizers, schedulers and
        # the resumable DataLoader have all been constructed.
        resume_ckpt = getattr(config, "resume_checkpoint", None)
        if resume_ckpt:
            self._restore_training_state(str(resume_ckpt))

    def _create_lr_scheduler(self, optimizer):
        """Create learning rate scheduler based on config.

        IMPORTANT: The scheduler is NOT stepped per-optimizer-call. Instead,
        both generator and critic schedulers are stepped once per global
        training step (in the training loop), so they stay synchronized
        even though the generator only trains every dfake_gen_update_ratio steps.

        Supported scheduler_type values:
        - None / "constant": No scheduling (constant LR)
        - "cosine_warmup": Linear warmup then cosine decay to min_lr
        """
        scheduler_type = getattr(self.config, "scheduler_type", None)
        if scheduler_type is None or scheduler_type == "constant":
            return None

        warmup_steps = getattr(self.config, "warmup_steps", 1000)
        max_steps = getattr(self.config, "max_steps", 20000)
        min_lr = getattr(self.config, "min_lr", 1e-7)
        base_lr = optimizer.param_groups[0]["lr"]

        if scheduler_type == "cosine_warmup":
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / max(1, warmup_steps)
                else:
                    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
                    progress = min(progress, 1.0)
                    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                    return max(min_lr / base_lr, cosine_decay)

            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            raise ValueError(f"Unknown scheduler_type: {scheduler_type}")


    def _move_conditioning_to_device(self, conditioning):
        return {
            key: value.to(device=self.device, dtype=self.dtype, non_blocking=True)
            if torch.is_tensor(value) else value
            for key, value in conditioning.items()
        }

    def _validate_preinstalled_bidirectional_delegate(self) -> None:
        """Fail early if causal benchmark fallback would need lazy delegate construction."""
        if not getattr(self.dmd, "generator_use_causal_wrapper", False):
            return

        has_delegate = getattr(self.dmd.generator, "has_bidirectional_delegate", None)
        if callable(has_delegate) and has_delegate():
            return

        raise RuntimeError(
            "Causal Stage-3 generator is missing a pre-installed bidirectional delegate before FSDP "
            "wrapping. Install it during model init (for example from "
            "bootstrap_bidirectional_ckpt_path / generator_ckpt) instead of relying on lazy "
            "delegate construction at benchmark time."
        )

    def _wrap_with_fsdp(self):
        """Wrap models with FSDP for distributed training."""
        config = self.config

        if not getattr(config, "fsdp_wrap_during_model_init", False):
            self.dmd.generator = fsdp_wrap(
                self.dmd.generator,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy=config.generator_fsdp_wrap_strategy,
            )

            if getattr(self.dmd.real_score, "fp8_mode", None) != "static":
                self.dmd.real_score = fsdp_wrap(
                    self.dmd.real_score,
                    sharding_strategy=config.sharding_strategy,
                    mixed_precision=config.mixed_precision,
                    wrap_strategy=config.real_score_fsdp_wrap_strategy,
                )

            self.dmd.fake_score = fsdp_wrap(
                self.dmd.fake_score,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy=config.fake_score_fsdp_wrap_strategy,
            )

            if getattr(config, "text_encoder_device", "cuda") != "cpu":
                self.dmd.text_encoder = fsdp_wrap(
                    self.dmd.text_encoder,
                    sharding_strategy=config.sharding_strategy,
                    mixed_precision=config.mixed_precision,
                    wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
                )

        # Keep VAEs on CPU to save GPU memory during training.
        # They are only needed for periodic visualization and benchmark decoding.
        # Use _vae_to_device() / _vae_to_cpu() to move them on-demand.
        if self.dmd.video_vae is not None:
            self.dmd.video_vae = self.dmd.video_vae.to(dtype=self.dtype)
        if self.dmd.audio_vae is not None:
            self.dmd.audio_vae = self.dmd.audio_vae.to(dtype=self.dtype)

    def _init_dataloader(self):
        """Initialize data loader."""
        from ltx_distillation.data import collate_text_prompts, collate_ode_data

        config = self.config

        self.backward_simulation = getattr(config, "backward_simulation", True)

        if self.backward_simulation:
            dataset = TextDataset(config.data_path)
            collate_fn = collate_text_prompts
        else:
            dataset = ODERegressionLMDBDataset(
                config.data_path,
                max_pair=int(1e8),
            )
            if getattr(config, "require_ode_manifest", False):
                manifest = getattr(dataset, "manifest", None)
                if not isinstance(manifest, dict):
                    raise RuntimeError(
                        f"Required LTX-2.3 ODE manifest is missing: {config.data_path}"
                    )
                if manifest.get("format_version") != 4:
                    raise RuntimeError(
                        f"Unsupported ODE format: {manifest.get('format_version')}"
                    )
                if manifest.get("producer") != "omniforcing-ltx23-full-architecture-v2":
                    raise RuntimeError(
                        f"Unsupported ODE producer: {manifest.get('producer')}"
                    )
                teacher_checkpoint = str(manifest.get("teacher_checkpoint", ""))
                expected_checkpoint = str(config.checkpoint_path)
                if not artifact_paths_match(teacher_checkpoint, expected_checkpoint):
                    raise RuntimeError(
                        "ODE teacher checkpoint mismatch: "
                        f"{teacher_checkpoint} != {expected_checkpoint}"
                    )
                generation_config = manifest.get("generation_config") or {}
                if list(generation_config.get("denoising_step_list") or []) != list(
                    config.denoising_step_list
                ):
                    raise RuntimeError("ODE denoising schedule does not match training")
                for key, expected in {
                    "video_height": int(config.video_height),
                    "video_width": int(config.video_width),
                }.items():
                    if generation_config.get(key) != expected:
                        raise RuntimeError(
                            f"ODE {key} mismatch: {generation_config.get(key)} != {expected}"
                        )
            collate_fn = collate_ode_data

        sampler = ResumableDistributedSampler(
            dataset,
            shuffle=True,
            drop_last=True,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            generator=torch.Generator(),
        )

        data_seed = int(config.seed) + self.global_rank * 1_000_003
        self.dataloader = ResumableDataIterator(dataloader, seed=data_seed)

    def _init_benchmark_prompts(self):
        """
        Load fixed benchmark prompts from the training prompt file.

        Reads the first ``benchmark_num_prompts`` lines from ``config.data_path``
        so that every benchmark run uses exactly the same prompts for comparison.

        **All ranks** load the prompts because FSDP-wrapped models require all
        ranks to participate in forward passes during benchmark inference.
        """
        config = self.config
        self.benchmark_enabled = getattr(config, "benchmark_enabled", True)
        self.benchmark_iters = int(getattr(config, "benchmark_iters", config.log_iters))
        self.benchmark_seed = getattr(config, "benchmark_seed", 12345)
        self.benchmark_num_prompts = getattr(config, "benchmark_num_prompts", 2)
        self.benchmark_video_fps = getattr(config, "benchmark_video_fps", 24)
        self.benchmark_audio_sample_rate = getattr(config, "benchmark_audio_sample_rate", 48000)
        if self.dmd.audio_vae is not None and self.dmd.audio_vae.vocoder is not None:
            model_audio_sample_rate = int(self.dmd.audio_vae.vocoder.output_sample_rate)
            if int(self.benchmark_audio_sample_rate) != model_audio_sample_rate:
                raise RuntimeError(
                    "benchmark_audio_sample_rate does not match the checkpoint vocoder: "
                    f"{self.benchmark_audio_sample_rate} != {model_audio_sample_rate}"
                )
        self.wandb_video_required = bool(
            getattr(config, "wandb_video_required", False)
        )
        self.benchmark_mode = str(getattr(config, "benchmark_mode", "bidirectional")).lower()
        if self.benchmark_mode not in {"bidirectional", "causal"}:
            if self.is_main_process:
                print(f"[Benchmark] Invalid benchmark_mode={self.benchmark_mode}, falling back to bidirectional.")
            self.benchmark_mode = "bidirectional"
        self.benchmark_num_frame_per_block = int(getattr(config, "benchmark_num_frame_per_block", getattr(config, "num_frame_per_block", 3)))
        self.benchmark_use_kv_cache = bool(getattr(config, "benchmark_use_kv_cache", False))
        self.benchmark_clear_cuda_cache_per_round = bool(getattr(config, "benchmark_clear_cuda_cache_per_round", True))
        self.benchmark_prompts = []

        if self.benchmark_iters <= 0:
            self.benchmark_enabled = False
            if self.is_main_process:
                print("[Benchmark] Disabled because benchmark_iters <= 0.")

        if self.benchmark_mode == "causal" and self.benchmark_use_kv_cache:
            if self.is_main_process:
                print(
                    "[Benchmark] benchmark_use_kv_cache=true requested, but the current "
                    "causal wrapper does not expose a stable KV-cache runtime API. "
                    "Falling back to prefix-rerun autoregressive benchmark mode."
                )
            self.benchmark_use_kv_cache = False

        if not self.benchmark_enabled:
            return

        try:
            # When backward_simulation=false, data_path is an LMDB directory.
            # Use benchmark_prompt_file if specified, otherwise fall back to data_path.
            data_path = getattr(config, "benchmark_prompt_file", None) or config.data_path
            with open(data_path, "r", encoding="utf-8") as f:
                all_prompts = [line.strip() for line in f if line.strip()]
            self.benchmark_prompts = all_prompts[: self.benchmark_num_prompts]
            if self.is_main_process:
                print(f"[Benchmark] Loaded {len(self.benchmark_prompts)} prompts from {data_path}")
                print(f"[Benchmark] mode={self.benchmark_mode}, kv_cache={self.benchmark_use_kv_cache}, frames_per_block={self.benchmark_num_frame_per_block}")
                for i, p in enumerate(self.benchmark_prompts):
                    print(f"  [{i}] {p[:80]}{'...' if len(p) > 80 else ''}")
        except Exception as e:
            if self.is_main_process:
                print(f"[Benchmark] Failed to load prompts: {e}")
            self.benchmark_enabled = False

    def _vae_to_device(self):
        """Move VAEs to GPU for decoding (visualization / benchmark)."""
        if self.dmd.video_vae is not None:
            self.dmd.video_vae = self.dmd.video_vae.to(device=self.device)
        if self.dmd.audio_vae is not None:
            self.dmd.audio_vae = self.dmd.audio_vae.to(device=self.device)

    def _vae_to_cpu(self):
        """Offload VAEs back to CPU to free GPU memory."""
        if self.dmd.video_vae is not None:
            self.dmd.video_vae = self.dmd.video_vae.to(device="cpu")
        if self.dmd.audio_vae is not None:
            self.dmd.audio_vae = self.dmd.audio_vae.to(device="cpu")
        torch.cuda.empty_cache()

    def _resume_signature(self) -> dict:
        config = self.config
        scheduler_type = str(getattr(config, "scheduler_type", "constant"))
        return {
            "training_stage": self.training_stage,
            "base_checkpoint": os.path.realpath(str(config.checkpoint_path)),
            "data_path": os.path.realpath(str(config.data_path)),
            "generator_task": str(config.generator_task),
            "training_mode": str(getattr(config, "training_mode", "")),
            "backward_simulation": bool(getattr(config, "backward_simulation", True)),
            "denoising_step_list": list(config.denoising_step_list),
            "batch_size": int(config.batch_size),
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "dfake_gen_update_ratio": int(config.dfake_gen_update_ratio),
            "scheduler_type": scheduler_type,
            "warmup_steps": int(getattr(config, "warmup_steps", 0)),
            "scheduler_max_steps": (
                int(config.max_steps) if scheduler_type != "constant" else None
            ),
            "min_lr": (
                float(getattr(config, "min_lr", 1e-7))
                if scheduler_type != "constant"
                else None
            ),
            "generator_lr": float(getattr(config, "generator_lr", config.lr)),
            "critic_lr": float(getattr(config, "critic_lr", config.lr)),
            "beta1": float(config.beta1),
            "beta2": float(config.beta2),
            "beta1_critic": float(getattr(config, "beta1_critic", config.beta1)),
            "beta2_critic": float(getattr(config, "beta2_critic", config.beta2)),
            "weight_decay": float(getattr(config, "weight_decay", 0.0)),
            "seed": int(config.seed),
            "mixed_precision": bool(config.mixed_precision),
            "sharding_strategy": str(config.sharding_strategy),
            "generator_fsdp_wrap_strategy": str(config.generator_fsdp_wrap_strategy),
            "fake_score_fsdp_wrap_strategy": str(config.fake_score_fsdp_wrap_strategy),
            "teacher_fp8_mode": str(getattr(config, "teacher_fp8_mode", "")),
            "trainable_fp8_mode": str(getattr(config, "trainable_fp8_mode", "")),
            "distillation_loss": str(config.distillation_loss),
            "denoising_loss_type": str(config.denoising_loss_type),
            "num_train_timestep": int(config.num_train_timestep),
            "disable_causal_mask": bool(getattr(config, "disable_causal_mask", False)),
            "context_noise": float(getattr(config, "context_noise", 0.0)),
            "num_frames": int(config.num_frames),
            "video_height": int(config.video_height),
            "video_width": int(config.video_width),
            "num_frame_per_block": int(getattr(config, "num_frame_per_block", 0)),
            "num_frame_per_block_first": int(
                getattr(config, "num_frame_per_block_first", 0)
            ),
            "real_video_guidance_scale": float(config.real_video_guidance_scale),
            "real_audio_guidance_scale": float(config.real_audio_guidance_scale),
        }

    @staticmethod
    def _resume_path_signature_matches(saved_path, current_path) -> bool:
        if saved_path == current_path:
            return True
        if not saved_path or not current_path:
            return False
        saved_name = os.path.basename(os.path.normpath(str(saved_path)))
        current_name = os.path.basename(os.path.normpath(str(current_path)))
        return bool(saved_name and saved_name == current_name)

    @classmethod
    def _resume_signature_mismatches(cls, saved_signature, current_signature) -> dict:
        saved_signature = saved_signature or {}
        current_signature = current_signature or {}
        path_keys = {"base_checkpoint", "data_path"}
        mismatches = {}
        for key in sorted(set(saved_signature) | set(current_signature)):
            saved_value = saved_signature.get(key)
            current_value = current_signature.get(key)
            if saved_value == current_value:
                continue
            if key in path_keys and cls._resume_path_signature_matches(
                saved_value, current_value
            ):
                continue
            mismatches[key] = (saved_value, current_value)
        return mismatches

    @staticmethod
    def _load_scheduler_state(scheduler, state, name: str) -> None:
        if scheduler is None and state is None:
            return
        if scheduler is None or state is None:
            raise RuntimeError(f"{name} scheduler changed across resume")
        scheduler.load_state_dict(state)

    def _restore_training_state(self, checkpoint_path: str) -> None:
        checkpoint_path = os.path.realpath(checkpoint_path)
        checkpoint_dir = os.path.dirname(checkpoint_path)
        manifest_path = os.path.join(checkpoint_dir, "trainer_state.json")
        if not os.path.isfile(manifest_path):
            raise RuntimeError(
                f"Exact resume metadata is missing: {manifest_path}. "
                "Use a stage initialization checkpoint for a weights-only warm start."
            )

        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if int(manifest.get("format_version", 0)) != 2:
            raise RuntimeError(
                f"Unsupported DMD trainer-state format: {manifest.get('format_version')}"
            )
        if manifest.get("training_stage") != self.training_stage:
            raise RuntimeError(
                "Training stage changed across resume: "
                f"{manifest.get('training_stage')} != {self.training_stage}"
            )
        if int(manifest.get("world_size", 0)) != self.world_size:
            raise RuntimeError(
                "Exact DMD resume requires the original FSDP topology: "
                f"checkpoint world_size={manifest.get('world_size')}, "
                f"current world_size={self.world_size}."
            )

        saved_signature = manifest.get("resume_signature")
        current_signature = self._resume_signature()
        mismatches = self._resume_signature_mismatches(saved_signature, current_signature)
        if mismatches:
            raise RuntimeError(f"Training configuration changed across resume: {mismatches}")

        rank_state_path = os.path.join(
            checkpoint_dir, f"trainer_state_rank_{self.global_rank:05d}.pt"
        )
        if not os.path.isfile(rank_state_path):
            raise RuntimeError(f"Exact resume shard is missing: {rank_state_path}")

        if self.is_main_process:
            print(f"[Resume] Loading exact {self.training_stage} state from {checkpoint_dir}")
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        if int(checkpoint.get("format_version", 0)) != 4:
            raise RuntimeError(
                f"Unsupported DMD model checkpoint format: {checkpoint.get('format_version')}"
            )
        saved_step = int(manifest.get("step", -1))
        if int(checkpoint.get("step", -2)) != saved_step:
            raise RuntimeError("model.pt and trainer_state.json have different steps")

        load_fsdp_state_dict(self.dmd.generator, checkpoint["generator"], strict=True)
        load_fsdp_state_dict(self.dmd.fake_score, checkpoint["critic"], strict=True)

        ema_present = bool(manifest.get("ema_present", False))
        if ema_present != ("generator_ema" in checkpoint):
            raise RuntimeError("EMA metadata does not match model.pt")
        if ema_present:
            if self.ema_weight <= 0:
                raise RuntimeError("Checkpoint contains EMA but EMA is disabled in the config")
            from ltx_distillation.ema import EMA_FSDP

            self.generator_ema = EMA_FSDP.from_state_dict(
                checkpoint["generator_ema"], decay=self.ema_weight
            )
        del checkpoint

        rank_state = torch.load(
            rank_state_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        if int(rank_state.get("format_version", 0)) != 2:
            raise RuntimeError(
                f"Unsupported rank trainer-state format: {rank_state.get('format_version')}"
            )
        if int(rank_state.get("rank", -1)) != self.global_rank:
            raise RuntimeError(f"Resume shard rank mismatch in {rank_state_path}")
        if int(rank_state.get("world_size", 0)) != self.world_size:
            raise RuntimeError(f"Resume shard world-size mismatch in {rank_state_path}")
        if int(rank_state.get("step", -1)) != saved_step:
            raise RuntimeError(f"Resume shard step mismatch in {rank_state_path}")

        self.generator_optimizer.load_state_dict(rank_state["generator_optimizer"])
        self.critic_optimizer.load_state_dict(rank_state["critic_optimizer"])
        self._load_scheduler_state(
            self.generator_scheduler,
            rank_state.get("generator_scheduler"),
            "generator",
        )
        self._load_scheduler_state(
            self.critic_scheduler,
            rank_state.get("critic_scheduler"),
            "critic",
        )
        self.dataloader.load_state_dict(rank_state["data_iterator"])
        restore_rng_state(rank_state["rng_state"], device=self.device)

        self.step = saved_step
        self.last_saved_step = saved_step
        if self.generator_ema is None and self.ema_weight > 0 and self.step >= self.ema_start_step:
            raise RuntimeError("Checkpoint is missing EMA state after ema_start_step")
        max_steps = int(getattr(self.config, "max_steps", 0))
        if max_steps <= self.step:
            raise RuntimeError(
                f"Resume max_steps must be greater than saved step {self.step}, got {max_steps}."
            )
        del rank_state
        barrier()
        if (
            self.is_main_process
            and os.environ.get("STEP3_DELETE_MATERIALIZED_RESUME", "0") == "1"
        ):
            marker_path = os.path.join(
                checkpoint_dir,
                "model.pt.assembled.json",
            )
            try:
                os.remove(checkpoint_path)
                print(
                    f"[Resume] Removed materialized local model after restore: "
                    f"{checkpoint_path}",
                    flush=True,
                )
            except FileNotFoundError:
                pass
            try:
                os.remove(marker_path)
            except FileNotFoundError:
                pass
        barrier()
        if self.is_main_process:
            print(
                f"[Resume] Restored model, both AdamW states, schedulers, EMA, RNG "
                f"and exact data cursor at step {self.step}."
            )

    def save(self):
        """Save inference weights and exact same-topology training state."""
        print("Gathering distributed model states...")

        generator_state_dict = fsdp_state_dict(self.dmd.generator)
        critic_state_dict = fsdp_state_dict(self.dmd.fake_score)
        checkpoint_dir = os.path.join(
            self.output_path,
            f"checkpoint_{self.step:06d}",
        )
        if self.is_main_process:
            os.makedirs(checkpoint_dir, exist_ok=True)
        barrier()

        saved_rng_state = capture_rng_state(device=self.device)
        rank_state = {
            "format_version": 2,
            "rank": self.global_rank,
            "world_size": self.world_size,
            "step": self.step,
            "generator_optimizer": self.generator_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "generator_scheduler": (
                self.generator_scheduler.state_dict()
                if self.generator_scheduler is not None
                else None
            ),
            "critic_scheduler": (
                self.critic_scheduler.state_dict()
                if self.critic_scheduler is not None
                else None
            ),
            "data_iterator": self.dataloader.state_dict(),
            "rng_state": saved_rng_state,
        }
        rank_state_path = os.path.join(
            checkpoint_dir, f"trainer_state_rank_{self.global_rank:05d}.pt"
        )
        rank_state_tmp = f"{rank_state_path}.tmp"
        torch.save(rank_state, rank_state_tmp)
        os.replace(rank_state_tmp, rank_state_path)
        del rank_state
        barrier()

        if self.is_main_process:
            state_dict = {
                "format_version": 4,
                "checkpoint_kind": "dmd_generator_critic_with_distributed_training_state",
                "training_stage": self.training_stage,
                "optimizer_state_saved": True,
                "resume_world_size": self.world_size,
                "resume_state_pattern": "trainer_state_rank_{rank:05d}.pt",
                "generator": generator_state_dict,
                "critic": critic_state_dict,
                "step": self.step,
            }
            if self.generator_ema is not None:
                state_dict["generator_ema"] = self.generator_ema.state_dict()

            save_path = os.path.join(checkpoint_dir, "model.pt")
            save_tmp = f"{save_path}.tmp"
            torch.save(state_dict, save_tmp)
            os.replace(save_tmp, save_path)

            manifest = {
                "format_version": 2,
                "training_stage": self.training_stage,
                "step": self.step,
                "world_size": self.world_size,
                "optimizer": "AdamW",
                "optimizer_state_saved": True,
                "scheduler_state_saved": True,
                "rng_state_saved": ["python", "numpy", "torch", "cuda"],
                "ema_present": self.generator_ema is not None,
                "rank_state_pattern": "trainer_state_rank_{rank:05d}.pt",
                "data_iterator_resume": "exact_epoch_batch",
                "resume_signature": self._resume_signature(),
            }
            manifest_path = os.path.join(checkpoint_dir, "trainer_state.json")
            manifest_tmp = f"{manifest_path}.tmp"
            with open(manifest_tmp, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(manifest_tmp, manifest_path)
            print(f"Checkpoint saved to {save_path}")
            split_backup_path = persist_checkpoint_to_split_storage(
                checkpoint_dir,
                self.config,
            )
            upload_checkpoint_to_hf(
                checkpoint_dir,
                self.config,
                output_path=self.output_path,
            )
            if (
                split_backup_path
                and bool(
                    self.config.get(
                        "delete_local_checkpoint_after_split_backup",
                        False,
                    )
                )
            ):
                shutil.rmtree(checkpoint_dir)
                print(
                    f"Removed local checkpoint after split backup and upload: "
                    f"{checkpoint_dir}",
                    flush=True,
                )

        barrier()
        del generator_state_dict, critic_state_dict
        # Checkpoint serialization and remote upload must not perturb the next
        # training step relative to an uninterrupted run.
        restore_rng_state(saved_rng_state, device=self.device)
        del saved_rng_state
        self.last_saved_step = self.step

    @staticmethod
    def _to_scalar(value):
        """Convert tensor-like values to Python scalars for WandB logging."""
        if torch.is_tensor(value):
            if value.numel() == 1:
                return value.item()
            return value.detach().float().mean().item()
        return value

    def _compute_layerwise_grad_norms(self, module, prefix):
        """
        Compute per-layer gradient L2 norm for monitoring.

        Aggregation strategy:
        - For transformer blocks, log at block granularity: blocks.{idx}
        - For others, log at up-to-2-level module granularity.
        """
        layer_sq_norm = {}
        fsdp_prefix = "_fsdp_wrapped_module."

        for name, param in module.named_parameters():
            if param.grad is None or not param.requires_grad:
                continue

            normalized_name = name[len(fsdp_prefix):] if name.startswith(fsdp_prefix) else name
            parts = normalized_name.split(".")
            if len(parts) >= 3 and parts[1] == "blocks" and parts[2].isdigit():
                layer_key = f"blocks.{parts[2]}"
            elif len(parts) >= 2:
                layer_key = f"{parts[0]}.{parts[1]}"
            else:
                layer_key = parts[0]

            grad_sq = param.grad.detach().float().pow(2).sum().item()
            layer_sq_norm[layer_key] = layer_sq_norm.get(layer_key, 0.0) + grad_sq

        return {
            f"train/{prefix}_grad_norm/{k}": math.sqrt(v) for k, v in layer_sq_norm.items()
        }

    def train_one_step(self):
        """Execute one training step."""
        # Set all models to eval mode first (disables dropout/batchnorm),
        # then re-enable train mode for generator and fake_score so that
        # gradient checkpointing remains active during their gradient-enabled
        # forward passes. This is critical for the 19B model's memory footprint.
        # The real_score (teacher) stays in eval mode since it's frozen.
        #
        # For backward simulation's @torch.no_grad() forward passes, the
        # generator is temporarily switched to eval() inside
        # _consistency_backward_simulation() to avoid FSDP+checkpoint conflicts.
        self.dmd.eval()
        self.dmd.generator.train()
        self.dmd.fake_score.train()

        # Pass current step to DMD for step-dependent loss weighting
        self.dmd.current_step = self.step

        config = self.config
        TRAIN_GENERATOR = self.step % config.dfake_gen_update_ratio == 0
        LOG_LAYERWISE_GRAD = self.step % self.layerwise_grad_log_interval == 0

        # Periodic cache clearing
        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Get batch
        if not self.backward_simulation:
            batch = next(self.dataloader)
            text_prompts = batch["prompts"]
            # ODE latent format: [B, T, F, C, H, W], take last timestep (clean)
            clean_video = batch["ode_latent"][:, -1].to(
                device=self.device,
                dtype=self.dtype,
            )
            # Audio ODE latent format: [B, T, F_a, C], take last timestep (clean)
            if "ode_audio_latent" in batch and batch["ode_audio_latent"] is not None:
                clean_audio = batch["ode_audio_latent"][:, -1].to(
                    device=self.device,
                    dtype=self.dtype,
                )
            else:
                clean_audio = None
        else:
            text_prompts = next(self.dataloader)
            clean_video = None
            clean_audio = None

        batch_size = len(text_prompts)

        # Compute latent shapes
        video_shape, audio_shape = compute_latent_shapes(
            num_frames=config.num_frames,
            video_height=config.video_height,
            video_width=config.video_width,
            batch_size=batch_size,
        )

        # Encode text
        with torch.no_grad():
            conditional_dict = self.dmd.text_encoder(text_prompts=text_prompts)
            conditional_dict = self._move_conditioning_to_device(conditional_dict)

            if not hasattr(self, "unconditional_dict"):
                unconditional_dict = self.dmd.text_encoder(
                    text_prompts=[config.negative_prompt] * batch_size
                )
                unconditional_dict = self._move_conditioning_to_device(unconditional_dict)
                unconditional_dict = {
                    k: v.detach() for k, v in unconditional_dict.items()
                }
                self.unconditional_dict = unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        # Train generator
        if TRAIN_GENERATOR:
            generator_loss, generator_log_dict = self.dmd.generator_loss(
                video_shape=video_shape,
                audio_shape=audio_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_video=clean_video,
                clean_audio=clean_audio,
            )

            self.generator_optimizer.zero_grad()
            generator_loss.backward()
            generator_layerwise_grad_dict = (
                self._compute_layerwise_grad_norms(self.dmd.generator, "generator")
                if LOG_LAYERWISE_GRAD else {}
            )
            # Use FSDP's clip_grad_norm_ if available, otherwise fall back to torch utility
            if hasattr(self.dmd.generator, 'clip_grad_norm_'):
                generator_grad_norm = self.dmd.generator.clip_grad_norm_(self.max_grad_norm)
            else:
                generator_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.dmd.generator.parameters(), self.max_grad_norm
                )
            self.generator_optimizer.step()
            from ltx_distillation.util import refresh_fp8_training_scales
            refresh_fp8_training_scales(self.dmd.generator)

            # EMA update
            if self.generator_ema is not None:
                self.generator_ema.update(self.dmd.generator)

            # ---- Memory cleanup between generator and critic training ----
            # Save scalar metrics before freeing the computation graph.
            # This is critical because step 0 first allocates Adam optimizer
            # states (momentum + variance 鈮?2脳 param size), and the remaining
            # graph/activation memory must be released before critic training.
            generator_loss_val = generator_loss.item()
            generator_grad_norm_val = generator_grad_norm.item()
            gen_grad_norm_video = generator_log_dict.get("dmdtrain_gradient_norm_video", 0)
            gen_grad_norm_audio = generator_log_dict.get("dmdtrain_gradient_norm_audio", 0)

            del generator_loss, generator_grad_norm
            torch.cuda.empty_cache()
        else:
            generator_log_dict = {}
            generator_loss_val = None
            generator_grad_norm_val = None
            gen_grad_norm_video = 0
            gen_grad_norm_audio = 0
            generator_layerwise_grad_dict = {}

        # Train critic
        critic_loss, critic_log_dict = self.dmd.critic_loss(
            video_shape=video_shape,
            audio_shape=audio_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_video=clean_video,
            clean_audio=clean_audio,
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_layerwise_grad_dict = (
            self._compute_layerwise_grad_norms(self.dmd.fake_score, "critic")
            if LOG_LAYERWISE_GRAD else {}
        )
        # Use FSDP's clip_grad_norm_ if available, otherwise fall back to torch utility
        if hasattr(self.dmd.fake_score, 'clip_grad_norm_'):
            critic_grad_norm = self.dmd.fake_score.clip_grad_norm_(self.max_grad_norm)
        else:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.dmd.fake_score.parameters(), self.max_grad_norm
            )
        self.critic_optimizer.step()
        from ltx_distillation.util import refresh_fp8_training_scales
        refresh_fp8_training_scales(self.dmd.fake_score)

        # From this point onward, self.step is the number of completed updates.
        self.step += 1

        # Logging (all scalars, no GPU tensors)
        if self.is_main_process:
            wandb_dict = {
                "train/critic_loss": critic_loss.item(),
                "train/critic_grad_norm": critic_grad_norm.item(),
            }

            # Add per-component critic losses from log_dict
            wandb_dict.update({
                f"train/{k}": self._to_scalar(v) for k, v in critic_log_dict.items()
            })
            wandb_dict.update(critic_layerwise_grad_dict)

            if TRAIN_GENERATOR:
                wandb_dict.update({
                    "train/generator_loss": generator_loss_val,
                    "train/generator_grad_norm": generator_grad_norm_val,
                    "train/dmdtrain_gradient_norm_video": gen_grad_norm_video,
                    "train/dmdtrain_gradient_norm_audio": gen_grad_norm_audio,
                })
                wandb_dict.update(generator_layerwise_grad_dict)
                for gk, gv in generator_log_dict.items():
                    wandb_dict[f"train/{gk}"] = self._to_scalar(gv)

            wandb_dict["train/lr_generator"] = self.generator_optimizer.param_groups[0]["lr"]
            wandb_dict["train/lr_critic"] = self.critic_optimizer.param_groups[0]["lr"]

            wandb.log(wandb_dict, step=self.step)

        del critic_loss, critic_grad_norm
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _run_benchmark_and_log(self):
        """
        Run 4-step inference on fixed benchmark prompts, distributing work
        across all ranks for maximum parallelism.

        **All ranks** must call this method because the generator and text
        encoder are FSDP-wrapped and require collective communication.

        Flow (per round, one prompt per rank):
        1. ALL ranks: encode 1 prompt each (FSDP collective, batch_size=1)
        2. ALL ranks: run inference pipeline (FSDP collective, batch_size=1)
        3. ALL ranks: decode video/audio with local VAE, save mp4 to shared FS
        4. Rank 0: collect all saved files, log to WandB

        This distributes N prompts across W ranks in ceil(N/W) rounds,
        reducing per-rank memory vs the old single-rank-decodes-all approach.

        RNG is forked per prompt for reproducibility without affecting training.
        """
        from ltx_distillation.inference.bidirectional_pipeline import (
            BidirectionalAVInferencePipeline,
        )
        from ltx_distillation.inference.causal_pipeline import (
            CausalAVInferencePipeline,
        )

        config = self.config

        # Free training intermediate memory before benchmark
        torch.cuda.empty_cache()

        num_prompts = len(self.benchmark_prompts)
        num_rounds = math.ceil(num_prompts / self.world_size)

        if self.is_main_process:
            print(
                f"[Benchmark] Step {self.step}: generating {num_prompts} samples "
                f"({self.benchmark_mode} mode) across {self.world_size} ranks "
                f"({num_rounds} round(s))..."
            )

        step_dir = os.path.join(
            self.output_path, "benchmark", f"step_{self.step:07d}"
        )
        os.makedirs(step_dir, exist_ok=True)

        video_shape_single, audio_shape_single = compute_latent_shapes(
            num_frames=config.num_frames,
            video_height=config.video_height,
            video_width=config.video_width,
            batch_size=1,
        )

        # Keep Stage 3 benchmark aligned with the Stage-2 ODE benchmark:
        # temporarily switch the FSDP-wrapped generator to eval() under no_grad,
        # then restore the previous mode afterwards.
        was_training = self.dmd.generator.training
        self.dmd.generator.eval()
        try:
            if self.benchmark_mode == "causal":
                pipeline = CausalAVInferencePipeline(
                    generator=self.dmd.generator,
                    add_noise_fn=self.dmd.add_noise,
                    denoising_sigmas=self.dmd.denoising_sigmas,
                    num_frame_per_block=self.benchmark_num_frame_per_block,
                    num_frame_per_block_first=getattr(config, "num_frame_per_block_first", 4),
                    context_noise=int(getattr(config, "context_noise", 0)),
                    num_train_timestep=int(getattr(config, "num_train_timestep", 1000)),
                    clear_cuda_cache_per_round=self.benchmark_clear_cuda_cache_per_round,
                )
            else:
                pipeline = BidirectionalAVInferencePipeline(
                    generator=self.dmd.generator,
                    add_noise_fn=self.dmd.add_noise,
                    denoising_sigmas=self.dmd.denoising_sigmas,
                )

            self._vae_to_device()

            # Timing: wall-clock for full benchmark, and per-video generation time
            benchmark_wall_start = time.perf_counter()
            my_total_generate_seconds = 0.0

            for round_idx in range(num_rounds):
                prompt_idx = round_idx * self.world_size + self.global_rank
                has_real_prompt = prompt_idx < num_prompts

                if has_real_prompt:
                    my_prompt = [self.benchmark_prompts[prompt_idx]]
                else:
                    my_prompt = [self.benchmark_prompts[0]]

                with torch.no_grad():
                    conditional_dict = self.dmd.text_encoder(text_prompts=my_prompt)

                prompt_seed = self.benchmark_seed + prompt_idx
                with torch.random.fork_rng(devices=[self.device]):
                    torch.manual_seed(prompt_seed)
                    torch.cuda.manual_seed(prompt_seed)

                    gen_start = time.perf_counter()
                    video_latent, audio_latent = pipeline.generate(
                        video_shape=tuple(video_shape_single),
                        audio_shape=tuple(audio_shape_single),
                        conditional_dict=conditional_dict,
                    )
                    gen_elapsed = time.perf_counter() - gen_start
                    my_total_generate_seconds += gen_elapsed

                if has_real_prompt:
                    self._decode_and_save_sample(
                        video_latent=video_latent,
                        audio_latent=audio_latent,
                        prompt_idx=prompt_idx,
                        step_dir=step_dir,
                    )

                del video_latent, audio_latent, conditional_dict
                if self.benchmark_clear_cuda_cache_per_round:
                    torch.cuda.empty_cache()

                barrier()
        finally:
            if was_training:
                self.dmd.generator.train()

        benchmark_wall_elapsed = time.perf_counter() - benchmark_wall_start

        # Gather total generation time from all ranks (each rank sums its own generate times)
        total_generate_tensor = torch.tensor(
            [my_total_generate_seconds], device=self.device, dtype=torch.float64
        )
        dist.all_reduce(total_generate_tensor, op=dist.ReduceOp.SUM)
        total_generate_seconds = total_generate_tensor.item()

        self._vae_to_cpu()

        barrier()

        # ---- Rank 0: log all samples to WandB and print benchmark timing ----
        if self.is_main_process:
            time_per_video_wall = benchmark_wall_elapsed / max(1, num_prompts)
            time_per_video_generate = total_generate_seconds / max(1, num_prompts)

            benchmark_wandb_dict = {}
            prompt_rows = []

            for idx in range(num_prompts):
                sample_path = os.path.join(step_dir, f"sample_{idx}.mp4")
                media_key = f"benchmark/sample_{idx}"
                video_media = wandb_video_from_path(
                    sample_path,
                    fps=self.benchmark_video_fps,
                    key=media_key,
                    required=self.wandb_video_required,
                )
                if video_media is not None:
                    benchmark_wandb_dict[media_key] = video_media
                    prompt_rows.append(
                        [idx, self.benchmark_prompts[idx], sample_path]
                    )

            if prompt_rows:
                table = wandb.Table(
                    columns=["index", "prompt", "local_path"],
                    data=prompt_rows,
                )
                benchmark_wandb_dict["benchmark/prompt_table"] = table

            if benchmark_wandb_dict:
                wandb.log(benchmark_wandb_dict, step=self.step)
                video_count = sum(
                    key.startswith("benchmark/sample_")
                    for key in benchmark_wandb_dict
                )
                print(
                    f"[WANDB_VIDEO] logged {video_count} benchmark video(s) "
                    f"at step {self.step}",
                    flush=True,
                )

            print(
                f"[Benchmark] Step {self.step}: {num_prompts} video(s) | "
                f"saved to {step_dir}",
                flush=True,
            )

        barrier()

    def _should_run_benchmark(self) -> bool:
        return (
            self.benchmark_enabled
            and len(self.benchmark_prompts) > 0
            and self.benchmark_iters > 0
            and self.step % self.benchmark_iters == 0
            and not getattr(self.config, "no_visualize", False)
        )

    def _decode_and_save_sample(
        self,
        video_latent: torch.Tensor,
        audio_latent: torch.Tensor,
        prompt_idx: int,
        step_dir: str,
    ):
        """
        Decode one (video, audio) latent pair and save as mp4 with audio.

        Called by every rank that owns a real benchmark prompt.  VAEs must
        already be on GPU (via ``_vae_to_device``) before calling this.
        """
        # Decode video 鈫?pixel  [1, C, F, H, W]  鈫? [0, 1]
        video_pixel = self.dmd.video_vae.decode_to_pixel(video_latent)

        # Decode audio 鈫?waveform  [1, 1, samples]
        audio_waveform = None
        try:
            audio_waveform = self.dmd.audio_vae.decode_to_waveform(audio_latent)
        except Exception as e:
            print(
                f"[Benchmark][Rank {self.global_rank}] Audio decode failed "
                f"for prompt {prompt_idx}: {e}"
            )

        # Prepare video tensor: -> uint8 [F, H, W, C]
        vid = video_pixel[0]  # [C, F, H, W]
        if vid.shape[0] == 3:
            vid = vid.permute(1, 0, 2, 3)  # -> [F, C, H, W]
        vid = vid.permute(0, 2, 3, 1)  # -> [F, H, W, C]
        vid = (vid.clamp(0, 1) * 255).cpu().to(torch.uint8)

        sample_path = os.path.join(step_dir, f"sample_{prompt_idx}.mp4")

        # Try writing mp4 with embedded audio track
        written_with_audio = False
        if audio_waveform is not None:
            try:
                wav_float = audio_waveform[0].cpu().float()  # [1, samples]
                from torchvision.io import write_video

                write_video(
                    sample_path,
                    vid,
                    fps=self.benchmark_video_fps,
                    audio_array=wav_float,
                    audio_fps=self.benchmark_audio_sample_rate,
                    audio_codec="aac",
                )
                written_with_audio = True
            except Exception as e:
                print(
                    f"[Benchmark][Rank {self.global_rank}] write_video with "
                    f"audio failed for prompt {prompt_idx}: {e}"
                )

        # Fallback: silent video + separate wav
        if not written_with_audio:
            try:
                from torchvision.io import write_video

                write_video(sample_path, vid, fps=self.benchmark_video_fps)
            except Exception as e:
                print(
                    f"[Benchmark][Rank {self.global_rank}] write_video (silent) "
                    f"failed for prompt {prompt_idx}: {e}"
                )
                return

            if audio_waveform is not None:
                try:
                    import torchaudio

                    wav = audio_waveform[0].cpu().float()
                    wav_path = os.path.join(
                        step_dir, f"sample_{prompt_idx}.wav"
                    )
                    torchaudio.save(
                        wav_path, wav, self.benchmark_audio_sample_rate
                    )
                except Exception as e:
                    print(
                        f"[Benchmark][Rank {self.global_rank}] torchaudio.save "
                        f"failed for prompt {prompt_idx}: {e}"
                    )

        # Free decoded tensors
        del video_pixel, audio_waveform
        torch.cuda.empty_cache()

    def train(self):
        """Main training loop."""
        while True:
            self.train_one_step()

            # Complete all state transitions for this step before checkpointing.
            if (
                self.generator_ema is None
                and self.ema_weight > 0
                and self.step >= self.ema_start_step
            ):
                from ltx_distillation.ema import EMA_FSDP
                if self.is_main_process:
                    print(f"[EMA] Initializing EMA with decay={self.ema_weight} at step {self.step}")
                self.generator_ema = EMA_FSDP(self.dmd.generator, decay=self.ema_weight)

            if self.generator_scheduler is not None:
                self.generator_scheduler.step()
            if self.critic_scheduler is not None:
                self.critic_scheduler.step()

            # Save checkpoint
            if (
                not getattr(self.config, "no_save", False)
                and self.save_iters > 0
                and self.step % self.save_iters == 0
            ):
                self.save()
                torch.cuda.empty_cache()

            barrier()

            # Persist and upload the checkpoint before a required video upload
            # can fail the shared 500-step artifact interval.
            if self._should_run_benchmark():
                self._run_benchmark_and_log()

            barrier()

            # Timing
            current_time = time.time()
            if self.is_main_process and self.previous_time is not None:
                wandb.log(
                    {"per_iteration_time": current_time - self.previous_time},
                    step=self.step,
                )
            self.previous_time = current_time

            # Optional: max steps limit
            max_steps = getattr(self.config, "max_steps", None)
            if max_steps and self.step >= max_steps:
                break

        if (
            not getattr(self.config, "no_save", False)
            and self.last_saved_step != self.step
        ):
            self.save()
        if self.is_main_process:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument("--resume_checkpoint", default=None)

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize
    if args.resume_checkpoint:
        config.resume_checkpoint = args.resume_checkpoint

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
