"""
ODE Regression Training Script for LTX-2 Causal Model Initialization.

This script trains a causal LTX-2 model using precomputed ODE trajectories.
It supports FSDP distributed training for multi-GPU/multi-node setups.

Usage:
    # Single node
    torchrun --nproc_per_node=8 train_ode.py --config_path configs/ltx2_causal_ode.yaml

    # Multi-node
    torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \\
        train_ode.py --config_path configs/ltx2_causal_ode.yaml

Reference: CausVid (https://arxiv.org/abs/2412.07772) Section 4.3
"""

import os
import json
import time
import logging
import argparse
import functools
import gc
import socket
from dataclasses import asdict
from collections import defaultdict
from datetime import timedelta
from typing import Dict, Any, Optional

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from omegaconf import OmegaConf, DictConfig
import wandb

from ltx_distillation.inference.ode_benchmark_pipeline import ODEAutoregressiveBenchmarkPipeline
from ltx_distillation.ode.data import ODERegressionLMDBDataset, collate_ode_batch
from ltx_distillation.ode.ode_regression import LTX2ODERegression, ODERegressionConfig
from ltx_distillation.util import (
    fsdp_state_dict as shared_fsdp_state_dict,
    ResumableDataIterator,
    ResumableDistributedSampler,
    capture_rng_state,
    restore_rng_state,
    upload_checkpoint_to_hf,
    validate_artifact_upload_config,
    resolve_wandb_api_key,
    wandb_video_from_path,
    artifact_paths_match,
)


# ============================================================================
# Distributed Training Utilities
# ============================================================================

def launch_distributed_job():
    """Initialize distributed training environment."""
    timeout_minutes = int(os.environ.get("DIST_TIMEOUT_MINUTES", "120"))
    if 'RANK' in os.environ:
        # Launched via torchrun
        dist.init_process_group(
            backend='nccl',
            timeout=timedelta(minutes=timeout_minutes),
        )
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    elif torch.cuda.is_available():
        # Single GPU fallback
        dist.init_process_group(
            backend='nccl',
            init_method='tcp://localhost:29500',
            world_size=1,
            rank=0,
            timeout=timedelta(minutes=timeout_minutes),
        )
        torch.cuda.set_device(0)
    else:
        raise RuntimeError("CUDA is required for training")


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def barrier():
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


def init_logging_folder(config: DictConfig):
    """Initialize output and wandb folders."""
    output_path = config.output_path
    os.makedirs(output_path, exist_ok=True)

    wandb_folder = os.path.join(output_path, "wandb")
    os.makedirs(wandb_folder, exist_ok=True)

    # Set wandb API key from config if provided (needed for multi-node)
    wandb_api_key = resolve_wandb_api_key(config)
    if wandb_api_key:
        os.environ["WANDB_API_KEY"] = wandb_api_key

    # Initialize wandb
    wandb_entity = config.get("wandb_entity", None) or None
    wandb.init(
        project=config.get("wandb_project", "OmniForcing"),
        entity=wandb_entity,
        name=config.get("wandb_name", "ltx2_causal_ode"),
        dir=wandb_folder,
        config=OmegaConf.to_container(config),
    )

    # Save config
    config_path = os.path.join(output_path, "config.yaml")
    OmegaConf.save(config, config_path)

    return output_path, wandb_folder


def get_sharding_strategy(name: str) -> ShardingStrategy:
    """Get FSDP sharding strategy by name."""
    strategies = {
        "full": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
        "hybrid_full": ShardingStrategy.HYBRID_SHARD,
        "hybrid_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
    }
    return strategies.get(name, ShardingStrategy.FULL_SHARD)


def fsdp_wrap(
    module: torch.nn.Module,
    sharding_strategy: str = "full",
    mixed_precision: bool = True,
    wrap_strategy: str = "size",
    min_num_params: int = 1e8,
    transformer_module: Optional[tuple] = None,
) -> FSDP:
    """Wrap module with FSDP.

    Args:
        module: Module to wrap
        sharding_strategy: FSDP sharding strategy name
        mixed_precision: Whether to use BF16 mixed precision
        wrap_strategy: "size" for size-based, "transformer" for block-based
        min_num_params: Minimum params for size-based wrapping
        transformer_module: Tuple of module classes for transformer wrapping
    """
    # Mixed precision policy
    # Match CausVid: param in bfloat16, but reduce/buffer in float32
    # for gradient all-reduce precision and buffer accuracy.
    mp_policy = None
    if mixed_precision:
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
            cast_forward_inputs=False,
        )

    # Wrap policy - use functools.partial to properly bind parameters
    auto_wrap_policy = None
    if wrap_strategy == "transformer" and transformer_module is not None:
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=set(transformer_module),
        )
    elif wrap_strategy == "size":
        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=int(min_num_params),
        )

    return FSDP(
        module,
        sharding_strategy=get_sharding_strategy(sharding_strategy),
        mixed_precision=mp_policy,
        auto_wrap_policy=auto_wrap_policy,
        device_id=torch.cuda.current_device(),
    )


def fsdp_state_dict(fsdp_module: FSDP) -> Dict[str, Any]:
    """Proxy to the shared rank0-only FSDP checkpoint helper."""
    return shared_fsdp_state_dict(fsdp_module)


# ============================================================================
# Trainer
# ============================================================================

class ODETrainer:
    """
    Trainer for ODE regression initialization.

    Handles:
    - FSDP distributed training
    - Gradient accumulation
    - Logging and checkpointing
    - Learning rate scheduling
    """

    def __init__(self, config: DictConfig):
        self.config = config
        self.training_stage = str(config.get("training_stage", "stage2_causal_ode"))
        validate_artifact_upload_config(config)

        # Initialize distributed
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        expected_world_size = int(config.get("expected_world_size", 0) or 0)
        if expected_world_size and self.world_size != expected_world_size:
            raise RuntimeError(
                "Distributed world-size mismatch: "
                f"expected {expected_world_size}, got {self.world_size}. "
                "The corrected Step-2 job must run with all requested GPUs."
            )

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.global_rank = global_rank
        self.is_main_process = global_rank == 0
        print(
            "[dist] "
            f"host={socket.gethostname()} rank={self.global_rank}/{self.world_size} "
            f"local_rank={os.environ.get('LOCAL_RANK', '0')} device={self.device}",
            flush=True,
        )

        # Diagnostic state (rank 0 only writes to log file)
        self._block_diag = []
        self.diag_logger = None

        # Random seed
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        # Logging
        self.output_path = config.output_path
        self.wandb_folder = os.path.join(self.output_path, "wandb")
        if self.is_main_process:
            self.output_path, self.wandb_folder = init_logging_folder(config)
            self._setup_diagnostic_log(config)
        else:
            os.makedirs(self.output_path, exist_ok=True)
        barrier()

        # Initialize model. Exact resume uses the same model file for the
        # pre-FSDP weight load, then restores rank-local optimizer state later.
        resume_checkpoint = config.get("resume_checkpoint")
        causal_model_checkpoint = config.get("causal_model_checkpoint")
        if resume_checkpoint:
            if (
                causal_model_checkpoint
                and os.path.realpath(str(causal_model_checkpoint))
                != os.path.realpath(str(resume_checkpoint))
            ):
                raise RuntimeError(
                    "resume_checkpoint and causal_model_checkpoint must refer to "
                    "the same model.pt when exact resume is requested."
                )
            causal_model_checkpoint = resume_checkpoint

        ode_config = ODERegressionConfig(
            checkpoint_path=config.get("checkpoint_path", ""),
            causal_model_checkpoint=causal_model_checkpoint,
            bidirectional_model_checkpoint=config.get("bidirectional_model_checkpoint"),
            text_encoder_checkpoint=config.text_encoder_checkpoint,
            denoising_step_list=tuple(config.denoising_step_list),
            generator_task=config.generator_task,
            num_frame_per_block=config.get("num_frame_per_block", 3),
            num_frame_per_block_first=config.get("num_frame_per_block_first", 4),
            gradient_checkpointing=config.gradient_checkpointing,
            mixed_precision=config.mixed_precision,
            uniform_timestep=config.get("uniform_timestep", False),
            loss_target=config.get("loss_target", "velocity"),
            disable_causal_mask=config.get("disable_causal_mask", False),
            enable_causal_log_rescale=config.get("enable_causal_log_rescale", False),
            text_encoder_device=config.get("text_encoder_device", "cuda"),
            video_height=config.get("video_height", 512),
            video_width=config.get("video_width", 768),
            # Loss weights
            video_loss_weight=config.get("video_loss_weight", 1.0),
            audio_loss_weight=config.get("audio_loss_weight", 0.0),
            require_ltx23_architecture=config.get("require_ltx23_architecture", True),
            trainable_fp8_mode=config.get("trainable_fp8_mode", "training"),
        )

        self.ode_model = LTX2ODERegression(ode_config, device=self.device)

        # Wrap with FSDP
        self.ode_model._load_models()  # Force load before FSDP wrap

        # Curriculum learning: skip A2V/V2A cross-modal attention
        self.skip_cross_modal_attention = config.get("skip_cross_modal_attention", False)
        if self.skip_cross_modal_attention:
            self._setup_skip_cross_modal(self.ode_model._generator)

        # Register diagnostic hooks BEFORE FSDP wrap (hooks survive wrapping)
        if self.is_main_process:
            self._register_diagnostic_hooks()

        # Determine generator FSDP wrap strategy
        gen_wrap_strategy = config.get("generator_fsdp_wrap_strategy", "transformer")
        gen_transformer_module = None
        if gen_wrap_strategy == "transformer":
            from ltx_causal.transformer.causal_block import CausalAVTransformerBlock
            gen_transformer_module = (CausalAVTransformerBlock,)

        self.ode_model._generator = fsdp_wrap(
            self.ode_model._generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=gen_wrap_strategy,
            transformer_module=gen_transformer_module,
        )
        if config.get("text_encoder_device", "cuda") != "cpu":
            self.ode_model._text_encoder = fsdp_wrap(
                self.ode_model._text_encoder,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy=config.get("text_encoder_fsdp_wrap_strategy", "size"),
            )
        # Pass diagnostic logger to ODE regression module
        if self.diag_logger:
            self.ode_model.diag_logger = self.diag_logger

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            [p for p in self.ode_model._generator.parameters() if p.requires_grad],
            lr=config.lr,
            betas=(config.get("beta1", 0.9), config.get("beta2", 0.999)),
            weight_decay=config.get("weight_decay", 0.0),
        )

        # Dataloader and fixed benchmark prompt source
        dataset = ODERegressionLMDBDataset(
            config.data_path,
            max_pair=config.get("max_pair", int(1e8)),
            load_audio=config.get("load_audio", True),
        )
        if config.get("require_ode_manifest", True):
            if dataset.manifest is None:
                raise RuntimeError(
                    f"ODE dataset has no LTX-2.3 provenance manifest: {config.data_path}. "
                    "Generate a fresh dataset instead of reusing ode_lmdb."
                )
            if dataset.manifest.get("format_version") != 4:
                raise RuntimeError(
                    f"Unsupported ODE dataset format in {config.data_path}: "
                    f"{dataset.manifest.get('format_version')}"
                )
            if dataset.manifest.get("producer") != "omniforcing-ltx23-full-architecture-v2":
                raise RuntimeError(
                    f"Unsupported ODE dataset producer in {config.data_path}: "
                    f"{dataset.manifest.get('producer')}"
                )
            teacher_checkpoint = dataset.manifest.get("teacher_checkpoint", "")
            expected_checkpoint = str(config.checkpoint_path)
            if not artifact_paths_match(teacher_checkpoint, expected_checkpoint):
                raise RuntimeError(
                    "ODE dataset teacher checkpoint does not match training checkpoint: "
                    f"{teacher_checkpoint} != {expected_checkpoint}"
                )
            generation_config = dataset.manifest.get("generation_config")
            if not isinstance(generation_config, dict):
                raise RuntimeError(
                    f"ODE dataset has no generation_config: {config.data_path}"
                )
            generated_steps = generation_config.get(
                "denoising_step_list"
            )
            if list(generated_steps or []) != list(config.denoising_step_list):
                raise RuntimeError(
                    "ODE dataset denoising schedule does not match training config: "
                    f"{generated_steps} != {list(config.denoising_step_list)}"
                )
            expected_generation_values = {
                "video_height": int(config.get("video_height", 512)),
                "video_width": int(config.get("video_width", 768)),
            }
            mismatched_generation_values = {
                key: (generation_config.get(key), expected)
                for key, expected in expected_generation_values.items()
                if generation_config.get(key) != expected
            }
            if mismatched_generation_values:
                raise RuntimeError(
                    "ODE dataset geometry does not match training config: "
                    f"{mismatched_generation_values}"
                )
            if not dataset.has_sigmas:
                raise RuntimeError(
                    f"ODE dataset has no exact scheduler sigmas: {config.data_path}"
                )
        self.dataset = dataset

        sampler = ResumableDistributedSampler(
            dataset, shuffle=True, drop_last=True
        )

        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=config.get("num_workers", 8),
            collate_fn=collate_ode_batch,
            pin_memory=True,
            generator=torch.Generator(),
        )
        data_seed = int(config.seed) + self.global_rank * 1_000_003
        self.data_iter = ResumableDataIterator(self.dataloader, seed=data_seed)

        # Benchmark uses the first N prompts from the training LMDB so that
        # every evaluation round is directly comparable across checkpoints.
        self._init_benchmark_state(dataset)

        # Visualization / benchmark decoding uses the original ltx-core decode
        # functions, which are already validated against the base pipeline.
        self.video_decoder = None
        self.audio_decoder = None
        self.vocoder = None
        self.audio_sample_rate = int(self.benchmark_audio_sample_rate)
        self.no_visualize = config.get("no_visualize", False)
        self.visualize_iters = config.get("visualize_iters", config.log_iters)

        should_load_decoders = (not self.no_visualize) or self.benchmark_enabled
        if self.is_main_process and should_load_decoders:
            reason = "visualization / benchmark logging"
            if self.no_visualize and self.benchmark_enabled:
                reason = "benchmark logging"
            self._load_decoders(reason=reason)

        if self.benchmark_enabled:
            benchmark_ready = torch.tensor(
                [1 if (not self.is_main_process or self.video_decoder is not None) else 0],
                device=self.device,
                dtype=torch.int32,
            )
            dist.broadcast(benchmark_ready, src=0)
            self.benchmark_enabled = bool(benchmark_ready.item())
            if self.is_main_process and not self.benchmark_enabled:
                print("[Benchmark] Disabled because VAE decoders could not be loaded on rank 0.")

        # Training state
        self.step = 0
        self.max_grad_norm = config.get("max_grad_norm", 10.0)
        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
        self.previous_time = None
        self._pending_visualization = None

        if resume_checkpoint:
            self._restore_training_state(resume_checkpoint)

        if self.is_main_process:
            if resume_checkpoint:
                print(
                    f"[Resume] Restored generator, AdamW state, RNG state, and step={self.step}."
                )
            elif config.get("causal_model_checkpoint"):
                print(
                    "[Warm Start] Loaded generator weights only. AdamW state and "
                    "the step counter start from zero; this is not an exact resume."
                )
            effective_batch = config.batch_size * self.gradient_accumulation_steps * self.world_size
            print(f"Gradient accumulation: {self.gradient_accumulation_steps} steps, "
                  f"effective batch size = {config.batch_size} x {self.gradient_accumulation_steps} x {self.world_size} = {effective_batch}")

    def _setup_skip_cross_modal(self, generator):
        """Skip A2V/V2A cross-modal attention for curriculum learning.

        Sets skip flag on all transformer blocks. Parameters are NOT frozen
        (FSDP requires uniform requires_grad within a flat parameter group).
        Since the forward path is skipped, these parameters receive zero
        gradient and are effectively frozen.
        """
        # Access the inner CausalLTXModel
        model = generator.model if hasattr(generator, 'model') else generator

        for block in model.transformer_blocks:
            block.skip_cross_modal_attention = True

        if self.is_main_process:
            print("[Curriculum] skip_cross_modal_attention=True: "
                  "A2V/V2A forward skipped (params kept trainable for FSDP compat, "
                  "but receive zero gradient)")

    def _load_decoders(self, reason: str) -> None:
        """Load video/audio decoders on rank 0 for visualization or benchmark logging."""
        if not self.is_main_process or self.video_decoder is not None:
            return

        try:
            from ltx_pipelines.utils.model_ledger import ModelLedger

            checkpoint_path = self.config.get("checkpoint_path", "")
            if not checkpoint_path:
                print(f"WARNING: checkpoint_path not set, skipping VAE loading for {reason}.")
                return

            print(f"Loading VAE decoders for {reason}...")
            ledger = ModelLedger(
                dtype=self.dtype,
                device=torch.device("cpu"),
                checkpoint_path=checkpoint_path,
            )
            self.video_decoder = ledger.video_decoder().to(self.device).eval()
            self.audio_decoder = ledger.audio_decoder().to(self.device).eval()
            self.vocoder = ledger.vocoder().to(self.device).eval()
            self.audio_sample_rate = int(self.vocoder.output_sample_rate)
            if (
                self.benchmark_audio_sample_rate_explicit is not None
                and self.benchmark_audio_sample_rate != self.audio_sample_rate
            ):
                raise RuntimeError(
                    "benchmark_audio_sample_rate does not match the checkpoint vocoder: "
                    f"{self.benchmark_audio_sample_rate} != {self.audio_sample_rate}"
                )
            self.benchmark_audio_sample_rate = self.audio_sample_rate
            self.video_decoder.requires_grad_(False)
            self.audio_decoder.requires_grad_(False)
            self.vocoder.requires_grad_(False)
            del ledger
            print(
                f"VAE decoders loaded for {reason} "
                f"(audio_sample_rate={self.audio_sample_rate})."
            )
        except Exception as e:
            print(f"WARNING: Failed to load VAE for {reason}: {e}")

    def _init_benchmark_state(self, dataset: ODERegressionLMDBDataset) -> None:
        """Initialize fixed ODE benchmark prompts and latent shapes."""
        from ltx_core.components.schedulers import LTX2Scheduler

        config = self.config
        self.benchmark_enabled = bool(getattr(config, "benchmark_enabled", True))
        self.benchmark_iters = int(getattr(config, "benchmark_iters", config.log_iters))
        self.benchmark_seed = int(getattr(config, "benchmark_seed", 12345))
        self.benchmark_num_prompts = int(getattr(config, "benchmark_num_prompts", 2))
        self.benchmark_video_fps = int(getattr(config, "benchmark_video_fps", 24))
        self.wandb_video_required = bool(
            getattr(config, "wandb_video_required", False)
        )
        self.benchmark_audio_sample_rate_explicit = getattr(config, "benchmark_audio_sample_rate", None)
        self.benchmark_audio_sample_rate = int(
            self.benchmark_audio_sample_rate_explicit
            if self.benchmark_audio_sample_rate_explicit is not None
            else 48000
        )
        self.benchmark_num_frame_per_block = int(
            getattr(config, "benchmark_num_frame_per_block", getattr(config, "num_frame_per_block", 3))
        )
        self.benchmark_clear_cuda_cache_per_round = bool(
            getattr(config, "benchmark_clear_cuda_cache_per_round", True)
        )
        self.benchmark_prompts = []
        self.benchmark_video_shape_single = tuple([1, *dataset.video_shape[2:]])
        self.benchmark_audio_shape_single = None
        if getattr(dataset, "has_audio", False):
            self.benchmark_audio_shape_single = tuple([1, *dataset.audio_shape[2:]])

        if getattr(dataset, "has_sigmas", False):
            try:
                # Reuse the exact sub-sampled sigma schedule stored alongside the
                # training trajectories so benchmark sampling matches ODE training.
                self.benchmark_denoising_sigmas = dataset.get_sigmas(0).to(self.device)
            except Exception as e:
                if self.is_main_process:
                    print(f"[Benchmark] Failed to load stored sigmas from LMDB, falling back to scheduler: {e}")
                num_denoising_steps = max(1, len(config.denoising_step_list) - 1)
                self.benchmark_denoising_sigmas = LTX2Scheduler().execute(
                    steps=num_denoising_steps
                ).to(self.device)
        else:
            num_denoising_steps = max(1, len(config.denoising_step_list) - 1)
            self.benchmark_denoising_sigmas = LTX2Scheduler().execute(
                steps=num_denoising_steps
            ).to(self.device)

        if not self.benchmark_enabled:
            pass

        if self.benchmark_iters <= 0:
            print("[Benchmark] Disabled because benchmark_iters <= 0.")
            self.benchmark_enabled = False

        # Load benchmark prompts: prefer benchmark_prompt_file, fallback to LMDB.
        prompt_file = config.get("benchmark_prompt_file", None)
        if prompt_file and os.path.exists(prompt_file):
            with open(prompt_file, "r") as f:
                self.benchmark_prompts = [l.strip() for l in f if l.strip()]
            self.benchmark_prompts = self.benchmark_prompts[:self.benchmark_num_prompts]
            prompt_source = prompt_file
        else:
            try:
                self.benchmark_prompts = dataset.get_prompts(self.benchmark_num_prompts)
                prompt_source = "LMDB"
            except Exception as e:
                if self.is_main_process:
                    print(f"[Benchmark] Failed to load prompts from LMDB: {e}")
                self.benchmark_enabled = False
                return

        if not self.benchmark_prompts:
            print("[Benchmark] Disabled because no prompts were loaded.")
            self.benchmark_enabled = False
            return

        if self.is_main_process:
            print(
                f"[Benchmark] Loaded {len(self.benchmark_prompts)} prompt(s) from {prompt_source}"
            )
            print(
                f"[Benchmark] block_frames={self.benchmark_num_frame_per_block}, "
                f"video_shape={self.benchmark_video_shape_single}, "
                f"audio_shape={self.benchmark_audio_shape_single}"
            )
            print(f"[Benchmark] denoising_sigmas={self.benchmark_denoising_sigmas.tolist()}")
            for i, prompt in enumerate(self.benchmark_prompts):
                suffix = "..." if len(prompt) > 80 else ""
                print(f"  [{i}] {prompt[:80]}{suffix}")

    @staticmethod
    def _benchmark_add_noise(
        original: torch.Tensor,
        noise: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Flow-matching noise injection used by the ODE benchmark pipeline."""
        if sigma.dim() == 1:
            sigma = sigma.reshape(-1, *([1] * (original.dim() - 1)))
        elif sigma.dim() == 2:
            sigma = sigma.reshape(*sigma.shape, *([1] * (original.dim() - 2)))
        sigma = sigma.to(dtype=original.dtype)
        return ((1 - sigma) * original + sigma * noise).to(dtype=original.dtype)

    def _should_run_benchmark(self) -> bool:
        """Return True when the periodic ODE benchmark should run."""
        return (
            self.benchmark_enabled
            and self.benchmark_iters > 0
            and len(self.benchmark_prompts) > 0
            and self.step % self.benchmark_iters == 0
        )

    def _save_benchmark_latents(
        self,
        video_latent: torch.Tensor,
        audio_latent: Optional[torch.Tensor],
        prompt_idx: int,
        step_dir: str,
    ) -> None:
        """Persist one benchmark latent sample to shared storage for rank-0 decode."""
        latent_path = os.path.join(step_dir, f"sample_{prompt_idx}.pt")
        payload = {"video_latent": video_latent.detach().cpu()}
        if audio_latent is not None:
            payload["audio_latent"] = audio_latent.detach().cpu()
        torch.save(payload, latent_path)

    @torch.no_grad()
    def _decode_benchmark_latents(self, latent_path: str, prompt_idx: int, step_dir: str) -> Optional[str]:
        """Decode saved latent tensors into media files for wandb logging."""
        from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
        from ltx_core.model.video_vae import decode_video as vae_decode_video
        from ltx_pipelines.utils.media_io import encode_video
        import torchaudio

        if self.video_decoder is None:
            return None

        payload = torch.load(latent_path, map_location="cpu", weights_only=False)
        video_latent = payload["video_latent"].to(device=self.device, dtype=self.dtype)
        audio_latent = payload.get("audio_latent")
        if audio_latent is not None:
            audio_latent = audio_latent.to(device=self.device, dtype=self.dtype)

        latent_for_vae = video_latent.permute(0, 2, 1, 3, 4)
        decoded_frames = list(vae_decode_video(latent_for_vae, self.video_decoder))
        frames = torch.cat(decoded_frames, dim=0)

        decoded_audio = None
        if audio_latent is not None and self.audio_decoder is not None and self.vocoder is not None:
            decoded_audio = vae_decode_audio(
                audio_latent.unflatten(-1, (8, 16)).permute(0, 2, 1, 3),
                self.audio_decoder,
                self.vocoder,
            )

        sample_path = os.path.join(step_dir, f"sample_{prompt_idx}.mp4")
        try:
            encode_video(
                video=iter([frames]),
                fps=self.benchmark_video_fps,
                audio=decoded_audio,
                audio_sample_rate=self.benchmark_audio_sample_rate if decoded_audio is not None else None,
                output_path=sample_path,
                video_chunks_number=1,
            )
        except Exception as e:
            print(f"[Benchmark] Failed to save sample_{prompt_idx}.mp4: {e}")
            sample_path = None

        if decoded_audio is not None:
            wav = decoded_audio.cpu()
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
            wav_path = os.path.join(step_dir, f"sample_{prompt_idx}.wav")
            try:
                torchaudio.save(wav_path, wav, sample_rate=self.benchmark_audio_sample_rate)
            except Exception as e:
                print(f"[Benchmark] Failed to save sample_{prompt_idx}.wav: {e}")

        del payload, video_latent, audio_latent, frames, decoded_frames, decoded_audio
        torch.cuda.empty_cache()
        return sample_path

    @torch.no_grad()
    def _run_benchmark_and_log(self) -> None:
        """Run periodic autoregressive benchmark inference and log decoded samples."""
        if not self.benchmark_prompts:
            return

        torch.cuda.empty_cache()

        num_prompts = len(self.benchmark_prompts)
        num_rounds = (num_prompts + self.world_size - 1) // self.world_size
        step_dir = os.path.join(self.output_path, "benchmark", f"step_{self.step:07d}")
        os.makedirs(step_dir, exist_ok=True)

        if self.is_main_process:
            print(
                f"[Benchmark] Step {self.step}: generating {num_prompts} sample(s) "
                f"across {self.world_size} rank(s) in {num_rounds} round(s)..."
            )

        was_training = self.ode_model._generator.training
        self.ode_model._generator.eval()

        # Pass the FSDP-wrapped generator directly. FSDP all-gathers parameters
        # inside forward(), so we don't need summon_full_params. The pipeline
        # calls self.generator(...) which goes through __call__ and FSDP hooks.
        pipeline = ODEAutoregressiveBenchmarkPipeline(
            generator=self.ode_model._generator,
            add_noise_fn=self._benchmark_add_noise,
            denoising_sigmas=self.benchmark_denoising_sigmas,
            num_frame_per_block=self.benchmark_num_frame_per_block,
            num_frame_per_block_first=getattr(self.config, "num_frame_per_block_first", 4),
            clear_cuda_cache_per_round=self.benchmark_clear_cuda_cache_per_round,
            transition_mode="euler",
            use_bidirectional_bootstrap=False,
        )

        benchmark_wall_start = time.perf_counter()
        my_total_generate_seconds = 0.0

        for round_idx in range(num_rounds):
            prompt_idx = round_idx * self.world_size + self.global_rank
            has_real_prompt = prompt_idx < num_prompts
            prompt_text = self.benchmark_prompts[prompt_idx] if has_real_prompt else self.benchmark_prompts[0]

            with torch.no_grad():
                conditional_dict = self.ode_model.text_encoder(text_prompts=[prompt_text])

            prompt_seed = self.benchmark_seed + prompt_idx
            with torch.random.fork_rng(devices=[self.device]):
                torch.manual_seed(prompt_seed)
                torch.cuda.manual_seed(prompt_seed)

                generate_start = time.perf_counter()
                video_latent, audio_latent = pipeline.generate(
                    video_shape=self.benchmark_video_shape_single,
                    audio_shape=self.benchmark_audio_shape_single,
                    conditional_dict=conditional_dict,
                )
                my_total_generate_seconds += time.perf_counter() - generate_start

            if has_real_prompt:
                self._save_benchmark_latents(
                    video_latent=video_latent,
                    audio_latent=audio_latent,
                    prompt_idx=prompt_idx,
                    step_dir=step_dir,
                )

            del video_latent, audio_latent, conditional_dict
            if self.benchmark_clear_cuda_cache_per_round:
                torch.cuda.empty_cache()

            barrier()

        benchmark_wall_elapsed = time.perf_counter() - benchmark_wall_start

        total_generate_tensor = torch.tensor(
            [my_total_generate_seconds], device=self.device, dtype=torch.float64
        )
        dist.all_reduce(total_generate_tensor, op=dist.ReduceOp.SUM)
        total_generate_seconds = total_generate_tensor.item()

        if was_training:
            self.ode_model._generator.train()

        barrier()

        if self.is_main_process:
            benchmark_wandb_dict = {
                "benchmark/time_wall_seconds": benchmark_wall_elapsed,
                "benchmark/time_generate_seconds": total_generate_seconds,
            }
            prompt_rows = []

            for idx in range(num_prompts):
                latent_path = os.path.join(step_dir, f"sample_{idx}.pt")
                sample_path = None
                if os.path.exists(latent_path):
                    sample_path = self._decode_benchmark_latents(
                        latent_path=latent_path,
                        prompt_idx=idx,
                        step_dir=step_dir,
                    )
                media_key = f"benchmark/sample_{idx}"
                expected_path = os.path.join(step_dir, f"sample_{idx}.mp4")
                video_media = wandb_video_from_path(
                    sample_path or expected_path,
                    fps=self.benchmark_video_fps,
                    key=media_key,
                    required=self.wandb_video_required,
                )
                if video_media is not None:
                    benchmark_wandb_dict[media_key] = video_media
                prompt_rows.append([idx, self.benchmark_prompts[idx], sample_path or latent_path])

            if prompt_rows:
                benchmark_wandb_dict["benchmark/prompt_table"] = wandb.Table(
                    columns=["index", "prompt", "local_path"],
                    data=prompt_rows,
                )

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
                f"[Benchmark] Step {self.step}: {num_prompts} sample(s) | "
                f"saved to {step_dir}",
                flush=True,
            )
            self._maybe_upload_benchmark_to_hf(step_dir)

        barrier()

    def _resume_signature(self) -> dict:
        config = self.config
        return {
            "training_stage": self.training_stage,
            "base_checkpoint": os.path.realpath(str(config.checkpoint_path)),
            "data_path": os.path.realpath(str(config.data_path)),
            "generator_task": str(config.generator_task),
            "denoising_step_list": list(config.denoising_step_list),
            "batch_size": int(config.batch_size),
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "lr": float(config.lr),
            "beta1": float(config.get("beta1", 0.9)),
            "beta2": float(config.get("beta2", 0.999)),
            "weight_decay": float(config.get("weight_decay", 0.0)),
            "seed": int(config.seed),
            "mixed_precision": bool(config.mixed_precision),
            "sharding_strategy": str(config.sharding_strategy),
            "generator_fsdp_wrap_strategy": str(
                config.get("generator_fsdp_wrap_strategy", "transformer")
            ),
            "teacher_fp8_mode": str(config.get("teacher_fp8_mode", "")),
            "trainable_fp8_mode": str(config.get("trainable_fp8_mode", "")),
            "loss_target": str(config.get("loss_target", "velocity")),
            "uniform_timestep": bool(config.get("uniform_timestep", False)),
            "disable_causal_mask": bool(config.get("disable_causal_mask", False)),
            "enable_causal_log_rescale": bool(
                config.get("enable_causal_log_rescale", False)
            ),
            "skip_cross_modal_attention": bool(
                config.get("skip_cross_modal_attention", False)
            ),
            "video_loss_weight": float(config.get("video_loss_weight", 1.0)),
            "audio_loss_weight": float(config.get("audio_loss_weight", 1.0)),
            "num_frames": int(config.num_frames),
            "video_height": int(config.video_height),
            "video_width": int(config.video_width),
            "num_frame_per_block": int(config.get("num_frame_per_block", 3)),
            "num_frame_per_block_first": int(
                config.get("num_frame_per_block_first", 4)
            ),
        }

    def _restore_training_state(self, checkpoint_path: str) -> None:
        checkpoint_dir = os.path.dirname(os.path.realpath(checkpoint_path))
        manifest_path = os.path.join(checkpoint_dir, "trainer_state.json")
        if not os.path.isfile(manifest_path):
            raise RuntimeError(
                f"Exact resume metadata is missing: {manifest_path}. "
                "Use causal_model_checkpoint for a weights-only warm start instead."
            )

        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if int(manifest.get("format_version", 0)) != 2:
            raise RuntimeError(
                f"Unsupported Step-2 trainer-state format: {manifest.get('format_version')}"
            )
        if manifest.get("training_stage") != self.training_stage:
            raise RuntimeError(
                "Training stage changed across resume: "
                f"{manifest.get('training_stage')} != {self.training_stage}"
            )
        saved_world_size = int(manifest.get("world_size", 0))
        if saved_world_size != self.world_size:
            raise RuntimeError(
                "Exact Step-2 resume requires the original FSDP topology: "
                f"checkpoint world_size={saved_world_size}, current world_size={self.world_size}."
            )
        saved_signature = manifest.get("resume_signature")
        current_signature = self._resume_signature()
        if saved_signature != current_signature:
            keys = sorted(set(saved_signature or {}) | set(current_signature))
            mismatches = {
                key: ((saved_signature or {}).get(key), current_signature.get(key))
                for key in keys
                if (saved_signature or {}).get(key) != current_signature.get(key)
            }
            raise RuntimeError(f"Training configuration changed across resume: {mismatches}")

        checkpoint = torch.load(
            os.path.realpath(checkpoint_path),
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        if int(checkpoint.get("format_version", 0)) != 4:
            raise RuntimeError(
                f"Unsupported Step-2 model checkpoint format: {checkpoint.get('format_version')}"
            )
        if checkpoint.get("training_stage") != self.training_stage:
            raise RuntimeError("Step-2 model checkpoint has the wrong training_stage")
        if int(checkpoint.get("step", -1)) != int(manifest.get("step", -2)):
            raise RuntimeError("model.pt and trainer_state.json have different steps")
        if not isinstance(checkpoint.get("generator"), dict):
            raise RuntimeError("Step-2 model checkpoint has no generator state")
        del checkpoint

        rank_state_path = os.path.join(
            checkpoint_dir, f"trainer_state_rank_{self.global_rank:05d}.pt"
        )
        if not os.path.isfile(rank_state_path):
            raise RuntimeError(f"Exact resume shard is missing: {rank_state_path}")

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

        self.optimizer.load_state_dict(rank_state["optimizer"])
        self.step = int(manifest["step"])
        max_steps = self.config.get("max_steps", float("inf"))
        if max_steps <= self.step:
            raise RuntimeError(
                f"Resume max_steps must be greater than saved step {self.step}, got {max_steps}."
            )
        if int(rank_state.get("step", -1)) != self.step:
            raise RuntimeError(f"Resume shard step mismatch in {rank_state_path}")
        self.data_iter.load_state_dict(rank_state["data_iterator"])
        restore_rng_state(rank_state["rng_state"], device=self.device)
        del rank_state
        barrier()

    def save(self):
        """Save inference weights and same-topology distributed training state."""
        print("Gathering distributed model states...")

        generator_state_dict = fsdp_state_dict(self.ode_model._generator)

        architecture = asdict(self.ode_model._make_model_config())
        rope_type = architecture.get("rope_type")
        if hasattr(rope_type, "value"):
            architecture["rope_type"] = rope_type.value

        checkpoint_dir = os.path.join(
            self.output_path,
            f"checkpoint_{self.step:06d}"
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
            "optimizer": self.optimizer.state_dict(),
            "data_iterator": self.data_iter.state_dict(),
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
                "checkpoint_kind": "generator_with_distributed_training_state",
                "training_stage": self.training_stage,
                "optimizer_state_saved": True,
                "resume_world_size": self.world_size,
                "resume_state_pattern": "trainer_state_rank_{rank:05d}.pt",
                "generator": generator_state_dict,
                "step": self.step,
                "base_checkpoint": str(self.config.get("checkpoint_path", "")),
                "architecture": architecture,
                "training_config": OmegaConf.to_container(self.config, resolve=True),
            }
            checkpoint_path = os.path.join(checkpoint_dir, "model.pt")
            checkpoint_tmp = f"{checkpoint_path}.tmp"
            torch.save(state_dict, checkpoint_tmp)
            os.replace(checkpoint_tmp, checkpoint_path)

            manifest = {
                "format_version": 2,
                "training_stage": self.training_stage,
                "step": self.step,
                "world_size": self.world_size,
                "optimizer": "AdamW",
                "optimizer_state_saved": True,
                "rng_state_saved": ["python", "numpy", "torch", "cuda"],
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
            print(f"Model saved to {checkpoint_path}")
            self._maybe_upload_checkpoint_to_hf(checkpoint_dir)

        barrier()
        del generator_state_dict
        restore_rng_state(saved_rng_state, device=self.device)
        del saved_rng_state
        gc.collect()
        torch.cuda.empty_cache()

    def _maybe_upload_checkpoint_to_hf(self, checkpoint_dir: str) -> None:
        """Optionally upload a completed checkpoint directory to Hugging Face."""
        upload_checkpoint_to_hf(
            checkpoint_dir,
            self.config,
            output_path=self.output_path,
        )

    def _maybe_upload_benchmark_to_hf(self, benchmark_dir: str) -> None:
        """Upload the decoded fixed-prompt videos beside the matching checkpoint."""
        if not bool(self.config.get("hf_upload_benchmark", False)):
            return

        original_prefix = str(self.config.get("hf_upload_path_prefix", ""))
        benchmark_prefix = (
            f"{original_prefix.rstrip('/')}/benchmark"
            if original_prefix
            else "benchmark"
        )
        upload_checkpoint_to_hf(
            benchmark_dir,
            self.config,
            output_path=self.output_path,
            path_prefix=benchmark_prefix,
        )

    def _should_visualize(self) -> bool:
        """Check if we should visualize on this step."""
        if self.no_visualize or not self.is_main_process:
            return False
        if self.video_decoder is None:
            return False
        completed_step = self.step + 1
        return completed_step % self.visualize_iters == 0

    @torch.no_grad()
    def _add_visualization(
        self,
        log_dict: dict,
        wandb_dict: dict,
        *,
        completed_step: Optional[int] = None,
    ) -> None:
        """Decode latents and save video/audio files to output folder.

        Uses the original ltx-core decode functions (vae_decode_video / vae_decode_audio)
        which are proven correct. Saves mp4 (video+audio merged) and .wav separately.

        Args:
            log_dict: Output from generator_loss(return_samples=True).
            wandb_dict: Dict to add WandB scalars and decoded video media into.
        """
        from ltx_core.model.video_vae import decode_video as vae_decode_video
        from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
        from ltx_pipelines.utils.media_io import encode_video
        import torchaudio

        completed_step = self.step + 1 if completed_step is None else int(completed_step)
        vis_dir = os.path.join(
            self.output_path, "visualizations", f"step_{completed_step:06d}"
        )
        os.makedirs(vis_dir, exist_ok=True)

        # --- Video decode ---
        decoded_videos = {}  # key 鈫?[F, H, W, 3] uint8
        for key in ("pred_video", "target_video", "noisy_video"):
            if key not in log_dict:
                continue
            try:
                latent = log_dict[key][:1].to(device=self.device, dtype=self.dtype)
                latent_f = latent.float()
                print(f"[Vis] {key}: shape={list(latent.shape)}, "
                      f"min={latent_f.min().item():.3f}, max={latent_f.max().item():.3f}, "
                      f"mean={latent_f.mean().item():.3f}, std={latent_f.std().item():.3f}")

                # [1, F, 128, H, W] 鈫?[1, 128, F, H, W]
                latent_for_vae = latent.permute(0, 2, 1, 3, 4)
                decoded_frames = list(vae_decode_video(latent_for_vae, self.video_decoder))
                all_frames = torch.cat(decoded_frames, dim=0)  # [F_out, H, W, 3] uint8
                decoded_videos[key] = all_frames
                print(f"[Vis] {key} decoded: {list(all_frames.shape)}")
            except Exception as e:
                import traceback
                print(f"[Vis] ERROR decoding {key}: {e}")
                traceback.print_exc()

        # --- Audio decode ---
        decoded_audios = {}  # key 鈫?[channels, samples] float
        if self.audio_decoder is not None and self.vocoder is not None:
            for key in ("pred_audio", "target_audio"):
                if key not in log_dict:
                    continue
                try:
                    latent = log_dict[key][:1].to(device=self.device, dtype=self.dtype)
                    print(f"[Vis] {key}: shape={list(latent.shape)}")
                    audio_latent = latent.unflatten(-1, (8, 16)).permute(0, 2, 1, 3)
                    decoded_audio = vae_decode_audio(
                        audio_latent, self.audio_decoder, self.vocoder,
                    )
                    decoded_audios[key] = decoded_audio
                    print(f"[Vis] {key} decoded: {list(decoded_audio.shape)}, "
                          f"duration={decoded_audio.shape[-1]/self.audio_sample_rate:.2f}s")
                except Exception as e:
                    import traceback
                    print(f"[Vis] ERROR decoding {key}: {e}")
                    traceback.print_exc()

        # --- Save files ---
        # Map video keys to matching audio keys
        audio_map = {"pred_video": "pred_audio", "target_video": "target_audio",
                     "noisy_video": None}

        for vkey, frames in decoded_videos.items():
            akey = audio_map.get(vkey)
            audio = decoded_audios.get(akey) if akey else None

            # Save mp4 (video + audio merged)
            mp4_path = os.path.join(vis_dir, f"{vkey}.mp4")
            try:
                encode_video(
                    video=iter([frames]),
                    fps=24,
                    audio=audio,
                    audio_sample_rate=self.audio_sample_rate if audio is not None else None,
                    output_path=mp4_path,
                    video_chunks_number=1,
                )
                print(f"[Vis] Saved {mp4_path}")
            except Exception as e:
                print(f"[Vis] ERROR saving {mp4_path}: {e}")

            media_key = f"visualization/{vkey}"
            video_media = wandb_video_from_path(
                mp4_path,
                fps=24,
                key=media_key,
                required=self.wandb_video_required and vkey == "pred_video",
            )
            if video_media is not None:
                wandb_dict[media_key] = video_media

        if self.wandb_video_required and "visualization/pred_video" not in wandb_dict:
            raise RuntimeError(
                "Required training visualization 'pred_video' was not generated "
                f"at step {completed_step}."
            )

        # Save audio as separate .wav files
        for akey, audio in decoded_audios.items():
            wav_path = os.path.join(vis_dir, f"{akey}.wav")
            try:
                wav = audio.cpu()
                if wav.ndim == 1:
                    wav = wav.unsqueeze(0)
                torchaudio.save(wav_path, wav, sample_rate=self.audio_sample_rate)
                print(f"[Vis] Saved {wav_path}")
            except Exception as e:
                print(f"[Vis] ERROR saving {wav_path}: {e}")

        print(f"[Vis] All files saved to {vis_dir}")

    def train_one_step(self):
        """Execute one training step."""
        # IMPORTANT: Keep FSDP-wrapped generator in train() mode.
        self.ode_model._generator.train()

        # Clear per-block diagnostics from previous step
        self._block_diag.clear()

        # Log mask info once at step 0
        if self.is_main_process and self.step == 0:
            batch_peek = next(iter(self.dataloader))
            self._log_mask_info(batch_peek)

        # Get batch
        batch = next(self.data_iter)
        prompts = batch["prompts"]
        video_latent = batch["video_latent"].to(device=self.device, dtype=self.dtype)
        audio_latent = batch.get("audio_latent")
        if audio_latent is not None:
            audio_latent = audio_latent.to(device=self.device, dtype=self.dtype)
        sigmas = batch.get("sigmas")
        if sigmas is not None:
            sigmas = sigmas.to(device=self.device, dtype=torch.float32)  # keep float32 for precision

        # Encode text
        with torch.no_grad():
            conditional_dict = self.ode_model.text_encoder(text_prompts=prompts)
            if self.config.get("text_encoder_device", "cuda") == "cpu":
                conditional_dict = {
                    key: value.to(device=self.device, dtype=self.dtype, non_blocking=True)
                    if torch.is_tensor(value)
                    else value
                    for key, value in conditional_dict.items()
                }

        # Compute loss (request samples on visualization steps)
        visualize = self._should_visualize()
        loss, log_dict = self.ode_model.generator_loss(
            video_latent=video_latent,
            conditional_dict=conditional_dict,
            audio_latent=audio_latent,
            sigmas=sigmas,
            return_samples=visualize,
        )

        # Gather losses for logging
        unnormalized_loss = log_dict["unnormalized_loss"]
        timestep = log_dict["timestep"]

        if self.world_size > 1:
            gathered_loss = torch.zeros(
                [self.world_size, *unnormalized_loss.shape],
                dtype=unnormalized_loss.dtype,
                device=self.device,
            )
            gathered_timestep = torch.zeros(
                [self.world_size, *timestep.shape],
                dtype=timestep.dtype,
                device=self.device,
            )
            dist.all_gather_into_tensor(gathered_loss, unnormalized_loss)
            dist.all_gather_into_tensor(gathered_timestep, timestep)
        else:
            gathered_loss = unnormalized_loss
            gathered_timestep = timestep

        # Loss breakdown by sigma bucket
        loss_breakdown = defaultdict(list)
        for i, t in enumerate(timestep):
            # Bucket by sigma value: 0.0-0.25, 0.25-0.50, 0.50-0.75, 0.75-1.0
            sigma_val = t.item()
            t_bucket = f"sigma_{int(sigma_val * 4) / 4:.2f}"
            loss_breakdown[t_bucket].append(unnormalized_loss[i].item())

        stats = {}
        for t_bucket, losses in loss_breakdown.items():
            stats[f"loss_at_time_{t_bucket}"] = sum(losses) / len(losses)

        # Backward pass with gradient accumulation.
        # Accumulate gradients over gradient_accumulation_steps micro-batches
        # before applying an optimizer step. Loss is scaled by 1/accum_steps
        # so the effective gradient equals the mean over all micro-batches.
        accum_steps = self.gradient_accumulation_steps
        is_first_micro = (self.step % accum_steps == 0)
        is_last_micro = ((self.step + 1) % accum_steps == 0)

        if is_first_micro:
            self.optimizer.zero_grad()

        scaled_loss = loss / accum_steps

        # Use no_sync() on intermediate micro-batches to skip redundant
        # FSDP all-reduce. Only the last micro-batch triggers gradient sync.
        if not is_last_micro and hasattr(self.ode_model._generator, 'no_sync'):
            with self.ode_model._generator.no_sync():
                scaled_loss.backward()
        else:
            scaled_loss.backward()

        # === Deep diagnostics: rank 0 only, write to log file ===
        should_diag = self.is_main_process and (self.step < 10 or self.step % 50 == 0)
        if should_diag:
            self._log_deep_diagnostics(loss, log_dict)

        if is_last_micro:
            # Gradient clipping + optimizer step
            grad_norm = self.ode_model._generator.clip_grad_norm_(self.max_grad_norm)
            self.optimizer.step()
            from ltx_distillation.util import refresh_fp8_training_scales
            refresh_fp8_training_scales(self.ode_model._generator)
        else:
            grad_norm = torch.tensor(0.0)

        # Logging
        if self.is_main_process:
            grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            log_data = {
                "loss": loss.item(),
                "video_loss": log_dict["video_loss"].item(),
                "grad_norm": grad_norm_val,
                **stats,
            }
            if log_dict.get("audio_loss") is not None:
                log_data["audio_loss"] = log_dict["audio_loss"].item()

            # Brief summary to stdout (all detail goes to log file)
            if self.step < 10 or self.step % 50 == 0 or grad_norm_val > 100:
                print(
                    f"[Step {self.step}] loss={loss.item():.6f} "
                    f"v_loss={log_dict['video_loss'].item():.6f} "
                    f"grad_norm={grad_norm_val:.2e} "
                    f"(details in diagnostics.log)"
                )

            # Preserve only detached visualization tensors. Decoding and WandB
            # media upload happen after this step's checkpoint/HF upload.
            if visualize:
                visualization_keys = (
                    "pred_video",
                    "target_video",
                    "noisy_video",
                    "pred_audio",
                    "target_audio",
                )
                self._pending_visualization = {
                    key: log_dict[key].detach()
                    for key in visualization_keys
                    if key in log_dict and torch.is_tensor(log_dict[key])
                }

            completed_step = self.step + 1
            wandb.log(log_data, step=completed_step)

    def _flush_pending_visualization(self) -> None:
        if not self.is_main_process or self._pending_visualization is None:
            return

        wandb_media = {}
        pending = self._pending_visualization
        self._pending_visualization = None
        self._add_visualization(
            pending,
            wandb_media,
            completed_step=self.step,
        )
        wandb.log(wandb_media, step=self.step)
        video_count = sum(
            key.startswith("visualization/") for key in wandb_media
        )
        print(
            f"[WANDB_VIDEO] logged {video_count} training video(s) "
            f"at step {self.step}",
            flush=True,
        )
        del pending

    # ========================================================================
    # Diagnostic Logging System (rank 0 only, writes to diagnostics.log)
    # ========================================================================

    def _setup_diagnostic_log(self, config):
        """Create file-based diagnostic logger."""
        log_path = os.path.join(self.output_path, "diagnostics.log")
        logger = logging.getLogger("ode_diag")
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        fh = logging.FileHandler(log_path, mode='w')
        fh.setLevel(logging.DEBUG)
        fmt = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.propagate = False
        self.diag_logger = logger
        logger.info("=" * 100)
        logger.info("ODE TRAINING DIAGNOSTICS LOG")
        logger.info("=" * 100)
        logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")
        logger.info("=" * 100)
        print(f"[Diagnostics] Logging to {log_path}")

    def _register_diagnostic_hooks(self):
        """Register forward hooks on transformer blocks for per-layer activation tracing.

        Hooks are registered BEFORE FSDP wrapping and survive the wrapping process.
        With gradient_checkpointing, forward runs twice per block; we skip duplicates.
        """
        gen = self.ode_model._generator
        model = gen.model  # CausalLTXModel

        # Hook on patchify_proj to see transformer input
        # IMPORTANT: All hooks use torch.no_grad() to avoid creating autograd nodes
        # that conflict with gradient checkpointing (which compares saved tensor
        # counts between original forward and recomputation).
        def patchify_hook(mod, inp, out):
            with torch.no_grad():
                of = out.detach().float()
                self._block_diag.append({
                    'idx': 'v_patchify',
                    'norm': of.norm().item(),
                    'mean': of.mean().item(),
                    'std': of.std().item(),
                    'absmax': of.abs().max().item(),
                })
        model.patchify_proj.register_forward_hook(patchify_hook)

        def audio_patchify_hook(mod, inp, out):
            with torch.no_grad():
                of = out.detach().float()
                self._block_diag.append({
                    'idx': 'a_patchify',
                    'norm': of.norm().item(),
                    'mean': of.mean().item(),
                    'std': of.std().item(),
                    'absmax': of.abs().max().item(),
                })
        model.audio_patchify_proj.register_forward_hook(audio_patchify_hook)

        # Hook on each transformer block
        for i, block in enumerate(model.transformer_blocks):
            def block_hook(mod, inp, out, idx=i):
                with torch.no_grad():
                    # Skip duplicate from gradient checkpointing re-run
                    if any(d.get('idx') == idx for d in self._block_diag):
                        return
                    video_args, audio_args = out
                    vx = video_args.x.detach().float()
                    ax = audio_args.x.detach().float()
                    stats = {
                        'idx': idx,
                        'vx_norm': vx.norm().item(),
                        'vx_mean': vx.mean().item(),
                        'vx_std': vx.std().item(),
                        'vx_absmax': vx.abs().max().item(),
                        'ax_norm': ax.norm().item(),
                        'ax_mean': ax.mean().item(),
                        'ax_std': ax.std().item(),
                        'ax_absmax': ax.abs().max().item(),
                    }
                    # Read gate stats if available (set by CausalAVTransformerBlock)
                    if hasattr(mod, '_gate_stats') and mod._gate_stats:
                        stats.update(mod._gate_stats)
                        mod._gate_stats = {}
                    self._block_diag.append(stats)
            block.register_forward_hook(block_hook)
            # Enable gate stat collection on this block
            block._store_gate_stats = True

        # Hook on output norm layers
        def norm_out_hook(mod, inp, out):
            with torch.no_grad():
                of = out.detach().float()
                self._block_diag.append({
                    'idx': 'v_norm_out',
                    'norm': of.norm().item(),
                    'mean': of.mean().item(),
                    'std': of.std().item(),
                    'absmax': of.abs().max().item(),
                })
        model.norm_out.register_forward_hook(norm_out_hook)

        def audio_norm_out_hook(mod, inp, out):
            with torch.no_grad():
                of = out.detach().float()
                self._block_diag.append({
                    'idx': 'a_norm_out',
                    'norm': of.norm().item(),
                    'mean': of.mean().item(),
                    'std': of.std().item(),
                    'absmax': of.abs().max().item(),
                })
        model.audio_norm_out.register_forward_hook(audio_norm_out_hook)

        if self.diag_logger:
            self.diag_logger.info(f"Registered diagnostic hooks on {len(model.transformer_blocks)} blocks + patchify + norm_out")

    def _log_deep_diagnostics(self, loss, log_dict):
        """Comprehensive per-step diagnostics written to log file.

        Logs:
        1. Per-block activation norms (from forward hooks)
        2. Per-block gate values (scale/shift/gate from AdaLN)
        3. Per-module gradient norms (grouped + individual top-N)
        4. Critical parameter value stats (scale_shift_table, patchify, etc.)
        5. RoPE frequency stats (if captured)
        """
        log = self.diag_logger
        if log is None:
            return

        log.info("")
        log.info("=" * 100)
        log.info(f"STEP {self.step} | loss={loss.item():.6e} | v_loss={log_dict['video_loss'].item():.6e}")
        log.info("=" * 100)

        # ---- 1. Per-block activation norms ----
        log.info("")
        log.info(f"--- PER-BLOCK ACTIVATION NORMS ({len(self._block_diag)} entries) ---")
        log.info(f"{'idx':>12s}  {'norm':>12s}  {'mean':>12s}  {'std':>12s}  {'absmax':>12s}")

        # Separate special entries and block entries
        for stats in self._block_diag:
            idx = stats['idx']
            if isinstance(idx, str):
                # Special entries: patchify, norm_out
                log.info(f"  [{idx:>10s}]  {stats['norm']:>12.4e}  {stats['mean']:>12.4e}  {stats['std']:>12.4e}  {stats['absmax']:>12.4e}")
            else:
                # Block entries: video + audio side by side
                vline = (f"  Block {idx:2d} V:  {stats['vx_norm']:>12.4e}  {stats['vx_mean']:>12.4e}  "
                         f"{stats['vx_std']:>12.4e}  {stats['vx_absmax']:>12.4e}")
                aline = (f"           A:  {stats['ax_norm']:>12.4e}  {stats['ax_mean']:>12.4e}  "
                         f"{stats['ax_std']:>12.4e}  {stats['ax_absmax']:>12.4e}")
                log.info(vline)
                log.info(aline)
                # Gate values (if collected)
                if 'vgate_msa_mean' in stats:
                    gline = (f"      gates V: msa={stats['vgate_msa_mean']:.4e}+/-{stats['vgate_msa_std']:.4e}  "
                             f"mlp={stats.get('vgate_mlp_mean', 0):.4e}+/-{stats.get('vgate_mlp_std', 0):.4e}  "
                             f"scale_msa={stats.get('vscale_msa_mean', 0):.4e}+/-{stats.get('vscale_msa_std', 0):.4e}")
                    log.info(gline)
                if 'agate_msa_mean' in stats:
                    gline = (f"      gates A: msa={stats['agate_msa_mean']:.4e}+/-{stats['agate_msa_std']:.4e}  "
                             f"mlp={stats.get('agate_mlp_mean', 0):.4e}+/-{stats.get('agate_mlp_std', 0):.4e}")
                    log.info(gline)
                if 'gate_a2v_mean' in stats:
                    gline = (f"      cross:   a2v_gate={stats['gate_a2v_mean']:.4e}  "
                             f"v2a_gate={stats.get('gate_v2a_mean', 0):.4e}")
                    log.info(gline)
                # Per-attention output norms and backward gradient norms
                attn_keys = ['vx_self_attn', 'vx_text_attn', 'ax_self_attn', 'ax_text_attn', 'a2v_attn', 'v2a_attn']
                if any(f'{k}_out_norm' in stats for k in attn_keys):
                    parts = []
                    for k in attn_keys:
                        out_n = stats.get(f'{k}_out_norm')
                        out_m = stats.get(f'{k}_out_absmax')
                        grad_n = stats.get(f'grad_{k}_norm')
                        grad_m = stats.get(f'grad_{k}_absmax')
                        if out_n is not None:
                            p = f"{k}: fwd={out_n:.4e}"
                            if out_m is not None:
                                p += f"(max={out_m:.4e})"
                            if grad_n is not None:
                                p += f" bwd={grad_n:.4e}(max={grad_m:.4e})"
                            else:
                                p += " bwd=N/A"
                            parts.append(p)
                    for p in parts:
                        log.info(f"      attn: {p}")

        # ---- 2. Gradient norms ----
        grad_norms = {}
        total_norm_sq = 0.0
        for name, param in self.ode_model._generator.named_parameters():
            if param.grad is not None:
                pnorm = param.grad.data.float().norm(2).item()
                grad_norms[name] = pnorm
                total_norm_sq += pnorm ** 2
        total_norm = total_norm_sq ** 0.5

        # Group by module: transformer_blocks.{i}.{submodule}
        module_norms = defaultdict(float)
        for name, norm in grad_norms.items():
            parts = name.split(".")
            if len(parts) >= 4 and 'transformer_blocks' in name:
                # model.transformer_blocks.0.attn1.to_q.weight 鈫?model.transformer_blocks.0
                key = ".".join(parts[:4])
            elif len(parts) >= 3:
                key = ".".join(parts[:3])
            else:
                key = name
            module_norms[key] += norm ** 2
        module_norms = {k: v ** 0.5 for k, v in module_norms.items()}

        sorted_modules = sorted(module_norms.items(), key=lambda x: x[1], reverse=True)

        log.info("")
        log.info(f"--- GRADIENT NORMS (total={total_norm:.4e}, before clipping) ---")
        for name, norm in sorted_modules[:25]:
            pct = (norm / total_norm * 100) if total_norm > 0 else 0
            log.info(f"  {name:65s} {norm:>12.4e} ({pct:5.1f}%)")

        # Top 15 individual parameters
        sorted_params = sorted(grad_norms.items(), key=lambda x: x[1], reverse=True)
        log.info("")
        log.info("--- TOP 15 INDIVIDUAL PARAMETER GRADIENTS ---")
        for name, norm in sorted_params[:15]:
            log.info(f"  {name:80s} {norm:>12.4e}")

        # ---- 3. Critical parameter value stats ----
        log.info("")
        log.info("--- CRITICAL PARAMETER VALUES ---")
        keywords = ['scale_shift_table', 'patchify_proj.weight', 'patchify_proj.bias',
                    'proj_out.weight', 'proj_out.bias', 'norm_out',
                    'caption_projection', 'adaln_single']
        for name, param in self.ode_model._generator.named_parameters():
            if any(kw in name for kw in keywords):
                pf = param.data.float()
                log.info(f"  {name:65s} shape={str(list(param.shape)):20s} "
                         f"min={pf.min().item():>12.4e} max={pf.max().item():>12.4e} "
                         f"mean={pf.mean().item():>12.4e} std={pf.std().item():>12.4e}")

        # ---- 4. Log to wandb ----
        wandb_grad = {"grad/total_norm_preclip": total_norm}
        for name, norm in sorted_modules[:5]:
            wandb_grad[f"grad/{name.replace('.', '/')}"] = norm
        wandb.log(wandb_grad, step=self.step)

    def _log_mask_info(self, sample_batch):
        """Log causal mask details (called once at step 0).

        Builds masks for the data shape from the first batch and logs:
        - Type, shape, dtype of each mask
        - Sparsity (fraction of masked entries)
        - Sample slices (top-left corner) for visual inspection
        """
        log = self.diag_logger
        if log is None:
            return

        if self.config.get("disable_causal_mask", False):
            log.info("")
            log.info("=" * 100)
            log.info("CAUSAL MASK DISABLED 鈥?fully bidirectional (no mask)")
            log.info("=" * 100)

        video_latent = sample_batch["video_latent"]
        audio_latent = sample_batch.get("audio_latent")

        B, T, F_v, C, H, W = video_latent.shape
        if audio_latent is not None:
            _, _, F_a, _ = audio_latent.shape
        else:
            from ltx_causal.config import VIDEO_LATENT_FPS, AUDIO_LATENT_FPS
            video_duration = F_v / VIDEO_LATENT_FPS
            F_a = round(video_duration * AUDIO_LATENT_FPS)

        log.info("")
        log.info("=" * 100)
        log.info("CAUSAL MASK INFO")
        log.info("=" * 100)
        log.info(f"Data shapes: video=[B={B}, T={T}, F_v={F_v}, C={C}, H={H}, W={W}]")
        log.info(f"             audio F_a={F_a}")
        log.info(f"Video: frame_seqlen=384, total_tokens={F_v * 384}")
        log.info(f"Audio: frame_seqlen=1, total_tokens={F_a}")

        # Build masks
        from ltx_causal.attention.mask_builder import build_all_causal_masks, FLEX_ATTENTION_AVAILABLE
        from ltx_causal.config import CausalMaskConfig
        log.info(f"FLEX_ATTENTION_AVAILABLE = {FLEX_ATTENTION_AVAILABLE}")

        mask_config = CausalMaskConfig(
            num_frame_per_block=self.config.get("num_frame_per_block", 3),
            num_frame_per_block_first=self.config.get("num_frame_per_block_first", 4),
        )
        masks = build_all_causal_masks(F_v, F_a, config=mask_config, device=self.device)

        for name, mask in masks.items():
            log.info(f"\n  --- {name} ---")
            if isinstance(mask, torch.Tensor):
                total = mask.numel()
                if mask.dtype == torch.bool:
                    true_count = mask.sum().item()
                else:
                    true_count = (mask > float('-inf') / 2).sum().item()
                sparsity = 1.0 - true_count / total if total > 0 else 0
                log.info(f"  Type: Tensor, Shape: {list(mask.shape)}, Dtype: {mask.dtype}")
                log.info(f"  Sparsity: {sparsity:.4f} ({true_count}/{total} attend, {total - true_count} masked)")

                # Sample slice (top-left 30x30 for 2D masks)
                if mask.ndim == 2:
                    rows = min(30, mask.shape[0])
                    cols = min(30, mask.shape[1])
                    sample = mask[:rows, :cols]
                    log.info(f"  Sample (top-left {rows}x{cols}):")
                    if mask.dtype == torch.bool:
                        for r in range(rows):
                            line = ''.join('1' if sample[r, c].item() else '.' for c in range(cols))
                            log.info(f"    row {r:3d}: {line}")
                    else:
                        for r in range(rows):
                            line = ''.join('1' if sample[r, c].item() > float('-inf') / 2 else '.' for c in range(cols))
                            log.info(f"    row {r:3d}: {line}")

                    # Also show a slice from the middle (around block boundary)
                    if mask.shape[0] > 400 and mask.shape[1] > 400:
                        mid = 384  # video block boundary
                        r_start = max(0, mid - 5)
                        r_end = min(mask.shape[0], mid + 5)
                        c_start = max(0, mid - 5)
                        c_end = min(mask.shape[1], mid + 5)
                        sample_mid = mask[r_start:r_end, c_start:c_end]
                        log.info(f"  Sample around block boundary (rows {r_start}-{r_end}, cols {c_start}-{c_end}):")
                        for r in range(sample_mid.shape[0]):
                            if mask.dtype == torch.bool:
                                line = ''.join('1' if sample_mid[r, c].item() else '.' for c in range(sample_mid.shape[1]))
                            else:
                                line = ''.join('1' if sample_mid[r, c].item() > float('-inf') / 2 else '.' for c in range(sample_mid.shape[1]))
                            log.info(f"    row {r_start + r:5d}: {line}")
            else:
                # BlockMask (flexattention) - attribute names vary across PyTorch versions
                log.info(f"  Type: {type(mask).__name__} (Flexattention BlockMask)")
                # Discover available attributes
                public_attrs = [a for a in dir(mask) if not a.startswith('_')]
                log.info(f"  Available attributes: {public_attrs}")
                try:
                    # Try common attribute name variants across PyTorch versions
                    for q_attr in ('Q_LEN', 'q_len', 'seq_len_q', 'shape'):
                        if hasattr(mask, q_attr):
                            log.info(f"  {q_attr}={getattr(mask, q_attr)}")
                    for kv_attr in ('KV_LEN', 'kv_len', 'seq_len_kv'):
                        if hasattr(mask, kv_attr):
                            log.info(f"  {kv_attr}={getattr(mask, kv_attr)}")
                    for bs_attr in ('BLOCK_SIZE', 'block_size'):
                        if hasattr(mask, bs_attr):
                            log.info(f"  {bs_attr}={getattr(mask, bs_attr)}")
                    # Try to materialize the mask for visualization
                    if hasattr(mask, 'to_dense'):
                        dense = mask.to_dense()
                        total = dense.numel()
                        true_count = dense.sum().item()
                        log.info(f"  Dense shape: {list(dense.shape)}, "
                                 f"attend: {true_count}/{total} ({true_count/total*100:.1f}%)")
                    elif hasattr(mask, 'as_tensor'):
                        dense = mask.as_tensor()
                        log.info(f"  as_tensor shape: {list(dense.shape)}")
                except Exception as e:
                    log.info(f"  (Could not extract BlockMask details: {e})")

        log.info("")
        log.info("=" * 100)

    def train(self):
        """Main training loop."""
        max_steps = self.config.get("max_steps", float('inf'))
        log_iters = self.config.log_iters
        save_iters = int(self.config.get("save_iters", log_iters))
        if save_iters <= 0:
            raise ValueError(f"save_iters must be positive, got {save_iters}")
        if save_iters % self.gradient_accumulation_steps != 0:
            raise ValueError(
                "save_iters must be divisible by gradient_accumulation_steps so "
                "checkpoints are written only after complete optimizer updates."
            )
        if max_steps != float('inf') and max_steps % self.gradient_accumulation_steps != 0:
            raise ValueError(
                "max_steps must be divisible by gradient_accumulation_steps so the "
                "final checkpoint does not contain partial accumulated gradients."
            )
        last_saved_step = None

        while self.step < max_steps:
            self.train_one_step()
            self.step += 1

            # Checkpointing (save_iters defaults to log_iters for backward compat)
            if not self.config.get("no_save", False) and self.step % save_iters == 0:
                self.save()
                last_saved_step = self.step
                torch.cuda.empty_cache()
                # Ensure generator stays in train mode after save
                # (FSDP state_dict gathering can disrupt internal state)
                self.ode_model._generator.train()

            barrier()

            self._flush_pending_visualization()

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

        # Final save
        if (
            not self.config.get("no_save", False)
            and self.step != last_saved_step
        ):
            self.save()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="ODE Regression Training for LTX-2")
    parser.add_argument("--config_path", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed")
    parser.add_argument("--no_save", action="store_true", help="Disable checkpointing")
    parser.add_argument("--resume_checkpoint", default=None)

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    if args.no_save:
        config.no_save = True
    if args.resume_checkpoint:
        config.resume_checkpoint = args.resume_checkpoint

    trainer = ODETrainer(config)
    trainer.train()

    if trainer.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()



