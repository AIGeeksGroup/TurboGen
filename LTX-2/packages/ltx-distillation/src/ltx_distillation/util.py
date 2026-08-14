"""
Utility functions for DMD distillation training.

Includes:
- FSDP wrapping and state dict handling
- Distributed training utilities
- Logging and checkpointing helpers
"""

import json
import os
import random
import subprocess
import sys
import time
from typing import Optional, Tuple, Any
from datetime import datetime, timedelta
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def artifact_path_identity(path: Any) -> str:
    """Return a storage-location-independent identity for a model artifact."""
    value = str(path or "").strip()
    return os.path.basename(os.path.normpath(value)) if value else ""


def artifact_paths_match(first: Any, second: Any) -> bool:
    """Compare artifact paths after removing machine-specific directories."""
    first_identity = artifact_path_identity(first)
    return bool(first_identity) and first_identity == artifact_path_identity(second)


def capture_rng_state(device: Optional[torch.device] = None) -> dict:
    """Capture every RNG used by the training code on the current rank."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def restore_rng_state(state: dict, device: Optional[torch.device] = None) -> None:
    """Strictly restore a state produced by :func:`capture_rng_state`."""
    required = {"python", "numpy", "torch"}
    if torch.cuda.is_available():
        required.add("cuda")
    missing = sorted(required.difference(state))
    if missing:
        raise RuntimeError(f"Checkpoint RNG state is incomplete: missing {missing}")

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available():
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_rng_state(state["cuda"], device=device)


def launch_distributed_job() -> None:
    """Initialize distributed training environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    torch.cuda.set_device(local_rank)

    if world_size > 1:
        timeout_minutes = int(os.environ.get("DIST_TIMEOUT_MINUTES", "120"))
        if rank == 0:
            print(f"[dist] init_process_group timeout_minutes={timeout_minutes}", flush=True)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=timedelta(minutes=timeout_minutes),
        )

    return rank, world_size, local_rank


def barrier() -> None:
    """Synchronization barrier for distributed training."""
    if dist.is_initialized():
        dist.barrier()


def get_sharding_strategy(strategy_name: str) -> ShardingStrategy:
    """Get FSDP sharding strategy by name."""
    strategy_map = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
        "hybrid_full": ShardingStrategy.HYBRID_SHARD,
        "hybrid_grad_op": ShardingStrategy._HYBRID_SHARD_ZERO2,
    }
    return strategy_map.get(strategy_name, ShardingStrategy.HYBRID_SHARD)


def fsdp_wrap(
    module: nn.Module,
    sharding_strategy: str = "hybrid_full",
    mixed_precision: bool = True,
    wrap_strategy: str = "size",
    transformer_module: Optional[Tuple[type, ...]] = None,
    min_num_params: int = 1e8,
) -> FSDP:
    """
    Wrap module with FSDP for distributed training.

    Args:
        module: Module to wrap
        sharding_strategy: Sharding strategy name
        mixed_precision: Use bfloat16 mixed precision
        wrap_strategy: "size" or "transformer"
        transformer_module: Transformer block classes for transformer wrapping
        min_num_params: Minimum parameters for size-based wrapping

    Returns:
        FSDP-wrapped module
    """
    # Mixed precision policy
    # Match CausVid: param in bfloat16, but reduce/buffer in float32
    # for gradient all-reduce precision and buffer accuracy.
    if mixed_precision:
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
            cast_forward_inputs=False,
        )
    else:
        mp_policy = None

    # Wrap policy
    if wrap_strategy == "transformer" and transformer_module is not None:
        wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_module,
        )
    else:
        wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=int(min_num_params),
        )

    # CPU offload

    # Wrap with FSDP
    wrapped = FSDP(
        module,
        sharding_strategy=get_sharding_strategy(sharding_strategy),
        mixed_precision=mp_policy,
        auto_wrap_policy=wrap_policy,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
    )

    return wrapped


def fsdp_state_dict(module: FSDP) -> dict:
    """
    Get full state dict from FSDP module.

    Gathers sharded parameters to rank 0.
    """
    from torch.distributed.fsdp import (
        FullStateDictConfig,
        StateDictType,
    )

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    with FSDP.state_dict_type(module, StateDictType.FULL_STATE_DICT, save_policy):
        state_dict = module.state_dict()

    return state_dict


def load_fsdp_state_dict(module: nn.Module, state_dict: dict, *, strict: bool = True):
    """Load a rank-complete state dict into FSDP or a regular module."""
    if not isinstance(module, FSDP):
        return module.load_state_dict(state_dict, strict=strict)

    from torch.distributed.fsdp import FullStateDictConfig, StateDictType

    load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
    with FSDP.state_dict_type(module, StateDictType.FULL_STATE_DICT, load_policy):
        return module.load_state_dict(state_dict, strict=strict)


def refresh_fp8_training_scales(module: nn.Module) -> None:
    """Refresh torchao FP8 weight scales after an optimizer update."""
    from ltx_core.model.transformer import precompute_fsdp_fp8_scales

    precompute_fsdp_fp8_scales(module)


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    """Read a value from an OmegaConf object, mapping, or namespace."""
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _resolve_hf_token(config: Any) -> str:
    """Resolve a Hugging Face token without printing it."""
    token = os.environ.get("HF_TOKEN", "").strip()
    token_file = str(_config_get(config, "hf_upload_token_file", "") or "")
    if not token and token_file and os.path.isfile(token_file):
        with open(token_file, "r", encoding="utf-8") as handle:
            token = handle.read().strip()
    if not token:
        try:
            from huggingface_hub import get_token

            token = (get_token() or "").strip()
        except Exception:
            token = ""
    return token


def resolve_wandb_api_key(config: Any) -> str:
    """Resolve a WandB API key from config, environment, or local login."""
    api_key = str(_config_get(config, "wandb_api_key", "") or "").strip()
    token_file = str(_config_get(config, "wandb_api_key_file", "") or "")
    if not api_key:
        api_key = os.environ.get("WANDB_API_KEY", "").strip()
    if not api_key and token_file and os.path.isfile(token_file):
        with open(token_file, "r", encoding="utf-8") as handle:
            api_key = handle.read().strip()
    if not api_key:
        try:
            import wandb

            api_key = (wandb.api.api_key or "").strip()
        except Exception:
            api_key = ""
    return api_key


def validate_artifact_upload_config(config: Any) -> None:
    """Fail before model loading when required artifact publishing cannot run."""
    if (
        bool(_config_get(config, "hf_upload_required", False))
        and not bool(_config_get(config, "no_save", False))
    ):
        if not str(_config_get(config, "hf_upload_repo_id", "") or "").strip():
            raise RuntimeError("Required Hugging Face repository is not configured")
        if not bool(_config_get(config, "hf_upload_blocking", False)):
            raise RuntimeError("Required Hugging Face uploads must be blocking")
        if not _resolve_hf_token(config):
            raise RuntimeError(
                "Required Hugging Face upload has no token. Set HF_TOKEN, "
                "hf_upload_token_file, or run `huggingface-cli login`."
            )

    if (
        bool(_config_get(config, "wandb_video_required", False))
        and not bool(_config_get(config, "no_visualize", False))
    ):
        wandb_mode = os.environ.get("WANDB_MODE", "online").strip().lower()
        if wandb_mode == "disabled":
            raise RuntimeError(
                "wandb_video_required=true is incompatible with WANDB_MODE=disabled"
            )
        if wandb_mode not in {"offline", "dryrun"} and not resolve_wandb_api_key(config):
            raise RuntimeError(
                "Required WandB video logging has no API key. Set WANDB_API_KEY, "
                "wandb_api_key_file, or run `wandb login`."
            )


def persist_checkpoint_to_split_storage(
    checkpoint_dir: str,
    config: Any,
) -> Optional[str]:
    """Persist a local checkpoint on storage with a practical per-file limit."""
    backup_root = str(
        _config_get(config, "split_checkpoint_backup_root", "") or ""
    ).strip()
    required = bool(_config_get(config, "split_checkpoint_required", False))
    if not backup_root:
        if required:
            raise RuntimeError("Required split checkpoint backup root is not configured")
        return None

    checkpoint_dir = os.path.realpath(checkpoint_dir)
    checkpoint_name = os.path.basename(os.path.normpath(checkpoint_dir))
    destination_dir = os.path.join(os.path.realpath(backup_root), checkpoint_name)
    success_path = os.path.join(destination_dir, "_SUCCESS")
    part_size = int(
        _config_get(config, "split_checkpoint_part_size", 16 * 1024**3)
    )
    visibility_timeout = int(
        _config_get(config, "split_checkpoint_visibility_timeout", 1800)
    )
    if part_size <= 0:
        raise RuntimeError(f"split_checkpoint_part_size must be positive: {part_size}")

    model_path = os.path.join(checkpoint_dir, "model.pt")
    trainer_state_path = os.path.join(checkpoint_dir, "trainer_state.json")
    if not os.path.isfile(model_path) or os.path.getsize(model_path) <= 0:
        raise RuntimeError(f"Local checkpoint model is missing or empty: {model_path}")
    if not os.path.isfile(trainer_state_path) or os.path.getsize(trainer_state_path) <= 0:
        raise RuntimeError(
            f"Local checkpoint trainer state is missing or empty: {trainer_state_path}"
        )

    with open(trainer_state_path, "r", encoding="utf-8") as handle:
        trainer_state = json.load(handle)
    world_size = int(trainer_state.get("world_size", 0))
    if world_size <= 0:
        raise RuntimeError(f"Invalid checkpoint world_size: {world_size}")
    checkpoint_step = int(trainer_state.get("step", -1))
    if checkpoint_step < 0:
        raise RuntimeError(f"Invalid checkpoint step: {checkpoint_step}")
    model_size = os.path.getsize(model_path)
    rank_pattern = str(
        trainer_state.get("rank_state_pattern")
        or "trainer_state_rank_{rank:05d}.pt"
    )
    rank_files = []
    for rank in range(world_size):
        name = rank_pattern.format(rank=rank)
        path = os.path.join(checkpoint_dir, name)
        if not os.path.isfile(path) or os.path.getsize(path) <= 0:
            raise RuntimeError(f"Local checkpoint rank state is missing or empty: {path}")
        rank_files.append((name, path, os.path.getsize(path)))

    def wait_for_size(path: str, expected_size: int) -> None:
        deadline = time.monotonic() + visibility_timeout
        last_size = -1
        while time.monotonic() < deadline:
            try:
                last_size = os.path.getsize(path)
            except OSError:
                last_size = -1
            if last_size == expected_size:
                return
            time.sleep(5)
        raise RuntimeError(
            f"Shared checkpoint file did not become visible with the expected size: "
            f"{path}, actual={last_size}, expected={expected_size}"
        )

    def write_exact(source, target_path: str, expected_size: int) -> None:
        if os.path.isfile(target_path) and os.path.getsize(target_path) == expected_size:
            source.seek(expected_size, os.SEEK_CUR)
            return

        copied = 0
        with open(target_path, "wb", buffering=0) as target:
            while copied < expected_size:
                chunk = source.read(min(64 * 1024**2, expected_size - copied))
                if not chunk:
                    raise RuntimeError(
                        f"Unexpected EOF while writing split checkpoint file: {target_path}"
                    )
                view = memoryview(chunk)
                while view:
                    written = target.write(view)
                    if not written:
                        raise RuntimeError(f"Short write while creating {target_path}")
                    view = view[written:]
                copied += len(chunk)
            os.fsync(target.fileno())

        wait_for_size(target_path, expected_size)

    def copy_file(source_path: str, target_path: str, expected_size: int) -> None:
        with open(source_path, "rb", buffering=0) as source:
            write_exact(source, target_path, expected_size)

    expected_success = {
        "format_version": 1,
        "checkpoint": checkpoint_name,
        "step": checkpoint_step,
        "model_size": model_size,
        "world_size": world_size,
    }
    if os.path.isfile(success_path) and os.path.getsize(success_path) > 0:
        with open(success_path, "r", encoding="utf-8") as handle:
            existing_success = json.load(handle)
        if all(existing_success.get(key) == value for key, value in expected_success.items()):
            print(
                f"[SPLIT_BACKUP] reusing completed checkpoint: {destination_dir}",
                flush=True,
            )
            return destination_dir
        raise RuntimeError(
            f"Split checkpoint success marker does not match local checkpoint: "
            f"{success_path}"
        )

    os.makedirs(destination_dir, exist_ok=True)
    parts = []
    with open(model_path, "rb", buffering=0) as source:
        offset = 0
        index = 0
        while offset < model_size:
            current_size = min(part_size, model_size - offset)
            name = f"model.pt.part-{index:05d}"
            target_path = os.path.join(destination_dir, name)
            print(
                f"[SPLIT_BACKUP] writing {checkpoint_name}/{name} "
                f"({current_size} bytes)",
                flush=True,
            )
            write_exact(source, target_path, current_size)
            parts.append({"name": name, "size": current_size})
            offset += current_size
            index += 1

    copy_file(
        trainer_state_path,
        os.path.join(destination_dir, "trainer_state.json"),
        os.path.getsize(trainer_state_path),
    )
    for name, source_path, size in rank_files:
        print(
            f"[SPLIT_BACKUP] writing {checkpoint_name}/{name} ({size} bytes)",
            flush=True,
        )
        copy_file(source_path, os.path.join(destination_dir, name), size)

    parts_manifest = {
        "format_version": 1,
        "original_filename": "model.pt",
        "original_size": model_size,
        "part_size": part_size,
        "parts": parts,
    }
    parts_manifest_path = os.path.join(destination_dir, "model.pt.parts.json")
    parts_manifest_bytes = (
        json.dumps(parts_manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    with open(parts_manifest_path, "wb", buffering=0) as handle:
        handle.write(parts_manifest_bytes)
        os.fsync(handle.fileno())
    wait_for_size(parts_manifest_path, len(parts_manifest_bytes))

    for item in parts:
        wait_for_size(
            os.path.join(destination_dir, item["name"]),
            int(item["size"]),
        )
    wait_for_size(
        os.path.join(destination_dir, "trainer_state.json"),
        os.path.getsize(trainer_state_path),
    )
    for name, _, size in rank_files:
        wait_for_size(os.path.join(destination_dir, name), size)
    wait_for_size(
        os.path.join(destination_dir, "model.pt.parts.json"),
        len(parts_manifest_bytes),
    )
    success_payload = dict(expected_success)
    success_payload["part_count"] = len(parts)
    success_bytes = (json.dumps(success_payload, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    with open(success_path, "wb", buffering=0) as handle:
        handle.write(success_bytes)
        os.fsync(handle.fileno())
    wait_for_size(success_path, len(success_bytes))
    print(
        f"[SPLIT_BACKUP] persisted checkpoint to {destination_dir}",
        flush=True,
    )
    return destination_dir


def upload_checkpoint_to_hf(
    checkpoint_dir: str,
    config: Any,
    output_path: Optional[str] = None,
    *,
    path_prefix: Optional[str] = None,
    api_factory: Any = None,
) -> bool:
    """Upload a completed checkpoint directory according to training config.

    Required uploads are synchronous and raise on any configuration, token, or
    network failure. Optional non-blocking uploads run in a detached process so
    training does not wait for the transfer.
    """
    repo_id = str(_config_get(config, "hf_upload_repo_id", "") or "").strip()
    required = bool(_config_get(config, "hf_upload_required", False))
    if not repo_id:
        if required:
            raise RuntimeError("Required Hugging Face repository is not configured")
        return False

    checkpoint_exists = os.path.isdir(checkpoint_dir)
    if checkpoint_exists:
        with os.scandir(checkpoint_dir) as entries:
            checkpoint_exists = any(entries)
    if not checkpoint_exists:
        message = f"Checkpoint directory is missing or empty: {checkpoint_dir}"
        if required:
            raise RuntimeError(message)
        print(f"[HF_UPLOAD] skipped: {message}", flush=True)
        return False

    token = _resolve_hf_token(config)
    if not token:
        if required:
            raise RuntimeError(
                "Required Hugging Face upload has no token. Set HF_TOKEN, "
                "hf_upload_token_file, or run `huggingface-cli login`."
            )
        print("[HF_UPLOAD] skipped: no Hugging Face token is available", flush=True)
        return False

    configured_prefix = str(_config_get(config, "hf_upload_path_prefix", "") or "")
    selected_prefix = configured_prefix if path_prefix is None else str(path_prefix)
    checkpoint_name = os.path.basename(os.path.normpath(checkpoint_dir))
    path_in_repo = (
        f"{selected_prefix.rstrip('/')}/{checkpoint_name}"
        if selected_prefix
        else checkpoint_name
    )
    repo_type = str(_config_get(config, "hf_upload_repo_type", "model"))
    create_repo = bool(_config_get(config, "hf_upload_create_repo", True))
    blocking = bool(_config_get(config, "hf_upload_blocking", False))
    if required and not blocking:
        raise RuntimeError("Required Hugging Face uploads must be blocking")

    commit_message = f"Upload {checkpoint_name}"
    try:
        if blocking:
            if api_factory is None:
                from huggingface_hub import HfApi

                api_factory = HfApi
            api = api_factory(token=token)
            print(
                f"[HF_UPLOAD] uploading {checkpoint_dir} to "
                f"{repo_id}/{path_in_repo}",
                flush=True,
            )
            if create_repo:
                api.create_repo(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    token=token,
                    exist_ok=True,
                )
            api.upload_folder(
                folder_path=checkpoint_dir,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type=repo_type,
                token=token,
                commit_message=commit_message,
            )
            print(
                f"[HF_UPLOAD] uploaded {checkpoint_dir} to "
                f"{repo_id}/{path_in_repo}",
                flush=True,
            )
            return True

        code = r'''
import os
import sys
from huggingface_hub import HfApi

checkpoint_dir, repo_id, repo_type, path_in_repo, create_repo, commit_message = sys.argv[1:]
token = os.environ["HF_TOKEN"]
api = HfApi(token=token)
if create_repo == "1":
    api.create_repo(repo_id=repo_id, repo_type=repo_type, token=token, exist_ok=True)
api.upload_folder(
    folder_path=checkpoint_dir,
    path_in_repo=path_in_repo,
    repo_id=repo_id,
    repo_type=repo_type,
    token=token,
    commit_message=commit_message,
)
print(f"[HF_UPLOAD] uploaded {checkpoint_dir} to {repo_id}/{path_in_repo}", flush=True)
'''
        cmd = [
            sys.executable,
            "-c",
            code,
            checkpoint_dir,
            repo_id,
            repo_type,
            path_in_repo,
            "1" if create_repo else "0",
            commit_message,
        ]
        env = os.environ.copy()
        env["HF_TOKEN"] = token
        env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        log_dir = output_path or os.path.dirname(os.path.normpath(checkpoint_dir))
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "hf_upload.log")
        with open(log_path, "ab", buffering=0) as log_file:
            subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                close_fds=True,
                env=env,
                start_new_session=True,
            )
        print(
            f"[HF_UPLOAD] queued {checkpoint_dir} to {repo_id}/{path_in_repo}",
            flush=True,
        )
        return True
    except Exception as exc:
        message = f"[HF_UPLOAD] failed: {type(exc).__name__}: {exc}"
        if required:
            raise RuntimeError(message) from exc
        print(message, flush=True)
        return False


def wandb_video_from_path(
    video_path: Optional[str],
    *,
    fps: int,
    key: str,
    required: bool = False,
) -> Any:
    """Validate a rendered MP4 and construct a WandB video media object."""
    path = str(video_path or "")
    if not path or not os.path.isfile(path) or os.path.getsize(path) <= 0:
        message = f"WandB video '{key}' is missing or empty: {path or '<unset>'}"
        if required:
            raise RuntimeError(message)
        print(f"[WANDB_VIDEO] skipped: {message}", flush=True)
        return None

    try:
        import wandb

        media = wandb.Video(path, fps=fps, format="mp4")
    except Exception as exc:
        message = (
            f"Failed to prepare WandB video '{key}' from {path}: "
            f"{type(exc).__name__}: {exc}"
        )
        if required:
            raise RuntimeError(message) from exc
        print(f"[WANDB_VIDEO] skipped: {message}", flush=True)
        return None

    print(f"[WANDB_VIDEO] prepared {key}: {path}", flush=True)
    return media


def init_logging_folder(config) -> Tuple[str, str]:
    """
    Initialize output and wandb folders.

    The run directory name follows the pattern: ``{MMDD}_{HHMMSS}_{wandb_name}``
    and the WandB run name is set to the same directory name so they stay in sync.

    A copy of the full config is saved as ``config.yaml`` inside the run directory.

    Args:
        config: Configuration object with output_path and wandb settings

    Returns:
        Tuple of (output_path, wandb_folder)
    """
    import wandb
    from omegaconf import OmegaConf

    # Create output directory 鈥?naming: {MMDD}_{HHMMSS}_{wandb_name}
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    run_dir_name = f"{timestamp}_{config.wandb_name}"
    output_path = os.path.join(config.output_path, run_dir_name)
    os.makedirs(output_path, exist_ok=True)

    # Save a copy of the config for reproducibility
    OmegaConf.save(config, os.path.join(output_path, "config.yaml"))

    # Initialize wandb
    configured_wandb_path = str(
        _config_get(config, "wandb_output_path", "") or ""
    ).strip()
    wandb_folder = (
        os.path.realpath(configured_wandb_path)
        if configured_wandb_path
        else os.path.join(output_path, "wandb")
    )
    os.makedirs(wandb_folder, exist_ok=True)

    # Set wandb API key from config (required for multi-node without shared ~/.netrc)
    wandb_api_key = resolve_wandb_api_key(config)
    if wandb_api_key:
        os.environ["WANDB_API_KEY"] = wandb_api_key

    # Get wandb entity (None means use default logged-in account)
    wandb_entity = getattr(config, "wandb_entity", None)
    if wandb_entity == "null" or wandb_entity == "":
        wandb_entity = None

    wandb_kwargs = dict(
        project=config.wandb_project,
        entity=wandb_entity,
        name=run_dir_name,
        config=dict(config),
        dir=wandb_folder,
        settings=wandb.Settings(
            init_timeout=int(
                _config_get(
                    config,
                    "wandb_init_timeout",
                    os.environ.get("WANDB_INIT_TIMEOUT", "600"),
                )
            ),
            symlink=False,
            disable_git=True,
        ),
    )
    print(
        f"[WandB] mode={os.environ.get('WANDB_MODE', 'online')}, "
        f"dir={wandb_folder}",
        flush=True,
    )
    try:
        wandb.init(**wandb_kwargs)
    except Exception as exc:
        if (
            bool(getattr(config, "wandb_video_required", False))
            and not bool(getattr(config, "no_visualize", False))
        ):
            raise RuntimeError(
                "Required WandB initialization failed; videos cannot be uploaded: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        # If rank 0 dies here, other ranks only report a later NCCL/TCPStore
        # failure during output_path broadcast. Fall back to disabled WandB so
        # training can proceed and the root cause remains visible on rank 0.
        print(
            f"[WandB] init failed ({type(exc).__name__}: {exc}). "
            "Falling back to disabled WandB mode."
        )
        os.environ["WANDB_MODE"] = "disabled"
        wandb.init(mode="disabled", **wandb_kwargs)

    return output_path, wandb_folder


def prepare_for_saving(tensor: torch.Tensor, max_frames: int = 16) -> Any:
    """
    Prepare tensor for wandb logging/saving.

    Args:
        tensor: Video tensor [B, C, F, H, W] or [B, F, C, H, W]
        max_frames: Maximum frames to save

    Returns:
        Wandb-compatible video object
    """
    import wandb

    # Ensure correct format [B, F, C, H, W]
    if tensor.dim() == 5:
        if tensor.shape[1] == 3:
            # [B, C, F, H, W] -> [B, F, C, H, W]
            tensor = tensor.permute(0, 2, 1, 3, 4)

    # Take first sample and limit frames
    video = tensor[0, :max_frames].cpu()

    # Normalize to [0, 255] uint8
    video = (video.clamp(0, 1) * 255).to(torch.uint8)

    # Convert to [F, H, W, C] for wandb
    video = video.permute(0, 2, 3, 1).numpy()

    return wandb.Video(video, fps=8, format="mp4")


class ResumableDistributedSampler(torch.utils.data.distributed.DistributedSampler):
    """DistributedSampler that can begin partway through its current epoch."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_index = 0

    def set_start_index(self, start_index: int) -> None:
        start_index = int(start_index)
        if not 0 <= start_index <= self.num_samples:
            raise ValueError(
                f"Sampler start_index must be in [0, {self.num_samples}], got {start_index}"
            )
        self.start_index = start_index

    def __iter__(self):
        indices = list(super().__iter__())
        return iter(indices[self.start_index :])


class ResumableDataIterator:
    """Infinite DataLoader iterator with an exact epoch/batch resume cursor.

    The datasets used by OmniForcing are deterministic. Reconstructing the
    DistributedSampler epoch and skipping the already-consumed batches therefore
    resumes at the same next sample on every rank.
    """

    FORMAT_VERSION = 1

    def __init__(self, dataloader, *, seed: int):
        if len(dataloader) <= 0:
            raise ValueError("Cannot create a resumable iterator for an empty DataLoader")
        self.dataloader = dataloader
        self.seed = int(seed)
        self.epoch = 0
        self.batch_offset = 0
        self._iterator = None

    def __iter__(self):
        return self

    def _start_epoch(self) -> None:
        sampler = getattr(self.dataloader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self.epoch)
        if hasattr(sampler, "set_start_index"):
            sampler.set_start_index(self.batch_offset * self.dataloader.batch_size)

        # Isolate DataLoader worker seeding from the model RNG and make it a
        # deterministic function of the saved epoch.
        generator = getattr(self.dataloader, "generator", None)
        if generator is not None:
            generator.manual_seed(self.seed + self.epoch)
        self._iterator = iter(self.dataloader)

    def __next__(self):
        while True:
            if self._iterator is None:
                self._start_epoch()
            try:
                batch = next(self._iterator)
            except StopIteration:
                self.epoch += 1
                self.batch_offset = 0
                self._iterator = None
                continue
            self.batch_offset += 1
            return batch

    def state_dict(self) -> dict:
        epoch = self.epoch
        batch_offset = self.batch_offset
        batches_per_epoch = len(self.dataloader)
        if batch_offset == batches_per_epoch:
            epoch += 1
            batch_offset = 0

        sampler = getattr(self.dataloader, "sampler", None)
        return {
            "format_version": self.FORMAT_VERSION,
            "epoch": epoch,
            "batch_offset": batch_offset,
            "batches_per_epoch": batches_per_epoch,
            "dataset_size": len(self.dataloader.dataset),
            "batch_size": self.dataloader.batch_size,
            "drop_last": self.dataloader.drop_last,
            "sampler_class": type(sampler).__name__,
            "sampler_seed": getattr(sampler, "seed", None),
            "sampler_rank": getattr(sampler, "rank", None),
            "sampler_num_replicas": getattr(sampler, "num_replicas", None),
            "iterator_seed": self.seed,
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state.get("format_version", 0)) != self.FORMAT_VERSION:
            raise RuntimeError(
                f"Unsupported data iterator state format: {state.get('format_version')}"
            )

        current = self.state_dict()
        immutable_keys = (
            "batches_per_epoch",
            "dataset_size",
            "batch_size",
            "drop_last",
            "sampler_class",
            "sampler_seed",
            "sampler_rank",
            "sampler_num_replicas",
            "iterator_seed",
        )
        mismatches = {
            key: (state.get(key), current.get(key))
            for key in immutable_keys
            if state.get(key) != current.get(key)
        }
        if mismatches:
            raise RuntimeError(f"DataLoader changed across resume: {mismatches}")

        epoch = int(state.get("epoch", -1))
        batch_offset = int(state.get("batch_offset", -1))
        if epoch < 0 or not 0 <= batch_offset < len(self.dataloader):
            raise RuntimeError(
                f"Invalid data iterator cursor: epoch={epoch}, batch_offset={batch_offset}"
            )

        self.epoch = epoch
        self.batch_offset = batch_offset
        self._iterator = None
        self._start_epoch()
        sampler = getattr(self.dataloader, "sampler", None)
        if hasattr(sampler, "set_start_index"):
            return

        self.batch_offset = 0
        for _ in range(batch_offset):
            try:
                next(self._iterator)
            except StopIteration as exc:
                raise RuntimeError("Data iterator checkpoint cursor exceeds epoch length") from exc
            self.batch_offset += 1


def cycle(dataloader):
    """Backward-compatible infinite iterator for callers without checkpointing."""
    return ResumableDataIterator(dataloader, seed=0)


class AverageMeter:
    """Compute and store running average."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_grad_norm(model: nn.Module) -> float:
    """Compute gradient norm."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5
