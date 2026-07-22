#!/usr/bin/env python
"""Validate the local LTX-2.3 FP8 runtime and the project FSDP integration."""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path


def _add_repo_paths() -> None:
    packages_dir = Path(__file__).resolve().parents[2]
    for package in ("ltx-distillation", "ltx-causal", "ltx-core", "ltx-pipelines"):
        source = packages_dir / package / "src"
        if source.exists():
            sys.path.insert(0, str(source))


_add_repo_paths()

import torch
import torch.distributed as dist


def validate_runtime(checkpoint: Path) -> dict:
    import torchao
    from ltx_core.model.transformer import checkpoint_fp8_module_names
    from safetensors import safe_open

    if not torch.cuda.is_available():
        raise RuntimeError("LTX-2.3 FP8 requires CUDA")
    if not hasattr(torch, "_scaled_mm"):
        raise RuntimeError("This PyTorch build has no torch._scaled_mm")
    if torchao.__version__ != "0.11.0":
        raise RuntimeError(f"Expected torchao==0.11.0, got {torchao.__version__}")

    device_count = torch.cuda.device_count()
    devices = []
    for index in range(device_count):
        capability = torch.cuda.get_device_capability(index)
        if capability < (9, 0):
            raise RuntimeError(f"GPU {index} does not provide native FP8 support: sm_{capability[0]}{capability[1]}")
        devices.append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(capability),
            }
        )

    quantized_names = checkpoint_fp8_module_names(checkpoint)
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        config = json.loads((handle.metadata() or {}).get("config", "{}"))
        vocoder_tensor_count = sum(
            key.startswith("vocoder.") for key in handle.keys()
        )
    transformer = config.get("transformer", {})
    required = {
        "rope_type": "split",
        "apply_gated_attention": True,
        "cross_attention_adaln": True,
        "audio_num_attention_heads": 32,
        "audio_attention_head_dim": 64,
        "frequencies_precision": "float64",
        "use_middle_indices_grid": True,
        "causal_temporal_positioning": True,
    }
    mismatched = {
        key: (transformer.get(key), expected)
        for key, expected in required.items()
        if transformer.get(key) != expected
    }
    if mismatched:
        raise RuntimeError(f"LTX-2.3 architecture metadata mismatch: {mismatched}")
    if len(quantized_names) != 1496:
        raise RuntimeError(f"Expected 1496 FP8 Linear layers, got {len(quantized_names)}")
    vocoder = config.get("vocoder", {})
    bwe = vocoder.get("bwe", {})
    if vocoder_tensor_count != 1227 or bwe.get("output_sampling_rate") != 48000:
        raise RuntimeError(
            "Incomplete LTX-2.3 audio runtime metadata: "
            f"vocoder_tensors={vocoder_tensor_count}, "
            f"output_sampling_rate={bwe.get('output_sampling_rate')}"
        )

    return {
        "torch": torch.__version__,
        "torchao": torchao.__version__,
        "devices": devices,
        "fp8_linear_layers": len(quantized_names),
        "vocoder_tensors": vocoder_tensor_count,
        "audio_output_sample_rate": bwe["output_sampling_rate"],
        "architecture": required,
    }


def distributed_smoke() -> None:
    from ltx_core.model.transformer import convert_to_fp8_training
    from ltx_distillation.ema import EMA_FSDP
    from ltx_distillation.util import fsdp_state_dict, fsdp_wrap, refresh_fp8_training_scales
    from torchao.float8.float8_linear import Float8Linear

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group("nccl")

    torch.manual_seed(1234)
    model = torch.nn.Sequential(
        torch.nn.Linear(32, 64, bias=True),
        torch.nn.GELU(),
        torch.nn.Linear(64, 32, bias=True),
    ).to(dtype=torch.bfloat16)
    convert_to_fp8_training(model, {"0", "2"})
    model = model.to(device=torch.device("cuda", local_rank))
    wrapped = fsdp_wrap(
        model,
        sharding_strategy="full_shard",
        mixed_precision=True,
        wrap_strategy="size",
        min_num_params=1_000_000,
    )
    optimizer = torch.optim.AdamW(wrapped.parameters(), lr=1e-3)
    ema = EMA_FSDP(wrapped, decay=0.9)

    inputs = torch.randn(16, 32, device=local_rank, dtype=torch.bfloat16)
    loss = wrapped(inputs).float().square().mean()
    loss.backward()
    optimizer.step()
    refresh_fp8_training_scales(wrapped)
    ema.update(wrapped)

    optimizer_buffer = io.BytesIO()
    torch.save(optimizer.state_dict(), optimizer_buffer)
    optimizer_buffer.seek(0)
    restored_optimizer_state = torch.load(optimizer_buffer, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(restored_optimizer_state)
    ema.copy_to(wrapped)

    fp8_layers = [module for module in wrapped.modules() if isinstance(module, Float8Linear)]
    if len(fp8_layers) != 2:
        raise RuntimeError(f"FSDP smoke expected 2 Float8Linear layers, got {len(fp8_layers)}")
    global_grad_presence = torch.tensor(
        [layer.weight.grad is not None for layer in fp8_layers],
        device=local_rank,
        dtype=torch.int32,
    )
    if dist.is_initialized():
        dist.all_reduce(global_grad_presence, op=dist.ReduceOp.MAX)
    if not bool(global_grad_presence.all()):
        raise RuntimeError("FSDP FP8 training produced a globally missing weight gradient")

    full_state = fsdp_state_dict(wrapped)
    if rank == 0 and set(full_state) != {"0.weight", "0.bias", "2.weight", "2.bias"}:
        raise RuntimeError(f"Unexpected FSDP state keys: {sorted(full_state)}")
    if rank == 0:
        print(
            json.dumps(
                {
                    "FSDP_FP8_VALID": 1,
                    "world_size": world_size,
                    "loss": float(loss.item()),
                    "fp8_layers": len(fp8_layers),
                    "state_tensors": len(full_state),
                    "ema_tensors": len(ema.state_dict()),
                    "optimizer_state_entries": len(restored_optimizer_state["state"]),
                },
                indent=2,
            ),
            flush=True,
        )
    if dist.is_initialized():
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--distributed-smoke", action="store_true")
    args = parser.parse_args()

    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps({"RUNTIME_VALID": 1, **validate_runtime(args.checkpoint)}, indent=2), flush=True)
    if args.distributed_smoke:
        distributed_smoke()


if __name__ == "__main__":
    main()
