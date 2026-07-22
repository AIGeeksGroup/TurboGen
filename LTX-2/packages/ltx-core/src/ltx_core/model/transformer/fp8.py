"""FP8 execution helpers for LTX-2.3 transformers."""

from __future__ import annotations

import json
from collections.abc import Iterable
from contextlib import AbstractContextManager
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

from ltx_core.loader.module_ops import ModuleOps


FP8_DTYPE = torch.float8_e4m3fn


def _validate_ltx23_metadata(metadata: dict, checkpoint_path: str | Path) -> None:
    model_version = metadata.get("model_version", "")
    if not model_version.startswith("2.3"):
        raise RuntimeError(
            f"Expected an LTX-2.3 checkpoint, got model_version={model_version!r}: {checkpoint_path}"
        )
    config = json.loads(metadata.get("config", "{}"))
    transformer = config.get("transformer", {})
    required = {
        "num_layers": 48,
        "num_attention_heads": 32,
        "attention_head_dim": 128,
        "cross_attention_dim": 4096,
        "audio_num_attention_heads": 32,
        "audio_attention_head_dim": 64,
        "audio_cross_attention_dim": 2048,
        "in_channels": 128,
        "out_channels": 128,
        "audio_out_channels": 128,
        "rope_type": "split",
        "frequencies_precision": "float64",
        "use_middle_indices_grid": True,
        "causal_temporal_positioning": True,
        "apply_gated_attention": True,
        "cross_attention_adaln": True,
        "av_cross_ada_norm": True,
        "caption_proj_before_connector": True,
        "caption_projection_first_linear": False,
        "caption_projection_second_linear": False,
    }
    mismatched = {
        key: (transformer.get(key), expected)
        for key, expected in required.items()
        if transformer.get(key) != expected
    }
    if mismatched:
        raise RuntimeError(f"Incomplete LTX-2.3 architecture metadata: {mismatched}")


def checkpoint_fp8_module_names(checkpoint_path: str | Path) -> frozenset[str]:
    """Return strictly validated transformer module names declared as FP8."""
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        _validate_ltx23_metadata(metadata, checkpoint_path)
        raw = metadata.get("_quantization_metadata")
        if not raw:
            raise RuntimeError(f"Checkpoint has no _quantization_metadata: {checkpoint_path}")
        quantization = json.loads(raw)
        layers = quantization.get("layers")
        if not isinstance(layers, dict) or not layers:
            raise RuntimeError(f"Checkpoint has no quantized layers: {checkpoint_path}")

        declared = {
            name
            for name, spec in layers.items()
            if isinstance(spec, dict) and spec.get("format") == "float8_e4m3fn"
        }
        keys = set(handle.keys())
        actual = {
            key.removesuffix(".weight")
            for key in keys
            if key.endswith(".weight") and handle.get_slice(key).get_dtype() == "F8_E4M3"
        }
        if declared != actual:
            raise RuntimeError(
                "FP8 checkpoint metadata/tensor mismatch: "
                f"metadata_only={sorted(declared - actual)[:20]}, "
                f"tensor_only={sorted(actual - declared)[:20]}"
            )
        incomplete = {
            name: [
                scale_key
                for scale_key in (f"{name}.weight_scale", f"{name}.input_scale")
                if scale_key not in keys
            ]
            for name in declared
        }
        incomplete = {name: missing for name, missing in incomplete.items() if missing}
        if incomplete:
            raise RuntimeError(f"FP8 checkpoint has incomplete scales: {list(incomplete.items())[:20]}")

    prefixes = ("model.diffusion_model.", "model.velocity_model.", "model.")
    result = set()
    for name in declared:
        for prefix in prefixes:
            if name.startswith(prefix):
                result.add(name.removeprefix(prefix))
                break
        else:
            result.add(name)
    if not result:
        raise RuntimeError(f"Checkpoint declares no LTX transformer FP8 layers: {checkpoint_path}")
    return frozenset(result)


class StaticFP8Linear(nn.Module):
    """Frozen FP8 linear layer backed directly by scaled-mm checkpoint tensors."""

    def __init__(self, in_features: int, out_features: int, bias: bool, device: torch.device | None = None):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.register_buffer("weight", torch.empty(out_features, in_features, dtype=FP8_DTYPE, device=device))
        self.register_buffer("weight_scale", torch.empty((), dtype=torch.float32, device=device))
        self.register_buffer("input_scale", torch.empty((), dtype=torch.float32, device=device))
        if bias:
            self.register_buffer("bias", torch.empty(out_features, dtype=torch.bfloat16, device=device))
        else:
            self.register_buffer("bias", None)

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "StaticFP8Linear":
        return cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.device.type != "cuda":
            raise RuntimeError("StaticFP8Linear requires a CUDA device")
        if input.dtype not in (torch.bfloat16, torch.float16):
            raise RuntimeError(f"StaticFP8Linear expects BF16/FP16 activations, got {input.dtype}")
        if not torch.isfinite(self.input_scale) or float(self.input_scale.item()) <= 0:
            raise RuntimeError("Invalid FP8 input_scale")
        if not torch.isfinite(self.weight_scale) or float(self.weight_scale.item()) <= 0:
            raise RuntimeError("Invalid FP8 weight_scale")

        from torchao.float8.inference import addmm_float8_unwrapped_inference, preprocess_data
        from torchao.float8.inference import Float8MMConfig

        original_shape = input.shape
        input_2d = input.reshape(-1, self.in_features)
        fp8_limit = torch.finfo(FP8_DTYPE).max
        input_fp8 = (input_2d.float() / self.input_scale).clamp(-fp8_limit, fp8_limit).to(FP8_DTYPE)
        weight_t = self.weight.t()
        input_fp8, weight_t = preprocess_data(
            input_fp8,
            weight_t,
            Float8MMConfig(use_fast_accum=True, pad_inner_dim=False),
        )
        output = addmm_float8_unwrapped_inference(
            input_fp8,
            self.input_scale,
            weight_t,
            self.weight_scale,
            output_dtype=input.dtype,
            bias=self.bias,
            use_fast_accum=True,
        )
        return output.reshape(*original_shape[:-1], self.out_features)

    def _apply(self, fn, recurse: bool = True):
        """Move the layer without allowing a parent ``to(dtype=...)`` to corrupt FP8 state."""
        weight = self._buffers.pop("weight")
        weight_scale = self._buffers.pop("weight_scale")
        input_scale = self._buffers.pop("input_scale")
        try:
            super()._apply(fn, recurse=recurse)
            probe = torch.empty(0, dtype=torch.uint8, device=weight.device)
            target_device = fn(probe).device
            self._buffers["weight"] = weight.to(device=target_device)
            self._buffers["weight_scale"] = weight_scale.to(device=target_device)
            self._buffers["input_scale"] = input_scale.to(device=target_device)
        except Exception:
            self._buffers["weight"] = weight
            self._buffers["weight_scale"] = weight_scale
            self._buffers["input_scale"] = input_scale
            raise
        return self

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        expected_dtypes = {
            "weight": FP8_DTYPE,
            "weight_scale": torch.float32,
            "input_scale": torch.float32,
        }
        for name, expected_dtype in expected_dtypes.items():
            key = prefix + name
            value = state_dict.get(key)
            if value is not None and value.dtype != expected_dtype:
                error_msgs.append(
                    f"Static FP8 checkpoint dtype mismatch for {key}: "
                    f"got {value.dtype}, expected {expected_dtype}"
                )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


def _replace_named_linears(module: nn.Module, names: Iterable[str], replacement_factory) -> int:
    replaced = 0
    for name in sorted(set(names), key=lambda value: value.count("."), reverse=True):
        try:
            child = module.get_submodule(name)
        except AttributeError as exc:
            raise RuntimeError(f"Quantized checkpoint layer is absent from model: {name}") from exc
        if not isinstance(child, nn.Linear):
            raise TypeError(f"Quantized checkpoint layer is not nn.Linear: {name} ({type(child).__name__})")
        parent_name, _, child_name = name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        setattr(parent, child_name, replacement_factory(child))
        replaced += 1
    return replaced


def static_fp8_module_op(checkpoint_path: str | Path) -> ModuleOps:
    names = checkpoint_fp8_module_names(checkpoint_path)

    def mutate(model: nn.Module) -> nn.Module:
        replaced = _replace_named_linears(model, names, StaticFP8Linear.from_linear)
        if replaced != len(names):
            raise RuntimeError(f"Static FP8 conversion incomplete: replaced={replaced}, expected={len(names)}")
        print(f"[fp8] Native static scaled-mm Linear layers: {replaced}", flush=True)
        return model

    return ModuleOps(
        name="ltx23_static_fp8_scaled_mm",
        matcher=lambda _model: True,
        mutator=mutate,
    )


def convert_to_static_fp8(module: nn.Module, quantized_module_names: Iterable[str]) -> nn.Module:
    """Replace selected Linear layers with frozen static scaled-mm modules."""
    names = frozenset(quantized_module_names)
    replaced = _replace_named_linears(module, names, StaticFP8Linear.from_linear)
    if replaced != len(names):
        raise RuntimeError(f"Static FP8 conversion incomplete: replaced={replaced}, expected={len(names)}")
    return module


def convert_to_fp8_training(module: nn.Module, quantized_module_names: Iterable[str]) -> nn.Module:
    """Replace checkpoint-quantized Linear layers with torchao FP8 training layers."""
    try:
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training
    except ImportError as exc:
        raise RuntimeError("FP8 training requires torchao==0.11.0") from exc

    names = frozenset(quantized_module_names)
    config = Float8LinearConfig(
        # torchao 0.11 implements FP8 all-gather only for composable FSDP2.
        # OmniForcing uses FSDP1 for checkpointing/EMA, so it must all-gather
        # the BF16 master while Float8Linear still runs all three GEMMs in FP8.
        enable_fsdp_float8_all_gather=False,
        force_recompute_fp8_weight_in_bwd=True,
        pad_inner_dim=True,
    )
    convert_to_float8_training(
        module,
        module_filter_fn=lambda child, fqn: isinstance(child, nn.Linear) and fqn in names,
        config=config,
    )

    from torchao.float8.float8_linear import Float8Linear

    converted = {name for name, child in module.named_modules() if isinstance(child, Float8Linear)}
    missing = sorted(names - converted)
    unexpected = sorted(converted - names)
    if missing or unexpected:
        raise RuntimeError(
            "FP8 training conversion did not match checkpoint metadata: "
            f"missing={missing[:20]}, unexpected={unexpected[:20]}"
        )
    return module


def precompute_fsdp_fp8_scales(module: nn.Module) -> None:
    """Refresh torchao FSDP FP8 weight scales after an optimizer update."""
    try:
        from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp
    except ImportError as exc:
        raise RuntimeError("FP8 training requires torchao==0.11.0") from exc
    precompute_float8_dynamic_scale_for_fsdp(module)


def _plain_tensor(tensor: torch.Tensor) -> torch.Tensor:
    inner = getattr(tensor, "_tensor", None)
    return inner if torch.is_tensor(inner) else tensor


def quantize_weight_to_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a trained master weight using a scalar dequantization scale."""
    weight = _plain_tensor(weight).detach().float()
    amax = weight.abs().amax().clamp_min(torch.finfo(torch.float32).tiny)
    scale = (amax / torch.finfo(FP8_DTYPE).max).to(torch.float32)
    quantized = (weight / scale).clamp(-torch.finfo(FP8_DTYPE).max, torch.finfo(FP8_DTYPE).max).to(FP8_DTYPE)
    return quantized, scale


class FP8InputScaleCalibrator(AbstractContextManager):
    """Collect per-Linear activation maxima for static FP8 student export."""

    def __init__(self, module: nn.Module):
        try:
            from torchao.float8.float8_linear import Float8Linear
        except ImportError as exc:
            raise RuntimeError("FP8 calibration requires torchao==0.11.0") from exc
        self.module = module
        self.layers = {name: child for name, child in module.named_modules() if isinstance(child, Float8Linear)}
        if not self.layers:
            raise RuntimeError("No torchao Float8Linear layers found for calibration")
        self.amax = {name: torch.zeros((), dtype=torch.float32) for name in self.layers}
        self.handles = []

    def __enter__(self) -> "FP8InputScaleCalibrator":
        for name, layer in self.layers.items():
            def collect(_module, inputs, layer_name=name):
                value = inputs[0].detach().float().abs().amax().cpu()
                self.amax[layer_name] = torch.maximum(self.amax[layer_name], value)

            self.handles.append(layer.register_forward_pre_hook(collect))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def input_scales(self) -> dict[str, torch.Tensor]:
        missing = sorted(name for name, value in self.amax.items() if float(value.item()) <= 0.0)
        if missing:
            raise RuntimeError(f"Calibration did not observe FP8 layers: {missing[:20]}")
        limit = torch.finfo(FP8_DTYPE).max
        return {name: (value / limit).clamp_min(torch.finfo(torch.float32).tiny) for name, value in self.amax.items()}


def fp8_inference_state_dict(
    module: nn.Module,
    input_scales: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Create a strict static-FP8 state dict from a torchao training module."""
    try:
        from torchao.float8.float8_linear import Float8Linear
    except ImportError as exc:
        raise RuntimeError("FP8 export requires torchao==0.11.0") from exc

    fp8_layers = {name: child for name, child in module.named_modules() if isinstance(child, Float8Linear)}
    if set(fp8_layers) != set(input_scales):
        raise RuntimeError(
            "FP8 export scale coverage mismatch: "
            f"missing={sorted(set(fp8_layers) - set(input_scales))[:20]}, "
            f"unexpected={sorted(set(input_scales) - set(fp8_layers))[:20]}"
        )

    result: dict[str, torch.Tensor] = {}
    fp8_weight_keys = {f"{name}.weight" for name in fp8_layers}
    for key, value in module.state_dict().items():
        if key in fp8_weight_keys:
            continue
        result[key] = _plain_tensor(value).detach().cpu().contiguous()

    for name, layer in fp8_layers.items():
        weight, weight_scale = quantize_weight_to_fp8(layer.weight)
        result[f"{name}.weight"] = weight.cpu().contiguous()
        result[f"{name}.weight_scale"] = weight_scale.cpu()
        result[f"{name}.input_scale"] = input_scales[name].detach().cpu().to(torch.float32)
    return result
