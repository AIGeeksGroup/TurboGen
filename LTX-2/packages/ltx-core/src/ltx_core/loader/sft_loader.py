import json

import safetensors
import torch

from ltx_core.loader.primitives import StateDict, StateDictLoader
from ltx_core.loader.sd_ops import SDOps


class SafetensorsStateDictLoader(StateDictLoader):
    """
    Loads weights from safetensors files without metadata support.
    Use this for loading raw weight files. For model files that include
    configuration metadata, use SafetensorsModelStateDictLoader instead.
    """

    _PREQUANT_SCALE_SUFFIXES = (".weight_scale", ".bias_scale", ".input_scale")

    def __init__(
        self,
        fold_prequantized_fp8: bool = False,
        preserve_prequantized_fp8: bool = False,
    ):
        if fold_prequantized_fp8 and preserve_prequantized_fp8:
            raise ValueError("FP8 weights cannot be folded and preserved simultaneously")
        self.fold_prequantized_fp8 = bool(fold_prequantized_fp8)
        self.preserve_prequantized_fp8 = bool(preserve_prequantized_fp8)

    def metadata(self, path: str) -> dict:
        raise NotImplementedError("Not implemented")

    def _read_prequant_scales(
        self,
        model_paths: list[str],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        if not self.fold_prequantized_fp8:
            return {}

        scales: dict[str, torch.Tensor] = {}
        for shard_path in model_paths:
            with safetensors.safe_open(shard_path, framework="pt", device=str(device)) as handle:
                for name in handle.keys():
                    if not name.endswith((".weight_scale", ".bias_scale")):
                        continue
                    scale = handle.get_tensor(name).to(device=device, dtype=torch.float32)
                    if scale.numel() != 1:
                        raise ValueError(
                            f"Unsupported pre-quantized FP8 scale shape {tuple(scale.shape)} for {name}"
                        )
                    scales[name.removesuffix("_scale")] = scale
        return scales

    def load(self, path: str | list[str], sd_ops: SDOps | None, device: torch.device | None = None) -> StateDict:
        """
        Load state dict from path or paths (for sharded model storage) and apply sd_ops
        """
        sd = {}
        size = 0
        dtype = set()
        device = device or torch.device("cpu")
        model_paths = path if isinstance(path, list) else [path]
        prequant_scales = self._read_prequant_scales(model_paths, device)
        if self.preserve_prequantized_fp8:
            self._validate_prequantized_fp8(model_paths)
        folded_count = 0
        for shard_path in model_paths:
            with safetensors.safe_open(shard_path, framework="pt", device=str(device)) as f:
                safetensor_keys = f.keys()
                for name in safetensor_keys:
                    if self.fold_prequantized_fp8 and name.endswith(self._PREQUANT_SCALE_SUFFIXES):
                        continue
                    expected_name = name if sd_ops is None else sd_ops.apply_to_key(name)
                    if expected_name is None:
                        continue
                    value = f.get_tensor(name).to(device=device, non_blocking=True, copy=False)
                    scale = prequant_scales.get(name)
                    if value.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                        if self.fold_prequantized_fp8 and scale is None:
                            raise ValueError(
                                f"Pre-quantized FP8 tensor has no companion scale: {name}"
                            )
                    if scale is not None and value.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                        value = (value.to(torch.float32) * scale).to(torch.bfloat16)
                        folded_count += 1
                    key_value_pairs = ((expected_name, value),)
                    if sd_ops is not None:
                        key_value_pairs = sd_ops.apply_to_key_value(expected_name, value)
                    for key, value in key_value_pairs:
                        if key in sd:
                            raise ValueError(f"Duplicate tensor after checkpoint key mapping: {key}")
                        size += value.nbytes
                        dtype.add(value.dtype)
                        sd[key] = value

        if folded_count:
            print(
                f"[checkpoint] Reconstructed {folded_count} BF16 optimizer master weights "
                "from static FP8 tensors and weight scales",
                flush=True,
            )

        return StateDict(sd=sd, device=device, size=size, dtype=dtype)

    @staticmethod
    def _validate_prequantized_fp8(model_paths: list[str]) -> None:
        keys: set[str] = set()
        fp8_weights: set[str] = set()
        for shard_path in model_paths:
            with safetensors.safe_open(shard_path, framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    if name in keys:
                        raise ValueError(f"Duplicate tensor across checkpoint shards: {name}")
                    keys.add(name)
                    if handle.get_slice(name).get_dtype() in ("F8_E4M3", "F8_E5M2"):
                        fp8_weights.add(name)

        missing_scales = []
        for weight_name in sorted(fp8_weights):
            if not weight_name.endswith(".weight"):
                missing_scales.append((weight_name, "unsupported FP8 tensor kind"))
                continue
            module_name = weight_name.removesuffix(".weight")
            for suffix in (".weight_scale", ".input_scale"):
                scale_name = module_name + suffix
                if scale_name not in keys:
                    missing_scales.append((weight_name, scale_name))
        if missing_scales:
            raise ValueError(f"Incomplete pre-quantized FP8 checkpoint: {missing_scales[:20]}")


class SafetensorsModelStateDictLoader(StateDictLoader):
    """
    Loads weights and configuration metadata from safetensors model files.
    Unlike SafetensorsStateDictLoader, this loader can read model configuration
    from the safetensors file metadata via the metadata() method.
    """

    def __init__(self, weight_loader: SafetensorsStateDictLoader | None = None):
        self.weight_loader = weight_loader if weight_loader is not None else SafetensorsStateDictLoader()

    def metadata(self, path: str) -> dict:
        with safetensors.safe_open(path, framework="pt") as f:
            metadata = f.metadata() or {}
            config = json.loads(metadata.get("config", "{}"))
            transformer_config = config.setdefault("transformer", {})
            vae_config = config.setdefault("vae", {})
            vocoder_config = config.setdefault("vocoder", {})

            def infer_vae_base_channels() -> None:
                for key in f.keys():
                    if key.endswith("vae.decoder.conv_in.conv.weight"):
                        shape = f.get_slice(key).get_shape()
                        vae_config["decoder_base_channels"] = shape[0]
                    elif key.endswith("vae.encoder.conv_in.conv.weight"):
                        shape = f.get_slice(key).get_shape()
                        vae_config["encoder_base_channels"] = shape[0]

            def infer_vocoder_config() -> None:
                upsample_kernel_sizes = {}
                for key in f.keys():
                    if key.endswith("vocoder.conv_pre.weight"):
                        shape = f.get_slice(key).get_shape()
                        vocoder_config["upsample_initial_channel"] = shape[0]
                        vocoder_config["stereo"] = shape[1] == 128
                    elif ".vocoder.ups." in key and key.endswith(".weight"):
                        parts = key.split(".")
                        up_idx = parts[parts.index("ups") + 1]
                        if up_idx.isdigit():
                            shape = f.get_slice(key).get_shape()
                            upsample_kernel_sizes[int(up_idx)] = shape[2]
                    elif key.endswith("vocoder.conv_post.weight"):
                        shape = f.get_slice(key).get_shape()
                        vocoder_config["stereo"] = shape[0] == 2

                if upsample_kernel_sizes:
                    vocoder_config["upsample_kernel_sizes"] = [
                        upsample_kernel_sizes[i] for i in sorted(upsample_kernel_sizes)
                    ]

            def infer_connector(prefix: str, key_prefix: str) -> None:
                for key in f.keys():
                    if key.endswith(f"{key_prefix}.transformer_1d_blocks.0.attn1.to_q.weight"):
                        shape = f.get_slice(key).get_shape()
                        inner_dim = shape[0]
                        transformer_config.setdefault(f"{prefix}connector_inner_dim", inner_dim)
                        transformer_config.setdefault(f"{prefix}connector_attention_head_dim", 128 if inner_dim % 128 == 0 else 64)
                        transformer_config.setdefault(
                            f"{prefix}connector_num_attention_heads",
                            inner_dim // transformer_config[f"{prefix}connector_attention_head_dim"],
                        )
                        return

            infer_connector("", "model.diffusion_model.video_embeddings_connector")
            infer_connector("audio_", "model.diffusion_model.audio_embeddings_connector")
            infer_vae_base_channels()
            infer_vocoder_config()
            return config

    def load(self, path: str | list[str], sd_ops: SDOps | None = None, device: torch.device | None = None) -> StateDict:
        return self.weight_loader.load(path, sd_ops, device)

