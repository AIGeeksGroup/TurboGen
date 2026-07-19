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

    def metadata(self, path: str) -> dict:
        raise NotImplementedError("Not implemented")

    def load(self, path: str | list[str], sd_ops: SDOps, device: torch.device | None = None) -> StateDict:
        """
        Load state dict from path or paths (for sharded model storage) and apply sd_ops
        """
        sd = {}
        size = 0
        dtype = set()
        device = device or torch.device("cpu")
        model_paths = path if isinstance(path, list) else [path]
        for shard_path in model_paths:
            with safetensors.safe_open(shard_path, framework="pt", device=str(device)) as f:
                safetensor_keys = f.keys()
                for name in safetensor_keys:
                    expected_name = name if sd_ops is None else sd_ops.apply_to_key(name)
                    if expected_name is None:
                        continue
                    value = f.get_tensor(name).to(device=device, non_blocking=True, copy=False)
                    key_value_pairs = ((expected_name, value),)
                    if sd_ops is not None:
                        key_value_pairs = sd_ops.apply_to_key_value(expected_name, value)
                    for key, value in key_value_pairs:
                        size += value.nbytes
                        dtype.add(value.dtype)
                        sd[key] = value

        return StateDict(sd=sd, device=device, size=size, dtype=dtype)


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

