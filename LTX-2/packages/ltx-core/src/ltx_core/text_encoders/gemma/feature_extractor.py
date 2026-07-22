import math

import torch
from einops import rearrange

from ltx_core.model.model_protocol import ModelConfigurator


_GEMMA_HIDDEN_SIZE = 3840
_GEMMA_HIDDEN_LAYERS = 49
_V2_EXPECTED_CONFIG = {
    "caption_proj_before_connector": True,
    "caption_projection_first_linear": False,
    "caption_proj_input_norm": False,
    "caption_projection_second_linear": False,
}


def _stack_hidden_states(hidden_states: tuple[torch.Tensor, ...] | torch.Tensor) -> torch.Tensor:
    if isinstance(hidden_states, (list, tuple)):
        return torch.stack(hidden_states, dim=-1)
    return hidden_states


def _norm_and_concat_padded_batch(
    encoded_text: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    batch_size, _, hidden_size, num_layers = encoded_text.shape
    mask = rearrange(attention_mask.bool(), "b t -> b t 1 1")
    sequence_lengths = attention_mask.sum(dim=-1)
    masked = encoded_text.masked_fill(~mask, 0.0)
    denominator = (sequence_lengths * hidden_size).view(batch_size, 1, 1, 1)
    mean = masked.sum(dim=(1, 2), keepdim=True) / (denominator + 1e-6)
    minimum = encoded_text.masked_fill(~mask, float("inf")).amin(dim=(1, 2), keepdim=True)
    maximum = encoded_text.masked_fill(~mask, float("-inf")).amax(dim=(1, 2), keepdim=True)
    normalized = 8 * (encoded_text - mean) / (maximum - minimum + 1e-6)
    normalized = normalized.reshape(batch_size, -1, hidden_size * num_layers)
    flat_mask = rearrange(mask, "b t 1 1 -> b t 1").expand_as(normalized)
    return normalized.masked_fill(~flat_mask, 0.0)


def _norm_and_concat_per_token_rms(
    encoded_text: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    batch_size, sequence_length, hidden_size, num_layers = encoded_text.shape
    variance = encoded_text.square().mean(dim=2, keepdim=True)
    normalized = encoded_text * torch.rsqrt(variance + 1e-6)
    normalized = normalized.reshape(batch_size, sequence_length, hidden_size * num_layers)
    return torch.where(
        attention_mask.bool().unsqueeze(-1),
        normalized,
        torch.zeros_like(normalized),
    )


class GemmaFeaturesExtractorProjLinear(
    torch.nn.Module,
    ModelConfigurator["GemmaFeaturesExtractorProjLinear"],
):
    """Version-aware Gemma feature extractor for LTX-2 19B and LTX-2.3 22B."""

    def __init__(
        self,
        *,
        video_dim: int | None = None,
        audio_dim: int | None = None,
    ) -> None:
        super().__init__()
        flat_dim = _GEMMA_HIDDEN_SIZE * _GEMMA_HIDDEN_LAYERS
        self.is_v2 = video_dim is not None
        if self.is_v2:
            self.video_aggregate_embed = torch.nn.Linear(flat_dim, video_dim, bias=True)
            self.audio_aggregate_embed = torch.nn.Linear(flat_dim, audio_dim, bias=True)
        else:
            self.aggregate_embed = torch.nn.Linear(flat_dim, _GEMMA_HIDDEN_SIZE, bias=False)

    def forward(
        self,
        hidden_states: tuple[torch.Tensor, ...] | torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = _stack_hidden_states(hidden_states)
        output_dtype = encoded.dtype

        if not self.is_v2:
            normalized = _norm_and_concat_padded_batch(encoded, attention_mask)
            features = self.aggregate_embed(normalized.to(output_dtype))
            return features, features

        normalized = _norm_and_concat_per_token_rms(encoded, attention_mask).to(output_dtype)
        source_dim = encoded.shape[2]
        video_input = normalized * math.sqrt(self.video_aggregate_embed.out_features / source_dim)
        audio_input = normalized * math.sqrt(self.audio_aggregate_embed.out_features / source_dim)
        return self.video_aggregate_embed(video_input), self.audio_aggregate_embed(audio_input)

    @classmethod
    def from_config(
        cls: type["GemmaFeaturesExtractorProjLinear"],
        config: dict,
    ) -> "GemmaFeaturesExtractorProjLinear":
        transformer_config = config.get("transformer", {})
        overlapping = transformer_config.keys() & _V2_EXPECTED_CONFIG.keys()
        if not overlapping:
            return cls()

        missing = _V2_EXPECTED_CONFIG.keys() - overlapping
        if missing:
            raise ValueError(f"Incomplete LTX-2.3 text config; missing: {sorted(missing)}")
        mismatched = {
            key: (transformer_config[key], expected)
            for key, expected in _V2_EXPECTED_CONFIG.items()
            if transformer_config[key] != expected
        }
        if mismatched:
            raise ValueError(f"Unsupported LTX-2.3 text config values: {mismatched}")

        video_dim = transformer_config["num_attention_heads"] * transformer_config["attention_head_dim"]
        audio_dim = (
            transformer_config["audio_num_attention_heads"]
            * transformer_config["audio_attention_head_dim"]
        )
        return cls(video_dim=video_dim, audio_dim=audio_dim)
