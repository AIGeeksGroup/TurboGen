"""Regression tests for the LTX-2.3 causal architecture port."""

import pytest
import torch

from ltx_causal.rope.causal_rope import CausalRopeType
from ltx_causal.transformer.causal_model import CausalLTXModel, CausalLTXModelConfig
from ltx_core.model.transformer.model import LTXModel
from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.model.transformer.fp8 import (
    StaticFP8Linear,
    convert_to_fp8_training,
    convert_to_static_fp8,
    fp8_inference_state_dict,
)
from ltx_core.text_encoders.gemma.embeddings_connector import Embeddings1DConnector


def _tiny_causal_config(**overrides) -> CausalLTXModelConfig:
    values = {
        "num_layers": 1,
        "video_dim": 12,
        "audio_dim": 12,
        "video_heads": 2,
        "audio_heads": 2,
        "video_d_head": 6,
        "audio_d_head": 6,
        "cross_attention_dim": 12,
        "audio_cross_attention_dim": 12,
        "in_channels": 4,
        "out_channels": 4,
        "audio_in_channels": 4,
        "audio_out_channels": 4,
        "caption_channels": 12,
        "audio_caption_channels": 12,
        "use_caption_projection": False,
        "apply_gated_attention": True,
        "cross_attention_adaln": True,
        "rope_type": CausalRopeType.SPLIT,
        "video_frame_seqlen": 1,
    }
    values.update(overrides)
    return CausalLTXModelConfig(**values)


def test_ltx23_metadata_builds_complete_architecture():
    config = CausalLTXModelConfig.from_checkpoint_config(
        {
            "transformer": {
                "num_layers": 2,
                "num_attention_heads": 4,
                "attention_head_dim": 8,
                "audio_num_attention_heads": 2,
                "audio_attention_head_dim": 8,
                "rope_type": "split",
                "caption_proj_before_connector": True,
                "caption_projection_first_linear": False,
                "caption_projection_second_linear": False,
                "cross_attention_adaln": True,
                "apply_gated_attention": True,
                "frequencies_precision": "float64",
                "use_middle_indices_grid": True,
                "causal_temporal_positioning": True,
            }
        }
    )

    assert config.video_dim == 32
    assert config.audio_dim == 16
    assert config.rope_type is CausalRopeType.SPLIT
    assert config.use_caption_projection is False
    assert config.cross_attention_adaln is True
    assert config.apply_gated_attention is True
    assert config.double_precision_rope is True
    assert config.use_middle_indices_grid is True
    assert config.causal_temporal_positioning is True


def test_causal_and_bidirectional_parameter_layouts_match():
    causal_config = _tiny_causal_config()
    with torch.device("meta"):
        causal = CausalLTXModel(causal_config)
        bidirectional = LTXModel(
            num_attention_heads=causal_config.video_heads,
            attention_head_dim=causal_config.video_d_head,
            in_channels=causal_config.in_channels,
            out_channels=causal_config.out_channels,
            num_layers=causal_config.num_layers,
            cross_attention_dim=causal_config.cross_attention_dim,
            caption_channels=causal_config.caption_channels,
            audio_num_attention_heads=causal_config.audio_heads,
            audio_attention_head_dim=causal_config.audio_d_head,
            audio_in_channels=causal_config.in_channels,
            audio_out_channels=causal_config.out_channels,
            audio_cross_attention_dim=causal_config.audio_cross_attention_dim,
            rope_type=LTXRopeType.SPLIT,
            use_caption_projection=False,
            apply_gated_attention=True,
            cross_attention_adaln=True,
        )

    causal_shapes = {key: tuple(value.shape) for key, value in causal.state_dict().items()}
    bidirectional_shapes = {
        key: tuple(value.shape) for key, value in bidirectional.state_dict().items()
    }
    assert causal_shapes == bidirectional_shapes


class _RecordingAdaLN(torch.nn.Module):
    def __init__(self, dim: int, coefficient: int):
        super().__init__()
        self.dim = dim
        self.coefficient = coefficient
        self.last_timestep = None

    def forward(self, timestep, hidden_dtype=None):
        self.last_timestep = timestep.detach().clone()
        dtype = hidden_dtype or timestep.dtype
        values = torch.zeros(
            timestep.numel(),
            self.coefficient * self.dim,
            dtype=dtype,
            device=timestep.device,
        )
        embedded = torch.zeros(
            timestep.numel(),
            self.dim,
            dtype=dtype,
            device=timestep.device,
        )
        return values, embedded


def test_cross_modal_gate_uses_other_modality_scalar_sigma():
    model = CausalLTXModel(_tiny_causal_config(num_layers=0))
    scale_shift = _RecordingAdaLN(dim=12, coefficient=4)
    gate = _RecordingAdaLN(dim=12, coefficient=1)

    model._prepare_cross_attention_timestep(
        modality_timestep=torch.tensor([[0.2, 0.3]]),
        cross_modality_sigma=torch.tensor([0.8]),
        cross_scale_shift_adaln=scale_shift,
        cross_gate_adaln=gate,
        batch_size=1,
        hidden_dtype=torch.float32,
    )

    assert torch.equal(scale_shift.last_timestep, torch.tensor([200.0, 300.0]))
    assert torch.equal(gate.last_timestep, torch.tensor([0.8]))


def test_prompt_adaln_uses_explicit_scalar_sigma():
    model = CausalLTXModel(_tiny_causal_config(num_layers=0))
    prompt_adaln = _RecordingAdaLN(dim=12, coefficient=2)

    model._prepare_prompt_timestep(
        sigma=torch.tensor([0.7]),
        prompt_adaln=prompt_adaln,
        batch_size=1,
        hidden_dtype=torch.float32,
    )

    assert torch.equal(prompt_adaln.last_timestep, torch.tensor([700.0]))


def test_temporal_cross_rope_expands_split_layout_without_flattening_heads():
    cos = torch.arange(1 * 2 * 3 * 4, dtype=torch.float32).reshape(1, 2, 3, 4)
    sin = -cos

    expanded_cos, expanded_sin = CausalLTXModel._expand_temporal_rope_to_tokens(
        (cos, sin),
        frame_seqlen=2,
    )

    assert expanded_cos.shape == (1, 2, 6, 4)
    assert expanded_sin.shape == expanded_cos.shape
    assert torch.equal(expanded_cos[:, :, 0], cos[:, :, 0])
    assert torch.equal(expanded_cos[:, :, 1], cos[:, :, 0])
    assert torch.equal(expanded_cos[:, :, 2], cos[:, :, 1])
    assert torch.equal(expanded_sin, -expanded_cos)


def test_ltx23_connector_rejects_feature_dimension_mismatch():
    connector = Embeddings1DConnector(
        attention_head_dim=4,
        num_attention_heads=2,
        num_layers=0,
        num_learnable_registers=None,
    )

    with pytest.raises(ValueError, match="Refusing to pad or truncate"):
        connector(torch.zeros((1, 4, 7)))


def test_static_fp8_linear_preserves_checkpoint_dtypes_across_to():
    layer = StaticFP8Linear(16, 16, bias=True)

    layer.to(dtype=torch.float16)

    assert layer.weight.dtype is torch.float8_e4m3fn
    assert layer.weight_scale.dtype is torch.float32
    assert layer.input_scale.dtype is torch.float32
    assert layer.bias.dtype is torch.float16


def test_static_fp8_linear_rejects_non_fp8_weight():
    layer = StaticFP8Linear(16, 16, bias=False)
    state_dict = {
        "weight": torch.zeros(16, 16, dtype=torch.bfloat16),
        "weight_scale": torch.ones((), dtype=torch.float32),
        "input_scale": torch.ones((), dtype=torch.float32),
    }

    with pytest.raises(RuntimeError, match="dtype mismatch"):
        layer.load_state_dict(state_dict, strict=True)


def test_fp8_training_export_strict_static_reload():
    pytest.importorskip("torchao")
    training_model = torch.nn.Sequential(torch.nn.Linear(16, 16, bias=True))
    convert_to_fp8_training(training_model, {"0"})
    exported = fp8_inference_state_dict(
        training_model,
        {"0": torch.tensor(0.125, dtype=torch.float32)},
    )

    static_model = torch.nn.Sequential(torch.nn.Linear(16, 16, bias=True))
    convert_to_static_fp8(static_model, {"0"})
    static_model.load_state_dict(exported, strict=True)

    assert isinstance(static_model[0], StaticFP8Linear)
    assert static_model[0].weight.dtype is torch.float8_e4m3fn
    assert static_model[0].weight_scale.dtype is torch.float32
    assert static_model[0].input_scale.dtype is torch.float32
