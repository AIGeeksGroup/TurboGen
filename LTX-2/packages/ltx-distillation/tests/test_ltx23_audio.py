"""Regression tests for the native LTX-2.3 vocoder and BWE path."""

import torch

from ltx_core.model.audio_vae.model_configurator import (
    VOCODER_COMFY_KEYS_FILTER,
    VocoderConfigurator,
)
from ltx_core.model.audio_vae.vocoder import VocoderWithBWE


def _ltx23_audio_config():
    common = {
        "resblock_kernel_sizes": [3],
        "resblock_dilation_sizes": [[1, 3, 5]],
        "upsample_initial_channel": 8,
        "resblock": "AMP1",
        "activation": "snakebeta",
        "stereo": True,
        "use_bias_at_final": False,
        "use_tanh_at_final": False,
    }
    return {
        "vocoder": {
            "vocoder": {
                **common,
                "upsample_rates": [2],
                "upsample_kernel_sizes": [4],
            },
            "bwe": {
                **common,
                "upsample_rates": [3],
                "upsample_kernel_sizes": [6],
                "input_sampling_rate": 16000,
                "output_sampling_rate": 48000,
                "hop_length": 2,
                "n_fft": 16,
                "num_mels": 64,
            },
        }
    }


def test_ltx23_config_builds_native_bwe_vocoder():
    with torch.device("meta"):
        vocoder = VocoderConfigurator.from_config(_ltx23_audio_config())

    assert isinstance(vocoder, VocoderWithBWE)
    assert vocoder.input_sampling_rate == 16000
    assert vocoder.output_sample_rate == 48000
    state_keys = set(vocoder.state_dict())
    assert "vocoder.act_post.act.alpha" in state_keys
    assert "bwe_generator.act_post.act.beta" in state_keys
    assert "mel_stft.stft_fn.forward_basis" in state_keys


def test_ltx23_vocoder_key_mapping_preserves_nested_components():
    dummy = torch.empty(())
    expected = {
        "vocoder.vocoder.conv_pre.weight": "vocoder.conv_pre.weight",
        "vocoder.bwe_generator.conv_pre.weight": "bwe_generator.conv_pre.weight",
        "vocoder.mel_stft.mel_basis": "mel_stft.mel_basis",
    }

    for source, target in expected.items():
        selected = VOCODER_COMFY_KEYS_FILTER.apply_to_key(source)
        assert selected is not None
        mapped = VOCODER_COMFY_KEYS_FILTER.apply_to_key_value(selected, dummy)
        assert [result.new_key for result in mapped] == [target]
