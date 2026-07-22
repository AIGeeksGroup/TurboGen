"""Tests for periodic checkpoint and WandB media publishing helpers."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ltx_distillation import util


class RecordingHfApi:
    instances = []

    def __init__(self, token):
        self.token = token
        self.create_calls = []
        self.upload_calls = []
        self.__class__.instances.append(self)

    def create_repo(self, **kwargs):
        self.create_calls.append(kwargs)

    def upload_folder(self, **kwargs):
        self.upload_calls.append(kwargs)


def _required_hf_config():
    return {
        "hf_upload_repo_id": "example/omniforcing",
        "hf_upload_repo_type": "model",
        "hf_upload_path_prefix": "step2_ltx23",
        "hf_upload_create_repo": True,
        "hf_upload_blocking": True,
        "hf_upload_required": True,
    }


def test_required_hf_upload_uses_expected_repository_path(tmp_path, monkeypatch):
    checkpoint_dir = tmp_path / "checkpoint_000500"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.pt").write_bytes(b"checkpoint")
    monkeypatch.setenv("HF_TOKEN", "test-token")
    RecordingHfApi.instances.clear()

    uploaded = util.upload_checkpoint_to_hf(
        str(checkpoint_dir),
        _required_hf_config(),
        api_factory=RecordingHfApi,
    )

    assert uploaded is True
    api = RecordingHfApi.instances[-1]
    assert api.token == "test-token"
    assert api.create_calls == [
        {
            "repo_id": "example/omniforcing",
            "repo_type": "model",
            "token": "test-token",
            "exist_ok": True,
        }
    ]
    assert api.upload_calls[0]["folder_path"] == str(checkpoint_dir)
    assert api.upload_calls[0]["path_in_repo"] == (
        "step2_ltx23/checkpoint_000500"
    )
    assert api.upload_calls[0]["commit_message"] == "Upload checkpoint_000500"


def test_required_hf_upload_rejects_missing_token(tmp_path, monkeypatch):
    checkpoint_dir = tmp_path / "checkpoint_000500"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.pt").write_bytes(b"checkpoint")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(util, "_resolve_hf_token", lambda config: "")

    with pytest.raises(RuntimeError, match="has no token"):
        util.upload_checkpoint_to_hf(
            str(checkpoint_dir),
            _required_hf_config(),
            api_factory=RecordingHfApi,
        )


def test_upload_preflight_fails_before_training_without_hf_token(monkeypatch):
    config = {
        **_required_hf_config(),
        "wandb_video_required": True,
    }
    monkeypatch.setattr(util, "_resolve_hf_token", lambda config: "")

    with pytest.raises(RuntimeError, match="has no token"):
        util.validate_artifact_upload_config(config)


def test_upload_preflight_allows_offline_wandb_debug(monkeypatch):
    config = {
        **_required_hf_config(),
        "wandb_video_required": True,
    }
    monkeypatch.setattr(util, "_resolve_hf_token", lambda config: "test-token")
    monkeypatch.setenv("WANDB_MODE", "offline")

    util.validate_artifact_upload_config(config)


def test_upload_preflight_rejects_disabled_wandb(monkeypatch):
    config = {
        **_required_hf_config(),
        "wandb_video_required": True,
    }
    monkeypatch.setattr(util, "_resolve_hf_token", lambda config: "test-token")
    monkeypatch.setenv("WANDB_MODE", "disabled")

    with pytest.raises(RuntimeError, match="WANDB_MODE=disabled"):
        util.validate_artifact_upload_config(config)


def test_required_wandb_video_rejects_missing_file(tmp_path):
    with pytest.raises(RuntimeError, match="missing or empty"):
        util.wandb_video_from_path(
            str(tmp_path / "missing.mp4"),
            fps=24,
            key="benchmark/sample_0",
            required=True,
        )


def test_distillation_save_uploads_completed_checkpoint(tmp_path, monkeypatch):
    import ltx_distillation.train_distillation as training

    trainer = training.Trainer.__new__(training.Trainer)
    trainer.dmd = SimpleNamespace(generator=object(), fake_score=object())
    trainer.generator_ema = None
    trainer.is_main_process = True
    trainer.output_path = str(tmp_path)
    trainer.config = _required_hf_config()
    trainer.step = 500
    trainer.last_saved_step = None
    trainer.global_rank = 0
    trainer.world_size = 1
    trainer.training_stage = "stage1_bidirectional_dmd"
    trainer.device = None
    trainer.generator_optimizer = SimpleNamespace(
        state_dict=lambda: {"generator_optimizer": True}
    )
    trainer.critic_optimizer = SimpleNamespace(
        state_dict=lambda: {"critic_optimizer": True}
    )
    trainer.generator_scheduler = None
    trainer.critic_scheduler = None
    trainer.dataloader = SimpleNamespace(
        state_dict=lambda: {"epoch": 2, "batch_offset": 3}
    )
    trainer._resume_signature = lambda: {"training_stage": trainer.training_stage}

    states = iter([{"generator.weight": 1}, {"critic.weight": 2}])
    monkeypatch.setattr(training, "fsdp_state_dict", lambda module: next(states))
    monkeypatch.setattr(
        training,
        "capture_rng_state",
        lambda device: {"python": 1, "numpy": 2, "torch": 3, "cuda": 4},
    )
    monkeypatch.setattr(training, "restore_rng_state", lambda state, device: None)
    upload_calls = []
    monkeypatch.setattr(
        training,
        "upload_checkpoint_to_hf",
        lambda checkpoint_dir, config, output_path: upload_calls.append(
            (checkpoint_dir, config, output_path)
        ),
    )

    trainer.save()

    checkpoint_dir = tmp_path / "checkpoint_000500"
    assert (checkpoint_dir / "model.pt").is_file()
    assert (checkpoint_dir / "trainer_state.json").is_file()
    assert (checkpoint_dir / "trainer_state_rank_00000.pt").is_file()
    assert not (checkpoint_dir / "model.pt.tmp").exists()
    assert upload_calls == [
        (str(checkpoint_dir), trainer.config, trainer.output_path)
    ]
    assert trainer.last_saved_step == 500


def test_distillation_uploads_checkpoint_before_benchmark_video(monkeypatch):
    import ltx_distillation.train_distillation as training

    trainer = training.Trainer.__new__(training.Trainer)
    trainer.config = SimpleNamespace(
        no_save=False,
        no_visualize=False,
        max_steps=1,
    )
    trainer.step = 0
    trainer.save_iters = 1
    trainer.last_saved_step = None
    trainer.ema_weight = 0.0
    trainer.ema_start_step = 200
    trainer.generator_ema = None
    trainer.generator_scheduler = None
    trainer.critic_scheduler = None
    trainer.benchmark_enabled = True
    trainer.benchmark_prompts = ["test prompt"]
    trainer.benchmark_iters = 1
    trainer.is_main_process = False
    trainer.previous_time = None

    events = []

    def train_one_step():
        trainer.step += 1
        events.append("train")

    def save():
        trainer.last_saved_step = trainer.step
        events.append("checkpoint")

    trainer.train_one_step = train_one_step
    trainer.save = save
    trainer._run_benchmark_and_log = lambda: events.append("video")
    monkeypatch.setattr(training, "barrier", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    trainer.train()

    assert events == ["train", "checkpoint", "video"]


def test_ode_uploads_checkpoint_before_training_video(monkeypatch):
    import ltx_distillation.ode.train_ode as training
    from omegaconf import OmegaConf

    trainer = training.ODETrainer.__new__(training.ODETrainer)
    trainer.config = OmegaConf.create(
        {"max_steps": 1, "log_iters": 1, "save_iters": 1, "no_save": False}
    )
    trainer.gradient_accumulation_steps = 1
    trainer.step = 0
    trainer.is_main_process = False
    trainer.previous_time = None
    trainer.ode_model = SimpleNamespace(
        _generator=SimpleNamespace(train=lambda: None)
    )

    events = []
    trainer.train_one_step = lambda: events.append("train")
    trainer.save = lambda: events.append("checkpoint")
    trainer._flush_pending_visualization = lambda: events.append("video")
    trainer._should_run_benchmark = lambda: False
    monkeypatch.setattr(training, "barrier", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    trainer.train()

    assert events == ["train", "checkpoint", "video"]


@pytest.mark.parametrize(
    ("config_name", "expected_prefix", "expected_stage"),
    [
        ("stage1_bidirectional_dmd.yaml", "step1_ltx23", "stage1_bidirectional_dmd"),
        ("stage2_causal_ode.yaml", "step2_ltx23", "stage2_causal_ode"),
        ("stage3_causal_dmd.yaml", "step3_ltx23", "stage3_causal_dmd"),
    ],
)
def test_training_configs_publish_every_500_steps(
    config_name, expected_prefix, expected_stage
):
    from omegaconf import OmegaConf

    config_path = Path(__file__).parents[1] / "configs" / config_name
    config = OmegaConf.load(config_path)

    assert config.save_iters == 500
    assert config.benchmark_iters == 500
    assert config.wandb_video_required is True
    assert config.hf_upload_required is True
    assert config.hf_upload_blocking is True
    assert config.hf_upload_path_prefix == expected_prefix
    assert config.hf_upload_token_file == "../../../.hf_token"
    assert config.wandb_api_key_file == "../../../.wandb_key"
    assert config.benchmark_audio_sample_rate == 48000
    assert config.training_stage == expected_stage
    assert config.resume_checkpoint is None


@pytest.mark.parametrize(
    "script_name",
    [
        "train_stage1_bidirectional_dmd.sh",
        "train_stage2_causal_ode.sh",
        "train_stage3_causal_dmd.sh",
    ],
)
def test_training_scripts_default_to_verified_hf_mirror(script_name):
    script_path = Path(__file__).parents[1] / "scripts" / script_name
    script = script_path.read_text(encoding="utf-8")

    assert 'HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"' in script
    assert 'HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"' in script
