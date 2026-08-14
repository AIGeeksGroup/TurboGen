"""Tests for exact training-state checkpoint helpers."""

import random

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from ltx_distillation.util import (
    ResumableDataIterator,
    ResumableDistributedSampler,
    capture_rng_state,
    restore_rng_state,
)


def _make_iterator(*, batch_size=2, rank=0, world_size=1):
    dataset = torch.utils.data.TensorDataset(torch.arange(13))
    sampler = ResumableDistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=17,
        drop_last=True,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        generator=torch.Generator(),
    )
    return ResumableDataIterator(dataloader, seed=1234 + rank)


def _values(iterator, batches):
    return [next(iterator)[0].tolist() for _ in range(batches)]


def test_resumable_data_iterator_continues_at_exact_next_batch():
    uninterrupted = _make_iterator()
    _values(uninterrupted, 4)
    saved_state = uninterrupted.state_dict()
    expected = _values(uninterrupted, 8)

    resumed = _make_iterator()
    resumed.load_state_dict(saved_state)

    assert _values(resumed, 8) == expected


def test_resumable_data_iterator_canonicalizes_epoch_boundary():
    uninterrupted = _make_iterator()
    _values(uninterrupted, len(uninterrupted.dataloader))

    saved_state = uninterrupted.state_dict()
    assert saved_state["epoch"] == 1
    assert saved_state["batch_offset"] == 0

    resumed = _make_iterator()
    resumed.load_state_dict(saved_state)
    assert _values(resumed, 3) == _values(uninterrupted, 3)


def test_resumable_data_iterator_rejects_changed_loader():
    iterator = _make_iterator(batch_size=2)
    _values(iterator, 1)
    resumed = _make_iterator(batch_size=3)

    with pytest.raises(RuntimeError, match="DataLoader changed across resume"):
        resumed.load_state_dict(iterator.state_dict())


def test_resumable_sampler_does_not_reload_skipped_samples():
    class CountingDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.accesses = 0

        def __len__(self):
            return 20

        def __getitem__(self, index):
            self.accesses += 1
            return index

    def make_counting_iterator():
        dataset = CountingDataset()
        sampler = ResumableDistributedSampler(
            dataset, num_replicas=1, rank=0, shuffle=True, seed=5
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=2,
            sampler=sampler,
            generator=torch.Generator(),
        )
        return dataset, ResumableDataIterator(loader, seed=9)

    _, original = make_counting_iterator()
    for _ in range(6):
        next(original)
    state = original.state_dict()

    resumed_dataset, resumed = make_counting_iterator()
    resumed.load_state_dict(state)
    assert resumed_dataset.accesses == 0
    next(resumed)
    assert resumed_dataset.accesses == 2


def test_rng_state_restores_python_numpy_and_torch():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    state = capture_rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(3))

    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    restore_rng_state(state)
    actual = (random.random(), np.random.rand(), torch.rand(3))

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def _make_tiny_dmd_trainer(tmp_path):
    import ltx_distillation.train_distillation as training

    trainer = training.Trainer.__new__(training.Trainer)
    trainer.config = OmegaConf.create({"max_steps": 20})
    trainer.training_stage = "stage3_causal_dmd"
    trainer.global_rank = 0
    trainer.world_size = 1
    trainer.is_main_process = True
    trainer.device = (
        torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    trainer.output_path = str(tmp_path)
    trainer.step = 3
    trainer.last_saved_step = None
    trainer.gradient_accumulation_steps = 1
    trainer.ema_weight = 0.0
    trainer.ema_start_step = 200
    trainer.generator_ema = None
    trainer.dmd = type("TinyDMD", (), {})()
    trainer.dmd.generator = torch.nn.Linear(3, 2)
    trainer.dmd.fake_score = torch.nn.Linear(3, 1)
    trainer.generator_optimizer = torch.optim.AdamW(
        trainer.dmd.generator.parameters(), lr=1e-3
    )
    trainer.critic_optimizer = torch.optim.AdamW(
        trainer.dmd.fake_score.parameters(), lr=2e-3
    )
    trainer.generator_scheduler = torch.optim.lr_scheduler.LambdaLR(
        trainer.generator_optimizer, lambda step: 1.0 / (step + 1)
    )
    trainer.critic_scheduler = torch.optim.lr_scheduler.LambdaLR(
        trainer.critic_optimizer, lambda step: 1.0 / (step + 1)
    )
    trainer.dataloader = _make_iterator()
    trainer._resume_signature = lambda: {"test_signature": 1}
    return trainer


def _initialize_adam_state(trainer):
    trainer.generator_optimizer.zero_grad()
    trainer.dmd.generator(torch.ones(1, 3)).sum().backward()
    trainer.generator_optimizer.step()
    trainer.critic_optimizer.zero_grad()
    trainer.dmd.fake_score(torch.ones(1, 3)).sum().backward()
    trainer.critic_optimizer.step()


def test_dmd_checkpoint_restores_complete_training_state(tmp_path, monkeypatch):
    import ltx_distillation.train_distillation as training

    monkeypatch.setattr(training, "fsdp_state_dict", lambda module: module.state_dict())
    monkeypatch.setattr(training, "barrier", lambda: None)
    monkeypatch.setattr(training, "upload_checkpoint_to_hf", lambda *args, **kwargs: True)

    original = _make_tiny_dmd_trainer(tmp_path)
    _initialize_adam_state(original)
    _values(original.dataloader, 4)
    original.generator_scheduler.step()
    original.critic_scheduler.step()
    original.save()

    checkpoint_path = tmp_path / "checkpoint_000003" / "model.pt"
    expected_parameters = {
        name: value.detach().clone()
        for name, value in original.dmd.generator.state_dict().items()
    }
    expected_next_batch = next(original.dataloader)[0].tolist()
    expected_random = (random.random(), np.random.rand(), torch.rand(2))

    resumed = _make_tiny_dmd_trainer(tmp_path / "resumed")
    resumed._restore_training_state(str(checkpoint_path))

    for name, value in resumed.dmd.generator.state_dict().items():
        assert torch.equal(value, expected_parameters[name])
    assert next(resumed.dataloader)[0].tolist() == expected_next_batch
    assert resumed.generator_optimizer.state_dict()["state"]
    assert resumed.critic_optimizer.state_dict()["state"]
    assert resumed.generator_scheduler.state_dict() == original.generator_scheduler.state_dict()
    assert resumed.critic_scheduler.state_dict() == original.critic_scheduler.state_dict()
    assert random.random() == expected_random[0]
    assert np.random.rand() == expected_random[1]
    assert torch.equal(torch.rand(2), expected_random[2])


def test_dmd_resume_signature_allows_relocated_absolute_paths():
    import ltx_distillation.train_distillation as training

    saved_signature = {
        "base_checkpoint": (
            "/raid/mc1max/zeyu/omniforcing/OmniTurbo/checkpoints/"
            "LTX-2.3-fp8/ltx-2.3-22b-dev-fp8.safetensors"
        ),
        "data_path": (
            "/raid/mc1max/zeyu/omniforcing/OmniTurbo/LTX-2/"
            "packages/ltx-distillation/ode_lmdb"
        ),
        "training_stage": "stage3_causal_dmd",
    }
    current_signature = {
        "base_checkpoint": (
            "/data/minghua/zzy/OmniForcing/checkpoints/"
            "LTX-2.3-fp8/ltx-2.3-22b-dev-fp8.safetensors"
        ),
        "data_path": (
            "/data/minghua/zzy/OmniForcing/LTX-2/packages/"
            "ltx-distillation/ode_lmdb"
        ),
        "training_stage": "stage3_causal_dmd",
    }

    assert training.Trainer._resume_signature_mismatches(
        saved_signature, current_signature
    ) == {}


def test_dmd_resume_signature_still_rejects_real_config_changes():
    import ltx_distillation.train_distillation as training

    mismatches = training.Trainer._resume_signature_mismatches(
        {
            "base_checkpoint": "/old/path/model.safetensors",
            "data_path": "/old/path/ode_lmdb",
            "generator_lr": 2e-5,
        },
        {
            "base_checkpoint": "/new/path/model.safetensors",
            "data_path": "/new/path/ode_lmdb",
            "generator_lr": 1e-5,
        },
    )

    assert mismatches == {"generator_lr": (2e-5, 1e-5)}

