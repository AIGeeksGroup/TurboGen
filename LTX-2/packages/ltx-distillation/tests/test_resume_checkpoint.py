import json
from pathlib import Path

import pytest

import ltx_distillation.resume_checkpoint as resume_checkpoint
from ltx_distillation.resume_checkpoint import (
    inspect_checkpoint,
    materialize_checkpoint,
)


def _make_split_checkpoint(tmp_path: Path) -> Path:
    checkpoint_dir = tmp_path / "checkpoint_000003"
    checkpoint_dir.mkdir()

    model = b"abcdefghij"
    parts = (model[:4], model[4:8], model[8:])
    for index, part in enumerate(parts):
        (checkpoint_dir / f"model.pt.part-{index:05d}").write_bytes(part)
    (checkpoint_dir / "model.pt").write_bytes(b"")

    (checkpoint_dir / "model.pt.parts.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "original_filename": "model.pt",
                "original_size": len(model),
                "part_size": 4,
                "parts": [
                    {"name": f"model.pt.part-{index:05d}", "size": len(part)}
                    for index, part in enumerate(parts)
                ],
            }
        ),
        encoding="utf-8",
    )
    (checkpoint_dir / "trainer_state.json").write_text(
        json.dumps(
            {
                "format_version": 2,
                "training_stage": "stage3_causal_dmd",
                "step": 3,
                "world_size": 1,
                "rank_state_pattern": "trainer_state_rank_{rank:05d}.pt",
            }
        ),
        encoding="utf-8",
    )
    (checkpoint_dir / "trainer_state_rank_00000.pt").write_bytes(b"state")
    (checkpoint_dir / "_SUCCESS").write_text(
        json.dumps(
            {
                "format_version": 1,
                "checkpoint": checkpoint_dir.name,
                "step": 3,
                "model_size": len(model),
                "world_size": 1,
                "part_count": len(parts),
            }
        ),
        encoding="utf-8",
    )
    return checkpoint_dir


def test_split_checkpoint_is_valid_and_materialized_once(tmp_path):
    checkpoint_dir = _make_split_checkpoint(tmp_path)
    info = inspect_checkpoint(
        checkpoint_dir,
        expected_step=3,
        expected_model_size=10,
        expected_part_size=4,
    )
    assert info.storage == "split:3"
    assert info.world_size == 1

    runtime_dir = tmp_path / "runtime"
    model_path = materialize_checkpoint(info, runtime_dir, workers=2)
    assert model_path.read_bytes() == b"abcdefghij"
    assert (runtime_dir / "model.pt.assembled.json").is_file()
    assert (runtime_dir / "trainer_state.json").is_symlink()
    first_mtime = model_path.stat().st_mtime_ns

    reused_path = materialize_checkpoint(info, runtime_dir, workers=2)
    assert reused_path == model_path
    assert reused_path.read_bytes() == b"abcdefghij"
    assert reused_path.stat().st_mtime_ns == first_mtime


def test_materialization_resumes_after_final_rename_interruption(tmp_path, monkeypatch):
    checkpoint_dir = _make_split_checkpoint(tmp_path)
    info = inspect_checkpoint(checkpoint_dir)
    runtime_dir = tmp_path / "runtime"
    original_replace = resume_checkpoint.os.replace

    def interrupt_final_rename(source, destination):
        if Path(destination) == runtime_dir / "model.pt":
            raise RuntimeError("simulated interruption")
        original_replace(source, destination)

    monkeypatch.setattr(resume_checkpoint.os, "replace", interrupt_final_rename)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        materialize_checkpoint(info, runtime_dir, workers=2)

    assert not (runtime_dir / "model.pt").exists()
    assert (runtime_dir / "model.pt.assembly.json").is_file()

    monkeypatch.setattr(resume_checkpoint.os, "replace", original_replace)
    model_path = materialize_checkpoint(info, runtime_dir, workers=2)
    assert model_path.read_bytes() == b"abcdefghij"
    assert not (runtime_dir / "model.pt.assembly.json").exists()
