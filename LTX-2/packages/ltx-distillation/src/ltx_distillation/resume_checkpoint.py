"""Validation and persistent materialization for exact training checkpoints."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_PART_SIZE = 16 * 1024**3
CHECKPOINT_PATTERN = re.compile(r"checkpoint_(\d+)")


@dataclass(frozen=True)
class ModelPart:
    path: Path
    offset: int
    size: int


@dataclass(frozen=True)
class CheckpointInfo:
    path: Path
    step: int
    world_size: int
    model_size: int
    part_size: int
    parts: tuple[ModelPart, ...]
    rank_pattern: str
    storage: str

    @property
    def model_path(self) -> Path:
        return self.path / "model.pt"

    @property
    def part_count(self) -> int:
        if self.parts:
            return len(self.parts)
        return math.ceil(self.model_size / self.part_size)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing file: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read JSON file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _positive_int(value: Any, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {label}: {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"invalid {label}: {parsed}")
    return parsed


def _checkpoint_step_from_name(path: Path) -> int | None:
    match = CHECKPOINT_PATTERN.fullmatch(path.name)
    return int(match.group(1)) if match else None


def inspect_checkpoint(
    checkpoint_dir: str | Path,
    *,
    expected_step: int | None = None,
    expected_model_size: int | None = None,
    expected_part_size: int | None = None,
    require_success: bool = False,
) -> CheckpointInfo:
    """Validate an exact-resume checkpoint and return its storage metadata."""

    path = Path(checkpoint_dir).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"missing checkpoint directory: {path}")

    trainer_manifest = _read_json(path / "trainer_state.json")
    if int(trainer_manifest.get("format_version", 0)) != 2:
        raise ValueError(
            "unsupported trainer_state.json format: "
            f"{trainer_manifest.get('format_version')!r}"
        )
    if trainer_manifest.get("training_stage") != "stage3_causal_dmd":
        raise ValueError(
            "training_stage mismatch: "
            f"{trainer_manifest.get('training_stage')!r}"
        )

    step = _positive_int(trainer_manifest.get("step"), "checkpoint step")
    directory_step = _checkpoint_step_from_name(path)
    if directory_step is not None and directory_step != step:
        raise ValueError(
            f"checkpoint directory step {directory_step} does not match manifest step {step}"
        )
    if expected_step is not None and step != expected_step:
        raise ValueError(f"checkpoint step mismatch: {step} != {expected_step}")

    world_size = _positive_int(trainer_manifest.get("world_size"), "world_size")
    rank_pattern = str(
        trainer_manifest.get("rank_state_pattern")
        or "trainer_state_rank_{rank:05d}.pt"
    )
    missing_rank_files = []
    for rank in range(world_size):
        try:
            rank_name = rank_pattern.format(rank=rank)
        except (IndexError, KeyError, ValueError) as exc:
            raise ValueError(f"invalid rank_state_pattern: {rank_pattern!r}") from exc
        rank_path = path / rank_name
        if not rank_path.is_file() or rank_path.stat().st_size <= 0:
            missing_rank_files.append(rank_name)
    if missing_rank_files:
        raise ValueError(
            "missing or empty rank state file(s): " + ", ".join(missing_rank_files)
        )

    model_path = path / "model.pt"
    model_size = model_path.stat().st_size if model_path.is_file() else -1
    parts_manifest_path = path / "model.pt.parts.json"
    parts_manifest = (
        _read_json(parts_manifest_path) if parts_manifest_path.is_file() else None
    )

    manifest_model_size = None
    manifest_part_size = None
    parsed_parts: list[ModelPart] = []
    if parts_manifest is not None:
        if int(parts_manifest.get("format_version", 0)) != 1:
            raise ValueError(
                "unsupported model.pt.parts.json format: "
                f"{parts_manifest.get('format_version')!r}"
            )
        if parts_manifest.get("original_filename") != "model.pt":
            raise ValueError("model.pt.parts.json has an unexpected original_filename")
        manifest_model_size = _positive_int(
            parts_manifest.get("original_size"), "split model size"
        )
        manifest_part_size = _positive_int(
            parts_manifest.get("part_size"), "split part size"
        )
        raw_parts = parts_manifest.get("parts")
        if not isinstance(raw_parts, list) or not raw_parts:
            raise ValueError("model.pt.parts.json has no parts")

        offset = 0
        for index, item in enumerate(raw_parts):
            if not isinstance(item, dict):
                raise ValueError(f"invalid split part entry at index {index}")
            expected_name = f"model.pt.part-{index:05d}"
            name = str(item.get("name", ""))
            size = _positive_int(item.get("size"), f"size for {expected_name}")
            if name != expected_name:
                raise ValueError(
                    f"unexpected split part name at index {index}: {name!r}"
                )
            parsed_parts.append(ModelPart(path / name, offset, size))
            offset += size
        if offset != manifest_model_size:
            raise ValueError(
                f"split part sizes total {offset}, expected {manifest_model_size}"
            )

    if manifest_model_size is not None and model_size == manifest_model_size:
        storage = "monolithic"
        resolved_model_size = manifest_model_size
        resolved_part_size = manifest_part_size or DEFAULT_PART_SIZE
        parts: tuple[ModelPart, ...] = ()
    elif model_size > 0 and parts_manifest is None:
        storage = "monolithic"
        resolved_model_size = model_size
        resolved_part_size = expected_part_size or DEFAULT_PART_SIZE
        parts = ()
    elif parts_manifest is not None:
        invalid_parts = []
        for part in parsed_parts:
            actual_size = part.path.stat().st_size if part.path.is_file() else -1
            if actual_size != part.size:
                invalid_parts.append(
                    f"{part.path.name}={actual_size} (expected {part.size})"
                )
        if invalid_parts:
            raise ValueError(
                "missing or invalid split model file(s): " + ", ".join(invalid_parts)
            )
        storage = f"split:{len(parsed_parts)}"
        resolved_model_size = manifest_model_size
        resolved_part_size = manifest_part_size
        parts = tuple(parsed_parts)
    else:
        raise ValueError(f"missing or empty model checkpoint: {model_path}")

    if expected_model_size is not None and resolved_model_size != expected_model_size:
        raise ValueError(
            f"model size mismatch: {resolved_model_size} != {expected_model_size}"
        )
    if expected_part_size is not None and resolved_part_size != expected_part_size:
        raise ValueError(
            f"part size mismatch: {resolved_part_size} != {expected_part_size}"
        )

    if require_success:
        success = _read_json(path / "_SUCCESS")
        expected_success = {
            "checkpoint": path.name,
            "step": step,
            "model_size": resolved_model_size,
            "world_size": world_size,
        }
        mismatches = {
            key: (success.get(key), value)
            for key, value in expected_success.items()
            if success.get(key) != value
        }
        if mismatches:
            raise ValueError(f"_SUCCESS marker mismatch: {mismatches}")

    return CheckpointInfo(
        path=path,
        step=step,
        world_size=world_size,
        model_size=resolved_model_size,
        part_size=resolved_part_size,
        parts=parts,
        rank_pattern=rank_pattern,
        storage=storage,
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _read_json_or_none(path: Path) -> dict[str, Any] | None:
    try:
        return _read_json(path)
    except ValueError:
        return None


def _link_metadata(info: CheckpointInfo, runtime_dir: Path) -> None:
    source_paths = [info.path / "trainer_state.json"]
    for rank in range(info.world_size):
        source_paths.append(info.path / info.rank_pattern.format(rank=rank))

    for source in source_paths:
        destination = runtime_dir / source.name
        temporary = runtime_dir / f".{source.name}.{os.getpid()}.link"
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        temporary.symlink_to(source)
        os.replace(temporary, destination)


def _all_part_markers_valid(
    marker_dir: Path,
    parts: Iterable[ModelPart],
    assembly_id: str,
) -> bool:
    for part in parts:
        marker = _read_json_or_none(marker_dir / f"{part.path.name}.json")
        expected = {
            "assembly_id": assembly_id,
            "name": part.path.name,
            "offset": part.offset,
            "size": part.size,
        }
        if marker != expected:
            return False
    return True


def materialize_checkpoint(
    info: CheckpointInfo,
    runtime_dir: str | Path,
    *,
    workers: int = 8,
) -> Path:
    """Build a persistent monolithic model, resuming at completed part boundaries."""

    if info.storage == "monolithic":
        return info.model_path

    runtime_path = Path(runtime_dir).expanduser().resolve()
    runtime_path.mkdir(parents=True, exist_ok=True)
    lock_root = Path(
        os.environ.get("OMNIFORCING_LOCK_ROOT", "/tmp/omniforcing-locks")
    ).expanduser()
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_key = hashlib.sha256(str(runtime_path).encode("utf-8")).hexdigest()
    lock_path = lock_root / f"step3-materialize-{lock_key}.lock"
    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        return _materialize_checkpoint_locked(info, runtime_path, workers=workers)


def _materialize_checkpoint_locked(
    info: CheckpointInfo,
    runtime_path: Path,
    *,
    workers: int,
) -> Path:
    output_path = runtime_path / "model.pt"
    partial_path = runtime_path / "model.pt.assembling"
    final_marker_path = runtime_path / "model.pt.assembled.json"
    assembly_state_path = runtime_path / "model.pt.assembly.json"
    part_marker_dir = runtime_path / ".model.pt.assembly-parts"

    fingerprint = {
        "format_version": 2,
        "source_checkpoint_dir": str(info.path),
        "expected_size": info.model_size,
        "parts": [
            {
                "name": part.path.name,
                "size": part.size,
                "mtime_ns": part.path.stat().st_mtime_ns,
            }
            for part in info.parts
        ],
    }
    fingerprint_bytes = json.dumps(
        fingerprint, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assembly_id = hashlib.sha256(fingerprint_bytes).hexdigest()
    assembly_state = {
        "format_version": 1,
        "assembly_id": assembly_id,
        "fingerprint": fingerprint,
    }

    if (
        output_path.is_file()
        and output_path.stat().st_size == info.model_size
        and _read_json_or_none(final_marker_path) == fingerprint
    ):
        _link_metadata(info, runtime_path)
        print(
            f"[STEP3_RESUME] reusing persistent assembled model: {output_path}",
            file=sys.stderr,
            flush=True,
        )
        return output_path

    state_matches = _read_json_or_none(assembly_state_path) == assembly_state
    partial_matches = (
        partial_path.is_file() and partial_path.stat().st_size == info.model_size
    )
    markers_match = _all_part_markers_valid(
        part_marker_dir, info.parts, assembly_id
    )

    if (
        output_path.is_file()
        and output_path.stat().st_size == info.model_size
        and state_matches
        and markers_match
    ):
        _write_json_atomic(final_marker_path, fingerprint)
        _link_metadata(info, runtime_path)
        print(
            f"[STEP3_RESUME] recovered completed assembled model: {output_path}",
            file=sys.stderr,
            flush=True,
        )
        return output_path

    if output_path.exists():
        raise RuntimeError(
            f"existing runtime model has no matching provenance marker: {output_path}; "
            "use a different STEP3_RESUME_RUNTIME_DIR or remove this generated cache"
        )

    if not (state_matches and partial_matches):
        part_marker_dir.mkdir(parents=True, exist_ok=True)
        for marker in part_marker_dir.glob("*.json"):
            marker.unlink()
        with partial_path.open("wb") as handle:
            handle.truncate(info.model_size)
        _write_json_atomic(assembly_state_path, assembly_state)
        completed_parts: set[str] = set()
    else:
        completed_parts = {
            part.path.name
            for part in info.parts
            if _read_json_or_none(part_marker_dir / f"{part.path.name}.json")
            == {
                "assembly_id": assembly_id,
                "name": part.path.name,
                "offset": part.offset,
                "size": part.size,
            }
        }

    pending_parts = [
        part for part in info.parts if part.path.name not in completed_parts
    ]
    completed_bytes = sum(
        part.size for part in info.parts if part.path.name in completed_parts
    )
    required_bytes = sum(part.size for part in pending_parts) + 10 * 1024**3
    free_bytes = shutil.disk_usage(runtime_path).free
    if free_bytes < required_bytes:
        raise RuntimeError(
            "not enough persistent space to assemble Stage 3 model: "
            f"free={free_bytes}, required={required_bytes}, dir={runtime_path}"
        )

    worker_count = max(1, min(int(workers), len(pending_parts) or 1))
    print(
        f"[STEP3_RESUME] assembling {len(info.parts)} model parts into {output_path} "
        f"with {worker_count} workers; reusing {len(completed_parts)} completed part(s)",
        file=sys.stderr,
        flush=True,
    )

    output_fd = os.open(partial_path, os.O_WRONLY)
    progress = {"bytes": completed_bytes}
    progress_lock = threading.Lock()
    fsync_lock = threading.Lock()

    def copy_part(part: ModelPart) -> None:
        copied = 0
        with part.path.open("rb", buffering=0) as source:
            while copied < part.size:
                chunk = source.read(min(16 * 1024**2, part.size - copied))
                if not chunk:
                    raise RuntimeError(f"unexpected EOF while reading {part.path}")
                view = memoryview(chunk)
                while view:
                    written = os.pwrite(output_fd, view, part.offset + copied)
                    if written <= 0:
                        raise RuntimeError(f"short local write for {part.path}: {written}")
                    copied += written
                    view = view[written:]
                    with progress_lock:
                        progress["bytes"] += written

        with fsync_lock:
            os.fsync(output_fd)
        marker_payload = {
            "assembly_id": assembly_id,
            "name": part.path.name,
            "offset": part.offset,
            "size": part.size,
        }
        _write_json_atomic(
            part_marker_dir / f"{part.path.name}.json", marker_payload
        )
        print(
            f"[STEP3_RESUME] assembled part: {part.path.name}",
            file=sys.stderr,
            flush=True,
        )

    try:
        if pending_parts:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=worker_count
            ) as executor:
                pending = {
                    executor.submit(copy_part, part) for part in pending_parts
                }
                while pending:
                    done, pending = concurrent.futures.wait(pending, timeout=30)
                    for future in done:
                        future.result()
                    with progress_lock:
                        copied_bytes = progress["bytes"]
                    print(
                        "[STEP3_RESUME] assembly progress: "
                        f"{copied_bytes}/{info.model_size} bytes "
                        f"({copied_bytes * 100 / info.model_size:.1f}%)",
                        file=sys.stderr,
                        flush=True,
                    )
        os.fsync(output_fd)
    finally:
        os.close(output_fd)

    if not _all_part_markers_valid(part_marker_dir, info.parts, assembly_id):
        raise RuntimeError("not all model parts have durable completion markers")
    if partial_path.stat().st_size != info.model_size:
        raise RuntimeError(
            f"assembled model size mismatch: {partial_path.stat().st_size} "
            f"!= {info.model_size}"
        )

    os.replace(partial_path, output_path)
    _write_json_atomic(final_marker_path, fingerprint)
    _link_metadata(info, runtime_path)
    try:
        assembly_state_path.unlink()
    except FileNotFoundError:
        pass
    for marker in part_marker_dir.glob("*.json"):
        marker.unlink()
    try:
        part_marker_dir.rmdir()
    except OSError:
        pass

    print(
        f"[STEP3_RESUME] persistent assembled model is ready: {output_path}",
        file=sys.stderr,
        flush=True,
    )
    return output_path


def _optional_positive_int(value: str | None, label: str) -> int | None:
    if value in (None, ""):
        return None
    return _positive_int(value, label)


def _add_expectation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--expected-step")
    parser.add_argument("--expected-model-size")
    parser.add_argument("--expected-part-size")


def _expectations(args: argparse.Namespace) -> dict[str, int | None]:
    return {
        "expected_step": _optional_positive_int(args.expected_step, "expected step"),
        "expected_model_size": _optional_positive_int(
            args.expected_model_size, "expected model size"
        ),
        "expected_part_size": _optional_positive_int(
            args.expected_part_size, "expected part size"
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--checkpoint-dir", required=True)
    _add_expectation_arguments(validate_parser)

    materialize_parser = subparsers.add_parser("materialize")
    materialize_parser.add_argument("--checkpoint-dir", required=True)
    materialize_parser.add_argument("--runtime-dir", required=True)
    materialize_parser.add_argument("--workers", type=int, default=8)
    _add_expectation_arguments(materialize_parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    expectations = _expectations(args)
    try:
        info = inspect_checkpoint(args.checkpoint_dir, **expectations)
        if args.command == "validate":
            print(
                "[STEP3_RESUME_VALIDATE] valid: "
                f"dir={info.path}, step={info.step}, world_size={info.world_size}, "
                f"model_storage={info.storage}",
                flush=True,
            )
            return 0

        model_path = materialize_checkpoint(
            info,
            args.runtime_dir,
            workers=args.workers,
        )
        print(model_path)
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[STEP3_RESUME] invalid: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
