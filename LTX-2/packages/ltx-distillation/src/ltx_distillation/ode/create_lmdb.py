"""
LMDB Creation Script for ODE Trajectory Pairs.

This script converts individual .pt trajectory files into an LMDB database
for efficient training data loading.

Each .pt file contains:
    - "prompt": str
    - "video_trajectory": [1, T, F, C, H, W] video latents at T timesteps
    - "audio_trajectory": [1, T, F_a, C] audio latents at T timesteps

The LMDB stores:
    - video_latents_{idx}_data: video trajectory bytes
    - audio_latents_{idx}_data: audio trajectory bytes
    - sigmas_{idx}_data: sigma values (float32) per trajectory entry
    - prompts_{idx}_data: prompt string bytes
    - *_shape: array shapes for reconstruction
"""

import os
import glob
import argparse
import json
from typing import Dict, Any, Set
from tqdm import tqdm

import numpy as np
import torch
import lmdb


ODE_FORMAT_VERSION = 4
ODE_PRODUCER = "omniforcing-ltx23-full-architecture-v2"


def store_arrays_to_lmdb(
    env: lmdb.Environment,
    arrays_dict: Dict[str, Any],
    start_index: int = 0,
) -> int:
    """
    Store arrays in LMDB database.

    Args:
        env: LMDB environment
        arrays_dict: Dictionary with array names and values
        start_index: Starting index for keys

    Returns:
        Number of entries stored
    """
    count = 0
    with env.begin(write=True) as txn:
        for array_name, array in arrays_dict.items():
            if isinstance(array, (list, np.ndarray)):
                for i, row in enumerate(array):
                    if isinstance(row, str):
                        row_bytes = row.encode('utf-8')
                    else:
                        row_bytes = row.tobytes()

                    data_key = f'{array_name}_{start_index + i}_data'.encode()
                    txn.put(data_key, row_bytes)
                count = max(count, len(array))
            else:
                # Single value
                if isinstance(array, str):
                    row_bytes = array.encode('utf-8')
                else:
                    row_bytes = array.tobytes()
                data_key = f'{array_name}_{start_index}_data'.encode()
                txn.put(data_key, row_bytes)
                count = 1

    return count


def get_array_shape_from_lmdb(
    env: lmdb.Environment,
    array_name: str,
) -> tuple:
    """Get array shape from LMDB metadata."""
    with env.begin() as txn:
        shape_bytes = txn.get(f"{array_name}_shape".encode())
        if shape_bytes is None:
            raise KeyError(f"Shape not found for {array_name}")
        shape_str = shape_bytes.decode()
        shape = tuple(map(int, shape_str.split()))
    return shape


def retrieve_row_from_lmdb(
    env: lmdb.Environment,
    array_name: str,
    dtype,
    row_index: int,
    shape: tuple = None,
):
    """Retrieve a specific row from LMDB."""
    data_key = f'{array_name}_{row_index}_data'.encode()

    with env.begin() as txn:
        row_bytes = txn.get(data_key)

    if row_bytes is None:
        raise KeyError(f"Key not found: {array_name}_{row_index}")

    if dtype == str:
        return row_bytes.decode('utf-8')
    else:
        array = np.frombuffer(row_bytes, dtype=dtype)
        if shape is not None and len(shape) > 0:
            array = array.reshape(shape)
        return array


def process_trajectory_file(
    file_path: str,
    seen_prompts: Set[str],
    manifest: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Process a single trajectory .pt file.

    Args:
        file_path: Path to .pt file
        seen_prompts: Set of already processed prompts (for deduplication)

    Returns:
        Dictionary with processed arrays, or None if duplicate
    """
    data = torch.load(file_path, map_location='cpu', weights_only=False)

    if manifest is not None:
        if data.get("format_version") != manifest.get("format_version"):
            raise ValueError(f"Trajectory format mismatch in {file_path}")
        if data.get("producer") != manifest.get("producer"):
            raise ValueError(f"Trajectory producer mismatch in {file_path}")
        if os.path.abspath(data.get("teacher_checkpoint", "")) != os.path.abspath(
            manifest.get("teacher_checkpoint", "")
        ):
            raise ValueError(f"Teacher checkpoint provenance mismatch in {file_path}")
        if data.get("generation_config") != manifest.get("generation_config"):
            raise ValueError(f"Generation config provenance mismatch in {file_path}")

    prompt = data.get('prompt', '')
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Missing prompt in {file_path}")

    # Deduplicate by prompt
    if prompt in seen_prompts:
        return None

    seen_prompts.add(prompt)

    # Extract trajectories
    video_trajectory = data.get('video_trajectory')
    audio_trajectory = data.get('audio_trajectory')
    sigmas = data.get('sigmas')  # [T] actual sigma values per trajectory entry

    if video_trajectory is None:
        raise ValueError(f"Missing video_trajectory in {file_path}")
    if not torch.is_tensor(video_trajectory) or video_trajectory.ndim != 6:
        raise ValueError(
            f"Invalid video_trajectory shape in {file_path}: "
            f"{getattr(video_trajectory, 'shape', None)}"
        )
    if video_trajectory.shape[0] != 1:
        raise ValueError(f"Expected one trajectory per file in {file_path}")
    if audio_trajectory is not None:
        if not torch.is_tensor(audio_trajectory) or audio_trajectory.ndim != 4:
            raise ValueError(
                f"Invalid audio_trajectory shape in {file_path}: "
                f"{getattr(audio_trajectory, 'shape', None)}"
            )
        if audio_trajectory.shape[:2] != video_trajectory.shape[:2]:
            raise ValueError(f"Audio/video trajectory length mismatch in {file_path}")
    if sigmas is None or not torch.is_tensor(sigmas) or sigmas.ndim != 1:
        raise ValueError(f"Missing or invalid sigma trajectory in {file_path}")
    if sigmas.shape[0] != video_trajectory.shape[1]:
        raise ValueError(f"Sigma/video trajectory length mismatch in {file_path}")

    # Convert to numpy float16 for storage efficiency
    video_latents = video_trajectory.half().numpy()
    audio_latents = audio_trajectory.half().numpy() if audio_trajectory is not None else None
    sigmas_np = sigmas.float().numpy() if sigmas is not None else None

    return {
        'video_latents': video_latents,
        'audio_latents': audio_latents,
        'prompts': [prompt],
        'sigmas': sigmas_np,
    }


def create_lmdb_from_trajectories(
    data_path: str,
    lmdb_path: str,
    map_size: int = 5_000_000_000_000,  # 5TB default
    require_manifest: bool = False,
) -> None:
    """
    Create LMDB database from trajectory .pt files.

    Args:
        data_path: Directory containing .pt trajectory files
        lmdb_path: Output LMDB path
        map_size: Maximum database size in bytes
    """
    # Find all .pt files
    all_files = sorted(glob.glob(os.path.join(data_path, "*.pt")))
    print(f"Found {len(all_files)} trajectory files")

    if len(all_files) == 0:
        raise ValueError(f"No .pt files found in {data_path}")

    if os.path.exists(lmdb_path):
        raise FileExistsError(f"Refusing to overwrite existing LMDB: {lmdb_path}")

    manifest_path = os.path.join(data_path, "manifest.json")
    manifest = None
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("format_version") != ODE_FORMAT_VERSION:
            raise ValueError(f"Unsupported ODE manifest version: {manifest_path}")
        if manifest.get("producer") != ODE_PRODUCER:
            raise ValueError(f"Unsupported ODE manifest producer: {manifest_path}")
    elif require_manifest:
        raise ValueError(f"Required ODE manifest not found: {manifest_path}")

    if require_manifest:
        validation_seen: Set[str] = set()
        expected_shapes = None
        for file_path in tqdm(all_files, desc="Validating trajectory provenance"):
            data_dict = process_trajectory_file(file_path, validation_seen, manifest)
            if data_dict is None:
                raise ValueError(f"Duplicate prompt found while validating {file_path}")
            shape_signature = (
                tuple(data_dict['video_latents'].shape),
                tuple(data_dict['audio_latents'].shape) if data_dict['audio_latents'] is not None else None,
                tuple(data_dict['sigmas'].shape),
            )
            if expected_shapes is None:
                expected_shapes = shape_signature
            elif shape_signature != expected_shapes:
                raise ValueError(
                    f"Trajectory shape mismatch in {file_path}: "
                    f"{shape_signature} != {expected_shapes}"
                )

    # Create LMDB
    os.makedirs(os.path.dirname(lmdb_path) if os.path.dirname(lmdb_path) else '.', exist_ok=True)
    env = lmdb.open(lmdb_path, map_size=map_size)

    counter = 0
    seen_prompts: Set[str] = set()
    last_video_shape = None
    last_audio_shape = None
    last_sigmas_shape = None

    for file_path in tqdm(all_files, desc="Processing trajectory files"):
        try:
            data_dict = process_trajectory_file(file_path, seen_prompts, manifest)
        except Exception as e:
            if require_manifest:
                env.close()
                raise RuntimeError(f"Failed to convert required trajectory {file_path}") from e
            print(f"Error processing {file_path}: {e}")
            continue

        if data_dict is None:
            # Duplicate prompt, skip
            continue

        # Store arrays
        video_latents = data_dict['video_latents']
        audio_latents = data_dict['audio_latents']
        prompts = data_dict['prompts']
        sigmas = data_dict['sigmas']  # [T] float32 actual sigma values

        with env.begin(write=True) as txn:
            for i, prompt in enumerate(prompts):
                # Store video latents
                video_key = f'video_latents_{counter}_data'.encode()
                txn.put(video_key, video_latents[i].tobytes())

                # Store audio latents (if available)
                if audio_latents is not None:
                    audio_key = f'audio_latents_{counter}_data'.encode()
                    txn.put(audio_key, audio_latents[i].tobytes())

                # Store sigma values (float32, same for all prompts in this file)
                if sigmas is not None:
                    sigma_key = f'sigmas_{counter}_data'.encode()
                    txn.put(sigma_key, sigmas.tobytes())

                # Store prompt
                prompt_key = f'prompts_{counter}_data'.encode()
                txn.put(prompt_key, prompt.encode('utf-8'))

                counter += 1

        last_video_shape = video_latents.shape
        last_audio_shape = audio_latents.shape if audio_latents is not None else None
        last_sigmas_shape = sigmas.shape if sigmas is not None else None

    if counter == 0:
        env.close()
        raise RuntimeError(f"No valid trajectories were written from {data_path}")

    # Save shapes to LMDB
    if last_video_shape is not None:
        with env.begin(write=True) as txn:
            # Video shape: [B, T, F, C, H, W] -> store [T, F, C, H, W] per entry
            video_entry_shape = list(last_video_shape[1:])  # Remove batch dim
            video_entry_shape = [counter] + video_entry_shape  # Add total count
            shape_str = " ".join(map(str, video_entry_shape))
            txn.put("video_latents_shape".encode(), shape_str.encode())

            # Audio shape: [B, T, F_a, C] -> store [T, F_a, C] per entry
            if last_audio_shape is not None:
                audio_entry_shape = list(last_audio_shape[1:])
                audio_entry_shape = [counter] + audio_entry_shape
                shape_str = " ".join(map(str, audio_entry_shape))
                txn.put("audio_latents_shape".encode(), shape_str.encode())

            # Sigmas shape: [T] per entry (same T as video trajectory dim)
            if last_sigmas_shape is not None:
                sigmas_entry_shape = [counter] + list(last_sigmas_shape)
                shape_str = " ".join(map(str, sigmas_entry_shape))
                txn.put("sigmas_shape".encode(), shape_str.encode())

            # Prompts shape
            txn.put("prompts_shape".encode(), f"{counter}".encode())
            if manifest is not None:
                txn.put(
                    "manifest_json".encode(),
                    json.dumps(manifest, sort_keys=True).encode("utf-8"),
                )

    env.close()
    print(f"Created LMDB at {lmdb_path} with {counter} entries")


def main():
    """Command line interface for LMDB creation."""
    parser = argparse.ArgumentParser(
        description="Convert ODE trajectory .pt files to LMDB format"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to directory containing .pt trajectory files"
    )
    parser.add_argument(
        "--lmdb_path",
        type=str,
        required=True,
        help="Output LMDB database path"
    )
    parser.add_argument(
        "--map_size",
        type=int,
        default=5_000_000_000_000,
        help="Maximum LMDB size in bytes (default: 5TB)"
    )
    parser.add_argument(
        "--require_manifest",
        action="store_true",
        help="Require versioned LTX-2.3 ODE provenance metadata",
    )

    args = parser.parse_args()

    create_lmdb_from_trajectories(
        data_path=args.data_path,
        lmdb_path=args.lmdb_path,
        map_size=args.map_size,
        require_manifest=args.require_manifest,
    )


if __name__ == "__main__":
    main()
