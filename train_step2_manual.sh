#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${OMNIFORCING_PROJECT_ROOT:-${SCRIPT_DIR}}"
cd "$PROJECT_ROOT"

DISTILLATION_ROOT="$PROJECT_ROOT/LTX-2/packages/ltx-distillation"
ENV_ARCHIVE="${OMNIFORCING_ENV_ARCHIVE:-/data/minghua/zzy/omniforcing-conda.tar.zst}"
ENV_DIR="${OMNIFORCING_ENV_DIR:-/tmp/omniforcing-conda}"
ENV_MARKER="$ENV_DIR/.omniforcing_ready"
CONFIG_PATH="${STEP2_CONFIG_PATH:-$DISTILLATION_ROOT/configs/stage2_causal_ode.yaml}"
MODEL_PATH="$PROJECT_ROOT/checkpoints/LTX-2.3-fp8/ltx-2.3-22b-dev-fp8.safetensors"
GEMMA_PATH="$PROJECT_ROOT/checkpoints/gemma-3-12b-it-qat-q4_0-unquantized"
PAIR_DIR="$DISTILLATION_ROOT/ode_pairs"
LMDB_DIR="$DISTILLATION_ROOT/ode_lmdb"
HF_REPO_ID="${HF_ODE_REPO_ID:-aaachier/OmniForcing-backup}"
HF_REPO_TYPE="${HF_ODE_REPO_TYPE:-model}"
EXPECTED_PAIR_COUNT="${STEP1_EXPECTED_PAIR_COUNT:-8}"
DOWNLOAD_STEP1_DATA="${DOWNLOAD_STEP1_DATA:-1}"
STEP2_PREPARE_ONLY="${STEP2_PREPARE_ONLY:-0}"
STEP1_INVALID_BACKUP_ROOT="${STEP1_INVALID_BACKUP_ROOT:-${PROJECT_ROOT}.step1-invalid}"

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29510}"
NUM_CPUS="${NUM_CPUS:-128}"

[[ -s "$CONFIG_PATH" ]] || {
    echo "Missing Step 2 config: $CONFIG_PATH" >&2
    exit 1
}
CONFIG_PATH="$(realpath "$CONFIG_PATH")"
[[ -s "$MODEL_PATH" ]] || {
    echo "Missing LTX-2.3 FP8 checkpoint: $MODEL_PATH" >&2
    exit 1
}
[[ -d "$GEMMA_PATH" ]] || {
    echo "Missing Gemma checkpoint: $GEMMA_PATH" >&2
    exit 1
}

if [[ -z "${HF_TOKEN:-}" && -s "$PROJECT_ROOT/.hf_token" ]]; then
    export HF_TOKEN="$(tr -d '\r\n' < "$PROJECT_ROOT/.hf_token")"
fi
if [[ -z "${WANDB_API_KEY:-}" && -s "$PROJECT_ROOT/.wandb_key" ]]; then
    export WANDB_API_KEY="$(tr -d '\r\n' < "$PROJECT_ROOT/.wandb_key")"
fi
[[ -n "${HF_TOKEN:-}" ]] || {
    echo "Missing Hugging Face token: set HF_TOKEN or create $PROJECT_ROOT/.hf_token" >&2
    exit 1
}
[[ -n "${WANDB_API_KEY:-}" ]] || {
    echo "Missing WandB key: set WANDB_API_KEY or create $PROJECT_ROOT/.wandb_key" >&2
    exit 1
}

if [[ ! -f "$ENV_MARKER" || ! -x "$ENV_DIR/bin/python" ]]; then
    [[ -s "$ENV_ARCHIVE" ]] || {
        echo "Missing conda environment archive: $ENV_ARCHIVE" >&2
        echo "Set OMNIFORCING_ENV_ARCHIVE to the archive on this machine." >&2
        exit 1
    }
    mkdir -p "$ENV_DIR"
    echo "Preparing conda environment in $ENV_DIR ..."
    tar --zstd -xf "$ENV_ARCHIVE" -C "$ENV_DIR"
    "$ENV_DIR/bin/python" "$ENV_DIR/bin/conda-unpack"
    touch "$ENV_MARKER"
fi

set +u
source "$ENV_DIR/bin/activate"
set -u

export PYTHONPATH="$DISTILLATION_ROOT/src:$PROJECT_ROOT/LTX-2/packages/ltx-causal/src:$PROJECT_ROOT/LTX-2/packages/ltx-core/src:$PROJECT_ROOT/LTX-2/packages/ltx-pipelines/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export WANDB_MODE="${WANDB_MODE:-online}"
export NNODES
export NODE_RANK
export MASTER_ADDR
export MASTER_PORT
export NUM_CPUS
export PYTHONUNBUFFERED=1

validate_step1_artifacts() {
    local pair_dir="$1"
    local lmdb_dir="$2"

    python - "$pair_dir" "$lmdb_dir" "$EXPECTED_PAIR_COUNT" "$MODEL_PATH" <<'PY'
import json
import os
import sys
from pathlib import Path

import torch

from ltx_distillation.ode.data import ODERegressionLMDBDataset


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


pair_dir = Path(sys.argv[1])
lmdb_dir = Path(sys.argv[2])
expected_count = int(sys.argv[3])
base_checkpoint = Path(sys.argv[4])
expected_schedule = [1000, 909, 725, 421, 0]
expected_producer = "omniforcing-ltx23-full-architecture-v2"
dataset = None

try:
    require(pair_dir.is_dir(), f"missing ode_pairs directory: {pair_dir}")
    require(lmdb_dir.is_dir(), f"missing ode_lmdb directory: {lmdb_dir}")
    require((lmdb_dir / "data.mdb").stat().st_size > 0, "missing or empty data.mdb")
    require((lmdb_dir / "lock.mdb").exists(), "missing lock.mdb")

    expected_names = [f"{index:06d}.pt" for index in range(expected_count)]
    actual_names = sorted(path.name for path in pair_dir.glob("*.pt"))
    require(
        actual_names == expected_names,
        f"ODE pair files differ: {actual_names} != {expected_names}",
    )

    manifest_path = pair_dir / "manifest.json"
    require(manifest_path.is_file(), f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("format_version") == 4, "ODE format_version must be 4")
    require(
        manifest.get("producer") == expected_producer,
        f"unexpected ODE producer: {manifest.get('producer')}",
    )

    generation = manifest.get("generation_config")
    require(isinstance(generation, dict), "manifest generation_config is missing")
    expected_generation = {
        "denoising_step_list": expected_schedule,
        "num_frames": 121,
        "video_height": 512,
        "video_width": 768,
    }
    for key, expected in expected_generation.items():
        require(
            generation.get(key) == expected,
            f"manifest {key} mismatch: {generation.get(key)} != {expected}",
        )

    expected_teacher = base_checkpoint.name
    actual_teacher = Path(str(manifest.get("teacher_checkpoint", ""))).name
    require(
        actual_teacher == expected_teacher,
        f"teacher checkpoint mismatch: {actual_teacher} != {expected_teacher}",
    )

    pair_prompts = []
    expected_video_shape = (
        1,
        len(expected_schedule),
        1 + (121 - 1) // 8,
        128,
        512 // 32,
        768 // 32,
    )
    for name in expected_names:
        path = pair_dir / name
        payload = torch.load(path, map_location="cpu", weights_only=False)
        require(payload.get("format_version") == 4, f"{name}: wrong format_version")
        require(payload.get("producer") == expected_producer, f"{name}: wrong producer")
        require(payload.get("generation_config") == generation, f"{name}: generation_config mismatch")
        require(
            Path(str(payload.get("teacher_checkpoint", ""))).name == expected_teacher,
            f"{name}: teacher checkpoint mismatch",
        )
        prompt = payload.get("prompt")
        require(isinstance(prompt, str) and prompt.strip(), f"{name}: prompt is missing")
        pair_prompts.append(prompt)

        video = payload.get("video_trajectory")
        audio = payload.get("audio_trajectory")
        sigmas = payload.get("sigmas")
        require(torch.is_tensor(video), f"{name}: video_trajectory is missing")
        require(tuple(video.shape) == expected_video_shape, f"{name}: video shape {tuple(video.shape)}")
        require(torch.is_tensor(audio) and audio.ndim == 4, f"{name}: audio_trajectory is invalid")
        require(tuple(audio.shape[:2]) == (1, len(expected_schedule)), f"{name}: audio shape {tuple(audio.shape)}")
        require(
            torch.is_tensor(sigmas) and tuple(sigmas.shape) == (len(expected_schedule),),
            f"{name}: sigma trajectory is invalid",
        )

    require(len(set(pair_prompts)) == expected_count, "ODE pair prompts are not unique")

    dataset = ODERegressionLMDBDataset(str(lmdb_dir), load_audio=True)
    require(len(dataset) == expected_count, f"LMDB sample count {len(dataset)} != {expected_count}")
    require(dataset.manifest == manifest, "LMDB manifest does not match ode_pairs manifest")
    require(dataset.has_audio, "LMDB has no audio trajectories")
    require(dataset.has_sigmas, "LMDB has no sigma trajectories")
    require(tuple(dataset.video_shape) == (expected_count, *expected_video_shape[1:]), f"LMDB video shape {dataset.video_shape}")
    require(len(dataset.audio_shape) == 4, f"LMDB audio shape {dataset.audio_shape}")
    require(tuple(dataset.audio_shape[:2]) == (expected_count, len(expected_schedule)), f"LMDB audio shape {dataset.audio_shape}")
    require(tuple(dataset.sigmas_shape) == (expected_count, len(expected_schedule)), f"LMDB sigma shape {dataset.sigmas_shape}")
    require(dataset.get_prompts(expected_count) == pair_prompts, "LMDB prompts do not match ode_pairs")

    sigma_values = dataset.get_sigmas(0)
    require(torch.isfinite(sigma_values).all().item(), "LMDB sigma row contains NaN or Inf")
    require(sigma_values[-1].item() == 0.0, "LMDB sigma row does not end at zero")
    require(torch.all(sigma_values[:-1] > sigma_values[1:]).item(), "LMDB sigmas are not decreasing")

    print(
        "[STEP1_VALIDATE] valid: "
        f"pairs={expected_count}, video_shape={tuple(dataset.video_shape)}, "
        f"audio_shape={tuple(dataset.audio_shape)}, sigmas={sigma_values.tolist()}",
        flush=True,
    )
except Exception as exc:
    print(
        f"[STEP1_VALIDATE] invalid: {type(exc).__name__}: {exc}",
        file=sys.stderr,
        flush=True,
    )
    raise SystemExit(1)
finally:
    if dataset is not None:
        dataset.close()
PY
}

ensure_step1_data() {
    if validate_step1_artifacts "$PAIR_DIR" "$LMDB_DIR"; then
        echo "Step 1 artifacts are valid and ready locally."
        return
    fi

    if [[ "$DOWNLOAD_STEP1_DATA" != "1" ]]; then
        echo "Step 1 artifacts are missing, incomplete, or from the wrong run." >&2
        echo "Set DOWNLOAD_STEP1_DATA=1 to download them from $HF_REPO_ID." >&2
        exit 1
    fi

    local backup_dir download_dir
    download_dir="$(mktemp -d "$DISTILLATION_ROOT/.step1-download.XXXXXX")"
    echo "Downloading ode_pairs/ and ode_lmdb/ from $HF_REPO_ID ..."
    if ! python - "$download_dir" "$HF_REPO_ID" "$HF_REPO_TYPE" <<'PY'
import os
import sys

from huggingface_hub import snapshot_download

staging_dir, repo_id, repo_type = sys.argv[1:]
snapshot_download(
    repo_id=repo_id,
    repo_type=repo_type,
    revision=os.environ.get("HF_ODE_REVISION", "main"),
    token=os.environ["HF_TOKEN"],
    local_dir=staging_dir,
    allow_patterns=["ode_pairs/*", "ode_lmdb/*"],
)
PY
    then
        echo "Failed to download Step 1 artifacts; existing local data was left untouched." >&2
        rm -rf -- "$download_dir"
        exit 1
    fi

    if ! validate_step1_artifacts "$download_dir/ode_pairs" "$download_dir/ode_lmdb"; then
        echo "Downloaded Step 1 artifacts failed validation; existing local data was left untouched." >&2
        rm -rf -- "$download_dir"
        exit 1
    fi

    if [[ -e "$PAIR_DIR" || -L "$PAIR_DIR" || -e "$LMDB_DIR" || -L "$LMDB_DIR" ]]; then
        backup_dir="$STEP1_INVALID_BACKUP_ROOT/$(date -u +%Y%m%dT%H%M%SZ)-$$"
        mkdir -p "$backup_dir"
        if [[ -e "$PAIR_DIR" || -L "$PAIR_DIR" ]]; then
            mv "$PAIR_DIR" "$backup_dir/ode_pairs"
        fi
        if [[ -e "$LMDB_DIR" || -L "$LMDB_DIR" ]]; then
            mv "$LMDB_DIR" "$backup_dir/ode_lmdb"
        fi
        echo "Moved old or invalid Step 1 artifacts to: $backup_dir"
    fi

    mv "$download_dir/ode_pairs" "$PAIR_DIR"
    mv "$download_dir/ode_lmdb" "$LMDB_DIR"
    rm -rf -- "$download_dir"
    validate_step1_artifacts "$PAIR_DIR" "$LMDB_DIR"
    echo "Correct Step 1 artifacts are ready: $PAIR_DIR and $LMDB_DIR"
}

ensure_step1_data

if [[ "${STEP2_PREPARE_ONLY}" == "1" ]]; then
    echo "Step 2 preparation complete; STEP2_PREPARE_ONLY=1, training was not started."
    exit 0
fi

if [[ "$NNODES" != "1" || "$NODE_RANK" != "0" ]]; then
    echo "This Step 2 entrypoint supports one node only; got NNODES=$NNODES, NODE_RANK=$NODE_RANK." >&2
    exit 1
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    if [[ -n "${STEP2_GPU_IDS:-}" ]]; then
        export CUDA_VISIBLE_DEVICES="$STEP2_GPU_IDS"
    else
        if ! command -v nvidia-smi >/dev/null 2>&1; then
            echo "nvidia-smi is unavailable; set CUDA_VISIBLE_DEVICES or STEP2_GPU_IDS explicitly." >&2
            exit 1
        fi
        physical_gpu_count="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
        if [[ "$physical_gpu_count" -ge 8 ]]; then
            export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
        elif [[ "$physical_gpu_count" -eq 4 ]]; then
            export CUDA_VISIBLE_DEVICES="0,1,2,3"
        else
            echo "Step 2 requires exactly 4 or 8 visible GPUs; detected $physical_gpu_count physical GPUs." >&2
            echo "Set CUDA_VISIBLE_DEVICES or STEP2_GPU_IDS to expose exactly 4 or 8 GPUs." >&2
            exit 1
        fi
    fi
fi

VISIBLE_GPU_COUNT="$(python -c 'import torch; print(torch.cuda.device_count())')"
if [[ "$VISIBLE_GPU_COUNT" != "4" && "$VISIBLE_GPU_COUNT" != "8" ]]; then
    echo "Step 2 requires exactly 4 or 8 visible CUDA devices; found $VISIBLE_GPU_COUNT." >&2
    echo "Current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
    exit 1
fi

NUM_GPUS="$VISIBLE_GPU_COUNT"
NPROC_PER_NODE="$VISIBLE_GPU_COUNT"
export NUM_GPUS
export NPROC_PER_NODE

TRAIN_CONFIG_PATH="$CONFIG_PATH"
RUNTIME_CONFIG_PATH=""
cleanup_runtime_config() {
    if [[ -n "$RUNTIME_CONFIG_PATH" && -f "$RUNTIME_CONFIG_PATH" ]]; then
        rm -f -- "$RUNTIME_CONFIG_PATH"
    fi
}
trap cleanup_runtime_config EXIT

if [[ "$NUM_GPUS" == "4" ]]; then
    RUNTIME_CONFIG_PATH="$(mktemp "${TMPDIR:-/tmp}/omniforcing-step2-4gpu.XXXXXX.yaml")"
    python - "$CONFIG_PATH" "$RUNTIME_CONFIG_PATH" <<'PY'
import sys

from omegaconf import OmegaConf

source_path, runtime_path = sys.argv[1:]
config = OmegaConf.load(source_path)
config.expected_world_size = 4
OmegaConf.save(config=config, f=runtime_path)
PY
    TRAIN_CONFIG_PATH="$RUNTIME_CONFIG_PATH"
fi

echo "========================================================"
echo "Step 2: ${NUM_GPUS}-GPU causal ODE regression"
echo "========================================================"
echo "Project:       $PROJECT_ROOT"
echo "Base config:   $CONFIG_PATH"
echo "Runtime config: $TRAIN_CONFIG_PATH"
echo "Visible GPUs:  $CUDA_VISIBLE_DEVICES"
echo "GPU processes: $NPROC_PER_NODE"
echo "Step 1 LMDB:   $LMDB_DIR"
echo "WandB:         $WANDB_MODE"
echo "HF repository: $HF_REPO_ID"
echo "========================================================"

EXTRA_ARGS=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
    [[ -f "$RESUME_CHECKPOINT" ]] || {
        echo "Missing resume model: $RESUME_CHECKPOINT" >&2
        exit 1
    }
    [[ "$(basename "$RESUME_CHECKPOINT")" == "model.pt" ]] || {
        echo "RESUME_CHECKPOINT must point to a saved model.pt file." >&2
        exit 1
    }
    RESUME_CHECKPOINT="$(realpath "$RESUME_CHECKPOINT")"
    EXTRA_ARGS+=(--resume_checkpoint "$RESUME_CHECKPOINT")
fi

echo "Starting Step 2 with $NUM_GPUS GPUs on one node."
bash \
    "$PROJECT_ROOT/LTX-2/packages/ltx-distillation/scripts/train_stage2_causal_ode.sh" \
    "$TRAIN_CONFIG_PATH" \
    "${EXTRA_ARGS[@]}" \
    "$@"
