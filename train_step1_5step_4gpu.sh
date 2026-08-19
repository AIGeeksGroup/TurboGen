#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISTILLATION_ROOT="${PROJECT_ROOT}/LTX-2/packages/ltx-distillation"
BASE_CONFIG="${DISTILLATION_ROOT}/configs/stage1_bidirectional_dmd.yaml"
RUNTIME_CONFIG="$(mktemp "${TMPDIR:-/tmp}/omniforcing-step1-5step.XXXXXX.yaml")"
trap 'rm -f "${RUNTIME_CONFIG}"' EXIT
export NUM_GPUS="${NUM_GPUS:-4}" NNODES="${NNODES:-1}" NUM_CPUS="${NUM_CPUS:-64}"
export WANDB_MODE="${WANDB_MODE:-online}" WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}" PYTHONUNBUFFERED=1
[[ "${NUM_GPUS}" == 4 && "${NNODES}" == 1 ]] || { echo 'Require NUM_GPUS=4 and NNODES=1' >&2; exit 1; }
[[ -s "${PROJECT_ROOT}/checkpoints/LTX-2.3-fp8/ltx-2.3-22b-dev-fp8.safetensors" ]] || { echo 'Missing teacher checkpoint' >&2; exit 1; }
[[ -d "${PROJECT_ROOT}/checkpoints/gemma-3-12b-it-qat-q4_0-unquantized" ]] || { echo 'Missing Gemma checkpoint' >&2; exit 1; }
[[ -s "${PROJECT_ROOT}/.hf_token" || -n "${HF_TOKEN:-}" ]] || { echo 'Missing HF_TOKEN or .hf_token' >&2; exit 1; }
[[ -s "${PROJECT_ROOT}/.wandb_key" || -n "${WANDB_API_KEY:-}" ]] || { echo 'Missing WANDB_API_KEY or .wandb_key' >&2; exit 1; }
python - "${BASE_CONFIG}" "${RUNTIME_CONFIG}" "${PROJECT_ROOT}" <<'PY'
import sys
from pathlib import Path
from omegaconf import OmegaConf
base, out, root = map(Path, sys.argv[1:])
cfg = OmegaConf.load(base)
cfg.checkpoint_path = str((root/'checkpoints/LTX-2.3-fp8/ltx-2.3-22b-dev-fp8.safetensors').resolve())
cfg.gemma_path = str((root/'checkpoints/gemma-3-12b-it-qat-q4_0-unquantized').resolve())
cfg.data_path = str((root/'LTX-2/packages/ltx-distillation/prompts/benchmark_512.txt').resolve())
cfg.benchmark_prompt_file = str((root/'LTX-2/packages/ltx-distillation/prompts/benchmark_8.txt').resolve())
cfg.wandb_api_key_file = str((root/'.wandb_key').resolve())
cfg.hf_upload_token_file = str((root/'.hf_token').resolve())
cfg.output_path = str((root/'outputs/stage1_bidirectional_dmd_5step').resolve())
cfg.wandb_name = 'stage1_bidirectional_dmd_5step_4gpu'
cfg.expected_world_size = 4
cfg.save_iters = cfg.benchmark_iters = 500
cfg.wandb_init_timeout = 300
cfg.denoising_step_list = [1000, 900, 750, 500, 250, 0]
OmegaConf.save(cfg, out)
PY
echo "Starting Step 1: 5-step bidirectional DMD, 4 GPUs"
cd "${DISTILLATION_ROOT}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" bash scripts/train_stage1_bidirectional_dmd.sh "${RUNTIME_CONFIG}"
