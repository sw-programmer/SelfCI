#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF_USAGE'
Usage:
  ./run_grpo.sh [hf_repo_or_local_path] [extra args...]

Arguments:
  hf_repo_or_local_path   Optional. Hugging Face model repo id or local path.
                          Default: Qwen/Qwen2.5-7B-Instruct
  extra args              Optional. Forwarded directly to rl/grpo_train.py.

Environment variables:
  GPU_ID                        CUDA device id to use (default: 0)
  OUTPUT_DIR                    Override output directory
  RUN_NAME                      Override run name
  EPOCHS                        Number of epochs (default: 30)
  BATCH_SIZE                    Per-device train batch size (default: 16)
  EVAL_BATCH_SIZE               Per-device eval batch size (default: 15)
  NUM_GENERATIONS               Number of sampled completions per prompt (default: 16)
  MINI_BATCH_SIZE               Effective GRPO mini-batch size (default: 128)
  LEARNING_RATE                 Learning rate (default: 1e-6)
  BETA                          KL beta (default: 1e-3)
  CLIP_RATIO                    GRPO clipping epsilon (default: 0.2)
  ENTROPY_PENALTY               Entropy regularization coefficient (default: 0.0)
  TEMPERATURE                   Sampling temperature (default: 0.7)
  VLLM_GPU_MEMORY_UTILIZATION   vLLM colocate GPU memory fraction (default: 0.3)
  TRUST_REMOTE_CODE             Set to true to pass --trust-remote-code (default: true)

Example:
  ./run_grpo.sh
  ./run_grpo.sh meta-llama/Llama-3.1-8B-Instruct
  GPU_ID=0 ./run_grpo.sh /path/to/model --epochs 2
EOF_USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

DEFAULT_MODEL="Qwen/Qwen2.5-7B-Instruct"
HF_MODEL="${DEFAULT_MODEL}"

if [[ "${1:-}" != "" && "${1:-}" != --* ]]; then
  HF_MODEL="$1"
  shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_PATH="huseyinatahaninan/ContextualIntegritySyntheticDataset"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-15}"
NUM_GENERATIONS="${NUM_GENERATIONS:-16}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-128}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
BETA="${BETA:-1e-3}"
CLIP_RATIO="${CLIP_RATIO:-0.2}"
ENTROPY_PENALTY="${ENTROPY_PENALTY:-0.0}"
TEMPERATURE="${TEMPERATURE:-0.7}"
EPOCHS="${EPOCHS:-30}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.3}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"
RUN_NAME_OVERRIDE="${RUN_NAME:-}"
OUTPUT_DIR_OVERRIDE="${OUTPUT_DIR:-}"

# Reflect CLI overrides in the resolved launch metadata.
for ((i=1; i<=$#; i++)); do
  arg="${!i}"
  if [[ "${arg}" == "--epochs" ]]; then
    j=$((i + 1))
    if [[ ${j} -le $# ]]; then
      EPOCHS="${!j}"
    fi
  elif [[ "${arg}" == --epochs=* ]]; then
    EPOCHS="${arg#*=}"
  elif [[ "${arg}" == "--eval-batch-size" ]]; then
    j=$((i + 1))
    if [[ ${j} -le $# ]]; then
      EVAL_BATCH_SIZE="${!j}"
    fi
  elif [[ "${arg}" == --eval-batch-size=* ]]; then
    EVAL_BATCH_SIZE="${arg#*=}"
  elif [[ "${arg}" == "--batch-size" ]]; then
    j=$((i + 1))
    if [[ ${j} -le $# ]]; then
      BATCH_SIZE="${!j}"
    fi
  elif [[ "${arg}" == --batch-size=* ]]; then
    BATCH_SIZE="${arg#*=}"
  elif [[ "${arg}" == "--num-generations" ]]; then
    j=$((i + 1))
    if [[ ${j} -le $# ]]; then
      NUM_GENERATIONS="${!j}"
    fi
  elif [[ "${arg}" == --num-generations=* ]]; then
    NUM_GENERATIONS="${arg#*=}"
  elif [[ "${arg}" == "--mini-batch-size" ]]; then
    j=$((i + 1))
    if [[ ${j} -le $# ]]; then
      MINI_BATCH_SIZE="${!j}"
    fi
  elif [[ "${arg}" == --mini-batch-size=* ]]; then
    MINI_BATCH_SIZE="${arg#*=}"
  elif [[ "${arg}" == "--learning-rate" ]]; then
    j=$((i + 1))
    if [[ ${j} -le $# ]]; then
      LEARNING_RATE="${!j}"
    fi
  elif [[ "${arg}" == --learning-rate=* ]]; then
    LEARNING_RATE="${arg#*=}"
  elif [[ "${arg}" == "--beta" ]]; then
    j=$((i + 1))
    if [[ ${j} -le $# ]]; then
      BETA="${!j}"
    fi
  elif [[ "${arg}" == --beta=* ]]; then
    BETA="${arg#*=}"
  elif [[ "${arg}" == "--clip-ratio" ]]; then
    j=$((i + 1))
    if [[ ${j} -le $# ]]; then
      CLIP_RATIO="${!j}"
    fi
  elif [[ "${arg}" == --clip-ratio=* ]]; then
    CLIP_RATIO="${arg#*=}"
  elif [[ "${arg}" == "--entropy-penalty" ]]; then
    j=$((i + 1))
    if [[ ${j} -le $# ]]; then
      ENTROPY_PENALTY="${!j}"
    fi
  elif [[ "${arg}" == --entropy-penalty=* ]]; then
    ENTROPY_PENALTY="${arg#*=}"
  elif [[ "${arg}" == "--temperature" ]]; then
    j=$((i + 1))
    if [[ ${j} -le $# ]]; then
      TEMPERATURE="${!j}"
    fi
  elif [[ "${arg}" == --temperature=* ]]; then
    TEMPERATURE="${arg#*=}"
  elif [[ "${arg}" == "--run-name" ]]; then
    j=$((i + 1))
    if [[ ${j} -le $# ]]; then
      RUN_NAME_OVERRIDE="${!j}"
    fi
  elif [[ "${arg}" == --run-name=* ]]; then
    RUN_NAME_OVERRIDE="${arg#*=}"
  elif [[ "${arg}" == "--output-dir" ]]; then
    j=$((i + 1))
    if [[ ${j} -le $# ]]; then
      OUTPUT_DIR_OVERRIDE="${!j}"
    fi
  elif [[ "${arg}" == --output-dir=* ]]; then
    OUTPUT_DIR_OVERRIDE="${arg#*=}"
  fi
done

MODEL_TAG="${HF_MODEL##*/}"
LR_TAG="${LEARNING_RATE//./p}"
BETA_TAG="${BETA//./p}"
CLIP_TAG="${CLIP_RATIO//./p}"
ENTROPY_TAG="${ENTROPY_PENALTY//./p}"
TEMP_TAG="${TEMPERATURE//./p}"

DEFAULT_RUN_NAME="grpo_${MODEL_TAG}_bs${BATCH_SIZE}_g${NUM_GENERATIONS}_mb${MINI_BATCH_SIZE}"
DEFAULT_RUN_NAME+="_lr${LR_TAG}_b${BETA_TAG}_clip${CLIP_TAG}"
if [[ "${ENTROPY_PENALTY}" != "0" && "${ENTROPY_PENALTY}" != "0.0" ]]; then
  DEFAULT_RUN_NAME+="_ent${ENTROPY_TAG}"
fi
DEFAULT_RUN_NAME+="_temp${TEMP_TAG}_ep${EPOCHS}"

RUN_NAME="${RUN_NAME_OVERRIDE:-$DEFAULT_RUN_NAME}"
OUTPUT_DIR="${OUTPUT_DIR_OVERRIDE:-./output/${RUN_NAME}}"

CMD=(
  accelerate launch "${SCRIPT_DIR}/rl/run_grpo.py"
  --model-name-or-path "${HF_MODEL}"
  --output-dir "${OUTPUT_DIR}"
  --run-name "${RUN_NAME}"
  --batch-size "${BATCH_SIZE}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --num-generations "${NUM_GENERATIONS}"
  --mini-batch-size "${MINI_BATCH_SIZE}"
  --learning-rate "${LEARNING_RATE}"
  --beta "${BETA}"
  --clip-ratio "${CLIP_RATIO}"
  --entropy-penalty "${ENTROPY_PENALTY}"
  --temperature "${TEMPERATURE}"
  --epochs "${EPOCHS}"
  --vllm-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
)

if [[ "${TRUST_REMOTE_CODE}" == "true" ]]; then
  CMD+=(--trust-remote-code)
fi

# Forward any additional args directly to run_grpo.py.
if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

echo "Launching GRPO training..."
echo "  GPU            : ${GPU_ID}"
echo "  Model          : ${HF_MODEL}"
echo "  Dataset        : ${DATASET_PATH} (fixed splits: train/eval/test)"
echo "  Batch size     : ${BATCH_SIZE}"
echo "  Eval batch     : ${EVAL_BATCH_SIZE}"
echo "  Generations    : ${NUM_GENERATIONS}"
echo "  Mini-batch     : ${MINI_BATCH_SIZE}"
echo "  Learning rate  : ${LEARNING_RATE}"
echo "  Beta           : ${BETA}"
echo "  Clip ratio     : ${CLIP_RATIO}"
echo "  Entropy        : ${ENTROPY_PENALTY}"
echo "  Temperature    : ${TEMPERATURE}"
echo "  Epochs         : ${EPOCHS}"
echo "  Output         : ${OUTPUT_DIR}"
echo "  Run name       : ${RUN_NAME}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CMD[@]}"
