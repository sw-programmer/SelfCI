#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF_USAGE'
Usage:
  ./run_feedback_weighted_asymmetric_bidirectional_kl.sh [student_model] [teacher_model] [extra args...]

Arguments:
  student_model      Optional. Student model repo id or local path.
                     Default: Qwen/Qwen2.5-7B-Instruct
  teacher_model      Optional. Teacher model repo id or local path.
                     Default: Qwen/Qwen2.5-7B-Instruct
  extra args         Optional. Forwarded directly to distill/feedback_weighted_asymmetric_bidirectional_kl_train.py.

Environment variables:
  GPU_ID                        CUDA device id to use (default: 0)
  DATASET_PATH                  Dataset repo id/path
                                (default: Sangsang/ContextualIntegritySyntheticDataset_Qwen2.5-7B-Instruct_merged)
  STUDENT_MODEL                 Default student model if first positional arg is omitted
  TEACHER_MODEL                 Default teacher model if second positional arg is omitted
  OUTPUT_DIR                    Override output directory
  RUN_NAME                      Override run name
  EPOCHS                        Used for run-name suffix if --epochs is not provided (default: 30)
  FEEDBACK_WEIGHT_TYPE          fixed|adaptive (default: fixed)
  KL_SUPPORT                    full|topk|sampled (default: full)
  KL_TOPK                       Top-k size used when KL_SUPPORT=topk (default: 50)
  BACKWARD_WEIGHT               Reverse-KL weight on disallowed-only feedback when FEEDBACK_WEIGHT_TYPE=fixed (default: 1.0)
  FORWARD_WEIGHT                Forward-KL weight on allowed-only feedback when FEEDBACK_WEIGHT_TYPE=fixed (default: 1.0)
  TEACHER_UPDATE                Teacher update mode (none|ema|interp|ema+interp; default: none)
  EMA_DECAY                     EMA decay for ema/ema+interp (default: 0.999)
  STUDENT_TARGET_WEIGHT         Detached-student interpolation weight for interp/ema+interp (default: 0.3)
  RESUME_FROM_CHECKPOINT        HF Trainer checkpoint path, or "last" for latest checkpoint in OUTPUT_DIR
  WANDB_RUN_ID                  Existing W&B run id to continue logging into
  WANDB_RESUME                  W&B resume policy (allow|must|never)
  VLLM_GPU_MEMORY_UTILIZATION   vLLM GPU memory fraction (default: 0.3)
  TRUST_REMOTE_CODE             Set to true to pass --trust-remote-code (default: true)

Examples:
  ./run_selfci.sh Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-7B-Instruct
  GPU_ID=1 ./run_selfci.sh --epochs 2 --backward-weight 1.0 --forward-weight 1.0
EOF_USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

DEFAULT_STUDENT="Qwen/Qwen2.5-7B-Instruct"
DEFAULT_TEACHER="Qwen/Qwen2.5-7B-Instruct"

STUDENT_MODEL="${STUDENT_MODEL:-$DEFAULT_STUDENT}"
TEACHER_MODEL="${TEACHER_MODEL:-$DEFAULT_TEACHER}"

if [[ "${1:-}" != "" && "${1:-}" != --* ]]; then
  STUDENT_MODEL="$1"
  shift
fi

if [[ "${1:-}" != "" && "${1:-}" != --* ]]; then
  TEACHER_MODEL="$1"
  shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_PATH="${DATASET_PATH:-wgcyeo/ContextualIntegritySyntheticDataset_Llama-3.1-8B-Instruct_all}"
GPU_ID="${GPU_ID:-0}"
EPOCHS="${EPOCHS:-30}"
FEEDBACK_WEIGHT_TYPE="${FEEDBACK_WEIGHT_TYPE:-fixed}"
KL_SUPPORT="${KL_SUPPORT:-full}"
KL_TOPK="${KL_TOPK:-50}"
BACKWARD_WEIGHT="${BACKWARD_WEIGHT:-1.0}"
FORWARD_WEIGHT="${FORWARD_WEIGHT:-1.0}"
TEACHER_UPDATE="${TEACHER_UPDATE:-none}"
EMA_DECAY="${EMA_DECAY:-0.999}"
STUDENT_TARGET_WEIGHT="${STUDENT_TARGET_WEIGHT:-0.3}"
RUN_NAME_OVERRIDE="${RUN_NAME:-}"
OUTPUT_DIR_OVERRIDE="${OUTPUT_DIR:-}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"
WANDB_RESUME="${WANDB_RESUME:-}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.3}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"

for ((i=1; i<=$#; i++)); do
  arg="${!i}"
  if [[ "${arg}" == "--epochs" ]]; then
    j=$((i + 1))
    if [[ ${j} -le $# ]]; then
      EPOCHS="${!j}"
    fi
  elif [[ "${arg}" == --epochs=* ]]; then
    EPOCHS="${arg#*=}"
  elif [[ "${arg}" == "--backward-weight" ]]; then
    j=$((i + 1))
    if [[ ${j} -le $# ]]; then
      BACKWARD_WEIGHT="${!j}"
    fi
  elif [[ "${arg}" == --backward-weight=* ]]; then
    BACKWARD_WEIGHT="${arg#*=}"
  elif [[ "${arg}" == "--forward-weight" ]]; then
    j=$((i + 1))
    if [[ ${j} -le $# ]]; then
      FORWARD_WEIGHT="${!j}"
    fi
  elif [[ "${arg}" == --forward-weight=* ]]; then
    FORWARD_WEIGHT="${arg#*=}"
  elif [[ "${arg}" == "--feedback-weight-type" ]]; then
    j=$((i + 1))
    if [[ ${j} -le $# ]]; then
      FEEDBACK_WEIGHT_TYPE="${!j}"
    fi
  elif [[ "${arg}" == --feedback-weight-type=* ]]; then
    FEEDBACK_WEIGHT_TYPE="${arg#*=}"
  elif [[ "${arg}" == "--kl-support" ]]; then
    j=$((i + 1))
    if [[ ${j} -le $# ]]; then
      KL_SUPPORT="${!j}"
    fi
  elif [[ "${arg}" == --kl-support=* ]]; then
    KL_SUPPORT="${arg#*=}"
  elif [[ "${arg}" == "--kl-topk" ]]; then
    j=$((i + 1))
    if [[ ${j} -le $# ]]; then
      KL_TOPK="${!j}"
    fi
  elif [[ "${arg}" == --kl-topk=* ]]; then
    KL_TOPK="${arg#*=}"
  elif [[ "${arg}" == "--teacher-update" ]]; then
    j=$((i + 1))
    if [[ ${j} -le $# ]]; then
      TEACHER_UPDATE="${!j}"
    fi
  elif [[ "${arg}" == --teacher-update=* ]]; then
    TEACHER_UPDATE="${arg#*=}"
  elif [[ "${arg}" == "--ema-decay" ]]; then
    j=$((i + 1))
    if [[ ${j} -le $# ]]; then
      EMA_DECAY="${!j}"
    fi
  elif [[ "${arg}" == --ema-decay=* ]]; then
    EMA_DECAY="${arg#*=}"
  elif [[ "${arg}" == "--student-target-weight" ]]; then
    j=$((i + 1))
    if [[ ${j} -le $# ]]; then
      STUDENT_TARGET_WEIGHT="${!j}"
    fi
  elif [[ "${arg}" == --student-target-weight=* ]]; then
    STUDENT_TARGET_WEIGHT="${arg#*=}"
  elif [[ "${arg}" == "--lambda-allow" ]]; then
    j=$((i + 1))
    if [[ ${j} -le $# ]]; then
      FORWARD_WEIGHT="${!j}"
    fi
  elif [[ "${arg}" == --lambda-allow=* ]]; then
    FORWARD_WEIGHT="${arg#*=}"
  elif [[ "${arg}" == "--lambda-disallow" ]]; then
    j=$((i + 1))
    if [[ ${j} -le $# ]]; then
      BACKWARD_WEIGHT="${!j}"
    fi
  elif [[ "${arg}" == --lambda-disallow=* ]]; then
    BACKWARD_WEIGHT="${arg#*=}"
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
  elif [[ "${arg}" == "--resume-from-checkpoint" ]]; then
    j=$((i + 1))
    if [[ ${j} -le $# ]]; then
      RESUME_FROM_CHECKPOINT="${!j}"
    fi
  elif [[ "${arg}" == --resume-from-checkpoint=* ]]; then
    RESUME_FROM_CHECKPOINT="${arg#*=}"
  elif [[ "${arg}" == "--wandb-run-id" ]]; then
    j=$((i + 1))
    if [[ ${j} -le $# ]]; then
      WANDB_RUN_ID="${!j}"
    fi
  elif [[ "${arg}" == --wandb-run-id=* ]]; then
    WANDB_RUN_ID="${arg#*=}"
  elif [[ "${arg}" == "--wandb-resume" ]]; then
    j=$((i + 1))
    if [[ ${j} -le $# ]]; then
      WANDB_RESUME="${!j}"
    fi
  elif [[ "${arg}" == --wandb-resume=* ]]; then
    WANDB_RESUME="${arg#*=}"
  fi
done

STUDENT_TAG="${STUDENT_MODEL##*/}"
TEACHER_TAG="${TEACHER_MODEL##*/}"
TEACHER_UPDATE_TAG="${TEACHER_UPDATE//+/_plus_}"
BACKWARD_TAG="${BACKWARD_WEIGHT//./p}"
FORWARD_TAG="${FORWARD_WEIGHT//./p}"
EMA_TAG="${EMA_DECAY//./p}"
STW_TAG="${STUDENT_TARGET_WEIGHT//./p}"

DEFAULT_RUN_NAME="selfci_${FEEDBACK_WEIGHT_TYPE}_${TEACHER_UPDATE_TAG}_${STUDENT_TAG}"
if [[ "${KL_SUPPORT}" == "topk" ]]; then
  DEFAULT_RUN_NAME+="_kltopk${KL_TOPK}"
elif [[ "${KL_SUPPORT}" == "sampled" ]]; then
  DEFAULT_RUN_NAME+="_klsampled"
fi
if [[ "${TEACHER_UPDATE}" != "ema" && "${TEACHER_UPDATE}" != "ema+interp" ]]; then
  DEFAULT_RUN_NAME+="_from_${TEACHER_TAG}"
fi
if [[ "${FEEDBACK_WEIGHT_TYPE}" == "fixed" ]]; then
  DEFAULT_RUN_NAME+="_bw${BACKWARD_TAG}_fw${FORWARD_TAG}"
fi
if [[ "${TEACHER_UPDATE}" == "ema" || "${TEACHER_UPDATE}" == "ema+interp" ]]; then
  DEFAULT_RUN_NAME+="_ema${EMA_TAG}"
fi
if [[ "${TEACHER_UPDATE}" == "interp" || "${TEACHER_UPDATE}" == "ema+interp" ]]; then
  DEFAULT_RUN_NAME+="_stw${STW_TAG}"
fi
DEFAULT_RUN_NAME+="_ep${EPOCHS}"
RUN_NAME="${RUN_NAME_OVERRIDE:-$DEFAULT_RUN_NAME}"
OUTPUT_DIR="${OUTPUT_DIR_OVERRIDE:-./output/${RUN_NAME}}"

if [[ -n "${RESUME_FROM_CHECKPOINT}" && -z "${OUTPUT_DIR_OVERRIDE}" ]]; then
  resume_token="${RESUME_FROM_CHECKPOINT,,}"
  if [[ "${resume_token}" != "1" && "${resume_token}" != "true" && "${resume_token}" != "yes" && "${resume_token}" != "last" && "${resume_token}" != "latest" && "${resume_token}" != "auto" ]]; then
    OUTPUT_DIR="$(dirname "${RESUME_FROM_CHECKPOINT}")"
    if [[ -z "${RUN_NAME_OVERRIDE}" ]]; then
      RUN_NAME="$(basename "${OUTPUT_DIR}")"
    fi
  fi
fi

CMD=(
  accelerate launch "${SCRIPT_DIR}/src/run_selfci.py"
  --student-model-name-or-path "${STUDENT_MODEL}"
  --teacher-model-name-or-path "${TEACHER_MODEL}"
  --dataset-name-or-path "${DATASET_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --run-name "${RUN_NAME}"
  --feedback-weight-type "${FEEDBACK_WEIGHT_TYPE}"
  --kl-support "${KL_SUPPORT}"
  --kl-topk "${KL_TOPK}"
  --teacher-update "${TEACHER_UPDATE}"
  --ema-decay "${EMA_DECAY}"
  --student-target-weight "${STUDENT_TARGET_WEIGHT}"
  --vllm-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
)

if [[ "${FEEDBACK_WEIGHT_TYPE}" == "fixed" ]]; then
  CMD+=(
    --backward-weight "${BACKWARD_WEIGHT}"
    --forward-weight "${FORWARD_WEIGHT}"
  )
fi

if [[ "${TRUST_REMOTE_CODE}" == "true" ]]; then
  CMD+=(--trust-remote-code)
fi

if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  CMD+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

if [[ -n "${WANDB_RUN_ID}" ]]; then
  CMD+=(--wandb-run-id "${WANDB_RUN_ID}")
fi

if [[ -n "${WANDB_RESUME}" ]]; then
  CMD+=(--wandb-resume "${WANDB_RESUME}")
fi

if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

echo "Launching weighted asymmetric bidirectional KL on-policy distillation..."
echo "  GPU               : ${GPU_ID}"
echo "  Student           : ${STUDENT_MODEL}"
echo "  Teacher           : ${TEACHER_MODEL}"
echo "  Dataset           : ${DATASET_PATH}"
echo "  Weight type       : ${FEEDBACK_WEIGHT_TYPE}"
if [[ "${FEEDBACK_WEIGHT_TYPE}" == "fixed" ]]; then
  echo "  Backward weight   : ${BACKWARD_WEIGHT}"
  echo "  Forward weight    : ${FORWARD_WEIGHT}"
fi
echo "  Teacher update    : ${TEACHER_UPDATE}"
if [[ "${TEACHER_UPDATE}" == "ema" || "${TEACHER_UPDATE}" == "ema+interp" ]]; then
  echo "  EMA decay         : ${EMA_DECAY}"
fi
if [[ "${TEACHER_UPDATE}" == "interp" || "${TEACHER_UPDATE}" == "ema+interp" ]]; then
  echo "  Student tgt wt    : ${STUDENT_TARGET_WEIGHT}"
fi
echo "  Output            : ${OUTPUT_DIR}"
echo "  Run name          : ${RUN_NAME}"
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  echo "  Resume checkpoint : ${RESUME_FROM_CHECKPOINT}"
fi
if [[ -n "${WANDB_RUN_ID}" ]]; then
  echo "  W&B run id        : ${WANDB_RUN_ID}"
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CMD[@]}"
