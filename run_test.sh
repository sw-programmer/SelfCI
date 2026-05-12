#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME_OR_PATH="${1:-}"
shift $(( $# >= 1 ? 1 : $# ))

if [[ -z "${MODEL_NAME_OR_PATH}" ]]; then
  echo "Usage: ./run_test.sh <model_name_or_path> [extra args...]" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_ID="${GPU_ID:-0}"
if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "Neither python nor python3 was found in PATH." >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" "${SCRIPT_DIR}/run_test.py" \
  --model-name-or-path "${MODEL_NAME_OR_PATH}" \
  "$@"
