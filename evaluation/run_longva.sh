#!/usr/bin/env bash
set -euo pipefail

# LongVA-7B-DPO 推理脚本
# 使用方法: bash run_longva.sh

cd "$(dirname "$0")"

# 配置参数
INPUT_JSON="${INPUT_JSON:-../data/ECbench_qa.json}"
OUTPUT_JSON="${OUTPUT_JSON:-../data/output_longva_pred.json}"
RGB_ROOT="${RGB_ROOT:-../data/rgb_video/rgb_video}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/LongVA-7B-DPO}"
NUM_FRAMES="${NUM_FRAMES:-32}"
MAX_ITEMS="${MAX_ITEMS:-}"
INDICES="${INDICES:-}"
OVERWRITE="${OVERWRITE:-false}"
SAVE_EVERY="${SAVE_EVERY:-20}"
DEBUG="${DEBUG:-false}"
DEVICE="${DEVICE:-cuda:0}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "Error: neither python3 nor python is available in PATH." >&2
    exit 1
  fi
fi

# 构建命令参数
ARGS=(
  --input "${INPUT_JSON}"
  --output "${OUTPUT_JSON}"
  --rgb-root "${RGB_ROOT}"
  --model-path "${MODEL_PATH}"
  --num-frames "${NUM_FRAMES}"
  --save-every "${SAVE_EVERY}"
  --device "${DEVICE}"
)

if [ -n "${MAX_ITEMS}" ]; then
  ARGS+=(--max-items "${MAX_ITEMS}")
fi

if [ -n "${INDICES}" ]; then
  ARGS+=(--indices "${INDICES}")
fi

if [ "${OVERWRITE}" = "true" ]; then
  ARGS+=(--overwrite)
fi

if [ "${DEBUG}" = "true" ]; then
  ARGS+=(--debug)
fi

echo "=========================================="
echo "Running LongVA-7B-DPO Inference"
echo "=========================================="
echo "Input: ${INPUT_JSON}"
echo "Output: ${OUTPUT_JSON}"
echo "Model: ${MODEL_PATH}"
echo "Num Frames: ${NUM_FRAMES}"
echo "Device: ${DEVICE}"
echo "=========================================="

"${PYTHON_BIN}" generate_pred_longva.py "${ARGS[@]}"

echo ""
echo "Inference completed! Output saved to: ${OUTPUT_JSON}"
