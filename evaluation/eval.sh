#!/usr/bin/env bash
set -euo pipefail

OUTPUT_JSON="${OUTPUT_JSON:-../data/output_openai_pred.json}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "Error: neither python3 nor python is available in PATH." >&2
    exit 1
  fi
fi

"${PYTHON_BIN}" score_by_gpt.py --answer "${OUTPUT_JSON}"
"${PYTHON_BIN}" static_score.py --answer "${OUTPUT_JSON}"
