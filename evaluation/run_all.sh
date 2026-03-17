#!/usr/bin/env bash
set -euo pipefail

# 运行所有模型的推理脚本
# 使用方法: bash run_all.sh
# 注意：此脚本会处理所有数据，不设置MAX_ITEMS限制

cd "$(dirname "$0")"

echo "=========================================="
echo "ECBench Benchmark - Running All Models"
echo "=========================================="
echo ""

# 检查输入文件是否存在
INPUT_JSON="${INPUT_JSON:-../data/ECbench_qa.json}"
if [ ! -f "${INPUT_JSON}" ]; then
    echo "Error: Input file not found: ${INPUT_JSON}"
    echo "Please ensure ECbench_qa.json exists in the data directory."
    exit 1
fi

# 确保不设置MAX_ITEMS限制（处理所有数据）
unset MAX_ITEMS

# 设置OVERWRITE=true以重新处理所有数据
export OVERWRITE=true

# 运行InternVL2-8B
echo "=========================================="
echo "Step 1/2: Running InternVL2-8B (All Data)"
echo "=========================================="
bash run_internvl2.sh
echo ""

# 运行LongVA-7B-DPO
echo "=========================================="
echo "Step 2/2: Running LongVA-7B-DPO (All Data)"
echo "=========================================="
bash run_longva.sh
echo ""

echo "=========================================="
echo "All models completed!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  - InternVL2: ../data/output_internvl2_pred.json"
echo "  - LongVA: ../data/output_longva_pred.json"
echo ""
echo "To evaluate the results, run:"
echo "  cd evaluation"
echo "  export OUTPUT_JSON=../data/output_internvl2_pred.json"
echo "  bash eval.sh"
echo "  # or for LongVA:"
echo "  export OUTPUT_JSON=../data/output_longva_pred.json"
echo "  bash eval.sh"
