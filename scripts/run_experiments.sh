#!/bin/bash
# Reproduce all experiments from the paper.
# Run on a machine with an NVIDIA GPU (L4/A100 recommended).

set -e

echo "=============================="
echo "Function Calling Fine-Tuning"
echo "Full Experiment Reproduction"
echo "=============================="

# Model comparison experiments
MODELS=(
    "Qwen/Qwen2.5-1.5B-Instruct"
    "Qwen/Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "microsoft/Phi-3.5-mini-instruct"
)

OUTPUT_NAMES=(
    "qwen25_1.5b"
    "qwen25_3b"
    "qwen25_7b"
    "llama32_3b"
    "phi35_mini"
)

echo ""
echo "Phase 1: Multi-Model Comparison"
echo "================================"

for i in "${!MODELS[@]}"; do
    MODEL=${MODELS[$i]}
    NAME=${OUTPUT_NAMES[$i]}
    echo ""
    echo "Training: $MODEL"
    
    python src/train.py \
        --model_id "$MODEL" \
        --output_dir "./output/$NAME" \
        --train_samples 10000 \
        --lora_rank 16 \
        --epochs 3

    echo "Evaluating: $MODEL (fine-tuned)"
    python src/evaluate.py \
        --model_id "$MODEL" \
        --adapter_path "./output/$NAME" \
        --save_path "./results/${NAME}_finetuned.json"

    echo "Evaluating: $MODEL (zero-shot)"
    python src/evaluate.py \
        --model_id "$MODEL" \
        --save_path "./results/${NAME}_zeroshot.json"
done

echo ""
echo "Phase 2: Data Scaling (Qwen2.5-3B)"
echo "===================================="

for N in 500 1000 3000 10000 30000; do
    echo ""
    echo "Training with $N samples"
    
    python src/train.py \
        --model_id "Qwen/Qwen2.5-3B-Instruct" \
        --output_dir "./output/data_${N}" \
        --train_samples $N \
        --lora_rank 16 \
        --epochs 3

    python src/evaluate.py \
        --model_id "Qwen/Qwen2.5-3B-Instruct" \
        --adapter_path "./output/data_${N}" \
        --save_path "./results/data_${N}.json"
done

echo ""
echo "Generating figures..."
python results/generate_figures.py

echo ""
echo "All experiments complete!"
