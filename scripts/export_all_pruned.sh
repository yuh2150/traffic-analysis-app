#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Directory containing the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Active Python interpreter from env_cv environment
PYTHON_BIN="/home/huy/miniconda3/envs/env_cv/bin/python"

echo "=========================================================="
echo "Starting batch export of pruned and baseline models to ONNX"
echo "=========================================================="

# List of checkpoints to export: (model, checkpoint_path)
checkpoints=(
    "detr:checkpoints/detr/baseline/best.pt"
    "detr:checkpoints/detr/pruned/magnitude_0.3.pt"
    "detr:checkpoints/detr/pruned/magnitude_0.5.pt"
    "detr:checkpoints/detr/pruned/magnitude_0.7.pt"
    "yolov5s:checkpoints/yolov5s/baseline/best.pt"
    "yolov5s:checkpoints/yolov5s/pruned/magnitude_0.3.pt"
    "yolov5s:checkpoints/yolov5s/pruned/magnitude_0.5.pt"
    "yolov5s:checkpoints/yolov5s/pruned/magnitude_0.7.pt"
)

for item in "${checkpoints[@]}"; do
    model="${item%%:*}"
    ckpt="${item#*:}"
    
    ckpt_full_path="$PROJECT_ROOT/$ckpt"
    
    if [ -f "$ckpt_full_path" ]; then
        echo ""
        echo "----------------------------------------------------------"
        echo "Exporting $model from checkpoint: $ckpt"
        echo "----------------------------------------------------------"
        
        $PYTHON_BIN "$SCRIPT_DIR/export_onnx.py" \
            --model "$model" \
            --checkpoint "$ckpt_full_path" \
            --verify
    else
        echo "Warning: Checkpoint not found at $ckpt_full_path. Skipping."
    fi
done

echo ""
echo "=========================================================="
echo "Batch export finished successfully!"
echo "=========================================================="
