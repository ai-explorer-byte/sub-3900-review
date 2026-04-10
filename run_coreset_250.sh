#!/bin/bash

# Run the pipeline for all models with Coreset scorer and Budget 250

set -e

CONFIG="config_coreset_250.yaml"
OUTPUT_DIR="output_coreset250"

echo "=========================================="
echo "Coreset Scorer | Budget: 250"
echo "Config: $CONFIG"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Get list of models dynamically
MODELS=$(python main.py --list_models --model_name dummy --config "$CONFIG" 2>/dev/null | grep "^  - " | sed 's/^  - //')

if [ -z "$MODELS" ]; then
    echo "Error: No models found"
    exit 1
fi

echo "Found models:"
echo "$MODELS"
echo "=========================================="

for model in $MODELS; do
    echo ""
    echo "Running pipeline for model: $model"
    echo "=========================================="
    python main.py --model_name "$model" --config "$CONFIG" --output_dir "$OUTPUT_DIR" --log_level INFO
    # Small delay to ensure wandb fully syncs before next run
    sleep 2
done

# Sync any remaining offline runs (if any)
echo ""
echo "Syncing any remaining wandb data..."
wandb sync wandb/ 2>/dev/null || true

echo ""
echo "Coreset Budget 250 - All models processed successfully!"
echo "Results saved to: $OUTPUT_DIR/"
