#!/bin/bash

# Run the pipeline for all available models with gamma=1.0

set -e

CONFIG="config_gamma_1.yaml"

# Get list of models dynamically
MODELS=$(python main.py --list_models --model_name dummy 2>/dev/null | grep "^  - " | sed 's/^  - //')

if [ -z "$MODELS" ]; then
    echo "Error: No models found"
    exit 1
fi

echo "Found models:"
echo "$MODELS"
echo "Using config: $CONFIG"
echo "=========================================="

for model in $MODELS; do
    echo ""
    echo "Running pipeline for model: $model"
    echo "Config: $CONFIG"
    echo "=========================================="
    python main.py --model_name "$model" --config "$CONFIG" --log_level INFO
    # Small delay to ensure wandb fully syncs before next run
    sleep 2
done

# Sync any remaining offline runs (if any)
echo ""
echo "Syncing any remaining wandb data..."
wandb sync wandb/ 2>/dev/null || true

echo ""
echo "All models processed successfully!"
