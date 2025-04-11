#!/bin/bash
# This script runs joint training with specific model paths

# Navigate to project root directory
cd "$(dirname "$0")/.."

# Find the latest model versions
STNPP_MODEL=$(ls -t trained_models/stnpp/stnpp_best_v*.pt | head -1)
QAL_MODEL=$(ls -t trained_models/qal/qal_best_v*.pt | head -1)
PROXY_MODEL=$(ls -t trained_models/proxy/proxy_network_best_v*.pt | head -1)

echo "Using models:"
echo "STNPP: $STNPP_MODEL"
echo "QAL: $QAL_MODEL"
echo "PROXY: $PROXY_MODEL"

# Run joint training with specific model paths
python scripts/train_joint.py \
  --stnpp_model "$STNPP_MODEL" \
  --qal_model "$QAL_MODEL" \
  --proxy_model "$PROXY_MODEL" \
  --dataset datasets/MOTChallenge/processed \
  --batch_size 8 \
  --epochs 20 \
  --lr 1e-5 \
  --lambda_distortion 1.0 \
  --lambda_rate 0.1 \
  --lambda_perception 0.01 \
  --output_dir trained_models/joint \
  --log_dir logs/joint 