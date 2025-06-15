#!/bin/bash

# Distributed Training Setup and Validation
# Configures multi-GPU training environment

set -e

echo "🚀 DISTRIBUTED TRAINING SETUP"
echo "============================="

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found. CUDA not available."
    exit 1
fi

# Get GPU information
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "🔍 Detected ${GPU_COUNT} GPU(s)"

if [[ ${GPU_COUNT} -eq 0 ]]; then
    echo "❌ No GPUs detected"
    exit 1
fi

# Display GPU details 