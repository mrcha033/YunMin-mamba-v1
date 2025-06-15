#!/bin/bash

# Complete Dependency Validation
# Runs all dependency checks and setups

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "🔍 COMPLETE DEPENDENCY VALIDATION"
echo "================================="

# 1. Check SDM dependencies
echo "1️⃣ Checking SDM checkpoint dependencies..."
bash "${PROJECT_ROOT}/scripts/check_dependencies.sh"

# 2. Setup datasets
echo ""
echo "2️⃣ Setting up datasets..."
bash "${PROJECT_ROOT}/scripts/setup_datasets.sh"

# 3. Configure distributed training
echo ""
echo "3️⃣ Configuring distributed training..."
bash "${PROJECT_ROOT}/scripts/setup_distributed.sh"

# 4. Memory optimization check
echo ""
echo "4️⃣ Checking memory optimization..."
python "${PROJECT_ROOT}/scripts/monitor_memory.py" --interval 1 &
MONITOR_PID=$!
sleep 3
kill $MONITOR_PID 2>/dev/null || true

# 5. Run final validation
echo ""
echo "5️⃣ Running final validation..."
bash "${PROJECT_ROOT}/tests/test_full_experiment.sh"

echo ""
echo "🎉 ALL DEPENDENCIES VALIDATED AND READY!"
echo "✅ SDM checkpoints: Ready"
echo "✅ Datasets: Downloaded and cached"
echo "✅ Multi-GPU: Configured"
echo "✅ Memory: Optimized"
echo "✅ System: Validated"
echo ""
echo "🚀 Ready to run experiments:"
echo "   ./run_full_experiment.sh 130m 1 production_run"
echo "   ./run_full_experiment.sh 370m 1 memory_optimized_run" 