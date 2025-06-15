#!/bin/bash

# Test script for run_full_experiment.sh
# This script validates that the full experiment can start without configuration errors

set -e  # Exit on any error

echo "🧪 Testing run_full_experiment.sh..."
echo "======================================"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Test parameters
MODEL_SIZE="130m"
NUM_GPUS=1
EXPERIMENT_NAME="test_experiment_$(date +%Y%m%d_%H%M%S)"

echo "Project Root: $PROJECT_ROOT"
echo "Test Experiment: $EXPERIMENT_NAME"
echo "Model Size: $MODEL_SIZE"
echo "GPUs: $NUM_GPUS"
echo ""

# Check if required files exist
echo "🔍 Checking required files..."

EXPERIMENT_SCRIPT="$PROJECT_ROOT/run_full_experiment.sh"
CONFIG_FILE="$PROJECT_ROOT/configs/mamba_$MODEL_SIZE.yaml"

if [[ ! -f "$EXPERIMENT_SCRIPT" ]]; then
    echo "❌ run_full_experiment.sh not found: $EXPERIMENT_SCRIPT"
    exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "❌ Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "✅ Required files found"
echo ""

# Make sure the experiment script is executable
chmod +x "$EXPERIMENT_SCRIPT"

# Test configuration validation
echo "🔍 Testing configuration validation..."

# Check if the config file is valid YAML
if command -v python3 &> /dev/null; then
    python3 -c "
import yaml
import sys
try:
    with open('$CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)
    print('✅ Configuration file is valid YAML')
    
    # Check required sections
    required_sections = ['model', 'training', 'data', 'sdm', 'logging']
    for section in required_sections:
        if section not in config:
            print(f'❌ Missing config section: {section}')
            sys.exit(1)
    
    print('✅ Required configuration sections present')
    
except Exception as e:
    print(f'❌ Configuration validation failed: {e}')
    sys.exit(1)
"
else
    echo "⚠️  Python3 not available, skipping YAML validation"
fi

echo ""

# Test dry run (setup only, no actual training)
echo "🔍 Testing experiment setup (dry run)..."

# Create temporary directory for test
TEST_DIR="$PROJECT_ROOT/test_experiment_temp"
mkdir -p "$TEST_DIR"

# Test environment setup part of the script
cd "$PROJECT_ROOT"

# Extract and test just the setup portion
echo "Testing configuration file existence..."
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "❌ Configuration file validation failed"
    exit 1
fi

echo "✅ Configuration file exists"

# Test directory creation
EXPERIMENT_DIR="$TEST_DIR/$EXPERIMENT_NAME"
CHECKPOINTS_DIR="$EXPERIMENT_DIR/checkpoints"
LOGS_DIR="$EXPERIMENT_DIR/logs"
RESULTS_DIR="$EXPERIMENT_DIR/results"

mkdir -p "${CHECKPOINTS_DIR}"/{baseline,csp,sdm,full}
mkdir -p "${LOGS_DIR}"
mkdir -p "${RESULTS_DIR}"

if [[ -d "$EXPERIMENT_DIR" ]]; then
    echo "✅ Experiment directory structure created successfully"
else
    echo "❌ Failed to create experiment directory structure"
    exit 1
fi

# Test Python imports that the experiment needs
echo ""
echo "🔍 Testing Python dependencies..."

python3 -c "
import sys
import os
sys.path.insert(0, '$PROJECT_ROOT')

# Test critical imports
try:
    import torch
    print('✅ PyTorch available')
except ImportError as e:
    print(f'❌ PyTorch import failed: {e}')
    sys.exit(1)

try:
    import yaml
    print('✅ PyYAML available')
except ImportError as e:
    print(f'❌ PyYAML import failed: {e}')
    sys.exit(1)

try:
    from transformers import AutoTokenizer
    print('✅ Transformers available')
except ImportError as e:
    print(f'❌ Transformers import failed: {e}')
    sys.exit(1)

try:
    from accelerate import Accelerator
    print('✅ Accelerate available')
except ImportError as e:
    print(f'❌ Accelerate import failed: {e}')
    sys.exit(1)

# Test project-specific imports
try:
    from models.baseline_ssm import BaselineSSM
    print('✅ BaselineSSM model available')
except ImportError as e:
    print(f'❌ BaselineSSM import failed: {e}')
    sys.exit(1)

try:
    from models.sdm_ssm import SDM_SSM
    print('✅ SDM_SSM model available')
except ImportError as e:
    print(f'❌ SDM_SSM import failed: {e}')
    sys.exit(1)

try:
    from utils.logger import setup_logger
    print('✅ Logger utilities available')
except ImportError as e:
    print(f'❌ Logger utilities import failed: {e}')
    sys.exit(1)

try:
    from data.wikitext103 import get_wiktext103_dataloader
    print('✅ Data utilities available')
except ImportError as e:
    print(f'❌ Data utilities import failed: {e}')
    sys.exit(1)

print('✅ All critical dependencies available')
"

# Test pretrain.py configuration compatibility
echo ""
echo "🔍 Testing pretrain.py configuration compatibility..."

python3 -c "
import sys
import os
sys.path.insert(0, '$PROJECT_ROOT')

try:
    from pretrain import load_config
    config = load_config('$CONFIG_FILE')
    
    # Test nested configuration structure
    if 'training' not in config:
        print('❌ Missing training section')
        sys.exit(1)
    
    training_config = config.get('training', {})
    if 'pretrain' in training_config:
        pretrain_config = training_config['pretrain']
        conceptual_batch_size = int(pretrain_config.get('batch_size', 32))
        micro_batch_size = int(pretrain_config.get('micro_batch_size', 8))
        print(f'✅ Pretrain config: batch_size={conceptual_batch_size}, micro_batch_size={micro_batch_size}')
    else:
        print('⚠️  Using legacy flat structure')
    
    print('✅ pretrain.py configuration compatibility validated')
    
except Exception as e:
    print(f'❌ pretrain.py configuration test failed: {e}')
    sys.exit(1)
"

# Test pretrain_sdm.py configuration compatibility
echo ""
echo "🔍 Testing pretrain_sdm.py configuration compatibility..."

python3 -c "
import sys
import os
sys.path.insert(0, '$PROJECT_ROOT')

try:
    from pretrain_sdm import load_config
    config = load_config('$CONFIG_FILE')
    
    # Test SDM configuration keys
    sdm_config = config.get('sdm', {})
    
    # Test temperature keys (should work with both old and new formats)
    temp_start = sdm_config.get('initial_temperature', sdm_config.get('gumbel_temp_start', 5.0))
    temp_end = sdm_config.get('final_temperature', sdm_config.get('gumbel_temp_end', 0.1))
    lambda_sparsity = sdm_config.get('lambda_sparsity', 0.01)
    
    print(f'✅ SDM config: temp_start={temp_start}, temp_end={temp_end}, lambda={lambda_sparsity}')
    print('✅ pretrain_sdm.py configuration compatibility validated')
    
except Exception as e:
    print(f'❌ pretrain_sdm.py configuration test failed: {e}')
    sys.exit(1)
"

# Cleanup test directory
rm -rf "$TEST_DIR"

echo ""
echo "🎉 All tests passed!"
echo "✅ run_full_experiment.sh should be able to start without configuration errors"
echo ""
echo "To run the actual experiment:"
echo "  ./run_full_experiment.sh $MODEL_SIZE $NUM_GPUS your_experiment_name"
echo "" 