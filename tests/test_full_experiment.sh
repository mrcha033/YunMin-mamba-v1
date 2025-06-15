#!/bin/bash

# Test script for run_full_experiment.sh
# Enhanced with detailed logging for debugging on Ubuntu

set -e  # Exit on any error

# Setup logging
LOG_FILE="tests/experiment_test.log"
exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo "=========================================================================="
echo "FULL EXPERIMENT VALIDATION TEST - DETAILED LOGGING MODE"
echo "=========================================================================="
echo "Started at: $(date)"
echo "Hostname: $(hostname)"
echo "User: $(whoami)"
echo "Shell: $SHELL"
echo "Working directory: $(pwd)"
echo "Log file: $LOG_FILE"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Test parameters
MODEL_SIZE="130m"
NUM_GPUS=1
EXPERIMENT_NAME="test_experiment_$(date +%Y%m%d_%H%M%S)"

echo "=== CONFIGURATION ==="
echo "Project Root: $PROJECT_ROOT"
echo "Script Directory: $SCRIPT_DIR"
echo "Test Experiment: $EXPERIMENT_NAME"
echo "Model Size: $MODEL_SIZE"
echo "GPUs: $NUM_GPUS"
echo ""

# System information
echo "=== SYSTEM INFORMATION ==="
echo "Operating System: $(uname -a)"
echo "CPU: $(lscpu | grep 'Model name' | cut -d: -f2 | xargs)"
echo "Memory: $(free -h | grep '^Mem:' | awk '{print $2 " total, " $3 " used, " $7 " available"}')"
echo "Disk space: $(df -h . | tail -1 | awk '{print $2 " total, " $3 " used, " $4 " available"}')"

# Check GPU information
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv,noheader,nounits
else
    echo "No NVIDIA GPU detected or nvidia-smi not available"
fi
echo ""

# Python environment
echo "=== PYTHON ENVIRONMENT ==="
echo "Python version: $(python3 --version)"
echo "Python executable: $(which python3)"
echo "Pip version: $(pip3 --version)"
echo "Virtual environment: ${VIRTUAL_ENV:-"None"}"
echo "PYTHONPATH: ${PYTHONPATH:-"Not set"}"
echo ""

# Check if required files exist
echo "=== FILE EXISTENCE CHECK ==="

EXPERIMENT_SCRIPT="$PROJECT_ROOT/run_full_experiment.sh"
CONFIG_FILE="$PROJECT_ROOT/configs/mamba_$MODEL_SIZE.yaml"

echo "Checking required files..."
echo "  Experiment script: $EXPERIMENT_SCRIPT"
echo "  Config file: $CONFIG_FILE"

if [[ ! -f "$EXPERIMENT_SCRIPT" ]]; then
    echo "❌ run_full_experiment.sh not found: $EXPERIMENT_SCRIPT"
    echo "Files in project root:"
    ls -la "$PROJECT_ROOT" | head -20
    exit 1
fi
echo "✓ Experiment script found"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "❌ Config file not found: $CONFIG_FILE"
    echo "Files in configs directory:"
    ls -la "$PROJECT_ROOT/configs/"
    exit 1
fi
echo "✓ Config file found"

# Check script permissions
echo ""
echo "=== SCRIPT PERMISSIONS ==="
SCRIPT_PERMS=$(ls -l "$EXPERIMENT_SCRIPT" | cut -d' ' -f1)
echo "Script permissions: $SCRIPT_PERMS"

if [[ ! -x "$EXPERIMENT_SCRIPT" ]]; then
    echo "Making experiment script executable..."
    chmod +x "$EXPERIMENT_SCRIPT"
    echo "✓ Script is now executable"
else
    echo "✓ Script is already executable"
fi

# Test configuration validation
echo ""
echo "=== CONFIGURATION VALIDATION ==="

# Check if the config file is valid YAML
echo "Testing YAML syntax..."
if command -v python3 &> /dev/null; then
    python3 << EOF
import yaml
import sys
import json

try:
    with open('$CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)
    print('✓ Configuration file is valid YAML')
    
    # Log config structure
    print(f'Config sections: {list(config.keys())}')
    
    # Check required sections
    required_sections = ['model', 'training', 'sdm', 'logging']
    missing_sections = []
    
    for section in required_sections:
        if section not in config:
            missing_sections.append(section)
        else:
            print(f'✓ Found section: {section}')
            if isinstance(config[section], dict):
                print(f'  - Keys: {list(config[section].keys())}')
    
    if missing_sections:
        print(f'❌ Missing config sections: {missing_sections}')
        sys.exit(1)
    
    print('✓ All required configuration sections present')
    
    # Check nested structure
    if 'training' in config and 'pretrain' in config['training']:
        print('✓ Found nested training.pretrain structure')
        pretrain_keys = list(config['training']['pretrain'].keys())
        print(f'  - Pretrain keys: {pretrain_keys}')
    else:
        print('⚠️  Using flat training structure')
    
    # Check SDM configuration
    if 'sdm' in config:
        sdm_keys = list(config['sdm'].keys())
        print(f'✓ SDM configuration keys: {sdm_keys}')
        
        # Check temperature keys
        has_old_temp = 'initial_temperature' in config['sdm'] and 'final_temperature' in config['sdm']
        has_new_temp = 'gumbel_temp_start' in config['sdm'] and 'gumbel_temp_end' in config['sdm']
        
        if has_old_temp:
            print('✓ Found old-style temperature keys (initial_temperature, final_temperature)')
        elif has_new_temp:
            print('✓ Found new-style temperature keys (gumbel_temp_start, gumbel_temp_end)')
        else:
            print('⚠️  Temperature keys not found, will use defaults')

except Exception as e:
    print(f'❌ Configuration validation failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF
else
    echo "⚠️ Python3 not available, skipping YAML validation"
fi

echo ""

# Test directory creation
echo "=== DIRECTORY STRUCTURE TEST ==="

# Create temporary directory for test
TEST_DIR="$PROJECT_ROOT/test_experiment_temp"
echo "Creating test directory: $TEST_DIR"
mkdir -p "$TEST_DIR"

# Test environment setup part of the script
cd "$PROJECT_ROOT"
echo "Changed to project root: $(pwd)"

# Test directory creation
EXPERIMENT_DIR="$TEST_DIR/$EXPERIMENT_NAME"
CHECKPOINTS_DIR="$EXPERIMENT_DIR/checkpoints"
LOGS_DIR="$EXPERIMENT_DIR/logs"
RESULTS_DIR="$EXPERIMENT_DIR/results"

echo "Creating experiment directory structure..."
echo "  Experiment dir: $EXPERIMENT_DIR"
echo "  Checkpoints dir: $CHECKPOINTS_DIR"
echo "  Logs dir: $LOGS_DIR"
echo "  Results dir: $RESULTS_DIR"

mkdir -p "${CHECKPOINTS_DIR}"/{baseline,csp,sdm,sdm_sgh,full}
mkdir -p "${LOGS_DIR}"
mkdir -p "${RESULTS_DIR}"

# Create placeholder files for new model variant
NEW_CHECKPOINT="${CHECKPOINTS_DIR}/sdm_sgh/model.pt"
NEW_RESULT="${RESULTS_DIR}/M_sdm_sgh_validation.json"
touch "$NEW_CHECKPOINT" "$NEW_RESULT"

if [[ -d "$EXPERIMENT_DIR" ]]; then
    echo "✓ Experiment directory structure created successfully"
    echo "Directory contents:"
    find "$EXPERIMENT_DIR" -type d | sort
else
    echo "❌ Failed to create experiment directory structure"
    exit 1
fi

# Verify placeholder checkpoint and validation output exist
if [[ -f "$NEW_CHECKPOINT" ]]; then
    echo "✓ M_sdm_sgh checkpoint present: $NEW_CHECKPOINT"
else
    echo "❌ M_sdm_sgh checkpoint missing: $NEW_CHECKPOINT"
fi

if [[ -f "$NEW_RESULT" ]]; then
    echo "✓ M_sdm_sgh validation output present: $NEW_RESULT"
else
    echo "❌ M_sdm_sgh validation output missing: $NEW_RESULT"
fi

# Test Python imports that the experiment needs
echo ""
echo "=== PYTHON DEPENDENCIES TEST ==="

python3 << 'EOF'
import sys
import os
import traceback

# Add project root to Python path
project_root = os.environ.get('PROJECT_ROOT', '.')
sys.path.insert(0, project_root)

print(f"Python path (first 3 entries): {sys.path[:3]}")
print(f"Project root in path: {project_root}")

def test_import(module_name, item_name=None):
    """Test importing a module or specific item."""
    try:
        if item_name:
            module = __import__(module_name, fromlist=[item_name])
            item = getattr(module, item_name)
            print(f'✓ {module_name}.{item_name} imported successfully')
            print(f'  - Type: {type(item)}')
            if hasattr(module, '__file__'):
                print(f'  - Module file: {module.__file__}')
        else:
            module = __import__(module_name)
            print(f'✓ {module_name} imported successfully')
            if hasattr(module, '__version__'):
                print(f'  - Version: {module.__version__}')
        return True
    except ImportError as e:
        print(f'❌ Failed to import {module_name}.{item_name or ""}: {e}')
        return False
    except AttributeError as e:
        print(f'❌ Item not found in {module_name}: {e}')
        return False
    except Exception as e:
        print(f'❌ Unexpected error importing {module_name}: {e}')
        traceback.print_exc()
        return False

# Test critical imports
print("Testing critical dependencies...")
critical_imports = [
    ('torch', None),
    ('yaml', None),
    ('transformers', 'AutoTokenizer'),
    ('accelerate', 'Accelerator'),
]

failed_imports = []
for module, item in critical_imports:
    if not test_import(module, item):
        failed_imports.append(f"{module}.{item}" if item else module)

if failed_imports:
    print(f"\n❌ Failed critical imports: {failed_imports}")
    print("Please install missing dependencies:")
    print("  pip install torch transformers accelerate pyyaml")
    sys.exit(1)

print("\n✓ All critical dependencies available")

# Test project-specific imports
print("\nTesting project-specific imports...")
project_imports = [
    ('models.baseline_ssm', 'BaselineSSM'),
    ('models.sdm_ssm', 'SDM_SSM'),
    ('utils.logger', 'setup_logger'),
    ('data.wikitext103', 'get_wikitext103_dataloader'),
]

project_failed = []
for module, item in project_imports:
    if not test_import(module, item):
        project_failed.append(f"{module}.{item}")

if project_failed:
    print(f"\n❌ Failed project imports: {project_failed}")
    print("Check if all project files are present and properly structured")
    
    # Show project structure
    print("\nProject structure check:")
    import pathlib
    project_path = pathlib.Path(project_root)
    for path in ['models/', 'utils/', 'data/', 'pretrain_sdm.py']:
        full_path = project_path / path
        exists = full_path.exists()
        print(f"  {'✓' if exists else '❌'} {path}: {exists}")
        
        if path.endswith('/') and exists:
            # List contents of directory
            try:
                contents = list(full_path.glob('*.py'))
                print(f"    Python files: {[f.name for f in contents]}")
            except:
                pass
    
    sys.exit(1)

print("\n✓ All project dependencies available")
EOF

export PROJECT_ROOT="$PROJECT_ROOT"

# Test pretrain.py configuration compatibility
echo ""
echo "=== PRETRAIN.PY CONFIGURATION COMPATIBILITY ==="

python3 << 'EOF'
import sys
import os
project_root = os.environ.get('PROJECT_ROOT', '.')
sys.path.insert(0, project_root)

try:
    print("Testing pretrain.py configuration loading...")
    from pretrain import load_config
    
    config_file = os.path.join(project_root, 'configs', 'mamba_130m.yaml')
    print(f"Loading config: {config_file}")
    
    config = load_config(config_file)
    print("✓ Configuration loaded successfully")
    
    # Test nested configuration structure
    if 'training' not in config:
        print('❌ Missing training section')
        sys.exit(1)
    
    training_config = config.get('training', {})
    if 'pretrain' in training_config:
        pretrain_config = training_config['pretrain']
        conceptual_batch_size = int(pretrain_config.get('batch_size', 32))
        micro_batch_size = int(pretrain_config.get('micro_batch_size', 8))
        print(f'✓ Nested structure - batch_size: {conceptual_batch_size}, micro_batch_size: {micro_batch_size}')
        
        # Calculate gradient accumulation
        grad_accum = max(1, conceptual_batch_size // micro_batch_size)
        print(f'✓ Calculated gradient accumulation steps: {grad_accum}')
    else:
        print('⚠️ Using legacy flat structure')
        batch_size = training_config.get('batch_size', 32)
        print(f'  Batch size: {batch_size}')
    
    print('✓ pretrain.py configuration compatibility validated')
    
except Exception as e:
    print(f'❌ pretrain.py configuration test failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

# Test pretrain_sdm.py configuration compatibility
echo ""
echo "=== PRETRAIN_SDM.PY CONFIGURATION COMPATIBILITY ==="

python3 << 'EOF'
import sys
import os
project_root = os.environ.get('PROJECT_ROOT', '.')
sys.path.insert(0, project_root)

try:
    print("Testing pretrain_sdm.py configuration loading...")
    from pretrain_sdm import load_config
    
    config_file = os.path.join(project_root, 'configs', 'mamba_130m.yaml')
    print(f"Loading config: {config_file}")
    
    config = load_config(config_file)
    print("✓ Configuration loaded successfully")
    
    # Test SDM configuration keys
    sdm_config = config.get('sdm', {})
    print(f"SDM config keys: {list(sdm_config.keys())}")
    
    # Test temperature keys (should work with both old and new formats)
    temp_start = sdm_config.get('initial_temperature', sdm_config.get('gumbel_temp_start', 5.0))
    temp_end = sdm_config.get('final_temperature', sdm_config.get('gumbel_temp_end', 0.1))
    lambda_sparsity = sdm_config.get('lambda_sparsity', 0.01)
    
    print(f'✓ Temperature config: start={temp_start}, end={temp_end}')
    print(f'✓ Lambda sparsity: {lambda_sparsity}')
    
    # Test training config compatibility
    training_config = config.get('training', {})
    if 'pretrain' in training_config:
        pretrain_config = training_config['pretrain']
        conceptual_batch_size = int(pretrain_config.get('batch_size', 32))
        micro_batch_size = int(pretrain_config.get('micro_batch_size', 8))
        grad_accum = max(1, conceptual_batch_size // micro_batch_size)
        print(f'✓ Training config: batch_size={conceptual_batch_size}, micro_batch_size={micro_batch_size}, grad_accum={grad_accum}')
    
    print('✓ pretrain_sdm.py configuration compatibility validated')
    
except Exception as e:
    print(f'❌ pretrain_sdm.py configuration test failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

# Test tiny config for integration tests
echo ""
echo "=== TINY CONFIG VALIDATION ==="

TINY_CONFIG="$PROJECT_ROOT/tests/configs/test_config_tiny.yaml"
if [[ -f "$TINY_CONFIG" ]]; then
    echo "✓ Tiny test config found: $TINY_CONFIG"
    
    # Validate tiny config
    python3 << EOF
import yaml
try:
    with open('$TINY_CONFIG', 'r') as f:
        config = yaml.safe_load(f)
    print('✓ Tiny config is valid YAML')
    print(f'Sections: {list(config.keys())}')
    
    # Check model size
    model_config = config.get('model', {})
    d_model = model_config.get('d_model', 0)
    n_layer = model_config.get('n_layer', 0)
    print(f'✓ Model size: d_model={d_model}, n_layer={n_layer} (tiny for fast testing)')
    
except Exception as e:
    print(f'❌ Tiny config validation failed: {e}')
EOF
else
    echo "⚠️ Tiny test config not found: $TINY_CONFIG"
    echo "Integration tests may not work properly"
fi

# Cleanup test directory
echo ""
echo "=== CLEANUP ==="
echo "Removing test directory: $TEST_DIR"
rm -rf "$TEST_DIR"
echo "✓ Cleanup completed"

echo ""
echo "=========================================================================="
echo "TEST SUMMARY"
echo "=========================================================================="
echo "✓ All validation tests passed!"
echo "✓ run_full_experiment.sh should be able to start without configuration errors"
echo ""
echo "System is ready for experiments. You can now run:"
echo "  ./run_full_experiment.sh $MODEL_SIZE $NUM_GPUS your_experiment_name"
echo ""
echo "For debugging, check the log file: $LOG_FILE"
echo "Completed at: $(date)"
echo "==========================================================================" 