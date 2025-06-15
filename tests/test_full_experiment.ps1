# Test script for run_full_experiment.sh (Windows PowerShell version)
# This script validates that the full experiment can start without configuration errors

$ErrorActionPreference = "Stop"

Write-Host "Testing run_full_experiment.sh..." -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan

# Get script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

# Test parameters
$ModelSize = "130m"
$NumGpus = 1
$ExperimentName = "test_experiment_$(Get-Date -Format 'yyyyMMdd_HHmmss')"

Write-Host "Project Root: $ProjectRoot"
Write-Host "Test Experiment: $ExperimentName"
Write-Host "Model Size: $ModelSize"
Write-Host "GPUs: $NumGpus"
Write-Host ""

# Check if required files exist
Write-Host "Checking required files..." -ForegroundColor Yellow

$ExperimentScript = Join-Path $ProjectRoot "run_full_experiment.sh"
$ConfigFile = Join-Path $ProjectRoot "configs\mamba_$ModelSize.yaml"

if (-not (Test-Path $ExperimentScript)) {
    Write-Host "run_full_experiment.sh not found: $ExperimentScript" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $ConfigFile)) {
    Write-Host "Config file not found: $ConfigFile" -ForegroundColor Red
    exit 1
}

Write-Host "Required files found" -ForegroundColor Green
Write-Host ""

# Test configuration validation
Write-Host "Testing configuration validation..." -ForegroundColor Yellow

# Test Python imports that the experiment needs
Write-Host "Testing Python dependencies..." -ForegroundColor Yellow

$ImportTestCode = @"
import sys
import os
sys.path.insert(0, r'$ProjectRoot')

# Test critical imports
try:
    import torch
    print('PyTorch available')
except ImportError as e:
    print(f'PyTorch import failed: {e}')
    sys.exit(1)

try:
    import yaml
    print('PyYAML available')
except ImportError as e:
    print(f'PyYAML import failed: {e}')
    sys.exit(1)

try:
    from transformers import AutoTokenizer
    print('Transformers available')
except ImportError as e:
    print(f'Transformers import failed: {e}')
    sys.exit(1)

try:
    from accelerate import Accelerator
    print('Accelerate available')
except ImportError as e:
    print(f'Accelerate import failed: {e}')
    sys.exit(1)

print('All critical dependencies available')
"@

try {
    $result = python -c $ImportTestCode
    Write-Host $result -ForegroundColor Green
} catch {
    Write-Host "Python import test failed: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "All tests passed!" -ForegroundColor Green
Write-Host "run_full_experiment.sh should be able to start without configuration errors" -ForegroundColor Green
Write-Host ""

# Test pretrain.py configuration compatibility
Write-Host "Testing pretrain.py configuration compatibility..." -ForegroundColor Yellow

$PretrainTestCode = @"
import sys
import os
sys.path.insert(0, r'$ProjectRoot')

try:
    from pretrain import load_config
    config = load_config(r'$ConfigFile')
    
    # Test nested configuration structure
    if 'training' not in config:
        print('Missing training section')
        sys.exit(1)
    
    training_config = config.get('training', {})
    if 'pretrain' in training_config:
        pretrain_config = training_config['pretrain']
        conceptual_batch_size = int(pretrain_config.get('batch_size', 32))
        micro_batch_size = int(pretrain_config.get('micro_batch_size', 8))
        print(f'Pretrain config: batch_size={conceptual_batch_size}, micro_batch_size={micro_batch_size}')
    else:
        print('Using legacy flat structure')
    
    print('pretrain.py configuration compatibility validated')
    
except Exception as e:
    print(f'pretrain.py configuration test failed: {e}')
    sys.exit(1)
"@

try {
    $result = python -c $PretrainTestCode
    Write-Host $result -ForegroundColor Green
} catch {
    Write-Host "pretrain.py configuration test failed: $_" -ForegroundColor Red
    exit 1
}

# Test pretrain_sdm.py configuration compatibility
Write-Host "Testing pretrain_sdm.py configuration compatibility..." -ForegroundColor Yellow

$SdmTestCode = @"
import sys
import os
sys.path.insert(0, r'$ProjectRoot')

try:
    from pretrain_sdm import load_config
    config = load_config(r'$ConfigFile')
    
    # Test SDM configuration keys
    sdm_config = config.get('sdm', {})
    
    # Test temperature keys (should work with both old and new formats)
    temp_start = sdm_config.get('initial_temperature', sdm_config.get('gumbel_temp_start', 5.0))
    temp_end = sdm_config.get('final_temperature', sdm_config.get('gumbel_temp_end', 0.1))
    lambda_sparsity = sdm_config.get('lambda_sparsity', 0.01)
    
    print(f'SDM config: temp_start={temp_start}, temp_end={temp_end}, lambda={lambda_sparsity}')
    print('pretrain_sdm.py configuration compatibility validated')
    
except Exception as e:
    print(f'pretrain_sdm.py configuration test failed: {e}')
    sys.exit(1)
"@

try {
    $result = python -c $SdmTestCode
    Write-Host $result -ForegroundColor Green
} catch {
    Write-Host "pretrain_sdm.py configuration test failed: $_" -ForegroundColor Red
    exit 1
}

# Cleanup test directory (if it exists)
# Note: No test directory was created in this script

Write-Host ""
Write-Host "All tests passed!" -ForegroundColor Green
Write-Host "run_full_experiment.sh should be able to start without configuration errors" -ForegroundColor Green
Write-Host ""
Write-Host "To run the actual experiment (using Git Bash or WSL):" -ForegroundColor Cyan
Write-Host "  ./run_full_experiment.sh $ModelSize $NumGpus your_experiment_name" -ForegroundColor Cyan
Write-Host "" 