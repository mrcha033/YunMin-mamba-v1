# Test Suite for Hardware-Data-Parameter Co-Design Framework

This directory contains comprehensive tests for the co-design framework, with special focus on `pretrain_sdm.py` and the full experiment pipeline.

## Test Structure

```
tests/
├── __init__.py                      # Test package initialization
├── README.md                        # This file
├── run_all_tests.py                # Master test runner
├── test_pretrain_sdm.py            # Unit tests for pretrain_sdm.py
├── test_pretrain_sdm_integration.py # Integration tests
├── test_full_experiment.sh         # Full experiment validation
└── configs/
    └── test_config_tiny.yaml       # Minimal config for fast testing
```

## Quick Start

### Run All Tests
```bash
# Run comprehensive test suite
python tests/run_all_tests.py
```

### Run Individual Test Categories

#### 1. Unit Tests (pytest)
```bash
# Run unit tests with pytest
python -m pytest tests/test_pretrain_sdm.py -v
```

#### 2. Integration Tests
```bash
# Run integration tests for pretrain_sdm.py
python tests/test_pretrain_sdm_integration.py
```

#### 3. Full Experiment Validation
```bash
# Test run_full_experiment.sh compatibility
bash tests/test_full_experiment.sh
```

## Test Categories

### 1. Unit Tests (`test_pretrain_sdm.py`)

Tests individual components of `pretrain_sdm.py`:

- **Configuration Loading**: Validates YAML config parsing and structure
- **SDM Components**: Tests sparsity loss calculation and temperature scheduling
- **Configuration Compatibility**: Ensures both old and new config formats work

Key test classes:
- `TestConfigLoading`: Configuration file validation
- `TestSDMComponents`: SDM-specific functionality
- `TestConfigCompatibility`: Cross-format compatibility

### 2. Integration Tests (`test_pretrain_sdm_integration.py`)

Tests the actual execution of `pretrain_sdm.py`:

- **Model Imports**: Verifies all required models can be imported
- **Utility Imports**: Checks logger and data utilities
- **Configuration Validation**: End-to-end config loading
- **Basic Execution**: Runs `pretrain_sdm.py` with minimal config

### 3. Full Experiment Validation (`test_full_experiment.sh`)

Validates the complete experiment pipeline:

- **File Existence**: Checks all required files are present
- **Configuration Validation**: YAML syntax and structure validation  
- **Python Dependencies**: Verifies all required packages are installed
- **Directory Structure**: Tests experiment directory creation
- **Script Compatibility**: Validates both `pretrain.py` and `pretrain_sdm.py` configs

## Test Configuration

### Minimal Test Config (`configs/test_config_tiny.yaml`)

Ultra-small configuration for fast testing:
- Model: 64-dim, 2 layers, 1000 vocab
- Training: 4 batch size, 1 epoch
- Data: 128 sequence length, no multiprocessing
- SDM: Standard sparsity settings
- Logging: Disabled wandb for tests

## Expected Test Results

### ✅ All Tests Pass
When all tests pass, you'll see:
```
🎉 All tests passed! Your system is ready for experiments.

Next steps:
  1. Run a small test experiment:
     ./run_full_experiment.sh 130m 1 test_run
  
  2. Monitor the logs for any issues:
     tail -f experiments/test_run/logs/experiment.log
```

### ❌ Common Issues and Solutions

#### Configuration Errors
```
❌ Configuration key error detected!
```
**Solution**: Update config files to use consistent nested structure

#### Missing Dependencies
```
❌ PyTorch import failed
❌ Transformers import failed
```
**Solution**: Install requirements
```bash
pip install -r requirements.txt
```

#### Module Import Errors
```
❌ BaselineSSM import failed
❌ SDM_SSM import failed
```
**Solution**: Check PYTHONPATH and ensure all model files exist

#### CUDA Memory Issues
```
⚠️ CUDA out of memory (expected with larger models)
```
**Solution**: This is acceptable for testing - means GPU is available but model is too large

## Troubleshooting

### Test Execution Issues

1. **Permission Denied**:
   ```bash
   chmod +x tests/test_full_experiment.sh
   chmod +x tests/test_pretrain_sdm_integration.py
   ```

2. **Python Path Issues**:
   ```bash
   export PYTHONPATH="${PYTHONPATH}:/path/to/YunMin-mamba-v1"
   ```

3. **Missing pytest**:
   ```bash
   pip install pytest
   ```

### Configuration Issues

1. **KeyError for config keys**: Usually means config structure mismatch
   - Check if using nested (`training.pretrain`) vs flat structure
   - Verify SDM parameter names match expectations

2. **YAML Syntax Errors**: Validate YAML files
   ```bash
   python -c "import yaml; yaml.safe_load(open('config.yaml'))"
   ```

## Adding New Tests

### Adding Unit Tests

1. Add test methods to existing classes in `test_pretrain_sdm.py`
2. Follow naming convention: `test_<functionality>`
3. Use pytest fixtures for common setup

### Adding Integration Tests

1. Add test functions to `test_pretrain_sdm_integration.py`
2. Use subprocess for external command testing
3. Include timeout and error handling

### Test Configuration

For new test configs:
1. Place in `tests/configs/`
2. Use minimal model sizes for speed
3. Disable wandb logging
4. Set short training loops

## Continuous Integration

These tests are designed to be run in CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Run Test Suite
  run: python tests/run_all_tests.py
```

## Performance Benchmarks

Typical test execution times:
- Unit Tests: ~30-60 seconds
- Integration Tests: ~2-5 minutes  
- Full Experiment Validation: ~1-2 minutes
- **Total**: ~5-8 minutes for complete test suite

## Test Coverage

Current test coverage:
- ✅ Configuration loading and validation
- ✅ Model imports and initialization
- ✅ SDM component functionality
- ✅ Training script compatibility
- ✅ Full experiment pipeline validation
- ✅ Cross-platform compatibility (Linux/Windows)

Future coverage (TODO):
- [ ] Actual training loop execution
- [ ] Checkpoint loading/saving
- [ ] Multi-GPU testing
- [ ] Performance regression tests 

# Testing Infrastructure for YunMin-mamba-v1

This directory contains comprehensive tests to validate the `run_full_experiment.sh` script and ensure that the project can execute without configuration errors.

## Overview

The test suite is designed to catch configuration issues early and provide detailed debugging information for troubleshooting. It includes unit tests, integration tests, and validation scripts for multiple platforms.

## Test Files

### Core Test Files

- **`test_pretrain_sdm.py`** - Unit tests for individual components using pytest
- **`test_pretrain_sdm_integration.py`** - Integration tests with detailed logging and error analysis
- **`run_all_tests.py`** - Master test runner that orchestrates all tests

### Platform-Specific Scripts

- **`test_full_experiment.sh`** - Comprehensive validation script for Linux/Ubuntu with detailed logging
- **`test_full_experiment.ps1`** - PowerShell equivalent for Windows systems

### Configuration Files

- **`configs/test_config_tiny.yaml`** - Minimal configuration for fast testing
- **`__init__.py`** - Makes the directory a Python package

## Detailed Logging Features

### Ubuntu/Linux Logging (`test_full_experiment.sh`)

The bash script provides extensive logging with:

#### System Information Gathering
- Operating system details (`uname -a`)
- CPU information (`lscpu`)
- Memory usage (`free -h`)
- Disk space (`df -h`)
- GPU information (`nvidia-smi`)
- Python environment details
- Virtual environment status
- PYTHONPATH configuration

#### Configuration Validation
- YAML syntax validation
- Required configuration sections check
- Nested vs flat structure detection
- SDM parameter validation
- Temperature key compatibility check
- Batch size and gradient accumulation validation

#### Dependencies Testing
- Critical imports (torch, transformers, accelerate, yaml)
- Project-specific imports (models, utils, data modules)
- Import error categorization and debugging
- Module version information
- File path validation

#### Execution Environment
- Script permissions verification
- Directory structure validation
- File existence checks
- Project structure analysis
- Python path configuration

#### Log Files
- **`tests/experiment_test.log`** - Complete execution log with timestamps
- All output is simultaneously displayed and logged
- Error details with full stack traces
- Performance timing information

### Integration Test Logging (`test_pretrain_sdm_integration.py`)

Provides detailed logging for:

#### Environment Analysis
- Python version and executable path
- CUDA availability and configuration
- PyTorch device information
- Project structure validation
- Configuration file analysis

#### Error Categorization
```python
ERROR_CATEGORIES = {
    'KeyError': 'Configuration key missing',
    'ModuleNotFoundError': 'Missing Python dependency',
    'ImportError': 'Module import failure',
    'FileNotFoundError': 'Missing required file',
    'RuntimeError': 'Runtime execution error',
    'torch.cuda.OutOfMemoryError': 'GPU memory insufficient',
    'yaml.scanner.ScannerError': 'YAML syntax error',
    'AttributeError': 'Missing class/function attribute'
}
```

#### Configuration Testing
- Dynamic gradient accumulation calculation
- Temperature parameter compatibility
- Nested configuration structure support
- SDM parameter validation
- Model architecture validation

#### Execution Monitoring
- Step-by-step execution tracking
- Performance metrics collection
- Memory usage monitoring
- Error recovery and diagnosis

## Running Tests

### Quick Test (Recommended)
```bash
# Run all tests with the master runner
python tests/run_all_tests.py
```

### Individual Tests

#### Unit Tests
```bash
# Run pytest unit tests
python -m pytest tests/test_pretrain_sdm.py -v
```

#### Integration Tests
```bash
# Run detailed integration tests
python tests/test_pretrain_sdm_integration.py
```

#### Full System Validation

**Ubuntu/Linux:**
```bash
# Comprehensive system validation with detailed logging
chmod +x tests/test_full_experiment.sh
./tests/test_full_experiment.sh
```

**Windows:**
```powershell
# PowerShell validation script
.\tests\test_full_experiment.ps1
```

## Understanding Test Output

### Success Indicators
- ✓ - Test passed successfully
- ⚠️ - Warning (non-critical issue)
- ❌ - Error (critical failure)

### Log Analysis

#### System Readiness
Look for these sections in the logs:
- **SYSTEM INFORMATION** - Hardware and OS details
- **PYTHON ENVIRONMENT** - Python setup validation
- **CONFIGURATION VALIDATION** - Config file syntax and structure
- **DEPENDENCIES TEST** - All required packages available
- **COMPATIBILITY** - pretrain_sdm.py configuration compatibility

#### Common Issues and Solutions

**Configuration KeyError:**
```
❌ KeyError: 'gradient_accumulation_steps'
```
- **Solution**: Tests now calculate this dynamically from batch_size/micro_batch_size
- **Check**: Ensure both `batch_size` and `micro_batch_size` are specified

**Import Errors:**
```
❌ ModuleNotFoundError: No module named 'transformers'
```
- **Solution**: Install missing dependencies
- **Command**: `pip install torch transformers accelerate pyyaml`

**Temperature Configuration:**
```
⚠️ Temperature keys not found, will use defaults
```
- **Solution**: Tests support both old and new temperature key formats
- **Check**: Either `initial_temperature`/`final_temperature` or `gumbel_temp_start`/`gumbel_temp_end`

**CUDA Issues:**
```
❌ torch.cuda.OutOfMemoryError: CUDA out of memory
```
- **Solution**: Use smaller batch sizes in test configuration
- **Check**: GPU memory availability with `nvidia-smi`

## Test Configuration

### Tiny Test Configuration
The `configs/test_config_tiny.yaml` provides minimal settings for fast testing:
- Small model dimensions (d_model=64, n_layer=2)
- Reduced batch sizes
- Shorter training sequences
- Minimal logging

### Configuration Compatibility
Tests validate compatibility between:
- Main configuration files (`mamba_130m.yaml`, `mamba_370m.yaml`)
- SDM-specific configurations
- Nested vs flat configuration structures
- Legacy vs modern parameter naming

## Debugging Guide

### Log File Analysis

**1. Check System Information**
```bash
# View system details from log
grep -A 10 "=== SYSTEM INFORMATION ===" tests/experiment_test.log
```

**2. Analyze Configuration Issues**
```bash
# Check configuration validation
grep -A 20 "=== CONFIGURATION VALIDATION ===" tests/experiment_test.log
```

**3. Review Dependencies**
```bash
# Check dependency status
grep -A 30 "=== PYTHON DEPENDENCIES TEST ===" tests/experiment_test.log
```

**4. Examine Import Failures**
```bash
# Find import errors
grep "❌.*import" tests/experiment_test.log
```

### Performance Monitoring

The integration tests provide timing information:
```
Environment setup: 0.123s
Configuration loading: 0.045s
Model imports: 0.234s
Total execution: 0.402s
```

### Memory Usage Tracking

Monitor memory usage during testing:
```
Memory before: 2.1GB used, 13.9GB available
Memory after: 2.3GB used, 13.7GB available
Memory delta: +0.2GB
```

## Continuous Integration

### Pre-commit Testing
Run before committing changes:
```bash
# Quick validation
python tests/run_all_tests.py

# Full system test
./tests/test_full_experiment.sh
```

### Ubuntu Deployment
For Ubuntu servers, the test suite provides:
- Complete environment validation
- Dependencies verification
- Configuration compatibility check
- GPU setup validation
- Performance benchmarking

### Docker Testing
Tests can be run in Docker containers:
```dockerfile
# Add to Dockerfile
COPY tests/ /workspace/tests/
RUN chmod +x /workspace/tests/test_full_experiment.sh
RUN /workspace/tests/test_full_experiment.sh
```

## Contributing

When adding new features:
1. Update relevant test files
2. Add configuration validation
3. Include error handling tests
4. Update documentation
5. Run full test suite before committing

## Troubleshooting

### Common Test Failures

**1. Permission Denied**
```bash
chmod +x tests/test_full_experiment.sh
```

**2. Python Path Issues**
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**3. Missing Test Dependencies**
```bash
pip install pytest pyyaml
```

**4. GPU Test Failures**
- Tests are designed to work without GPU
- GPU tests are optional and logged separately
- Check CUDA installation if GPU tests fail

### Getting Help

1. **Check log files** - Always examine the detailed logs first
2. **Run individual tests** - Isolate specific component failures
3. **Verify environment** - Ensure all dependencies are installed
4. **Test with tiny config** - Use minimal configuration for debugging
5. **Check project structure** - Verify all required files exist

The test suite is designed to provide comprehensive validation and detailed debugging information to ensure reliable deployment of the YunMin-mamba-v1 project. 