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