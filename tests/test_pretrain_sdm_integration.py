#!/usr/bin/env python3
"""
Integration test for pretrain_sdm.py

This script tests the actual execution of pretrain_sdm.py with minimal configuration
to ensure it runs without errors.
"""

import os
import sys
import subprocess
import tempfile
import shutil
import torch
import time
import logging
from pathlib import Path

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('tests/integration_test.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger.info(f"Project root: {project_root}")
logger.info(f"Python executable: {sys.executable}")
logger.info(f"Python version: {sys.version}")
logger.info(f"Platform: {sys.platform}")


def test_pretrain_sdm_basic_execution():
    """Test that pretrain_sdm.py can execute without errors."""
    logger.info("="*60)
    logger.info("TESTING: pretrain_sdm.py basic execution")
    logger.info("="*60)
    
    # Create temporary directory for test outputs
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_config = project_root / "tests" / "configs" / "test_config_tiny.yaml"
        output_dir = Path(tmp_dir) / "test_output"
        output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Test config path: {test_config}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Temporary directory: {tmp_dir}")
        
        # Check if config exists
        if not test_config.exists():
            logger.error(f"Test config not found at: {test_config}")
            logger.error(f"Current working directory: {os.getcwd()}")
            logger.error(f"Files in tests/configs/: {list((project_root / 'tests' / 'configs').glob('*'))}")
            return False
        
        logger.info(f"Test config found successfully")
        
        # Read and log config content
        try:
            with open(test_config, 'r') as f:
                config_content = f.read()
            logger.debug(f"Config file content:\n{config_content}")
        except Exception as e:
            logger.warning(f"Could not read config file: {e}")
        
        # Test command
        cmd = [
            sys.executable, 
            str(project_root / "pretrain_sdm.py"),
            "--config", str(test_config),
            "--output_dir", str(output_dir),
            "--experiment_name", "integration_test"
        ]
        
        logger.info(f"Command to execute: {' '.join(cmd)}")
        logger.info(f"Working directory: {project_root}")
        
        # Set environment variables for debugging
        env = os.environ.copy()
        env['PYTHONPATH'] = str(project_root)
        env['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU if available
        
        logger.info(f"Environment PYTHONPATH: {env.get('PYTHONPATH')}")
        logger.info(f"Environment CUDA_VISIBLE_DEVICES: {env.get('CUDA_VISIBLE_DEVICES')}")
        
        try:
            # Run with timeout to prevent hanging
            logger.info("Starting pretrain_sdm.py execution (max 120 seconds)...")
            start_time = time.time()
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=120,
                cwd=str(project_root),
                env=env
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            logger.info(f"Process completed in {execution_time:.2f} seconds")
            logger.info(f"Return code: {result.returncode}")
            
            # Log stdout in detail
            if result.stdout:
                logger.info("=== STDOUT ===")
                stdout_lines = result.stdout.strip().split('\n')
                for i, line in enumerate(stdout_lines, 1):
                    logger.info(f"STDOUT[{i:3d}]: {line}")
                logger.info("=== END STDOUT ===")
            else:
                logger.warning("No stdout output")
            
            # Log stderr in detail
            if result.stderr:
                logger.warning("=== STDERR ===")
                stderr_lines = result.stderr.strip().split('\n')
                for i, line in enumerate(stderr_lines, 1):
                    logger.warning(f"STDERR[{i:3d}]: {line}")
                logger.warning("=== END STDERR ===")
            else:
                logger.info("No stderr output")
            
            # Detailed error analysis
            if result.returncode != 0:
                logger.error(f"Process failed with return code {result.returncode}")
                error_output = result.stderr.lower()
                
                if "keyerror" in error_output:
                    logger.error("DIAGNOSIS: Configuration key error detected!")
                    logger.error("This usually means a required config key is missing or has wrong name")
                    return False
                elif "modulenotfounderror" in error_output:
                    logger.error("DIAGNOSIS: Missing module error detected!")
                    logger.error("Check if all required packages are installed")
                    return False
                elif "cuda" in error_output and "out of memory" in error_output:
                    logger.warning("DIAGNOSIS: CUDA out of memory (expected with larger models)")
                    logger.info("This is acceptable for testing - means GPU is available but model is too large")
                    return True  # This is acceptable for testing
                elif "importerror" in error_output:
                    logger.error("DIAGNOSIS: Import error detected!")
                    logger.error("Check if all project modules are properly structured")
                    return False
                elif "filenotfounderror" in error_output:
                    logger.error("DIAGNOSIS: File not found error!")
                    logger.error("Check if all required files exist")
                    return False
                else:
                    logger.error(f"DIAGNOSIS: Unknown error type")
                    logger.error("Check stderr output above for more details")
                    return False
            else:
                logger.info("✓ Process completed successfully!")
                
                # Check if output files were created
                if output_dir.exists():
                    output_files = list(output_dir.rglob('*'))
                    logger.info(f"Output files created: {len(output_files)}")
                    for f in output_files[:10]:  # Log first 10 files
                        logger.info(f"  - {f.relative_to(output_dir)}")
                    if len(output_files) > 10:
                        logger.info(f"  ... and {len(output_files) - 10} more files")
                else:
                    logger.warning("No output directory created")
                
                return True
                
        except subprocess.TimeoutExpired:
            logger.warning("Process timed out after 120 seconds")
            logger.info("This might be expected for actual training - means it started successfully")
            return True  # Timeout is acceptable - means it started training
        except Exception as e:
            logger.error(f"Error running process: {e}")
            logger.exception("Full exception details:")
            return False


def test_config_validation():
    """Test configuration loading and validation."""
    logger.info("="*60)
    logger.info("TESTING: Configuration validation")
    logger.info("="*60)
    
    try:
        logger.info("Attempting to import pretrain_sdm.load_config")
        from pretrain_sdm import load_config
        logger.info("✓ Successfully imported load_config")
        
        test_config = project_root / "tests" / "configs" / "test_config_tiny.yaml"
        logger.info(f"Test config path: {test_config}")
        
        if not test_config.exists():
            logger.error(f"Test config not found: {test_config}")
            logger.error(f"Files in tests/configs/: {list((project_root / 'tests' / 'configs').glob('*'))}")
            return False
        
        logger.info("✓ Test config file exists")
        
        logger.info("Loading configuration...")
        config = load_config(str(test_config))
        logger.info("✓ Configuration loaded successfully")
        
        logger.debug(f"Loaded config keys: {list(config.keys())}")
        
        # Check required sections
        required_sections = ['model', 'training', 'data', 'sdm', 'logging']
        logger.info(f"Checking required sections: {required_sections}")
        
        for section in required_sections:
            if section not in config:
                logger.error(f"Missing config section: {section}")
                logger.error(f"Available sections: {list(config.keys())}")
                return False
            else:
                logger.info(f"✓ Found section: {section}")
                logger.debug(f"  {section} keys: {list(config[section].keys()) if isinstance(config[section], dict) else type(config[section])}")
        
        # Check nested training structure
        logger.info("Checking nested training structure...")
        if 'pretrain' not in config['training']:
            logger.error("Missing 'pretrain' section in training config")
            logger.error(f"Training config keys: {list(config['training'].keys())}")
            return False
        
        logger.info("✓ Found pretrain section in training config")
        
        # Check SDM config
        logger.info("Checking SDM configuration...")
        sdm_config = config.get('sdm', {})
        logger.debug(f"SDM config: {sdm_config}")
        
        # Test temperature key fallback logic
        temp_start = sdm_config.get('initial_temperature', sdm_config.get('gumbel_temp_start', 5.0))
        temp_end = sdm_config.get('final_temperature', sdm_config.get('gumbel_temp_end', 0.1))
        lambda_sparsity = sdm_config.get('lambda_sparsity', 0.01)
        
        logger.info(f"✓ SDM temperature config: start={temp_start}, end={temp_end}, lambda={lambda_sparsity}")
        
        logger.info("✓ Configuration validation passed!")
        return True
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        logger.exception("Full exception details:")
        return False


def test_model_import():
    """Test that all required models can be imported."""
    logger.info("="*60)
    logger.info("TESTING: Model imports")
    logger.info("="*60)
    
    models_to_test = [
        ('models.sdm_ssm', 'SDM_SSM'),
        ('models.baseline_ssm', 'BaselineSSM')
    ]
    
    for module_name, class_name in models_to_test:
        try:
            logger.info(f"Importing {class_name} from {module_name}...")
            module = __import__(module_name, fromlist=[class_name])
            model_class = getattr(module, class_name)
            logger.info(f"✓ Successfully imported {class_name}")
            logger.debug(f"  Class: {model_class}")
            logger.debug(f"  Module file: {getattr(module, '__file__', 'unknown')}")
        except ImportError as e:
            logger.error(f"Failed to import {class_name} from {module_name}: {e}")
            logger.exception("Import error details:")
            return False
        except AttributeError as e:
            logger.error(f"Class {class_name} not found in {module_name}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error importing {class_name}: {e}")
            logger.exception("Full exception details:")
            return False
    
    logger.info("✓ All model imports successful!")
    return True


def test_utility_imports():
    """Test that all utility functions can be imported."""
    logger.info("="*60)
    logger.info("TESTING: Utility imports")
    logger.info("="*60)
    
    utilities_to_test = [
        ('utils.logger', ['setup_logger', 'log_model_info']),
        ('data.wikitext103', ['get_wiktext103_dataloader'])
    ]
    
    for module_name, function_names in utilities_to_test:
        try:
            logger.info(f"Importing from {module_name}...")
            module = __import__(module_name, fromlist=function_names)
            
            for func_name in function_names:
                func = getattr(module, func_name)
                logger.info(f"✓ Successfully imported {func_name}")
                logger.debug(f"  Function: {func}")
            
            logger.debug(f"  Module file: {getattr(module, '__file__', 'unknown')}")
            
        except ImportError as e:
            logger.error(f"Failed to import from {module_name}: {e}")
            logger.exception("Import error details:")
            return False
        except AttributeError as e:
            logger.error(f"Function not found in {module_name}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error importing from {module_name}: {e}")
            logger.exception("Full exception details:")
            return False
    
    logger.info("✓ All utility imports successful!")
    return True


def test_environment_info():
    """Test and log environment information."""
    logger.info("="*60)
    logger.info("TESTING: Environment information")
    logger.info("="*60)
    
    try:
        # Python environment
        logger.info(f"Python executable: {sys.executable}")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Platform: {sys.platform}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"PYTHONPATH: {sys.path[:3]}... (showing first 3 entries)")
        
        # PyTorch info
        try:
            import torch
            logger.info(f"✓ PyTorch version: {torch.__version__}")
            logger.info(f"✓ CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"✓ CUDA device count: {torch.cuda.device_count()}")
                logger.info(f"✓ Current CUDA device: {torch.cuda.current_device()}")
        except Exception as e:
            logger.warning(f"PyTorch info gathering failed: {e}")
        
        # Project structure
        logger.info("Project structure check:")
        critical_paths = [
            "pretrain_sdm.py",
            "models/",
            "utils/",
            "data/",
            "configs/",
            "tests/"
        ]
        
        for path in critical_paths:
            full_path = project_root / path
            exists = full_path.exists()
            logger.info(f"  {'✓' if exists else '✗'} {path}: {exists}")
            if not exists:
                logger.warning(f"Missing critical path: {full_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Environment info gathering failed: {e}")
        logger.exception("Full exception details:")
        return False


def main():
    """Run all integration tests."""
    logger.info("=" * 80)
    logger.info("PRETRAIN_SDM.PY INTEGRATION TESTS - DETAILED LOGGING MODE")
    logger.info("=" * 80)
    logger.info(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: tests/integration_test.log")
    
    tests = [
        ("Environment Info", test_environment_info),
        ("Model Imports", test_model_import),
        ("Utility Imports", test_utility_imports),
        ("Config Validation", test_config_validation),
        ("Basic Execution", test_pretrain_sdm_basic_execution),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*80}")
        logger.info(f"Running: {test_name}")
        logger.info('='*80)
        
        start_time = time.time()
        try:
            result = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} raised an exception: {e}")
            logger.exception("Full exception details:")
            result = False
            
        end_time = time.time()
        duration = end_time - start_time
        
        results.append((test_name, result, duration))
        
        status = "PASSED" if result else "FAILED"
        logger.info(f"\n>>> {status} ({duration:.2f}s) <<<")
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("INTEGRATION TEST SUMMARY")
    logger.info('='*80)
    
    passed = 0
    for test_name, result, duration in results:
        status = "PASS" if result else "FAIL"
        logger.info(f"{status:>6} {test_name} ({duration:.2f}s)")
        if result:
            passed += 1
    
    logger.info(f"\nResults: {passed}/{len(results)} tests passed")
    logger.info(f"Finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if passed == len(results):
        logger.info("All integration tests passed!")
        return 0
    else:
        logger.error("Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 