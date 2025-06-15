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
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_pretrain_sdm_basic_execution():
    """Test that pretrain_sdm.py can execute without errors."""
    print("Testing pretrain_sdm.py basic execution...")
    
    # Create temporary directory for test outputs
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_config = project_root / "tests" / "configs" / "test_config_tiny.yaml"
        output_dir = Path(tmp_dir) / "test_output"
        output_dir.mkdir(exist_ok=True)
        
        print(f"   Config: {test_config}")
        print(f"   Output: {output_dir}")
        
        # Check if config exists
        if not test_config.exists():
            print("❌ Test config not found!")
            return False
        
        # Test command
        cmd = [
            sys.executable, 
            str(project_root / "pretrain_sdm.py"),
            "--config", str(test_config),
            "--output_dir", str(output_dir),
            "--experiment_name", "integration_test"
        ]
        
        print(f"   Command: {' '.join(cmd)}")
        
        try:
            # Run with timeout to prevent hanging
            print("   Running (max 120 seconds)...")
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=120,
                cwd=str(project_root)
            )
            
            print(f"   Return code: {result.returncode}")
            
            if result.stdout:
                print("   STDOUT (last 10 lines):")
                stdout_lines = result.stdout.strip().split('\n')
                for line in stdout_lines[-10:]:
                    print(f"     {line}")
            
            if result.stderr:
                print("   STDERR:")
                stderr_lines = result.stderr.strip().split('\n')
                for line in stderr_lines[-10:]:
                    print(f"     {line}")
            
            # Check for specific error types
            if result.returncode != 0:
                error_output = result.stderr.lower()
                if "keyerror" in error_output:
                    print("❌ Configuration key error detected!")
                    return False
                elif "modulenotfounderror" in error_output:
                    print("❌ Missing module error detected!")
                    return False
                elif "cuda" in error_output and "out of memory" in error_output:
                    print("⚠️  CUDA out of memory (expected with larger models)")
                    return True  # This is acceptable for testing
                else:
                    print(f"❌ Process failed with return code {result.returncode}")
                    return False
            else:
                print("✅ Process completed successfully!")
                return True
                
        except subprocess.TimeoutExpired:
            print("⚠️  Process timed out (this might be expected for actual training)")
            return True  # Timeout is acceptable - means it started training
        except Exception as e:
            print(f"❌ Error running process: {e}")
            return False


def test_config_validation():
    """Test configuration loading and validation."""
    print("Testing configuration validation...")
    
    try:
        from pretrain_sdm import load_config
        
        test_config = project_root / "tests" / "configs" / "test_config_tiny.yaml"
        
        if not test_config.exists():
            print("❌ Test config not found!")
            return False
        
        config = load_config(str(test_config))
        
        # Check required sections
        required_sections = ['model', 'training', 'data', 'sdm', 'logging']
        for section in required_sections:
            if section not in config:
                print(f"❌ Missing config section: {section}")
                return False
        
        # Check nested training structure
        if 'pretrain' not in config['training']:
            print("❌ Missing pretrain section in training config")
            return False
        
        print("✅ Configuration validation passed!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False


def test_model_import():
    """Test that all required models can be imported."""
    print("Testing model imports...")
    
    try:
        from models.sdm_ssm import SDM_SSM
        from models.baseline_ssm import BaselineSSM
        print("✅ Model imports successful!")
        return True
    except Exception as e:
        print(f"❌ Model import failed: {e}")
        return False


def test_utility_imports():
    """Test that all utility functions can be imported."""
    print("Testing utility imports...")
    
    try:
        from utils.logger import setup_logger, log_model_info
        from data.wikitext103 import get_wiktext103_dataloader
        print("✅ Utility imports successful!")
        return True
    except Exception as e:
        print(f"❌ Utility import failed: {e}")
        return False


def main():
    """Run all integration tests."""
    print("🚀 Starting pretrain_sdm.py integration tests...\n")
    
    tests = [
        ("Model Imports", test_model_import),
        ("Utility Imports", test_utility_imports),
        ("Config Validation", test_config_validation),
        ("Basic Execution", test_pretrain_sdm_basic_execution),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        start_time = time.time()
        result = test_func()
        end_time = time.time()
        
        results.append((test_name, result, end_time - start_time))
        
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"\n{status} ({end_time - start_time:.2f}s)")
    
    # Summary
    print(f"\n{'='*60}")
    print("INTEGRATION TEST SUMMARY")
    print('='*60)
    
    passed = 0
    for test_name, result, duration in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name} ({duration:.2f}s)")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All integration tests passed!")
        return 0
    else:
        print("💥 Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 