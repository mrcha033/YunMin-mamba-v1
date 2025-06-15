#!/usr/bin/env python3
"""
Master test runner for the Hardware-Data-Parameter Co-Design Framework

This script runs all tests to ensure the system is working correctly:
1. Unit tests for pretrain_sdm.py
2. Integration tests for pretrain_sdm.py 
3. Full experiment validation tests
4. Configuration compatibility tests
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_command(cmd, description, timeout=300):
    """Run a command and return success status."""
    print(f"🧪 {description}")
    print(f"   Command: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    
    start_time = time.time()
    
    try:
        if isinstance(cmd, str):
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, 
                timeout=timeout, cwd=str(project_root)
            )
        else:
            result = subprocess.run(
                cmd, capture_output=True, text=True, 
                timeout=timeout, cwd=str(project_root)
            )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"   ✅ SUCCESS ({duration:.2f}s)")
            if result.stdout.strip():
                print("   Output (last 5 lines):")
                for line in result.stdout.strip().split('\n')[-5:]:
                    print(f"     {line}")
            return True, duration, result.stdout, result.stderr
        else:
            print(f"   ❌ FAILED ({duration:.2f}s)")
            print(f"   Return code: {result.returncode}")
            if result.stderr.strip():
                print("   Error output:")
                for line in result.stderr.strip().split('\n')[-10:]:
                    print(f"     {line}")
            return False, duration, result.stdout, result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"   ⏰ TIMEOUT ({timeout}s)")
        return False, timeout, "", "Process timed out"
    except Exception as e:
        print(f"   💥 ERROR: {e}")
        return False, 0, "", str(e)


def main():
    """Run all tests."""
    print("🚀 Hardware-Data-Parameter Co-Design Framework - Test Suite")
    print("=" * 70)
    print(f"Project Root: {project_root}")
    print(f"Python: {sys.executable}")
    print("")
    
    # Test definitions
    tests = [
        {
            "name": "Unit Tests (pytest)", 
            "cmd": [sys.executable, "-m", "pytest", "tests/test_pretrain_sdm.py", "-v"],
            "description": "Running unit tests with pytest",
            "timeout": 60
        },
        {
            "name": "Integration Tests", 
            "cmd": [sys.executable, "tests/test_pretrain_sdm_integration.py"],
            "description": "Running integration tests for pretrain_sdm.py",
            "timeout": 300
        },
        {
            "name": "Full Experiment Validation", 
            "cmd": "chmod +x tests/test_full_experiment.sh && tests/test_full_experiment.sh",
            "description": "Validating run_full_experiment.sh compatibility",
            "timeout": 120
        }
    ]
    
    results = []
    total_start_time = time.time()
    
    # Run each test
    for i, test in enumerate(tests, 1):
        print(f"\n{'-' * 50}")
        print(f"Test {i}/{len(tests)}: {test['name']}")
        print('-' * 50)
        
        success, duration, stdout, stderr = run_command(
            test["cmd"], 
            test["description"], 
            test.get("timeout", 300)
        )
        
        results.append({
            "name": test["name"],
            "success": success,
            "duration": duration,
            "stdout": stdout,
            "stderr": stderr
        })
    
    total_end_time = time.time()
    
    # Summary report
    print(f"\n{'=' * 70}")
    print("TEST SUMMARY REPORT")
    print('=' * 70)
    
    passed = sum(1 for r in results if r["success"])
    total = len(results)
    
    for result in results:
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        print(f"{status} {result['name']} ({result['duration']:.2f}s)")
    
    print(f"\nResults: {passed}/{total} tests passed")
    print(f"Total time: {total_end_time - total_start_time:.2f}s")
    
    # Detailed failure analysis
    failed_tests = [r for r in results if not r["success"]]
    if failed_tests:
        print(f"\n{'-' * 50}")
        print("FAILURE ANALYSIS")
        print('-' * 50)
        
        for test in failed_tests:
            print(f"\n❌ {test['name']}:")
            if test["stderr"]:
                print("   Error details:")
                for line in test["stderr"].split('\n')[-5:]:
                    if line.strip():
                        print(f"     {line}")
    
    # Recommendations
    print(f"\n{'-' * 50}")
    print("RECOMMENDATIONS")
    print('-' * 50)
    
    if passed == total:
        print("🎉 All tests passed! Your system is ready for experiments.")
        print("")
        print("Next steps:")
        print("  1. Run a small test experiment:")
        print("     ./run_full_experiment.sh 130m 1 test_run")
        print("")
        print("  2. Monitor the logs for any issues:")
        print("     tail -f experiments/test_run/logs/experiment.log")
        
    else:
        print("💥 Some tests failed. Please address the issues above.")
        print("")
        print("Common solutions:")
        print("  - Install missing dependencies: pip install -r requirements.txt")
        print("  - Check Python path and imports")
        print("  - Verify configuration files are valid")
        print("  - Ensure CUDA is available if using GPU")
    
    print("")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main()) 