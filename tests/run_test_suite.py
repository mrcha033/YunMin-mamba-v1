#!/usr/bin/env python3
"""
Comprehensive Test Suite Runner for YunMin-mamba-v1
Provides different test modes with detailed logging and error analysis
"""

import os
import sys
import time
import subprocess
import platform
import argparse
import json
from pathlib import Path
from datetime import datetime

def setup_logging():
    """Setup comprehensive logging for test execution."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"tests/test_suite_{timestamp}.log"
    
    # Create tests directory if it doesn't exist
    os.makedirs("tests", exist_ok=True)
    
    return log_file

def log_system_info(log_file):
    """Log comprehensive system information."""
    with open(log_file, "a") as f:
        f.write("=" * 80 + "\n")
        f.write("TEST SUITE EXECUTION LOG\n")
        f.write("=" * 80 + "\n")
        f.write(f"Started at: {datetime.now()}\n")
        f.write(f"Platform: {platform.platform()}\n")
        f.write(f"Python: {sys.version}\n")
        f.write(f"Working directory: {os.getcwd()}\n")
        f.write(f"Log file: {log_file}\n")
        f.write("\n")

def run_command(cmd, log_file, description="Command"):
    """Run a command and log its output."""
    print(f"Running: {description}")
    
    with open(log_file, "a") as f:
        f.write(f"\n=== {description} ===\n")
        f.write(f"Command: {cmd}\n")
        f.write(f"Started: {datetime.now()}\n")
        f.write("-" * 40 + "\n")
    
    start_time = time.time()
    
    try:
        if isinstance(cmd, str):
            # Shell command
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
        else:
            # List command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
        
        duration = time.time() - start_time
        
        with open(log_file, "a") as f:
            f.write(f"Exit code: {result.returncode}\n")
            f.write(f"Duration: {duration:.2f}s\n")
            f.write(f"STDOUT:\n{result.stdout}\n")
            if result.stderr:
                f.write(f"STDERR:\n{result.stderr}\n")
            f.write("-" * 40 + "\n")
        
        if result.returncode == 0:
            print(f"✓ {description} completed successfully ({duration:.2f}s)")
            return True, result.stdout, result.stderr
        else:
            print(f"❌ {description} failed (exit code: {result.returncode})")
            return False, result.stdout, result.stderr
    
    except subprocess.TimeoutExpired:
        print(f"❌ {description} timed out after 5 minutes")
        with open(log_file, "a") as f:
            f.write("ERROR: Command timed out after 5 minutes\n")
        return False, "", "Timeout"
    
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        with open(log_file, "a") as f:
            f.write(f"ERROR: Exception occurred: {e}\n")
        return False, "", str(e)

def test_unit_tests(log_file):
    """Run pytest unit tests."""
    print("\n" + "=" * 50)
    print("RUNNING UNIT TESTS")
    print("=" * 50)
    
    cmd = [sys.executable, "-m", "pytest", "tests/test_pretrain_sdm.py", "-v", "--tb=short"]
    return run_command(cmd, log_file, "Unit Tests (pytest)")

def test_integration_tests(log_file):
    """Run integration tests."""
    print("\n" + "=" * 50)
    print("RUNNING INTEGRATION TESTS")
    print("=" * 50)
    
    cmd = [sys.executable, "tests/test_pretrain_sdm_integration.py"]
    return run_command(cmd, log_file, "Integration Tests")

def test_system_validation(log_file):
    """Run full system validation."""
    print("\n" + "=" * 50)
    print("RUNNING SYSTEM VALIDATION")
    print("=" * 50)
    
    system = platform.system().lower()
    
    if system == "linux" or system == "darwin":  # Linux or macOS
        script_path = "tests/test_full_experiment.sh"
        if os.path.exists(script_path):
            # Make script executable
            os.chmod(script_path, 0o755)
            cmd = ["bash", script_path]
            return run_command(cmd, log_file, "System Validation (Bash)")
        else:
            print(f"❌ Bash validation script not found: {script_path}")
            return False, "", "Script not found"
    
    elif system == "windows":
        script_path = "tests/test_full_experiment.ps1"
        if os.path.exists(script_path):
            cmd = ["powershell", "-ExecutionPolicy", "Bypass", "-File", script_path]
            return run_command(cmd, log_file, "System Validation (PowerShell)")
        else:
            print(f"❌ PowerShell validation script not found: {script_path}")
            return False, "", "Script not found"
    
    else:
        print(f"❌ Unsupported platform: {system}")
        return False, "", "Unsupported platform"

def test_configuration_files(log_file):
    """Test configuration file compatibility."""
    print("\n" + "=" * 50)
    print("TESTING CONFIGURATION FILES")
    print("=" * 50)
    
    config_files = [
        "configs/mamba_130m.yaml",
        "configs/mamba_370m.yaml",
        "tests/configs/test_config_tiny.yaml"
    ]
    
    all_passed = True
    
    for config_file in config_files:
        if not os.path.exists(config_file):
            print(f"⚠️ Config file not found: {config_file}")
            continue
        
        # Test YAML loading
        cmd = [
            sys.executable, "-c",
            f"import yaml; yaml.safe_load(open('{config_file}', 'r')); print('✓ {config_file} is valid YAML')"
        ]
        
        success, stdout, stderr = run_command(cmd, log_file, f"YAML Validation: {config_file}")
        if not success:
            all_passed = False
    
    return all_passed, "", ""

def generate_summary_report(log_file, results):
    """Generate a comprehensive summary report."""
    print("\n" + "=" * 80)
    print("TEST EXECUTION SUMMARY")
    print("=" * 80)
    
    passed_tests = sum(1 for success, _, _ in results.values() if success)
    total_tests = len(results)
    
    print(f"Tests passed: {passed_tests}/{total_tests}")
    print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
    print(f"Log file: {log_file}")
    
    # Write summary to log file
    with open(log_file, "a") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("EXECUTION SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Completed at: {datetime.now()}\n")
        f.write(f"Tests passed: {passed_tests}/{total_tests}\n")
        f.write(f"Success rate: {(passed_tests/total_tests)*100:.1f}%\n\n")
        
        f.write("Test Results:\n")
        for test_name, (success, stdout, stderr) in results.items():
            status = "PASS" if success else "FAIL"
            f.write(f"  {test_name}: {status}\n")
        
        f.write("\n")
    
    # Detailed results
    print("\nDetailed Results:")
    for test_name, (success, stdout, stderr) in results.items():
        status = "✓" if success else "❌"
        print(f"  {status} {test_name}")
        if not success and stderr:
            # Show first few lines of error
            error_lines = stderr.split('\n')[:3]
            for line in error_lines:
                if line.strip():
                    print(f"    Error: {line.strip()}")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed! System is ready for experiments.")
        return True
    else:
        print(f"\n⚠️ {total_tests - passed_tests} test(s) failed. Check log file for details.")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive test suite runner for YunMin-mamba-v1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Test Modes:
  unit         Run pytest unit tests only
  integration  Run integration tests only
  system       Run full system validation only
  config       Test configuration files only
  all          Run all tests (default)
  
Examples:
  python tests/run_test_suite.py                    # Run all tests
  python tests/run_test_suite.py --mode unit        # Unit tests only
  python tests/run_test_suite.py --mode system      # System validation only
  python tests/run_test_suite.py --verbose          # Detailed output
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["unit", "integration", "system", "config", "all"],
        default="all",
        help="Test mode to run (default: all)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show verbose output"
    )
    
    parser.add_argument(
        "--log-file",
        help="Custom log file path"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = args.log_file or setup_logging()
    log_system_info(log_file)
    
    print("YunMin-mamba-v1 Test Suite Runner")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Platform: {platform.system()}")
    print(f"Log file: {log_file}")
    print(f"Verbose: {args.verbose}")
    
    # Track results
    results = {}
    start_time = time.time()
    
    # Run tests based on mode
    if args.mode in ["all", "config"]:
        success, stdout, stderr = test_configuration_files(log_file)
        results["Configuration Files"] = (success, stdout, stderr)
    
    if args.mode in ["all", "unit"]:
        success, stdout, stderr = test_unit_tests(log_file)
        results["Unit Tests"] = (success, stdout, stderr)
    
    if args.mode in ["all", "integration"]:
        success, stdout, stderr = test_integration_tests(log_file)
        results["Integration Tests"] = (success, stdout, stderr)
    
    if args.mode in ["all", "system"]:
        success, stdout, stderr = test_system_validation(log_file)
        results["System Validation"] = (success, stdout, stderr)
    
    # Generate summary
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f}s")
    
    all_passed = generate_summary_report(log_file, results)
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main() 