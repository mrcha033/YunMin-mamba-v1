"""
Quick test utility for testing different training modes.
"""

import subprocess
import time
import sys
import os


def quick_test(mode: str, timeout: int = 30) -> bool:
    """
    Run a quick test with the specified mode.
    
    Args:
        mode: Either "baseline" or "ia3"
        timeout: Maximum time to wait for the test to complete
        
    Returns:
        True if test passes, False otherwise
    """
    print(f"Running quick test in {mode} mode...")
    
    # Build command based on mode
    cmd = ["python", "train.py", "--epochs", "1", "--batch-size", "4"]
    
    if mode == "ia3":
        cmd.append("--ia3")
    elif mode != "baseline":
        print(f"Unknown mode: {mode}")
        return False
    
    try:
        print(f"Command: {' '.join(cmd)}")
        start_time = time.time()
        result = subprocess.run(cmd, timeout=timeout)
        end_time = time.time()
        
        print(f"Test completed in {end_time - start_time:.2f} seconds")
        
        if result.returncode == 0:
            print(f"✅ {mode} test passed")
            return True
        else:
            print(f"❌ {mode} test failed with return code {result.returncode}")
            if result.stderr:
                print(f"Error output: {result.stderr}")
            if result.stdout:
                print(f"Standard output: {result.stdout[:500]}...")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ {mode} test timed out after {timeout} seconds")
        return False
    except Exception as e:
        print(f"❌ {mode} test failed with exception: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = "baseline"
    
    success = quick_test(mode)
    sys.exit(0 if success else 1) 