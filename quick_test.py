#!/usr/bin/env python3
"""
Quick Test Script for YunMin Training Pipeline
Minimal configuration for rapid validation
"""

import subprocess
import time

def quick_test(mode):
    """Run a quick test with minimal configuration"""
    print(f"\n🧪 Quick test: {mode} mode")
    
    cmd = [
        "python", "train_yunmin.py",
        "--mode", mode,
        "--epochs", "1",
        "--batch_size", "2", 
        "--max_length", "128",
        "--lr", "1e-4"
    ]
    
    print(f"🔄 Command: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, timeout=300)  # 5 min timeout
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {mode} test passed in {duration:.1f}s")
            return True
        else:
            print(f"❌ {mode} test failed (code: {result.returncode})")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {mode} test timed out")
        return False
    except Exception as e:
        print(f"💥 {mode} test crashed: {e}")
        return False

def main():
    """Run quick tests for all modes"""
    print("🚀 YunMin Quick Test Suite")
    print("=" * 40)
    
    modes = ["baseline", "lora", "scan", "hybrid", "ia3", "ia3_lora"]
    results = {}
    
    for mode in modes:
        results[mode] = quick_test(mode)
        time.sleep(2)  # Brief pause
    
    print("\n📊 Quick Test Results:")
    print("-" * 40)
    for mode, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{mode:<12} {status}")
    
    all_passed = all(results.values())
    print(f"\n🎯 Overall: {'All tests passed!' if all_passed else 'Some tests failed'}")
    
    if all_passed:
        print("🚀 Ready for full experiments!")
    else:
        print("🔧 Please fix issues before running full experiments")

if __name__ == "__main__":
    main() 