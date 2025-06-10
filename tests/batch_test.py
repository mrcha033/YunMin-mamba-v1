from quick_test import quick_test

# All available test modes
MODES = ["baseline", "ia3", "masking", "full"]


def run_all():
    """Run comprehensive test suite across all modes."""
    print("🧪 Running Adaptive Hybrid-PEFT Mamba Test Suite")
    print("=" * 50)
    
    results = {}
    
    for mode in MODES:
        print(f"\n📋 Testing {mode} mode...")
        try:
            results[mode] = quick_test(mode)
        except Exception as e:
            print(f"❌ {mode} mode failed with exception: {e}")
            results[mode] = False
    
    # Summary report
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print("=" * 50)
    
    passed = 0
    for mode, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{mode:12}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(MODES)} tests passed")
    
    if all(results.values()):
        print("🎉 All tests passed! The implementation is ready for production.")
    else:
        print("⚠️  Some tests failed. Please review the errors above.")
    
    return all(results.values())


if __name__ == "__main__":
    success = run_all()
    raise SystemExit(0 if success else 1)
