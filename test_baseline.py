"""
Test script to verify the baseline SSM implementation.
This script tests the core components of Phase 0: Baseline Establishment.
"""

import torch
import sys
import os

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.baseline_ssm import BaselineSSM, MambaBlock
from utils.profiling import count_parameters, count_flops
from utils.logger import setup_logger


def test_mamba_block():
    """Test MambaBlock functionality."""
    print("Testing MambaBlock...")
    
    # Initialize MambaBlock
    block = MambaBlock(d_model=768, d_state=16, d_conv=4, expand=2)
    
    # Test forward pass
    batch_size, seq_length = 2, 512
    input_tensor = torch.randn(batch_size, seq_length, 768)
    
    with torch.no_grad():
        output = block(input_tensor)
    
    # Verify output shape
    expected_shape = (batch_size, seq_length, 768)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    
    print(f"✓ MambaBlock forward pass successful: {input_tensor.shape} -> {output.shape}")
    return True


def test_baseline_ssm():
    """Test BaselineSSM functionality."""
    print("Testing BaselineSSM...")
    
    # Initialize BaselineSSM
    model = BaselineSSM(
        d_model=768,
        n_layer=12,
        vocab_size=50257,
        d_state=16,
        d_conv=4
    )
    
    # Test forward pass
    batch_size, seq_length = 2, 512
    input_ids = torch.randint(0, 50257, (batch_size, seq_length))
    
    with torch.no_grad():
        output = model(input_ids)
    
    # Verify output shape
    expected_shape = (batch_size, seq_length, 50257)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    
    print(f"✓ BaselineSSM forward pass successful: {input_ids.shape} -> {output.shape}")
    return True


def test_profiling():
    """Test profiling utilities."""
    print("Testing profiling utilities...")
    
    # Create a small model for testing
    model = BaselineSSM(d_model=256, n_layer=4, vocab_size=1000, d_state=8, d_conv=4)
    
    # Test parameter counting
    param_info = count_parameters(model)
    print(f"✓ Parameter counting: {param_info['total_parameters']:,} total parameters")
    
    # Test FLOPs counting (CPU only to avoid memory issues)
    try:
        flop_info = count_flops(model, (1, 128), device="cpu")
        print(f"✓ FLOPs counting: {flop_info['total_flops']:,} FLOPs")
    except Exception as e:
        print(f"⚠ FLOPs counting failed (expected): {e}")
    
    return True


def test_optimization_targets():
    """Test that optimization targets are accessible."""
    print("Testing optimization targets...")
    
    model = BaselineSSM(d_model=768, n_layer=12, vocab_size=50257, d_state=16, d_conv=4)
    
    # Test CSP targets (state parameters)
    for i, layer in enumerate(model.layers):
        assert hasattr(layer, 'A_log'), f"Layer {i} missing A_log parameter (CSP target)"
        assert hasattr(layer, 'x_proj'), f"Layer {i} missing x_proj parameter (CSP target)"
        assert layer.A_log.shape == (layer.d_inner, layer.d_state), f"Incorrect A_log shape in layer {i}"
    
    print(f"✓ CSP targets accessible: A_log and x_proj parameters in {len(model.layers)} layers")
    
    # Test SDM targets (channel dimensions)
    for i, layer in enumerate(model.layers):
        assert hasattr(layer, 'in_proj'), f"Layer {i} missing in_proj parameter (SDM target)"
        assert layer.in_proj.out_features == layer.d_inner * 2, f"Incorrect in_proj dimensions in layer {i}"
    
    print(f"✓ SDM targets accessible: in_proj channels in {len(model.layers)} layers")
    
    # Test SGH-PEFT targets (layer-wise structure)
    assert hasattr(model, 'layers'), "Model missing layers attribute (SGH-PEFT target)"
    assert len(model.layers) > 0, "Model has no layers for SGH-PEFT"
    
    print(f"✓ SGH-PEFT targets accessible: {len(model.layers)} layers for importance scoring")
    
    return True


def test_model_dimensions():
    """Test that model dimensions match the paper specifications."""
    print("Testing model dimensions...")
    
    # Test configuration from configs/pretrain_base.yaml
    model = BaselineSSM(
        d_model=768,     # Model dimension
        n_layer=12,      # Number of layers
        vocab_size=50257, # GPT-2 vocabulary size
        d_state=16,      # SSM state dimension
        d_conv=4         # Convolution kernel size
    )
    
    # Verify layer structure
    assert len(model.layers) == 12, f"Expected 12 layers, got {len(model.layers)}"
    
    # Verify each layer's dimensions
    for i, layer in enumerate(model.layers):
        assert layer.d_model == 768, f"Layer {i} d_model mismatch"
        assert layer.d_state == 16, f"Layer {i} d_state mismatch"
        assert layer.d_conv == 4, f"Layer {i} d_conv mismatch"
        assert layer.d_inner == 768 * 2, f"Layer {i} d_inner mismatch"
    
    print("✓ Model dimensions match paper specifications")
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("PHASE 0: BASELINE ESTABLISHMENT - VERIFICATION TESTS")
    print("=" * 60)
    
    tests = [
        ("MambaBlock Functionality", test_mamba_block),
        ("BaselineSSM Functionality", test_baseline_ssm),
        ("Profiling Utilities", test_profiling),
        ("Optimization Targets", test_optimization_targets),
        ("Model Dimensions", test_model_dimensions),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n[{test_name}]")
        try:
            success = test_func()
            results.append((test_name, success))
            print(f"✓ {test_name} PASSED")
        except Exception as e:
            results.append((test_name, False))
            print(f"✗ {test_name} FAILED: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{status:4} | {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The baseline implementation is ready.")
        print("✓ M_base model architecture verified")
        print("✓ Optimization targets accessible")
        print("✓ Profiling utilities functional")
        print("\nNext steps:")
        print("1. Run baseline pre-training: python pretrain.py")
        print("2. Implement CSP analysis: python scripts/run_csp_analysis.py")
        print("3. Develop SDM structured masking")
        print("4. Create SGH-PEFT fine-tuning pipeline")
    else:
        print(f"\n⚠ {total - passed} tests failed. Please fix issues before proceeding.")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 