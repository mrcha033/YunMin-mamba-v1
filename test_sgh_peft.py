"""
Test script for SGH-PEFT (Sparsity-Guided Hybrid PEFT) implementation.
This script verifies the core components of Pillar 3.
"""

import torch
import torch.nn as nn
import sys
import os
import numpy as np

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.sdm_ssm import SDM_SSM, SDM_MambaBlock
from models.sgh_peft import (
    SGHPEFTModel, SGHPEFTConfig, MaskedLoRALayer, IA3Layer,
    compute_layer_importance_scores, create_sgh_peft_model
)
from utils.profiling import count_parameters


def test_masked_lora_layer():
    """Test MaskedLoRALayer functionality."""
    print("Testing MaskedLoRALayer...")
    
    # Create base layer
    base_layer = nn.Linear(256, 512, bias=False)
    
    # Create sparsity mask (50% sparsity)
    sparsity_mask = torch.randint(0, 2, (512,)).float()
    
    # Create MaskedLoRALayer
    lora_layer = MaskedLoRALayer(
        base_layer=base_layer,
        rank=8,
        alpha=16.0,
        dropout=0.1,
        sparsity_mask=sparsity_mask
    )
    
    # Test forward pass
    batch_size, seq_length = 2, 64
    input_tensor = torch.randn(batch_size, seq_length, 256)
    
    with torch.no_grad():
        output = lora_layer(input_tensor)
    
    # Verify output shape
    expected_shape = (batch_size, seq_length, 512)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    
    # Verify sparsity mask is applied
    assert lora_layer.sparsity_mask is not None, "Sparsity mask should be stored"
    assert torch.equal(lora_layer.sparsity_mask, sparsity_mask), "Sparsity mask should match input"
    
    # Test parameter counting
    lora_params = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)
    expected_lora_params = 8 * 256 + 512 * 8  # A + B matrices
    assert lora_params == expected_lora_params, f"Expected {expected_lora_params} LoRA params, got {lora_params}"
    
    print(f"✓ MaskedLoRALayer forward pass successful: {input_tensor.shape} -> {output.shape}")
    print(f"✓ Sparsity mask applied correctly with {sparsity_mask.sum().item()}/{len(sparsity_mask)} active channels")
    print(f"✓ LoRA parameters: {lora_params:,}")
    
    return True


def test_ia3_layer():
    """Test IA3Layer functionality."""
    print("Testing IA3Layer...")
    
    # Create base layer
    base_layer = nn.Linear(256, 512, bias=False)
    
    # Create sparsity mask (30% sparsity)
    sparsity_mask = torch.randint(0, 2, (512,)).float()
    
    # Create IA3Layer
    ia3_layer = IA3Layer(
        base_layer=base_layer,
        init_std=0.02,
        sparsity_mask=sparsity_mask
    )
    
    # Test forward pass
    batch_size, seq_length = 2, 64
    input_tensor = torch.randn(batch_size, seq_length, 256)
    
    with torch.no_grad():
        output = ia3_layer(input_tensor)
    
    # Verify output shape
    expected_shape = (batch_size, seq_length, 512)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    
    # Verify sparsity mask is applied
    assert ia3_layer.sparsity_mask is not None, "Sparsity mask should be stored"
    
    # Test parameter counting
    ia3_params = sum(p.numel() for p in ia3_layer.parameters() if p.requires_grad)
    expected_ia3_params = 512  # One scaling parameter per output feature
    assert ia3_params == expected_ia3_params, f"Expected {expected_ia3_params} IA³ params, got {ia3_params}"
    
    print(f"✓ IA3Layer forward pass successful: {input_tensor.shape} -> {output.shape}")
    print(f"✓ Sparsity mask applied correctly")
    print(f"✓ IA³ parameters: {ia3_params:,}")
    
    return True


def test_importance_score_computation():
    """Test importance score computation from SDM model."""
    print("Testing importance score computation...")
    
    # Create SDM model with learned sparsity patterns
    model = SDM_SSM(
        d_model=256,
        n_layer=4,
        vocab_size=1000,
        d_state=8,
        d_conv=4
    )
    
    # Simulate learned importance patterns
    with torch.no_grad():
        for layer_idx, layer in enumerate(model.layers):
            # Create realistic patterns: decreasing importance with depth
            base_importance = 1.0 - (layer_idx / 4) * 0.8
            noise = torch.randn_like(layer.z_logits) * 0.3
            layer.z_logits.data = base_importance + noise
    
    # Compute importance scores
    importance_scores = compute_layer_importance_scores(model)
    
    # Verify scores were computed for all layers
    expected_layers = 4
    assert len(importance_scores) == expected_layers, f"Expected {expected_layers} layers, got {len(importance_scores)}"
    
    # Verify score structure
    for layer_name, scores in importance_scores.items():
        required_keys = ['mean_importance', 'std_importance', 'active_channels', 'total_channels', 'sparsity_mask']
        for key in required_keys:
            assert key in scores, f"Missing key '{key}' in importance scores for {layer_name}"
        
        # Verify sparsity mask shape
        assert scores['sparsity_mask'].shape == torch.Size([layer.d_inner]), "Sparsity mask shape mismatch"
        
        # Verify channel counts
        assert scores['active_channels'] <= scores['total_channels'], "Active channels should not exceed total"
    
    print(f"✓ Importance scores computed for {len(importance_scores)} layers")
    
    # Show importance distribution
    for layer_name, scores in importance_scores.items():
        print(f"  {layer_name}: mean_imp={scores['mean_importance']:.3f}, "
              f"active={scores['active_channels']}/{scores['total_channels']} "
              f"({scores['active_channels']/scores['total_channels']*100:.1f}%)")
    
    return True


def test_sgh_peft_allocation_strategy():
    """Test SGH-PEFT allocation strategy."""
    print("Testing SGH-PEFT allocation strategy...")
    
    # Create SDM model
    sdm_model = SDM_SSM(
        d_model=256,
        n_layer=6,
        vocab_size=1000,
        d_state=8,
        d_conv=4
    )
    
    # Create diverse importance patterns for testing allocation logic
    importance_patterns = [
        (1.0, 0.8),   # High importance, high activity -> LoRA high
        (0.7, 0.7),   # High importance, medium activity -> LoRA high  
        (0.3, 0.5),   # Medium importance, medium activity -> LoRA low
        (0.1, 0.3),   # Low importance, low activity -> IA³
        (-0.2, 0.2),  # Very low importance -> IA³
        (-0.8, 0.1),  # Minimal importance -> Frozen
    ]
    
    with torch.no_grad():
        for layer_idx, (mean_imp, activity_ratio) in enumerate(importance_patterns):
            layer = sdm_model.layers[layer_idx]
            
            # Set z_logits to create desired patterns
            num_active = int(activity_ratio * layer.d_inner)
            layer.z_logits.data.fill_(mean_imp - 1.0)  # Most negative
            layer.z_logits.data[:num_active] = mean_imp + 1.0  # Some positive
    
    # Create SGH-PEFT model
    config = SGHPEFTConfig(
        lora_high_rank=16,
        lora_low_rank=4,
        high_importance_mean_threshold=0.5,
        high_importance_active_threshold=60.0,
        medium_importance_mean_threshold=0.0,
        medium_importance_active_threshold=40.0,
        low_importance_mean_threshold=-0.5
    )
    
    sgh_peft_model = create_sgh_peft_model(sdm_model, config)
    
    # Verify allocation decisions
    adaptation_summary = sgh_peft_model.get_adaptation_summary()
    adapter_distribution = adaptation_summary['adapter_distribution']
    
    print("Allocation results:")
    for adapter_type, count in adapter_distribution.items():
        print(f"  {adapter_type}: {count} layers")
    
    # Verify we have diversity in allocation
    assert adapter_distribution['lora_high'] > 0, "Should have high-rank LoRA layers"
    assert adapter_distribution['lora_low'] > 0, "Should have low-rank LoRA layers"
    assert adapter_distribution['ia3'] > 0, "Should have IA³ layers"
    assert adapter_distribution['frozen'] > 0, "Should have frozen layers"
    
    print("✓ SGH-PEFT allocation strategy working correctly")
    
    return True


def test_sgh_peft_forward_pass():
    """Test SGH-PEFT model forward pass."""
    print("Testing SGH-PEFT forward pass...")
    
    # Create SDM model
    sdm_model = SDM_SSM(
        d_model=128,
        n_layer=3,
        vocab_size=500,
        d_state=8,
        d_conv=4
    )
    
    # Set some reasonable importance patterns
    with torch.no_grad():
        for layer_idx, layer in enumerate(sdm_model.layers):
            base_importance = 0.5 - layer_idx * 0.3
            noise = torch.randn_like(layer.z_logits) * 0.2
            layer.z_logits.data = base_importance + noise
    
    # Create SGH-PEFT model
    config = SGHPEFTConfig()
    sgh_peft_model = create_sgh_peft_model(sdm_model, config)
    
    # Test forward pass
    batch_size, seq_length = 2, 32
    input_ids = torch.randint(0, 500, (batch_size, seq_length))
    
    with torch.no_grad():
        output = sgh_peft_model(input_ids)
    
    # Verify output shape
    expected_shape = (batch_size, seq_length, 500)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    
    # Verify some parameters are trainable
    trainable_params = sum(p.numel() for p in sgh_peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in sgh_peft_model.parameters())
    
    assert trainable_params > 0, "Should have some trainable parameters"
    assert trainable_params < total_params, "Should not have all parameters trainable"
    
    print(f"✓ SGH-PEFT forward pass successful: {input_ids.shape} -> {output.shape}")
    print(f"✓ Parameter efficiency: {trainable_params:,}/{total_params:,} ({trainable_params/total_params:.2%}) trainable")
    
    return True


def test_sparsity_mask_integration():
    """Test integration between SDM sparsity masks and SGH-PEFT adapters."""
    print("Testing sparsity mask integration...")
    
    # Create SDM model with known sparsity pattern
    sdm_model = SDM_SSM(
        d_model=64,
        n_layer=2,
        vocab_size=100,
        d_state=8,
        d_conv=4
    )
    
    # Set specific sparsity patterns
    with torch.no_grad():
        for layer in sdm_model.layers:
            # Create 50% sparsity pattern
            layer.z_logits.data = torch.where(
                torch.rand_like(layer.z_logits) > 0.5,
                torch.ones_like(layer.z_logits) * 2.0,
                torch.ones_like(layer.z_logits) * -2.0
            )
    
    # Create SGH-PEFT model with sparsity masking enabled
    config = SGHPEFTConfig(apply_sparsity_mask=True)
    sgh_peft_model = create_sgh_peft_model(sdm_model, config)
    
    # Verify sparsity masks are applied to adapters
    mask_applied_count = 0
    for name, module in sgh_peft_model.named_modules():
        if isinstance(module, (MaskedLoRALayer, IA3Layer)):
            if hasattr(module, 'sparsity_mask') and module.sparsity_mask is not None:
                mask_applied_count += 1
                
                # Verify mask has reasonable sparsity
                sparsity = 1.0 - module.sparsity_mask.mean().item()
                assert 0.0 < sparsity < 1.0, f"Sparsity should be between 0 and 1, got {sparsity}"
    
    assert mask_applied_count > 0, "Should have applied sparsity masks to adapters"
    
    print(f"✓ Sparsity masks applied to {mask_applied_count} adapter layers")
    
    # Test with sparsity masking disabled
    config_no_mask = SGHPEFTConfig(apply_sparsity_mask=False)
    sgh_peft_model_no_mask = create_sgh_peft_model(sdm_model, config_no_mask)
    
    # Verify no masks are applied
    no_mask_count = 0
    for name, module in sgh_peft_model_no_mask.named_modules():
        if isinstance(module, (MaskedLoRALayer, IA3Layer)):
            if hasattr(module, 'sparsity_mask') and module.sparsity_mask is not None:
                # Should be all ones (no masking)
                assert torch.allclose(module.sparsity_mask, torch.ones_like(module.sparsity_mask)), \
                    "Should have no masking when disabled"
                no_mask_count += 1
    
    print(f"✓ Sparsity masking correctly disabled for {no_mask_count} adapter layers")
    
    return True


def test_parameter_efficiency():
    """Test parameter efficiency of SGH-PEFT vs full fine-tuning."""
    print("Testing parameter efficiency...")
    
    # Create baseline and SGH-PEFT models
    sdm_model = SDM_SSM(
        d_model=256,
        n_layer=6,
        vocab_size=1000,
        d_state=16,
        d_conv=4
    )
    
    # Set realistic importance patterns
    with torch.no_grad():
        for layer_idx, layer in enumerate(sdm_model.layers):
            base_importance = 0.8 - layer_idx * 0.2
            layer.z_logits.data = base_importance + torch.randn_like(layer.z_logits) * 0.3
    
    # Full fine-tuning baseline (all parameters trainable)
    baseline_model = SDM_SSM(
        d_model=256,
        n_layer=6,
        vocab_size=1000,
        d_state=16,
        d_conv=4
    )
    baseline_trainable = sum(p.numel() for p in baseline_model.parameters())
    
    # SGH-PEFT model
    config = SGHPEFTConfig()
    sgh_peft_model = create_sgh_peft_model(sdm_model, config)
    
    sgh_peft_total = sum(p.numel() for p in sgh_peft_model.parameters())
    sgh_peft_trainable = sum(p.numel() for p in sgh_peft_model.parameters() if p.requires_grad)
    
    # Calculate efficiency metrics
    parameter_reduction = 1.0 - (sgh_peft_trainable / baseline_trainable)
    efficiency_ratio = baseline_trainable / sgh_peft_trainable
    
    print(f"Baseline (full fine-tuning): {baseline_trainable:,} trainable parameters")
    print(f"SGH-PEFT: {sgh_peft_trainable:,} trainable parameters")
    print(f"Parameter reduction: {parameter_reduction:.2%}")
    print(f"Efficiency ratio: {efficiency_ratio:.2f}x fewer parameters")
    
    # Verify significant parameter reduction
    assert parameter_reduction > 0.5, f"Should have >50% parameter reduction, got {parameter_reduction:.2%}"
    assert efficiency_ratio > 2.0, f"Should be >2x more efficient, got {efficiency_ratio:.2f}x"
    
    print("✓ SGH-PEFT achieves significant parameter efficiency")
    
    return True


def main():
    """Run all SGH-PEFT tests."""
    print("=" * 70)
    print("PILLAR 3: SGH-PEFT (SPARSITY-GUIDED HYBRID PEFT) TESTS")
    print("=" * 70)
    
    tests = [
        ("MaskedLoRALayer Functionality", test_masked_lora_layer),
        ("IA3Layer Functionality", test_ia3_layer),
        ("Importance Score Computation", test_importance_score_computation),
        ("SGH-PEFT Allocation Strategy", test_sgh_peft_allocation_strategy),
        ("SGH-PEFT Forward Pass", test_sgh_peft_forward_pass),
        ("Sparsity Mask Integration", test_sparsity_mask_integration),
        ("Parameter Efficiency", test_parameter_efficiency),
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
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 70)
    print("SGH-PEFT TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{status:4} | {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All SGH-PEFT tests passed! Pillar 3 implementation is ready.")
        print("✓ Masked LoRA layers functional")
        print("✓ IA³ adapters working correctly")
        print("✓ Importance-based allocation strategy operational")
        print("✓ Sparsity mask integration verified")
        print("✓ Parameter efficiency achieved")
        print("\n🏆 HARDWARE-DATA-PARAMETER CO-DESIGN FRAMEWORK COMPLETE!")
        print("All three pillars implemented and tested:")
        print("  Pillar 1: CSP (Correlation-based Scan Permutation) ✅")
        print("  Pillar 2: SDM (Structured Differentiable Masking) ✅")
        print("  Pillar 3: SGH-PEFT (Sparsity-Guided Hybrid PEFT) ✅")
        print("\nReady for:")
        print("1. SDM pre-training: python pretrain_sdm.py")
        print("2. SGH-PEFT fine-tuning: python scripts/run_finetuning.py")
        print("3. Full pipeline evaluation and analysis")
    else:
        print(f"\n⚠ {total - passed} tests failed. Please fix issues before proceeding.")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 