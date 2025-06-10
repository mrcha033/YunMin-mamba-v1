import torch
import sys
from pathlib import Path

# Add parent directory to path to import model
sys.path.append(str(Path(__file__).parent.parent))

from model import AdaptiveMambaModel


def quick_test(mode: str = "baseline") -> bool:
    """
    Comprehensive quick test for AdaptiveMambaModel functionality.
    
    Args:
        mode: Test mode - "baseline", "ia3", "masking", or "full"
    
    Returns:
        True if all tests pass, False otherwise
    """
    try:
        print(f"Running quick test in {mode} mode...")
        
        # Test configuration based on mode
        if mode == "baseline":
            config = {
                'enable_masking': False,
                'enable_scan_optimization': False
            }
        elif mode == "ia3":
            config = {
                'enable_masking': False,
                'enable_scan_optimization': False
            }
        elif mode == "masking":
            config = {
                'enable_masking': True,
                'masking_config': {
                    'tau': 0.5,
                    'init_sparsity': 0.3,
                    'target_sparsity': 0.5,
                    'sparsity_weight': 1e-5
                },
                'enable_scan_optimization': False
            }
        elif mode == "full":
            config = {
                'enable_masking': True,
                'masking_config': {
                    'tau': 0.5,
                    'init_sparsity': 0.3,
                    'target_sparsity': 0.5,
                    'sparsity_weight': 1e-5
                },
                'enable_scan_optimization': True
            }
        else:
            print(f"Unknown mode: {mode}")
            return False
        
        # Create small model for testing
        model = AdaptiveMambaModel(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            block_config=config
        )
        
        # Test forward pass
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        
        with torch.no_grad():
            output = model(input_ids)
        
        # Verify output shape
        expected_shape = (batch_size, seq_len, 1000)
        if output.shape != expected_shape:
            print(f"Shape mismatch: expected {expected_shape}, got {output.shape}")
            return False
        
        # Test importance scores functionality
        if hasattr(model.blocks[0], 'get_importance_scores'):
            scores = model.blocks[0].get_importance_scores()
            if not isinstance(scores, dict):
                print("Importance scores should return a dictionary")
                return False
        
        # Test regularization loss (if masking enabled)
        if config.get('enable_masking', False):
            reg_loss = model.get_total_regularization_loss()
            if not isinstance(reg_loss, torch.Tensor):
                print("Regularization loss should return a tensor")
                return False
        
        # Test PEFT compatibility
        inputs = model.prepare_inputs_for_generation(input_ids)
        if 'input_ids' not in inputs:
            print("prepare_inputs_for_generation should return dict with 'input_ids'")
            return False
        
        print(f"✅ {mode} mode test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    success = quick_test(mode)
    print(f"Test result: {'PASS' if success else 'FAIL'}")
    raise SystemExit(0 if success else 1)
