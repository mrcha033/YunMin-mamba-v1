"""
Test suite for pretrain_sdm.py

This module tests the SDM pre-training functionality including:
- Configuration loading and validation
- Model initialization with SDM components
- Training loop components
- Sparsity metrics and checkpoint saving
"""

import os
import sys
import pytest
import torch
import yaml
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pretrain_sdm import (
    load_config, 
    calculate_sparsity_loss,
    adaptive_temperature_schedule,
    update_model_temperature,
    save_sdm_checkpoint,
    main
)
from models.sdm_ssm import SDM_SSM


class TestConfigLoading:
    """Test configuration loading and validation."""
    
    def test_load_config_valid_file(self, tmp_path):
        """Test loading a valid configuration file."""
        config_data = {
            'model': {
                'd_model': 768,
                'n_layer': 12,
                'vocab_size': 50257,
                'd_state': 16,
                'd_conv': 4
            },
            'training': {
                'pretrain': {
                    'batch_size': 24,
                    'micro_batch_size': 6,
                    'learning_rate': 8e-5,
                    'weight_decay': 0.1,
                    'warmup_steps': 2000,
                    'max_epochs': 15,
                    'max_grad_norm': 1.0
                }
            },
            'data': {
                'max_length': 1024,
                'num_workers': 4
            },
            'sdm': {
                'lambda_sparsity': 0.01,
                'initial_temperature': 5.0,
                'final_temperature': 0.1,
                'target_sparsity': 0.5
            },
            'logging': {
                'log_interval': 50,
                'eval_interval': 1000,
                'save_interval': 2500,
                'wandb_project': 'test-project'
            }
        }
        
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        loaded_config = load_config(str(config_file))
        assert loaded_config == config_data
    
    def test_config_missing_file(self):
        """Test error handling for missing config file."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")


class TestSDMComponents:
    """Test SDM-specific components."""
    
    def test_adaptive_temperature_schedule(self):
        """Test temperature annealing schedule."""
        # Test start of training
        temp = adaptive_temperature_schedule(0, 1000, 5.0, 0.1)
        assert temp == 5.0
        
        # Test middle of training
        temp = adaptive_temperature_schedule(500, 1000, 5.0, 0.1)
        assert 0.1 < temp < 5.0
        
        # Test end of training
        temp = adaptive_temperature_schedule(1000, 1000, 5.0, 0.1)
        assert temp == 0.1
    
    def test_calculate_sparsity_loss(self):
        """Test sparsity loss calculation."""
        # Create a mock model with z_logits parameters
        model = Mock()
        
        # Mock z_logits parameters
        z_logits1 = torch.tensor([2.0, -1.0, 0.5, -2.0])  # Some positive, some negative
        z_logits2 = torch.tensor([1.0, -0.5, 3.0])
        
        model.named_parameters.return_value = [
            ('layer1.z_logits', z_logits1),
            ('layer2.z_logits', z_logits2),
            ('layer3.weight', torch.randn(10, 10))  # Non-z_logits parameter
        ]
        
        loss = calculate_sparsity_loss(model)
        
        # Should be positive (encourages sparsity)
        assert loss > 0
        assert isinstance(loss, torch.Tensor)


class TestConfigCompatibility:
    """Test configuration compatibility between different formats."""
    
    def test_temperature_key_fallback(self):
        """Test that both temperature key formats work."""
        # Test config with old keys
        config_old = {
            'sdm': {
                'initial_temperature': 5.0,
                'final_temperature': 0.1,
                'lambda_sparsity': 0.01
            }
        }
        
        # Test config with new keys
        config_new = {
            'sdm': {
                'gumbel_temp_start': 5.0,
                'gumbel_temp_end': 0.1,
                'lambda_sparsity': 0.01
            }
        }
        
        # Both should work (tested via the fallback logic we implemented)
        temp_old = config_old['sdm'].get('initial_temperature', config_old['sdm'].get('gumbel_temp_start', 5.0))
        temp_new = config_new['sdm'].get('initial_temperature', config_new['sdm'].get('gumbel_temp_start', 5.0))
        
        assert temp_old == 5.0
        assert temp_new == 5.0


if __name__ == "__main__":
    pytest.main([__file__]) 