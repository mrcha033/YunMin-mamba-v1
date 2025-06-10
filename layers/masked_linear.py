"""
Clean MaskedLinear implementation that encapsulates masking logic.
This follows the principle of composition over manual weight manipulation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .learned_mask import LearnedMask

class MaskedLinear(nn.Linear):
    """
    A Linear layer with learned masking applied transparently.
    
    This encapsulates all masking logic and eliminates the need for manual
    weight manipulation in the forward pass.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 tau: float = 0.5, init_sparsity: float = 0.5,
                 target_sparsity: Optional[float] = None, 
                 sparsity_weight: float = 1e-5):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            bias: Whether to include bias
            tau: Temperature for Gumbel-Sigmoid
            init_sparsity: Initial sparsity level
            target_sparsity: Target sparsity for regularization
            sparsity_weight: Weight for sparsity loss
        """
        super().__init__(in_features, out_features, bias)
        
        # Create mask generator for weights
        self.mask_generator = LearnedMask(
            self.weight.shape, 
            tau=tau, 
            init_sparsity=init_sparsity
        )
        
        # Sparsity regularization config
        self.target_sparsity = target_sparsity
        self.sparsity_weight = sparsity_weight
        
        # Track statistics
        self.register_buffer('total_forward_calls', torch.tensor(0))
        self.register_buffer('total_active_weights', torch.tensor(0.0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with transparent masking.
        The mask is applied internally - no manual weight manipulation needed.
        """
        # Generate mask (differentiable in training, deterministic in eval)
        mask = self.mask_generator(self.training)
        
        # Apply mask to weights
        masked_weight = self.weight * mask
        
        # Standard linear operation with masked weights
        output = F.linear(x, masked_weight, self.bias)
        
        # Update statistics (no gradients)
        with torch.no_grad():
            self.total_forward_calls += 1
            self.total_active_weights += mask.sum()
        
        return output
    
    def get_sparsity_loss(self) -> torch.Tensor:
        """
        Compute sparsity regularization loss.
        This centralizes all loss computation logic.
        """
        if self.target_sparsity is not None:
            # Re-calculate sparsity here, keeping it as a tensor to preserve gradients
            current_prob = torch.sigmoid(self.mask_generator.logits)
            current_sparsity = 1 - current_prob.mean()
            
            sparsity_loss = torch.abs(current_sparsity - self.target_sparsity)
        else:
            # Simple L1 penalty on mask logits
            sparsity_loss = torch.norm(self.mask_generator.logits, p=1)
        
        return self.sparsity_weight * sparsity_loss
    
    def get_importance_score(self, method: str = 'mask_probability') -> float:
        """
        Compute importance score for this layer using specified method.
        
        Available methods (corresponding to math_spec.md formulations):
        - 'mask_probability': Sigmoid(L) - average probability of weights being active
        - 'logit_magnitude': |L| - absolute value of logits (from math spec)
        - 'weight_magnitude': |W| - absolute value of weights 
        - 'combined': Sigmoid(L) * |W| - combination of mask and weight importance
        - 'mask_usage': Historical usage frequency over time
        
        Returns:
            Importance score (higher = more important for PEFT allocation)
        """
        with torch.no_grad():
            if method == 'mask_probability':
                # Average probability of weights being active: Sigmoid(L)
                probabilities = torch.sigmoid(self.mask_generator.logits)
                return probabilities.mean().item()
            
            elif method == 'logit_magnitude':
                # |L_{i,j}| from math specification - absolute logit values
                logit_magnitude = torch.abs(self.mask_generator.logits)
                return logit_magnitude.mean().item()
            
            elif method == 'weight_magnitude':
                # |W_{i,j}| - average weight magnitude
                return self.weight.abs().mean().item()
            
            elif method == 'combined':
                # Sigmoid(L) * |W| - combination of mask probability and weight magnitude
                mask_score = torch.sigmoid(self.mask_generator.logits).mean()
                weight_score = self.weight.abs().mean()
                return (mask_score * weight_score).item()
            
            elif method == 'mask_usage':
                # Historical usage frequency: ∑M_{i,j}^{(t)} / T
                if hasattr(self.mask_generator, 'mask_usage') and self.mask_generator.update_count > 0:
                    avg_usage = self.mask_generator.mask_usage / self.mask_generator.update_count
                    return avg_usage.mean().item()
                else:
                    # Fallback to mask probability if no usage history
                    probabilities = torch.sigmoid(self.mask_generator.logits)
                    return probabilities.mean().item()
            
            elif method == 'gradient_based':
                # Gradient-based importance (if gradients are available)
                if self.mask_generator.logits.grad is not None:
                    gradient_magnitude = torch.abs(self.mask_generator.logits.grad)
                    return gradient_magnitude.mean().item()
                else:
                    # Fallback to logit magnitude
                    return torch.abs(self.mask_generator.logits).mean().item()
            
            else:
                available_methods = ['mask_probability', 'logit_magnitude', 'weight_magnitude', 
                                   'combined', 'mask_usage', 'gradient_based']
                raise ValueError(f"Unknown importance method: {method}. "
                               f"Available methods: {available_methods}")
    
    def get_statistics(self) -> dict:
        """Get comprehensive statistics for monitoring."""
        mask_stats = self.mask_generator.get_mask_statistics()
        
        avg_active = (self.total_active_weights / self.total_forward_calls 
                     if self.total_forward_calls > 0 else 0.0)
        
        return {
            'current_sparsity': mask_stats.get('current_sparsity', 0.0),
            'average_active_weights': avg_active.item() if torch.is_tensor(avg_active) else float(avg_active),
            'total_forward_calls': self.total_forward_calls.item(),
            'importance_score': self.get_importance_score(),
            'sparsity_loss': self.get_sparsity_loss().item()
        }
    
    def reset_statistics(self):
        """Reset tracking statistics."""
        self.total_forward_calls.zero_()
        self.total_active_weights.zero_()

class MaskedConv1d(nn.Conv1d):
    """
    Conv1d layer with learned masking for convolution kernels.
    Similar principle to MaskedLinear but for convolutions.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, groups: int = 1,
                 bias: bool = True, tau: float = 0.5, init_sparsity: float = 0.5,
                 target_sparsity: Optional[float] = None, sparsity_weight: float = 1e-5):
        super().__init__(in_channels, out_channels, kernel_size, stride, 
                        padding, bias=bias, groups=groups)
        
        # Create mask for convolution weights
        self.mask_generator = LearnedMask(
            self.weight.shape,
            tau=tau,
            init_sparsity=init_sparsity
        )
        
        # Sparsity regularization config
        self.target_sparsity = target_sparsity
        self.sparsity_weight = sparsity_weight
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with masked convolution weights."""
        mask = self.mask_generator(self.training)
        masked_weight = self.weight * mask
        
        return F.conv1d(
            x, masked_weight, self.bias, 
            self.stride, self.padding, self.dilation, self.groups
        )
    
    def get_sparsity_loss(self) -> torch.Tensor:
        """Compute sparsity loss for convolution layer."""
        if self.target_sparsity is not None:
            # Re-calculate sparsity here, keeping it as a tensor to preserve gradients
            current_prob = torch.sigmoid(self.mask_generator.logits)
            current_sparsity = 1 - current_prob.mean()
            
            sparsity_loss = torch.abs(current_sparsity - self.target_sparsity)
        else:
            # Simple L1 penalty on mask logits
            sparsity_loss = torch.norm(self.mask_generator.logits, p=1)
        
        return self.sparsity_weight * sparsity_loss
    
    def get_importance_score(self, method: str = 'mask_probability') -> float:
        """
        Compute importance score for this Conv1d layer.
        Same methods as MaskedLinear but for convolution kernels.
        """
        with torch.no_grad():
            if method == 'mask_probability':
                probabilities = torch.sigmoid(self.mask_generator.logits)
                return probabilities.mean().item()
            
            elif method == 'logit_magnitude':
                logit_magnitude = torch.abs(self.mask_generator.logits)
                return logit_magnitude.mean().item()
            
            elif method == 'weight_magnitude':
                return self.weight.abs().mean().item()
            
            elif method == 'combined':
                mask_score = torch.sigmoid(self.mask_generator.logits).mean()
                weight_score = self.weight.abs().mean()
                return (mask_score * weight_score).item()
            
            elif method == 'mask_usage':
                if hasattr(self.mask_generator, 'mask_usage') and self.mask_generator.update_count > 0:
                    avg_usage = self.mask_generator.mask_usage / self.mask_generator.update_count
                    return avg_usage.mean().item()
                else:
                    probabilities = torch.sigmoid(self.mask_generator.logits)
                    return probabilities.mean().item()
            
            elif method == 'gradient_based':
                if self.mask_generator.logits.grad is not None:
                    gradient_magnitude = torch.abs(self.mask_generator.logits.grad)
                    return gradient_magnitude.mean().item()
                else:
                    return torch.abs(self.mask_generator.logits).mean().item()
            
            else:
                available_methods = ['mask_probability', 'logit_magnitude', 'weight_magnitude', 
                                   'combined', 'mask_usage', 'gradient_based']
                raise ValueError(f"Unknown importance method: {method}. "
                               f"Available methods: {available_methods}")

    def get_statistics(self) -> dict:
        """Get comprehensive statistics for monitoring."""
        mask_stats = self.mask_generator.get_mask_statistics()
        
        return {
            'current_sparsity': mask_stats.get('current_sparsity', 0.0),
            'importance_score': self.get_importance_score(),
            'sparsity_loss': self.get_sparsity_loss().item()
        }

# Utility function to convert existing modules
def convert_to_masked(module: nn.Module, masking_config: dict) -> nn.Module:
    """
    Convert a standard Linear or Conv1d layer to its masked version.
    
    Args:
        module: Original nn.Linear or nn.Conv1d module
        masking_config: Configuration for masking
    
    Returns:
        Converted masked module
    """
    if isinstance(module, nn.Linear):
        masked_module = MaskedLinear(
            module.in_features, 
            module.out_features, 
            bias=module.bias is not None,
            **masking_config
        )
        # Copy weights and bias
        masked_module.weight.data.copy_(module.weight.data)
        if module.bias is not None:
            masked_module.bias.data.copy_(module.bias.data)
        
        return masked_module
    
    elif isinstance(module, nn.Conv1d):
        masked_module = MaskedConv1d(
            module.in_channels,
            module.out_channels, 
            module.kernel_size[0],
            stride=module.stride[0],
            padding=module.padding[0],
            groups=module.groups,
            bias=module.bias is not None,
            **masking_config
        )
        # Copy weights and bias
        masked_module.weight.data.copy_(module.weight.data)
        if module.bias is not None:
            masked_module.bias.data.copy_(module.bias.data)
            
        return masked_module
    
    else:
        raise ValueError(f"Unsupported module type for masking: {type(module)}") 