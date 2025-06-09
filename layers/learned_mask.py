"""
Pillar 2: Learned Masking Implementation
Implements Gumbel-Sigmoid based differentiable binary masking for Mamba SSM components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class LearnedMask(nn.Module):
    """
    Learned binary mask using Gumbel-Sigmoid distribution.
    
    Mathematical formulation:
    M_{i,j} = Sigmoid((L_{i,j} + G_{i,j}) / τ)
    where G_{i,j} ~ Gumbel(0,1) and L_{i,j} are learnable logits.
    """
    
    def __init__(self, shape: Tuple[int, ...], tau: float = 0.5, init_sparsity: float = 0.5):
        """
        Args:
            shape: Shape of the mask tensor
            tau: Temperature parameter for Gumbel-Sigmoid
            init_sparsity: Initial sparsity level (0.5 = 50% sparse)
        """
        super().__init__()
        self.shape = shape
        self.tau = tau
        
        # Initialize logits to achieve desired initial sparsity
        # logit(p) = log(p / (1-p)) where p = 1 - init_sparsity
        init_value = torch.log(torch.tensor((1 - init_sparsity) / init_sparsity))
        self.logits = nn.Parameter(torch.full(shape, init_value.item()))
        
        # Track statistics
        self.register_buffer('mask_usage', torch.zeros(shape))
        self.register_buffer('update_count', torch.tensor(0))
    
    def sample_gumbel(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        """
        Sample from Gumbel(0, 1) distribution.
        G = -log(-log(U)) where U ~ Uniform(0, 1)
        """
        uniform = torch.rand(shape, device=device)
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        uniform = torch.clamp(uniform, eps, 1 - eps)
        return -torch.log(-torch.log(uniform))
    
    def forward(self, is_training: bool = True, hard: bool = True) -> torch.Tensor:
        """
        Generate binary mask using Gumbel-Sigmoid.
        
        Args:
            is_training: Whether in training mode (affects sampling)
            hard: Whether to use hard (binary) or soft masks
        
        Returns:
            Binary mask tensor of shape self.shape
        """
        if is_training:
            # Gumbel-Sigmoid sampling (differentiable)
            gumbel_noise = self.sample_gumbel(self.shape, self.logits.device)
            y_soft = torch.sigmoid((self.logits + gumbel_noise) / self.tau)
            
            if hard:
                # Straight-Through Estimator (STE)
                y_hard = (y_soft > 0.5).float()
                # Use STE: forward pass uses hard, backward pass uses soft
                mask = y_hard - y_soft.detach() + y_soft
            else:
                mask = y_soft
                
            # Update usage statistics
            with torch.no_grad():
                self.mask_usage += (y_soft > 0.5).float()
                self.update_count += 1
                
        else:
            # Test time: deterministic hard thresholding
            mask = (self.logits > 0).float()
        
        return mask
    
    def get_sparsity(self) -> float:
        """Get current sparsity level (percentage of zero elements)."""
        with torch.no_grad():
            prob = torch.sigmoid(self.logits)
            return (1 - prob.mean()).item()
    
    def get_mask_statistics(self) -> dict:
        """Get detailed mask usage statistics."""
        if self.update_count > 0:
            avg_usage = self.mask_usage / self.update_count
            return {
                'current_sparsity': self.get_sparsity(),
                'average_usage': avg_usage.mean().item(),
                'usage_std': avg_usage.std().item(),
                'total_updates': self.update_count.item()
            }
        else:
            return {'current_sparsity': self.get_sparsity()}

class AdaptiveSparsityMask(LearnedMask):
    """
    Extended mask with adaptive sparsity control based on importance scores.
    """
    
    def __init__(self, shape: Tuple[int, ...], target_sparsity: float = 0.5, 
                 sparsity_weight: float = 1e-3, **kwargs):
        """
        Args:
            target_sparsity: Target sparsity level to maintain
            sparsity_weight: Weight for sparsity regularization loss
        """
        super().__init__(shape, **kwargs)
        self.target_sparsity = target_sparsity
        self.sparsity_weight = sparsity_weight
    
    def sparsity_loss(self) -> torch.Tensor:
        """
        Compute regularization loss to maintain target sparsity.
        L_sparsity = |current_sparsity - target_sparsity|
        """
        current_sparsity = 1 - torch.sigmoid(self.logits).mean()
        return self.sparsity_weight * torch.abs(current_sparsity - self.target_sparsity)

class MambaMaskingLayer(nn.Module):
    """
    Wrapper that applies learned masks to Mamba SSM components.
    
    Supports masking of:
    - B projection: B_t = Linear(M_B ⊙ W_B)(x_t)
    - Convolution kernel: K_sparse = M ⊙ K
    """
    
    def __init__(self, original_layer: nn.Module, mask_type: str = 'weight'):
        """
        Args:
            original_layer: Original linear layer or conv layer to mask
            mask_type: Type of masking ('weight' or 'activation')
        """
        super().__init__()
        self.original_layer = original_layer
        self.mask_type = mask_type
        
        # Create mask based on layer type
        if hasattr(original_layer, 'weight'):
            self.mask = LearnedMask(original_layer.weight.shape)
        else:
            raise ValueError("Layer must have 'weight' attribute for masking")
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply masked computation."""
        if self.mask_type == 'weight':
            # Apply mask to weights: W_masked = M ⊙ W
            mask = self.mask(self.training)
            
            # Temporarily modify weights
            original_weight = self.original_layer.weight.data
            self.original_layer.weight.data = mask * original_weight
            
            # Forward pass
            output = self.original_layer(x, **kwargs)
            
            # Restore original weights
            self.original_layer.weight.data = original_weight
            
            return output
            
        elif self.mask_type == 'activation':
            # Apply mask to activations: y = M ⊙ f(x)
            output = self.original_layer(x, **kwargs)
            mask = self.mask(self.training)
            return mask * output
        
        else:
            raise ValueError(f"Unknown mask_type: {self.mask_type}")
    
    def get_mask_loss(self) -> torch.Tensor:
        """Get regularization loss from the mask."""
        if isinstance(self.mask, AdaptiveSparsityMask):
            return self.mask.sparsity_loss()
        else:
            return torch.tensor(0.0, device=self.mask.logits.device)

def compute_importance_score(mask_logits: torch.Tensor, method: str = 'magnitude') -> torch.Tensor:
    """
    Compute importance scores from mask logits.
    
    Args:
        mask_logits: Learned mask logits
        method: Method for computing importance ('magnitude', 'probability', 'gradient')
    
    Returns:
        Importance scores for each parameter
    """
    if method == 'magnitude':
        return torch.abs(mask_logits)
    elif method == 'probability':
        return torch.sigmoid(mask_logits)
    elif method == 'gradient':
        # Requires gradients to be computed
        if mask_logits.grad is not None:
            return torch.abs(mask_logits.grad)
        else:
            return torch.abs(mask_logits)  # Fallback to magnitude
    else:
        raise ValueError(f"Unknown importance method: {method}")

def apply_structured_pruning(mask: torch.Tensor, structure: str = 'channel') -> torch.Tensor:
    """
    Apply structured pruning to maintain hardware efficiency.
    
    Args:
        mask: Original element-wise mask
        structure: Pruning structure ('channel', 'block', 'row', 'column')
    
    Returns:
        Structured mask
    """
    if structure == 'channel':
        # Keep entire channels if any element is active
        channel_mask = mask.sum(dim=1, keepdim=True) > 0
        return mask * channel_mask.float()
    elif structure == 'row':
        row_mask = mask.sum(dim=1, keepdim=True) > 0
        return mask * row_mask.float()
    elif structure == 'column':
        col_mask = mask.sum(dim=0, keepdim=True) > 0
        return mask * col_mask.float()
    else:
        return mask  # No structured pruning