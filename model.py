"""
Refactored Adaptive Mamba Model Implementation
Clean, modular design with proper separation of concerns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

# Pillar 1: Variable-Aware Scan
try:
    from .layers.variable_scan import apply_permutation, invert_permutation, VariableScanOptimizer
    from .layers.masked_linear import MaskedLinear, MaskedConv1d, convert_to_masked
except ImportError:
    # Fallback for direct execution
    from layers.variable_scan import apply_permutation, invert_permutation, VariableScanOptimizer
    from layers.masked_linear import MaskedLinear, MaskedConv1d, convert_to_masked

try:
    # Use official Mamba implementation for optimized scan
    from mamba_ssm.modules.mamba_simple import Mamba as OfficialMamba
    MAMBA_AVAILABLE = True
except ImportError:
    print("Warning: mamba_ssm not available. Using fallback implementation.")
    MAMBA_AVAILABLE = False

class FallbackMamba(nn.Module):
    """
    Simple fallback implementation when mamba_ssm is not available.
    This is a simplified version for demonstration purposes.
    """
    
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            padding=d_conv - 1,
            groups=self.d_inner,
        )
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Simplified forward pass."""
        B, L, D = x.shape
        
        # Project and split
        xz = self.in_proj(x)  # (B, L, 2 * d_inner)
        x_part, z = xz.chunk(2, dim=-1)  # Each: (B, L, d_inner)
        
        # Apply convolution
        x_conv = self.conv1d(x_part.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv = F.silu(x_conv)
        
        # Simple gating (placeholder for SSM logic)
        y = x_conv * F.silu(z)
        
        # Output projection
        output = self.out_proj(y)
        return output

class AdaptiveMambaBlock(nn.Module):
    """
    Clean, refactored Adaptive Mamba Block with proper separation of concerns.
    
    This version:
    1. Uses official Mamba implementation when available
    2. Encapsulates masking logic in MaskedLinear layers
    3. Removes PEFT logic (handled in training script)
    4. Focuses on core functionality
    """
    
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, enable_masking: bool = True, 
                 masking_config: Optional[Dict] = None, **kwargs):
        """
        Args:
            d_model: Model dimension
            d_state: SSM state dimension  
            d_conv: Local convolution width
            expand: Expansion factor
            enable_masking: Whether to enable learned masking (Pillar 2)
            masking_config: Configuration for masking layers
            **kwargs: Additional arguments (e.g., initial_permutation for reproducibility)
        """
        super().__init__()
        self.d_model = d_model
        self.enable_masking = enable_masking
        
        # Default masking configuration
        self.masking_config = masking_config or {
            'tau': 0.5,
            'init_sparsity': 0.5,
            'target_sparsity': 0.3,
            'sparsity_weight': 1e-5
        }
        
        # Pillar 1: Scan permutation optimizer
        self.scan_optimizer = VariableScanOptimizer(d_model, update_frequency=1000)
        
        # Set initial permutation for reproducibility if provided
        if 'initial_permutation' in kwargs:
            self.scan_optimizer.current_permutation = kwargs['initial_permutation']
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Core Mamba mixer
        if MAMBA_AVAILABLE:
            # Use official optimized implementation
            self.mixer = OfficialMamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
            
            # Apply masking to key layers if enabled
            if enable_masking:
                self._apply_masking_to_official_mamba()
        else:
            # Use fallback implementation
            self.mixer = FallbackMamba(d_model, d_state, d_conv, expand)
            
            # Apply masking to fallback layers if enabled
            if enable_masking:
                self._apply_masking_to_fallback_mamba()
    
    def _apply_masking_to_official_mamba(self):
        """Apply masking to official Mamba implementation."""
        # Replace key linear layers with masked versions
        if hasattr(self.mixer, 'in_proj'):
            self.mixer.in_proj = convert_to_masked(self.mixer.in_proj, self.masking_config)
        if hasattr(self.mixer, 'out_proj'):
            self.mixer.out_proj = convert_to_masked(self.mixer.out_proj, self.masking_config)
        
        # Apply masking to convolution if present
        if hasattr(self.mixer, 'conv1d'):
            original_conv = self.mixer.conv1d
            self.mixer.conv1d = MaskedConv1d(
                original_conv.in_channels,
                original_conv.out_channels,
                original_conv.kernel_size[0],
                stride=original_conv.stride[0],
                padding=original_conv.padding[0],
                groups=original_conv.groups,
                bias=original_conv.bias is not None,
                **self.masking_config
            )
            # Copy weights
            self.mixer.conv1d.weight.data.copy_(original_conv.weight.data)
            if original_conv.bias is not None:
                self.mixer.conv1d.bias.data.copy_(original_conv.bias.data)
    
    def _apply_masking_to_fallback_mamba(self):
        """Apply masking to fallback Mamba implementation."""
        self.mixer.in_proj = convert_to_masked(self.mixer.in_proj, self.masking_config)
        self.mixer.out_proj = convert_to_masked(self.mixer.out_proj, self.masking_config)
        
        # Convert conv1d to masked version
        original_conv = self.mixer.conv1d
        self.mixer.conv1d = MaskedConv1d(
            original_conv.in_channels,
            original_conv.out_channels, 
            original_conv.kernel_size[0],
            stride=original_conv.stride[0],
            padding=original_conv.padding[0],
            groups=original_conv.groups,
            bias=original_conv.bias is not None,
            **self.masking_config
        )
        # Copy weights
        self.mixer.conv1d.weight.data.copy_(original_conv.weight.data)
        if original_conv.bias is not None:
            self.mixer.conv1d.bias.data.copy_(original_conv.bias.data)
    
    def forward(self, x: torch.Tensor, update_scan: bool = False) -> torch.Tensor:
        """
        Clean forward pass with all three pillars.
        
        Args:
            x: Input tensor of shape (B, L, d_model)
            update_scan: Whether to update scan permutation
        
        Returns:
            Output tensor of shape (B, L, d_model)
        """
        # Pillar 1: Apply variable-aware scan
        if update_scan:
            self.scan_optimizer.update_permutation(x)
        
        pi = self.scan_optimizer.get_permutation().to(x.device)
        x_permuted = apply_permutation(x, pi)
        
        # Core computation with residual connection
        # The mixer handles all internal logic (projections, convolutions, SSM)
        # Masking is applied transparently within MaskedLinear layers
        residual = self.mixer(self.norm(x_permuted))
        
        # Pillar 1: Restore original order
        residual_restored = invert_permutation(residual, pi)
        
        # Residual connection
        return x + residual_restored
    
    def get_masking_statistics(self) -> Dict[str, Any]:
        """Get statistics from all masked layers."""
        stats = {}
        
        if self.enable_masking:
            # Collect stats from all MaskedLinear and MaskedConv1d layers
            for name, module in self.named_modules():
                if isinstance(module, (MaskedLinear, MaskedConv1d)):
                    stats[name] = module.get_statistics()
        
        return stats
    
    def get_sparsity_loss(self) -> torch.Tensor:
        """Get total sparsity regularization loss."""
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        
        if self.enable_masking:
            for module in self.modules():
                if isinstance(module, (MaskedLinear, MaskedConv1d)):
                    total_loss += module.get_sparsity_loss()
        
        return total_loss
    
    def get_importance_scores(self, method: str = 'mask_probability') -> Dict[str, float]:
        """
        Get importance scores for all layers.
        Used by training script for PEFT selection.
        """
        import warnings
        scores = {}
        fallback_used = False
        
        for name, module in self.named_modules():
            if isinstance(module, MaskedLinear):
                scores[name] = module.get_importance_score(method)
            elif isinstance(module, nn.Linear) and not self.enable_masking:
                # Fallback for non-masked layers
                scores[name] = module.weight.abs().mean().item()
                fallback_used = True
        
        if fallback_used:
            warnings.warn(
                "Fallback importance calculation used (based on weight magnitudes). "
                "Consider enabling masking for more principled importance estimation.",
                UserWarning
            )
        
        return scores
    
    def get_scan_permutation(self) -> torch.Tensor:
        """Get current scan permutation."""
        return self.scan_optimizer.get_permutation()

class AdaptiveMambaModel(nn.Module):
    """
    Complete Adaptive Mamba model with multiple blocks.
    """
    
    def __init__(self, vocab_size: int, d_model: int, n_layers: int,
                 block_config: Optional[Dict] = None, **kwargs):
        """
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            n_layers: Number of Mamba blocks
            block_config: Configuration for individual blocks
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Mamba blocks
        self.blocks = nn.ModuleList([
            AdaptiveMambaBlock(d_model, **(block_config or {}))
            for _ in range(n_layers)
        ])
        
        # Output head
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embedding.weight
    
    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass through the model."""
        x = self.embedding(input_ids)
        
        for i, block in enumerate(self.blocks):
            # Update scan permutation less frequently for efficiency
            update_scan = i == 0 and kwargs.get('update_scan', False)
            x = block(x, update_scan=update_scan)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits
    
    def get_total_regularization_loss(self) -> torch.Tensor:
        """Get total regularization loss from all blocks."""
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for block in self.blocks:
            total_loss += block.get_sparsity_loss()
        return total_loss