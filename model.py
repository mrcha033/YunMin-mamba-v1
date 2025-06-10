"""
YunMin Mamba Model
Adaptive Hybrid-PEFT Mamba
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

# Hugging Face integration
try:
    from transformers import PretrainedConfig
    HF_AVAILABLE = True
except ImportError:
    # Fallback base class if Transformers not available
    class PretrainedConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
        
        def to_dict(self):
            return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        
        def save_pretrained(self, save_directory):
            import json
            import os
            os.makedirs(save_directory, exist_ok=True)
            with open(os.path.join(save_directory, 'config.json'), 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        
        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
            import json
            import os
            config_path = os.path.join(pretrained_model_name_or_path, 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                config_dict.update(kwargs)
                return cls(**config_dict)
            else:
                return cls(**kwargs)
    
    HF_AVAILABLE = False


class AdaptiveMambaConfig(PretrainedConfig):
    """
    Configuration class for AdaptiveMambaModel.
    
    Inherits from PretrainedConfig to enable Hugging Face ecosystem integration
    including save_pretrained(), from_pretrained(), and PEFT compatibility.
    """
    
    model_type = "adaptive_mamba"
    
    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 768,
        n_layers: int = 12,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        # Masking configuration (Pillar 2)
        enable_masking: bool = True,
        masking_tau: float = 0.5,
        masking_init_sparsity: float = 0.5,
        masking_target_sparsity: float = 0.3,
        sparsity_weight: float = 1e-5,
        # Scan optimization (Pillar 1)
        enable_scan_optimization: bool = True,
        scan_mode: str = 'dynamic',  # 'static' or 'dynamic'
        scan_update_frequency: int = 1000,
        # General model settings
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = True,
        # Compatibility aliases
        **kwargs
    ):
        """
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            n_layers: Number of transformer blocks
            d_state: SSM state dimension
            d_conv: Local convolution width  
            expand: Expansion factor for inner dimension
            enable_masking: Enable learned masking (Pillar 2)
            masking_tau: Temperature for Gumbel-Sigmoid sampling
            masking_init_sparsity: Initial sparsity level
            masking_target_sparsity: Target sparsity for regularization
            sparsity_weight: Weight for sparsity loss
            enable_scan_optimization: Enable variable-aware scan (Pillar 1)
            scan_mode: 'static' (computed once) or 'dynamic' (periodic updates)
            scan_update_frequency: Steps between scan updates (dynamic mode only)
        """
        # Store all configuration
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        
        # Pillar configurations
        self.enable_masking = enable_masking
        self.masking_tau = masking_tau
        self.masking_init_sparsity = masking_init_sparsity
        self.masking_target_sparsity = masking_target_sparsity
        self.sparsity_weight = sparsity_weight
        
        self.enable_scan_optimization = enable_scan_optimization
        self.scan_mode = scan_mode
        self.scan_update_frequency = scan_update_frequency
        
        # Token configuration
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        
        # Compatibility aliases for PEFT and other libraries
        self.hidden_size = d_model
        self.num_hidden_layers = n_layers
        self.intermediate_size = int(expand * d_model)
        
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )
    
    def get_masking_config(self) -> Dict[str, Any]:
        """Get masking configuration for block initialization."""
        return {
            'tau': self.masking_tau,
            'init_sparsity': self.masking_init_sparsity,
            'target_sparsity': self.masking_target_sparsity,
            'sparsity_weight': self.sparsity_weight
        }
    
    def get_scan_config(self) -> Dict[str, Any]:
        """Get scan optimization configuration."""
        return {
            'mode': self.scan_mode,
            'update_frequency': self.scan_update_frequency
        }

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
    """Simplified recurrent SSM used when ``mamba_ssm`` is unavailable.

    This module loosely emulates the official Mamba mixer using a basic
    recurrent formulation. It does **not** implement the selective (S6)
    state mechanism and therefore is **not** a performance-equivalent
    substitute for the real implementation. The state ``s`` of dimension
    ``d_state`` is updated at each timestep via ``s <- A @ s + B @ x`` and
    contributes to the output projection.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.d_state = d_state

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

        # Parameters for the simple SSM: s <- A @ s + B @ x
        self.A = nn.Parameter(torch.eye(d_state))
        self.B = nn.Parameter(torch.randn(d_state, self.d_inner) * 0.01)
        self.C = nn.Parameter(torch.randn(d_state, self.d_inner) * 0.01)

        # Registered buffer so state persists across calls but is not a parameter
        self.register_buffer("ssm_state", torch.zeros(1, d_state))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the simplified SSM mixer."""
        B, L, _ = x.shape

        # Ensure state matches current batch size
        if self.ssm_state.size(0) != B:
            self.ssm_state = torch.zeros(B, self.d_state, device=x.device, dtype=x.dtype)

        # Project and split
        xz = self.in_proj(x)  # (B, L, 2 * d_inner)
        x_part, z = xz.chunk(2, dim=-1)  # Each: (B, L, d_inner)

        # Apply convolution
        x_conv = self.conv1d(x_part.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv = F.silu(x_conv)

        outputs = []
        state = self.ssm_state
        for t in range(L):
            inp = x_part[:, t]
            # Update recurrent state
            state = torch.matmul(state, self.A.T) + torch.matmul(inp, self.B.T)
            gate = torch.sigmoid(z[:, t])
            state_proj = torch.matmul(state, self.C)
            out = (x_conv[:, t] + state_proj) * gate
            outputs.append(out.unsqueeze(1))

        # Persist state for subsequent calls
        self.ssm_state = state.detach()

        y = torch.cat(outputs, dim=1)
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
        scan_mode = kwargs.get('scan_mode', 'dynamic')
        scan_update_freq = kwargs.get('scan_update_frequency', 1000)
        self.scan_optimizer = VariableScanOptimizer(
            d_model, 
            mode=scan_mode,
            update_frequency=scan_update_freq
        )
        
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
    Complete Adaptive Mamba model with multiple blocks implementing the 
    Self-Optimizing paradigm through three synergistic pillars.
    
    Example usage with importance-driven PEFT allocation:
    
    ```python
    # Create model with masking enabled
    model = AdaptiveMambaModel(
        vocab_size=10000,
        d_model=256,
        n_layers=6,
        block_config={
            'enable_masking': True,
            'masking_config': {'tau': 0.5, 'target_sparsity': 0.5}
        }
    )
    
    # Extract importance scores for PEFT allocation
    importance_scores = {}
    for i, block in enumerate(model.blocks):
        block_scores = block.get_importance_scores(method='mask_probability')
        importance_scores[f'block_{i}'] = block_scores
    
    # Use scores for adaptive PEFT application
    from layers.peft_manager import PEFTManager
    peft_manager = PEFTManager(
        importance_threshold=0.7,
        peft_application_ratio=0.2
    )
    peft_manager.apply_peft_to_model(model, importance_scores)
    
    # Model is now self-optimized with LoRA on high-importance layers
    # and IA³ on medium-importance layers
    ```
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
        
        # Use standardized HuggingFace-compatible config
        if isinstance(block_config, AdaptiveMambaConfig):
            self.config = block_config
        else:
            self.config = AdaptiveMambaConfig(
                vocab_size=vocab_size,
                d_model=d_model,
                n_layers=n_layers,
                **(block_config or {})
            )
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Mamba blocks - use config for consistent settings
        block_kwargs = {
            'd_state': self.config.d_state,
            'd_conv': self.config.d_conv,
            'expand': self.config.expand,
            'enable_masking': self.config.enable_masking,
            'masking_config': self.config.get_masking_config(),
            'scan_mode': self.config.scan_mode,
            'scan_update_frequency': self.config.scan_update_frequency
        }
        
        self.blocks = nn.ModuleList([
            AdaptiveMambaBlock(self.config.d_model, **block_kwargs)
            for _ in range(self.config.n_layers)
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

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """Compatibility helper for PEFT generation APIs."""
        return {"input_ids": input_ids}
    
    def get_total_regularization_loss(self) -> torch.Tensor:
        """Get total regularization loss from all blocks."""
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for block in self.blocks:
            total_loss += block.get_sparsity_loss()
        return total_loss
