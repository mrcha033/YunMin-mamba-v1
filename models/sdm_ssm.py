"""
Structured Differentiable Masking (SDM) SSM Implementation - Pillar 2

This module implements the SDM-enhanced SSM model that learns channel-wise sparsity
during pre-training using Gumbel-Sigmoid sampling for differentiable binary masking.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple

# Import the refactored, high-performance SelectiveSSM module
from .ssm_scan import SelectiveSSM


class SDM_SSM(nn.Module):
    """
    The complete SDM-enhanced SSM model (M_SDM) that learns sparsity during pre-training.
    
    This model serves dual purposes:
    1. Learns a sparse, hardware-friendly architecture through structured masking
    2. Generates importance scores for SGH-PEFT fine-tuning (Pillar 3)
    
    This implementation uses the unified `SelectiveSSM` block with `use_sdm=True`
    to ensure high performance and accurate modeling.
    """
    
    def __init__(self, d_model: int, n_layer: int, vocab_size: int, d_state: int, d_conv: int, expand: int = 2, gumbel_temp: float = 1.0):
        """
        Args:
            d_model: Model dimension
            n_layer: Number of layers
            vocab_size: Vocabulary size
            d_state: SSM state dimension
            d_conv: Convolution kernel size
            expand: Expansion factor for inner dimension
            gumbel_temp: Temperature for Gumbel-Sigmoid sampling
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_layer = n_layer
        self.vocab_size = vocab_size
        
        # Model components
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # [Pillar 2: SDM] Stack of SDM-enhanced MambaBlocks
        # Each layer is a high-performance SelectiveSSM with SDM enabled.
        self.layers = nn.ModuleList([
            SelectiveSSM(
                d_model=d_model, 
                d_state=d_state, 
                d_conv=d_conv,
                expand=expand,
                use_sdm=True,  # <-- Critical change: Enable SDM in the core block
                gumbel_temp=gumbel_temp,
                pscan=True # Ensure parallel scan is used
            )
            for _ in range(n_layer)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, return_last_hidden_states: bool = False):
        """
        Forward pass for the full SDM model.
        
        Returns:
            A tuple of (logits, all_masks, last_hidden_states).
            - logits: The final model output.
            - all_masks: A list of masks from each layer, for sparsity regularization loss.
            - last_hidden_states: Hidden states from the final layer (if requested).
        """
        x = self.embedding(input_ids)
        
        all_masks = []
        last_hidden_states = None

        for i, layer in enumerate(self.layers):
            is_last_layer = i == len(self.layers) - 1
            
            # The refactored SelectiveSSM now returns (output, mask, hidden_states)
            y, mask, hidden_states = layer(x, return_hidden_states=is_last_layer and return_last_hidden_states)
            
            if mask is not None:
                all_masks.append(mask)
                
            if hidden_states is not None:
                last_hidden_states = hidden_states

            x = x + y # Residual connection
            
        x = self.norm(x)
        logits = self.lm_head(x)
        
        # Return all masks for sparsity loss calculation in the training loop
        return logits, all_masks, last_hidden_states

    def get_layer_importance_scores(self) -> Dict[int, torch.Tensor]:
        """
        Extract learned importance scores from each layer.
        
        [Pillar 3: SGH-PEFT] These scores will guide the allocation of fine-tuning resources.
        
        Returns:
            Dictionary mapping layer index to importance logits
        """
        importance_scores = {}
        
        for layer_idx, layer in enumerate(self.layers):
            if hasattr(layer, 'z_logits'):
                importance_scores[layer_idx] = layer.z_logits.detach().clone()
        
        return importance_scores

    def get_sparsity_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive sparsity statistics across all layers.
        
        Returns:
            Dictionary with overall sparsity metrics
        """
        layer_stats = []
        total_channels = 0
        total_kept = 0
        
        for layer_idx, layer in enumerate(self.layers):
            if hasattr(layer, 'get_sparsity_stats'):
                stats = layer.get_sparsity_stats()
                if stats:
                    stats['layer_idx'] = layer_idx
                    layer_stats.append(stats)
                    total_channels += stats['total_channels']
                    total_kept += stats['num_channels_kept']
        
        if total_channels == 0:
            return {'error': 'No layers with SDM enabled.'}

        overall_sparsity = 1.0 - (total_kept / total_channels)
        
        return {
            'layer_stats': layer_stats,
            'overall_sparsity': overall_sparsity,
            'total_channels': total_channels,
            'total_kept': total_kept,
            'compression_ratio': total_channels / total_kept if total_kept > 0 else float('inf')
        }
    
    def apply_learned_sparsity(self) -> 'SDM_SSM':
        """
        Convert the model to use deterministic masks based on learned importance.
        This creates the final sparse model for deployment.
        
        Returns:
            Self with deterministic masking enabled
        """
        self.eval()  # Switch to inference mode for deterministic masking
        
        with torch.no_grad():
            for layer in self.layers:
                if hasattr(layer, '_create_mask'):
                    layer._create_mask()
        
        return self

    def get_inference_model_size(self) -> Dict[str, int]:
        """
        Calculate the effective model size after applying learned sparsity.
        
        Returns:
            Dictionary with size metrics
        """
        # This method would need to be re-evaluated based on the new structure
        # of SelectiveSSM. For now, we assume a similar logic can be applied.
        # The core idea remains: count parameters but discount those in pruned channels.
        
        total_params = sum(p.numel() for p in self.parameters())
        effective_params = 0
        
        for name, param in self.named_parameters():
            is_masked = False
            # Check if this parameter is part of a layer that has a z_logits mask
            if 'layers.' in name:
                layer_idx_str = name.split('.')[1]
                if layer_idx_str.isdigit():
                    layer_idx = int(layer_idx_str)
                    layer = self.layers[layer_idx]

                    if layer.use_sdm:
                        # Parameters affected by d_inner masking
                        affected_params = ['conv1d.weight', 'conv1d.bias', 'dt_proj.weight', 'dt_proj.bias', 
                                         'A_log', 'D', 'out_proj.weight']
                        
                        # Parameters affected on a different dimension
                        # x_proj: (dt_rank + 2*d_state, d_inner) -> dim 1
                        # in_proj: (2*d_inner, d_model) -> dim 0

                        mask = (layer.z_logits > 0).float()
                        kept_channels_ratio = mask.mean()

                        if any(p in name for p in affected_params):
                            effective_params += param.numel() * kept_channels_ratio
                            is_masked = True
                        elif 'x_proj' in name or 'out_proj' in name: # dim 1 is d_inner
                            effective_params += param.numel() * kept_channels_ratio
                            is_masked = True
                        elif 'in_proj' in name: # dim 0 is d_inner*2
                            effective_params += param.numel() * kept_channels_ratio
                            is_masked = True

            if not is_masked:
                effective_params += param.numel()
        
        return {
            'total_parameters': total_params,
            'effective_parameters': int(effective_params),
            'parameter_reduction': 1.0 - (effective_params / total_params) if total_params > 0 else 0.0
        } 