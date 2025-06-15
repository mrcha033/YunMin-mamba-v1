"""
SGH-PEFT Implementation - Pillar 3: Sparsity-Guided Hybrid PEFT

This module implements the parameter-aware design that leverages SDM importance scores
to intelligently allocate fine-tuning resources across layers using hybrid LoRA/IA³ adapters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import numpy as np

# Import the high-performance SelectiveSSM module which replaces the old SDM_MambaBlock
from .ssm_scan import SelectiveSSM
from .sdm_ssm import SDM_SSM


@dataclass
class SGHPEFTConfig:
    """Configuration for SGH-PEFT hybrid adaptation strategy."""
    # LoRA configuration
    lora_rank: int = 16
    lora_alpha_factor: int = 2  # alpha = rank * factor
    lora_dropout: float = 0.05
    
    # IA³ configuration
    ia3_init_std: float = 0.02
    
    # Importance-based allocation threshold (as a percentile)
    # As per the paper: layers above this threshold get LoRA, others get IA³.
    # e.g., 75.0 means layers with importance in the top 25% get LoRA.
    peft_threshold_percentile: float = 75.0
    
    # Training configuration
    apply_sparsity_mask: bool = True
    freeze_base_model: bool = True


class StandardLoRALayer(nn.Module):
    """A standard LoRA layer without any sparsity masking."""
    def __init__(self, base_layer: nn.Module, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        for param in self.base_layer.parameters():
            param.requires_grad = False
            
        if hasattr(base_layer, 'in_features') and hasattr(base_layer, 'out_features'):
            self.in_features = base_layer.in_features
            self.out_features = base_layer.out_features
        else:
            raise ValueError(f"Base layer {type(base_layer)} must have in_features and out_features attributes")
            
        self.lora_A = nn.Parameter(torch.randn(rank, self.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base_layer(x)
        lora_input = self.dropout(x)
        
        original_shape = lora_input.shape
        lora_input_flat = lora_input.reshape(-1, self.in_features)
        
        lora_intermediate = torch.matmul(lora_input_flat, self.lora_A.T)
        lora_output_flat = torch.matmul(lora_intermediate, self.lora_B.T)
        
        lora_output = lora_output_flat.reshape(*original_shape[:-1], self.out_features)
        
        return base_output + lora_output * self.scaling


class MaskedLoRALayer(nn.Module):
    """
    Custom LoRA layer that applies SDM sparsity masks to updates.
    
    [Pillar 3: SGH-PEFT] This is the core innovation - LoRA updates are masked
    to respect the learned sparsity structure from SDM, ensuring synergy between
    data-aware sparsity and parameter-efficient fine-tuning.
    """
    
    def __init__(
        self,
        base_layer: nn.Module,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        sparsity_mask: Optional[torch.Tensor] = None
    ):
        super().__init__()
        
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Freeze base layer parameters
        for param in self.base_layer.parameters():
            param.requires_grad = False
        
        # Get dimensions from base layer
        if hasattr(base_layer, 'in_features') and hasattr(base_layer, 'out_features'):
            self.in_features = base_layer.in_features
            self.out_features = base_layer.out_features
        elif isinstance(base_layer, nn.Linear):
             self.in_features = base_layer.in_features
             self.out_features = base_layer.out_features
        else:
            raise ValueError(f"Base layer {type(base_layer)} must have in_features and out_features attributes")
        
        # LoRA parameters
        self.lora_A = nn.Parameter(torch.randn(rank, self.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # [Pillar 3: SGH-PEFT] Sparsity mask from SDM.
        # This mask is pre-processed to match the layer's output dimension.
        if sparsity_mask is not None and sparsity_mask.numel() == self.out_features:
            self.register_buffer('sparsity_mask', sparsity_mask.float())
        else:
            # Default to no masking if dimensions mismatch or mask is not provided
            self.register_buffer('sparsity_mask', torch.ones(self.out_features, dtype=torch.float))
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize LoRA parameters using Kaiming initialization."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with masked LoRA update.
        
        Args:
            x: Input tensor
            
        Returns:
            Output with masked LoRA adaptation
        """
        # Base layer forward pass
        base_output = self.base_layer(x)
        
        # LoRA forward pass
        lora_input = self.dropout(x)
        
        # Reshape input for matrix multiplication
        original_shape = lora_input.shape
        lora_input_flat = lora_input.reshape(-1, self.in_features)  # (batch*seq, in_features)
        
        # First LoRA transformation: (batch*seq, in_features) @ (in_features, rank) -> (batch*seq, rank)
        lora_intermediate = torch.matmul(lora_input_flat, self.lora_A.T)  # (batch*seq, rank)
        
        # Second LoRA transformation: (batch*seq, rank) @ (rank, out_features) -> (batch*seq, out_features)
        lora_output_flat = torch.matmul(lora_intermediate, self.lora_B.T)  # (batch*seq, out_features)
        
        # Reshape back to original shape
        lora_output = lora_output_flat.reshape(*original_shape[:-1], self.out_features)
        
        # [Pillar 3: SGH-PEFT] Apply sparsity mask to LoRA update
        # The mask is now guaranteed to match the output dimension
        lora_output = lora_output * self.sparsity_mask.view(1, 1, -1)
        
        # Scale and combine
        lora_output = lora_output * self.scaling
        
        return base_output + lora_output


class IA3Layer(nn.Module):
    """
    IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations) layer.
    
    [Pillar 3: SGH-PEFT] Used for medium-low importance layers where we want
    minimal parameter overhead but some adaptation capability.
    """
    
    def __init__(
        self,
        base_layer: nn.Module,
        init_std: float = 0.02,
        sparsity_mask: Optional[torch.Tensor] = None
    ):
        super().__init__()
        
        self.base_layer = base_layer
        
        # Freeze base layer parameters
        for param in self.base_layer.parameters():
            param.requires_grad = False
        
        # Get output features
        if hasattr(base_layer, 'out_features'):
            out_features = base_layer.out_features
        elif isinstance(base_layer, nn.Linear):
            out_features = base_layer.out_features
        else:
            raise ValueError(f"Base layer {type(base_layer)} must have out_features attribute")
        
        # IA³ scaling parameters (one per output feature)
        self.ia3_scaling = nn.Parameter(torch.ones(out_features))
        
        # [Pillar 3: SGH-PEFT] Sparsity mask from SDM
        # Mask is pre-processed to match the layer's output dimension.
        if sparsity_mask is not None and sparsity_mask.numel() == out_features:
            self.register_buffer('sparsity_mask', sparsity_mask.float())
        else:
            self.register_buffer('sparsity_mask', torch.ones(out_features, dtype=torch.float))
        
        # Initialize scaling factors
        self.reset_parameters(init_std)
    
    def reset_parameters(self, init_std: float):
        """Initialize IA³ parameters."""
        nn.init.normal_(self.ia3_scaling, mean=1.0, std=init_std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with IA³ scaling.
        
        Args:
            x: Input tensor
            
        Returns:
            Output with IA³ scaling adaptation
        """
        # Base layer forward pass
        base_output = self.base_layer(x)
        
        # [Pillar 3: SGH-PEFT] Apply sparsity-aware IA³ scaling 
        # The mask is pre-applied to the scaling factors.
        effective_scaling = torch.where(
            self.sparsity_mask > 0.5,
            self.ia3_scaling,
            torch.ones_like(self.ia3_scaling)
        )
        
        return base_output * effective_scaling.view(1, 1, -1)


class SGHPEFTModel(nn.Module):
    """
    SGH-PEFT model that wraps an SDM model with hybrid LoRA/IA³ adapters
    allocated based on learned importance scores.
    
    [Pillar 3: SGH-PEFT] This represents the final M_final model that combines:
    - M_base: Baseline SSM architecture
    - CSP: Hardware-optimized scan permutation (Pillar 1) 
    - SDM: Data-learned channel sparsity (Pillar 2)
    - SGH-PEFT: Parameter-efficient hybrid adaptation (Pillar 3)
    """
    
    def __init__(
        self,
        base_model: SDM_SSM,
        config: SGHPEFTConfig,
        layer_importance_scores: Dict[str, Dict[str, Any]]
    ):
        super().__init__()
        
        self.config = config
        self.layer_importance_scores = layer_importance_scores
        
        # Store base model components (frozen)
        self.embedding = base_model.embedding
        self.norm = base_model.norm
        self.lm_head = base_model.lm_head
        
        # Freeze base model if specified
        if config.freeze_base_model:
            for param in [self.embedding.parameters(), self.norm.parameters(), self.lm_head.parameters()]:
                for p in param:
                    p.requires_grad = False
        
        # Create adapted layers with hybrid PEFT
        self.adapted_layers = nn.ModuleList()
        self.adaptation_info = {}
        
        # Calculate actual threshold values from percentiles
        self._calculate_percentile_thresholds()
        
        self._create_adapted_layers(base_model.layers)
        
        # Log adaptation statistics
        self._log_adaptation_stats()
    
    def _calculate_percentile_thresholds(self):
        """Calculate the single importance score threshold from the configured percentile."""
        if not self.layer_importance_scores:
            # Fallback if scores are not available
            self.peft_threshold_val_ = 0.5
            return

        all_scores = [s['mean_importance'] for s in self.layer_importance_scores.values()]

        if not all_scores:
            self.peft_threshold_val_ = 0.5
            return
            
        self.peft_threshold_val_ = np.percentile(all_scores, self.config.peft_threshold_percentile)
        
        print(f"SGH-PEFT Threshold (from {self.config.peft_threshold_percentile}th percentile): "
              f"LoRA if importance >= {self.peft_threshold_val_:.3f}, else IA³.")

    def _create_adapted_layers(self, base_layers: nn.ModuleList):
        """Create adapted layers based on importance scores."""
        
        for layer_idx, base_layer in enumerate(base_layers):
            layer_name = f"layers.{layer_idx}"
            
            # Get importance metrics for this layer
            if layer_name in self.layer_importance_scores:
                importance = self.layer_importance_scores[layer_name]
                mean_imp = importance['mean_importance']
                active_perc = importance.get('active_channels', 0) / importance.get('total_channels', 1) * 100
                sparsity_mask = importance.get('sparsity_mask')
            else:
                # Default to low importance if not found
                mean_imp = -1.0
                active_perc = 0.0
                sparsity_mask = None
            
            # Apply allocation strategy
            adapter_type = self._determine_adapter_type(mean_imp)
            
            # Create adapted layer
            adapted_layer = self._create_single_adapted_layer(
                base_layer, adapter_type, sparsity_mask
            )
            
            self.adapted_layers.append(adapted_layer)
            
            # Store adaptation info
            self.adaptation_info[layer_idx] = {
                'adapter_type': adapter_type,
                'mean_importance': mean_imp,
                'active_percentage': active_perc,
                'trainable_params': sum(p.numel() for p in adapted_layer.parameters() if p.requires_grad)
            }
    
    def _determine_adapter_type(self, mean_imp: float) -> str:
        """
        Determine adapter type based on importance score and the calculated percentile threshold.
        This follows the paper's logic: high-importance layers get LoRA, others get IA³.
        """
        if mean_imp >= self.peft_threshold_val_:
            return "lora"
        else:
            return "ia3"
    
    def _create_single_adapted_layer(
        self, 
        base_layer: SelectiveSSM, 
        adapter_type: str,
        sparsity_mask: Optional[torch.Tensor]
    ) -> nn.Module:
        """Create a single adapted layer based on the adapter type."""
        
        if adapter_type == "frozen":
            # Freeze all parameters
            for param in base_layer.parameters():
                param.requires_grad = False
            return base_layer
        
        # Create an adapted version of the SelectiveSSM block
        adapted_block = AdaptedMambaBlock(base_layer, adapter_type, self.config, sparsity_mask)
        
        return adapted_block
    
    def _log_adaptation_stats(self):
        """Log adaptation statistics."""
        total_params = 0
        trainable_params = 0
        adapter_counts = {'lora': 0, 'ia3': 0, 'frozen': 0}
        
        for info in self.adaptation_info.values():
            # Handle cases where a key might not exist if logic changes
            adapter_type = info.get('adapter_type', 'frozen')
            if adapter_type in adapter_counts:
                adapter_counts[adapter_type] += 1
            trainable_params += info['trainable_params']
        
        # Count total parameters
        total_params = sum(p.numel() for p in self.parameters())
        
        print("\n" + "=" * 60)
        print("SGH-PEFT ADAPTATION SUMMARY")
        print("=" * 60)
        print(f"LoRA layers:   {adapter_counts['lora']}")
        print(f"IA³ layers:    {adapter_counts['ia3']}")
        print(f"Frozen layers: {adapter_counts['frozen']}")
        print(f"Total parameters:     {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable ratio:      {(trainable_params/total_params)*100:.4f}%")
    
    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass through SGH-PEFT adapted model."""
        
        x = self.embedding(input_ids)
        
        # Pass through adapted layers
        for adapted_layer in self.adapted_layers:
            # Each adapted layer is a full block, so no residual connection here
            x = adapted_layer(x)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits
    
    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get detailed adaptation summary for analysis."""
        return {
            'adaptation_info': self.adaptation_info,
            'total_layers': len(self.adapted_layers),
            'adapter_distribution': {
                adapter_type: sum(1 for info in self.adaptation_info.values() 
                                if info['adapter_type'] == adapter_type)
                for adapter_type in ['lora', 'ia3', 'frozen']
            },
            'total_trainable_params': sum(info['trainable_params'] 
                                        for info in self.adaptation_info.values())
        }


class AdaptedMambaBlock(nn.Module):
    """
    MambaBlock with SGH-PEFT adapters applied to key projections.
    
    [Pillar 3: SGH-PEFT] This block applies the appropriate adapter type
    (LoRA high/low rank, IA³) to the input and output projections while
    preserving the core SSM functionality.
    """
    
    def __init__(
        self,
        base_block: SelectiveSSM,
        adapter_type: str,
        config: SGHPEFTConfig,
        sparsity_mask: Optional[torch.Tensor]
    ):
        super().__init__()
        
        self.base_block = base_block # This contains the core SSM logic
        self.adapter_type = adapter_type
        self.config = config
        
        # Freeze the entire base block
        for param in self.base_block.parameters():
            param.requires_grad = False

        # Create adapted projections for in_proj and out_proj
        # The sparsity mask for d_inner is used for both
        self.adapted_in_proj = self._create_adapted_projection(
            self.base_block.in_proj, sparsity_mask
        )
        # For out_proj, the adapter input is d_inner, so we use that part of the mask
        self.adapted_out_proj = self._create_adapted_projection(
            self.base_block.out_proj, sparsity_mask
        )

    def _create_adapted_projection(
        self, 
        base_projection: nn.Linear, 
        d_inner_mask: Optional[torch.Tensor]
    ) -> nn.Module:
        """Create an adapted projection layer (LoRA or IA3)."""

        if self.adapter_type == "frozen":
            return base_projection

        # Determine the correct mask for the specific projection
        mask_to_use = None
        if self.config.apply_sparsity_mask and d_inner_mask is not None:
            # in_proj has out_features = d_inner * 2
            if base_projection.out_features == d_inner_mask.numel() * 2:
                mask_to_use = d_inner_mask.repeat(2)
            # out_proj has in_features = d_inner
            elif base_projection.in_features == d_inner_mask.numel():
                mask_to_use = d_inner_mask
        
        if self.adapter_type == "lora":
            return MaskedLoRALayer(
                base_layer=base_projection,
                rank=self.config.lora_rank,
                alpha=self.config.lora_rank * self.config.lora_alpha_factor,
                dropout=self.config.lora_dropout,
                sparsity_mask=mask_to_use
            )
        elif self.adapter_type == "ia3":
            # IA3 acts on the output of a layer, so the mask should match out_features
            if self.config.apply_sparsity_mask and d_inner_mask is not None:
                 if base_projection.out_features == d_inner_mask.numel() * 2: # in_proj case
                     mask_to_use = d_inner_mask.repeat(2)
                 elif base_projection.out_features == self.base_block.d_model: # out_proj case
                     # IA3 on out_proj scales the final output, so no mask is applied here
                     # as the sparsity was already handled internally by the LoRA on d_inner.
                     # This is a design choice to avoid double-masking.
                     mask_to_use = None

            return IA3Layer(
                base_layer=base_projection,
                init_std=self.config.ia3_init_std,
                sparsity_mask=mask_to_use
            )
        else: # Should not happen due to outer logic
            return base_projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the adapted block, reusing the base block's logic."""
        
        # Temporarily replace the base block's projections with our adapted ones
        original_in_proj = self.base_block.in_proj
        original_out_proj = self.base_block.out_proj
        
        self.base_block.in_proj = self.adapted_in_proj
        self.base_block.out_proj = self.adapted_out_proj
        
        # Run the forward pass of the original, high-performance SelectiveSSM block
        # The base_block itself is frozen, but the adapted layers inside are trainable
        # The base_block's forward will call our adapted layers.
        # It returns (output, mask, hidden_states), but we only need the output here.
        output, _, _ = self.base_block(x)

        # Restore original projections to keep the base_block clean
        self.base_block.in_proj = original_in_proj
        self.base_block.out_proj = original_out_proj

        # The residual connection is applied outside, in the SGHPEFTModel's forward loop
        return x + output

def compute_layer_importance_scores(sdm_model: SDM_SSM) -> Dict[str, Dict[str, Any]]:
    """
    Extract layer-wise importance scores from SDM model based on sigmoid of z_logits.
    
    [Pillar 3: SGH-PEFT] This function bridges SDM and SGH-PEFT by extracting
    the learned importance information and converting it to allocation decisions.
    The importance score is defined as the mean of the sigmoid of the logits,
    as per the paper's description of layer-level score α_l.
    
    Args:
        sdm_model: Pre-trained SDM model with learned z_logits
        
    Returns:
        Dictionary mapping layer names to importance metrics
    """
    print("Computing layer-wise importance scores from SDM logits...")
    
    importance_scores = {}
    
    with torch.no_grad():
        for layer_idx, layer in enumerate(sdm_model.layers):
            if hasattr(layer, 'z_logits'):
                layer_name = f"layers.{layer_idx}"
                
                logits = layer.z_logits.detach().float()
                # 논문에 명시된 대로 sigmoid(z_c)를 사용하여 중요도 점수 s_c를 계산
                scores = torch.sigmoid(logits)
                
                # Compute importance metrics from scores, not logits
                mean_importance = scores.mean().item() # This is now α_l
                std_importance = scores.std().item()
                max_importance = scores.max().item()
                min_importance = scores.min().item()
                
                # Active channels (positive logits -> score > 0.5)
                active_mask = (logits > 0).float()
                active_channels = active_mask.sum().item()
                total_channels = len(logits)
                
                # Store comprehensive information
                importance_scores[layer_name] = {
                    'mean_importance': mean_importance, # This is now α_l
                    'std_importance': std_importance,
                    'max_importance': max_importance,
                    'min_importance': min_importance,
                    'active_channels': int(active_channels),
                    'total_channels': int(total_channels),
                    'sparsity_level': 1.0 - (active_channels / total_channels),
                    'sparsity_mask': active_mask.clone()
                }
                
                print(f"Layer {layer_idx:2d}: mean_imp(α_l)={mean_importance:6.3f}, "
                      f"active={active_channels:3.0f}/{total_channels} "
                      f"({active_channels/total_channels*100:5.1f}%)")
    
    return importance_scores

def create_sgh_peft_model(
    sdm_model: SDM_SSM,
    peft_config: Optional[SGHPEFTConfig] = None
) -> SGHPEFTModel:
    """
    High-level factory function to create an SGH-PEFT model from a pre-trained SDM model.
    This function orchestrates the entire process as described in the paper:
    1. Computes layer-wise importance scores from the SDM model.
    2. Initializes an SGH-PEFT configuration.
    3. Creates and returns the SGH-PEFT model.
    
    Args:
        sdm_model: A pre-trained SDM_SSM model instance.
        peft_config: Optional SGHPEFTConfig. If None, a default config is used.
        
    Returns:
        An SGHPEFTModel ready for fine-tuning.
    """
    print("🚀 Creating SGH-PEFT model from pre-trained SDM model...")
    
    # 1. Compute layer importance scores from the SDM model
    layer_importance = compute_layer_importance_scores(sdm_model)
    
    # 2. Initialize PEFT config
    if peft_config is None:
        peft_config = SGHPEFTConfig()
        
    # 3. Create the SGH-PEFT model
    sgh_model = SGHPEFTModel(
        base_model=sdm_model,
        config=peft_config,
        layer_importance_scores=layer_importance
    )
    
    print("✅ SGH-PEFT model created successfully.")
    
    return sgh_model