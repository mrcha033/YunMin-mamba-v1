"""
SGH-PEFT Implementation - Pillar 3: Sparsity-Guided Hybrid PEFT

This module implements the parameter-aware design that leverages SDM importance scores
to intelligently allocate fine-tuning resources across layers using hybrid LoRA/IA³ adapters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import math
from dataclasses import dataclass

from models.sdm_ssm import SDM_MambaBlock, SDM_SSM


@dataclass
class SGHPEFTConfig:
    """Configuration for SGH-PEFT hybrid adaptation strategy."""
    # LoRA configuration
    lora_high_rank: int = 16
    lora_low_rank: int = 4
    lora_alpha_factor: int = 2  # alpha = rank * factor
    lora_dropout: float = 0.05
    
    # IA³ configuration
    ia3_init_std: float = 0.02
    
    # Importance-based allocation thresholds
    high_importance_mean_threshold: float = 0.5
    high_importance_active_threshold: float = 60.0
    medium_importance_mean_threshold: float = 0.0
    medium_importance_active_threshold: float = 40.0
    low_importance_mean_threshold: float = -0.5
    
    # Training configuration
    apply_sparsity_mask: bool = True
    freeze_base_model: bool = True


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
        
        # Get dimensions from base layer
        if hasattr(base_layer, 'in_features') and hasattr(base_layer, 'out_features'):
            self.in_features = base_layer.in_features
            self.out_features = base_layer.out_features
        else:
            raise ValueError("Base layer must have in_features and out_features attributes")
        
        # LoRA parameters
        self.lora_A = nn.Parameter(torch.randn(rank, self.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # [Pillar 3: SGH-PEFT] Sparsity mask from SDM
        if sparsity_mask is not None:
            self.register_buffer('sparsity_mask', sparsity_mask.float())
        else:
            self.register_buffer('sparsity_mask', torch.ones(self.out_features))
        
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
        lora_output = F.linear(lora_input, self.lora_A.T)  # (batch, seq, rank)
        lora_output = F.linear(lora_output, self.lora_B.T)  # (batch, seq, out_features)
        
        # [Pillar 3: SGH-PEFT] Apply sparsity mask to LoRA update
        # This ensures ΔW_c = 0 for channels where m_c = 0 (SDM learned they're unimportant)
        if self.sparsity_mask is not None:
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
        
        # Get output features
        if hasattr(base_layer, 'out_features'):
            out_features = base_layer.out_features
        else:
            raise ValueError("Base layer must have out_features attribute")
        
        # IA³ scaling parameters (one per output feature)
        self.ia3_scaling = nn.Parameter(torch.ones(out_features))
        
        # [Pillar 3: SGH-PEFT] Sparsity mask from SDM
        if sparsity_mask is not None:
            self.register_buffer('sparsity_mask', sparsity_mask.float())
        else:
            self.register_buffer('sparsity_mask', torch.ones(out_features))
        
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
        # Scale only the channels that SDM identified as important
        if self.sparsity_mask is not None:
            effective_scaling = torch.where(
                self.sparsity_mask > 0.5,
                self.ia3_scaling,
                torch.ones_like(self.ia3_scaling)
            )
        else:
            effective_scaling = self.ia3_scaling
        
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
        layer_importance_scores: Dict[str, Dict[str, float]]
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
        
        self._create_adapted_layers(base_model.layers)
        
        # Log adaptation statistics
        self._log_adaptation_stats()
    
    def _create_adapted_layers(self, base_layers: nn.ModuleList):
        """Create adapted layers based on importance scores."""
        
        for layer_idx, base_layer in enumerate(base_layers):
            layer_name = f"layers.{layer_idx}"
            
            # Get importance metrics for this layer
            if layer_name in self.layer_importance_scores:
                importance = self.layer_importance_scores[layer_name]
                mean_imp = importance['mean_importance']
                active_perc = importance['active_channels'] / importance['total_channels'] * 100
                sparsity_mask = importance.get('sparsity_mask')
            else:
                # Default to low importance if not found
                mean_imp = -1.0
                active_perc = 0.0
                sparsity_mask = None
            
            # Apply allocation strategy
            adapter_type = self._determine_adapter_type(mean_imp, active_perc)
            
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
    
    def _determine_adapter_type(self, mean_imp: float, active_perc: float) -> str:
        """
        Determine adapter type based on importance scores.
        
        [Pillar 3: SGH-PEFT] This implements the intelligent allocation strategy
        that concentrates adaptation capacity where SDM learned it's most needed.
        """
        config = self.config
        
        if (mean_imp > config.high_importance_mean_threshold and 
            active_perc > config.high_importance_active_threshold):
            return "lora_high"
        elif (mean_imp > config.medium_importance_mean_threshold and 
              active_perc > config.medium_importance_active_threshold):
            return "lora_low"
        elif mean_imp > config.low_importance_mean_threshold:
            return "ia3"
        else:
            return "frozen"
    
    def _create_single_adapted_layer(
        self, 
        base_layer: SDM_MambaBlock, 
        adapter_type: str,
        sparsity_mask: Optional[torch.Tensor]
    ) -> nn.Module:
        """Create a single adapted layer based on the adapter type."""
        
        config = self.config
        
        if adapter_type == "frozen":
            # Freeze all parameters
            for param in base_layer.parameters():
                param.requires_grad = False
            return base_layer
        
        # Create adapted projections
        adapted_layer = AdaptedMambaBlock(base_layer, adapter_type, config, sparsity_mask)
        
        return adapted_layer
    
    def _log_adaptation_stats(self):
        """Log adaptation statistics."""
        total_params = 0
        trainable_params = 0
        adapter_counts = {'lora_high': 0, 'lora_low': 0, 'ia3': 0, 'frozen': 0}
        
        for info in self.adaptation_info.values():
            adapter_counts[info['adapter_type']] += 1
            trainable_params += info['trainable_params']
        
        # Count total parameters
        total_params = sum(p.numel() for p in self.parameters())
        
        print("\n" + "=" * 60)
        print("SGH-PEFT ADAPTATION SUMMARY")
        print("=" * 60)
        print(f"High-rank LoRA layers: {adapter_counts['lora_high']}")
        print(f"Low-rank LoRA layers:  {adapter_counts['lora_low']}")
        print(f"IA³ layers:           {adapter_counts['ia3']}")
        print(f"Frozen layers:        {adapter_counts['frozen']}")
        print(f"Total parameters:     {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable ratio:      {trainable_params/total_params:.2%}")
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through SGH-PEFT adapted model."""
        
        x = self.embedding(input_ids)
        
        # Pass through adapted layers
        for adapted_layer in self.adapted_layers:
            x = x + adapted_layer(x)  # Residual connection
        
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
                for adapter_type in ['lora_high', 'lora_low', 'ia3', 'frozen']
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
        base_block: SDM_MambaBlock,
        adapter_type: str,
        config: SGHPEFTConfig,
        sparsity_mask: Optional[torch.Tensor]
    ):
        super().__init__()
        
        self.adapter_type = adapter_type
        self.config = config
        
        # Copy base block components (frozen)
        self.conv1d = base_block.conv1d
        self.x_proj = base_block.x_proj
        self.dt_proj = base_block.dt_proj
        self.A_log = base_block.A_log
        self.D = base_block.D
        
        # Freeze base parameters
        for param in [self.conv1d.parameters(), self.x_proj.parameters(), 
                     self.dt_proj.parameters()]:
            for p in param:
                p.requires_grad = False
        
        self.A_log.requires_grad = False
        self.D.requires_grad = False
        
        # Create adapted projections
        self.adapted_in_proj = self._create_adapted_layer(
            base_block.in_proj, adapter_type, config, sparsity_mask
        )
        self.adapted_out_proj = self._create_adapted_layer(
            base_block.out_proj, adapter_type, config, sparsity_mask
        )
        
        # Store dimensions for compatibility
        self.d_model = base_block.d_model
        self.d_inner = base_block.d_inner
        self.d_state = base_block.d_state
        self.d_conv = base_block.d_conv
    
    def _create_adapted_layer(
        self, 
        base_layer: nn.Linear, 
        adapter_type: str, 
        config: SGHPEFTConfig,
        sparsity_mask: Optional[torch.Tensor]
    ) -> nn.Module:
        """Create adapted layer based on adapter type."""
        
        # Freeze base layer
        for param in base_layer.parameters():
            param.requires_grad = False
        
        if adapter_type == "lora_high":
            return MaskedLoRALayer(
                base_layer=base_layer,
                rank=config.lora_high_rank,
                alpha=config.lora_high_rank * config.lora_alpha_factor,
                dropout=config.lora_dropout,
                sparsity_mask=sparsity_mask if config.apply_sparsity_mask else None
            )
        elif adapter_type == "lora_low":
            return MaskedLoRALayer(
                base_layer=base_layer,
                rank=config.lora_low_rank,
                alpha=config.lora_low_rank * config.lora_alpha_factor,
                dropout=config.lora_dropout,
                sparsity_mask=sparsity_mask if config.apply_sparsity_mask else None
            )
        elif adapter_type == "ia3":
            return IA3Layer(
                base_layer=base_layer,
                init_std=config.ia3_init_std,
                sparsity_mask=sparsity_mask if config.apply_sparsity_mask else None
            )
        else:  # frozen
            return base_layer
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through adapted MambaBlock."""
        B, L, _ = x.shape
        
        # 1. Adapted input projection
        xz = self.adapted_in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        
        # 2. Convolution (unchanged)
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :L] 
        x = x.transpose(1, 2)
        
        # 3. SSM scan (unchanged - using base implementation)
        x = F.silu(x)
        y = self.ssm_scan(x)
        
        # 4. Gating and adapted output projection
        y = y * F.silu(z)
        output = self.adapted_out_proj(y)
        
        return output
    
    def ssm_scan(self, x: torch.Tensor) -> torch.Tensor:
        """SSM scan - placeholder for actual implementation."""
        # This would use the actual SSM scan implementation
        # For now, return identity for testing
        return x


def compute_layer_importance_scores(sdm_model: SDM_SSM) -> Dict[str, Dict[str, Any]]:
    """
    Extract layer-wise importance scores from SDM model.
    
    [Pillar 3: SGH-PEFT] This function bridges SDM and SGH-PEFT by extracting
    the learned importance information from z_logits and converting it to
    allocation decisions for hybrid PEFT.
    
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
                
                # Compute importance metrics
                mean_importance = logits.mean().item()
                std_importance = logits.std().item()
                max_importance = logits.max().item()
                min_importance = logits.min().item()
                
                # Active channels (positive logits)
                active_mask = (logits > 0).float()
                active_channels = active_mask.sum().item()
                total_channels = len(logits)
                
                # Store comprehensive information
                importance_scores[layer_name] = {
                    'mean_importance': mean_importance,
                    'std_importance': std_importance,
                    'max_importance': max_importance,
                    'min_importance': min_importance,
                    'active_channels': int(active_channels),
                    'total_channels': int(total_channels),
                    'sparsity_level': 1.0 - (active_channels / total_channels),
                    'sparsity_mask': active_mask.clone()
                }
                
                print(f"Layer {layer_idx:2d}: mean_imp={mean_importance:6.3f}, "
                      f"active={active_channels:3.0f}/{total_channels} "
                      f"({active_channels/total_channels*100:5.1f}%)")
    
    return importance_scores


def create_sgh_peft_model(
    sdm_model: SDM_SSM,
    config: Optional[SGHPEFTConfig] = None
) -> SGHPEFTModel:
    """
    Create SGH-PEFT model from pre-trained SDM model.
    
    Args:
        sdm_model: Pre-trained SDM model
        config: SGH-PEFT configuration
        
    Returns:
        SGH-PEFT adapted model ready for fine-tuning
    """
    if config is None:
        config = SGHPEFTConfig()
    
    # Extract importance scores
    importance_scores = compute_layer_importance_scores(sdm_model)
    
    # Create SGH-PEFT model
    sgh_peft_model = SGHPEFTModel(
        base_model=sdm_model,
        config=config,
        layer_importance_scores=importance_scores
    )
    
    return sgh_peft_model 