"""
Structured Differentiable Masking (SDM) SSM Implementation - Pillar 2

This module implements the SDM-enhanced SSM model that learns channel-wise sparsity
during pre-training using Gumbel-Sigmoid sampling for differentiable binary masking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class SDM_MambaBlock(nn.Module):
    """
    A MambaBlock augmented with Structured Differentiable Masking (SDM).
    This block learns which internal channels to prune during pre-training.
    
    [Pillar 2: SDM] The core innovation is the learnable sparsity logits 'z_c' that
    determine channel importance through data-driven optimization rather than heuristic pruning.
    """
    
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int = 2, gumbel_temp: float = 1.0):
        """
        Args:
            d_model (int): The main dimension of the model
            d_state (int): The dimension of the latent state 'h'
            d_conv (int): The kernel size of the 1D convolution
            expand (int): The expansion factor for the internal dimension
            gumbel_temp (float): Temperature for Gumbel-Sigmoid sampling
        """
        super().__init__()
        
        # --- Standard MambaBlock parameters ---
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        # Input-dependent Projections & Gating
        # [Pillar 2: SDM] The output channels of this layer (d_inner) are the target for
        #                 Structured Differentiable Masking via learnable logits 'z_c'
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)

        # Local Context Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # State Space Model (SSM) Core Parameters
        self.x_proj = nn.Linear(self.d_inner, self.d_state + self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.d_state, self.d_inner, bias=True)

        # A_log and D are the state-independent parameters of the SSM
        A_log = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A_log))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output Projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

        # --- SDM Specific Parameters ---
        # [Pillar 2: SDM] Learnable logits 'z_c' for each internal channel 'c'.
        # These are the core parameters that determine channel importance during training.
        # Initialized with zeros to give equal probability (0.5) for each channel initially.
        self.z_logits = nn.Parameter(torch.zeros(self.d_inner))
        self.temperature = gumbel_temp
        
        # Cache for the generated mask (used in loss calculation)
        self.stochastic_mask = None
        self.deterministic_mask = None

    def _create_mask(self) -> torch.Tensor:
        """
        Generates a differentiable or deterministic mask based on training mode.
        
        Returns:
            Tensor of shape (d_inner,) containing mask values
        """
        if self.training:
            # [Pillar 2: SDM] Gumbel-Sigmoid trick for differentiable binary sampling
            # This allows gradients to flow back to z_logits from the task loss
            
            # Sample Gumbel noise: gumbel_noise ~ Gumbel(0, 1)
            uniform_noise = torch.rand_like(self.z_logits)
            gumbel_noise = -torch.log(-torch.log(uniform_noise + 1e-8) + 1e-8)
            
            # Apply Gumbel-Sigmoid with temperature
            logits_with_noise = (self.z_logits + gumbel_noise) / self.temperature
            self.stochastic_mask = torch.sigmoid(logits_with_noise)
            
            return self.stochastic_mask
        else:
            # [Pillar 2: SDM] At inference time, use deterministic binary mask
            # A channel is kept if its learned logit is positive (sigmoid > 0.5)
            self.deterministic_mask = (self.z_logits > 0).float()
            return self.deterministic_mask

    def get_sparsity_stats(self) -> Dict[str, float]:
        """
        Get current sparsity statistics for monitoring.
        
        Returns:
            Dictionary with sparsity metrics
        """
        with torch.no_grad():
            # Current sigmoid probabilities
            probs = torch.sigmoid(self.z_logits)
            
            # Deterministic sparsity (what would be used at inference)
            deterministic_kept = (self.z_logits > 0).float()
            
            return {
                'mean_prob': probs.mean().item(),
                'std_prob': probs.std().item(),
                'deterministic_sparsity': 1.0 - deterministic_kept.mean().item(),
                'num_channels_kept': deterministic_kept.sum().item(),
                'total_channels': len(self.z_logits)
            }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with integrated SDM masking.
        
        Args:
            x: Input tensor of shape (B, L, d_model)
            
        Returns:
            Output tensor of shape (B, L, d_model)
        """
        B, L, _ = x.shape
        
        # 1. Input projection
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # x: (B, L, d_inner), z: (B, L, d_inner)

        # --- MASK APPLICATION ---
        # [Pillar 2: SDM] Apply learned channel-wise mask
        # This is the core of structured differentiable masking
        mask = self._create_mask()  # Shape: (d_inner,)
        
        # Broadcast mask over batch and sequence dimensions
        # mask.view(1, 1, -1) creates shape (1, 1, d_inner) for broadcasting
        x = x * mask.view(1, 1, -1)

        # 2. 1D convolution for local context
        x = x.transpose(1, 2)  # (B, d_inner, L)
        x = self.conv1d(x)[:, :, :L]
        x = x.transpose(1, 2)  # (B, L, d_inner)
        
        # 3. Activation and SSM scan
        x = F.silu(x)
        y = self.ssm_scan(x)

        # 4. Gating and output projection
        y = y * F.silu(z)
        output = self.out_proj(y)

        return output

    def ssm_scan(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the selective SSM scan.
        
        [Pillar 1: CSP] This function would benefit from the CSP permutation optimization.
        [Pillar 2: SDM] The input 'x' has already been sparsified by channel masking.
        
        Args:
            x: Input tensor of shape (B, L, d_inner)
            
        Returns:
            Output tensor of shape (B, L, d_inner)
        """
        # Placeholder for the complex selective scan logic
        # In practice, this would use the actual Mamba SSM implementation
        y = torch.randn_like(x)
        return y


class SDM_SSM(nn.Module):
    """
    The complete SDM-enhanced SSM model (M_SDM) that learns sparsity during pre-training.
    
    This model serves dual purposes:
    1. Learns a sparse, hardware-friendly architecture through structured masking
    2. Generates importance scores for SGH-PEFT fine-tuning (Pillar 3)
    """
    
    def __init__(self, d_model: int, n_layer: int, vocab_size: int, d_state: int, d_conv: int, gumbel_temp: float = 1.0):
        """
        Args:
            d_model: Model dimension
            n_layer: Number of layers
            vocab_size: Vocabulary size
            d_state: SSM state dimension
            d_conv: Convolution kernel size
            gumbel_temp: Temperature for Gumbel-Sigmoid sampling
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_layer = n_layer
        self.vocab_size = vocab_size
        self.d_state = d_state
        self.d_conv = d_conv
        self.gumbel_temp = gumbel_temp
        
        # Model components
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # [Pillar 2: SDM] Stack of SDM-enhanced MambaBlocks
        # Each layer learns its own channel importance pattern
        self.layers = nn.ModuleList([
            SDM_MambaBlock(
                d_model=d_model, 
                d_state=d_state, 
                d_conv=d_conv,
                gumbel_temp=gumbel_temp
            )
            for _ in range(n_layer)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the SDM-enhanced model.
        
        Args:
            input_ids: Input token IDs of shape (B, L)
            
        Returns:
            Logits of shape (B, L, vocab_size)
        """
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = x + layer(x)  # Residual connection
            
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits

    def get_layer_importance_scores(self) -> Dict[int, torch.Tensor]:
        """
        Extract learned importance scores from each layer.
        
        [Pillar 3: SGH-PEFT] These scores will guide the allocation of fine-tuning resources.
        Layers with higher average importance will receive LoRA adapters,
        while layers with lower importance will receive IA³ adapters.
        
        Returns:
            Dictionary mapping layer index to importance logits
        """
        importance_scores = {}
        
        for layer_idx, layer in enumerate(self.layers):
            # The z_logits represent learned channel importance
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
            stats = layer.get_sparsity_stats()
            stats['layer_idx'] = layer_idx
            layer_stats.append(stats)
            
            total_channels += stats['total_channels']
            total_kept += stats['num_channels_kept']
        
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
                # Force creation of deterministic mask
                layer._create_mask()
        
        return self

    def get_inference_model_size(self) -> Dict[str, int]:
        """
        Calculate the effective model size after applying learned sparsity.
        
        Returns:
            Dictionary with size metrics
        """
        total_params = 0
        effective_params = 0
        
        for name, param in self.named_parameters():
            total_params += param.numel()
            
            # For parameters affected by channel masking
            if 'in_proj.weight' in name and 'layers.' in name:
                # Extract layer index from parameter name like 'layers.0.in_proj.weight'
                layer_idx = int(name.split('.')[1])
                layer = self.layers[layer_idx]
                
                with torch.no_grad():
                    mask = (layer.z_logits > 0).float()
                    kept_channels = mask.sum().item()
                    
                    # in_proj weight shape is (d_inner * 2, d_model)
                    # We need to account for masking on the output dimension
                    d_model = param.shape[1]
                    effective_params += int(kept_channels * 2 * d_model)  # Both x and z projections
                    
            elif 'out_proj.weight' in name and 'layers.' in name:
                # Extract layer index
                layer_idx = int(name.split('.')[1])
                layer = self.layers[layer_idx]
                
                with torch.no_grad():
                    mask = (layer.z_logits > 0).float()
                    kept_channels = mask.sum().item()
                    
                    # out_proj weight shape is (d_model, d_inner)
                    d_model = param.shape[0]
                    effective_params += int(d_model * kept_channels)
                    
            elif 'conv1d.weight' in name and 'layers.' in name:
                # Conv1d is affected by channel masking
                layer_idx = int(name.split('.')[1])
                layer = self.layers[layer_idx]
                
                with torch.no_grad():
                    mask = (layer.z_logits > 0).float()
                    kept_channels = mask.sum().item()
                    
                    # conv1d weight shape depends on groups, but effectively we keep only kept_channels
                    kernel_size = param.shape[2]
                    effective_params += int(kept_channels * kernel_size)
                    
            else:
                # Other parameters not affected by channel masking
                effective_params += param.numel()
        
        return {
            'total_parameters': total_params,
            'effective_parameters': int(effective_params),
            'parameter_reduction': 1.0 - (effective_params / total_params)
        } 