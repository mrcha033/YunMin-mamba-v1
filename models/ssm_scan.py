"""
State Space Model (SSM) Selective Scan Implementation

This module implements the core selective scan algorithm for Mamba SSM models.
The selective scan is the computational heart of the SSM, enabling efficient
sequence modeling with sub-quadratic complexity.

Mathematical Foundation:
    h_t = A * h_{t-1} + B * x_t
    y_t = C * h_t + D * x_t

Where A, B, C are input-dependent (selective) and learned through the model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model implementation.
    
    This implements the core SSM computation with input-dependent parameters
    A, B, C that make the model "selective" - able to focus on relevant parts
    of the sequence while forgetting irrelevant information.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        bias: bool = False,
        conv_bias: bool = True,
        pscan: bool = True,
        use_sdm: bool = False,
        gumbel_temp: float = 1.0
    ):
        """
        Initialize Selective SSM.
        
        Args:
            d_model: Model dimension
            d_state: SSM state dimension  
            d_conv: Local convolution width
            expand: Block expansion factor
            dt_rank: Rank of Δ (discretization parameter)
            dt_min: Minimum value of Δ
            dt_max: Maximum value of Δ
            dt_init: How to initialize Δ
            dt_scale: Scaling factor for Δ
            bias: Whether to use bias in linear layers
            conv_bias: Whether to use bias in conv layer
            pscan: Whether to use parallel scan
            use_sdm: Whether to enable Structured Differentiable Masking
            gumbel_temp: Temperature for Gumbel-Sigmoid sampling if SDM is used
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        
        # Fix dt_rank to always be an integer
        if dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)
        elif isinstance(dt_rank, str):
            # Handle any other string cases
            self.dt_rank = math.ceil(self.d_model / 16)
        else:
            self.dt_rank = int(dt_rank)
            
        self.pscan = pscan
        self.use_sdm = use_sdm
        
        # Input projections
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        
        # Convolution for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        
        # SSM projections
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # Initialize special dt projection to preserve variance
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        
        # Initialize dt bias to be between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_min)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # Inverse of softplus
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        
        # S4D real initialization for A
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))  # Keep A_log in log space for stability
        
        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

        # --- SDM Specific Parameters ---
        if self.use_sdm:
            self.z_logits = nn.Parameter(torch.zeros(self.d_inner))
            self.temperature = gumbel_temp
            self.stochastic_mask = None
            self.deterministic_mask = None

    def forward(self, x: torch.Tensor, return_hidden_states: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass through selective SSM.
        
        Args:
            x: Input tensor of shape (B, L, D)
            return_hidden_states: If true, also returns the hidden state sequence.

        Returns:
            A tuple of (output, mask, hidden_states).
            - output: Output tensor of shape (B, L, D)
            - mask: The SDM mask used (if use_sdm=True), otherwise None.
            - hidden_states: Sequence of hidden states (if requested), otherwise None.
        """
        B, L, D = x.shape
        
        # Input projection: split into main path and gate
        xz = self.in_proj(x)  # (B, L, 2 * d_inner)
        x, z = xz.chunk(2, dim=-1)  # Each: (B, L, d_inner)
        
        # Convolution for local context
        x = x.transpose(1, 2)  # (B, d_inner, L)
        x = self.conv1d(x)[:, :, :L]  # Causal conv
        x = x.transpose(1, 2)  # (B, L, d_inner)
        
        # Activation
        x = F.silu(x)
        
        # --- SDM MASKING ---
        mask = None
        if self.use_sdm:
            mask = self._create_mask()
            x = x * mask.view(1, 1, -1)

        # SSM computation
        y, h_seq = self.selective_scan(x)
        
        # Gating mechanism
        z = F.silu(z)
        output = y * z
        
        # Output projection
        output = self.out_proj(output)
        
        hidden_states_output = h_seq if return_hidden_states else None
        
        return output, mask, hidden_states_output
    
    def selective_scan(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the selective scan operation.
        
        This is the core of the SSM computation where we:
        1. Generate input-dependent parameters A, B, C, Δ
        2. Discretize the continuous system  
        3. Perform the state space computation
        
        Args:
            x: Input tensor of shape (B, L, d_inner)
            
        Returns:
            Output tensor of shape (B, L, d_inner)
        """
        B, L, D = x.shape
        
        # Generate input-dependent SSM parameters
        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)
        
        # Split projections
        dt, B_proj, C_proj = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        # Process dt (discretization parameter)
        dt = F.softplus(self.dt_proj(dt))  # (B, L, d_inner)
        
        # Get A matrix (convert from log space)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        
        # Perform selective scan
        if self.pscan:
            y, h_seq = self.selective_scan_parallel(x, dt, A, B_proj, C_proj)
        else:
            y, h_seq = self.selective_scan_sequential(x, dt, A, B_proj, C_proj)
        
        # Apply skip connection
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        
        return y, h_seq
    
    def selective_scan_parallel(
        self, 
        x: torch.Tensor, 
        dt: torch.Tensor, 
        A: torch.Tensor, 
        B: torch.Tensor, 
        C: torch.Tensor
    ) -> torch.Tensor:
        """
        Parallel selective scan using associative scan.
        
        This is more efficient for training but requires more memory.
        """
        B_batch, L, D = x.shape
        N = A.shape[-1]
        
        # Discretize A and B
        # A_discrete = exp(dt * A)
        # B_discrete = dt * B
        dt = dt.unsqueeze(-1)  # (B, L, D, 1)
        A = A.unsqueeze(0).unsqueeze(0)  # (1, 1, D, N)
        
        # Discretization
        dA = torch.exp(dt * A)  # (B, L, D, N)
        dB = dt * B.unsqueeze(2)  # (B, L, D, N)
        
        # Reshape for scan
        dA = dA.contiguous()
        dB = dB.contiguous()
        x = x.unsqueeze(-1)  # (B, L, D, 1)
        
        # Parallel scan using custom implementation
        h = self.parallel_scan(dA, dB * x)  # (B, L, D, N)
        
        # Output computation: y = C * h
        # C is (B, L, N), h is (B, L, D, N) 
        # We need to sum over the N dimension: h * C -> (B, L, D)
        C = C.unsqueeze(2)  # (B, L, 1, N) 
        y = torch.sum(h * C, dim=-1)  # (B, L, D)
        
        return y, h # Return hidden states as well
    
    def selective_scan_sequential(
        self, 
        x: torch.Tensor, 
        dt: torch.Tensor, 
        A: torch.Tensor, 
        B: torch.Tensor, 
        C: torch.Tensor
    ) -> torch.Tensor:
        """
        Sequential selective scan, fully implemented to return hidden states.
        
        This is slower but uses less memory. Useful for inference or very long sequences.
        """
        B_batch, L, D = x.shape
        N = A.shape[-1]
        
        # Initialize hidden state
        h = torch.zeros(B_batch, D, N, device=x.device)
        
        y_list = []
        h_list = []

        # Iterate over sequence length
        for i in range(L):
            # Discretize A and B for current timestep
            dA = torch.exp(dt[:, i].unsqueeze(-1) * A)
            dB = dt[:, i].unsqueeze(-1) * B[:, i].unsqueeze(1)
            
            # Update hidden state (avoiding in-place operation)
            h = dA * h + dB * x[:, i].unsqueeze(-1)
            h_list.append(h)
            
            # Calculate output for current timestep
            current_y = torch.einsum('bdn,bdn->bd', h, C[:, i])
            y_list.append(current_y)
            
        y = torch.stack(y_list, dim=1)

        # Stack hidden states and permute to (L, B, D, N) for compatibility with CSP
        h_seq = torch.stack(h_list, dim=1) # (B, L, D, N)
        h_seq = h_seq.permute(1, 0, 2, 3) # (L, B, D, N)

        return y, h_seq

    def _create_mask(self) -> torch.Tensor:
        """
        Create a binary mask for SDM using Gumbel-Sigmoid.
        
        Returns:
            Tensor of shape (d_inner,) containing mask values
        """
        if not self.use_sdm:
            return None

        if self.training:
            # Gumbel-Sigmoid trick for differentiable binary sampling
            uniform_noise = torch.rand_like(self.z_logits)
            gumbel_noise = -torch.log(-torch.log(uniform_noise + 1e-8) + 1e-8)
            
            logits_with_noise = (self.z_logits + gumbel_noise) / self.temperature
            self.stochastic_mask = torch.sigmoid(logits_with_noise)
            return self.stochastic_mask
        else:
            # At inference time, use deterministic binary mask
            self.deterministic_mask = (self.z_logits > 0).float()
            return self.deterministic_mask
            
    def get_sparsity_stats(self) -> Optional[Dict[str, float]]:
        """
        Get current sparsity statistics for monitoring.
        (Only used if use_sdm=True)
        """
        if not self.use_sdm:
            return None
            
        with torch.no_grad():
            probs = torch.sigmoid(self.z_logits)
            deterministic_kept = (self.z_logits > 0).float()
            
            return {
                'mean_prob': probs.mean().item(),
                'std_prob': probs.std().item(),
                'deterministic_sparsity': 1.0 - deterministic_kept.mean().item(),
                'num_channels_kept': deterministic_kept.sum().item(),
                'total_channels': len(self.z_logits)
            }

    def parallel_scan(self, A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        Custom parallel scan implementation.
        
        Computes: h_t = A_t * h_{t-1} + X_t
        
        Args:
            A: Transition matrices of shape (B, L, D, N)
            X: Input terms of shape (B, L, D, N)
            
        Returns:
            h: Hidden states of shape (B, L, D, N)
        """
        B, L, D, N = A.shape
        
        # Convert to associative scan format
        # Each element is (A_i, X_i) representing h_i = A_i * h_{i-1} + X_i
        
        # For the associative scan, we need:
        # - A values stay the same
        # - X values are the inputs
        
        # Initialize output
        h = torch.zeros_like(X)
        
        # Simple sequential implementation (can be optimized to parallel)
        h[:, 0] = X[:, 0]  # h_0 = X_0 (no previous state)
        
        for t in range(1, L):
            h[:, t] = A[:, t] * h[:, t-1] + X[:, t]
        
        return h


def create_ssm_scan_function(
    d_model: int,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    **kwargs
) -> SelectiveSSM:
    """
    Factory function to create SSM scan with standard Mamba parameters.
    
    Args:
        d_model: Model dimension
        d_state: SSM state dimension
        d_conv: Convolution kernel size
        expand: Expansion factor
        **kwargs: Additional SSM parameters
        
    Returns:
        Configured SelectiveSSM module
    """
    return SelectiveSSM(
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        **kwargs
    )


class OptimizedSSMScan(nn.Module):
    """
    Hardware-optimized SSM scan with memory efficiency improvements.
    
    This version includes optimizations for:
    - Memory usage reduction
    - Numerical stability  
    - Hardware acceleration compatibility
    """
    
    def __init__(self, d_inner: int, d_state: int):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        
        # Use more stable initialization
        self.register_buffer('eps', torch.tensor(1e-6))
        
    def scan_with_checkpointing(
        self, 
        x: torch.Tensor, 
        dt: torch.Tensor, 
        A: torch.Tensor, 
        B: torch.Tensor, 
        C: torch.Tensor,
        checkpoint_segments: int = 4
    ) -> torch.Tensor:
        """
        Memory-efficient scan with gradient checkpointing.
        
        This reduces memory usage by recomputing forward pass segments
        during backward pass instead of storing all intermediate states.
        """
        B_batch, L, D = x.shape
        
        if checkpoint_segments <= 1:
            # Standard computation without checkpointing
            return self._basic_scan(x, dt, A, B, C)
        
        # Split sequence into segments for checkpointing
        segment_size = L // checkpoint_segments
        segments_x = []
        segments_dt = []
        segments_B = []
        segments_C = []
        
        for i in range(checkpoint_segments):
            start_idx = i * segment_size
            end_idx = min((i + 1) * segment_size, L)
            
            segments_x.append(x[:, start_idx:end_idx])
            segments_dt.append(dt[:, start_idx:end_idx])
            segments_B.append(B[:, start_idx:end_idx])
            segments_C.append(C[:, start_idx:end_idx])
        
        # Process segments with checkpointing
        outputs = []
        h = torch.zeros(B_batch, D, self.d_state, device=x.device, dtype=x.dtype)
        
        for seg_x, seg_dt, seg_B, seg_C in zip(segments_x, segments_dt, segments_B, segments_C):
            if self.training:
                # Use checkpointing during training
                from torch.utils.checkpoint import checkpoint
                seg_out, h = checkpoint(self._segment_scan, seg_x, seg_dt, A, seg_B, seg_C, h)
            else:
                # No checkpointing during inference
                seg_out, h = self._segment_scan(seg_x, seg_dt, A, seg_B, seg_C, h)
            
            outputs.append(seg_out)
        
        return torch.cat(outputs, dim=1)
    
    def _segment_scan(
        self, 
        x: torch.Tensor, 
        dt: torch.Tensor, 
        A: torch.Tensor, 
        B: torch.Tensor, 
        C: torch.Tensor,
        h_init: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a single segment of the sequence."""
        B_batch, L_seg, D = x.shape
        h = h_init.clone()
        
        outputs = []
        
        for t in range(L_seg):
            # Current timestep parameters
            dt_t = dt[:, t, :].unsqueeze(-1)  # (B, D, 1)
            B_t = B[:, t, :].unsqueeze(1)    # (B, 1, N)
            C_t = C[:, t, :].unsqueeze(1)    # (B, 1, N)
            x_t = x[:, t, :].unsqueeze(-1)   # (B, D, 1)
            
            # Discretization with numerical stability
            dA = torch.exp(torch.clamp(dt_t * A.unsqueeze(0), max=10.0))
            dB = dt_t * B_t
            
            # State update
            h = dA * h + dB * x_t
            
            # Output computation
            y_t = torch.sum(C_t * h, dim=-1)  # (B, D)
            outputs.append(y_t)
        
        return torch.stack(outputs, dim=1), h
    
    def _basic_scan(
        self, 
        x: torch.Tensor, 
        dt: torch.Tensor, 
        A: torch.Tensor, 
        B: torch.Tensor, 
        C: torch.Tensor
    ) -> torch.Tensor:
        """Basic scan without checkpointing."""
        B_batch, L, D = x.shape
        h = torch.zeros(B_batch, D, self.d_state, device=x.device, dtype=x.dtype)
        
        outputs = []
        
        for t in range(L):
            dt_t = dt[:, t, :].unsqueeze(-1)
            B_t = B[:, t, :].unsqueeze(1)
            C_t = C[:, t, :].unsqueeze(1)
            x_t = x[:, t, :].unsqueeze(-1)
            
            dA = torch.exp(torch.clamp(dt_t * A.unsqueeze(0), max=10.0))
            dB = dt_t * B_t
            
            h = dA * h + dB * x_t
            y_t = torch.sum(C_t * h, dim=-1)
            outputs.append(y_t)
        
        return torch.stack(outputs, dim=1)


# Export main functions for use in other modules
__all__ = [
    'SelectiveSSM',
    'create_ssm_scan_function', 
    'OptimizedSSMScan'
] 