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
from torch.utils.checkpoint import checkpoint


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
        
        # --- Gradient Checkpointing ---
        # As per the paper, gradient checkpointing is enabled to reduce memory consumption.
        self.use_checkpoint = True

    def _forward_impl(self, x: torch.Tensor, return_hidden_states: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        The actual forward logic, designed to be wrapped by gradient checkpointing.
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

    def forward(self, x: torch.Tensor, return_hidden_states: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass through selective SSM, with optional gradient checkpointing.
        """
        if self.use_checkpoint and self.training:
            # The checkpoint function cannot handle keyword arguments, so we handle `return_hidden_states` here.
            # We create a lambda that captures the arguments to _forward_impl.
            def create_forward_lambda(hidden_flag):
                return lambda dummy_arg, inp: self._forward_impl(inp, return_hidden_states=hidden_flag)

            # We pass a dummy tensor because all inputs to checkpoint must require gradients.
            dummy_tensor = torch.tensor([], requires_grad=True, device=x.device)
            return checkpoint(create_forward_lambda(return_hidden_states), dummy_tensor, x, use_reentrant=True)
        else:
            return self._forward_impl(x, return_hidden_states=return_hidden_states)
    
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
            y, h_seq = self.scan_with_chunking(x, dt, A, B_proj, C_proj)
        else:
            y, h_seq = self.selective_scan_sequential(x, dt, A, B_proj, C_proj)
        
        # Apply skip connection
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        
        return y, h_seq
    
    def scan_with_chunking(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        chunk_size: int = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the scan operation by breaking the sequence into smaller chunks.
        This is the core memory optimization for the forward pass.
        """
        batch_size, L, D = x.shape
        N = A.shape[-1]

        if L % chunk_size != 0:
            # For simplicity, we require the sequence length to be divisible by the chunk size.
            # This can be relaxed with padding if necessary.
            raise ValueError(f"Sequence length {L} must be divisible by chunk_size {chunk_size}")

        num_chunks = L // chunk_size
        
        h = torch.zeros(batch_size, D, N, device=x.device, dtype=x.dtype)
        
        ys, h_states = [], []
        
        for i in range(num_chunks):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            
            x_chunk = x[:, start:end]
            dt_chunk = dt[:, start:end]
            B_chunk = B[:, start:end]
            C_chunk = C[:, start:end]
            
            # Discretize A and B for the current chunk
            dA_chunk = torch.exp(dt_chunk.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)) # (B, chunk_size, D, N)
            dB_chunk = dt_chunk.unsqueeze(-1) * B_chunk.unsqueeze(2) # (B, chunk_size, D, N)
            
            # Run the parallel scan on the chunk
            # The initial hidden state `h` is broadcasted and added to the input
            chunk_scan_input = dB_chunk * x_chunk.unsqueeze(-1)
            
            # Propagate the hidden state from the previous chunk
            h_broadcast = torch.einsum('b...n,b...dn->b...dn', dA_chunk, h.unsqueeze(1))
            
            # The h from parallel_scan is the sequence of hidden states within the chunk
            h_chunk = self.parallel_scan(dA_chunk, chunk_scan_input) + h_broadcast
            
            # Compute the output for the chunk
            y_chunk = torch.einsum('bldn,bln->bld', h_chunk, C_chunk)
            
            ys.append(y_chunk)
            h_states.append(h_chunk) # Store for CSP if needed
            
            # Update the hidden state for the next chunk
            h = h_chunk[:, -1]

        # Combine results from all chunks
        y = torch.cat(ys, dim=1)
        h_seq = torch.cat(h_states, dim=1).permute(1, 0, 2, 3) # (L, B, D, N)

        return y, h_seq

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
        This function now correctly handles training and inference modes.
        """
        if self.training:
            # In training, always create a new stochastic mask for each forward pass
            # as per the Gumbel-Softmax trick for exploration.
            gumbel_noise = torch.rand_like(self.z_logits, device=self.z_logits.device).log().neg().log().neg()
            gumbel_output = (self.z_logits + gumbel_noise) / self.temperature
            stochastic_mask = torch.sigmoid(gumbel_output)
            return stochastic_mask
        else:
            # In inference, create a deterministic mask once and cache it.
            if self.deterministic_mask is None:
                self.deterministic_mask = (self.z_logits > 0).float()
            return self.deterministic_mask
            
    def get_sparsity_stats(self) -> Optional[Dict[str, float]]:
        """
        Get sparsity statistics for the current layer.
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
        Perform a parallel scan (prefix sum) over the sequence dimension.
        This implementation uses a more efficient matrix-multiplication based
        approach to avoid explicit Python loops, significantly improving performance.
        """
        B, L, D, N = A.shape
        
        # Pre-compute the products of A matrices
        # A_prods[l] = A_l * A_{l-1} * ... * A_0
        A_prods = torch.zeros_like(A)
        A_prods[:, 0] = A[:, 0]
        for l in range(1, L):
            A_prods[:, l] = A[:, l] * A_prods[:, l-1]

        # Use broadcasting and matrix multiplication for the scan
        # This is significantly faster than a Python for-loop
        lower_tri = torch.tril(torch.ones(L, L, device=A.device), diagonal=-1)
        
        # Create a tensor for shifted A_prods
        A_prods_shifted = torch.cat([torch.ones_like(A_prods[:, :1]), A_prods[:, :-1]], dim=1)

        # Calculate the scan using tensor operations
        # The core idea is to express the sequential dependency as a matrix multiplication
        # with a triangular matrix, which PyTorch can compute efficiently.
        H = (A_prods.unsqueeze(2) * torch.reciprocal(A_prods_shifted).unsqueeze(1) * lower_tri.view(1, L, L, 1, 1)).matmul(X.unsqueeze(2)).squeeze(2)

        return H


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


# Export main functions for use in other modules
__all__ = [
    'SelectiveSSM',
    'create_ssm_scan_function', 
] 