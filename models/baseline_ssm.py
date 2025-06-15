# models/baseline_ssm.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple

# CSP integration
try:
    from .csp_permutation import run_csp_optimization, CSPConfig
    CSP_AVAILABLE = True
except ImportError:
    CSP_AVAILABLE = False

class MambaBlock(nn.Module):
    """
    A single block of the State Space Model, based on the Mamba architecture.
    This module is the primary subject of our three-pillar co-design optimization.
    """
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int = 2):
        """
        Args:
            d_model (int): The main dimension of the model.
            d_state (int): The dimension of the latent state 'h'. The paper refers to this as 'D' or 'N'.
            d_conv (int): The kernel size of the 1D convolution.
            expand (int): The expansion factor for the internal dimension.
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        # --- Input-dependent Projections & Gating ---
        # Projects input 'x' to an expanded dimension for the main path and the gate.
        # [Pillar 2: SDM] The output channels of this layer (d_inner) are the target for
        #                 Structured Differentiable Masking. We will introduce learnable logits 'z_c'
        #                 here to learn a channel-wise sparse structure during pre-training.
        # [Pillar 3: SGH-PEFT] During fine-tuning, the importance score 'α_l' of this layer,
        #                       derived from SDM's 'z_c', will determine whether to apply
        #                       LoRA or IA³.
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)

        # --- Local Context Convolution ---
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # --- State Space Model (SSM) Core Parameters ---
        # These parameters define the dynamics of the state transition h_t = f(h_{t-1}, x_t).
        self.x_proj = nn.Linear(self.d_inner, self.d_state + self.d_state, bias=False) # Projects for B and C
        self.dt_proj = nn.Linear(self.d_state, self.d_inner, bias=True) # Projects for Δt

        # A_log and D are the state-independent parameters of the SSM.
        # [Pillar 1: CSP] The state-interacting parameters (A, B, C) are the targets for
        #                 Correlation-based Scan Permutation. The optimal permutation 'π*' found
        #                 offline will be used to permanently reorder the 'd_state' dimension
        #                 of these parameters' weight tensors (e.g., A_new = A[π*, :][:, π*]).
        A_log = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A_log))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # --- Output Projection ---
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x: torch.Tensor, return_hidden_states: bool = False):
        """
        Forward pass for the MambaBlock.
        Args:
            x (torch.Tensor): Input tensor of shape (B, L, D_model).
            return_hidden_states (bool): If True, returns the sequence of hidden states.
        """
        B, L, _ = x.shape
        
        # 1. Input projection
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # x: (B, L, d_inner), z: (B, L, d_inner)

        # 2. 1D convolution for local context
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :L]
        x = x.transpose(1, 2)
        
        # 3. Activation and SSM scan
        x = F.silu(x)
        y, h_seq = self.ssm_scan(x) # h_seq will be returned

        # 4. Gating and output projection
        y = y * F.silu(z)
        output = self.out_proj(y)
        
        if return_hidden_states:
            return output, h_seq
        return output

    def ssm_scan(self, x: torch.Tensor):
        """
        Performs the selective SSM scan. This is the computational core.
        
        [Pillar 1: CSP] The performance of this function is memory-bound. The core operation
                       involves sequentially updating the hidden state 'h_t' of size (B, d_inner, d_state).
                       The arbitrary memory layout of the 'd_state' dimension leads to poor cache
                       locality and high wall-clock latency. CSP directly addresses this by
                       reordering the 'd_state' dimension to maximize physical cache hits.
        
        Args:
            x: Input tensor of shape (B, L, d_inner)
            
        Returns:
            Output tensor of shape (B, L, d_inner)
        """
        B, L, D = x.shape
        
        # Generate input-dependent SSM parameters
        x_dbl = self.x_proj(x)  # (B, L, 2*d_state)
        
        # Split into B and C projections
        B_proj, C_proj = torch.split(x_dbl, [self.d_state, self.d_state], dim=-1)
        
        # Process dt (discretization parameter) - simplified for baseline
        # In full Mamba, this would use a separate dt projection
        dt = torch.ones(B, L, self.d_inner, device=x.device, dtype=x.dtype) * 0.1
        
        # Get A matrix (convert from log space)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        
        # Perform selective scan
        y, h_seq = self.selective_scan_sequential(x, dt, A, B_proj, C_proj)
        
        # Apply skip connection
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        
        return y, h_seq
    
    def selective_scan_sequential(
        self, 
        x: torch.Tensor, 
        dt: torch.Tensor, 
        A: torch.Tensor, 
        B: torch.Tensor, 
        C: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sequential selective scan implementation.
        
        This implements the core SSM recurrence:
        h_t = exp(dt * A) * h_{t-1} + dt * B * x_t
        y_t = C * h_t
        
        Args:
            x: Input tensor (B, L, d_inner)
            dt: Discretization parameter (B, L, d_inner)
            A: Transition matrix (d_inner, d_state)
            B: Input projection (B, L, d_state)
            C: Output projection (B, L, d_state)
            
        Returns:
            Tuple of:
            - Output tensor (B, L, d_inner)
            - Hidden states sequence (L, B, d_inner, d_state)
        """
        B_batch, L, D = x.shape
        N = A.shape[-1]
        
        # Initialize hidden state
        h = torch.zeros(B_batch, D, N, device=x.device, dtype=x.dtype)
        
        # Output buffer
        ys = []
        hs = [] # Buffer for hidden states
        
        for t in range(L):
            # Get parameters for this timestep
            dt_t = dt[:, t, :]  # (B, D)
            B_t = B[:, t, :]    # (B, N)
            C_t = C[:, t, :]    # (B, N)
            x_t = x[:, t, :]    # (B, D)
            
            # Discretize A and B
            dA = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))  # (B, D, N)
            dB = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)  # (B, D, N)
            
            # State update: h = A * h + B * x
            h = dA * h + dB * x_t.unsqueeze(-1)  # (B, D, N)
            hs.append(h)
            
            # Output: y = C * h
            y_t = torch.einsum('bdn,bn->bd', h, C_t)  # (B, D)
            ys.append(y_t)
        
        y = torch.stack(ys, dim=1)  # (B, L, D)
        h_sequence = torch.stack(hs, dim=0) # (L, B, D, N)
        return y, h_sequence


class BaselineSSM(nn.Module):
    """
    The full baseline SSM, referred to as M_base in the research proposal.
    It consists of an embedding layer, a stack of MambaBlocks, and a final language model head.
    """
    def __init__(self, d_model: int, n_layer: int, vocab_size: int, d_state: int, d_conv: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # [Pillar 3: SGH-PEFT] The list of layers is the target for our Sparsity-Guided
        #                      Hybrid PEFT strategy. After SDM pre-training, we will iterate
        #                      through `self.layers`, compute the layer-level importance 'α_l'
        #                      for each block, and conditionally attach either a LoRA or IA³ adapter.
        self.layers = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv)
            for _ in range(n_layer)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, return_last_hidden_states: bool = False):
        """
        Args:
            input_ids (torch.Tensor): The input token IDs. Shape: (B, L)
            return_last_hidden_states (bool): If True, returns the hidden states 'h'
                                              from the final MambaBlock for CSP analysis.
        """
        x = self.embedding(input_ids)
        
        last_hidden_states = None

        for i, layer in enumerate(self.layers):
            is_last_layer = i == len(self.layers) - 1
            if return_last_hidden_states and is_last_layer:
                y, last_hidden_states = layer(x, return_hidden_states=True)
                x = x + y # Main residual connection
            else:
                y = layer(x)
                x = x + y # Main residual connection
            
        x = self.norm(x)
        logits = self.lm_head(x)
        
        if return_last_hidden_states:
            # Note: The returned hidden states are from BEFORE the final layer norm.
            return logits, last_hidden_states

        return logits

    def apply_csp_optimization(
        self, 
        dataloader: torch.utils.data.DataLoader,
        config: Optional['CSPConfig'] = None
    ) -> Tuple['BaselineSSM', Dict[str, Any]]:
        """
        Apply CSP (Correlation-based Scan Permutation) optimization.
        
        [Pillar 1: CSP] This optimizes the SSM state dimension ordering to
        maximize cache locality and reduce memory access latency.
        
        Args:
            dataloader: Data for correlation analysis
            config: CSP configuration (optional)
            
        Returns:
            Tuple of (optimized_model, optimization_results)
        """
        if not CSP_AVAILABLE:
            raise ImportError("CSP optimization requires the csp_permutation module")
        
        device = next(self.parameters()).device
        return run_csp_optimization(self, dataloader, config, device) 