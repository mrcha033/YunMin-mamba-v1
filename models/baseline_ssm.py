# models/baseline_ssm.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

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

    def forward(self, x: torch.Tensor):
        """
        Forward pass for the MambaBlock.
        Args:
            x (torch.Tensor): Input tensor of shape (B, L, D_model).
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
        y = self.ssm_scan(x)

        # 4. Gating and output projection
        y = y * F.silu(z)
        output = self.out_proj(y)

        return output

    def ssm_scan(self, x: torch.Tensor):
        """
        Performs the selective SSM scan. This is the computational core.
        [Pillar 1: CSP] The performance of this function is memory-bound. The core operation
                       involves sequentially updating the hidden state 'h_t' of size (B, d_inner, d_state).
                       The arbitrary memory layout of the 'd_state' dimension leads to poor cache
                       locality and high wall-clock latency. CSP directly addresses this by
                       reordering the 'd_state' dimension to maximize physical cache hits.
                       A high-performance implementation requires a custom CUDA kernel, but the
                       permutation logic can be applied to a PyTorch-based scan as well.
        """
        # A placeholder for the complex selective scan logic.
        # The actual implementation involves discretization of (A, B) and a parallel scan.
        # We will use an established implementation (e.g., from the 'mamba-ssm' package) here.
        # The key is that this function operates on a hidden state 'h' of dimension 'd_state',
        # which is the target of CSP's permutation.
        y = torch.randn_like(x) # Placeholder for actual scan output
        return y


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

    def forward(self, input_ids: torch.Tensor):
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = x + layer(x) # Residual connection
            
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits 