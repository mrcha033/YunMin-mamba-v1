"""
Pillar 1: Variable-Aware Scan Implementation
Implements scan path optimization based on state variable correlation matrices.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional

def compute_correlation_matrix(hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Compute correlation matrix Σ_{i,j} = E[(h_i - μ_i)(h_j - μ_j)]
    
    Args:
        hidden_states: Tensor of shape (B, L, D) where D is state dimension
    
    Returns:
        Correlation matrix of shape (D, D)
    """
    B, L, D = hidden_states.shape
    # Flatten batch and sequence dimensions
    h_flat = hidden_states.view(-1, D)  # (B*L, D)
    
    # Center the data
    h_centered = h_flat - h_flat.mean(dim=0, keepdim=True)
    
    # Compute correlation matrix
    cov_matrix = torch.mm(h_centered.T, h_centered) / (B * L - 1)
    
    # Convert to correlation matrix
    std_diag = torch.sqrt(torch.diag(cov_matrix))
    correlation_matrix = cov_matrix / torch.outer(std_diag, std_diag)
    
    return correlation_matrix

def compute_cost_matrix(correlation_matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute path cost matrix: Cost(i,j) = 1 - |ρ_{i,j}|
    
    Args:
        correlation_matrix: Correlation matrix of shape (D, D)
    
    Returns:
        Cost matrix of shape (D, D)
    """
    return 1.0 - torch.abs(correlation_matrix)

def nearest_neighbor_tsp(cost_matrix: torch.Tensor, start_idx: int = 0) -> torch.Tensor:
    """
    Nearest Neighbor heuristic for TSP approximation.
    π* = arg min_π ∑_{t=1}^{d-1} Cost(π(t), π(t+1))
    
    Args:
        cost_matrix: Cost matrix of shape (D, D)
        start_idx: Starting index for the tour
    
    Returns:
        Permutation tensor representing optimal scan path
    """
    device = cost_matrix.device
    D = cost_matrix.shape[0]
    
    visited = torch.zeros(D, dtype=torch.bool, device=device)
    path = torch.zeros(D, dtype=torch.long, device=device)
    
    current = start_idx
    path[0] = current
    visited[current] = True
    
    for i in range(1, D):
        # Find nearest unvisited node
        costs = cost_matrix[current].clone()
        costs[visited] = float('inf')  # Mask visited nodes
        next_node = torch.argmin(costs)
        
        path[i] = next_node
        visited[next_node] = True
        current = next_node
    
    return path

def compute_scan_permutation(hidden_states: torch.Tensor, 
                           cost_fn=None) -> torch.Tensor:
    """
    Compute optimal scan permutation π* based on state correlations.
    
    Args:
        hidden_states: Sample hidden states for correlation analysis
        cost_fn: Optional custom cost function
    
    Returns:
        Optimal permutation tensor (on the same device as input)
    """
    # ### FIX ### Ensure all computations happen on the same device as input
    device = hidden_states.device
    
    # Step 1: Compute correlation matrix Σ
    correlation_matrix = compute_correlation_matrix(hidden_states)
    
    # Step 2: Compute cost matrix
    if cost_fn is None:
        cost_matrix = compute_cost_matrix(correlation_matrix)
    else:
        cost_matrix = cost_fn(correlation_matrix)
    
    # Step 3: Solve TSP using nearest neighbor heuristic
    optimal_permutation = nearest_neighbor_tsp(cost_matrix)
    
    # ### FIX ### Ensure the result is on the correct device
    if optimal_permutation.device != device:
        optimal_permutation = optimal_permutation.to(device)
    
    return optimal_permutation

def apply_permutation(x: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
    """
    Apply permutation π to input tensor along the last dimension.
    h_t' = h_t[π*]
    
    Args:
        x: Input tensor of shape (B, L, D)
        pi: Permutation tensor of shape (D,)
    
    Returns:
        Permuted tensor of shape (B, L, D)
    """
    # ### FIX ### Ensure permutation tensor is on the same device as input
    if pi.device != x.device:
        pi = pi.to(x.device)
    
    return x[:, :, pi]

def invert_permutation(x: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
    """
    Invert permutation to restore original order.
    
    Args:
        x: Permuted tensor of shape (B, L, D)
        pi: Original permutation tensor of shape (D,)
    
    Returns:
        Tensor restored to original order
    """
    # ### FIX ### Ensure permutation tensor is on the same device as input
    if pi.device != x.device:
        pi = pi.to(x.device)
    
    # Compute inverse permutation
    pi_inv = torch.zeros_like(pi, device=x.device)  # Use x.device for consistency
    pi_inv[pi] = torch.arange(len(pi), device=x.device, dtype=pi.dtype)
    
    return x[:, :, pi_inv]

class VariableScanOptimizer:
    """
    Utility class for managing scan path optimization across training.
    
    Supports two modes:
    - 'static': Compute permutation once during initialization
    - 'dynamic': Update permutation periodically during training
    
    Device-aware version that ensures all tensors are on the correct device.
    """
    
    def __init__(self, d_model: int, mode: str = 'dynamic', update_frequency: int = 1000):
        """
        Args:
            d_model: Model dimension
            mode: 'static' or 'dynamic' - controls when permutation is computed
            update_frequency: Steps between permutation updates (only for dynamic mode)
        """
        assert mode in ['static', 'dynamic'], f"Mode must be 'static' or 'dynamic', got {mode}"
        
        self.d_model = d_model
        self.mode = mode
        self.update_frequency = update_frequency
        self.step_count = 0
        self.is_initialized = False
        
        # Initialize internal state - starts on CPU, will be moved to correct device via to()
        self._permutation = torch.arange(d_model, dtype=torch.long)
        self.device = torch.device('cpu')  # Track current device
    
    def to(self, device) -> 'VariableScanOptimizer':
        """
        Move the internal state of the optimizer to the specified device.
        Mimics the behavior of nn.Module.to().
        """
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self._permutation = self._permutation.to(device)
        return self
    
    def update_permutation(self, hidden_states: torch.Tensor) -> bool:
        """
        Update scan permutation based on mode.
        
        Args:
            hidden_states: Sample hidden states for correlation analysis
            
        Returns:
            True if permutation was updated, False otherwise
        """
        # Defensive check: ensure optimizer is on same device as input
        input_device = hidden_states.device
        if self.device != input_device:
            print(f"Warning: Optimizer device ({self.device}) differs from input device ({input_device}). Auto-moving optimizer.")
            self.to(input_device)
        
        if self.mode == 'static':
            # Static mode: compute permutation only once
            if not self.is_initialized:
                # compute_scan_permutation returns tensor on same device as hidden_states
                self._permutation = compute_scan_permutation(hidden_states)
                # ### FIX ### Update internal device tracking
                self.device = self._permutation.device
                self.is_initialized = True
                return True
            return False
        
        elif self.mode == 'dynamic':
            # Dynamic mode: update permutation periodically
            self.step_count += 1
            
            if self.step_count % self.update_frequency == 0:
                # compute_scan_permutation returns tensor on same device as hidden_states
                self._permutation = compute_scan_permutation(hidden_states)
                # ### FIX ### Update internal device tracking
                self.device = self._permutation.device
                return True
            
            return False
    
    def get_permutation(self) -> torch.Tensor:
        """Get current scan permutation, ensuring it's on the correct device."""
        # Always return tensor on the current device, with defensive check
        if self._permutation.device != self.device:
            self._permutation = self._permutation.to(self.device)
        return self._permutation

    def get_inverse_permutation(self) -> torch.Tensor:
        """Get inverse of the current scan permutation."""
        perm = self.get_permutation()  # Always get tensor on correct device
        inverse = torch.empty_like(perm, device=perm.device)
        inverse[perm] = torch.arange(
            self.d_model, device=perm.device, dtype=perm.dtype
        )
        return inverse
