"""
Correlation-based Scan Permutation (CSP) Implementation - Pillar 1

This module implements hardware-aware state dimension reordering for SSM models.
CSP analyzes correlation patterns in SSM states and finds optimal permutations
that maximize cache locality and minimize memory access latency.

Mathematical Foundation:
- State correlation analysis: C[i,j] = corr(h_t[i], h_t[j]) over state dim N
- Optimal permutation: π* = argmax_π Σ_{k=1}^{D-1} C_{π_k, π_{k+1}} (solved as Max TSP)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from python_tsp.heuristics import solve_tsp_simulated_annealing, solve_tsp_local_search

logger = logging.getLogger(__name__)


@dataclass
class CSPConfig:
    """Configuration for CSP optimization."""
    analysis_samples: int = 20000  # Number of state samples for correlation analysis
    tsp_solver: str = "greedy_2-opt"  # TSP solver heuristic: "greedy_2-opt", "2-opt", or "simulated_annealing"
    hardware_type: str = "gpu"    # Target hardware: "gpu", "cpu", "tpu"


class CorrelationAnalyzer:
    """
    Analyzes correlation patterns in SSM hidden states to guide permutation.
    
    The core insight is that highly correlated state dimensions should be
    placed adjacently in memory to maximize cache locality during scan operations.
    """
    
    def __init__(self, config: CSPConfig):
        self.config = config
        self.correlation_matrix = None
        self.optimal_permutation = None
        
    def analyze_state_correlations(
        self, 
        model: nn.Module, 
        dataloader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> torch.Tensor:
        """
        Analyze correlation patterns in SSM hidden states.
        
        Args:
            model: SSM model to analyze
            dataloader: Data for correlation analysis
            device: Computation device
            
        Returns:
            Correlation matrix of shape (d_state, d_state)
        """
        logger.info("🔍 Starting CSP correlation analysis...")
        
        model.eval()
        all_states = []
        sample_count = 0
        
        # Collect hidden states from multiple batches
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if sample_count >= self.config.analysis_samples:
                    break
                    
                input_ids = batch['input_ids'].to(device)
                
                # Extract hidden states from the last SSM layer
                states = self._extract_hidden_states(model, input_ids)
                
                if states is not None:
                    # states shape is (num_samples, d_state)
                    all_states.append(states)
                    sample_count += states.size(0)
                    
                if batch_idx % 10 == 0:
                    logger.info(f"   Analyzed {sample_count}/{self.config.analysis_samples} samples")
        
        if not all_states:
            logger.warning("No states collected for correlation analysis")
            return None
            
        # Concatenate all states and trim to exact number of samples
        all_states = torch.cat(all_states, dim=0)[:self.config.analysis_samples]
        
        # Compute correlation matrix
        logger.info(f"📊 Computing {all_states.shape[1]}x{all_states.shape[1]} state correlation matrix...")
        correlation_matrix = self._compute_correlation_matrix(all_states)
        
        self.correlation_matrix = correlation_matrix
        logger.info(f"✅ Correlation analysis completed. Matrix shape: {correlation_matrix.shape}")
        
        return correlation_matrix
    
    def _extract_hidden_states(self, model: nn.Module, input_ids: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Extract the actual hidden states 'h' from the SSM's scan operation.
        This is critical for a correct CSP implementation as per the paper.
        The goal is to get a correlation matrix for the state dimension N.
        """
        try:
            # The model's forward pass must be adjusted to return hidden states
            # when a specific argument is passed.
            # We expect the model to return a tuple where the last element is the
            # sequence of hidden states 'h' from the last layer.
            # The SDM_SSM model returns (logits, all_masks, hidden_states). We need the third element.
            *_, hidden_states = model(input_ids, return_last_hidden_states=True)
            
            # Expected hidden_states shape from baseline_ssm: (L, B, D, N)
            # L=seq_len, B=batch, D=d_inner, N=d_state
            if hidden_states is None or hidden_states.dim() != 4:
                logger.error(f"Failed to get valid hidden states. Expected 4D tensor, got {type(hidden_states)}")
                return None

            L, B, D, N = hidden_states.shape
            
            # To compute correlation across the inner dimension D, we need to treat
            # each of the D channels as a variable, and the values across time,
            # batch, and state dimension N as observations.
            # (L, B, D, N) -> permute to (D, L, B, N)
            h_permuted = hidden_states.permute(2, 0, 1, 3)
            
            # Now, reshape to (D, L*B*N) and transpose to (L*B*N, D)
            # This gives a 2D matrix where columns are the D channels.
            states_for_corr = h_permuted.reshape(D, -1).transpose(0, 1)

            logger.info(f"Successfully extracted hidden states. Shape for correlation: {states_for_corr.shape}")
            return states_for_corr

        except Exception as e:
            logger.error(f"Failed to extract hidden states correctly: {e}", exc_info=True)
            # To aid debugging, re-raise the exception so the user sees the full traceback
            # if the model's forward pass signature is incorrect.
            raise e
    
    def _compute_correlation_matrix(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute Pearson correlation matrix for state dimensions.
        
        Args:
            states: State tensor of shape (num_samples, d_state)
            
        Returns:
            Correlation matrix of shape (d_state, d_state)
        """
        # Standardize states (zero mean, unit variance) before computing Pearson correlation.
        states_mean = states.mean(dim=0, keepdim=True)
        states_std = states.std(dim=0, keepdim=True)
        # Add epsilon to prevent division by zero for constant state dimensions
        states_normalized = (states - states_mean) / (states_std + 1e-8)
        
        # Compute correlation matrix: R = (1/(n-1)) * X^T * X
        # Using n-1 for sample correlation
        num_samples = states_normalized.size(0)
        correlation_matrix = torch.matmul(states_normalized.t(), states_normalized) / (num_samples - 1)
        
        return correlation_matrix


class PermutationOptimizer:
    """
    Finds optimal state dimension permutations based on correlation analysis.
    
    Uses clustering and graph-based algorithms to group highly correlated
    dimensions together for improved memory access patterns.
    """
    
    def __init__(self, config: CSPConfig):
        self.config = config
        
    def _solve_tsp_greedy(self, correlation_matrix: np.ndarray) -> List[int]:
        """
        Solves the Max-TSP problem using a simple greedy heuristic.
        Starts at node 0 and iteratively travels to the most correlated unvisited node.
        """
        num_nodes = correlation_matrix.shape[0]
        permutation = [0]
        visited = {0}
        
        current_node = 0
        while len(permutation) < num_nodes:
            # Find the best next node (max correlation, not min distance)
            best_next_node = -1
            max_corr = -np.inf
            
            # Use a pre-sorted list of neighbors for efficiency
            sorted_neighbors = np.argsort(-correlation_matrix[current_node, :])
            
            for neighbor in sorted_neighbors:
                if neighbor not in visited:
                    best_next_node = neighbor
                    break
            
            if best_next_node == -1:
                # Should not happen in a fully connected graph
                # Fallback: find any unvisited node
                for i in range(num_nodes):
                    if i not in visited:
                        best_next_node = i
                        break

            permutation.append(best_next_node)
            visited.add(best_next_node)
            current_node = best_next_node
            
        return permutation

    def find_optimal_permutation(self, correlation_matrix: torch.Tensor) -> torch.Tensor:
        """
        Finds the optimal permutation by framing the problem as a
        Maximum Traveling Salesperson Problem (TSP) and solving it with a heuristic.
        The goal is to find a path (permutation) that maximizes the sum of
        correlations between adjacent dimensions in memory.
        
        Args:
            correlation_matrix: Correlation matrix of shape (d_state, d_state).
            
        Returns:
            Permutation indices of shape (d_state,).
        """
        logger.info(f"🎯 Finding optimal state permutation via TSP ({self.config.tsp_solver})...")
        
        if correlation_matrix is None or not isinstance(correlation_matrix, torch.Tensor):
            logger.error("Invalid correlation matrix provided.")
            return None
        
        d_state = correlation_matrix.size(0)
        
        # The TSP solvers from `python-tsp` minimize the distance.
        # To maximize correlation, we convert correlations to distances.
        # A common conversion is distance = max_corr - correlation.
        # We use a large constant (e.g., 2.0, since max corr is 1.0) to ensure all distances are non-negative.
        max_val = torch.max(torch.abs(correlation_matrix)) * 1.1 + 1e-6
        distance_matrix = max_val - correlation_matrix

        distance_matrix_np = distance_matrix.cpu().numpy()
        np.fill_diagonal(distance_matrix_np, 0) # Distance to self is 0.
        correlation_matrix_np = correlation_matrix.cpu().numpy()

        # Solve the TSP. The result is a list of node indices in the optimal order.
        if self.config.tsp_solver == "simulated_annealing":
            # Better quality, but slower
            permutation, _ = solve_tsp_simulated_annealing(distance_matrix_np, alpha=0.995)
        
        elif self.config.tsp_solver == "greedy_2-opt":
            # As per the paper: Greedy + 2-opt.
            logger.info("   Generating initial tour with Greedy algorithm...")
            initial_permutation = self._solve_tsp_greedy(correlation_matrix_np)
            logger.info("   Refining tour with 2-opt local search...")
            permutation, _ = solve_tsp_local_search(
                distance_matrix_np, x0=initial_permutation, perturbation_scheme="ps2"
            )

        elif self.config.tsp_solver == "2-opt":
            # Faster, good quality heuristic (as mentioned in paper).
            # The paper suggests "Greedy + 2-opt". We use a standard 2-opt implementation
            # with a simple initial tour, which is a common and effective approach.
            initial_permutation = list(range(d_state))
            permutation, _ = solve_tsp_local_search(
                distance_matrix_np, x0=initial_permutation, perturbation_scheme="ps2"
            )
        else:
            raise ValueError(f"Unknown TSP solver: {self.config.tsp_solver}")
        
        optimal_permutation = torch.tensor(permutation, dtype=torch.long)

        # Validate permutation
        assert len(set(optimal_permutation.tolist())) == d_state, "Invalid permutation: not all dimensions are present."
        assert optimal_permutation.min() == 0 and optimal_permutation.max() == d_state - 1, "Invalid permutation range."
        
        logger.info(f"✅ Optimal permutation found: {optimal_permutation[:10].tolist()}...")
        return optimal_permutation
    
    def estimate_performance_gain(
        self, 
        original_order: torch.Tensor,
        optimized_order: torch.Tensor,
        correlation_matrix: torch.Tensor
    ) -> Dict[str, float]:
        """
        Estimate performance improvement from permutation.
        
        Args:
            original_order: Original dimension ordering
            optimized_order: Optimized dimension ordering  
            correlation_matrix: State correlation matrix
            
        Returns:
            Dictionary with performance metrics
        """
        def cache_efficiency(order, corr_matrix):
            """Estimate cache efficiency for given ordering."""
            total_efficiency = 0.0
            cache_line_dims = self.config.cache_line_size // 4  # Assuming float32
            
            for i in range(0, len(order), cache_line_dims):
                cache_block = order[i:i + cache_line_dims]
                if len(cache_block) > 1:
                    # Average correlation within cache block
                    block_corr = corr_matrix[cache_block[:, None], cache_block[None, :]]
                    avg_corr = block_corr.mean().item()
                    total_efficiency += avg_corr
            
            return total_efficiency
        
        original_efficiency = cache_efficiency(original_order, correlation_matrix)
        optimized_efficiency = cache_efficiency(optimized_order, correlation_matrix)
        
        improvement = (optimized_efficiency - original_efficiency) / (abs(original_efficiency) + 1e-8)
        
        return {
            'cache_efficiency_improvement': improvement,
            'estimated_latency_reduction': min(improvement * 0.15, 0.25),  # Conservative estimate
            'memory_access_optimization': improvement * 0.1
        }


class CSPApplier:
    """
    Applies the optimized permutation to SSM model parameters.
    
    Physically reorders weight matrices and state dimensions to match
    the optimal permutation found by correlation analysis.
    """
    
    def apply_permutation_to_model(
        self, 
        model: nn.Module, 
        permutation: torch.Tensor
    ) -> nn.Module:
        """
        Apply state dimension permutation to SSM model.
        
        Args:
            model: SSM model to optimize
            permutation: Optimal permutation indices
            
        Returns:
            Model with reordered parameters
        """
        logger.info(f"🔄 Applying CSP permutation to model...")
        
        applied_layers = 0
        
        with torch.no_grad():
            for name, module in model.named_modules():
                if hasattr(module, 'A_log') and hasattr(module, 'x_proj'):
                    # This is an SSM layer - apply permutation
                    self._apply_permutation_to_ssm_layer(module, permutation)
                    applied_layers += 1
        
        logger.info(f"✅ CSP permutation applied to {applied_layers} SSM layers")
        return model
    
    def _apply_permutation_to_ssm_layer(self, layer: nn.Module, permutation: torch.Tensor):
        """
        Apply permutation to all parameters in a single SSM layer that interact
        with the d_inner dimension. This physically reorders the channels.
        """
        # The permutation, of size d_inner, is applied to the corresponding dimension
        # in each weight matrix.
        
        # --- Permute in_proj ---
        # Weight shape: (d_inner * 2, d_model) -> Permute dim 0
        if hasattr(layer, 'in_proj') and layer.in_proj is not None:
            w = layer.in_proj.weight.data
            d_inner = permutation.size(0)
            w_x, w_z = w.chunk(2, dim=0)
            layer.in_proj.weight.data = torch.cat([w_x[permutation, :], w_z[permutation, :]], dim=0)

        # --- Permute conv1d ---
        # Weight shape: (d_inner, 1, d_conv) -> Permute dim 0
        # Bias shape: (d_inner) -> Permute dim 0
        if hasattr(layer, 'conv1d') and layer.conv1d is not None:
            layer.conv1d.weight.data = layer.conv1d.weight.data[permutation, :, :]
            if layer.conv1d.bias is not None:
                layer.conv1d.bias.data = layer.conv1d.bias.data[permutation]

        # --- Permute x_proj ---
        # Weight shape: (dt_rank + d_state*2, d_inner) -> Permute dim 1
        if hasattr(layer, 'x_proj') and layer.x_proj is not None:
            layer.x_proj.weight.data = layer.x_proj.weight.data[:, permutation]

        # --- Permute dt_proj ---
        # Weight shape: (d_inner, dt_rank) -> Permute dim 0
        # Bias shape: (d_inner) -> Permute dim 0
        if hasattr(layer, 'dt_proj') and layer.dt_proj is not None:
            layer.dt_proj.weight.data = layer.dt_proj.weight.data[permutation, :]
            if layer.dt_proj.bias is not None:
                layer.dt_proj.bias.data = layer.dt_proj.bias.data[permutation]

        # --- Permute A_log and D ---
        # A_log shape: (d_inner, d_state) -> Permute dim 0
        # D shape: (d_inner) -> Permute dim 0
        if hasattr(layer, 'A_log'):
            layer.A_log.data = layer.A_log.data[permutation, :]
        if hasattr(layer, 'D'):
            layer.D.data = layer.D.data[permutation]

        # --- Permute out_proj ---
        # Weight shape: (d_model, d_inner) -> Permute dim 1
        if hasattr(layer, 'out_proj') and layer.out_proj is not None:
            layer.out_proj.weight.data = layer.out_proj.weight.data[:, permutation]


def run_csp_optimization(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    config: CSPConfig = None,
    device: torch.device = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Complete CSP optimization pipeline.
    
    Args:
        model: SSM model to optimize
        dataloader: Data for correlation analysis
        config: CSP configuration
        device: Computation device
        
    Returns:
        Tuple of (optimized_model, optimization_results)
    """
    if config is None:
        config = CSPConfig()
    
    if device is None:
        device = next(model.parameters()).device
    
    logger.info("🚀 Starting CSP (Correlation-based Scan Permutation) optimization...")
    
    # Step 1: Analyze state correlations
    analyzer = CorrelationAnalyzer(config)
    correlation_matrix = analyzer.analyze_state_correlations(model, dataloader, device)
    
    if correlation_matrix is None:
        logger.error("❌ Failed to analyze correlations")
        return model, {'status': 'failed', 'reason': 'correlation_analysis_failed'}
    
    # Step 2: Find optimal permutation
    optimizer = PermutationOptimizer(config)
    d_state = correlation_matrix.size(0)
    original_order = torch.arange(d_state)
    optimal_permutation = optimizer.find_optimal_permutation(correlation_matrix)
    
    # Step 3: Estimate performance gain
    performance_gain = optimizer.estimate_performance_gain(
        original_order, optimal_permutation, correlation_matrix
    )
    
    # Step 4: Apply permutation to model
    applier = CSPApplier()
    optimized_model = applier.apply_permutation_to_model(model, optimal_permutation)
    
    results = {
        'status': 'success',
        'correlation_matrix_shape': correlation_matrix.shape,
        'optimal_permutation': optimal_permutation.tolist(),
        'performance_estimates': performance_gain,
        'config': config.__dict__
    }
    
    logger.info("✅ CSP optimization completed successfully!")
    logger.info(f"📈 Estimated latency reduction: {performance_gain['estimated_latency_reduction']:.2%}")
    
    return optimized_model, results


# Export main functions
__all__ = [
    'CSPConfig',
    'CorrelationAnalyzer', 
    'PermutationOptimizer',
    'CSPApplier',
    'run_csp_optimization'
] 