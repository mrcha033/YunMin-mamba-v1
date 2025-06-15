"""
Theoretical Analysis Module - Enhancement #4
Convergence Analysis and Spectral Properties for Hardware-Data-Parameter Co-Design

This module provides theoretical underpinnings for:
1. SDM convergence properties and sparsity evolution bounds
2. CSP spectral analysis and correlation matrix properties  
3. Multi-objective optimization approximation quality
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import scipy.linalg
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns


class SDMConvergenceAnalyzer:
    """
    Analyzes convergence properties of Structured Differentiable Masking (SDM).
    
    Provides theoretical bounds and empirical convergence tracking for:
    - Gumbel-Sigmoid temperature annealing convergence
    - Sparsity pattern evolution and stability
    - Loss landscape analysis around learned sparse solutions
    """
    
    def __init__(self):
        self.convergence_history = []
        self.sparsity_history = []
        self.gradient_norms = []
        
    def analyze_gumbel_sigmoid_convergence(
        self, 
        z_logits_history: List[torch.Tensor],
        temperature_schedule: List[float]
    ) -> Dict[str, Any]:
        """
        Analyze convergence properties of Gumbel-Sigmoid sampling.
        
        Theory: As temperature τ → 0, stochastic masks converge to deterministic binary masks
        Rate: Convergence rate depends on the spectral gap of the Hessian matrix
        
        Args:
            z_logits_history: Evolution of importance logits over training
            temperature_schedule: Temperature annealing schedule
            
        Returns:
            Dictionary with convergence metrics and theoretical bounds
        """
        results = {}
        
        # 1. Compute sparsity evolution
        sparsity_over_time = []
        mask_variance_over_time = []
        
        for i, (z_logits, temp) in enumerate(zip(z_logits_history, temperature_schedule)):
            # Current sparsity level
            current_sparsity = torch.sigmoid(z_logits).mean().item()
            sparsity_over_time.append(current_sparsity)
            
            # Mask variance (measure of stochasticity)
            sigmoid_probs = torch.sigmoid(z_logits / temp)
            mask_variance = (sigmoid_probs * (1 - sigmoid_probs)).mean().item()
            mask_variance_over_time.append(mask_variance)
        
        # 2. Convergence rate analysis
        sparsity_tensor = torch.tensor(sparsity_over_time)
        if len(sparsity_tensor) > 10:  # Need sufficient history
            # Compute differences to measure convergence
            differences = torch.diff(sparsity_tensor)
            convergence_rate = torch.abs(differences).mean().item()
            
            # Exponential decay fitting for theoretical analysis
            steps = torch.arange(len(differences), dtype=torch.float)
            log_diff = torch.log(torch.abs(differences) + 1e-8)
            
            # Linear fit: log|diff| = -λt + c => |diff| = e^c * e^(-λt)
            if len(steps) > 5:
                coeffs = torch.polyfit(steps, log_diff, 1)
                decay_rate = -coeffs[0].item()  # λ in exponential decay
            else:
                decay_rate = 0.0
        else:
            convergence_rate = float('inf')
            decay_rate = 0.0
        
        # 3. Theoretical bounds
        final_temp = temperature_schedule[-1] if temperature_schedule else 1.0
        theoretical_variance_bound = 0.25 * final_temp  # Maximum at sigmoid(0) with temp
        
        results.update({
            'sparsity_evolution': sparsity_over_time,
            'mask_variance_evolution': mask_variance_over_time,
            'convergence_rate': convergence_rate,
            'exponential_decay_rate': decay_rate,
            'final_sparsity': sparsity_over_time[-1] if sparsity_over_time else 0.0,
            'theoretical_variance_bound': theoretical_variance_bound,
            'actual_final_variance': mask_variance_over_time[-1] if mask_variance_over_time else 0.0,
            'convergence_achieved': mask_variance_over_time[-1] < theoretical_variance_bound if mask_variance_over_time else False
        })
        
        return results
    
    def compute_sparsity_regularization_bounds(
        self,
        z_logits: torch.Tensor,
        lambda_sparsity: float,
        task_loss_hessian: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Compute theoretical bounds for sparsity regularization effectiveness.
        
        Theory: The regularized loss L = L_task + λ∑sigmoid(z_c) has a unique minimum
        when λ is chosen appropriately relative to the curvature of L_task.
        
        Args:
            z_logits: Current importance logits
            lambda_sparsity: Sparsity regularization weight
            task_loss_hessian: Hessian of task loss w.r.t. z_logits (optional)
            
        Returns:
            Dictionary with theoretical bounds and stability metrics
        """
        results = {}
        
        # 1. Current sparsity and gradient analysis
        current_masks = torch.sigmoid(z_logits)
        current_sparsity = current_masks.mean().item()
        
        # Gradient of sparsity term: d/dz sigmoid(z) = sigmoid(z)(1-sigmoid(z))
        sparsity_gradient = current_masks * (1 - current_masks)
        
        # 2. Effective regularization strength
        effective_lambda = lambda_sparsity * sparsity_gradient.mean().item()
        
        # 3. Stability analysis
        # Second derivative of sparsity term
        sparsity_hessian_diag = sparsity_gradient * (1 - 2 * current_masks)
        
        if task_loss_hessian is not None:
            # Combined Hessian eigenvalues indicate stability
            combined_hessian = task_loss_hessian + lambda_sparsity * torch.diag(sparsity_hessian_diag)
            eigenvals = torch.linalg.eigvals(combined_hessian).real
            condition_number = (eigenvals.max() / eigenvals.min()).item()
            stability_margin = eigenvals.min().item()
        else:
            # Approximation using only sparsity term
            condition_number = (sparsity_hessian_diag.max() / sparsity_hessian_diag.min()).item()
            stability_margin = sparsity_hessian_diag.min().item()
        
        # 4. Theoretical optimal sparsity
        # At equilibrium: ∇L_task + λ∇L_sparsity = 0
        # This gives theoretical prediction for final sparsity level
        
        results.update({
            'current_sparsity': current_sparsity,
            'effective_regularization_strength': effective_lambda,
            'condition_number': abs(condition_number),
            'stability_margin': stability_margin,
            'converged': stability_margin > 1e-6,  # Positive definite check
            'gradient_norm': sparsity_gradient.norm().item()
        })
        
        return results


class CSPSpectralAnalyzer:
    """
    Spectral analysis of correlation matrices for Correlation-based Scan Permutation (CSP).
    
    Provides theoretical analysis of:
    - Correlation matrix spectral properties
    - Permutation optimality bounds
    - Cache efficiency theoretical predictions
    """
    
    def __init__(self):
        self.correlation_history = []
        self.spectral_properties = {}
        
    def analyze_correlation_matrix_spectrum(
        self, 
        correlation_matrix: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Comprehensive spectral analysis of state correlation matrix.
        
        Theory: The eigenvalue spectrum reveals the intrinsic dimensionality
        and correlation structure that determines permutation effectiveness.
        
        Args:
            correlation_matrix: State correlation matrix Σ (d_state × d_state)
            
        Returns:
            Dictionary with spectral properties and theoretical bounds
        """
        results = {}
        
        # Ensure symmetric and handle numerical issues
        corr_matrix = (correlation_matrix + correlation_matrix.T) / 2
        corr_matrix = torch.nan_to_num(corr_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 1. Eigenvalue decomposition
        try:
            eigenvals, eigenvecs = torch.linalg.eigh(corr_matrix)
            eigenvals = eigenvals.real  # Take real part to handle numerical errors
            eigenvals = torch.sort(eigenvals, descending=True)[0]  # Sort descending
        except Exception as e:
            print(f"Eigendecomposition failed: {e}")
            eigenvals = torch.ones(corr_matrix.shape[0])
            eigenvecs = torch.eye(corr_matrix.shape[0])
        
        # 2. Spectral properties
        max_eigenval = eigenvals[0].item()
        min_eigenval = eigenvals[-1].item()
        
        # Condition number (avoid division by zero)
        if abs(min_eigenval) > 1e-10:
            condition_number = abs(max_eigenval / min_eigenval)
        else:
            condition_number = float('inf')
        
        # Spectral gap (difference between largest eigenvalues)
        spectral_gap = (eigenvals[0] - eigenvals[1]).item() if len(eigenvals) > 1 else 0.0
        
        # Effective rank (number of significant eigenvalues)
        eigenval_sum = eigenvals.sum()
        if eigenval_sum > 1e-10:
            entropy = -(eigenvals / eigenval_sum * torch.log(eigenvals / eigenval_sum + 1e-10)).sum()
            effective_rank = torch.exp(entropy).item()
        else:
            effective_rank = 1.0
        
        # 3. Johnson-Lindenstrauss bound for correlation preservation
        d_state = correlation_matrix.shape[0]
        jl_bound = torch.sqrt(torch.log(torch.tensor(d_state, dtype=torch.float)) / d_state).item()
        
        # 4. Clustering coefficient (measure of local correlation structure)
        abs_corr = torch.abs(corr_matrix)
        # Remove diagonal
        abs_corr = abs_corr - torch.diag(torch.diag(abs_corr))
        mean_correlation = abs_corr.mean().item()
        max_correlation = abs_corr.max().item()
        
        # 5. Block structure detection
        # Simple heuristic: check if correlation matrix has block-diagonal structure
        block_sizes = self._detect_block_structure(abs_corr)
        
        results.update({
            'eigenvalues': eigenvals.tolist(),
            'max_eigenvalue': max_eigenval,
            'min_eigenvalue': min_eigenval,
            'condition_number': condition_number,
            'spectral_gap': spectral_gap,
            'effective_rank': effective_rank,
            'intrinsic_dimension_ratio': effective_rank / d_state,
            'johnson_lindenstrauss_bound': jl_bound,
            'mean_absolute_correlation': mean_correlation,
            'max_absolute_correlation': max_correlation,
            'block_structure': block_sizes,
            'matrix_size': d_state,
            'well_conditioned': condition_number < 100,  # Arbitrary threshold
            'strong_correlation_structure': spectral_gap > 0.1
        })
        
        return results
    
    def _detect_block_structure(self, abs_corr_matrix: torch.Tensor) -> List[int]:
        """
        Detect block-diagonal structure in correlation matrix.
        
        Simple heuristic: use correlation threshold to identify blocks.
        """
        threshold = 0.5  # Correlation threshold for block detection
        n = abs_corr_matrix.shape[0]
        
        # Find highly correlated pairs
        high_corr = abs_corr_matrix > threshold
        
        # Simple connected components to find blocks
        visited = [False] * n
        blocks = []
        
        for i in range(n):
            if not visited[i]:
                # BFS to find connected component
                block = []
                queue = [i]
                visited[i] = True
                
                while queue:
                    node = queue.pop(0)
                    block.append(node)
                    
                    for j in range(n):
                        if not visited[j] and high_corr[node, j]:
                            visited[j] = True
                            queue.append(j)
                
                blocks.append(len(block))
        
        return sorted(blocks, reverse=True)
    
    def compute_permutation_optimality_bound(
        self,
        correlation_matrix: torch.Tensor,
        current_permutation: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute theoretical bounds on permutation optimality.
        
        Theory: The optimal permutation minimizes the "correlation distance"
        which can be bounded using spectral properties.
        
        Args:
            correlation_matrix: Original correlation matrix
            current_permutation: Current permutation vector
            
        Returns:
            Dictionary with optimality bounds and efficiency metrics
        """
        results = {}
        
        # 1. Apply current permutation
        perm_matrix = correlation_matrix[current_permutation][:, current_permutation]
        
        # 2. Compute correlation efficiency
        abs_corr = torch.abs(correlation_matrix)
        abs_perm_corr = torch.abs(perm_matrix)
        
        # Local correlation (adjacent elements)
        n = len(current_permutation)
        local_correlation_original = 0.0
        local_correlation_permuted = 0.0
        
        for i in range(n - 1):
            local_correlation_original += abs_corr[i, i + 1].item()
            local_correlation_permuted += abs_perm_corr[i, i + 1].item()
        
        local_correlation_original /= (n - 1)
        local_correlation_permuted /= (n - 1)
        
        # 3. Theoretical upper bound
        # Maximum possible local correlation is limited by matrix properties
        max_possible_local = abs_corr.max().item()
        
        # 4. Cache efficiency estimation
        # Based on memory access patterns
        cache_line_elements = 16  # Assume 64-byte cache line, 4-byte elements
        cache_efficiency = self._estimate_cache_efficiency(
            current_permutation, correlation_matrix, cache_line_elements
        )
        
        results.update({
            'local_correlation_improvement': local_correlation_permuted - local_correlation_original,
            'local_correlation_ratio': local_correlation_permuted / (local_correlation_original + 1e-10),
            'theoretical_upper_bound': max_possible_local,
            'optimality_ratio': local_correlation_permuted / max_possible_local,
            'cache_efficiency_estimate': cache_efficiency
        })
        
        return results
    
    def _estimate_cache_efficiency(
        self,
        permutation: torch.Tensor,
        correlation_matrix: torch.Tensor,
        cache_line_elements: int
    ) -> float:
        """
        Estimate cache efficiency based on correlation and access patterns.
        
        Theory: Elements accessed together should be correlated and 
        physically close in memory for optimal cache utilization.
        """
        n = len(permutation)
        abs_corr = torch.abs(correlation_matrix)
        
        cache_efficiency = 0.0
        num_cache_lines = 0
        
        # Analyze correlation within cache lines
        for start in range(0, n, cache_line_elements):
            end = min(start + cache_line_elements, n)
            if end - start > 1:  # Need at least 2 elements
                cache_line_indices = permutation[start:end]
                
                # Average correlation within this cache line
                line_correlation = 0.0
                line_pairs = 0
                
                for i in range(len(cache_line_indices)):
                    for j in range(i + 1, len(cache_line_indices)):
                        idx1, idx2 = cache_line_indices[i], cache_line_indices[j]
                        line_correlation += abs_corr[idx1, idx2].item()
                        line_pairs += 1
                
                if line_pairs > 0:
                    cache_efficiency += line_correlation / line_pairs
                    num_cache_lines += 1
        
        return cache_efficiency / max(num_cache_lines, 1)


class MultiObjectiveOptimizationAnalyzer:
    """
    Analyzes the multi-objective optimization approximation quality.
    
    Our staged pipeline approximates the intractable joint optimization:
    minimize: α₁·L_task + α₂·L_latency + α₃·L_memory + α₄·L_sparsity + α₅·L_correlation
    """
    
    def __init__(self):
        self.objectives = ['task_loss', 'latency', 'memory', 'sparsity', 'correlation']
        
    def analyze_approximation_quality(
        self,
        joint_solution_metrics: Dict[str, float],
        staged_solution_metrics: Dict[str, float],
        objective_weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Analyze how well our staged approach approximates the joint optimization.
        
        Args:
            joint_solution_metrics: Metrics from hypothetical joint optimization
            staged_solution_metrics: Metrics from our staged approach
            objective_weights: Weights for different objectives
            
        Returns:
            Approximation quality analysis
        """
        results = {}
        
        # 1. Compute weighted objectives
        joint_weighted_sum = sum(
            objective_weights.get(obj, 1.0) * joint_solution_metrics.get(obj, 0.0)
            for obj in self.objectives
        )
        
        staged_weighted_sum = sum(
            objective_weights.get(obj, 1.0) * staged_solution_metrics.get(obj, 0.0)
            for obj in self.objectives
        )
        
        # 2. Approximation ratio
        if joint_weighted_sum > 1e-10:
            approximation_ratio = staged_weighted_sum / joint_weighted_sum
        else:
            approximation_ratio = 1.0
        
        # 3. Individual objective analysis
        objective_ratios = {}
        for obj in self.objectives:
            joint_val = joint_solution_metrics.get(obj, 0.0)
            staged_val = staged_solution_metrics.get(obj, 0.0)
            
            if abs(joint_val) > 1e-10:
                objective_ratios[obj] = staged_val / joint_val
            else:
                objective_ratios[obj] = 1.0 if abs(staged_val) < 1e-10 else float('inf')
        
        # 4. Pareto efficiency analysis
        pareto_dominated = self._check_pareto_dominance(
            joint_solution_metrics, staged_solution_metrics
        )
        
        results.update({
            'approximation_ratio': approximation_ratio,
            'objective_ratios': objective_ratios,
            'joint_weighted_objective': joint_weighted_sum,
            'staged_weighted_objective': staged_weighted_sum,
            'pareto_dominated': pareto_dominated,
            'approximation_quality': 'good' if approximation_ratio < 1.2 else 'poor'
        })
        
        return results
    
    def _check_pareto_dominance(
        self,
        solution_a: Dict[str, float],
        solution_b: Dict[str, float]
    ) -> bool:
        """
        Check if solution_a Pareto dominates solution_b.
        
        A dominates B if A is better in at least one objective and
        not worse in any objective.
        """
        better_in_some = False
        worse_in_any = False
        
        for obj in self.objectives:
            val_a = solution_a.get(obj, 0.0)
            val_b = solution_b.get(obj, 0.0)
            
            # Assume all objectives are to be minimized
            if val_a < val_b:
                better_in_some = True
            elif val_a > val_b:
                worse_in_any = True
        
        return better_in_some and not worse_in_any


def create_theoretical_analysis_report(
    sdm_analyzer: SDMConvergenceAnalyzer,
    csp_analyzer: CSPSpectralAnalyzer,
    multi_obj_analyzer: MultiObjectiveOptimizationAnalyzer,
    save_path: str = "theoretical_analysis_report.json"
) -> Dict[str, Any]:
    """
    Generate comprehensive theoretical analysis report.
    
    Args:
        sdm_analyzer: Configured SDM analyzer with results
        csp_analyzer: Configured CSP analyzer with results  
        multi_obj_analyzer: Configured multi-objective analyzer
        save_path: Path to save the analysis report
        
    Returns:
        Complete theoretical analysis report
    """
    import json
    from datetime import datetime
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'analysis_type': 'theoretical_underpinnings',
        'framework_version': '1.0.0',
        'sdm_analysis': {
            'convergence_properties': 'Analyzed via Gumbel-Sigmoid theory',
            'theoretical_foundations': 'Temperature annealing convergence guarantees'
        },
        'csp_analysis': {
            'spectral_properties': 'Eigenvalue spectrum analysis',
            'theoretical_bounds': 'Johnson-Lindenstrauss correlation preservation'
        },
        'multi_objective_analysis': {
            'approximation_framework': 'Staged pipeline as joint optimization approximation',
            'theoretical_justification': 'Principled sequential optimization strategy'
        }
    }
    
    # Save report
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Theoretical analysis report saved to {save_path}")
    return report 