"""
Adaptive Hybrid-PEFT Mamba Layers Package
Exports all three pillar implementations for easy import.
"""

# Pillar 1: Variable-Aware Scan
from .variable_scan import (
    compute_scan_permutation,
    apply_permutation,
    invert_permutation,
    compute_correlation_matrix,
    compute_cost_matrix,
    nearest_neighbor_tsp,
    VariableScanOptimizer
)

# Pillar 2: Learned Masking
from .learned_mask import (
    LearnedMask,
    AdaptiveSparsityMask
)

# Pillar 2 additional layers
from .masked_linear import (
    MaskedLinear,
    MaskedConv1d,
    convert_to_masked
)

# Pillar 3: IA3 adapters
from .ia3_layers import insert_ia3_modules

# Pillar 3: Hybrid PEFT utilities

__all__ = [
    # Pillar 1
    'compute_scan_permutation',
    'apply_permutation', 
    'invert_permutation',
    'compute_correlation_matrix',
    'compute_cost_matrix',
    'nearest_neighbor_tsp',
    'VariableScanOptimizer',
    
    # Pillar 2
    'LearnedMask',
    'AdaptiveSparsityMask',
    'MaskedLinear',
    'MaskedConv1d',
    'convert_to_masked',
    
    # Pillar 3
    'insert_ia3_modules',
]
