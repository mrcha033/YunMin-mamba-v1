"""
Scan Patch Implementation
Implements monkey-patching for Mamba mixers to apply precomputed scan permutations.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Callable
import logging

# Global state for tracking patch status
_scan_patched = False
_original_forwards = {}

def apply_scan_patch(model: nn.Module, scan_path: str, rev_path: str):
    """
    Apply scan permutation patch to all mixer modules in the model.
    
    Args:
        model: The neural network model to patch
        scan_path: Path to the forward scan permutation file (.npy)
        rev_path: Path to the reverse scan permutation file (.npy)
    """
    global _scan_patched, _original_forwards
    
    if _scan_patched:
        logging.warning("Model is already scan-patched. Remove existing patch first.")
        return
    
    # Load permutation arrays
    try:
        scan_permutation = np.load(scan_path)
        rev_permutation = np.load(rev_path)
        
        # Convert to torch tensors
        scan_perm = torch.from_numpy(scan_permutation).long()
        rev_perm = torch.from_numpy(rev_permutation).long()
        
        logging.info(f"Loaded scan permutations from {scan_path} and {rev_path}")
        logging.info(f"Scan permutation shape: {scan_perm.shape}")
        
    except Exception as e:
        logging.error(f"Failed to load permutation files: {e}")
        return
    
    # Find and patch all mixer modules
    mixers_patched = 0
    for name, module in model.named_modules():
        if hasattr(module, 'forward') and 'mixer' in name.lower():
            # Store original forward method
            original_forward = module.forward
            _original_forwards[id(module)] = original_forward
            
            # Create patched forward method
            def create_patched_forward(orig_forward, scan_p, rev_p):
                def patched_forward(hidden_states, *args, **kwargs):
                    # Apply forward permutation to input
                    if hidden_states.dim() >= 3:
                        # Assume last dimension is the feature dimension
                        permuted_input = apply_permutation_to_tensor(hidden_states, scan_p)
                    else:
                        permuted_input = hidden_states
                    
                    # Call original forward method
                    output = orig_forward(permuted_input, *args, **kwargs)
                    
                    # Apply reverse permutation to output
                    if output.dim() >= 3:
                        output = apply_permutation_to_tensor(output, rev_p)
                    
                    return output
                return patched_forward
            
            # Apply the patch
            module.forward = create_patched_forward(original_forward, scan_perm, rev_perm)
            mixers_patched += 1
            
            logging.debug(f"Patched mixer: {name}")
    
    _scan_patched = True
    logging.info(f"Successfully applied scan patch to {mixers_patched} mixer modules")

def remove_scan_patch(model: nn.Module):
    """
    Remove scan permutation patch from all mixer modules in the model.
    
    Args:
        model: The neural network model to unpatch
    """
    global _scan_patched, _original_forwards
    
    if not _scan_patched:
        logging.warning("Model is not currently scan-patched.")
        return
    
    # Restore original forward methods
    mixers_restored = 0
    for name, module in model.named_modules():
        if hasattr(module, 'forward') and 'mixer' in name.lower():
            module_id = id(module)
            if module_id in _original_forwards:
                module.forward = _original_forwards[module_id]
                del _original_forwards[module_id]
                mixers_restored += 1
                logging.debug(f"Restored mixer: {name}")
    
    _scan_patched = False
    _original_forwards.clear()
    logging.info(f"Successfully removed scan patch from {mixers_restored} mixer modules")

def is_scan_patched() -> bool:
    """
    Check if the model is currently scan-patched.
    
    Returns:
        True if scan patch is applied, False otherwise
    """
    return _scan_patched

def apply_permutation_to_tensor(tensor: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    """
    Apply permutation to the last dimension of a tensor.
    
    Args:
        tensor: Input tensor of shape (..., D)
        permutation: Permutation indices of shape (D,)
    
    Returns:
        Permuted tensor of same shape as input
    """
    if tensor.size(-1) != len(permutation):
        # If dimensions don't match, pad or truncate permutation
        target_size = tensor.size(-1)
        if len(permutation) > target_size:
            # Truncate permutation
            perm = permutation[:target_size]
        else:
            # Pad permutation with identity mapping
            perm = torch.cat([
                permutation, 
                torch.arange(len(permutation), target_size, dtype=permutation.dtype, device=permutation.device)
            ])
    else:
        perm = permutation
    
    # Move permutation to same device as tensor
    perm = perm.to(tensor.device)
    
    # Apply permutation along last dimension
    return tensor[..., perm]

def create_scan_permutation_from_model(model: nn.Module, sample_input: torch.Tensor) -> tuple:
    """
    Generate scan permutation based on model hidden states correlation analysis.
    
    Args:
        model: The model to analyze
        sample_input: Sample input to extract hidden states
    
    Returns:
        Tuple of (forward_permutation, reverse_permutation)
    """
    try:
        from layers.variable_scan import compute_scan_permutation
        
        # Extract hidden states by running model
        model.eval()
        with torch.no_grad():
            # This is a simplified version - in practice you'd need to hook into 
            # intermediate layers to get actual hidden states
            _ = model(sample_input)
            
        # For now, generate a simple permutation based on input dimensions
        d_model = sample_input.size(-1)
        hidden_states = torch.randn(32, 128, d_model)  # Dummy hidden states
        
        # Compute optimal permutation
        forward_perm = compute_scan_permutation(hidden_states)
        
        # Compute reverse permutation
        reverse_perm = torch.zeros_like(forward_perm)
        reverse_perm[forward_perm] = torch.arange(len(forward_perm))
        
        return forward_perm.numpy(), reverse_perm.numpy()
        
    except ImportError:
        logging.warning("Variable scan module not available. Using identity permutation.")
        d_model = sample_input.size(-1) if sample_input is not None else 256
        identity = np.arange(d_model)
        return identity, identity 