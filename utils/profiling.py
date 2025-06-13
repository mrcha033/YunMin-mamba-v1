"""
Profiling utilities for FLOPs counting and performance analysis.
"""

import time
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any
from contextlib import contextmanager
try:
    from fvcore.nn import flop_count
    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False


def count_flops(model: nn.Module, input_shape: Tuple[int, ...], device: str = "cpu") -> Dict[str, Any]:
    """
    Count FLOPs for a model with given input shape.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (batch_size, seq_length, ...)
        device: Device to run the model on
    
    Returns:
        Dictionary containing FLOPs information
    """
    if not FVCORE_AVAILABLE:
        return {
            "error": "fvcore not available for FLOPs counting",
            "total_flops": 0,
            "input_shape": input_shape,
            "device": device
        }
    
    model.eval()
    model = model.to(device)
    
    # Create dummy input
    if len(input_shape) == 2:  # (batch_size, seq_length)
        dummy_input = torch.randint(0, 1000, input_shape, device=device)
    else:
        dummy_input = torch.randn(*input_shape, device=device)
    
    # Count FLOPs
    try:
        with torch.no_grad():
            flop_dict, _ = flop_count(model, (dummy_input,))
        
        total_flops = sum(flop_dict.values())
        
        return {
            "total_flops": total_flops,
            "flops_per_layer": flop_dict,
            "input_shape": input_shape,
            "device": device
        }
    except Exception as e:
        return {
            "error": f"FLOPs counting failed: {str(e)}",
            "total_flops": 0,
            "input_shape": input_shape,
            "device": device
        }


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary containing parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count parameters by layer type
    param_by_type = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            module_params = sum(p.numel() for p in module.parameters())
            if module_params > 0:
                module_type = type(module).__name__
                if module_type not in param_by_type:
                    param_by_type[module_type] = 0
                param_by_type[module_type] += module_params
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
        "parameters_by_type": param_by_type
    }


@contextmanager
def measure_time():
    """
    Context manager for measuring execution time.
    
    Usage:
        with measure_time() as timer:
            # Your code here
            pass
        print(f"Execution time: {timer.elapsed:.4f}s")
    """
    class Timer:
        def __init__(self):
            self.start_time = None
            self.elapsed = None
    
    timer = Timer()
    timer.start_time = time.time()
    
    try:
        yield timer
    finally:
        timer.elapsed = time.time() - timer.start_time


def measure_latency(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: str = "cuda",
    num_warmup: int = 10,
    num_runs: int = 100
) -> Dict[str, float]:
    """
    Measure model inference latency.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        device: Device to run inference on
        num_warmup: Number of warmup runs
        num_runs: Number of measurement runs
    
    Returns:
        Dictionary containing latency statistics
    """
    model.eval()
    model = model.to(device)
    
    # Create dummy input
    if len(input_shape) == 2:  # (batch_size, seq_length)
        dummy_input = torch.randint(0, 1000, input_shape, device=device)
    else:
        dummy_input = torch.randn(*input_shape, device=device)
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)
    
    # Synchronize for accurate timing
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Measurement runs
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(dummy_input)
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms
    
    return {
        "mean_latency_ms": sum(latencies) / len(latencies),
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies),
        "std_latency_ms": (sum((x - sum(latencies) / len(latencies)) ** 2 for x in latencies) / len(latencies)) ** 0.5,
        "num_runs": num_runs,
        "input_shape": input_shape,
        "device": device
    }


def memory_usage(model: nn.Module, device: str = "cuda") -> Dict[str, float]:
    """
    Measure GPU memory usage of the model.
    
    Args:
        model: PyTorch model
        device: Device ("cuda" only)
    
    Returns:
        Dictionary containing memory usage information
    """
    if device != "cuda":
        return {"error": "Memory profiling only supported on CUDA"}
    
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Measure memory before model allocation
    memory_before = torch.cuda.memory_allocated()
    
    model = model.to(device)
    
    # Measure memory after model allocation
    memory_after = torch.cuda.memory_allocated()
    
    return {
        "model_memory_mb": (memory_after - memory_before) / (1024 ** 2),
        "peak_memory_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
        "total_memory_mb": torch.cuda.get_device_properties(device).total_memory / (1024 ** 2),
        "device": device
    } 