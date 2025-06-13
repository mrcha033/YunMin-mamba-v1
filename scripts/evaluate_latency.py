"""
High-Precision Hardware Profiling for A100 GPUs

This script provides comprehensive hardware performance profiling with:
- High-precision CUDA event timing
- GPU memory profiling and peak usage tracking
- Throughput measurement with various batch sizes
- Detailed kernel-level profiling
- A100-specific optimizations and measurements

Features:
- torch.profiler integration for detailed kernel analysis
- CUDA event-based timing for microsecond precision
- Memory efficiency analysis
- Batch size scaling analysis
- Hardware utilization metrics
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import yaml

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.baseline_ssm import BaselineSSM
from models.sdm_ssm import SDM_SSM
from models.sgh_peft import SGHPEFTModel
from utils.profiling import count_parameters, estimate_flops

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LatencyProfile:
    """Container for latency profiling results."""
    model_name: str
    batch_size: int
    sequence_length: int
    
    # Timing metrics (milliseconds)
    mean_latency: float
    std_latency: float
    min_latency: float
    max_latency: float
    median_latency: float
    p95_latency: float
    p99_latency: float
    
    # Throughput metrics
    tokens_per_second: float
    samples_per_second: float
    
    # Memory metrics (MB)
    peak_memory_mb: float
    allocated_memory_mb: float
    reserved_memory_mb: float
    
    # Hardware utilization
    gpu_utilization: Optional[float] = None
    memory_utilization: Optional[float] = None
    
    # Additional metrics
    warmup_iterations: int = 10
    measurement_iterations: int = 100
    cuda_events_used: bool = True


@dataclass
class ThroughputProfile:
    """Container for throughput profiling results."""
    model_name: str
    batch_sizes: List[int]
    sequence_length: int
    
    # Throughput data
    throughput_data: Dict[int, float]  # batch_size -> tokens/sec
    latency_data: Dict[int, float]     # batch_size -> latency (ms)
    memory_data: Dict[int, float]      # batch_size -> peak memory (MB)
    
    # Optimal operating points
    optimal_batch_size: int
    max_throughput: float
    memory_efficient_batch_size: int


class A100Profiler:
    """
    High-precision profiler optimized for NVIDIA A100 GPUs.
    
    Features:
    - CUDA event-based timing for maximum precision
    - Memory profiling with detailed breakdown
    - Kernel-level analysis with torch.profiler
    - Batch size scaling analysis
    - Hardware utilization monitoring
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        warmup_iterations: int = 10,
        measurement_iterations: int = 100,
        enable_profiling: bool = True
    ):
        """
        Initialize A100 profiler.
        
        Args:
            model: Model to profile
            device: Device to run profiling on
            warmup_iterations: Number of warmup iterations
            measurement_iterations: Number of measurement iterations
            enable_profiling: Whether to enable detailed profiling
        """
        self.model = model.to(device)
        self.device = device
        self.warmup_iterations = warmup_iterations
        self.measurement_iterations = measurement_iterations
        self.enable_profiling = enable_profiling
        
        # Verify CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available for profiling")
        
        # Get GPU information
        self.gpu_name = torch.cuda.get_device_name(device)
        self.gpu_memory_total = torch.cuda.get_device_properties(device).total_memory
        
        logger.info(f"Profiling on: {self.gpu_name}")
        logger.info(f"Total GPU memory: {self.gpu_memory_total / 1024**3:.2f} GB")
        
        # Set model to evaluation mode
        self.model.eval()
    
    def profile_latency(
        self,
        batch_size: int = 1,
        sequence_length: int = 1024,
        model_name: str = "model"
    ) -> LatencyProfile:
        """
        Profile model latency with high precision.
        
        Args:
            batch_size: Batch size for profiling
            sequence_length: Sequence length
            model_name: Name for identification
            
        Returns:
            LatencyProfile with detailed timing information
        """
        logger.info(f"Profiling latency: {model_name} (batch={batch_size}, seq_len={sequence_length})")
        
        # Create dummy input
        dummy_input = torch.randint(
            0, 50257, (batch_size, sequence_length), 
            device=self.device, dtype=torch.long
        )
        
        # Reset memory stats
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.empty_cache()
        
        # Warmup
        logger.info("Warming up...")
        with torch.no_grad():
            for _ in range(self.warmup_iterations):
                _ = self.model(dummy_input)
        
        # Synchronize before measurement
        torch.cuda.synchronize()
        
        # Measure latency with CUDA events
        logger.info(f"Measuring latency over {self.measurement_iterations} iterations...")
        
        latencies = []
        start_events = []
        end_events = []
        
        # Create CUDA events for precise timing
        for _ in range(self.measurement_iterations):
            start_events.append(torch.cuda.Event(enable_timing=True))
            end_events.append(torch.cuda.Event(enable_timing=True))
        
        # Measurement loop
        with torch.no_grad():
            for i in range(self.measurement_iterations):
                # Record start event
                start_events[i].record()
                
                # Forward pass
                _ = self.model(dummy_input)
                
                # Record end event
                end_events[i].record()
        
        # Synchronize and calculate latencies
        torch.cuda.synchronize()
        
        for i in range(self.measurement_iterations):
            latency_ms = start_events[i].elapsed_time(end_events[i])
            latencies.append(latency_ms)
        
        # Calculate statistics
        latencies = np.array(latencies)
        mean_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        median_latency = np.median(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        # Calculate throughput
        tokens_per_second = (batch_size * sequence_length * 1000) / mean_latency
        samples_per_second = (batch_size * 1000) / mean_latency
        
        # Get memory statistics
        peak_memory_bytes = torch.cuda.max_memory_allocated(self.device)
        allocated_memory_bytes = torch.cuda.memory_allocated(self.device)
        reserved_memory_bytes = torch.cuda.memory_reserved(self.device)
        
        peak_memory_mb = peak_memory_bytes / (1024 ** 2)
        allocated_memory_mb = allocated_memory_bytes / (1024 ** 2)
        reserved_memory_mb = reserved_memory_bytes / (1024 ** 2)
        
        # Log results
        logger.info(f"Latency results for {model_name}:")
        logger.info(f"  Mean latency: {mean_latency:.3f} ± {std_latency:.3f} ms")
        logger.info(f"  Min/Max latency: {min_latency:.3f} / {max_latency:.3f} ms")
        logger.info(f"  P95/P99 latency: {p95_latency:.3f} / {p99_latency:.3f} ms")
        logger.info(f"  Throughput: {tokens_per_second:.0f} tokens/sec, {samples_per_second:.1f} samples/sec")
        logger.info(f"  Peak memory: {peak_memory_mb:.1f} MB")
        
        return LatencyProfile(
            model_name=model_name,
            batch_size=batch_size,
            sequence_length=sequence_length,
            mean_latency=mean_latency,
            std_latency=std_latency,
            min_latency=min_latency,
            max_latency=max_latency,
            median_latency=median_latency,
            p95_latency=p95_latency,
            p99_latency=p99_latency,
            tokens_per_second=tokens_per_second,
            samples_per_second=samples_per_second,
            peak_memory_mb=peak_memory_mb,
            allocated_memory_mb=allocated_memory_mb,
            reserved_memory_mb=reserved_memory_mb,
            warmup_iterations=self.warmup_iterations,
            measurement_iterations=self.measurement_iterations,
            cuda_events_used=True
        )
    
    def profile_throughput_scaling(
        self,
        batch_sizes: List[int],
        sequence_length: int = 1024,
        model_name: str = "model"
    ) -> ThroughputProfile:
        """
        Profile throughput scaling across different batch sizes.
        
        Args:
            batch_sizes: List of batch sizes to test
            sequence_length: Sequence length
            model_name: Name for identification
            
        Returns:
            ThroughputProfile with scaling analysis
        """
        logger.info(f"Profiling throughput scaling: {model_name}")
        logger.info(f"Batch sizes: {batch_sizes}")
        
        throughput_data = {}
        latency_data = {}
        memory_data = {}
        
        for batch_size in batch_sizes:
            try:
                logger.info(f"Testing batch size: {batch_size}")
                
                # Profile this batch size
                profile = self.profile_latency(
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                    model_name=f"{model_name}_bs{batch_size}"
                )
                
                throughput_data[batch_size] = profile.tokens_per_second
                latency_data[batch_size] = profile.mean_latency
                memory_data[batch_size] = profile.peak_memory_mb
                
            except torch.cuda.OutOfMemoryError:
                logger.warning(f"OOM at batch size {batch_size}, stopping scaling test")
                break
            except Exception as e:
                logger.error(f"Error at batch size {batch_size}: {e}")
                continue
        
        # Find optimal operating points
        if throughput_data:
            optimal_batch_size = max(throughput_data.keys(), key=lambda k: throughput_data[k])
            max_throughput = throughput_data[optimal_batch_size]
            
            # Memory-efficient batch size (highest throughput under memory threshold)
            memory_threshold = self.gpu_memory_total * 0.8 / (1024 ** 2)  # 80% of total memory
            memory_efficient_candidates = [
                bs for bs, mem in memory_data.items() if mem < memory_threshold
            ]
            
            if memory_efficient_candidates:
                memory_efficient_batch_size = max(
                    memory_efficient_candidates, 
                    key=lambda k: throughput_data[k]
                )
            else:
                memory_efficient_batch_size = min(throughput_data.keys())
        else:
            optimal_batch_size = 1
            max_throughput = 0.0
            memory_efficient_batch_size = 1
        
        logger.info(f"Throughput scaling results:")
        logger.info(f"  Optimal batch size: {optimal_batch_size} ({max_throughput:.0f} tokens/sec)")
        logger.info(f"  Memory-efficient batch size: {memory_efficient_batch_size}")
        
        return ThroughputProfile(
            model_name=model_name,
            batch_sizes=list(throughput_data.keys()),
            sequence_length=sequence_length,
            throughput_data=throughput_data,
            latency_data=latency_data,
            memory_data=memory_data,
            optimal_batch_size=optimal_batch_size,
            max_throughput=max_throughput,
            memory_efficient_batch_size=memory_efficient_batch_size
        )
    
    def profile_with_torch_profiler(
        self,
        batch_size: int = 1,
        sequence_length: int = 1024,
        model_name: str = "model",
        output_dir: str = "profiling_results"
    ) -> Dict[str, Any]:
        """
        Profile model with torch.profiler for detailed kernel analysis.
        
        Args:
            batch_size: Batch size for profiling
            sequence_length: Sequence length
            model_name: Name for identification
            output_dir: Directory to save profiling results
            
        Returns:
            Dictionary with profiling summary
        """
        if not self.enable_profiling:
            logger.info("Detailed profiling disabled, skipping torch.profiler")
            return {}
        
        logger.info(f"Running detailed profiling with torch.profiler: {model_name}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create dummy input
        dummy_input = torch.randint(
            0, 50257, (batch_size, sequence_length),
            device=self.device, dtype=torch.long
        )
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = self.model(dummy_input)
        
        # Profile with torch.profiler
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True
        ) as prof:
            with torch.no_grad():
                for _ in range(10):  # Profile 10 iterations
                    _ = self.model(dummy_input)
        
        # Save profiling results
        trace_file = os.path.join(output_dir, f"{model_name}_trace.json")
        prof.export_chrome_trace(trace_file)
        
        # Get profiling summary
        key_averages = prof.key_averages()
        
        # Extract key metrics
        total_cuda_time = sum([item.cuda_time for item in key_averages])
        total_cpu_time = sum([item.cpu_time for item in key_averages])
        
        # Get top operations by CUDA time
        cuda_ops = sorted(key_averages, key=lambda x: x.cuda_time, reverse=True)[:10]
        
        # Memory profiling
        memory_events = prof.profiler.kineto_results.events()
        peak_memory = max([event.cuda_memory_usage for event in memory_events if hasattr(event, 'cuda_memory_usage')], default=0)
        
        profiling_summary = {
            "model_name": model_name,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "total_cuda_time_us": total_cuda_time,
            "total_cpu_time_us": total_cpu_time,
            "peak_memory_bytes": peak_memory,
            "trace_file": trace_file,
            "top_cuda_ops": [
                {
                    "name": op.key,
                    "cuda_time_us": op.cuda_time,
                    "cpu_time_us": op.cpu_time,
                    "count": op.count,
                    "input_shapes": str(op.input_shapes) if op.input_shapes else None
                }
                for op in cuda_ops
            ]
        }
        
        logger.info(f"Detailed profiling completed:")
        logger.info(f"  Total CUDA time: {total_cuda_time / 1000:.2f} ms")
        logger.info(f"  Total CPU time: {total_cpu_time / 1000:.2f} ms")
        logger.info(f"  Trace saved to: {trace_file}")
        
        return profiling_summary


def load_model_from_checkpoint(
    checkpoint_path: str,
    config: Dict[str, Any],
    device: str = "cuda"
) -> nn.Module:
    """Load model from checkpoint."""
    logger.info(f"Loading model from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Determine model type
    if 'model_type' in checkpoint:
        model_type = checkpoint['model_type']
    elif 'sdm_applied' in checkpoint:
        model_type = 'sdm'
    elif 'sgh_peft_applied' in checkpoint:
        model_type = 'sgh_peft'
    else:
        model_type = 'baseline'
    
    # Create model
    if model_type == 'sdm':
        model = SDM_SSM(**config)
    elif model_type == 'sgh_peft':
        base_model = SDM_SSM(**config)
        model = SGHPEFTModel(base_model)
    else:
        model = BaselineSSM(**config)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    
    return model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="High-precision hardware profiling for A100 GPUs"
    )
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to model configuration file")
    parser.add_argument("--model_name", type=str, default=None,
                       help="Model name for identification")
    
    # Profiling settings
    parser.add_argument("--batch_sizes", type=int, nargs="+", 
                       default=[1, 2, 4, 8, 16, 32],
                       help="Batch sizes to test for throughput scaling")
    parser.add_argument("--sequence_length", type=int, default=1024,
                       help="Sequence length for profiling")
    parser.add_argument("--warmup_iterations", type=int, default=10,
                       help="Number of warmup iterations")
    parser.add_argument("--measurement_iterations", type=int, default=100,
                       help="Number of measurement iterations")
    
    # Output settings
    parser.add_argument("--output_dir", type=str, default="profiling_results",
                       help="Output directory for results")
    parser.add_argument("--enable_detailed_profiling", action="store_true",
                       help="Enable detailed torch.profiler analysis")
    
    # System settings
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run profiling on")
    
    return parser.parse_args()


def main():
    """Main profiling function."""
    args = parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model_path):
        logger.error(f"Model path does not exist: {args.model_path}")
        return 1
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        model_config = config.get('model', {})
    else:
        model_config = {
            'd_model': 768,
            'n_layer': 12,
            'vocab_size': 50257,
            'd_state': 16,
            'd_conv': 4
        }
    
    # Generate model name if not provided
    if args.model_name is None:
        args.model_name = Path(args.model_path).stem
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("Starting high-precision hardware profiling...")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Batch sizes: {args.batch_sizes}")
    logger.info(f"Sequence length: {args.sequence_length}")
    
    try:
        # Load model
        model = load_model_from_checkpoint(args.model_path, model_config, args.device)
        
        # Create profiler
        profiler = A100Profiler(
            model=model,
            device=args.device,
            warmup_iterations=args.warmup_iterations,
            measurement_iterations=args.measurement_iterations,
            enable_profiling=args.enable_detailed_profiling
        )
        
        # Profile single batch latency
        logger.info("\n" + "="*60)
        logger.info("SINGLE BATCH LATENCY PROFILING")
        logger.info("="*60)
        
        single_batch_profile = profiler.profile_latency(
            batch_size=1,
            sequence_length=args.sequence_length,
            model_name=args.model_name
        )
        
        # Profile throughput scaling
        logger.info("\n" + "="*60)
        logger.info("THROUGHPUT SCALING PROFILING")
        logger.info("="*60)
        
        throughput_profile = profiler.profile_throughput_scaling(
            batch_sizes=args.batch_sizes,
            sequence_length=args.sequence_length,
            model_name=args.model_name
        )
        
        # Detailed profiling
        detailed_profile = {}
        if args.enable_detailed_profiling:
            logger.info("\n" + "="*60)
            logger.info("DETAILED KERNEL PROFILING")
            logger.info("="*60)
            
            detailed_profile = profiler.profile_with_torch_profiler(
                batch_size=1,
                sequence_length=args.sequence_length,
                model_name=args.model_name,
                output_dir=args.output_dir
            )
        
        # Save results
        results = {
            "model_name": args.model_name,
            "model_path": args.model_path,
            "device": args.device,
            "gpu_name": profiler.gpu_name,
            "gpu_memory_total_gb": profiler.gpu_memory_total / (1024**3),
            "profiling_settings": {
                "sequence_length": args.sequence_length,
                "warmup_iterations": args.warmup_iterations,
                "measurement_iterations": args.measurement_iterations
            },
            "single_batch_profile": asdict(single_batch_profile),
            "throughput_profile": asdict(throughput_profile),
            "detailed_profile": detailed_profile
        }
        
        # Save to JSON
        output_file = os.path.join(args.output_dir, f"{args.model_name}_profiling.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n✅ Profiling completed successfully!")
        logger.info(f"📊 Results saved to: {output_file}")
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("PROFILING SUMMARY")
        logger.info("="*60)
        logger.info(f"Model: {args.model_name}")
        logger.info(f"Single batch latency: {single_batch_profile.mean_latency:.3f} ± {single_batch_profile.std_latency:.3f} ms")
        logger.info(f"Peak throughput: {throughput_profile.max_throughput:.0f} tokens/sec (batch size {throughput_profile.optimal_batch_size})")
        logger.info(f"Memory efficient batch size: {throughput_profile.memory_efficient_batch_size}")
        logger.info(f"Peak memory usage: {single_batch_profile.peak_memory_mb:.1f} MB")
        
        return 0
        
    except Exception as e:
        logger.error(f"Profiling failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 