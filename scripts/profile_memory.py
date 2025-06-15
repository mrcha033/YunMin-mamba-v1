"""
Comprehensive GPU Memory Profiling

This script provides detailed GPU memory analysis for:
- Training memory requirements (forward + backward + optimizer states)
- Inference memory usage and peak allocation
- Memory efficiency comparison between model variants
- Batch size scaling and memory optimization
- A100-specific memory utilization analysis

Features:
- Peak memory tracking during training and inference
- Optimizer state memory analysis
- Memory fragmentation detection
- Batch size optimization for memory efficiency
- Detailed memory breakdown by component
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
import gc

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import yaml

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.baseline_ssm import BaselineSSM
from models.sdm_ssm import SDM_SSM
from models.sgh_peft import SGHPEFTModel
from utils.profiling import count_parameters
from data.wikitext103 import get_wikitext103_dataloader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Container for memory usage at a specific point."""
    allocated_mb: float
    reserved_mb: float
    peak_allocated_mb: float
    peak_reserved_mb: float
    free_mb: float
    total_mb: float
    utilization_percent: float
    timestamp: str


@dataclass
class InferenceMemoryProfile:
    """Container for inference memory profiling results."""
    model_name: str
    batch_size: int
    sequence_length: int
    
    # Memory usage (MB)
    model_memory_mb: float
    input_memory_mb: float
    activation_memory_mb: float
    peak_memory_mb: float
    total_allocated_mb: float
    
    # Memory efficiency metrics
    memory_per_token: float
    memory_per_parameter: float
    memory_utilization_percent: float
    
    # Batch scaling analysis
    memory_scaling_factor: float  # How memory scales with batch size
    optimal_batch_size: int       # Largest batch size that fits in memory


@dataclass
class TrainingMemoryProfile:
    """Container for training memory profiling results."""
    model_name: str
    batch_size: int
    sequence_length: int
    optimizer_name: str
    
    # Memory breakdown (MB)
    model_parameters_mb: float
    gradients_mb: float
    optimizer_states_mb: float
    activations_mb: float
    peak_memory_mb: float
    total_allocated_mb: float
    
    # Training-specific metrics
    memory_per_trainable_param: float
    gradient_memory_ratio: float      # gradients / parameters
    optimizer_memory_ratio: float     # optimizer states / parameters
    
    # Memory efficiency
    memory_utilization_percent: float
    can_fit_larger_batch: bool
    recommended_batch_size: int


class GPUMemoryProfiler:
    """
    Comprehensive GPU memory profiler for deep learning models.
    
    Features:
    - Detailed memory breakdown by component
    - Training vs inference memory analysis
    - Batch size optimization
    - Memory efficiency recommendations
    - A100-specific optimizations
    """
    
    def __init__(
        self,
        device: str = "cuda",
        memory_threshold: float = 0.9  # Use up to 90% of GPU memory
    ):
        """
        Initialize GPU memory profiler.
        
        Args:
            device: CUDA device to profile
            memory_threshold: Maximum memory utilization threshold
        """
        self.device = device
        self.memory_threshold = memory_threshold
        
        # Verify CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available for memory profiling")
        
        # Get GPU information
        self.gpu_properties = torch.cuda.get_device_properties(device)
        self.total_memory_mb = self.gpu_properties.total_memory / (1024 ** 2)
        self.gpu_name = self.gpu_properties.name
        
        logger.info(f"Memory profiling on: {self.gpu_name}")
        logger.info(f"Total GPU memory: {self.total_memory_mb:.0f} MB ({self.total_memory_mb/1024:.1f} GB)")
        logger.info(f"Memory threshold: {memory_threshold*100:.0f}%")
    
    def get_memory_snapshot(self, label: str = "") -> MemorySnapshot:
        """Get current memory usage snapshot."""
        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        peak_allocated = torch.cuda.max_memory_allocated(self.device)
        peak_reserved = torch.cuda.max_memory_reserved(self.device)
        
        allocated_mb = allocated / (1024 ** 2)
        reserved_mb = reserved / (1024 ** 2)
        peak_allocated_mb = peak_allocated / (1024 ** 2)
        peak_reserved_mb = peak_reserved / (1024 ** 2)
        free_mb = self.total_memory_mb - reserved_mb
        utilization_percent = (reserved_mb / self.total_memory_mb) * 100
        
        snapshot = MemorySnapshot(
            allocated_mb=allocated_mb,
            reserved_mb=reserved_mb,
            peak_allocated_mb=peak_allocated_mb,
            peak_reserved_mb=peak_reserved_mb,
            free_mb=free_mb,
            total_mb=self.total_memory_mb,
            utilization_percent=utilization_percent,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        if label:
            logger.info(f"Memory snapshot ({label}): "
                       f"{allocated_mb:.1f} MB allocated, "
                       f"{reserved_mb:.1f} MB reserved, "
                       f"{utilization_percent:.1f}% utilization")
        
        return snapshot
    
    def profile_model_memory(self, model: nn.Module) -> Dict[str, float]:
        """Profile memory usage of model parameters."""
        param_memory = 0
        buffer_memory = 0
        
        for param in model.parameters():
            param_memory += param.numel() * param.element_size()
        
        for buffer in model.buffers():
            buffer_memory += buffer.numel() * buffer.element_size()
        
        total_memory = param_memory + buffer_memory
        
        return {
            "parameters_mb": param_memory / (1024 ** 2),
            "buffers_mb": buffer_memory / (1024 ** 2),
            "total_mb": total_memory / (1024 ** 2)
        }
    
    def profile_inference_memory(
        self,
        model: nn.Module,
        batch_size: int = 1,
        sequence_length: int = 1024,
        model_name: str = "model"
    ) -> InferenceMemoryProfile:
        """
        Profile memory usage during inference.
        
        Args:
            model: Model to profile
            batch_size: Batch size for profiling
            sequence_length: Sequence length
            model_name: Name for identification
            
        Returns:
            InferenceMemoryProfile with detailed memory analysis
        """
        logger.info(f"Profiling inference memory: {model_name} (batch={batch_size}, seq_len={sequence_length})")
        
        # Reset memory stats and clear cache
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.empty_cache()
        gc.collect()
        
        model.eval()
        model.to(self.device)
        
        # Measure model memory
        baseline_snapshot = self.get_memory_snapshot("baseline")
        model_memory = self.profile_model_memory(model)
        
        # Create input tensor
        dummy_input = torch.randint(
            0, 50257, (batch_size, sequence_length),
            device=self.device, dtype=torch.long
        )
        
        input_snapshot = self.get_memory_snapshot("with input")
        input_memory_mb = input_snapshot.allocated_mb - baseline_snapshot.allocated_mb
        
        # Forward pass
        with torch.no_grad():
            _ = model(dummy_input)
        
        forward_snapshot = self.get_memory_snapshot("after forward")
        
        # Calculate memory components
        activation_memory_mb = forward_snapshot.peak_allocated_mb - input_snapshot.allocated_mb
        peak_memory_mb = forward_snapshot.peak_allocated_mb
        
        # Memory efficiency metrics
        total_tokens = batch_size * sequence_length
        memory_per_token = peak_memory_mb / total_tokens
        
        param_count = sum(p.numel() for p in model.parameters())
        memory_per_parameter = peak_memory_mb / param_count
        
        memory_utilization_percent = (peak_memory_mb / self.total_memory_mb) * 100
        
        # Estimate memory scaling with batch size
        # Assume linear scaling for activations, constant for model
        memory_scaling_factor = activation_memory_mb / batch_size
        
        # Find optimal batch size
        available_memory = self.total_memory_mb * self.memory_threshold
        fixed_memory = model_memory["total_mb"] + input_memory_mb
        variable_memory_per_batch = memory_scaling_factor
        
        if variable_memory_per_batch > 0:
            optimal_batch_size = int((available_memory - fixed_memory) / variable_memory_per_batch)
            optimal_batch_size = max(1, optimal_batch_size)
        else:
            optimal_batch_size = batch_size
        
        logger.info(f"Inference memory profile for {model_name}:")
        logger.info(f"  Model memory: {model_memory['total_mb']:.1f} MB")
        logger.info(f"  Input memory: {input_memory_mb:.1f} MB")
        logger.info(f"  Activation memory: {activation_memory_mb:.1f} MB")
        logger.info(f"  Peak memory: {peak_memory_mb:.1f} MB")
        logger.info(f"  Memory per token: {memory_per_token:.3f} MB")
        logger.info(f"  Optimal batch size: {optimal_batch_size}")
        
        return InferenceMemoryProfile(
            model_name=model_name,
            batch_size=batch_size,
            sequence_length=sequence_length,
            model_memory_mb=model_memory["total_mb"],
            input_memory_mb=input_memory_mb,
            activation_memory_mb=activation_memory_mb,
            peak_memory_mb=peak_memory_mb,
            total_allocated_mb=forward_snapshot.allocated_mb,
            memory_per_token=memory_per_token,
            memory_per_parameter=memory_per_parameter,
            memory_utilization_percent=memory_utilization_percent,
            memory_scaling_factor=memory_scaling_factor,
            optimal_batch_size=optimal_batch_size
        )
    
    def profile_training_memory(
        self,
        model: nn.Module,
        batch_size: int = 8,
        sequence_length: int = 1024,
        optimizer_class: type = optim.AdamW,
        model_name: str = "model"
    ) -> TrainingMemoryProfile:
        """
        Profile memory usage during training.
        
        Args:
            model: Model to profile
            batch_size: Batch size for training
            sequence_length: Sequence length
            optimizer_class: Optimizer class to use
            model_name: Name for identification
            
        Returns:
            TrainingMemoryProfile with detailed training memory analysis
        """
        logger.info(f"Profiling training memory: {model_name} (batch={batch_size}, seq_len={sequence_length})")
        
        # Reset memory stats and clear cache
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.empty_cache()
        gc.collect()
        
        model.train()
        model.to(self.device)
        
        # Measure model parameters memory
        baseline_snapshot = self.get_memory_snapshot("baseline")
        model_memory = self.profile_model_memory(model)
        
        # Create optimizer
        optimizer = optimizer_class(model.parameters(), lr=1e-4)
        optimizer_snapshot = self.get_memory_snapshot("with optimizer")
        
        # Estimate optimizer state memory
        optimizer_memory_mb = optimizer_snapshot.allocated_mb - baseline_snapshot.allocated_mb - model_memory["total_mb"]
        
        # Create input and target tensors
        input_ids = torch.randint(
            0, 50257, (batch_size, sequence_length),
            device=self.device, dtype=torch.long
        )
        labels = input_ids.clone()
        
        input_snapshot = self.get_memory_snapshot("with input")
        
        # Forward pass
        outputs = model(input_ids)
        
        # Calculate loss
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs
        
        # Reshape for loss calculation
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        forward_snapshot = self.get_memory_snapshot("after forward")
        
        # Backward pass
        loss.backward()
        
        backward_snapshot = self.get_memory_snapshot("after backward")
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        final_snapshot = self.get_memory_snapshot("after optimizer step")
        
        # Calculate memory components
        activations_mb = forward_snapshot.peak_allocated_mb - input_snapshot.allocated_mb
        gradients_mb = backward_snapshot.peak_allocated_mb - forward_snapshot.peak_allocated_mb
        peak_memory_mb = max(forward_snapshot.peak_allocated_mb, backward_snapshot.peak_allocated_mb)
        
        # Training-specific metrics
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        memory_per_trainable_param = peak_memory_mb / trainable_params if trainable_params > 0 else 0
        
        gradient_memory_ratio = gradients_mb / model_memory["total_mb"] if model_memory["total_mb"] > 0 else 0
        optimizer_memory_ratio = optimizer_memory_mb / model_memory["total_mb"] if model_memory["total_mb"] > 0 else 0
        
        memory_utilization_percent = (peak_memory_mb / self.total_memory_mb) * 100
        
        # Check if larger batch size can fit
        available_memory = self.total_memory_mb * self.memory_threshold
        can_fit_larger_batch = peak_memory_mb < available_memory * 0.8  # Leave 20% margin
        
        # Recommend batch size
        if can_fit_larger_batch:
            memory_per_sample = peak_memory_mb / batch_size
            recommended_batch_size = int(available_memory * 0.8 / memory_per_sample)
            recommended_batch_size = max(batch_size, recommended_batch_size)
        else:
            # Try to find smaller batch size that fits
            memory_per_sample = peak_memory_mb / batch_size
            recommended_batch_size = int(available_memory * 0.8 / memory_per_sample)
            recommended_batch_size = max(1, recommended_batch_size)
        
        logger.info(f"Training memory profile for {model_name}:")
        logger.info(f"  Model parameters: {model_memory['total_mb']:.1f} MB")
        logger.info(f"  Gradients: {gradients_mb:.1f} MB")
        logger.info(f"  Optimizer states: {optimizer_memory_mb:.1f} MB")
        logger.info(f"  Activations: {activations_mb:.1f} MB")
        logger.info(f"  Peak memory: {peak_memory_mb:.1f} MB")
        logger.info(f"  Memory utilization: {memory_utilization_percent:.1f}%")
        logger.info(f"  Recommended batch size: {recommended_batch_size}")
        
        return TrainingMemoryProfile(
            model_name=model_name,
            batch_size=batch_size,
            sequence_length=sequence_length,
            optimizer_name=optimizer_class.__name__,
            model_parameters_mb=model_memory["total_mb"],
            gradients_mb=gradients_mb,
            optimizer_states_mb=optimizer_memory_mb,
            activations_mb=activations_mb,
            peak_memory_mb=peak_memory_mb,
            total_allocated_mb=final_snapshot.allocated_mb,
            memory_per_trainable_param=memory_per_trainable_param,
            gradient_memory_ratio=gradient_memory_ratio,
            optimizer_memory_ratio=optimizer_memory_ratio,
            memory_utilization_percent=memory_utilization_percent,
            can_fit_larger_batch=can_fit_larger_batch,
            recommended_batch_size=recommended_batch_size
        )
    
    def compare_model_memory_efficiency(
        self,
        models: Dict[str, nn.Module],
        batch_size: int = 8,
        sequence_length: int = 1024
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare memory efficiency across multiple models.
        
        Args:
            models: Dictionary of model_name -> model
            batch_size: Batch size for comparison
            sequence_length: Sequence length
            
        Returns:
            Dictionary with comparative analysis
        """
        logger.info(f"Comparing memory efficiency across {len(models)} models")
        
        results = {}
        
        for model_name, model in models.items():
            try:
                # Profile inference memory
                inference_profile = self.profile_inference_memory(
                    model, batch_size, sequence_length, model_name
                )
                
                # Profile training memory
                training_profile = self.profile_training_memory(
                    model, batch_size, sequence_length, optim.AdamW, model_name
                )
                
                # Calculate efficiency metrics
                param_count = sum(p.numel() for p in model.parameters())
                trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                efficiency_metrics = {
                    "parameter_efficiency": trainable_param_count / param_count if param_count > 0 else 0,
                    "memory_efficiency_inference": inference_profile.memory_utilization_percent,
                    "memory_efficiency_training": training_profile.memory_utilization_percent,
                    "memory_per_parameter": inference_profile.memory_per_parameter,
                    "optimal_batch_size_inference": inference_profile.optimal_batch_size,
                    "optimal_batch_size_training": training_profile.recommended_batch_size
                }
                
                results[model_name] = {
                    "inference_profile": asdict(inference_profile),
                    "training_profile": asdict(training_profile),
                    "efficiency_metrics": efficiency_metrics,
                    "parameter_count": param_count,
                    "trainable_parameter_count": trainable_param_count
                }
                
            except Exception as e:
                logger.error(f"Failed to profile {model_name}: {e}")
                results[model_name] = {"error": str(e)}
        
        # Generate comparative summary
        summary = self._generate_memory_comparison_summary(results)
        results["comparison_summary"] = summary
        
        return results
    
    def _generate_memory_comparison_summary(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of memory comparison."""
        valid_results = {k: v for k, v in results.items() if "error" not in v}
        
        if not valid_results:
            return {"error": "No valid results to compare"}
        
        # Find most memory efficient models
        inference_efficiency = {
            name: data["efficiency_metrics"]["memory_efficiency_inference"]
            for name, data in valid_results.items()
        }
        
        training_efficiency = {
            name: data["efficiency_metrics"]["memory_efficiency_training"]
            for name, data in valid_results.items()
        }
        
        parameter_efficiency = {
            name: data["efficiency_metrics"]["parameter_efficiency"]
            for name, data in valid_results.items()
        }
        
        summary = {
            "most_memory_efficient_inference": min(inference_efficiency.items(), key=lambda x: x[1]),
            "most_memory_efficient_training": min(training_efficiency.items(), key=lambda x: x[1]),
            "most_parameter_efficient": max(parameter_efficiency.items(), key=lambda x: x[1]),
            "average_inference_memory_usage": np.mean(list(inference_efficiency.values())),
            "average_training_memory_usage": np.mean(list(training_efficiency.values())),
            "average_parameter_efficiency": np.mean(list(parameter_efficiency.values()))
        }
        
        return summary


def load_model_from_checkpoint(
    checkpoint_path: str,
    config: Dict[str, Any],
    device: str = "cuda"
) -> nn.Module:
    """Load model from checkpoint."""
    logger.info(f"Loading model from {checkpoint_path}")
    
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
    
    return model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Comprehensive GPU memory profiling for deep learning models"
    )
    
    parser.add_argument("--model_paths", type=str, nargs="+", required=True,
                       help="Paths to model checkpoints")
    parser.add_argument("--model_names", type=str, nargs="+", default=None,
                       help="Names for models (default: use checkpoint filenames)")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to model configuration file")
    
    # Profiling settings
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 4, 8, 16],
                       help="Batch sizes to test")
    parser.add_argument("--sequence_length", type=int, default=1024,
                       help="Sequence length for profiling")
    parser.add_argument("--profile_training", action="store_true",
                       help="Profile training memory usage")
    parser.add_argument("--profile_inference", action="store_true", default=True,
                       help="Profile inference memory usage")
    
    # Output settings
    parser.add_argument("--output_dir", type=str, default="memory_profiling_results",
                       help="Output directory for results")
    
    # System settings
    parser.add_argument("--device", type=str, default="cuda",
                       help="CUDA device to use")
    parser.add_argument("--memory_threshold", type=float, default=0.9,
                       help="Maximum memory utilization threshold")
    
    return parser.parse_args()


def main():
    """Main memory profiling function."""
    args = parse_args()
    
    # Validate inputs
    for model_path in args.model_paths:
        if not os.path.exists(model_path):
            logger.error(f"Model path does not exist: {model_path}")
            return 1
    
    # Generate model names if not provided
    if args.model_names is None:
        args.model_names = [Path(path).stem for path in args.model_paths]
    
    if len(args.model_names) != len(args.model_paths):
        logger.error("Number of model names must match number of model paths")
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
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("Starting comprehensive GPU memory profiling...")
    logger.info(f"Models: {args.model_names}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Batch sizes: {args.batch_sizes}")
    logger.info(f"Sequence length: {args.sequence_length}")
    
    try:
        # Create profiler
        profiler = GPUMemoryProfiler(
            device=args.device,
            memory_threshold=args.memory_threshold
        )
        
        # Load models
        models = {}
        for model_path, model_name in zip(args.model_paths, args.model_names):
            models[model_name] = load_model_from_checkpoint(model_path, model_config, args.device)
        
        # Profile each batch size
        all_results = {}
        
        for batch_size in args.batch_sizes:
            logger.info(f"\n{'='*60}")
            logger.info(f"PROFILING BATCH SIZE: {batch_size}")
            logger.info(f"{'='*60}")
            
            batch_results = {}
            
            for model_name, model in models.items():
                try:
                    model_results = {}
                    
                    if args.profile_inference:
                        inference_profile = profiler.profile_inference_memory(
                            model, batch_size, args.sequence_length, model_name
                        )
                        model_results["inference"] = asdict(inference_profile)
                    
                    if args.profile_training:
                        training_profile = profiler.profile_training_memory(
                            model, batch_size, args.sequence_length, optim.AdamW, model_name
                        )
                        model_results["training"] = asdict(training_profile)
                    
                    batch_results[model_name] = model_results
                    
                except torch.cuda.OutOfMemoryError:
                    logger.warning(f"OOM for {model_name} at batch size {batch_size}")
                    batch_results[model_name] = {"error": "OutOfMemoryError"}
                except Exception as e:
                    logger.error(f"Error profiling {model_name} at batch size {batch_size}: {e}")
                    batch_results[model_name] = {"error": str(e)}
            
            all_results[f"batch_size_{batch_size}"] = batch_results
        
        # Generate comparative analysis
        logger.info(f"\n{'='*60}")
        logger.info("COMPARATIVE ANALYSIS")
        logger.info(f"{'='*60}")
        
        # Use first successful batch size for comparison
        comparison_batch_size = args.batch_sizes[0]
        comparison_results = profiler.compare_model_memory_efficiency(
            models, comparison_batch_size, args.sequence_length
        )
        
        # Combine all results
        final_results = {
            "profiling_settings": {
                "batch_sizes": args.batch_sizes,
                "sequence_length": args.sequence_length,
                "device": args.device,
                "gpu_name": profiler.gpu_name,
                "total_memory_mb": profiler.total_memory_mb,
                "memory_threshold": args.memory_threshold
            },
            "batch_size_results": all_results,
            "comparative_analysis": comparison_results
        }
        
        # Save results
        output_file = os.path.join(args.output_dir, "memory_profiling_results.json")
        with open(output_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        logger.info(f"\n✅ Memory profiling completed successfully!")
        logger.info(f"📊 Results saved to: {output_file}")
        
        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("MEMORY PROFILING SUMMARY")
        logger.info(f"{'='*60}")
        
        if "comparison_summary" in comparison_results:
            summary = comparison_results["comparison_summary"]
            logger.info(f"Most memory efficient (inference): {summary.get('most_memory_efficient_inference', 'N/A')}")
            logger.info(f"Most memory efficient (training): {summary.get('most_memory_efficient_training', 'N/A')}")
            logger.info(f"Most parameter efficient: {summary.get('most_parameter_efficient', 'N/A')}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Memory profiling failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 