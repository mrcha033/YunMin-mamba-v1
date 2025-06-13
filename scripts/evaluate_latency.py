"""
Latency evaluation script for measuring wall-clock performance and throughput.
"""

import argparse
import torch
import json
from models.baseline_ssm import BaselineSSM
from utils.profiling import measure_latency, memory_usage, count_parameters
from utils.logger import setup_logger


def evaluate_model_performance(model, device="cuda", batch_sizes=[1, 4, 8, 16], seq_lengths=[512, 1024, 2048]):
    """
    Comprehensive performance evaluation of the model.
    
    Args:
        model: PyTorch model to evaluate
        device: Device to run evaluation on
        batch_sizes: List of batch sizes to test
        seq_lengths: List of sequence lengths to test
    
    Returns:
        Dictionary containing all performance metrics
    """
    logger = setup_logger("latency_eval")
    results = {}
    
    # Parameter analysis
    param_info = count_parameters(model)
    results["parameters"] = param_info
    logger.info(f"Model parameters: {param_info['total_parameters']:,}")
    
    # Memory usage
    if device == "cuda":
        mem_info = memory_usage(model, device)
        results["memory"] = mem_info
        logger.info(f"Model memory: {mem_info.get('model_memory_mb', 'N/A'):.2f} MB")
    
    # Latency measurements
    results["latency"] = {}
    
    for batch_size in batch_sizes:
        for seq_length in seq_lengths:
            key = f"batch_{batch_size}_seq_{seq_length}"
            logger.info(f"Measuring latency for {key}...")
            
            try:
                latency_info = measure_latency(
                    model,
                    input_shape=(batch_size, seq_length),
                    device=device,
                    num_warmup=20,
                    num_runs=100
                )
                
                # Calculate throughput (tokens per second)
                total_tokens = batch_size * seq_length
                throughput = total_tokens / (latency_info["mean_latency_ms"] / 1000)
                latency_info["throughput_tokens_per_sec"] = throughput
                
                results["latency"][key] = latency_info
                
                logger.info(f"  Mean latency: {latency_info['mean_latency_ms']:.2f}ms")
                logger.info(f"  Throughput: {throughput:.0f} tokens/sec")
                
            except Exception as e:
                logger.error(f"Failed to measure latency for {key}: {e}")
                results["latency"][key] = {"error": str(e)}
    
    return results


def compare_models(baseline_path, optimized_path, device="cuda"):
    """
    Compare performance between baseline and optimized models.
    
    Args:
        baseline_path: Path to baseline model checkpoint
        optimized_path: Path to optimized model checkpoint
        device: Device to run comparison on
    """
    logger = setup_logger("model_comparison")
    
    # Load models (placeholder - actual implementation would load from checkpoints)
    logger.info("Loading baseline model...")
    baseline_model = BaselineSSM(d_model=768, n_layer=12, vocab_size=50257, d_state=16, d_conv=4)
    
    logger.info("Loading optimized model...")
    optimized_model = BaselineSSM(d_model=768, n_layer=12, vocab_size=50257, d_state=16, d_conv=4)
    
    # Evaluate both models
    logger.info("Evaluating baseline model...")
    baseline_results = evaluate_model_performance(baseline_model, device, 
                                                 batch_sizes=[1, 8], seq_lengths=[1024])
    
    logger.info("Evaluating optimized model...")
    optimized_results = evaluate_model_performance(optimized_model, device,
                                                  batch_sizes=[1, 8], seq_lengths=[1024])
    
    # Calculate improvements
    comparison = {
        "baseline": baseline_results,
        "optimized": optimized_results,
        "improvements": {}
    }
    
    # Compare latency improvements
    for key in baseline_results["latency"]:
        if key in optimized_results["latency"]:
            baseline_latency = baseline_results["latency"][key]["mean_latency_ms"]
            optimized_latency = optimized_results["latency"][key]["mean_latency_ms"]
            
            speedup = baseline_latency / optimized_latency
            improvement_pct = (1 - optimized_latency / baseline_latency) * 100
            
            comparison["improvements"][key] = {
                "speedup": speedup,
                "improvement_percent": improvement_pct
            }
            
            logger.info(f"{key}: {speedup:.2f}x speedup ({improvement_pct:.1f}% improvement)")
    
    return comparison


def main():
    parser = argparse.ArgumentParser(description="Evaluate model latency and throughput")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--baseline_path", type=str, default=None,
                        help="Path to baseline model for comparison")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run evaluation on")
    parser.add_argument("--output_file", type=str, default="latency_results.json",
                        help="Output file for results")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 4, 8, 16],
                        help="Batch sizes to test")
    parser.add_argument("--seq_lengths", type=int, nargs="+", default=[512, 1024, 2048],
                        help="Sequence lengths to test")
    
    args = parser.parse_args()
    
    logger = setup_logger("latency_eval")
    
    if args.baseline_path:
        # Comparison mode
        logger.info("Running comparison evaluation...")
        results = compare_models(args.baseline_path, args.model_path, args.device)
    else:
        # Single model evaluation
        logger.info("Loading model for evaluation...")
        model = BaselineSSM(d_model=768, n_layer=12, vocab_size=50257, d_state=16, d_conv=4)
        
        logger.info("Running performance evaluation...")
        results = evaluate_model_performance(model, args.device, args.batch_sizes, args.seq_lengths)
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {args.output_file}")
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main() 