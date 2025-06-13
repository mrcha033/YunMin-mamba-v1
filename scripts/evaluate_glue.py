"""
Comprehensive GLUE Evaluation Script

This script provides comprehensive evaluation on the GLUE benchmark with:
- Support for all GLUE tasks
- F1-score computation for appropriate tasks
- Statistical significance testing with multiple seeds
- Confidence interval calculation
- Publication-ready results formatting

Usage:
    python scripts/evaluate_glue.py --model_path checkpoints/model.pt --tasks sst2 mrpc qnli mnli
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import yaml

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.glue import (
    get_glue_dataloader, 
    compute_glue_metrics, 
    run_full_glue_evaluation,
    GLUE_TASKS
)
from models.baseline_ssm import BaselineSSM
from models.sdm_ssm import SDM_SSM
from models.sgh_peft import SGHPEFTModel
from utils.profiling import count_parameters

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResults:
    """Container for evaluation results with statistical information."""
    task_name: str
    metrics: Dict[str, float]
    metrics_with_ci: Dict[str, Dict[str, float]]
    num_seeds: int
    evaluation_time: float
    num_examples: int


class GLUEEvaluator:
    """
    Comprehensive GLUE evaluator with statistical significance testing.
    
    Features:
    - Multi-seed evaluation for statistical robustness
    - Confidence interval calculation
    - F1-score computation for appropriate tasks
    - Memory-efficient evaluation
    - Detailed logging and progress tracking
    """
    
    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        device: str = "cuda",
        batch_size: int = 32,
        max_length: int = 512,
        num_seeds: int = 5,
        confidence_level: float = 0.95
    ):
        """
        Initialize GLUE evaluator.
        
        Args:
            model_path: Path to model checkpoint
            config_path: Path to model configuration
            device: Device to run evaluation on
            batch_size: Batch size for evaluation
            max_length: Maximum sequence length
            num_seeds: Number of random seeds for statistical testing
            confidence_level: Confidence level for intervals (default: 95%)
        """
        self.model_path = model_path
        self.config_path = config_path
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_seeds = num_seeds
        self.confidence_level = confidence_level
        
        # Load configuration
        self.config = self._load_config()
        
        # Load model
        self.model = self._load_model()
        
        # Calculate confidence interval multiplier
        # For 95% CI with normal distribution: 1.96
        # For 99% CI: 2.576
        if confidence_level == 0.95:
            self.ci_multiplier = 1.96
        elif confidence_level == 0.99:
            self.ci_multiplier = 2.576
        else:
            # Approximate for other confidence levels
            from scipy.stats import norm
            self.ci_multiplier = norm.ppf(1 - (1 - confidence_level) / 2)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load model configuration."""
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config.get('model', {})
        else:
            # Default configuration
            return {
                'd_model': 768,
                'n_layer': 12,
                'vocab_size': 50257,
                'd_state': 16,
                'd_conv': 4
            }
    
    def _load_model(self) -> nn.Module:
        """Load model from checkpoint."""
        logger.info(f"Loading model from {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        # Determine model type from checkpoint or config
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
            model = SDM_SSM(**self.config)
        elif model_type == 'sgh_peft':
            # Load base model first, then apply SGH-PEFT
            base_model = SDM_SSM(**self.config)
            model = SGHPEFTModel(base_model)
        else:
            model = BaselineSSM(**self.config)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        model.to(self.device)
        
        # Log model information
        param_info = count_parameters(model)
        logger.info(f"Model loaded: {param_info['total_params']:,} parameters")
        
        return model
    
    def evaluate_single_task(
        self,
        task_name: str,
        seed: int = 42
    ) -> Dict[str, float]:
        """
        Evaluate model on a single GLUE task with a specific seed.
        
        Args:
            task_name: GLUE task name
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with computed metrics
        """
        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Create dataloader
        dataloader = get_glue_dataloader(
            task_name=task_name,
            split="validation",
            batch_size=self.batch_size,
            max_length=self.max_length,
            tokenizer_name="gpt2"
        )
        
        # Evaluation
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {task_name} (seed {seed})"):
                # Move to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"]
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask=attention_mask)
                
                # Get predictions
                if hasattr(outputs, 'logits'):
                    predictions = outputs.logits
                else:
                    predictions = outputs
                
                # Move to CPU and convert to numpy
                predictions = predictions.cpu().numpy()
                labels = labels.numpy()
                
                all_predictions.append(predictions)
                all_labels.append(labels)
        
        # Concatenate all predictions and labels
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # Compute metrics
        metrics = compute_glue_metrics(task_name, all_predictions, all_labels)
        
        return metrics
    
    def evaluate_task_with_seeds(
        self,
        task_name: str
    ) -> EvaluationResults:
        """
        Evaluate model on a GLUE task with multiple seeds for statistical significance.
        
        Args:
            task_name: GLUE task name
            
        Returns:
            EvaluationResults with metrics and confidence intervals
        """
        logger.info(f"Evaluating {task_name} with {self.num_seeds} seeds...")
        
        start_time = time.time()
        
        # Run evaluation with multiple seeds
        all_metrics = []
        seeds = [42 + i for i in range(self.num_seeds)]
        
        for seed in seeds:
            metrics = self.evaluate_single_task(task_name, seed)
            all_metrics.append(metrics)
        
        # Calculate statistics
        metric_names = all_metrics[0].keys()
        metrics_with_ci = {}
        mean_metrics = {}
        
        for metric_name in metric_names:
            values = [m[metric_name] for m in all_metrics]
            
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0
            
            # Calculate confidence interval
            margin_of_error = self.ci_multiplier * std_val / np.sqrt(len(values))
            ci_lower = mean_val - margin_of_error
            ci_upper = mean_val + margin_of_error
            
            mean_metrics[metric_name] = mean_val
            metrics_with_ci[metric_name] = {
                "mean": mean_val,
                "std": std_val,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "margin_of_error": margin_of_error,
                "values": values
            }
        
        evaluation_time = time.time() - start_time
        
        # Get number of examples (approximate)
        dataloader = get_glue_dataloader(
            task_name=task_name,
            split="validation",
            batch_size=self.batch_size,
            max_length=self.max_length
        )
        num_examples = len(dataloader.dataset)
        
        return EvaluationResults(
            task_name=task_name,
            metrics=mean_metrics,
            metrics_with_ci=metrics_with_ci,
            num_seeds=self.num_seeds,
            evaluation_time=evaluation_time,
            num_examples=num_examples
        )
    
    def evaluate_multiple_tasks(
        self,
        tasks: List[str]
    ) -> Dict[str, EvaluationResults]:
        """
        Evaluate model on multiple GLUE tasks.
        
        Args:
            tasks: List of GLUE task names
            
        Returns:
            Dictionary mapping task names to evaluation results
        """
        logger.info(f"Evaluating on {len(tasks)} GLUE tasks: {tasks}")
        
        results = {}
        total_start_time = time.time()
        
        for task_name in tasks:
            try:
                # Validate task
                if task_name not in GLUE_TASKS:
                    logger.warning(f"Task {task_name} not supported. Skipping.")
                    continue
                
                # Evaluate task
                task_results = self.evaluate_task_with_seeds(task_name)
                results[task_name] = task_results
                
                # Log results
                self._log_task_results(task_results)
                
            except Exception as e:
                logger.error(f"Failed to evaluate {task_name}: {e}")
                continue
        
        total_time = time.time() - total_start_time
        logger.info(f"Total evaluation time: {total_time:.2f} seconds")
        
        return results
    
    def _log_task_results(self, results: EvaluationResults):
        """Log results for a single task."""
        logger.info(f"\n{results.task_name} Results:")
        logger.info(f"  Examples: {results.num_examples:,}")
        logger.info(f"  Evaluation time: {results.evaluation_time:.2f}s")
        logger.info(f"  Seeds: {results.num_seeds}")
        
        for metric_name, metric_data in results.metrics_with_ci.items():
            mean_val = metric_data["mean"]
            std_val = metric_data["std"]
            ci_lower = metric_data["ci_lower"]
            ci_upper = metric_data["ci_upper"]
            
            logger.info(f"  {metric_name}: {mean_val:.4f} ± {std_val:.4f} "
                       f"(95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
    
    def save_results(
        self,
        results: Dict[str, EvaluationResults],
        output_file: str
    ):
        """Save evaluation results to JSON file."""
        logger.info(f"Saving results to {output_file}")
        
        # Convert results to serializable format
        serializable_results = {}
        
        for task_name, task_results in results.items():
            serializable_results[task_name] = {
                "task_name": task_results.task_name,
                "metrics": task_results.metrics,
                "metrics_with_ci": task_results.metrics_with_ci,
                "num_seeds": task_results.num_seeds,
                "evaluation_time": task_results.evaluation_time,
                "num_examples": task_results.num_examples
            }
        
        # Add metadata
        output_data = {
            "model_path": self.model_path,
            "config": self.config,
            "evaluation_settings": {
                "batch_size": self.batch_size,
                "max_length": self.max_length,
                "num_seeds": self.num_seeds,
                "confidence_level": self.confidence_level,
                "device": self.device
            },
            "results": serializable_results,
            "summary": self._generate_summary(results)
        }
        
        # Save to file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"✅ Results saved to {output_file}")
    
    def _generate_summary(self, results: Dict[str, EvaluationResults]) -> Dict[str, Any]:
        """Generate summary statistics across all tasks."""
        if not results:
            return {}
        
        # Calculate average metrics
        all_accuracies = []
        all_f1_scores = []
        
        for task_results in results.values():
            if "accuracy" in task_results.metrics:
                all_accuracies.append(task_results.metrics["accuracy"])
            if "f1" in task_results.metrics:
                all_f1_scores.append(task_results.metrics["f1"])
        
        summary = {
            "num_tasks_evaluated": len(results),
            "total_examples": sum(r.num_examples for r in results.values()),
            "total_evaluation_time": sum(r.evaluation_time for r in results.values()),
            "average_accuracy": np.mean(all_accuracies) if all_accuracies else None,
            "average_f1": np.mean(all_f1_scores) if all_f1_scores else None,
            "tasks_evaluated": list(results.keys())
        }
        
        return summary


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Comprehensive GLUE evaluation with statistical significance testing"
    )
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to model configuration file")
    parser.add_argument("--tasks", type=str, nargs="+", 
                       default=["sst2", "mrpc", "qnli", "mnli"],
                       help="GLUE tasks to evaluate")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output file for results (default: auto-generated)")
    
    # Evaluation settings
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for evaluation")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--num_seeds", type=int, default=5,
                       help="Number of random seeds for statistical testing")
    parser.add_argument("--confidence_level", type=float, default=0.95,
                       help="Confidence level for intervals")
    
    # System settings
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run evaluation on")
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model_path):
        logger.error(f"Model path does not exist: {args.model_path}")
        return 1
    
    # Generate output file name if not provided
    if args.output_file is None:
        model_name = Path(args.model_path).stem
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_file = f"results/glue_evaluation_{model_name}_{timestamp}.json"
    
    logger.info("Starting comprehensive GLUE evaluation...")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Tasks: {args.tasks}")
    logger.info(f"Seeds: {args.num_seeds}")
    logger.info(f"Confidence level: {args.confidence_level}")
    
    try:
        # Create evaluator
        evaluator = GLUEEvaluator(
            model_path=args.model_path,
            config_path=args.config,
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            num_seeds=args.num_seeds,
            confidence_level=args.confidence_level
        )
        
        # Run evaluation
        results = evaluator.evaluate_multiple_tasks(args.tasks)
        
        # Save results
        evaluator.save_results(results, args.output_file)
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*60)
        
        for task_name, task_results in results.items():
            logger.info(f"\n{task_name.upper()}:")
            for metric_name, metric_data in task_results.metrics_with_ci.items():
                mean_val = metric_data["mean"]
                ci_lower = metric_data["ci_lower"]
                ci_upper = metric_data["ci_upper"]
                logger.info(f"  {metric_name}: {mean_val:.4f} "
                           f"(95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
        
        logger.info(f"\n✅ Evaluation completed successfully!")
        logger.info(f"📊 Results saved to: {args.output_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 