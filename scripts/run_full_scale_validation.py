"""
Full-Scale Production-Ready Validation Suite

This script orchestrates the complete full-scale validation pipeline addressing all gaps:
1. Scale Factors: Full Mamba-130M/370M models with complete datasets
2. Metric Completeness: Full GLUE suite with F1-scores and confidence intervals
3. Hardware Validation: High-precision A100 profiling with memory analysis

Features:
- Complete model scaling (130M/370M parameters)
- Full WikiText-103 and GLUE benchmark datasets
- Statistical significance testing with multiple seeds
- High-precision hardware profiling
- Memory efficiency analysis
- Publication-ready results and visualizations
"""

import argparse
import json
import os
import sys
import time
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass, asdict
import subprocess
import yaml

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.evaluate_glue import GLUEEvaluator
from scripts.evaluate_latency import A100Profiler, load_model_from_checkpoint
from scripts.profile_memory import GPUMemoryProfiler
from scripts.analyze_results import ResultsAnalyzer
from utils.profiling import count_parameters

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class FullScaleValidationConfig:
    """Configuration for full-scale validation."""
    # Model configurations
    model_sizes: List[str]  # ["130m", "370m"]
    model_variants: List[str]  # ["M_base", "M_CSP", "M_SDM", "M_full"]
    
    # Dataset configurations
    use_full_datasets: bool = True
    wikitext_split: str = "train"
    glue_tasks: List[str] = None
    
    # Evaluation configurations
    num_seeds: int = 5
    confidence_level: float = 0.95
    batch_sizes: List[int] = None
    sequence_length: int = 1024
    
    # Hardware profiling
    enable_hardware_profiling: bool = True
    enable_memory_profiling: bool = True
    target_device: str = "cuda"
    
    # Output configurations
    output_dir: str = "full_scale_validation_results"
    save_detailed_logs: bool = True
    generate_plots: bool = True


class FullScaleValidator:
    """
    Complete full-scale validation orchestrator.
    
    This class coordinates all validation components to provide
    comprehensive, publication-ready experimental validation.
    """
    
    def __init__(self, config: FullScaleValidationConfig):
        """Initialize full-scale validator."""
        self.config = config
        
        # Set default values
        if self.config.glue_tasks is None:
            # Default subset of GLUE tasks
            self.config.glue_tasks = ["sst2", "mrpc", "qnli", "mnli"]
        
        if self.config.batch_sizes is None:
            self.config.batch_sizes = [1, 2, 4, 8, 16, 32]
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Setup logging
        log_file = os.path.join(self.config.output_dir, "validation.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info("Initialized Full-Scale Validator")
        logger.info(f"Configuration: {asdict(self.config)}")
    
    def validate_environment(self) -> bool:
        """Validate that the environment is ready for full-scale validation."""
        logger.info("Validating environment...")
        
        # Check CUDA availability
        import torch
        if not torch.cuda.is_available():
            logger.error("CUDA not available - required for full-scale validation")
            return False
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Check for A100 (recommended)
        if "A100" not in gpu_name:
            logger.warning(f"Recommended GPU is A100, found: {gpu_name}")
        
        # Check memory requirements
        if gpu_memory < 40:  # 40GB minimum for 370M model
            logger.warning(f"GPU memory may be insufficient for 370M model: {gpu_memory:.1f} GB")
        
        # Check disk space
        import shutil
        free_space_gb = shutil.disk_usage(self.config.output_dir).free / (1024**3)
        if free_space_gb < 50:  # 50GB minimum for full datasets and results
            logger.warning(f"Low disk space: {free_space_gb:.1f} GB available")
        
        # Check required packages
        required_packages = ["datasets", "transformers", "torch", "numpy", "scipy", "sklearn"]
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"Missing required packages: {missing_packages}")
            return False
        
        logger.info("✅ Environment validation passed")
        return True
    
    def prepare_full_scale_datasets(self) -> Dict[str, Any]:
        """Prepare full-scale datasets for validation."""
        logger.info("Preparing full-scale datasets...")
        
        dataset_info = {}
        
        if self.config.use_full_datasets:
            # Prepare WikiText-103
            logger.info("Preparing WikiText-103 dataset...")
            try:
                from data.wikitext103 import verify_dataset_loading, calculate_dataset_stats
                
                # Verify dataset loading
                if verify_dataset_loading():
                    stats = calculate_dataset_stats(split=self.config.wikitext_split)
                    dataset_info["wikitext103"] = {
                        "status": "ready",
                        "stats": stats
                    }
                    logger.info(f"✅ WikiText-103 ready: {stats['estimated_total_samples']:,} samples")
                else:
                    dataset_info["wikitext103"] = {"status": "failed"}
                    logger.error("❌ WikiText-103 preparation failed")
            
            except Exception as e:
                logger.error(f"WikiText-103 preparation error: {e}")
                dataset_info["wikitext103"] = {"status": "error", "message": str(e)}
            
            # Prepare GLUE datasets
            logger.info("Preparing GLUE datasets...")
            try:
                from data.glue import verify_glue_loading, GLUE_TASKS
                
                if verify_glue_loading():
                    glue_info = {}
                    for task in self.config.glue_tasks:
                        if task in GLUE_TASKS:
                            glue_info[task] = {
                                "status": "ready",
                                "config": asdict(GLUE_TASKS[task])
                            }
                        else:
                            glue_info[task] = {"status": "unsupported"}
                    
                    dataset_info["glue"] = glue_info
                    logger.info(f"✅ GLUE ready: {len([t for t in glue_info.values() if t['status'] == 'ready'])} tasks")
                else:
                    dataset_info["glue"] = {"status": "failed"}
                    logger.error("❌ GLUE preparation failed")
            
            except Exception as e:
                logger.error(f"GLUE preparation error: {e}")
                dataset_info["glue"] = {"status": "error", "message": str(e)}
        
        return dataset_info
    
    def run_comprehensive_glue_evaluation(
        self,
        model_paths: Dict[str, str],
        model_configs: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run comprehensive GLUE evaluation with statistical significance."""
        logger.info("Running comprehensive GLUE evaluation...")
        
        glue_results = {}
        
        for model_name, model_path in model_paths.items():
            if not os.path.exists(model_path):
                logger.warning(f"Model path not found: {model_path}")
                continue
            
            logger.info(f"Evaluating {model_name} on GLUE...")
            
            try:
                # Get model config
                config = model_configs.get(model_name, {})
                
                # Create GLUE evaluator
                evaluator = GLUEEvaluator(
                    model_path=model_path,
                    config_path=None,  # Pass config directly
                    device=self.config.target_device,
                    batch_size=32,  # Optimized for A100
                    max_length=512,  # Standard GLUE sequence length
                    num_seeds=self.config.num_seeds,
                    confidence_level=self.config.confidence_level
                )
                
                # Override config
                evaluator.config = config
                
                # Run evaluation on all GLUE tasks
                task_results = evaluator.evaluate_multiple_tasks(self.config.glue_tasks)
                
                # Save individual model results
                model_output_file = os.path.join(
                    self.config.output_dir, 
                    f"glue_results_{model_name}.json"
                )
                evaluator.save_results(task_results, model_output_file)
                
                glue_results[model_name] = {
                    "results": {task: asdict(result) for task, result in task_results.items()},
                    "output_file": model_output_file
                }
                
                logger.info(f"✅ {model_name} GLUE evaluation completed")
                
            except Exception as e:
                logger.error(f"GLUE evaluation failed for {model_name}: {e}")
                glue_results[model_name] = {"error": str(e)}
        
        return glue_results
    
    def run_hardware_profiling(
        self,
        model_paths: Dict[str, str],
        model_configs: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run comprehensive hardware profiling."""
        logger.info("Running comprehensive hardware profiling...")
        
        hardware_results = {}
        
        if not self.config.enable_hardware_profiling:
            logger.info("Hardware profiling disabled")
            return hardware_results
        
        for model_name, model_path in model_paths.items():
            if not os.path.exists(model_path):
                continue
            
            logger.info(f"Profiling hardware performance for {model_name}...")
            
            try:
                # Load model
                config = model_configs.get(model_name, {})
                model = load_model_from_checkpoint(model_path, config, self.config.target_device)
                
                # Create A100 profiler
                profiler = A100Profiler(
                    model=model,
                    device=self.config.target_device,
                    warmup_iterations=20,  # More warmup for precision
                    measurement_iterations=200,  # More measurements for statistical robustness
                    enable_profiling=True
                )
                
                # Profile single batch latency
                latency_profile = profiler.profile_latency(
                    batch_size=1,
                    sequence_length=self.config.sequence_length,
                    model_name=model_name
                )
                
                # Profile throughput scaling
                throughput_profile = profiler.profile_throughput_scaling(
                    batch_sizes=self.config.batch_sizes,
                    sequence_length=self.config.sequence_length,
                    model_name=model_name
                )
                
                # Detailed profiling
                detailed_profile = profiler.profile_with_torch_profiler(
                    batch_size=1,
                    sequence_length=self.config.sequence_length,
                    model_name=model_name,
                    output_dir=os.path.join(self.config.output_dir, "profiling_traces")
                )
                
                hardware_results[model_name] = {
                    "latency_profile": asdict(latency_profile),
                    "throughput_profile": asdict(throughput_profile),
                    "detailed_profile": detailed_profile
                }
                
                logger.info(f"✅ {model_name} hardware profiling completed")
                
            except Exception as e:
                logger.error(f"Hardware profiling failed for {model_name}: {e}")
                hardware_results[model_name] = {"error": str(e)}
        
        return hardware_results
    
    def run_memory_profiling(
        self,
        model_paths: Dict[str, str],
        model_configs: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run comprehensive memory profiling."""
        logger.info("Running comprehensive memory profiling...")
        
        memory_results = {}
        
        if not self.config.enable_memory_profiling:
            logger.info("Memory profiling disabled")
            return memory_results
        
        try:
            # Create memory profiler
            profiler = GPUMemoryProfiler(
                device=self.config.target_device,
                memory_threshold=0.9
            )
            
            # Load all models
            models = {}
            for model_name, model_path in model_paths.items():
                if os.path.exists(model_path):
                    config = model_configs.get(model_name, {})
                    models[model_name] = load_model_from_checkpoint(
                        model_path, config, self.config.target_device
                    )
            
            # Profile each batch size
            for batch_size in [1, 4, 8, 16]:  # Conservative batch sizes for memory analysis
                logger.info(f"Memory profiling with batch size {batch_size}...")
                
                batch_results = {}
                
                for model_name, model in models.items():
                    try:
                        # Inference memory profiling
                        inference_profile = profiler.profile_inference_memory(
                            model, batch_size, self.config.sequence_length, model_name
                        )
                        
                        # Training memory profiling
                        training_profile = profiler.profile_training_memory(
                            model, batch_size, self.config.sequence_length, 
                            optim_class=torch.optim.AdamW, model_name=model_name
                        )
                        
                        batch_results[model_name] = {
                            "inference": asdict(inference_profile),
                            "training": asdict(training_profile)
                        }
                        
                    except torch.cuda.OutOfMemoryError:
                        logger.warning(f"OOM for {model_name} at batch size {batch_size}")
                        batch_results[model_name] = {"error": "OutOfMemoryError"}
                    except Exception as e:
                        logger.error(f"Memory profiling error for {model_name}: {e}")
                        batch_results[model_name] = {"error": str(e)}
                
                memory_results[f"batch_size_{batch_size}"] = batch_results
            
            # Comparative analysis
            comparison_results = profiler.compare_model_memory_efficiency(
                models, batch_size=4, sequence_length=self.config.sequence_length
            )
            memory_results["comparative_analysis"] = comparison_results
            
            logger.info("✅ Memory profiling completed")
            
        except Exception as e:
            logger.error(f"Memory profiling failed: {e}")
            memory_results = {"error": str(e)}
        
        return memory_results
    
    def generate_comprehensive_analysis(
        self,
        glue_results: Dict[str, Any],
        hardware_results: Dict[str, Any],
        memory_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive analysis and visualizations."""
        logger.info("Generating comprehensive analysis...")
        
        try:
            # Create results analyzer
            analyzer = ResultsAnalyzer(
                results_dir=self.config.output_dir,
                output_dir=os.path.join(self.config.output_dir, "analysis")
            )
            
            # Combine all results
            combined_results = {
                "glue_evaluation": glue_results,
                "hardware_profiling": hardware_results,
                "memory_profiling": memory_results,
                "validation_config": asdict(self.config)
            }
            
            # Generate analysis
            analysis_results = analyzer.generate_comprehensive_analysis(combined_results)
            
            # Generate plots if enabled
            if self.config.generate_plots:
                plot_results = analyzer.generate_publication_plots(combined_results)
                analysis_results["plots"] = plot_results
            
            logger.info("✅ Comprehensive analysis completed")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Analysis generation failed: {e}")
            return {"error": str(e)}
    
    def run_full_validation(
        self,
        model_paths: Dict[str, str],
        model_configs: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run complete full-scale validation pipeline."""
        logger.info("Starting full-scale validation pipeline...")
        start_time = time.time()
        
        # Validate environment
        if not self.validate_environment():
            return {"error": "Environment validation failed"}
        
        # Prepare datasets
        dataset_info = self.prepare_full_scale_datasets()
        
        # Run GLUE evaluation
        glue_results = self.run_comprehensive_glue_evaluation(model_paths, model_configs)
        
        # Run hardware profiling
        hardware_results = self.run_hardware_profiling(model_paths, model_configs)
        
        # Run memory profiling
        memory_results = self.run_memory_profiling(model_paths, model_configs)
        
        # Generate comprehensive analysis
        analysis_results = self.generate_comprehensive_analysis(
            glue_results, hardware_results, memory_results
        )
        
        # Compile final results
        total_time = time.time() - start_time
        
        final_results = {
            "validation_summary": {
                "total_time_seconds": total_time,
                "models_evaluated": list(model_paths.keys()),
                "glue_tasks_evaluated": self.config.glue_tasks,
                "num_seeds": self.config.num_seeds,
                "confidence_level": self.config.confidence_level,
                "hardware_profiling_enabled": self.config.enable_hardware_profiling,
                "memory_profiling_enabled": self.config.enable_memory_profiling
            },
            "dataset_preparation": dataset_info,
            "glue_evaluation": glue_results,
            "hardware_profiling": hardware_results,
            "memory_profiling": memory_results,
            "comprehensive_analysis": analysis_results,
            "configuration": asdict(self.config)
        }
        
        # Save final results
        final_output_file = os.path.join(self.config.output_dir, "full_scale_validation_results.json")
        with open(final_output_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        logger.info(f"✅ Full-scale validation completed in {total_time:.2f} seconds")
        logger.info(f"📊 Final results saved to: {final_output_file}")
        
        return final_results


def load_model_configs(config_dir: str, model_sizes: List[str]) -> Dict[str, Dict[str, Any]]:
    """Load model configurations for different sizes."""
    configs = {}
    
    for size in model_sizes:
        config_file = os.path.join(config_dir, f"mamba_{size}.yaml")
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            configs[size] = config.get('model', {})
        else:
            logger.warning(f"Config file not found: {config_file}")
            # Default config
            configs[size] = {
                'd_model': 768 if size == "130m" else 1024,
                'n_layer': 12 if size == "130m" else 24,
                'vocab_size': 50257,
                'd_state': 16,
                'd_conv': 4
            }
    
    return configs


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Full-Scale Production-Ready Validation Suite"
    )
    
    # Model configurations
    parser.add_argument("--model_sizes", type=str, nargs="+", 
                       default=["130m"],
                       help="Model sizes to validate")
    parser.add_argument("--model_variants", type=str, nargs="+",
                       default=["M_base", "M_CSP", "M_SDM", "M_full"],
                       help="Model variants to validate")
    parser.add_argument("--checkpoints_dir", type=str, required=True,
                       help="Directory containing model checkpoints")
    parser.add_argument("--configs_dir", type=str, default="configs",
                       help="Directory containing model configurations")
    
    # Dataset configurations
    parser.add_argument("--use_full_datasets", action="store_true", default=True,
                       help="Use full-scale datasets")
    parser.add_argument("--glue_tasks", type=str, nargs="+",
                       default=["sst2", "mrpc", "qnli", "mnli"],
                       help="GLUE tasks to evaluate")
    
    # Evaluation configurations
    parser.add_argument("--num_seeds", type=int, default=5,
                       help="Number of random seeds for statistical testing")
    parser.add_argument("--confidence_level", type=float, default=0.95,
                       help="Confidence level for statistical intervals")
    parser.add_argument("--sequence_length", type=int, default=1024,
                       help="Sequence length for evaluation")
    
    # Hardware profiling
    parser.add_argument("--enable_hardware_profiling", action="store_true", default=True,
                       help="Enable hardware profiling")
    parser.add_argument("--enable_memory_profiling", action="store_true", default=True,
                       help="Enable memory profiling")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Target device for profiling")
    
    # Output configurations
    parser.add_argument("--output_dir", type=str, 
                       default="full_scale_validation_results",
                       help="Output directory for results")
    parser.add_argument("--generate_plots", action="store_true", default=True,
                       help="Generate publication-ready plots")
    
    return parser.parse_args()


def main():
    """Main full-scale validation function."""
    args = parse_args()
    
    # Create validation configuration
    config = FullScaleValidationConfig(
        model_sizes=args.model_sizes,
        model_variants=args.model_variants,
        use_full_datasets=args.use_full_datasets,
        glue_tasks=args.glue_tasks,
        num_seeds=args.num_seeds,
        confidence_level=args.confidence_level,
        sequence_length=args.sequence_length,
        enable_hardware_profiling=args.enable_hardware_profiling,
        enable_memory_profiling=args.enable_memory_profiling,
        target_device=args.device,
        output_dir=args.output_dir,
        generate_plots=args.generate_plots
    )
    
    # Load model configurations
    model_configs_by_size = load_model_configs(args.configs_dir, args.model_sizes)
    
    # Build model paths
    model_paths = {}
    model_configs = {}
    
    for size in args.model_sizes:
        for variant in args.model_variants:
            model_name = f"{variant}_{size}"
            model_path = os.path.join(args.checkpoints_dir, size, variant.lower(), "model.pt")
            
            if os.path.exists(model_path):
                model_paths[model_name] = model_path
                model_configs[model_name] = model_configs_by_size.get(size, {})
            else:
                logger.warning(f"Model checkpoint not found: {model_path}")
    
    if not model_paths:
        logger.error("No valid model checkpoints found")
        return 1
    
    logger.info(f"Found {len(model_paths)} model checkpoints to validate")
    
    try:
        # Create validator
        validator = FullScaleValidator(config)
        
        # Run full validation
        results = validator.run_full_validation(model_paths, model_configs)
        
        if "error" in results:
            logger.error(f"Validation failed: {results['error']}")
            return 1
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("FULL-SCALE VALIDATION SUMMARY")
        logger.info("="*80)
        
        summary = results["validation_summary"]
        logger.info(f"Total validation time: {summary['total_time_seconds']:.2f} seconds")
        logger.info(f"Models evaluated: {len(summary['models_evaluated'])}")
        logger.info(f"GLUE tasks evaluated: {len(summary['glue_tasks_evaluated'])}")
        logger.info(f"Statistical robustness: {summary['num_seeds']} seeds, {summary['confidence_level']*100:.0f}% CI")
        
        # Print key findings
        if "comprehensive_analysis" in results and "key_findings" in results["comprehensive_analysis"]:
            findings = results["comprehensive_analysis"]["key_findings"]
            logger.info("\nKey Findings:")
            for finding in findings:
                logger.info(f"  • {finding}")
        
        logger.info(f"\n📊 Complete results available in: {config.output_dir}")
        logger.info("🎉 Full-scale validation completed successfully!")
        
        return 0
        
    except Exception as e:
        logger.error(f"Full-scale validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 