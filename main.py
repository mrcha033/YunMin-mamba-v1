"""
Main Training Script for Hardware-Data-Parameter Co-Design Framework
GPU-Efficient Single-Run Execution for All Model Variants

This script orchestrates the complete training pipeline:
1. M_base: Baseline pre-training
2. M_csp: CSP analysis and permutation  
3. M_sdm: SDM pre-training with sparsity learning
4. M_sgh: SGH-PEFT fine-tuning
5. M_full: Complete pipeline integration
6. M_challenge: Challenge baseline for comparison

Usage:
    python main.py --config hparams.yaml --experiment_name full_pipeline
    python main.py --config hparams.yaml --stage sdm_only --model_size 130m
"""

import os
import sys
import argparse
import yaml
import time
import torch
import torch.distributed as dist
from pathlib import Path
from typing import Dict, Any, Optional
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.baseline_ssm import BaselineSSM
from models.sdm_ssm import SDM_SSM
from models.sgh_peft import create_sgh_peft_model, SGHPEFTConfig
from data.wikitext103 import get_wikitext103_dataloader
from data.glue import get_glue_dataloader
from utils.logger import setup_logger, setup_wandb
from utils.profiling import count_parameters, measure_latency

# Import training modules
from pretrain import main as pretrain_baseline
from pretrain_sdm import main as pretrain_sdm
from scripts.run_csp_analysis import main as run_csp
from scripts.run_finetuning import main as run_finetuning
from scripts.run_validation_suite import ValidationSuite


class CoDesignPipeline:
    """
    Orchestrates the complete hardware-data-parameter co-design pipeline.
    
    Designed for GPU efficiency: minimizes model loading/unloading and 
    maximizes utilization through intelligent scheduling.
    """
    
    def __init__(self, config: Dict[str, Any], experiment_name: str):
        self.config = config
        self.experiment_name = experiment_name
        self.device = torch.device(config['system']['device'])
        
        # Setup directories
        self.base_dir = Path(config['system']['base_output_dir'])
        self.experiment_dir = self.base_dir / experiment_name
        self.checkpoints_dir = self.experiment_dir / "checkpoints"
        self.logs_dir = self.experiment_dir / "logs"
        self.results_dir = self.experiment_dir / "results"
        
        # Create directories
        for directory in [self.experiment_dir, self.checkpoints_dir, self.logs_dir, self.results_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = setup_logger(
            name="codesign_pipeline",
            log_file=str(self.logs_dir / "pipeline.log")
        )
        
        # Initialize W&B if configured
        if config['logging'].get('use_wandb', False):
            setup_wandb(
                config=config,
                project=config['logging']['wandb_project'],
                run_name=experiment_name
            )
        
        # Model configurations
        self.model_size = config['experiment']['model_size']
        self.model_config = config['models'][self.model_size]
        
        # Training phases to execute
        self.phases = config['experiment']['phases']
        
        self.logger.info(f"🚀 Initialized Co-Design Pipeline: {experiment_name}")
        self.logger.info(f"📊 Model size: {self.model_size}")
        self.logger.info(f"🔄 Phases: {', '.join(self.phases)}")

    def run_complete_pipeline(self) -> Dict[str, Any]:
        """
        Execute the complete co-design pipeline efficiently.
        
        Returns:
            Dictionary with results from all phases
        """
        results = {
            'experiment_name': self.experiment_name,
            'model_size': self.model_size,
            'start_time': time.time(),
            'phases': {}
        }
        
        try:
            # Phase 1: Baseline Pre-training
            if 'baseline' in self.phases:
                self.logger.info("\n" + "="*70)
                self.logger.info("PHASE 1: BASELINE PRE-TRAINING (M_base)")
                self.logger.info("="*70)
                
                baseline_results = self.run_baseline_pretraining()
                results['phases']['baseline'] = baseline_results
                
                # GPU memory cleanup
                torch.cuda.empty_cache()
            
            # Phase 2: CSP Analysis  
            if 'csp' in self.phases:
                self.logger.info("\n" + "="*70)
                self.logger.info("PHASE 2: CSP ANALYSIS (M_csp)")
                self.logger.info("="*70)
                
                csp_results = self.run_csp_analysis()
                results['phases']['csp'] = csp_results
                
                torch.cuda.empty_cache()
            
            # Phase 3: SDM Pre-training
            if 'sdm' in self.phases:
                self.logger.info("\n" + "="*70)
                self.logger.info("PHASE 3: SDM PRE-TRAINING (M_sdm)")
                self.logger.info("="*70)
                
                sdm_results = self.run_sdm_pretraining()
                results['phases']['sdm'] = sdm_results
                
                torch.cuda.empty_cache()
            
            # Phase 4: SGH-PEFT Fine-tuning
            if 'sgh_peft' in self.phases:
                self.logger.info("\n" + "="*70)
                self.logger.info("PHASE 4: SGH-PEFT FINE-TUNING")
                self.logger.info("="*70)
                
                sgh_results = self.run_sgh_peft_finetuning()
                results['phases']['sgh_peft'] = sgh_results
                
                torch.cuda.empty_cache()
            
            # Phase 5: Challenge Baseline
            if 'challenge' in self.phases:
                self.logger.info("\n" + "="*70)
                self.logger.info("PHASE 5: CHALLENGE BASELINE (M_challenge)")
                self.logger.info("="*70)
                
                challenge_results = self.run_challenge_baseline()
                results['phases']['challenge'] = challenge_results
                
                torch.cuda.empty_cache()
            
            # Phase 6: Complete Integration (M_full)
            if 'full' in self.phases:
                self.logger.info("\n" + "="*70)
                self.logger.info("PHASE 6: FULL INTEGRATION (M_full)")
                self.logger.info("="*70)
                
                full_results = self.run_full_integration()
                results['phases']['full'] = full_results
                
                torch.cuda.empty_cache()
            
            # Phase 7: Comprehensive Validation
            if 'validation' in self.phases:
                self.logger.info("\n" + "="*70)
                self.logger.info("PHASE 7: COMPREHENSIVE VALIDATION")
                self.logger.info("="*70)
                
                validation_results = self.run_comprehensive_validation()
                results['phases']['validation'] = validation_results
            
            results['status'] = 'completed'
            results['end_time'] = time.time()
            results['total_time'] = results['end_time'] - results['start_time']
            
            self.logger.info(f"\n✅ Pipeline completed successfully!")
            self.logger.info(f"⏱️ Total time: {results['total_time']:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline failed: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)
            raise
        
        # Save results
        self.save_results(results)
        return results
    
    def run_baseline_pretraining(self) -> Dict[str, Any]:
        """Run baseline pre-training (M_base)."""
        checkpoint_path = self.checkpoints_dir / "baseline"
        checkpoint_path.mkdir(exist_ok=True)
        
        # Configure baseline training
        baseline_config = self.model_config.copy()
        baseline_config['training'] = self.config['training']['baseline']
        baseline_config['output_dir'] = str(checkpoint_path)
        
        # Save temporary config
        temp_config_path = checkpoint_path / "config.yaml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(baseline_config, f)
        
        self.logger.info("🔄 Starting baseline pre-training...")
        
        try:
            # Use existing pretrain.py logic
            from pretrain import train_baseline_model
            
            # Create model
            model = BaselineSSM(**baseline_config['model'])
            
            # Train model
            metrics = train_baseline_model(
                model=model,
                config=baseline_config,
                output_dir=str(checkpoint_path)
            )
            
            self.logger.info("✅ Baseline pre-training completed")
            return {
                'status': 'completed',
                'checkpoint_path': str(checkpoint_path / "model.pt"),
                'metrics': metrics
            }
            
        except Exception as e:
            self.logger.error(f"❌ Baseline pre-training failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def run_csp_analysis(self) -> Dict[str, Any]:
        """Run CSP analysis and apply permutation."""
        baseline_checkpoint = self.checkpoints_dir / "baseline" / "model.pt"
        csp_checkpoint_path = self.checkpoints_dir / "csp"
        csp_checkpoint_path.mkdir(exist_ok=True)
        
        self.logger.info("🔄 Running CSP analysis...")
        
        try:
            # Import CSP analysis
            from scripts.run_csp_analysis import run_csp_analysis
            
            csp_results = run_csp_analysis(
                model_path=str(baseline_checkpoint),
                output_path=str(csp_checkpoint_path / "model_csp.pt"),
                num_samples=self.config['csp']['num_samples'],
                device=str(self.device)
            )
            
            self.logger.info("✅ CSP analysis completed")
            return {
                'status': 'completed',
                'checkpoint_path': str(csp_checkpoint_path / "model_csp.pt"),
                'results': csp_results
            }
            
        except Exception as e:
            self.logger.error(f"❌ CSP analysis failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def run_sdm_pretraining(self) -> Dict[str, Any]:
        """Run SDM pre-training with sparsity learning."""
        csp_checkpoint = self.checkpoints_dir / "csp" / "model_csp.pt"
        sdm_checkpoint_path = self.checkpoints_dir / "sdm"
        sdm_checkpoint_path.mkdir(exist_ok=True)
        
        self.logger.info("🔄 Starting SDM pre-training...")
        
        try:
            from pretrain_sdm import train_sdm_model
            
            # Configure SDM training
            sdm_config = self.model_config.copy()
            sdm_config['training'] = self.config['training']['sdm']
            sdm_config['sdm'] = self.config['sdm']
            sdm_config['output_dir'] = str(sdm_checkpoint_path)
            
            # Train SDM model
            metrics = train_sdm_model(
                config=sdm_config,
                init_from=str(csp_checkpoint),
                output_dir=str(sdm_checkpoint_path)
            )
            
            self.logger.info("✅ SDM pre-training completed")
            return {
                'status': 'completed',
                'checkpoint_path': str(sdm_checkpoint_path / "model.pt"),
                'metrics': metrics
            }
            
        except Exception as e:
            self.logger.error(f"❌ SDM pre-training failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def run_sgh_peft_finetuning(self) -> Dict[str, Any]:
        """Run SGH-PEFT fine-tuning on GLUE tasks."""
        sdm_checkpoint = self.checkpoints_dir / "sdm" / "model.pt"
        sgh_checkpoint_path = self.checkpoints_dir / "sgh_peft"
        sgh_checkpoint_path.mkdir(exist_ok=True)
        
        self.logger.info("🔄 Starting SGH-PEFT fine-tuning...")
        
        try:
            from scripts.run_finetuning import run_sgh_peft_finetuning
            
            # Configure SGH-PEFT
            sgh_config = self.config['sgh_peft']
            
            # Fine-tune on specified GLUE tasks
            tasks = self.config['evaluation']['glue_tasks']
            results = {}
            
            for task in tasks:
                self.logger.info(f"  Fine-tuning on {task}...")
                
                task_results = run_sgh_peft_finetuning(
                    sdm_checkpoint_path=str(sdm_checkpoint),
                    task=task,
                    config=sgh_config,
                    output_dir=str(sgh_checkpoint_path / task)
                )
                
                results[task] = task_results
                
                # Save task-specific checkpoint
                torch.save(
                    task_results['model_state_dict'],
                    sgh_checkpoint_path / f"{task}_model.pt"
                )
            
            self.logger.info("✅ SGH-PEFT fine-tuning completed")
            return {
                'status': 'completed',
                'checkpoint_path': str(sgh_checkpoint_path),
                'task_results': results
            }
            
        except Exception as e:
            self.logger.error(f"❌ SGH-PEFT fine-tuning failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def run_challenge_baseline(self) -> Dict[str, Any]:
        """Run challenge baseline (magnitude pruning + uniform LoRA)."""
        baseline_checkpoint = self.checkpoints_dir / "baseline" / "model.pt"
        sdm_checkpoint = self.checkpoints_dir / "sdm" / "model.pt"
        challenge_checkpoint_path = self.checkpoints_dir / "challenge"
        challenge_checkpoint_path.mkdir(exist_ok=True)
        
        self.logger.info("🔄 Creating challenge baseline...")
        
        try:
            from scripts.run_challenge_baseline import create_challenge_baseline
            
            challenge_results = create_challenge_baseline(
                baseline_checkpoint=str(baseline_checkpoint),
                sdm_checkpoint=str(sdm_checkpoint),
                output_path=str(challenge_checkpoint_path / "model.pt"),
                config=self.config['challenge']
            )
            
            self.logger.info("✅ Challenge baseline created")
            return {
                'status': 'completed',
                'checkpoint_path': str(challenge_checkpoint_path / "model.pt"),
                'results': challenge_results
            }
            
        except Exception as e:
            self.logger.error(f"❌ Challenge baseline creation failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def run_full_integration(self) -> Dict[str, Any]:
        """Run complete integration (M_full)."""
        sdm_checkpoint = self.checkpoints_dir / "sdm" / "model.pt"
        full_checkpoint_path = self.checkpoints_dir / "full"
        full_checkpoint_path.mkdir(exist_ok=True)
        
        self.logger.info("🔄 Creating full integration model...")
        
        try:
            # Load SDM model
            sdm_model = torch.load(sdm_checkpoint, map_location='cpu')
            
            # Apply SGH-PEFT to create M_full
            sgh_config = SGHPEFTConfig(**self.config['sgh_peft'])
            full_model = create_sgh_peft_model(sdm_model, sgh_config)
            
            # Save M_full
            torch.save(
                full_model.state_dict(),
                full_checkpoint_path / "model.pt"
            )
            
            self.logger.info("✅ Full integration model created")
            return {
                'status': 'completed',
                'checkpoint_path': str(full_checkpoint_path / "model.pt")
            }
            
        except Exception as e:
            self.logger.error(f"❌ Full integration failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation on all model variants."""
        self.logger.info("🔄 Running comprehensive validation...")
        
        validation_results = {}
        validator = ValidationSuite(device=str(self.device))
        
        # Model variants to validate
        model_variants = {
            'M_base': self.checkpoints_dir / "baseline" / "model.pt",
            'M_csp': self.checkpoints_dir / "csp" / "model_csp.pt",
            'M_sdm': self.checkpoints_dir / "sdm" / "model.pt",
            'M_challenge': self.checkpoints_dir / "challenge" / "model.pt",
            'M_full': self.checkpoints_dir / "full" / "model.pt"
        }
        
        for variant_name, checkpoint_path in model_variants.items():
            if checkpoint_path.exists():
                self.logger.info(f"  Validating {variant_name}...")
                
                try:
                    results = validator.run_comprehensive_validation(
                        model_group=variant_name,
                        checkpoint_path=str(checkpoint_path),
                        config=self.model_config,
                        sdm_checkpoint=str(self.checkpoints_dir / "sdm" / "model.pt") if variant_name == 'M_challenge' else None
                    )
                    
                    validation_results[variant_name] = results
                    
                    # Save individual results
                    with open(self.results_dir / f"{variant_name}_validation.json", 'w') as f:
                        json.dump(results, f, indent=2)
                    
                except Exception as e:
                    self.logger.error(f"❌ Validation failed for {variant_name}: {e}")
                    validation_results[variant_name] = {'status': 'failed', 'error': str(e)}
            else:
                self.logger.warning(f"⚠️ Checkpoint not found for {variant_name}: {checkpoint_path}")
        
        self.logger.info("✅ Comprehensive validation completed")
        return validation_results
    
    def save_results(self, results: Dict[str, Any]):
        """Save final results to file."""
        results_file = self.results_dir / "pipeline_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"📁 Results saved to {results_file}")
        
        # Print summary
        self.print_summary(results)
    
    def print_summary(self, results: Dict[str, Any]):
        """Print pipeline execution summary."""
        self.logger.info("\n" + "="*70)
        self.logger.info("PIPELINE EXECUTION SUMMARY")
        self.logger.info("="*70)
        
        total_time = results.get('total_time', 0)
        self.logger.info(f"Experiment: {results['experiment_name']}")
        self.logger.info(f"Model Size: {results['model_size']}")
        self.logger.info(f"Total Time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
        self.logger.info(f"Status: {results['status']}")
        
        # Phase summary
        if 'phases' in results:
            self.logger.info("\nPhase Results:")
            for phase, result in results['phases'].items():
                status = result.get('status', 'unknown')
                self.logger.info(f"  {phase}: {status}")
        
        self.logger.info("="*70)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Hardware-Data-Parameter Co-Design Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="hparams.yaml",
        help="Path to hyperparameters configuration file"
    )
    
    parser.add_argument(
        "--experiment_name", 
        type=str, 
        default=None,
        help="Name of the experiment (auto-generated if not provided)"
    )
    
    parser.add_argument(
        "--stage",
        type=str,
        choices=['baseline', 'csp', 'sdm', 'sgh_peft', 'challenge', 'full', 'validation', 'all'],
        default='all',
        help="Specific stage to run (default: all)"
    )
    
    parser.add_argument(
        "--model_size",
        type=str,
        choices=['130m', '370m'],
        help="Override model size from config"
    )
    
    parser.add_argument(
        "--resume_from",
        type=str,
        help="Resume from checkpoint directory"
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run without actual training (for testing)"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return 1
    
    # Override config with command line arguments
    if args.model_size:
        config['experiment']['model_size'] = args.model_size
    
    if args.stage != 'all':
        config['experiment']['phases'] = [args.stage]
    
    # Generate experiment name if not provided
    if not args.experiment_name:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_size = config['experiment']['model_size']
        phases = "_".join(config['experiment']['phases'])
        args.experiment_name = f"codesign_{model_size}_{phases}_{timestamp}"
    
    # Initialize and run pipeline
    try:
        pipeline = CoDesignPipeline(config, args.experiment_name)
        
        if args.dry_run:
            pipeline.logger.info("🧪 Dry run mode - no actual training will be performed")
            return 0
        
        results = pipeline.run_complete_pipeline()
        
        if results['status'] == 'completed':
            pipeline.logger.info("🎉 Pipeline completed successfully!")
            return 0
        else:
            pipeline.logger.error("❌ Pipeline failed!")
            return 1
            
    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 