#!/usr/bin/env python3
"""
Experimental Setup Validation Script

This script automatically verifies that all hyperparameters and configurations
match the experimental description provided in the paper.

Usage:
    python scripts/validate_experimental_setup.py [--verbose] [--fix-issues]
"""

import argparse
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class ExperimentalSetupValidator:
    """Validates experimental setup against paper specifications."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.issues_found = []
        self.checks_passed = []
        
        # Expected configurations from experimental description
        self.expected_config = {
            "phase_a": {
                "optimizer": "AdamW",
                "learning_rate": 2e-4,
                "batch_size": 128,
                "epochs": 20,
                "warmup_steps_ratio": 0.1,
                "dataset": "WikiText-103"
            },
            "phase_b": {
                "optimizer": "AdamW", 
                "learning_rate": 1e-4,
                "batch_size": 32,
                "epochs": {
                    "sst2": 5,
                    "mnli": 10,
                    "qnli": 5,  # Not specified in description, using reasonable default
                    "mrpc": 8   # Not specified in description, using reasonable default
                },
                "early_stopping": True
            },
            "model_variants": [
                "M_base", "M_csp", "M_sdm", "M_sgh", 
                "M_sdm+sgh", "M_full", "M_challenge"
            ],
            "datasets": {
                "phase_a": ["WikiText-103"],
                "phase_b": ["SST-2", "MRPC", "QNLI", "MNLI"]
            },
            "hardware": {
                "gpu": "NVIDIA A100",
                "memory": "80GB",
                "cuda_version": "12.1",
                "pytorch_version": "2.1"
            },
            "metrics": {
                "performance": ["perplexity", "accuracy", "f1"],
                "efficiency": ["latency", "throughput", "flops", "trainable_parameters"]
            }
        }
    
    def log_issue(self, category: str, message: str, severity: str = "ERROR"):
        """Log validation issue."""
        issue = {
            "category": category,
            "message": message,
            "severity": severity
        }
        self.issues_found.append(issue)
        
        if severity == "ERROR":
            logger.error(f"[{category}] {message}")
        elif severity == "WARNING":
            logger.warning(f"[{category}] {message}")
        else:
            logger.info(f"[{category}] {message}")
    
    def log_success(self, category: str, message: str):
        """Log successful check."""
        self.checks_passed.append(f"[{category}] {message}")
        logger.info(f"✅ [{category}] {message}")
    
    def validate_config_file(self, config_path: Path, expected_values: Dict[str, Any], context: str) -> bool:
        """Validate a YAML configuration file."""
        if not config_path.exists():
            self.log_issue("CONFIG", f"Missing config file: {config_path}")
            return False
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            self.log_issue("CONFIG", f"Failed to load {config_path}: {e}")
            return False
        
        all_good = True
        
        # Check nested values
        for key_path, expected_value in expected_values.items():
            keys = key_path.split('.')
            current = config
            
            try:
                for key in keys:
                    current = current[key] 
                
                if current != expected_value:
                    self.log_issue(
                        "CONFIG",
                        f"{context} - {key_path}: expected {expected_value}, got {current}",
                        "ERROR"
                    )
                    all_good = False
                else:
                    self.log_success("CONFIG", f"{context} - {key_path}: ✓ {current}")
                    
            except (KeyError, TypeError):
                self.log_issue(
                    "CONFIG", 
                    f"{context} - Missing key: {key_path}",
                    "ERROR"
                )
                all_good = False
        
        return all_good
    
    def validate_phase_a_hyperparameters(self) -> bool:
        """Validate Phase A (pre-training) hyperparameters."""
        logger.info("🔍 Validating Phase A hyperparameters...")
        
        # Check SDM pre-training config
        sdm_config_path = self.project_root / "configs" / "pretrain_sdm.yaml"
        expected_sdm = {
            "training.pretrain.learning_rate": 2e-4,
            "training.pretrain.batch_size": 128,
            "training.pretrain.max_epochs": 20,
            "training.pretrain.warmup_steps_ratio": 0.1
        }
        
        result1 = self.validate_config_file(sdm_config_path, expected_sdm, "SDM Pre-training")
        
        # Check model configs  
        model_configs = ["mamba_130m.yaml", "mamba_370m.yaml"]
        results = [result1]
        
        for model_config in model_configs:
            config_path = self.project_root / "configs" / model_config
            expected_model = {
                "training.pretrain.learning_rate": 2e-4,
                "training.pretrain.batch_size": 128,
                "training.pretrain.max_epochs": 20,
                "training.pretrain.warmup_steps_ratio": 0.1
            }
            
            result = self.validate_config_file(config_path, expected_model, f"Model Config ({model_config})")
            results.append(result)
        
        return all(results)
    
    def validate_phase_b_hyperparameters(self) -> bool:
        """Validate Phase B (fine-tuning) hyperparameters."""
        logger.info("🔍 Validating Phase B hyperparameters...")
        
        # Check fine-tuning config
        finetune_config_path = self.project_root / "configs" / "finetune_glue.yaml"
        expected_finetune = {
            "training.finetune.learning_rate": 1e-4,
            "training.finetune.batch_size": 32
        }
        
        result1 = self.validate_config_file(finetune_config_path, expected_finetune, "GLUE Fine-tuning")
        
        # Check task-specific epochs in model configs
        model_configs = ["mamba_130m.yaml", "mamba_370m.yaml"]
        results = [result1]
        
        for model_config in model_configs:
            config_path = self.project_root / "configs" / model_config
            expected_epochs = {
                "training.finetune.learning_rate": 1e-4,
                "training.finetune.batch_size": 32,
                "training.finetune.epochs.sst2": 5,
                "training.finetune.epochs.mnli": 10
            }
            
            result = self.validate_config_file(config_path, expected_epochs, f"Task Epochs ({model_config})")
            results.append(result)
        
        return all(results)
    
    def validate_model_variants(self) -> bool:
        """Validate all model variants are implemented."""
        logger.info("🔍 Validating model variants implementation...")
        
        all_good = True
        
        # Check experiment script defines all variants
        experiment_script = self.project_root / "run_full_experiment.sh"
        if not experiment_script.exists():
            self.log_issue("MODELS", "Missing main experiment script: run_full_experiment.sh")
            all_good = False
        else:
            # Read script and check for model variants
            with open(experiment_script, 'r') as f:
                script_content = f.read()
            
            for variant in self.expected_config["model_variants"]:
                if variant not in script_content:
                    self.log_issue("MODELS", f"Model variant {variant} not found in experiment script")
                    all_good = False
                else:
                    self.log_success("MODELS", f"Model variant {variant} found in experiment script")
        
        # Check validation suite supports all variants
        validation_script = self.project_root / "scripts" / "run_validation_suite.py"
        if validation_script.exists():
            with open(validation_script, 'r') as f:
                script_content = f.read()
            
            for variant in self.expected_config["model_variants"]:
                if variant not in script_content:
                    self.log_issue("MODELS", f"Model variant {variant} not supported in validation suite", "WARNING")
                else:
                    self.log_success("MODELS", f"Model variant {variant} supported in validation suite")
        
        return all_good
    
    def validate_datasets(self) -> bool:
        """Validate dataset support."""
        logger.info("🔍 Validating dataset support...")
        
        all_good = True
        
        # Check WikiText-103 support
        wikitext_module = self.project_root / "data" / "wikitext103.py"
        if not wikitext_module.exists():
            self.log_issue("DATASETS", "Missing WikiText-103 data module")
            all_good = False
        else:
            self.log_success("DATASETS", "WikiText-103 data module found")
        
        # Check GLUE support
        glue_module = self.project_root / "data" / "glue.py"
        if not glue_module.exists():
            self.log_issue("DATASETS", "Missing GLUE data module")
            all_good = False
        else:
            self.log_success("DATASETS", "GLUE data module found")
            
            # Check GLUE tasks
            with open(glue_module, 'r') as f:
                glue_content = f.read()
            
            required_tasks = ["sst2", "mrpc", "qnli", "mnli"]
            for task in required_tasks:
                if task not in glue_content.lower():
                    self.log_issue("DATASETS", f"GLUE task {task} not found in GLUE module", "WARNING")
                else:
                    self.log_success("DATASETS", f"GLUE task {task.upper()} supported")
        
        return all_good
    
    def validate_evaluation_metrics(self) -> bool:
        """Validate evaluation metrics implementation."""
        logger.info("🔍 Validating evaluation metrics...")
        
        all_good = True
        
        # Check latency evaluation
        latency_script = self.project_root / "scripts" / "evaluate_latency.py"
        if not latency_script.exists():
            self.log_issue("METRICS", "Missing latency evaluation script")
            all_good = False
        else:
            self.log_success("METRICS", "Latency evaluation script found")
            
            # Check for A100 profiling
            with open(latency_script, 'r') as f:
                content = f.read()
            
            if "A100" in content:
                self.log_success("METRICS", "A100-specific profiling found")
            else:
                self.log_issue("METRICS", "A100-specific profiling not found", "WARNING")
        
        # Check GLUE evaluation
        glue_eval_script = self.project_root / "scripts" / "evaluate_glue.py"
        if not glue_eval_script.exists():
            self.log_issue("METRICS", "Missing GLUE evaluation script")
            all_good = False
        else:
            self.log_success("METRICS", "GLUE evaluation script found")
        
        # Check profiling utilities
        profiling_utils = self.project_root / "utils" / "profiling.py"
        if not profiling_utils.exists():
            self.log_issue("METRICS", "Missing profiling utilities")
            all_good = False
        else:
            self.log_success("METRICS", "Profiling utilities found")
        
        return all_good
    
    def validate_iso_sparsity_implementation(self) -> bool:
        """Validate iso-sparsity implementation for fair comparison."""
        logger.info("🔍 Validating iso-sparsity implementation...")
        
        all_good = True
        
        # Check challenge baseline script
        challenge_script = self.project_root / "scripts" / "run_challenge_baseline.py"
        if not challenge_script.exists():
            self.log_issue("SPARSITY", "Missing challenge baseline script")
            all_good = False
        else:
            with open(challenge_script, 'r') as f:
                content = f.read()
            
            # Check for sparsity verification
            if "sparsity_ratio" in content and "sdm_checkpoint" in content:
                self.log_success("SPARSITY", "Iso-sparsity implementation found")
            else:
                self.log_issue("SPARSITY", "Iso-sparsity implementation incomplete", "WARNING")
            
            # Check for verification logging
            if "DETECTED M_SDM SPARSITY" in content:
                self.log_success("SPARSITY", "Sparsity verification logging found")
            else:
                self.log_issue("SPARSITY", "Sparsity verification logging missing", "WARNING")
        
        return all_good
    
    def test_basic_functionality(self) -> bool:
        """Test basic functionality of key components."""
        logger.info("🔍 Testing basic functionality...")
        
        all_good = True
        
        try:
            # Test dataset loading
            sys.path.insert(0, str(self.project_root))
            
            # Test WikiText-103
            try:
                from data.wikitext103 import verify_dataset_loading
                if verify_dataset_loading():
                    self.log_success("FUNCTIONALITY", "WikiText-103 dataset loading works")
                else:
                    self.log_issue("FUNCTIONALITY", "WikiText-103 dataset loading failed", "WARNING")
            except Exception as e:
                self.log_issue("FUNCTIONALITY", f"WikiText-103 test failed: {e}", "WARNING")
            
            # Test GLUE loading
            try:
                from data.glue import verify_glue_loading
                if verify_glue_loading():
                    self.log_success("FUNCTIONALITY", "GLUE dataset loading works")
                else:
                    self.log_issue("FUNCTIONALITY", "GLUE dataset loading failed", "WARNING")
            except Exception as e:
                self.log_issue("FUNCTIONALITY", f"GLUE test failed: {e}", "WARNING")
                
        except Exception as e:
            self.log_issue("FUNCTIONALITY", f"Basic functionality test failed: {e}")
            all_good = False
        
        return all_good
    
    def run_full_validation(self) -> bool:
        """Run complete validation suite."""
        logger.info("🚀 Starting Experimental Setup Validation")
        logger.info("=" * 60)
        
        validation_results = []
        
        # Run all validation checks
        validation_results.append(self.validate_phase_a_hyperparameters())
        validation_results.append(self.validate_phase_b_hyperparameters())
        validation_results.append(self.validate_model_variants())
        validation_results.append(self.validate_datasets())
        validation_results.append(self.validate_evaluation_metrics())
        validation_results.append(self.validate_iso_sparsity_implementation())
        validation_results.append(self.test_basic_functionality())
        
        # Print summary
        logger.info("=" * 60)
        logger.info("🏁 VALIDATION SUMMARY")
        logger.info("=" * 60)
        
        logger.info(f"✅ Checks passed: {len(self.checks_passed)}")
        
        if self.issues_found:
            logger.info(f"❌ Issues found: {len(self.issues_found)}")
            
            # Group issues by severity
            errors = [i for i in self.issues_found if i["severity"] == "ERROR"]
            warnings = [i for i in self.issues_found if i["severity"] == "WARNING"]
            
            if errors:
                logger.error(f"🚨 CRITICAL ERRORS ({len(errors)}):")
                for error in errors:
                    logger.error(f"  • [{error['category']}] {error['message']}")
            
            if warnings:
                logger.warning(f"⚠️  WARNINGS ({len(warnings)}):")
                for warning in warnings:
                    logger.warning(f"  • [{warning['category']}] {warning['message']}")
        else:
            logger.info("🎉 No issues found!")
        
        all_passed = all(validation_results) and len([i for i in self.issues_found if i["severity"] == "ERROR"]) == 0
        
        if all_passed:
            logger.info("✅ EXPERIMENTAL SETUP VALIDATION: PASSED")
            logger.info("   Your setup matches the experimental description!")
        else:
            logger.error("❌ EXPERIMENTAL SETUP VALIDATION: FAILED")
            logger.error("   Please fix the issues above before running experiments.")
        
        return all_passed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate experimental setup against paper specifications"
    )
    
    parser.add_argument("--project-root", type=str, default=".",
                       help="Path to project root directory")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--fix-issues", action="store_true",
                       help="Automatically fix issues where possible")
    
    return parser.parse_args()


def main():
    """Main validation function."""
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run validation
    validator = ExperimentalSetupValidator(args.project_root)
    success = validator.run_full_validation()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 