#!/usr/bin/env python3
"""
Generate Final Report Script

This script generates a comprehensive final report for the hardware-data-parameter 
co-design experiment by analyzing all results and creating a summary.
"""

import argparse
import json
import os
import yaml
from pathlib import Path
from typing import Dict, Any, List
import glob


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate final experiment report")
    parser.add_argument("--experiment_dir", type=str, required=True,
                       help="Path to experiment directory")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to configuration file")
    parser.add_argument("--output_file", type=str, required=True,
                       help="Output file path for the final report")
    return parser.parse_args()


def collect_validation_results(experiment_dir: str) -> Dict[str, Any]:
    """Collect validation results from the experiment directory."""
    results_dir = Path(experiment_dir) / "results"
    validation_results = {}
    
    # Look for validation JSON files
    for result_file in results_dir.glob("*_validation.json"):
        model_name = result_file.stem.replace("_validation", "")
        try:
            with open(result_file, 'r') as f:
                validation_results[model_name] = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load {result_file}: {e}")
    
    return validation_results


def collect_checkpoint_info(experiment_dir: str) -> Dict[str, Any]:
    """Collect information about generated checkpoints."""
    checkpoints_dir = Path(experiment_dir) / "checkpoints"
    checkpoint_info = {}
    
    for checkpoint_dir in checkpoints_dir.iterdir():
        if checkpoint_dir.is_dir():
            model_name = checkpoint_dir.name
            model_files = list(checkpoint_dir.glob("*.pt"))
            
            checkpoint_info[model_name] = {
                "path": str(checkpoint_dir),
                "model_files": [str(f) for f in model_files],
                "file_count": len(model_files)
            }
    
    return checkpoint_info


def analyze_experiment_status(experiment_dir: str) -> Dict[str, Any]:
    """Analyze the overall experiment status."""
    experiment_path = Path(experiment_dir)
    
    status = {
        "experiment_completed": False,
        "phases_completed": [],
        "missing_components": [],
        "total_files": 0,
        "total_size_mb": 0
    }
    
    # Check for key directories and files
    expected_dirs = ["checkpoints", "results", "logs"]
    for dir_name in expected_dirs:
        dir_path = experiment_path / dir_name
        if dir_path.exists():
            status["phases_completed"].append(dir_name)
        else:
            status["missing_components"].append(dir_name)
    
    # Calculate total size
    try:
        total_size = sum(f.stat().st_size for f in experiment_path.rglob('*') if f.is_file())
        status["total_size_mb"] = round(total_size / (1024 * 1024), 2)
        status["total_files"] = len(list(experiment_path.rglob('*')))
    except Exception as e:
        print(f"Warning: Could not calculate experiment size: {e}")
    
    # Check if experiment appears complete
    required_components = ["checkpoints", "results"]
    status["experiment_completed"] = all(comp in status["phases_completed"] for comp in required_components)
    
    return status


def generate_performance_summary(validation_results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a performance summary from validation results."""
    summary = {
        "models_evaluated": len(validation_results),
        "best_model": None,
        "best_score": -float('inf'),
        "performance_comparison": {}
    }
    
    for model_name, results in validation_results.items():
        if isinstance(results, dict):
            # Extract main performance metric (assume accuracy or similar)
            score = 0.0
            if 'accuracy' in results:
                score = results['accuracy']
            elif 'score' in results:
                score = results['score']
            elif 'glue_average' in results:
                score = results['glue_average']
            
            summary["performance_comparison"][model_name] = {
                "score": score,
                "metrics": results
            }
            
            if score > summary["best_score"]:
                summary["best_score"] = score
                summary["best_model"] = model_name
    
    return summary


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration if provided."""
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load config {config_path}: {e}")
    
    return {}


def generate_final_report(experiment_dir: str, config_path: str = None) -> Dict[str, Any]:
    """Generate the complete final report."""
    
    # Collect all information
    validation_results = collect_validation_results(experiment_dir)
    checkpoint_info = collect_checkpoint_info(experiment_dir)
    experiment_status = analyze_experiment_status(experiment_dir)
    performance_summary = generate_performance_summary(validation_results)
    config = load_config(config_path)
    
    # Generate timestamp
    from datetime import datetime
    timestamp = datetime.now().isoformat()
    
    # Create final report
    final_report = {
        "experiment_info": {
            "experiment_dir": experiment_dir,
            "config_path": config_path,
            "generated_at": timestamp,
            "framework": "Hardware-Data-Parameter Co-Design for State Space Models"
        },
        "experiment_status": experiment_status,
        "configuration": config,
        "checkpoints": checkpoint_info,
        "validation_results": validation_results,
        "performance_summary": performance_summary,
        "conclusions": {
            "experiment_successful": experiment_status["experiment_completed"],
            "models_generated": len(checkpoint_info),
            "models_validated": len(validation_results),
            "best_performing_model": performance_summary.get("best_model", "N/A"),
            "recommendations": []
        }
    }
    
    # Add recommendations based on results
    if performance_summary["best_model"]:
        final_report["conclusions"]["recommendations"].append(
            f"Use {performance_summary['best_model']} as the optimal model for deployment"
        )
    
    if experiment_status["experiment_completed"]:
        final_report["conclusions"]["recommendations"].append(
            "All pipeline stages completed successfully - ready for production"
        )
    else:
        final_report["conclusions"]["recommendations"].append(
            f"Complete missing components: {', '.join(experiment_status['missing_components'])}"
        )
    
    return final_report


def main():
    """Main function."""
    args = parse_args()
    
    print("Generating final experiment report...")
    print(f"Experiment directory: {args.experiment_dir}")
    print(f"Output file: {args.output_file}")
    
    try:
        # Generate the report
        final_report = generate_final_report(args.experiment_dir, args.config)
        
        # Save the report
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, 'w') as f:
            json.dump(final_report, f, indent=4)
        
        print(f"✅ Final report generated successfully: {args.output_file}")
        
        # Print summary
        print("\nExperiment Summary:")
        print(f"  Status: {'✅ Complete' if final_report['experiment_status']['experiment_completed'] else '⚠️ Incomplete'}")
        print(f"  Models Generated: {final_report['conclusions']['models_generated']}")
        print(f"  Models Validated: {final_report['conclusions']['models_validated']}")
        print(f"  Best Model: {final_report['conclusions']['best_performing_model']}")
        print(f"  Total Size: {final_report['experiment_status']['total_size_mb']} MB")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error generating final report: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 