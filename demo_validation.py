"""
Demonstration Script for Complete Co-Design Framework Validation

This script demonstrates the complete validation framework using simulated models
and results to show how the hypothesis validation works.

Usage:
    python demo_validation.py
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.baseline_ssm import BaselineSSM
from models.sdm_ssm import SDM_SSM
from models.sgh_peft import create_sgh_peft_model, SGHPEFTConfig


def create_demo_models():
    """Create demonstration models for validation."""
    print("Creating demonstration models...")
    
    demo_dir = Path("demo_validation_results")
    demo_dir.mkdir(exist_ok=True)
    
    models_dir = demo_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Model configuration
    config = {
        'd_model': 256,
        'n_layer': 6,
        'vocab_size': 1000,
        'd_state': 8,
        'd_conv': 4
    }
    
    models = {}
    
    # 1. M_base: Baseline model
    print("Creating M_base...")
    base_model = BaselineSSM(**config)
    base_path = models_dir / "M_base.pt"
    torch.save({'model_state_dict': base_model.state_dict(), 'config': config}, base_path)
    models['M_base'] = str(base_path)
    
    # 2. M_csp: Base model with CSP (simulated)
    print("Creating M_csp...")
    csp_model = BaselineSSM(**config)
    csp_path = models_dir / "M_csp.pt"
    torch.save({'model_state_dict': csp_model.state_dict(), 'config': config, 'csp_applied': True}, csp_path)
    models['M_csp'] = str(csp_path)
    
    # 3. M_sdm: SDM model with learned sparsity
    print("Creating M_sdm...")
    sdm_model = SDM_SSM(**config, gumbel_temp=1.0)
    
    # Simulate learned sparsity patterns
    with torch.no_grad():
        for layer_idx, layer in enumerate(sdm_model.layers):
            # Create realistic sparsity: 20-40% depending on layer depth
            sparsity_ratio = 0.2 + (layer_idx / 6) * 0.2
            num_channels = layer.d_inner
            num_sparse = int(num_channels * sparsity_ratio)
            
            # Set z_logits to create structured sparsity
            layer.z_logits.data.fill_(1.0)  # Most channels active
            sparse_indices = torch.randperm(num_channels)[:num_sparse]
            layer.z_logits.data[sparse_indices] = -2.0  # Some channels inactive
    
    sdm_path = models_dir / "M_sdm.pt"
    torch.save({'model_state_dict': sdm_model.state_dict(), 'config': config, 'sdm_applied': True}, sdm_path)
    models['M_sdm'] = str(sdm_path)
    
    # 4. M_sgh: SGH-PEFT with proxy importance
    print("Creating M_sgh...")
    sgh_model = BaselineSSM(**config)
    sgh_path = models_dir / "M_sgh.pt"
    torch.save({'model_state_dict': sgh_model.state_dict(), 'config': config, 'sgh_proxy': True}, sgh_path)
    models['M_sgh'] = str(sgh_path)
    
    # 5. M_challenge: Magnitude pruning + uniform LoRA
    print("Creating M_challenge...")
    challenge_model = BaselineSSM(**config)
    challenge_path = models_dir / "M_challenge.pt"
    torch.save({'model_state_dict': challenge_model.state_dict(), 'config': config, 'challenge': True}, challenge_path)
    models['M_challenge'] = str(challenge_path)
    
    # 6. M_full: Complete co-design
    print("Creating M_full...")
    sgh_peft_config = SGHPEFTConfig(
        lora_high_rank=8, lora_low_rank=2,
        high_importance_mean_threshold=0.5,
        high_importance_active_threshold=60.0,
        medium_importance_mean_threshold=0.0,
        medium_importance_active_threshold=40.0,
        low_importance_mean_threshold=-0.5
    )
    
    full_model = create_sgh_peft_model(sdm_model, sgh_peft_config)
    full_path = models_dir / "M_full.pt"
    torch.save({
        'model_state_dict': full_model.state_dict(), 
        'config': config, 
        'full_pipeline': True,
        'adaptation_summary': full_model.get_adaptation_summary()
    }, full_path)
    models['M_full'] = str(full_path)
    
    print(f"✓ All demonstration models created in {models_dir}")
    return models


def simulate_validation_results(models):
    """Simulate realistic validation results for all models."""
    print("Simulating validation results...")
    
    # Base metrics (realistic for small model)
    base_metrics = {
        'flops_per_token': 500_000,  # 0.5M FLOPs per token
        'latency_ms_per_token': 2.5,
        'throughput_tokens_per_sec': 400,
        'total_parameters': 15_000_000,  # 15M parameters
        'perplexity': 8.5,
        'glue_sst2_accuracy': 0.82
    }
    
    # Model-specific variations
    model_variations = {
        'M_base': {
            'multipliers': {'flops': 1.0, 'latency': 1.0, 'params': 1.0, 'accuracy': 1.0, 'perplexity': 1.0},
            'trainable_ratio': 1.0
        },
        'M_csp': {
            'multipliers': {'flops': 1.0, 'latency': 0.85, 'params': 1.0, 'accuracy': 1.01, 'perplexity': 0.98},
            'trainable_ratio': 1.0
        },
        'M_sdm': {
            'multipliers': {'flops': 0.75, 'latency': 0.90, 'params': 0.75, 'accuracy': 0.99, 'perplexity': 1.02},
            'trainable_ratio': 1.0
        },
        'M_sgh': {
            'multipliers': {'flops': 1.0, 'latency': 1.0, 'params': 1.0, 'accuracy': 1.03, 'perplexity': 1.0},
            'trainable_ratio': 0.06  # 6% trainable with SGH-PEFT
        },
        'M_challenge': {
            'multipliers': {'flops': 0.75, 'latency': 0.92, 'params': 0.75, 'accuracy': 1.02, 'perplexity': 1.05},
            'trainable_ratio': 0.08  # 8% trainable with uniform LoRA
        },
        'M_full': {
            'multipliers': {'flops': 0.75, 'latency': 0.78, 'params': 0.75, 'accuracy': 1.05, 'perplexity': 0.97},
            'trainable_ratio': 0.04  # 4% trainable with optimal SGH-PEFT
        }
    }
    
    results = {}
    demo_dir = Path("demo_validation_results")
    results_dir = demo_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    for model_name in models.keys():
        variation = model_variations[model_name]
        multipliers = variation['multipliers']
        
        # Calculate metrics
        model_results = {
            'model_group': model_name,
            'flops_per_token': int(base_metrics['flops_per_token'] * multipliers['flops']),
            'latency_ms_per_token': base_metrics['latency_ms_per_token'] * multipliers['latency'],
            'throughput_tokens_per_sec': base_metrics['throughput_tokens_per_sec'] / multipliers['latency'],
            'total_parameters': int(base_metrics['total_parameters'] * multipliers['params']),
            'trainable_parameters': int(base_metrics['total_parameters'] * multipliers['params'] * variation['trainable_ratio']),
            'perplexity': base_metrics['perplexity'] * multipliers['perplexity'],
            'glue_sst2_accuracy': base_metrics['glue_sst2_accuracy'] * multipliers['accuracy']
        }
        
        # Add some realistic noise
        model_results['latency_ms_per_token'] += np.random.normal(0, 0.05)
        model_results['glue_sst2_accuracy'] += np.random.normal(0, 0.005)
        
        results[model_name] = model_results
        
        # Save individual result file
        result_file = results_dir / f"{model_name}_validation.json"
        with open(result_file, 'w') as f:
            json.dump(model_results, f, indent=2)
    
    print(f"✓ Validation results simulated and saved to {results_dir}")
    return results


def validate_hypotheses(results):
    """Validate all four hypotheses with the simulated results."""
    print("\nValidating hypotheses...")
    print("="*50)
    
    hypotheses = {}
    
    # H1: CSP reduces latency
    if 'M_base' in results and 'M_CSP' in results:
        base_latency = results['M_base']['latency_ms_per_token']
        csp_latency = results['M_CSP']['latency_ms_per_token']
        improvement = (base_latency - csp_latency) / base_latency * 100
        
        hypotheses['H1'] = {
            'name': 'CSP reduces inference latency',
            'validated': improvement > 0,
            'improvement': improvement,
            'baseline': base_latency,
            'treatment': csp_latency
        }
        
        status = "✅ VALIDATED" if improvement > 0 else "❌ FAILED"
        print(f"H1: CSP Latency Reduction - {status}")
        print(f"    M_base: {base_latency:.2f} ms/token")
        print(f"    M_CSP:  {csp_latency:.2f} ms/token")
        print(f"    Improvement: {improvement:.1f}%")
    
    # H2: SDM reduces FLOPs
    if 'M_base' in results and 'M_SDM' in results:
        base_flops = results['M_base']['flops_per_token']
        sdm_flops = results['M_SDM']['flops_per_token']
        reduction = (base_flops - sdm_flops) / base_flops * 100
        
        hypotheses['H2'] = {
            'name': 'SDM reduces computational FLOPs',
            'validated': reduction > 0,
            'reduction': reduction,
            'baseline': base_flops,
            'treatment': sdm_flops
        }
        
        status = "✅ VALIDATED" if reduction > 0 else "❌ FAILED"
        print(f"\nH2: SDM FLOPs Reduction - {status}")
        print(f"    M_base: {base_flops:,} FLOPs/token")
        print(f"    M_SDM:  {sdm_flops:,} FLOPs/token")
        print(f"    Reduction: {reduction:.1f}%")
    
    # H3: SGH-PEFT improves parameter efficiency
    if 'M_challenge' in results and 'M_SGH' in results:
        challenge_params = results['M_challenge']['trainable_parameters']
        sgh_params = results['M_SGH']['trainable_parameters']
        efficiency = (challenge_params - sgh_params) / challenge_params * 100
        
        challenge_acc = results['M_challenge']['glue_sst2_accuracy']
        sgh_acc = results['M_SGH']['glue_sst2_accuracy']
        
        hypotheses['H3'] = {
            'name': 'SGH-PEFT improves parameter efficiency',
            'validated': efficiency > 0 and sgh_acc >= challenge_acc,
            'efficiency_gain': efficiency,
            'accuracy_maintained': sgh_acc >= challenge_acc,
            'challenge_params': challenge_params,
            'sgh_params': sgh_params
        }
        
        status = "✅ VALIDATED" if efficiency > 0 and sgh_acc >= challenge_acc else "❌ FAILED"
        print(f"\nH3: SGH-PEFT Parameter Efficiency - {status}")
        print(f"    M_challenge: {challenge_params:,} trainable params, {challenge_acc:.3f} accuracy")
        print(f"    M_SGH:       {sgh_params:,} trainable params, {sgh_acc:.3f} accuracy")
        print(f"    Efficiency gain: {efficiency:.1f}%")
    
    # H4: M_full achieves synergistic dominance
    if 'M_base' in results and 'M_full' in results:
        base_latency = results['M_base']['latency_ms_per_token']
        base_flops = results['M_base']['flops_per_token']
        base_params = results['M_base']['trainable_parameters']
        base_acc = results['M_base']['glue_sst2_accuracy']
        
        full_latency = results['M_full']['latency_ms_per_token']
        full_flops = results['M_full']['flops_per_token']
        full_params = results['M_full']['trainable_parameters']
        full_acc = results['M_full']['glue_sst2_accuracy']
        
        latency_improvement = (base_latency - full_latency) / base_latency * 100
        flops_reduction = (base_flops - full_flops) / base_flops * 100
        param_efficiency = (base_params - full_params) / base_params * 100
        accuracy_improvement = (full_acc - base_acc) / base_acc * 100
        
        # Synergy score: all improvements are positive
        synergy_validated = (latency_improvement > 0 and flops_reduction > 0 and 
                           param_efficiency > 0 and accuracy_improvement > 0)
        
        hypotheses['H4'] = {
            'name': 'M_full achieves synergistic dominance',
            'validated': synergy_validated,
            'latency_improvement': latency_improvement,
            'flops_reduction': flops_reduction,
            'param_efficiency': param_efficiency,
            'accuracy_improvement': accuracy_improvement
        }
        
        status = "✅ VALIDATED" if synergy_validated else "❌ FAILED"
        print(f"\nH4: M_full Synergistic Dominance - {status}")
        print(f"    Latency improvement: {latency_improvement:.1f}%")
        print(f"    FLOPs reduction: {flops_reduction:.1f}%")
        print(f"    Parameter efficiency: {param_efficiency:.1f}%")
        print(f"    Accuracy improvement: {accuracy_improvement:.1f}%")
    
    return hypotheses


def generate_demo_plots(results):
    """Generate demonstration plots."""
    print(f"\nGenerating demonstration plots...")
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        demo_dir = Path("demo_validation_results")
        plots_dir = demo_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Extract data
        models = list(results.keys())
        latencies = [results[m]['latency_ms_per_token'] for m in models]
        flops = [results[m]['flops_per_token'] / 1e6 for m in models]  # Convert to millions
        trainable_ratios = [results[m]['trainable_parameters'] / results[m]['total_parameters'] * 100 for m in models]
        accuracies = [results[m]['glue_sst2_accuracy'] for m in models]
        
        # Model colors
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#e377c2']
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Hardware-Data-Parameter Co-Design: Validation Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Latency vs Accuracy
        ax1 = axes[0, 0]
        scatter1 = ax1.scatter(latencies, accuracies, c=colors[:len(models)], s=100, alpha=0.7)
        for i, model in enumerate(models):
            ax1.annotate(model, (latencies[i], accuracies[i]), xytext=(5, 5), 
                        textcoords='offset points', fontsize=9)
        ax1.set_xlabel('Latency (ms/token)')
        ax1.set_ylabel('GLUE SST-2 Accuracy')
        ax1.set_title('H1: Latency vs Accuracy')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: FLOPs vs Accuracy
        ax2 = axes[0, 1]
        scatter2 = ax2.scatter(flops, accuracies, c=colors[:len(models)], s=100, alpha=0.7)
        for i, model in enumerate(models):
            ax2.annotate(model, (flops[i], accuracies[i]), xytext=(5, 5), 
                        textcoords='offset points', fontsize=9)
        ax2.set_xlabel('FLOPs per token (M)')
        ax2.set_ylabel('GLUE SST-2 Accuracy')
        ax2.set_title('H2: FLOPs vs Accuracy')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Parameter Efficiency vs Accuracy
        ax3 = axes[1, 0]
        scatter3 = ax3.scatter(trainable_ratios, accuracies, c=colors[:len(models)], s=100, alpha=0.7)
        for i, model in enumerate(models):
            ax3.annotate(model, (trainable_ratios[i], accuracies[i]), xytext=(5, 5), 
                        textcoords='offset points', fontsize=9)
        ax3.set_xlabel('Trainable Parameters (%)')
        ax3.set_ylabel('GLUE SST-2 Accuracy')
        ax3.set_title('H3: Parameter Efficiency vs Accuracy')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Combined metrics bar chart
        ax4 = axes[1, 1]
        
        # Normalize metrics for comparison (higher is better)
        norm_latency = [(max(latencies) - lat) / max(latencies) for lat in latencies]  # Invert latency
        norm_flops = [(max(flops) - f) / max(flops) for f in flops]  # Invert FLOPs
        norm_params = [(max(trainable_ratios) - tr) / max(trainable_ratios) for tr in trainable_ratios]  # Invert trainable%
        norm_acc = [(acc - min(accuracies)) / (max(accuracies) - min(accuracies)) for acc in accuracies]
        
        x = np.arange(len(models))
        width = 0.2
        
        ax4.bar(x - 1.5*width, norm_latency, width, label='Latency', alpha=0.7)
        ax4.bar(x - 0.5*width, norm_flops, width, label='FLOPs', alpha=0.7)
        ax4.bar(x + 0.5*width, norm_params, width, label='Param Eff', alpha=0.7)
        ax4.bar(x + 1.5*width, norm_acc, width, label='Accuracy', alpha=0.7)
        
        ax4.set_xlabel('Model Variants')
        ax4.set_ylabel('Normalized Score (Higher=Better)')
        ax4.set_title('H4: Overall Performance Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(models, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = plots_dir / "demo_validation_results.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Demo plots saved to {plot_path}")
        
    except ImportError:
        print("⚠ Matplotlib not available, skipping plot generation")


def print_summary_table(results):
    """Print a summary table of all results."""
    print(f"\n📊 VALIDATION RESULTS SUMMARY")
    print("="*80)
    
    # Table header
    header = f"{'Model':<12} {'Latency':<10} {'FLOPs(M)':<10} {'Trainable%':<12} {'Accuracy':<10} {'Perplexity':<10}"
    print(header)
    print("-" * len(header))
    
    # Table rows
    for model_name, result in results.items():
        latency = result['latency_ms_per_token']
        flops = result['flops_per_token'] / 1e6
        trainable_pct = result['trainable_parameters'] / result['total_parameters'] * 100
        accuracy = result['glue_sst2_accuracy']
        perplexity = result['perplexity']
        
        row = f"{model_name:<12} {latency:<10.2f} {flops:<10.1f} {trainable_pct:<12.1f} {accuracy:<10.3f} {perplexity:<10.1f}"
        print(row)


def main():
    """Main demonstration function."""
    print("🎯 HARDWARE-DATA-PARAMETER CO-DESIGN FRAMEWORK")
    print("🎯 VALIDATION DEMONSTRATION")
    print("="*80)
    
    try:
        # Create demonstration models
        models = create_demo_models()
        
        # Simulate validation results
        results = simulate_validation_results(models)
        
        # Validate hypotheses
        hypotheses = validate_hypotheses(results)
        
        # Generate plots
        generate_demo_plots(results)
        
        # Print summary
        print_summary_table(results)
        
        # Final validation summary
        print(f"\n🏆 VALIDATION SUMMARY")
        print("="*30)
        
        validated_count = sum(1 for h in hypotheses.values() if h['validated'])
        total_count = len(hypotheses)
        
        print(f"Hypotheses validated: {validated_count}/{total_count}")
        
        for h_name, h_data in hypotheses.items():
            status = "✅" if h_data['validated'] else "❌"
            print(f"{status} {h_name}: {h_data['name']}")
        
        if validated_count == total_count:
            print(f"\n🎉 ALL HYPOTHESES VALIDATED!")
            print(f"🏆 The co-design framework demonstrates clear synergistic benefits!")
        else:
            print(f"\n⚠ {total_count - validated_count} hypotheses need attention")
        
        print(f"\n📁 Demo results saved to: demo_validation_results/")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Demo validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 