#!/usr/bin/env python3
"""
YunMin Correlation Scan Batch Experiment Runner
Executes all 6 modes sequentially and aggregates results
"""

import subprocess
import time
import os
import json
from datetime import datetime

def run_experiment(mode, seed=42):
    """Run a single experiment mode"""
    print(f"\n{'='*60}")
    print(f"🚀 STARTING EXPERIMENT: {mode.upper()} MODE")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run training
        cmd = [
            "python", "train_yunmin.py",
            "--mode", mode,
            "--seed", str(seed),
            "--epochs", "3",
            "--batch_size", "4",
            "--lr", "5e-5",
            "--max_length", "256"
        ]
        
        print(f"🔄 Executing: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {mode.upper()} completed successfully in {duration:.2f}s")
            return {
                "mode": mode,
                "status": "success",
                "duration": duration,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        else:
            print(f"❌ {mode.upper()} failed (code: {result.returncode})")
            print(f"Error: {result.stderr}")
            return {
                "mode": mode,
                "status": "failed",
                "duration": duration,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {mode.upper()} timed out after 1 hour")
        return {
            "mode": mode,
            "status": "timeout",
            "duration": 3600,
            "error": "Timeout after 1 hour"
        }
    except Exception as e:
        print(f"💥 {mode.upper()} crashed: {e}")
        return {
            "mode": mode,
            "status": "crashed",
            "duration": time.time() - start_time,
            "error": str(e)
        }

def parse_results_file(mode):
    """Parse results from the saved results file"""
    results_file = f"./results_yunmin_{mode}_seed42.txt"
    
    if not os.path.exists(results_file):
        return None
    
    results = {}
    try:
        with open(results_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if ': ' in line:
                key, value = line.split(': ', 1)
                
                # Parse numeric values
                if key == "Perplexity":
                    results["perplexity"] = float(value)
                elif key == "Eval Loss":
                    results["eval_loss"] = float(value)
                elif key == "Training Time":
                    results["training_time"] = float(value.replace('s', ''))
                elif key == "Total Parameters":
                    results["total_params"] = int(value.replace(',', ''))
                elif key == "Trainable Parameters":
                    # Parse "X,XXX (Y.YY%)"
                    parts = value.split(' (')
                    results["trainable_params"] = int(parts[0].replace(',', ''))
                    results["trainable_pct"] = float(parts[1].replace('%)', ''))
                elif key == "Scan Applied":
                    results["scan_applied"] = value == "True"
                elif key == "IA3 Applied":
                    results["ia3_applied"] = value == "True"
                elif key == "LoRA Applied":
                    results["lora_applied"] = value == "True"
                    
    except Exception as e:
        print(f"⚠️ Failed to parse results for {mode}: {e}")
        return None
        
    return results

def main():
    """Run all experiments"""
    print("🧪 YunMin Correlation Scan - Batch Experiment Runner")
    print("=" * 60)
    print("📋 Experiment Plan:")
    print("   1. Baseline (Full fine-tuning)")
    print("   2. LoRA-only (PEFT @ SSM-only)")
    print("   3. Scan-only (π insertion)")
    print("   4. Hybrid (LoRA + Scan)")
    print("   5. IA3-only (per-channel scaling)")
    print("   6. IA3 + LoRA")
    print("=" * 60)
    # Create results directory
    os.makedirs("./batch_results", exist_ok=True)
    
    modes = ["baseline", "lora", "scan", "hybrid", "ia3", "ia3_lora"]
    results = {}
    experiment_log = []
    total_start = time.time()
    
    for i, mode in enumerate(modes, 1):
        print(f"\n📊 Progress: {i}/{len(modes)} experiments")
        
        # Run experiment
        exp_result = run_experiment(mode)
        experiment_log.append(exp_result)
        
        # Parse results if successful
        if exp_result["status"] == "success":
            parsed_results = parse_results_file(mode)
            if parsed_results:
                results[mode] = parsed_results
                
        # Brief pause between experiments
        time.sleep(5)
    
    total_duration = time.time() - total_start
    
    # -------------------------
    # Generate Summary Report
    # -------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"./batch_results/experiment_summary_{timestamp}.json"
    
    summary = {
        "timestamp": timestamp,
        "total_duration": total_duration,
        "experiment_log": experiment_log,
        "results": results
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print Summary Table
    print("\n" + "="*80)
    print("📊 EXPERIMENT SUMMARY")
    print("="*80)
    
    if results:
        print(f"{'Mode':<12} {'PPL':<12} {'Time(s)':<10} {'Trainable%':<12} {'Memory(GB)':<12}")
        print("-" * 80)
        
        for mode in modes:
            if mode in results:
                r = results[mode]
                print(f"{mode:<12} {r.get('perplexity', 'N/A'):<12.2f} "
                      f"{r.get('training_time', 'N/A'):<10.1f} "
                      f"{r.get('trainable_pct', 'N/A'):<12.2f} "
                      f"{'TBD':<12}")
            else:
                print(f"{mode:<12} {'FAILED':<12} {'N/A':<10} {'N/A':<12} {'N/A':<12}")
    
    print("\n🎯 Key Findings:")
    if "baseline" in results and "hybrid" in results:
        baseline_ppl = results["baseline"]["perplexity"]
        hybrid_ppl = results["hybrid"]["perplexity"]
        improvement = ((baseline_ppl - hybrid_ppl) / baseline_ppl) * 100
        print(f"   • Hybrid vs Baseline PPL improvement: {improvement:.1f}%")
    
    if "lora" in results and "hybrid" in results:
        lora_ppl = results["lora"]["perplexity"]
        hybrid_ppl = results["hybrid"]["perplexity"]
        improvement = ((lora_ppl - hybrid_ppl) / lora_ppl) * 100
        print(f"   • Hybrid vs LoRA-only improvement: {improvement:.1f}%")
    
    print(f"\n📄 Full results saved to: {summary_file}")
    print(f"⏱️ Total experiment time: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
    print("\n🏁 All experiments completed!")

if __name__ == "__main__":
    main() 