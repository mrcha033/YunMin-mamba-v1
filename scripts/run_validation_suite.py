"""
Comprehensive Validation Suite for Hardware-Data-Parameter Co-Design Framework

This script validates the four main hypotheses (H1-H4) of the co-design framework:
- H1: CSP reduces latency while maintaining performance
- H2: SDM reduces FLOPs through learned sparsity  
- H3: SGH-PEFT improves parameter efficiency
- H4: M_full achieves synergistic dominance across all metrics

Usage:
    python scripts/run_validation_suite.py --model_group M_full --checkpoint checkpoints/full/model.pt --validate_all
"""

import argparse
import json
import os
import sys
import time
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
from fvcore.nn import FlopCountAnalysis

# Add project root to path with higher priority
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from models.baseline_ssm import BaselineSSM
from models.sdm_ssm import SDM_SSM
from models.sgh_peft import (
    SGHPEFTModel,
    SGHPEFTConfig,
    MaskedLoRALayer,
    create_sgh_peft_model,
)
from data.wikitext103 import get_wikitext103_dataloader
from data.glue import get_glue_dataloader
from utils.profiling import count_parameters, measure_latency
from transformers import AutoTokenizer


class ValidationSuite:
    """
    Comprehensive validation suite for the co-design framework.
    
    Supports evaluation of all model variants:
    - M_base: Original baseline model
    - M_csp: M_base + CSP permutation (Pillar 1)
    - M_sdm: M_base + SDM sparsity (Pillar 2)
    - M_sgh: M_base + SGH-PEFT with proxy importance
    - M_sdm+sgh: M_sdm fine-tuned with SGH-PEFT using learned sparsity masks
    - M_challenge: M_base + magnitude pruning + uniform LoRA
    - M_full: M_base + CSP + SDM + SGH-PEFT (all pillars)
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def load_model_for_group(self, group_name: str, checkpoint_path: str, config: Dict[str, Any]) -> nn.Module:
        """
        Factory function to load the correct model for a given group.
        
        Args:
            group_name: Model group identifier (M_base, M_CSP, M_SDM, etc.)
            checkpoint_path: Path to model checkpoint
            config: Model configuration
            
        Returns:
            Loaded and initialized model
        """
        print(f"Loading model for group: {group_name} from {checkpoint_path}")
        
        # Model configuration
        model_config = {
            'd_model': config.get('d_model', 768),
            'n_layer': config.get('n_layer', 12),
            'vocab_size': config.get('vocab_size', 50257),
            'd_state': config.get('d_state', 16),
            'd_conv': config.get('d_conv', 4)
        }
        
        if group_name in ['M_base', 'M_csp', 'M_challenge']:
            # Base model variants
            model = BaselineSSM(**model_config)
        elif group_name in ['M_sdm', 'M_sgh']:
            # SDM-based models
            model = SDM_SSM(**model_config, gumbel_temp=1.0)
        elif group_name in ['M_full', 'M_sdm+sgh']:
            # Full pipeline model (SDM + SGH-PEFT)
            base_model = SDM_SSM(**model_config, gumbel_temp=1.0)
            # Load base SDM checkpoint first
            if os.path.isfile(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                if 'model_state_dict' in checkpoint:
                    base_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    base_model.load_state_dict(checkpoint)
            
            # Create SGH-PEFT model
            model = create_sgh_peft_model(base_model)
            model.eval().to(self.device)
            return model
        else:
            raise ValueError(f"Unknown model group: {group_name}")
        
        # Load checkpoint if it exists
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Load state dict with error handling
            try:
                model.load_state_dict(state_dict, strict=True)
            except RuntimeError as e:
                print(f"Warning: Could not load checkpoint strictly: {e}")
                model.load_state_dict(state_dict, strict=False)
        else:
            print(f"Warning: No checkpoint found at {checkpoint_path}, using randomly initialized model")
        
        model.eval().to(self.device)
        return model
    
    def validate_pretrain_metrics(self, model: nn.Module, model_name: str) -> Dict[str, Any]:
        """
        Validate pre-training metrics: FLOPs and Perplexity.
        
        H2 Validation: Prove SDM effectively reduces computational cost.
        
        Args:
            model: Model to evaluate
            model_name: Model identifier
            
        Returns:
            Dictionary with FLOPs and perplexity metrics
        """
        print(f"Validating pre-training metrics for {model_name}...")
        
        results = {"model": model_name}
        
        # 1. FLOPs Calculation
        try:
            dummy_input = torch.randint(0, 50257, (1, 1024), device=self.device)
            with torch.no_grad():
                flops_analysis = FlopCountAnalysis(model, dummy_input)
                total_flops = flops_analysis.total()
                results["total_flops"] = int(total_flops)
                results["flops_per_token"] = int(total_flops / 1024)
                
                print(f"✓ FLOPs analysis completed: {total_flops:,} total, {total_flops//1024:,} per token")
        except Exception as e:
            print(f"Warning: FLOPs analysis failed: {e}")
            results["total_flops"] = -1
            results["flops_per_token"] = -1
        
        # 2. Perplexity Calculation on WikiText-103
        try:
            val_dataloader = get_wikitext103_dataloader(
                tokenizer=self.tokenizer,
                batch_size=8,
                max_length=1024,
                split="validation"
            )
            
            perplexity = self.calculate_perplexity(model, val_dataloader)
            results["perplexity"] = perplexity
            
            print(f"✓ Perplexity calculated: {perplexity:.4f}")
        except Exception as e:
            print(f"Warning: Perplexity calculation failed: {e}")
            results["perplexity"] = -1
        
        # 3. Model size metrics
        param_info = count_parameters(model)
        results.update(param_info)
        
        return results
    
    def calculate_perplexity(self, model: nn.Module, dataloader) -> float:
        """Calculate perplexity on a dataset."""
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= 50:  # Limit evaluation for speed
                    break
                
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch.get("attention_mask", None)
                
                # Create labels (shifted input_ids)
                labels = input_ids[:, 1:].contiguous()
                input_ids = input_ids[:, :-1].contiguous()
                
                # Forward pass
                logits = model(input_ids)
                
                # Calculate loss
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                # Accumulate
                seq_len = labels.numel()
                total_loss += loss.item() * seq_len
                total_tokens += seq_len
        
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return perplexity
    
    def validate_inference_speed(self, model: nn.Module, model_name: str) -> Dict[str, Any]:
        """
        Validate inference speed: latency and throughput.
        
        H1 Validation: Prove CSP reduces wall-clock latency.
        
        Args:
            model: Model to evaluate
            model_name: Model identifier
            
        Returns:
            Dictionary with latency and throughput metrics
        """
        print(f"Validating inference speed for {model_name}...")
        
        results = {"model": model_name}
        
        # 1. Latency measurement (batch_size=1, autoregressive)
        try:
            latency_info = self.measure_autoregressive_latency(model)
            results.update(latency_info)
            
            print(f"✓ Latency measured: {latency_info['latency_ms_per_token']:.2f} ms/token")
        except Exception as e:
            print(f"Warning: Latency measurement failed: {e}")
            results["latency_ms_per_token"] = -1
        
        # 2. Throughput measurement (large batch)
        try:
            throughput_info = self.measure_batch_throughput(model)
            results.update(throughput_info)
            
            print(f"✓ Throughput measured: {throughput_info['throughput_tokens_per_sec']:.2f} tokens/sec")
        except Exception as e:
            print(f"Warning: Throughput measurement failed: {e}")
            results["throughput_tokens_per_sec"] = -1
        
        return results
    
    def measure_autoregressive_latency(self, model: nn.Module) -> Dict[str, float]:
        """Measure autoregressive generation latency."""
        model.eval()
        
        # Generate 512 tokens autoregressively
        batch_size = 1
        seq_length = 512
        prompt_length = 32
        
        # Create prompt
        input_ids = torch.randint(0, 50257, (batch_size, prompt_length), device=self.device)
        
        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = model(input_ids)
        
        # Measure generation time
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            current_input = input_ids
            for _ in range(seq_length - prompt_length):
                logits = model(current_input)
                next_token = torch.argmax(logits[:, -1:], dim=-1)
                current_input = torch.cat([current_input, next_token], dim=1)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        total_time_ms = (end_time - start_time) * 1000
        generated_tokens = seq_length - prompt_length
        latency_ms_per_token = total_time_ms / generated_tokens
        
        return {
            "latency_ms_per_token": latency_ms_per_token,
            "total_generation_time_ms": total_time_ms,
            "generated_tokens": generated_tokens
        }
    
    def measure_batch_throughput(self, model: nn.Module) -> Dict[str, float]:
        """Measure batch processing throughput."""
        model.eval()
        
        # Use large batch to saturate GPU
        batch_size = 64
        seq_length = 512
        
        input_ids = torch.randint(0, 50257, (batch_size, seq_length), device=self.device)
        
        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = model(input_ids)
        
        # Measure throughput
        torch.cuda.synchronize()
        start_time = time.time()
        
        num_runs = 10
        for _ in range(num_runs):
            with torch.no_grad():
                _ = model(input_ids)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        total_time = end_time - start_time
        total_tokens = batch_size * seq_length * num_runs
        throughput = total_tokens / total_time
        
        return {
            "throughput_tokens_per_sec": throughput,
            "batch_size": batch_size,
            "sequence_length": seq_length,
            "total_time_sec": total_time
        }
    
    def validate_finetune_efficiency(self, model_group: str, base_checkpoint: str, config: Dict[str, Any], sdm_checkpoint: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate fine-tuning efficiency: trainable parameters and GLUE performance.
        
        H3 Validation: Prove SGH-PEFT is more parameter-efficient than standard LoRA.
        
        Args:
            model_group: Model group identifier
            base_checkpoint: Path to base model checkpoint
            config: Model configuration
            
        Returns:
            Dictionary with parameter efficiency and performance metrics
        """
        print(f"Validating fine-tuning efficiency for {model_group}...")
        
        results = {"model": model_group}
        
        try:
            # Load base model
            base_model = self.load_model_for_group(model_group, base_checkpoint, config)
            
            # Create fine-tuned model based on group
            if model_group in ['M_full', 'M_sdm+sgh']:
                # Already has SGH-PEFT applied
                finetuned_model = base_model
            elif model_group == 'M_sgh':
                # Apply SGH-PEFT with proxy importance scores
                finetuned_model = self.create_sgh_peft_with_proxy(base_model)
            elif model_group == 'M_challenge':
                # Apply magnitude pruning + uniform LoRA
                finetuned_model = self.create_magnitude_pruned_lora(base_model, sdm_checkpoint)
            else:
                # Standard LoRA for other models
                finetuned_model = self.create_standard_lora(base_model)
            
            # Count trainable parameters
            total_params = sum(p.numel() for p in finetuned_model.parameters())
            trainable_params = sum(p.numel() for p in finetuned_model.parameters() if p.requires_grad)
            
            results["total_parameters"] = total_params
            results["trainable_parameters"] = trainable_params
            results["trainable_ratio"] = trainable_params / total_params
            if model_group == 'M_challenge':
                results["pruning_sparsity"] = getattr(self, "last_pruning_sparsity", 0.0)
            
            # Run GLUE evaluation (simplified for SST-2)
            glue_score = self.evaluate_glue_task(finetuned_model, task="sst2")
            results["glue_sst2_accuracy"] = glue_score
            
            print(f"✓ Parameter efficiency: {trainable_params:,}/{total_params:,} ({trainable_params/total_params:.2%}) trainable")
            print(f"✓ GLUE SST-2 accuracy: {glue_score:.4f}")
            
        except Exception as e:
            print(f"Warning: Fine-tuning validation failed: {e}")
            results.update({
                "total_parameters": -1,
                "trainable_parameters": -1, 
                "trainable_ratio": -1,
                "glue_sst2_accuracy": -1
            })
        
        return results
    
    def create_sgh_peft_with_proxy(self, base_model: nn.Module) -> nn.Module:
        """Create SGH-PEFT model with proxy importance scores (weight magnitude)."""
        # Compute layer importance using average weight magnitude
        importance_scores: Dict[str, Dict[str, Any]] = {}

        with torch.no_grad():
            for idx, layer in enumerate(base_model.layers):
                layer_name = f"layers.{idx}"

                magnitudes = []
                for param in layer.parameters():
                    magnitudes.append(param.detach().abs().mean())

                if magnitudes:
                    mean_imp = torch.stack(magnitudes).mean().item()
                else:
                    mean_imp = 0.0

                d_inner = getattr(layer, "d_inner", layer.in_proj.weight.shape[0] // 2)

                importance_scores[layer_name] = {
                    "mean_importance": mean_imp,
                    "std_importance": 0.0,
                    "max_importance": mean_imp,
                    "min_importance": mean_imp,
                    "active_channels": d_inner,
                    "total_channels": d_inner,
                    "sparsity_level": 0.0,
                    "sparsity_mask": torch.ones(d_inner),
                }

        config = SGHPEFTConfig(apply_sparsity_mask=False, freeze_base_model=True)
        return create_sgh_peft_model(base_model, config, layer_importance_scores=importance_scores)
    
    def create_magnitude_pruned_lora(self, base_model: nn.Module, sdm_checkpoint: Optional[str] = None) -> nn.Module:
        """
        Create magnitude-pruned model with uniform LoRA.
        
        This implements the M_challenge baseline that combines:
        1. Magnitude-based channel pruning (17.6% sparsity to match M_SDM)
        2. Uniform LoRA adaptation across all layers
        
        Args:
            base_model: Base model to adapt
            sdm_checkpoint: Path to SDM checkpoint for pruning ratio
            
        Returns:
            Model with magnitude pruning + uniform LoRA
        """
        # Try to detect sparsity from SDM checkpoint if available
        sparsity_ratio = 0.176  # Default to match M_SDM parameter reduction (~17.6%)
        
        # Check if we can extract sparsity from an SDM checkpoint
        if sdm_checkpoint and os.path.isfile(sdm_checkpoint):
            checkpoint = torch.load(sdm_checkpoint, map_location="cpu")
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            z_keys = [k for k in state_dict.keys() if k.endswith("z_logits")]
            total = 0
            kept = 0
            for k in z_keys:
                z = state_dict[k]
                total += z.numel()
                kept += (z > 0).float().sum().item()
            if total > 0:
                sparsity_ratio = 1.0 - kept / total
                print(f"✓ SDM sparsity ratio detected: {sparsity_ratio:.2%}")
        else:
            print(f"Using default sparsity ratio: {sparsity_ratio:.2%}")

        self.last_pruning_sparsity = sparsity_ratio

        # Freeze original parameters
        for p in base_model.parameters():
            p.requires_grad = False

        # Apply magnitude-based pruning if sparsity > 0
        if sparsity_ratio > 0:
            print(f"Applying magnitude-based pruning with {sparsity_ratio:.2%} sparsity...")
            
            # Collect channel importance scores across all layers
            channel_scores = []
            for layer in base_model.layers:
                # Use magnitude of input projection weights as importance metric
                weight = layer.in_proj.weight.data[:layer.d_inner]  # First half for x projection
                channel_scores.append(weight.abs().mean(dim=1))
            
            # Global threshold based on magnitude
            flat_scores = torch.cat(channel_scores)
            k = int(len(flat_scores) * sparsity_ratio)
            threshold = flat_scores.kthvalue(k).values.item() if k > 0 else -float("inf")
            
            # Apply pruning masks to each layer
            idx = 0
            for layer in base_model.layers:
                n = layer.d_inner
                scores = flat_scores[idx:idx+n]
                idx += n
                
                # Create binary mask (1 = keep, 0 = prune)
                mask = (scores > threshold).float()
                
                # Apply mask to weights (zero out pruned channels)
                layer.in_proj.weight.data[:n] *= mask.view(-1, 1)      # x projection
                layer.in_proj.weight.data[n:] *= mask.view(-1, 1)      # z projection  
                layer.out_proj.weight.data *= mask.view(1, -1)         # output projection
                layer.conv1d.weight.data *= mask.view(-1, 1, 1)        # convolution

        # Apply uniform LoRA to all layers
        from models.sgh_peft import MaskedLoRALayer
        
        rank = 4
        alpha_factor = 2
        dropout = 0.05
        
        print(f"Applying uniform LoRA (rank={rank}, alpha={rank*alpha_factor}) to all layers...")
        
        for layer in base_model.layers:
            # Replace projections with LoRA-adapted versions
            layer.in_proj = MaskedLoRALayer(
                layer.in_proj,
                rank=rank,
                alpha=rank * alpha_factor,
                dropout=dropout,
            )
            layer.out_proj = MaskedLoRALayer(
                layer.out_proj,
                rank=rank,
                alpha=rank * alpha_factor,
                dropout=dropout,
            )
            
            # Freeze remaining parameters (conv1d, SSM parameters)
            for param in [
                layer.conv1d.weight,
                layer.x_proj.weight,
                layer.dt_proj.weight,
                layer.A_log,
                layer.D
            ]:
                param.requires_grad = False

        return base_model
    
    def verify_iso_sparsity(self, sdm_model_path: str, challenge_model_path: str) -> Dict[str, Any]:
        """
        Verify challenge baseline matches SDM sparsity exactly.
        
        This function ensures fair comparison by validating that M_challenge
        has the same sparsity level as M_SDM, as required by the experimental description.
        
        Args:
            sdm_model_path: Path to M_SDM checkpoint
            challenge_model_path: Path to M_challenge checkpoint
            
        Returns:
            Dictionary with sparsity comparison results
        """
        print("🔍 Verifying iso-sparsity between M_SDM and M_challenge...")
        
        results = {
            "sdm_path": sdm_model_path,
            "challenge_path": challenge_model_path,
            "sdm_sparsity": -1.0,
            "challenge_sparsity": -1.0,
            "sparsity_difference": -1.0,
            "iso_sparsity_verified": False,
            "tolerance": 0.01  # 1% tolerance
        }
        
        try:
            # Extract M_SDM sparsity from z_logits
            if os.path.isfile(sdm_model_path):
                sdm_checkpoint = torch.load(sdm_model_path, map_location="cpu")
                sdm_state_dict = sdm_checkpoint.get("model_state_dict", sdm_checkpoint)
                
                z_keys = [k for k in sdm_state_dict.keys() if k.endswith("z_logits")]
                if z_keys:
                    total_channels = kept_channels = 0
                    layer_sparsities = []
                    
                    for k in z_keys:
                        z = sdm_state_dict[k]
                        layer_total = z.numel()
                        layer_kept = (z > 0).float().sum().item()
                        layer_sparsity = 1.0 - (layer_kept / layer_total)
                        
                        total_channels += layer_total
                        kept_channels += layer_kept
                        layer_sparsities.append(layer_sparsity)
                    
                    if total_channels > 0:
                        results["sdm_sparsity"] = 1.0 - kept_channels / total_channels
                        print(f"  📊 M_SDM sparsity: {results['sdm_sparsity']:.4f} ({results['sdm_sparsity']:.2%})")
                        print(f"     Total channels: {total_channels}, Kept: {kept_channels}")
                        print(f"     Per-layer sparsity range: {min(layer_sparsities):.2%} - {max(layer_sparsities):.2%}")
                    else:
                        print("  ⚠️  No valid z_logits found in M_SDM checkpoint")
                else:
                    print("  ⚠️  No z_logits found in M_SDM checkpoint")
            else:
                print(f"  ❌ M_SDM checkpoint not found: {sdm_model_path}")
            
            # Extract M_challenge sparsity from pruned weights
            if os.path.isfile(challenge_model_path):
                challenge_checkpoint = torch.load(challenge_model_path, map_location="cpu")
                challenge_state_dict = challenge_checkpoint.get("model_state_dict", challenge_checkpoint)
                
                # Look for in_proj weights and count zero channels
                layer_weights = [k for k in challenge_state_dict.keys() if k.endswith("in_proj.weight")]
                
                if layer_weights:
                    total_channels = pruned_channels = 0
                    layer_sparsities = []
                    
                    for k in layer_weights:
                        weight = challenge_state_dict[k]
                        if weight.numel() > 0:
                            # Get first half of in_proj (x projection)
                            d_inner = weight.shape[0] // 2
                            x_proj_weight = weight[:d_inner]
                            
                            # Count channels with zero magnitude
                            channel_magnitudes = x_proj_weight.abs().mean(dim=1)
                            zero_channels = (channel_magnitudes < 1e-8).sum().item()
                            layer_sparsity = zero_channels / d_inner
                            
                            total_channels += d_inner
                            pruned_channels += zero_channels
                            layer_sparsities.append(layer_sparsity)
                    
                    if total_channels > 0:
                        results["challenge_sparsity"] = pruned_channels / total_channels
                        print(f"  📊 M_challenge sparsity: {results['challenge_sparsity']:.4f} ({results['challenge_sparsity']:.2%})")
                        print(f"     Total channels: {total_channels}, Pruned: {pruned_channels}")
                        print(f"     Per-layer sparsity range: {min(layer_sparsities):.2%} - {max(layer_sparsities):.2%}")
                    else:
                        print("  ⚠️  No valid weight tensors found in M_challenge checkpoint")
                else:
                    print("  ⚠️  No in_proj weights found in M_challenge checkpoint")
            else:
                print(f"  ❌ M_challenge checkpoint not found: {challenge_model_path}")
            
            # Compare sparsity levels
            if results["sdm_sparsity"] >= 0 and results["challenge_sparsity"] >= 0:
                results["sparsity_difference"] = abs(results["sdm_sparsity"] - results["challenge_sparsity"])
                results["iso_sparsity_verified"] = results["sparsity_difference"] <= results["tolerance"]
                
                print(f"\n  🎯 SPARSITY COMPARISON:")
                print(f"     M_SDM:       {results['sdm_sparsity']:.4f} ({results['sdm_sparsity']:.2%})")
                print(f"     M_challenge: {results['challenge_sparsity']:.4f} ({results['challenge_sparsity']:.2%})")
                print(f"     Difference:  {results['sparsity_difference']:.4f} ({results['sparsity_difference']:.2%})")
                print(f"     Tolerance:   {results['tolerance']:.4f} ({results['tolerance']:.2%})")
                
                if results["iso_sparsity_verified"]:
                    print(f"  ✅ ISO-SPARSITY VERIFIED: Fair comparison ensured!")
                else:
                    print(f"  ❌ ISO-SPARSITY VIOLATION: Sparsity difference exceeds tolerance!")
                    print(f"     This may lead to unfair comparison between learned vs. heuristic pruning.")
            else:
                print(f"  ⚠️  Could not extract sparsity from one or both checkpoints")
                
        except Exception as e:
            print(f"  ❌ Sparsity verification failed: {e}")
            import traceback
            traceback.print_exc()
        
        return results
    
    def create_standard_lora(self, base_model: nn.Module) -> nn.Module:
        """Create model with standard uniform LoRA."""
        model = base_model

        for p in model.parameters():
            p.requires_grad = False

        rank = 4
        alpha_factor = 2
        dropout = 0.05

        for layer in model.layers:
            layer.in_proj = MaskedLoRALayer(
                layer.in_proj,
                rank=rank,
                alpha=rank * alpha_factor,
                dropout=dropout,
            )
            layer.out_proj = MaskedLoRALayer(
                layer.out_proj,
                rank=rank,
                alpha=rank * alpha_factor,
                dropout=dropout,
            )

        return model
    
    def evaluate_glue_task(self, model: nn.Module, task: str = "sst2") -> float:
        """
        Evaluate model on GLUE task.
        
        Args:
            model: Model to evaluate
            task: GLUE task name
            
        Returns:
            Task accuracy/F1 score
        """
        try:
            # Create task-specific evaluation dataloader
            eval_dataloader = get_glue_dataloader(
                task_name=task,
                tokenizer=self.tokenizer,
                batch_size=16,
                max_length=512,
                split="validation",
            )

            model.eval()
            all_predictions = []
            all_labels = []

            with torch.no_grad():
                for batch in eval_dataloader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"]

                    outputs = model(input_ids, attention_mask=attention_mask)
                    if hasattr(outputs, "logits"):
                        predictions = outputs.logits
                    else:
                        predictions = outputs

                    all_predictions.append(predictions.cpu().numpy())
                    all_labels.append(labels.numpy())

            all_predictions = np.concatenate(all_predictions, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)

            metrics = compute_glue_metrics(task, all_predictions, all_labels)

            # Return accuracy/F1 if available, otherwise the first metric
            for key in ["accuracy", "f1"]:
                if key in metrics:
                    return metrics[key]
            return next(iter(metrics.values())) if metrics else -1.0

        except Exception as e:
            print(f"Warning: GLUE evaluation failed: {e}")
            return -1.0
    
    def run_comprehensive_validation(self, model_group: str, checkpoint_path: str, config: Dict[str, Any], sdm_checkpoint: Optional[str] = None) -> Dict[str, Any]:
        """
        Run comprehensive validation for all hypotheses.
        
        Args:
            model_group: Model group identifier
            checkpoint_path: Path to model checkpoint
            config: Model configuration
            
        Returns:
            Complete validation results
        """
        print(f"\n{'='*70}")
        print(f"COMPREHENSIVE VALIDATION: {model_group}")
        print(f"{'='*70}")
        
        results = {"model_group": model_group, "checkpoint_path": checkpoint_path}
        
        # Load model
        model = self.load_model_for_group(model_group, checkpoint_path, config)
        
        # H2: Pre-training metrics (FLOPs, perplexity)
        print("\n[H2] Validating pre-training metrics...")
        pretrain_results = self.validate_pretrain_metrics(model, model_group)
        results.update(pretrain_results)
        
        # H1: Inference speed (latency, throughput)
        print("\n[H1] Validating inference speed...")
        speed_results = self.validate_inference_speed(model, model_group)
        results.update(speed_results)
        
        # H3: Fine-tuning efficiency
        print("\n[H3] Validating fine-tuning efficiency...")
        finetune_results = self.validate_finetune_efficiency(model_group, checkpoint_path, config, sdm_checkpoint)
        results.update(finetune_results)
        
        # Calculate efficiency ratios for comparison
        if results.get("total_flops", -1) > 0 and results.get("trainable_parameters", -1) > 0:
            results["efficiency_score"] = (1e12 / results["total_flops"]) * (1e6 / results["trainable_parameters"])
        else:
            results["efficiency_score"] = -1
        
        return results
    
    def save_results(self, results: Dict[str, Any], model_group: str):
        """Save validation results to JSON file."""
        output_file = self.results_dir / f"{model_group}_validation.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\n✅ Validation results saved to {output_file}")
        
        # Print summary
        print(f"\n📊 VALIDATION SUMMARY: {model_group}")
        print(f"{'='*50}")
        print(f"FLOPs per token:      {results.get('flops_per_token', 'N/A'):,}")
        print(f"Perplexity:           {results.get('perplexity', 'N/A'):.4f}")
        print(f"Latency (ms/token):   {results.get('latency_ms_per_token', 'N/A'):.2f}")
        print(f"Throughput (tok/sec): {results.get('throughput_tokens_per_sec', 'N/A'):.2f}")
        print(f"Trainable params:     {results.get('trainable_parameters', 'N/A'):,}")
        print(f"GLUE SST-2 accuracy:  {results.get('glue_sst2_accuracy', 'N/A'):.4f}")
        print(f"Efficiency score:     {results.get('efficiency_score', 'N/A'):.2e}")
        if "pruning_sparsity" in results:
            print(f"Pruning sparsity:     {results['pruning_sparsity']:.2%}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive validation suite for the co-design framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model Groups:
  M_base      - Original baseline Mamba model
  M_csp       - M_base + CSP permutation (Pillar 1)
  M_sdm       - M_base + SDM sparsity (Pillar 2)
  M_sgh       - M_base + SGH-PEFT with proxy importance
  M_sdm+sgh   - M_sdm fine-tuned with SGH-PEFT using learned sparsity masks
  M_challenge - M_base + magnitude pruning + uniform LoRA
  M_full      - M_base + CSP + SDM + SGH-PEFT (all pillars)

Examples:
  # Validate M_full model with all tests
  python scripts/run_validation_suite.py --model_group M_full --checkpoint checkpoints/full/model.pt --validate_all
  
  # Validate only inference speed for M_CSP
  python scripts/run_validation_suite.py --model_group M_CSP --checkpoint checkpoints/csp/model.pt --validate_speed
        """
    )
    
    parser.add_argument("--model_group", type=str, required=True,
                       choices=['M_base', 'M_CSP', 'M_SDM', 'M_SGH', 'M_sdm_sgh', 'M_challenge', 'M_full'],
                       help="Model group identifier")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to the model checkpoint")
    parser.add_argument("--sdm_checkpoint", type=str, default=None,
                       help="Path to SDM checkpoint for pruning ratio")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml",
                       help="Path to model configuration file")
    
    # Validation options
    parser.add_argument("--validate_pretrain", action="store_true",
                       help="Validate FLOPs and Perplexity (H2)")
    parser.add_argument("--validate_speed", action="store_true", 
                       help="Validate Latency and Throughput (H1)")
    parser.add_argument("--validate_finetune", action="store_true",
                       help="Validate Fine-tuning efficiency (H3)")
    parser.add_argument("--validate_all", action="store_true",
                       help="Run all validation tests (H1, H2, H3)")
    
    # System options
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run validation on")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Output directory for results")
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if os.path.exists(config_path):
        import yaml
        with open(config_path, 'r') as f:
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


def main():
    """Main validation function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize validation suite
    validator = ValidationSuite(device=args.device)
    validator.results_dir = Path(args.output_dir)
    validator.results_dir.mkdir(exist_ok=True)
    
    # Determine which validations to run
    run_pretrain = args.validate_pretrain or args.validate_all
    run_speed = args.validate_speed or args.validate_all
    run_finetune = args.validate_finetune or args.validate_all
    
    if not any([run_pretrain, run_speed, run_finetune]):
        print("No validation tests specified. Use --validate_all or specific --validate_* flags.")
        return
    
    try:
        # Run comprehensive validation
        results = validator.run_comprehensive_validation(
            model_group=args.model_group,
            checkpoint_path=args.checkpoint,
            config=config,
            sdm_checkpoint=args.sdm_checkpoint
        )
        
        # Save results
        validator.save_results(results, args.model_group)
        
        print(f"\n🎉 Validation completed successfully for {args.model_group}!")
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 