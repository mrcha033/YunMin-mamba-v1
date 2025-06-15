import argparse
import os
import yaml
from pathlib import Path
import torch

from models.baseline_ssm import BaselineSSM
from models.sgh_peft import MaskedLoRALayer


def magnitude_pruned_lora(model: BaselineSSM, sdm_checkpoint: str = None) -> BaselineSSM:
    """
    Apply magnitude pruning + uniform LoRA with iso-sparsity verification.
    
    For fair comparison, M_challenge sparsity level matches M_SDM exactly.
    """
    sparsity_ratio = 0.176  # Default fallback
    
    # CRITICAL: Extract exact sparsity from M_SDM for iso-sparsity comparison
    if sdm_checkpoint and os.path.isfile(sdm_checkpoint):
        print(f"🔍 Loading SDM checkpoint for iso-sparsity verification: {sdm_checkpoint}")
        ckpt = torch.load(sdm_checkpoint, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)
        z_keys = [k for k in state_dict if k.endswith("z_logits")]
        
        total_channels = kept_channels = 0
        layer_sparsities = []
        
        for k in z_keys:
            z = state_dict[k]
            layer_total = z.numel()
            layer_kept = (z > 0).float().sum().item()
            layer_sparsity = 1.0 - (layer_kept / layer_total)
            
            total_channels += layer_total
            kept_channels += layer_kept
            layer_sparsities.append(layer_sparsity)
            
            print(f"    Layer {k}: {layer_sparsity:.2%} sparsity ({layer_kept}/{layer_total} channels kept)")
        
        if total_channels > 0:
            sparsity_ratio = 1.0 - kept_channels / total_channels
            print(f"✅ DETECTED M_SDM SPARSITY: {sparsity_ratio:.4f} ({sparsity_ratio:.2%})")
            print(f"   Total channels: {total_channels}, Kept: {kept_channels}, Pruned: {total_channels - kept_channels}")
            print(f"   Per-layer sparsity range: {min(layer_sparsities):.2%} - {max(layer_sparsities):.2%}")
        else:
            print("⚠️  No z_logits found in SDM checkpoint, using default sparsity")
    else:
        print(f"⚠️  SDM checkpoint not provided or not found, using default sparsity: {sparsity_ratio:.2%}")
        print("   This may result in unfair comparison - iso-sparsity not guaranteed!")

    for p in model.parameters():
        p.requires_grad = False

    if sparsity_ratio > 0:
        print(f"🔪 Applying magnitude pruning with {sparsity_ratio:.4f} ({sparsity_ratio:.2%}) sparsity")
        
        # Collect channel importance scores across all layers
        channel_scores = []
        for layer_idx, layer in enumerate(model.layers):
            weight = layer.in_proj.weight.data[: layer.d_inner]
            layer_scores = weight.abs().mean(dim=1)
            channel_scores.append(layer_scores)
            print(f"    Layer {layer_idx}: {layer.d_inner} channels, score range: {layer_scores.min():.4f} - {layer_scores.max():.4f}")
        
        # Global thresholding for consistent sparsity
        flat_scores = torch.cat(channel_scores)
        k = int(len(flat_scores) * sparsity_ratio)
        threshold = flat_scores.kthvalue(k).values.item() if k > 0 else -float("inf")
        print(f"    Global threshold: {threshold:.4f} (removing bottom {k}/{len(flat_scores)} channels)")
        
        # Apply magnitude-based pruning
        idx = 0
        pruned_per_layer = []
        for layer_idx, layer in enumerate(model.layers):
            n = layer.d_inner
            scores = flat_scores[idx : idx + n]
            idx += n
            
            mask = (scores > threshold).float()
            kept_channels = mask.sum().item()
            pruned_channels = n - kept_channels
            pruned_per_layer.append(pruned_channels / n)
            
            # Apply pruning mask to all relevant weights
            layer.in_proj.weight.data[:n] *= mask.view(-1, 1)
            layer.in_proj.weight.data[n:] *= mask.view(-1, 1)
            layer.out_proj.weight.data *= mask.view(1, -1)
            layer.conv1d.weight.data *= mask.view(-1, 1, 1)
            
            print(f"    Layer {layer_idx}: {pruned_channels}/{n} channels pruned ({pruned_channels/n:.2%} sparsity)")
        
        # Verify achieved sparsity
        achieved_sparsity = sum(pruned_per_layer) / len(pruned_per_layer)
        print(f"✅ ACHIEVED M_CHALLENGE SPARSITY: {achieved_sparsity:.4f} ({achieved_sparsity:.2%})")
        print(f"   Target vs Achieved: {sparsity_ratio:.4f} vs {achieved_sparsity:.4f} (diff: {abs(sparsity_ratio - achieved_sparsity):.4f})")
        
        if abs(sparsity_ratio - achieved_sparsity) > 0.01:  # 1% tolerance
            print("⚠️  WARNING: Achieved sparsity differs significantly from target!")
            print("   This may affect iso-sparsity comparison fairness.")

    rank = 4
    alpha_factor = 2
    dropout = 0.05
    print("Applying uniform LoRA adapters")
    for layer in model.layers:
        layer.in_proj = MaskedLoRALayer(layer.in_proj, rank=rank, alpha=rank * alpha_factor, dropout=dropout)
        layer.out_proj = MaskedLoRALayer(layer.out_proj, rank=rank, alpha=rank * alpha_factor, dropout=dropout)
        for param in [layer.conv1d.weight, layer.x_proj.weight, layer.dt_proj.weight, layer.A_log, layer.D]:
            param.requires_grad = False
    return model


def main():
    parser = argparse.ArgumentParser(description="Create M_challenge baseline")
    parser.add_argument("--checkpoint", required=True, help="Baseline checkpoint")
    parser.add_argument("--sdm_checkpoint", help="SDM checkpoint for sparsity ratio")
    parser.add_argument("--config", required=True, help="Model config YAML")
    parser.add_argument("--output", required=True, help="Output checkpoint path")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model = BaselineSSM(
        d_model=cfg.get("d_model", 768),
        n_layer=cfg.get("n_layer", 12),
        vocab_size=cfg.get("vocab_size", 50257),
        d_state=cfg.get("d_state", 16),
        d_conv=cfg.get("d_conv", 4),
    )

    if os.path.isfile(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)

    model = magnitude_pruned_lora(model, args.sdm_checkpoint)

    Path(os.path.dirname(args.output)).mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": cfg}, args.output)


if __name__ == "__main__":
    main()
