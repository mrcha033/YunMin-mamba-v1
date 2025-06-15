import argparse
import os
import yaml
from pathlib import Path
import torch

from models.baseline_ssm import BaselineSSM
from models.sgh_peft import MaskedLoRALayer


def magnitude_pruned_lora(model: BaselineSSM, sdm_checkpoint: str = None) -> BaselineSSM:
    sparsity_ratio = 0.176
    if sdm_checkpoint and os.path.isfile(sdm_checkpoint):
        ckpt = torch.load(sdm_checkpoint, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)
        z_keys = [k for k in state_dict if k.endswith("z_logits")]
        total = kept = 0
        for k in z_keys:
            z = state_dict[k]
            total += z.numel()
            kept += (z > 0).float().sum().item()
        if total > 0:
            sparsity_ratio = 1.0 - kept / total
            print(f"Detected sparsity ratio: {sparsity_ratio:.2%}")
    else:
        print(f"Using default sparsity ratio: {sparsity_ratio:.2%}")

    for p in model.parameters():
        p.requires_grad = False

    if sparsity_ratio > 0:
        print(f"Applying magnitude pruning with {sparsity_ratio:.2%} sparsity")
        channel_scores = []
        for layer in model.layers:
            weight = layer.in_proj.weight.data[: layer.d_inner]
            channel_scores.append(weight.abs().mean(dim=1))
        flat = torch.cat(channel_scores)
        k = int(len(flat) * sparsity_ratio)
        thr = flat.kthvalue(k).values.item() if k > 0 else -float("inf")
        idx = 0
        for layer in model.layers:
            n = layer.d_inner
            scores = flat[idx : idx + n]
            idx += n
            mask = (scores > thr).float()
            layer.in_proj.weight.data[:n] *= mask.view(-1, 1)
            layer.in_proj.weight.data[n:] *= mask.view(-1, 1)
            layer.out_proj.weight.data *= mask.view(1, -1)
            layer.conv1d.weight.data *= mask.view(-1, 1, 1)

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
