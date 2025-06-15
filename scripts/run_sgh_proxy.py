import argparse
import os
import yaml
from pathlib import Path
import torch

from models.baseline_ssm import BaselineSSM
from models.sgh_peft import SGHPEFTConfig, create_sgh_peft_model


def compute_proxy_importance(model: BaselineSSM):
    """Compute layer importance using weight magnitude."""
    scores = {}
    with torch.no_grad():
        for idx, layer in enumerate(model.layers):
            layer_name = f"layers.{idx}"
            mags = [p.detach().abs().mean() for p in layer.parameters()]
            mean_imp = torch.stack(mags).mean().item() if mags else 0.0
            d_inner = getattr(layer, "d_inner", layer.in_proj.weight.shape[0] // 2)
            scores[layer_name] = {
                "mean_importance": mean_imp,
                "std_importance": 0.0,
                "max_importance": mean_imp,
                "min_importance": mean_imp,
                "active_channels": d_inner,
                "total_channels": d_inner,
                "sparsity_level": 0.0,
                "sparsity_mask": torch.ones(d_inner),
            }
    return scores


def main():
    parser = argparse.ArgumentParser(description="Create M_SGH using proxy importance")
    parser.add_argument("--checkpoint", required=True, help="Baseline checkpoint")
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

    scores = compute_proxy_importance(model)
    sgh_model = create_sgh_peft_model(
        model,
        SGHPEFTConfig(apply_sparsity_mask=False, freeze_base_model=True),
        layer_importance_scores=scores,
    )

    Path(os.path.dirname(args.output)).mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": sgh_model.state_dict(), "config": cfg}, args.output)


if __name__ == "__main__":
    main()
