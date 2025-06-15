#!/usr/bin/env python
"""Create M_challenge baseline by applying magnitude pruning and uniform LoRA."""

import argparse
import os
import sys
import torch

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from models.baseline_ssm import BaselineSSM
from scripts.run_validation_suite import ValidationSuite, load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate M_challenge baseline (magnitude pruning + uniform LoRA)"
    )
    parser.add_argument("--base_model", type=str, required=True, help="Path to base checkpoint")
    parser.add_argument("--output_path", type=str, required=True, help="Where to save the new checkpoint")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml", help="Model config file")
    parser.add_argument("--sdm_checkpoint", type=str, default=None, help="Optional SDM checkpoint for sparsity ratio")
    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config(args.config)

    # Load base model
    model = BaselineSSM(
        d_model=config.get("d_model", 768),
        n_layer=config.get("n_layer", 12),
        vocab_size=config.get("vocab_size", 50257),
        d_state=config.get("d_state", 16),
        d_conv=config.get("d_conv", 4),
    )

    checkpoint = torch.load(args.base_model, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)

    # Apply magnitude pruning + uniform LoRA
    validator = ValidationSuite(device="cpu")
    model = validator.create_magnitude_pruned_lora(model, args.sdm_checkpoint)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, args.output_path)
    print(f"Saved challenge baseline to {args.output_path}")


if __name__ == "__main__":
    main()
