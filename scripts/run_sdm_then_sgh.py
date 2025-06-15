import argparse
import os
import yaml
from pathlib import Path
import torch

from models.sdm_ssm import SDM_SSM
from models.sgh_peft import SGHPEFTConfig, create_sgh_peft_model


def main():
    parser = argparse.ArgumentParser(description="Create M_sdm_sgh from SDM checkpoint")
    parser.add_argument("--sdm_checkpoint", required=True, help="Path to SDM checkpoint")
    parser.add_argument("--config", required=True, help="Model config YAML")
    parser.add_argument("--output", required=True, help="Output checkpoint path")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model = SDM_SSM(
        d_model=cfg.get("d_model", 768),
        n_layer=cfg.get("n_layer", 12),
        vocab_size=cfg.get("vocab_size", 50257),
        d_state=cfg.get("d_state", 16),
        d_conv=cfg.get("d_conv", 4),
        gumbel_temp=1.0,
    )

    if os.path.isfile(args.sdm_checkpoint):
        ckpt = torch.load(args.sdm_checkpoint, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)

    sgh_model = create_sgh_peft_model(model, SGHPEFTConfig())

    Path(os.path.dirname(args.output)).mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": sgh_model.state_dict(), "config": cfg}, args.output)


if __name__ == "__main__":
    main()
