import torch
import torch.nn as nn

class IA3Layer(nn.Module):
    """Per-channel scaling factor applied to activations."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.scaling = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scaling

def _get_parent(model: nn.Module, name: str):
    parts = name.split('.')
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]

def insert_ia3_modules(model: nn.Module,
                       layer_types=(nn.LayerNorm,)) -> None:
    """Wrap specified layer types with IA3 scaling layers."""
    targets = [n for n, m in model.named_modules() if isinstance(m, layer_types)]
    for name in targets:
        parent, attr = _get_parent(model, name)
        layer = getattr(parent, attr)
        if isinstance(layer, nn.Sequential) and any(isinstance(m, IA3Layer) for m in layer):
            continue
        hidden_size = layer.normalized_shape[-1]
        ia3 = IA3Layer(hidden_size)
        setattr(parent, attr, nn.Sequential(layer, ia3))
