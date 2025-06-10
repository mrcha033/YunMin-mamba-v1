import torch
import torch.nn as nn

class IA3Layer(nn.Module):
    """Per-channel scaling factor applied along a specific dimension."""

    def __init__(self, num_channels: int, dim: int = -1):
        super().__init__()
        self.scaling = nn.Parameter(torch.ones(num_channels))
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = [1] * x.dim()
        shape[self.dim] = -1
        return x * self.scaling.view(*shape)

def _get_parent(model: nn.Module, name: str):
    parts = name.split('.')
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]

def insert_ia3_modules(model: nn.Module, target_module_names=None,
                       layer_types=(nn.Linear, nn.Conv1d, nn.LayerNorm)) -> None:
    """Wrap specified modules with IA³ layers.

    Args:
        model: Model to modify.
        target_module_names: Specific module names to wrap. If ``None`` all
            modules matching ``layer_types`` are wrapped.
        layer_types: Types of layers eligible for IA³ wrapping.
    """
    if target_module_names is None:
        targets = [n for n, m in model.named_modules() if isinstance(m, layer_types)]
    else:
        targets = target_module_names
    for name in targets:
        parent, attr = _get_parent(model, name)
        layer = getattr(parent, attr)
        if isinstance(layer, nn.Sequential) and any(isinstance(m, IA3Layer) for m in layer):
            continue

        if isinstance(layer, nn.Linear):
            ia3 = IA3Layer(layer.out_features, dim=-1)
        elif isinstance(layer, nn.Conv1d):
            ia3 = IA3Layer(layer.out_channels, dim=1)
        elif isinstance(layer, nn.LayerNorm):
            ia3 = IA3Layer(layer.normalized_shape[-1], dim=-1)
        else:
            continue

        setattr(parent, attr, nn.Sequential(layer, ia3))
